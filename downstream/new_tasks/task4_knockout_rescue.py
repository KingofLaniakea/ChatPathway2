#!/usr/bin/env python3
"""Task 4: experimentally grounded knockout and rescue evaluation.

This module evaluates calibrated phenotype probabilities supplied by an
upstream scorer.  It does not derive a phenotype probability from free-form
prose, and it rejects attempts to label the time-only force ``F(t)`` as a
knockout/control input.  The current autonomous dynamics may be used with an
intervention encoded in the prompt/initial latent state; a genuinely explicit
control requires a separately trained intervention-conditioned model.

Cases with ``phenotype_available=false`` are counted and excluded.  Absence of
an annotation is never treated as a negative phenotype.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from downstream.common.io import mean, write_json, write_rows
from downstream.new_tasks.schemas import (
    SchemaError,
    load_json_records,
    require_binary,
    require_choice,
    require_fields,
    require_mapping,
    require_probability,
    require_sequence,
    require_text,
)


CONDITIONING_MODES = ("prompt_initial_condition", "explicit_intervention_conditioned_dynamics")
INTERVENTION_KINDS = ("knockout", "knockdown", "inhibition", "overexpression", "activation", "drug")
KO_KINDS = {"knockout", "knockdown", "inhibition"}
RESCUE_KINDS = {"overexpression", "activation", "drug"}


def _validate_provenance(value: Any, path: str) -> dict[str, Any]:
    result = dict(require_mapping(value, path))
    require_fields(result, ("dataset_id", "dataset_version", "split", "evidence_source"), path)
    require_text(result["dataset_id"], f"{path}.dataset_id")
    require_text(result["dataset_version"], f"{path}.dataset_version")
    require_choice(result["split"], ("test",), f"{path}.split")
    require_text(result["evidence_source"], f"{path}.evidence_source")
    return result


def _validate_scorer(value: Any, path: str) -> dict[str, Any]:
    result = dict(require_mapping(value, path))
    require_fields(result, ("scorer_id", "calibration_id", "calibration_split"), path)
    require_text(result["scorer_id"], f"{path}.scorer_id")
    require_text(result["calibration_id"], f"{path}.calibration_id")
    require_choice(result["calibration_split"], ("validation",), f"{path}.calibration_split")
    result["threshold"] = require_probability(result.get("threshold", 0.5), f"{path}.threshold")
    return result


def _validate_intervention(value: Any, path: str) -> dict[str, str]:
    item = dict(require_mapping(value, path))
    require_fields(item, ("kind", "target"), path)
    kind = require_choice(item["kind"], INTERVENTION_KINDS, f"{path}.kind")
    target = require_text(item["target"], f"{path}.target")
    return {"kind": kind, "target": target}


def _validate_state(value: Any, path: str) -> dict[str, Any]:
    state = dict(require_mapping(value, path))
    require_fields(
        state,
        ("state_id", "role", "interventions", "gold_positive", "predicted_probability"),
        path,
    )
    state["state_id"] = require_text(str(state["state_id"]), f"{path}.state_id")
    state["role"] = require_choice(state["role"], ("wt", "ko", "rescue"), f"{path}.role")
    interventions = require_sequence(state["interventions"], f"{path}.interventions")
    state["interventions"] = [
        _validate_intervention(item, f"{path}.interventions[{index}]") for index, item in enumerate(interventions)
    ]
    state["gold_positive"] = require_binary(state["gold_positive"], f"{path}.gold_positive")
    state["predicted_probability"] = require_probability(
        state["predicted_probability"], f"{path}.predicted_probability"
    )
    if state["role"] == "wt" and state["interventions"]:
        raise SchemaError(f"{path}: wt state must have no interventions.")
    if state["role"] == "ko" and not any(item["kind"] in KO_KINDS for item in state["interventions"]):
        raise SchemaError(f"{path}: ko state requires a knockout/knockdown/inhibition intervention.")
    if state["role"] == "rescue":
        state["parent_ko"] = require_text(state.get("parent_ko"), f"{path}.parent_ko")
        if not any(item["kind"] in RESCUE_KINDS for item in state["interventions"]):
            raise SchemaError(f"{path}: rescue state requires an overexpression/activation/drug intervention.")
    return state


def validate_case(value: Any, index: int) -> dict[str, Any]:
    path = f"record[{index}]"
    case = dict(require_mapping(value, path))
    require_fields(case, ("case_id", "phenotype_available"), path)
    case["case_id"] = require_text(str(case["case_id"]), f"{path}.case_id")
    if not isinstance(case["phenotype_available"], bool):
        raise SchemaError(f"{path}.phenotype_available must be boolean.")
    if not case["phenotype_available"]:
        if case.get("states"):
            raise SchemaError(f"{path}: phenotype_available=false cannot include scored states.")
        return {"case_id": case["case_id"], "phenotype_available": False}

    require_fields(
        case,
        (
            "dataset_provenance",
            "model_provenance",
            "phenotype",
            "phenotype_scorer",
            "dynamics_conditioning",
            "states",
        ),
        path,
    )
    case["dataset_provenance"] = _validate_provenance(case["dataset_provenance"], f"{path}.dataset_provenance")
    phenotype = dict(require_mapping(case["phenotype"], f"{path}.phenotype"))
    require_fields(phenotype, ("phenotype_id", "positive_definition"), f"{path}.phenotype")
    require_text(phenotype["phenotype_id"], f"{path}.phenotype.phenotype_id")
    require_text(phenotype["positive_definition"], f"{path}.phenotype.positive_definition")
    case["phenotype"] = phenotype
    case["phenotype_scorer"] = _validate_scorer(case["phenotype_scorer"], f"{path}.phenotype_scorer")
    model = dict(require_mapping(case["model_provenance"], f"{path}.model_provenance"))
    require_fields(
        model,
        ("base_checkpoint", "adapter_checkpoint", "dynamics_checkpoint"),
        f"{path}.model_provenance",
    )
    for field in ("base_checkpoint", "adapter_checkpoint", "dynamics_checkpoint"):
        require_text(model[field], f"{path}.model_provenance.{field}")
    case["model_provenance"] = model
    mode = str(case["dynamics_conditioning"]).strip()
    lowered = mode.casefold().replace(" ", "")
    if "f(t)" in lowered or "time_only_force" in lowered or lowered in {"force", "forcing"}:
        raise SchemaError(
            f"{path}.dynamics_conditioning cannot treat time-only F(t) as an intervention/control input."
        )
    case["dynamics_conditioning"] = require_choice(mode, CONDITIONING_MODES, f"{path}.dynamics_conditioning")
    if mode == "prompt_initial_condition":
        require_text(model.get("prompt_template_version"), f"{path}.model_provenance.prompt_template_version")
    else:
        require_text(model.get("conditioning_schema_id"), f"{path}.model_provenance.conditioning_schema_id")
    states_raw = require_sequence(case["states"], f"{path}.states", nonempty=True)
    states = [_validate_state(state, f"{path}.states[{state_index}]") for state_index, state in enumerate(states_raw)]
    ids = [state["state_id"] for state in states]
    if len(ids) != len(set(ids)):
        raise SchemaError(f"{path}.states state_id values must be unique.")
    wt_states = [state for state in states if state["role"] == "wt"]
    ko_states = [state for state in states if state["role"] == "ko"]
    if len(wt_states) != 1 or not ko_states:
        raise SchemaError(f"{path} requires exactly one wt state and at least one ko state.")
    by_id = {state["state_id"]: state for state in states}
    for state in states:
        if state["role"] != "rescue":
            continue
        parent = by_id.get(state["parent_ko"])
        if parent is None or parent["role"] != "ko":
            raise SchemaError(f"{path}: rescue {state['state_id']!r} references a non-KO parent.")
        parent_ko = {(item["kind"], item["target"]) for item in parent["interventions"] if item["kind"] in KO_KINDS}
        rescue_ko = {(item["kind"], item["target"]) for item in state["interventions"] if item["kind"] in KO_KINDS}
        if not parent_ko.issubset(rescue_ko):
            raise SchemaError(f"{path}: rescue {state['state_id']!r} must retain its parent KO intervention.")
    case["states"] = states
    return case


def _sign(value: float, tolerance: float = 1e-12) -> int:
    return 1 if value > tolerance else -1 if value < -tolerance else 0


def evaluate_cases(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cases = [validate_case(record, index) for index, record in enumerate(records)]
    if not cases:
        raise SchemaError("Task 4 input contains no cases.")
    missing = [case for case in cases if not case["phenotype_available"]]
    annotated = [case for case in cases if case["phenotype_available"]]
    endpoint_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    rescue_ranks: list[float] = []
    rescue_hits: list[float] = []
    effect_sign_values: list[float] = []

    for case in annotated:
        threshold = float(case["phenotype_scorer"]["threshold"])
        states = case["states"]
        by_id = {state["state_id"]: state for state in states}
        wt = next(state for state in states if state["role"] == "wt")
        state_rows = []
        for state in states:
            probability = float(state["predicted_probability"])
            gold = int(state["gold_positive"])
            row = {
                "case_id": case["case_id"],
                "state_id": state["state_id"],
                "role": state["role"],
                "gold_positive": gold,
                "predicted_probability": probability,
                "predicted_positive": int(probability >= threshold),
                "correct": float((probability >= threshold) == bool(gold)),
                "brier": (probability - gold) ** 2,
                "parent_ko": state.get("parent_ko", ""),
                "dynamics_conditioning": case["dynamics_conditioning"],
            }
            if state["role"] == "ko":
                gold_delta = gold - int(wt["gold_positive"])
                predicted_delta = probability - float(wt["predicted_probability"])
                if _sign(gold_delta):
                    row["ko_effect_direction_correct"] = float(_sign(gold_delta) == _sign(predicted_delta))
                    effect_sign_values.append(float(row["ko_effect_direction_correct"]))
            if state["role"] == "rescue":
                parent = by_id[state["parent_ko"]]
                denominator = float(wt["predicted_probability"]) - float(parent["predicted_probability"])
                row["predicted_recovery_fraction"] = (
                    (probability - float(parent["predicted_probability"])) / denominator
                    if abs(denominator) > 1e-12
                    else None
                )
            endpoint_rows.append(row)
            state_rows.append(row)

        rescues_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for state in states:
            if state["role"] == "rescue":
                rescues_by_parent[state["parent_ko"]].append(state)
        eligible_rescue_groups = 0
        for candidates in rescues_by_parent.values():
            positives = [state for state in candidates if state["gold_positive"] == 1]
            if not positives:
                continue
            eligible_rescue_groups += 1
            ranked = sorted(candidates, key=lambda state: float(state["predicted_probability"]), reverse=True)
            best_rank = min(index for index, state in enumerate(ranked, 1) if state["gold_positive"] == 1)
            rescue_ranks.append(1.0 / best_rank)
            rescue_hits.append(float(best_rank == 1))
        case_rows.append({
            "case_id": case["case_id"],
            "num_states": len(states),
            "num_rescues": sum(state["role"] == "rescue" for state in states),
            "eligible_rescue_groups": eligible_rescue_groups,
            "endpoint_accuracy": mean([float(row["correct"]) for row in state_rows]),
            "endpoint_brier": mean([float(row["brier"]) for row in state_rows]),
        })

    summary = {
        "task": "task4_knockout_rescue",
        "num_cases": len(cases),
        "phenotype_annotated_cases": len(annotated),
        "phenotype_missing_cases_excluded": len(missing),
        "num_scored_states": len(endpoint_rows),
        "metrics": {
            "endpoint_accuracy": mean([float(row["correct"]) for row in endpoint_rows]) if endpoint_rows else None,
            "endpoint_brier": mean([float(row["brier"]) for row in endpoint_rows]) if endpoint_rows else None,
            "ko_effect_direction_accuracy": mean(effect_sign_values) if effect_sign_values else None,
            "rescue_hit_at_1": mean(rescue_hits) if rescue_hits else None,
            "rescue_mrr": mean(rescue_ranks) if rescue_ranks else None,
            "eligible_rescue_groups": len(rescue_ranks),
        },
        "phenotype_policy": "Unannotated phenotype cases are excluded and counted, never mapped to gold_negative.",
        "conditioning_policy": (
            "Allowed: prompt/initial-condition intervention, or a separately trained explicit intervention-conditioned model. "
            "Forbidden: interpreting time-only F(t) as knockout u."
        ),
        "reportability_warning": (
            "Scientific claims require real intervention evidence and a phenotype scorer calibrated on validation data; "
            "synthetic records only test the evaluator."
        ),
    }
    return endpoint_rows, case_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Strict knockout/rescue JSON/JSONL records.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    endpoints, cases, summary = evaluate_cases(load_json_records(args.input))
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "endpoint_metrics.csv", endpoints)
    write_rows(output_dir / "case_metrics.csv", cases)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
