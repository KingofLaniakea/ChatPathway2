#!/usr/bin/env python3
"""Task 3: causal-direction, order-shuffle, and unrelated-pathway reranking.

Every candidate set contains one expert-validated positive plus negatives of
type ``direction_reversal``, ``step_shuffle``, or ``unrelated_pathway``.  The
default ranking signal is conditional LLM mean log probability.  Optional HNN
quantities are diagnostics, not causal labels; they may affect ranking only
through a versioned calibration fitted on a validation split.

In particular, this evaluator never assumes that total H must monotonically
decrease under ``(J-R) grad(H) + F(t)``: forcing can inject energy, and text
order is not automatically biological time.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from downstream.common.io import mean, write_json, write_rows
from downstream.common.sequence_scoring import conditional_score, load_model
from downstream.new_tasks.schemas import (
    SchemaError,
    load_json_object,
    load_json_records,
    require_choice,
    require_fields,
    require_integer,
    require_mapping,
    require_number,
    require_sequence,
    require_text,
)


NEGATIVE_TYPES = ("direction_reversal", "step_shuffle", "unrelated_pathway")
HNN_DIAGNOSTICS = (
    "hnn_rollout_error",
    "hnn_vector_field_residual",
    "energy_balance_residual",
)
FORBIDDEN_DIAGNOSTICS = ("energy_delta", "monotonic_energy_score", "causal_energy_label")


def _validate_negative_provenance(value: Any, negative_type: str, path: str) -> dict[str, Any]:
    provenance = dict(require_mapping(value, path))
    require_fields(provenance, ("construction_method", "validation_protocol"), path)
    require_text(provenance["construction_method"], f"{path}.construction_method")
    require_text(provenance["validation_protocol"], f"{path}.validation_protocol")
    if negative_type == "direction_reversal":
        relation_ids = require_sequence(
            provenance.get("reversed_relation_ids"), f"{path}.reversed_relation_ids", nonempty=True
        )
        provenance["reversed_relation_ids"] = [
            require_text(str(value), f"{path}.reversed_relation_ids[{index}]")
            for index, value in enumerate(relation_ids)
        ]
    elif negative_type == "step_shuffle":
        provenance["shuffle_unit"] = require_choice(
            provenance.get("shuffle_unit"),
            ("graph_layer", "causal_substep"),
            f"{path}.shuffle_unit",
        )
        original = list(require_sequence(provenance.get("original_order"), f"{path}.original_order", nonempty=True))
        shuffled = list(require_sequence(provenance.get("shuffled_order"), f"{path}.shuffled_order", nonempty=True))
        if len(original) != len(shuffled) or Counter(str(value) for value in original) != Counter(str(value) for value in shuffled):
            raise SchemaError(f"{path}: original_order and shuffled_order must be permutations of the same units.")
        if [str(value) for value in original] == [str(value) for value in shuffled]:
            raise SchemaError(f"{path}: shuffled_order must differ from original_order.")
    else:
        provenance["source_pathway_id"] = require_text(
            provenance.get("source_pathway_id"), f"{path}.source_pathway_id"
        )
        provenance["matching_protocol"] = require_text(
            provenance.get("matching_protocol"), f"{path}.matching_protocol"
        )
    return provenance


def validate_calibration(value: dict[str, Any]) -> dict[str, Any]:
    require_fields(value, ("schema_version", "calibration_id", "fit_split", "features"), "calibration")
    if require_integer(value["schema_version"], "calibration.schema_version") != 1:
        raise SchemaError("calibration.schema_version must be 1.")
    require_text(value["calibration_id"], "calibration.calibration_id")
    require_choice(value["fit_split"], ("validation",), "calibration.fit_split")
    features = require_mapping(value["features"], "calibration.features")
    if "llm_score" not in features:
        raise SchemaError("calibration.features must include llm_score.")
    allowed = {"llm_score", *HNN_DIAGNOSTICS}
    unknown = sorted(set(features) - allowed)
    if unknown:
        raise SchemaError(f"calibration contains unsupported features: {', '.join(unknown)}.")
    normalized: dict[str, dict[str, float]] = {}
    for name, raw in features.items():
        spec = require_mapping(raw, f"calibration.features.{name}")
        require_fields(spec, ("mean", "scale", "weight"), f"calibration.features.{name}")
        scale = require_number(spec["scale"], f"calibration.features.{name}.scale")
        weight = require_number(spec["weight"], f"calibration.features.{name}.weight")
        if scale <= 0:
            raise SchemaError(f"calibration.features.{name}.scale must be positive.")
        if name == "llm_score" and weight <= 0:
            raise SchemaError("llm_score calibration weight must be positive.")
        if name in HNN_DIAGNOSTICS and weight > 0:
            raise SchemaError(f"{name} is an error/residual, so its calibration weight must be non-positive.")
        normalized[name] = {
            "mean": require_number(spec["mean"], f"calibration.features.{name}.mean"),
            "scale": scale,
            "weight": weight,
        }
    return {**value, "features": normalized}


def calibrated_score(candidate: dict[str, Any], calibration: dict[str, Any]) -> float:
    total = 0.0
    diagnostics = candidate.get("hnn_diagnostics") or {}
    for name, spec in calibration["features"].items():
        raw = candidate.get("llm_score") if name == "llm_score" else diagnostics.get(name)
        if raw is None:
            raise SchemaError(f"candidate is missing calibrated feature {name!r}.")
        value = require_number(raw, f"candidate.{name}")
        total += spec["weight"] * (value - spec["mean"]) / spec["scale"]
    return total


def validate_case(value: Any, index: int) -> dict[str, Any]:
    case = dict(require_mapping(value, f"record[{index}]"))
    require_fields(
        case,
        ("id", "question", "expert_validated", "annotation_provenance", "candidates"),
        f"record[{index}]",
    )
    require_text(str(case["id"]), f"record[{index}].id")
    require_text(case["question"], f"record[{index}].question")
    if case["expert_validated"] is not True:
        raise SchemaError(f"record[{index}].expert_validated must be true before causal reranking.")
    annotation = require_mapping(case["annotation_provenance"], f"record[{index}].annotation_provenance")
    require_fields(
        annotation,
        ("annotation_id", "protocol_version", "source_dataset_id"),
        f"record[{index}].annotation_provenance",
    )
    for field in ("annotation_id", "protocol_version", "source_dataset_id"):
        require_text(annotation[field], f"record[{index}].annotation_provenance.{field}")
    candidates = require_sequence(case["candidates"], f"record[{index}].candidates", nonempty=True)
    normalized = []
    positives = 0
    ids = set()
    for candidate_index, raw in enumerate(candidates):
        path = f"record[{index}].candidates[{candidate_index}]"
        candidate = dict(require_mapping(raw, path))
        require_fields(candidate, ("id", "text", "label"), path)
        candidate_id = require_text(str(candidate["id"]), f"{path}.id")
        if candidate_id in ids:
            raise SchemaError(f"{path}.id must be unique within its case.")
        ids.add(candidate_id)
        require_text(candidate["text"], f"{path}.text")
        label = require_choice(candidate["label"], ("positive", "negative"), f"{path}.label")
        if label == "positive":
            positives += 1
            if candidate.get("negative_type") not in (None, "", "positive"):
                raise SchemaError(f"{path}.negative_type must be absent for a positive candidate.")
        else:
            negative_type = require_choice(candidate.get("negative_type"), NEGATIVE_TYPES, f"{path}.negative_type")
            candidate["negative_provenance"] = _validate_negative_provenance(
                candidate.get("negative_provenance"), negative_type, f"{path}.negative_provenance"
            )
        diagnostics = candidate.get("hnn_diagnostics", {})
        diagnostics = require_mapping(diagnostics, f"{path}.hnn_diagnostics")
        forbidden = [name for name in FORBIDDEN_DIAGNOSTICS if name in diagnostics]
        if forbidden:
            raise SchemaError(
                f"{path}.hnn_diagnostics contains forbidden monotonic/causal proxies: {', '.join(forbidden)}."
            )
        unknown = sorted(set(diagnostics) - set(HNN_DIAGNOSTICS))
        if unknown:
            raise SchemaError(f"{path}.hnn_diagnostics has unsupported fields: {', '.join(unknown)}.")
        candidate["hnn_diagnostics"] = {
            name: require_number(score, f"{path}.hnn_diagnostics.{name}") for name, score in diagnostics.items()
        }
        normalized.append(candidate)
    if positives != 1:
        raise SchemaError(f"record[{index}] must contain exactly one positive candidate; got {positives}.")
    if len(normalized) < 2:
        raise SchemaError(f"record[{index}] needs at least one negative candidate.")
    case["candidates"] = normalized
    return case


def _rank_metrics(candidates: list[dict[str, Any]], score_field: str) -> dict[str, float]:
    ranked = sorted(candidates, key=lambda item: float(item[score_field]), reverse=True)
    positive = next(item for item in candidates if item["label"] == "positive")
    positive_score = float(positive[score_field])
    positive_rank = next(index for index, item in enumerate(ranked, 1) if item["label"] == "positive")
    result = {
        "top1": float(positive_rank == 1),
        "mrr": 1.0 / positive_rank,
        "positive_rank": float(positive_rank),
        "overall_pairwise_rejection": mean([
            float(positive_score > float(item[score_field])) for item in candidates if item["label"] == "negative"
        ]),
    }
    for negative_type in NEGATIVE_TYPES:
        values = [
            float(positive_score > float(item[score_field]))
            for item in candidates
            if item.get("negative_type") == negative_type
        ]
        result[f"{negative_type}_rejection"] = mean(values) if values else float("nan")
    return result


def evaluate_cases(
    records: Iterable[dict[str, Any]],
    *,
    calibration: dict[str, Any] | None = None,
    llm_scorer: Callable[[str, str], float] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    prepared = [validate_case(record, index) for index, record in enumerate(records)]
    if not prepared:
        raise SchemaError("Task 3 input contains no candidate cases.")
    calibration_value = validate_calibration(calibration) if calibration is not None else None
    candidate_rows, sample_rows = [], []
    type_counts: dict[str, int] = defaultdict(int)
    for case in prepared:
        candidates = []
        for candidate in case["candidates"]:
            item = dict(candidate)
            if llm_scorer is not None:
                item["llm_score"] = llm_scorer(str(case["question"]), str(item["text"]))
            if "llm_score" not in item:
                raise SchemaError("Every candidate needs llm_score unless an online LLM scorer is supplied.")
            item["llm_score"] = require_number(item["llm_score"], "candidate.llm_score")
            item["ranking_score"] = (
                calibrated_score(item, calibration_value) if calibration_value is not None else item["llm_score"]
            )
            if item["label"] == "negative":
                type_counts[str(item["negative_type"])] += 1
            candidates.append(item)
            candidate_rows.append({"sample_id": case["id"], **item})
        llm_metrics = _rank_metrics(candidates, "llm_score")
        ranking_metrics = _rank_metrics(candidates, "ranking_score")
        row: dict[str, Any] = {"sample_id": case["id"]}
        row.update({f"llm_{key}": value for key, value in llm_metrics.items()})
        row.update({f"ranking_{key}": value for key, value in ranking_metrics.items()})
        positive = next(item for item in candidates if item["label"] == "positive")
        for diagnostic in HNN_DIAGNOSTICS:
            if diagnostic not in positive["hnn_diagnostics"]:
                continue
            negatives = [
                item["hnn_diagnostics"][diagnostic]
                for item in candidates
                if item["label"] == "negative" and diagnostic in item["hnn_diagnostics"]
            ]
            if negatives:
                # Positive candidates should have lower error/residual, so a
                # positive value is a favorable descriptive gap.
                row[f"diagnostic_{diagnostic}_gap"] = mean(negatives) - positive["hnn_diagnostics"][diagnostic]
        sample_rows.append(row)

    aggregate_fields = (
        "llm_top1",
        "llm_mrr",
        "llm_overall_pairwise_rejection",
        "ranking_top1",
        "ranking_mrr",
        "ranking_overall_pairwise_rejection",
    )
    per_type = {}
    for negative_type in NEGATIVE_TYPES:
        for prefix in ("llm", "ranking"):
            field = f"{prefix}_{negative_type}_rejection"
            values = [float(row[field]) for row in sample_rows if field in row and row[field] == row[field]]
            per_type[field] = mean(values) if values else None
    summary = {
        "task": "task3_causal_reranking",
        "num_cases": len(sample_rows),
        "negative_type_counts": dict(type_counts),
        "calibration": {
            "used": calibration_value is not None,
            "calibration_id": calibration_value.get("calibration_id") if calibration_value else None,
            "fit_split": calibration_value.get("fit_split") if calibration_value else None,
        },
        "metrics": {
            **{field: mean([float(row[field]) for row in sample_rows]) for field in aggregate_fields},
            **per_type,
        },
        "hnn_policy": (
            "HNN quantities are diagnostics unless a versioned validation-fit calibration is supplied. "
            "No total-energy monotonicity or causal truth is assumed."
        ),
        "warning": "expert_validated=true asserts candidate validity; preserve the annotation artifact and version in experiment provenance.",
    }
    return candidate_rows, sample_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Strict candidate-set JSON/JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", help="Compute conditional LLM scores online.")
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--calibration", help="Optional validation-fit feature calibration JSON.")
    args = parser.parse_args()
    scorer = None
    if args.base_model:
        tokenizer, model, device = load_model(args.base_model, args.adapter, args.device)
        scorer = lambda question, text: conditional_score(tokenizer, model, device, question, text, args.max_length)
    calibration = load_json_object(args.calibration) if args.calibration else None
    candidate_rows, sample_rows, summary = evaluate_cases(
        load_json_records(args.input), calibration=calibration, llm_scorer=scorer
    )
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "candidates.csv", candidate_rows)
    write_rows(output_dir / "sample_metrics.csv", sample_rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
