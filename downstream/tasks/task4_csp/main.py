#!/usr/bin/env python3
"""Task IV: multi-step Conditional Step Prediction (CSP) evaluation.

The maintained dataset emits one JSON object with ``remaining_steps`` and
``predicted_phenotype``. Each pathway step can summarize multiple reaction or
relation events, so evaluation is order-aware at the step object level and
uses entity/relation-set overlap inside each aligned step. Missing phenotype
annotations are scored only for required-null compliance, never as biological
negative labels.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

from downstream.common.entities import extract_entities, precision_recall_f1
from downstream.common.io import load_records, mean, write_json, write_rows
from downstream.common.pathway_json import (
    ParsedPathwayPayload,
    parse_pathway_payload,
    phenotype_text,
    record_id,
)


RELATIONS = (
    "mediates a functional link",
    "is shared in successive reactions",
    "is converted to",
    "dephosphorylates",
    "phosphorylates",
    "ubiquitinates",
    "dissociates",
    "associates",
    "methylates",
    "acetylates",
    "transports",
    "catalyzes",
    "activates",
    "inhibits",
    "regulates",
    "represses",
    "expresses",
    "produces",
    "converts",
    "degrades",
    "induces",
    "recruits",
    "cleaves",
    "forms",
    "binds",
)
RELATION_RE = re.compile(r"\b(" + "|".join(re.escape(value) for value in RELATIONS) + r")\b", re.IGNORECASE)


def parse_steps(value: Any) -> ParsedPathwayPayload:
    return parse_pathway_payload(value)


def normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def relation_set(value: str) -> set[str]:
    return {match.group(1).casefold() for match in RELATION_RE.finditer(value)}


def set_f1(left: set[str], right: set[str]) -> float:
    return float(precision_recall_f1(left, right)["f1"])


def _as_payload(value: Any) -> ParsedPathwayPayload:
    return value if isinstance(value, ParsedPathwayPayload) else parse_pathway_payload(value)


def evaluate_pair(target: Any, predicted: Any) -> dict[str, float | int | str]:
    target_payload = _as_payload(target)
    predicted_payload = _as_payload(predicted)
    target_steps = target_payload.steps
    predicted_steps = predicted_payload.steps
    denominator = max(len(target_steps), len(predicted_steps), 1)

    exact = step_index = 0.0
    entity_f1 = relation_f1 = 0.0
    for expected, actual in zip(target_steps, predicted_steps):
        expected_events = expected.substeps or (expected.text,)
        actual_events = actual.substeps or (actual.text,)
        exact += float(
            Counter(normalized_text(value) for value in expected_events)
            == Counter(normalized_text(value) for value in actual_events)
        )
        expected_entities = extract_entities(expected.text)
        actual_entities = extract_entities(actual.text)
        entity_f1 += set_f1(actual_entities, expected_entities)
        relation_f1 += set_f1(relation_set(actual.text), relation_set(expected.text))
        step_index += float(
            expected.step is not None
            and actual.step is not None
            and expected.step == actual.step
        )

    target_phenotype = phenotype_text(target_payload.phenotype)
    predicted_phenotype = phenotype_text(predicted_payload.phenotype)
    phenotype_available = target_phenotype is not None
    phenotype_exact = (
        float(normalized_text(target_phenotype) == normalized_text(predicted_phenotype or ""))
        if phenotype_available
        else ""
    )
    missing_null_compliance = float(predicted_phenotype is None) if not phenotype_available else ""

    return {
        "target_steps": len(target_steps),
        "predicted_steps": len(predicted_steps),
        "paired_steps": min(len(target_steps), len(predicted_steps)),
        "step_count_accuracy": float(len(target_steps) == len(predicted_steps)),
        "exact_match": exact / denominator,
        "reactant_match": entity_f1 / denominator,
        "reaction_match": relation_f1 / denominator,
        "step_index_match": step_index / denominator,
        "json_validity": float(predicted_payload.json_valid),
        "parse_validity": float(predicted_payload.schema_valid),
        "phenotype_target_available": int(phenotype_available),
        "phenotype_exact_available": phenotype_exact,
        "missing_phenotype_null_compliance": missing_null_compliance,
        "prediction_parse_error": predicted_payload.error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-column", default="answer")
    parser.add_argument("--predicted-column", default="predicted_answer")
    args = parser.parse_args()

    rows = []
    for index, record in enumerate(load_records(args.input)):
        row = {
            "id": record_id(record, index),
            "record_id": record.get("record_id", ""),
            "source_json": record.get("source_json", ""),
        }
        row.update(evaluate_pair(record.get(args.target_column), record.get(args.predicted_column)))
        rows.append(row)

    available_phenotype = [row for row in rows if row["phenotype_target_available"]]
    missing_phenotype = [row for row in rows if not row["phenotype_target_available"]]
    summary = {
        "num_records": len(rows),
        "phenotype_available_records": len(available_phenotype),
        "phenotype_missing_records": len(missing_phenotype),
        "metrics": {
            key: mean([float(row[key]) for row in rows])
            for key in (
                "step_count_accuracy",
                "exact_match",
                "reactant_match",
                "reaction_match",
                "step_index_match",
                "json_validity",
                "parse_validity",
            )
        },
        "phenotype_metrics": {
            "exact_when_available": mean(
                [float(row["phenotype_exact_available"]) for row in available_phenotype]
            ),
            "null_compliance_when_unannotated": mean(
                [float(row["missing_phenotype_null_compliance"]) for row in missing_phenotype]
            ),
            "warning": "Null compliance for unannotated examples is a format metric, not phenotype accuracy.",
        },
    }
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
