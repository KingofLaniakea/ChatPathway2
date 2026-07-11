#!/usr/bin/env python3
"""Optional Task 6: BioMaze multiple-choice evaluation.

BioMaze remains an independent external benchmark.  This module scores frozen
prediction artifacts; it does not turn model-generated pathways into BioMaze
ground truth or claim zero-shot status without a versioned contamination audit.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from downstream.common.io import mean, write_json, write_rows
from downstream.new_tasks.schemas import (
    SchemaError,
    load_json_object,
    load_json_records,
    require_choice,
    require_fields,
    require_integer,
    require_mapping,
    require_sequence,
    require_text,
)


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    require_fields(
        value,
        (
            "schema_version",
            "dataset_id",
            "dataset_version",
            "split",
            "source",
            "license",
            "evaluation_protocol",
            "contamination_audit",
            "model_checkpoint",
        ),
        "manifest",
    )
    if require_integer(value["schema_version"], "manifest.schema_version") != 1:
        raise SchemaError("manifest.schema_version must be 1.")
    for field in ("dataset_id", "dataset_version", "source", "license", "evaluation_protocol", "model_checkpoint"):
        require_text(value[field], f"manifest.{field}")
    require_choice(value["split"], ("validation", "test"), "manifest.split")
    audit = require_mapping(value["contamination_audit"], "manifest.contamination_audit")
    require_fields(audit, ("status", "method"), "manifest.contamination_audit")
    require_choice(audit["status"], ("not_detected", "possible", "unknown"), "manifest.contamination_audit.status")
    require_text(audit["method"], "manifest.contamination_audit.method")
    return dict(value)


def _choices(value: Any, path: str) -> dict[str, str]:
    if isinstance(value, dict):
        result = {}
        for key, text in value.items():
            label = require_text(str(key), f"{path}.key").upper()
            if label in result:
                raise SchemaError(f"{path} contains duplicate case-insensitive label {label!r}.")
            result[label] = require_text(text, f"{path}.{key}")
    else:
        values = require_sequence(value, path, nonempty=True)
        if len(values) > 26:
            raise SchemaError(f"{path} supports at most 26 choices.")
        result = {chr(ord("A") + index): require_text(text, f"{path}[{index}]") for index, text in enumerate(values)}
    if len(result) < 2:
        raise SchemaError(f"{path} requires at least two choices.")
    return result


def validate_record(value: Any, index: int) -> dict[str, Any]:
    path = f"record[{index}]"
    record = dict(require_mapping(value, path))
    require_fields(record, ("id", "question", "choices", "gold_option", "predicted_option"), path)
    record["id"] = require_text(str(record["id"]), f"{path}.id")
    record["question"] = require_text(record["question"], f"{path}.question")
    record["choices"] = _choices(record["choices"], f"{path}.choices")
    record["gold_option"] = require_text(record["gold_option"], f"{path}.gold_option").upper()
    if record["gold_option"] not in record["choices"]:
        raise SchemaError(f"{path}.gold_option must name a provided choice.")
    record["predicted_option"] = require_text(record["predicted_option"], f"{path}.predicted_option").upper()
    tags = record.get("tags", [])
    record["tags"] = [require_text(tag, f"{path}.tags[{tag_index}]") for tag_index, tag in enumerate(require_sequence(tags, f"{path}.tags"))]
    return record


def evaluate_records(
    records: Iterable[dict[str, Any]], manifest: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = validate_manifest(manifest)
    prepared = [validate_record(record, index) for index, record in enumerate(records)]
    if not prepared:
        raise SchemaError("Task 6 input contains no questions.")
    rows = []
    tag_values: dict[str, list[float]] = defaultdict(list)
    for record in prepared:
        valid = record["predicted_option"] in record["choices"]
        correct = float(valid and record["predicted_option"] == record["gold_option"])
        rows.append({
            "sample_id": record["id"],
            "gold_option": record["gold_option"],
            "predicted_option": record["predicted_option"],
            "answer_valid": float(valid),
            "correct": correct,
            "tags": "|".join(record["tags"]),
        })
        for tag in record["tags"]:
            tag_values[tag].append(correct)
    return rows, {
        "task": "task6_biomaze",
        "manifest": metadata,
        "num_questions": len(rows),
        "metrics": {
            "accuracy": mean([float(row["correct"]) for row in rows]),
            "answer_validity": mean([float(row["answer_valid"]) for row in rows]),
        },
        "accuracy_by_tag": {tag: mean(values) for tag, values in sorted(tag_values.items())},
        "claim_policy": (
            "Report as zero-shot only when the manifest contamination audit and checkpoint chronology support that claim. "
            "BioMaze is independent of pathway-text Task 0--5 ground truth."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Frozen BioMaze prediction JSON/JSONL.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows, summary = evaluate_records(load_json_records(args.input), load_json_object(args.manifest))
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
