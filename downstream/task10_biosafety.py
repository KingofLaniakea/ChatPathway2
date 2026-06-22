#!/usr/bin/env python3
"""Task X: declarative BioSafety-style downstream analysis.

The supplied PDF names this task but does not define a taxonomy. This generic
evaluator therefore requires all semantics in its input instead of hard-coding
an invented biosafety policy. Each record contains:

``{id, gold: {risk_labels: [...], evidence_ids: [...], severity: number?},
prediction: {risk_labels: [...], evidence_ids: [...], severity: number?}}``.

It reports set-level risk-label and evidence grounding metrics, plus severity
error where calibrated severity labels are supplied. A task-specific taxonomy,
annotation guidance, and evidence corpus remain prerequisites for a scientific
claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from downstream.entities import precision_recall_f1
from downstream.io import as_float, load_records, mean, write_json, write_rows


def string_set(value: Any) -> set[str]:
    return {str(item).strip() for item in value} if isinstance(value, list) else set()


def evaluate(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for index, record in enumerate(records):
        gold, prediction = record.get("gold"), record.get("prediction")
        if not isinstance(gold, dict) or not isinstance(prediction, dict):
            raise ValueError(f"Record {index} needs gold and prediction objects.")
        risk = precision_recall_f1(string_set(prediction.get("risk_labels")), string_set(gold.get("risk_labels")))
        evidence = precision_recall_f1(string_set(prediction.get("evidence_ids")), string_set(gold.get("evidence_ids")))
        gold_severity, predicted_severity = as_float(gold.get("severity")), as_float(prediction.get("severity"))
        rows.append({
            "id": record.get("id", index),
            **{f"risk_{key}": value for key, value in risk.items()},
            **{f"evidence_{key}": value for key, value in evidence.items()},
            "severity_absolute_error": abs(gold_severity - predicted_severity) if gold_severity is not None and predicted_severity is not None else "",
        })
    summary = {"num_records": len(rows), "risk_metrics": {
        key: mean([float(row[f"risk_{key}"]) for row in rows]) for key in ("precision", "recall", "f1", "exact_match")
    }, "evidence_metrics": {
        key: mean([float(row[f"evidence_{key}"]) for row in rows]) for key in ("precision", "recall", "f1", "exact_match")
    }}
    severity = [float(row["severity_absolute_error"]) for row in rows if row["severity_absolute_error"] != ""]
    summary["mean_severity_absolute_error"] = mean(severity) if severity else None
    summary["warning"] = "The PDF does not specify a BioSafety taxonomy or corpus; labels/evidence IDs must come from a versioned task definition."
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="CSV/JSON/JSONL declarative biosafety records.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows, summary = evaluate(load_records(args.input))
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
