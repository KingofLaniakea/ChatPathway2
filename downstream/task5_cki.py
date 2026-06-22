#!/usr/bin/env python3
"""Task V: Steady-State Counterfactual Knockout Gate Inference (CKI).

This module deliberately evaluates model *outputs* rather than inventing a
phenotype probability from generated prose. Build a record per intervention
with calibrated ``wt_survival`` and ``ko_survival`` probabilities, optional
endpoint distributions, and a gold/predicted gate label. This makes CSR, JSD,
GateAcc, and SLM auditable once a CKI dataset and scorer exist.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from downstream.io import as_float, load_records, mean, write_json, write_rows


def json_value(value: Any, fallback: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return fallback
    return value if value is not None else fallback


def js_divergence(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    if min(left) < 0 or min(right) < 0 or sum(left) <= 0 or sum(right) <= 0:
        return None
    p = [value / sum(left) for value in left]
    q = [value / sum(right) for value in right]
    midpoint = [(a + b) / 2 for a, b in zip(p, q)]
    def kl(a: list[float], b: list[float]) -> float:
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0 and y > 0)
    return (kl(p, midpoint) + kl(q, midpoint)) / 2


def evaluate(records: list[dict[str, Any]], epsilon: float = 1e-8) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        wt = as_float(record.get("wt_survival"))
        ko = as_float(record.get("ko_survival"))
        if wt is None or ko is None:
            raise ValueError(f"Record {index} requires numeric wt_survival and ko_survival.")
        wt_distribution = json_value(record.get("wt_distribution"), [])
        ko_distribution = json_value(record.get("ko_distribution"), [])
        if not isinstance(wt_distribution, list) or not isinstance(ko_distribution, list):
            raise ValueError("wt_distribution and ko_distribution must be JSON arrays when provided.")
        gold_gate = str(record.get("gold_gate", "")).strip().lower()
        predicted_gate = str(record.get("predicted_gate", "")).strip().lower()
        row = {
            "id": record.get("id", index),
            "case_id": record.get("case_id", record.get("id", index)),
            "ko_set_size": int(as_float(record.get("ko_set_size"), 1) or 1),
            "wt_survival": wt,
            "ko_survival": ko,
            "csr": ko / max(wt, epsilon),
            "gold_gate": gold_gate,
            "predicted_gate": predicted_gate,
            "gate_correct": float(bool(gold_gate) and gold_gate == predicted_gate),
            "jsd": js_divergence([float(x) for x in wt_distribution], [float(x) for x in ko_distribution]),
        }
        rows.append(row)
        grouped[str(row["case_id"])].append(row)

    slm_values = []
    for case_rows in grouped.values():
        singles = [row["ko_survival"] for row in case_rows if row["ko_set_size"] == 1]
        doubles = [row["ko_survival"] for row in case_rows if row["ko_set_size"] >= 2]
        if singles and doubles:
            slm_values.append(float(min(doubles) < min(singles)))
    valid_gate = [row["gate_correct"] for row in rows if row["gold_gate"] and row["predicted_gate"]]
    valid_jsd = [float(row["jsd"]) for row in rows if row["jsd"] is not None]
    summary = {
        "num_interventions": len(rows),
        "gate_accuracy": mean(valid_gate),
        "mean_csr": mean([float(row["csr"]) for row in rows]),
        "mean_jsd": mean(valid_jsd),
        "slm": mean(slm_values),
        "slm_eligible_cases": len(slm_values),
        "warning": "No calibrated CKI dataset/scorer is included in this repository; metrics are valid only for supplied survival probabilities.",
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="CSV/JSON/JSONL intervention records.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows, summary = evaluate(load_records(args.input))
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
