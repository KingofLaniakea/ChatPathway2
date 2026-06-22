#!/usr/bin/env python3
"""Task VIII: directional reranking with expert-validated negatives.

Input is JSON/JSONL records of the form:
``{"id", "question", "candidates": [{"text", "label": "positive"|"negative",
"negative_type": "direction_reversal", "score"?}]}``.

The evaluator intentionally does not auto-create direction negatives: reversing
text can change biology, entity roles, or grammatical plausibility. A candidate
set must be curated so the negative differs only in causal direction/mechanism.
Scores are either supplied or calculated as conditional LLM log probabilities.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from downstream.common.io import load_records, mean, write_json, write_rows
from downstream.common.sequence_scoring import conditional_score, load_model


def evaluate_case(candidates: list[dict[str, Any]]) -> dict[str, float]:
    positives = [candidate for candidate in candidates if candidate.get("label") == "positive"]
    negatives = [candidate for candidate in candidates if candidate.get("label") == "negative"]
    if not positives or not negatives:
        raise ValueError("Each case needs at least one positive and one negative candidate.")
    if any(candidate.get("negative_type") != "direction_reversal" for candidate in negatives):
        raise ValueError("All negatives must explicitly declare negative_type='direction_reversal'.")
    ranked = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
    best_positive = max(float(candidate["score"]) for candidate in positives)
    best_negative = max(float(candidate["score"]) for candidate in negatives)
    return {
        "directionality_accuracy": float(ranked[0]["label"] == "positive"),
        "wrong_direction_rejection_rate": float(best_positive > best_negative),
        "score_gap": best_positive - best_negative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Directional candidate-set JSON/JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", help="Calculate scores instead of consuming candidate score fields.")
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=1072)
    args = parser.parse_args()
    tokenizer = model = device = None
    if args.base_model:
        tokenizer, model, device = load_model(args.base_model, args.adapter, args.device)
    rows, candidate_rows = [], []
    for index, record in enumerate(load_records(args.input)):
        question = str(record.get("question", ""))
        candidates = record.get("candidates")
        if not isinstance(candidates, list):
            raise ValueError(f"Record {index} has no candidate list.")
        prepared = []
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict) or not candidate.get("text"):
                raise ValueError(f"Record {index} candidate {candidate_index} needs a text field.")
            item = dict(candidate)
            if model is not None:
                item["score"] = conditional_score(tokenizer, model, device, question, str(item["text"]), args.max_length)
            if "score" not in item:
                raise ValueError("Pass --base-model or supply a numeric score for every candidate.")
            prepared.append(item)
            candidate_rows.append({"sample_id": record.get("id", index), "candidate_index": candidate_index, **item})
        rows.append({"sample_id": record.get("id", index), **evaluate_case(prepared)})
    summary = {"num_cases": len(rows), "metrics": {
        key: mean([float(row[key]) for row in rows])
        for key in ("directionality_accuracy", "wrong_direction_rejection_rate", "score_gap")
    }, "warning": "Scores are meaningful only after expert validation that negative candidates differ solely in direction/mechanism."}
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "candidates.csv", candidate_rows)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
