#!/usr/bin/env python3
"""Task VII: ordered-pathway step-shuffling robustness.

For each gold continuation, the evaluator creates fixed-seed shuffled-order
negatives and asks whether the model ranks the original order higher under
conditional log likelihood. It emits candidates even when no model is supplied,
so candidate construction can be reviewed before expensive scoring.

This task is valid only for examples with explicit, independently meaningful
step boundaries. It does not turn arbitrary prose into a causal benchmark.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from downstream.io import load_records, mean, write_json, write_rows
from downstream.sequence_scoring import conditional_score, load_model


def step_lines(text: str) -> list[str]:
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


def shuffled_candidates(lines: list[str], count: int, rng: random.Random) -> list[str]:
    if len(lines) < 3:
        return []
    results, seen = [], {tuple(lines)}
    attempts = 0
    while len(results) < count and attempts < count * 30:
        proposal = lines[:]
        rng.shuffle(proposal)
        key = tuple(proposal)
        if key not in seen:
            seen.add(key)
            results.append("\n".join(proposal))
        attempts += 1
    return results


def rank_summary(candidates: list[dict[str, Any]]) -> dict[str, float]:
    ranked = sorted(candidates, key=lambda item: float(item["score"]), reverse=True)
    gold_rank = next(index for index, item in enumerate(ranked, 1) if item["label"] == "gold")
    negatives = [float(item["score"]) for item in candidates if item["label"] == "shuffled"]
    gold_score = next(float(item["score"]) for item in candidates if item["label"] == "gold")
    return {
        "gold_rank": gold_rank,
        "hit_at_1": float(gold_rank == 1),
        "mrr": 1.0 / gold_rank,
        "shuffle_rejection_rate": mean([float(gold_score > score) for score in negatives]),
        "mean_score_margin": gold_score - mean(negatives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="CSV/JSON/JSONL with question and answer columns.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--question-column", default="question")
    parser.add_argument("--answer-column", default="answer")
    parser.add_argument("--num-negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--base-model", help="Enable conditional log-probability scoring.")
    parser.add_argument("--adapter", help="Optional LoRA adapter for scoring.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    tokenizer = model = device = None
    if args.base_model:
        tokenizer, model, device = load_model(args.base_model, args.adapter, args.device)
    rng = random.Random(args.seed)
    candidate_rows, metric_rows = [], []
    for index, record in enumerate(load_records(args.input)[: args.limit]):
        lines = step_lines(record.get(args.answer_column, ""))
        negatives = shuffled_candidates(lines, args.num_negatives, rng)
        if not negatives:
            continue
        question = str(record.get(args.question_column, ""))
        candidates = [{"label": "gold", "text": "\n".join(lines)}] + [{"label": "shuffled", "text": text} for text in negatives]
        for candidate_index, candidate in enumerate(candidates):
            row = {"sample_id": record.get("id", record.get("entry_id", index)), "candidate_index": candidate_index, **candidate}
            if model is not None:
                row["score"] = conditional_score(tokenizer, model, device, question, candidate["text"], args.max_length)
            candidate_rows.append(row)
        if model is not None:
            scored = candidate_rows[-len(candidates):]
            metric_rows.append({"sample_id": record.get("id", record.get("entry_id", index)), **rank_summary(scored)})
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "candidates.csv", candidate_rows)
    write_rows(output_dir / "sample_metrics.csv", metric_rows)
    summary = {
        "num_eligible_examples": len({row["sample_id"] for row in candidate_rows}),
        "scored": model is not None,
        "seed": args.seed,
        "metrics": {key: mean([float(row[key]) for row in metric_rows]) for key in ("hit_at_1", "mrr", "shuffle_rejection_rate", "mean_score_margin")} if metric_rows else {},
        "warning": "Review candidate boundaries and use held-out pathways before reporting robustness results.",
    }
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
