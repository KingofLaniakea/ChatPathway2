#!/usr/bin/env python3
"""Task VI: perturbed-cell generation metrics.

The evaluator consumes a portable NPZ artifact containing float arrays
``control``, ``observed``, and ``predicted`` of shape ``[cells, genes]``. It
does not claim that an LLM has generated a cell profile; an upstream adapter
must first decode model output into the same normalized gene space. This keeps
metric code independent from a particular Cell2Sentence representation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from downstream.io import mean, write_json, write_rows


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(values.size, dtype=float)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + end - 1) / 2
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return pearson(average_ranks(np.asarray(left)), average_ranks(np.asarray(right)))


def evaluate(control: np.ndarray, observed: np.ndarray, predicted: np.ndarray, top_k: int) -> tuple[list[dict[str, float | int]], dict[str, float | int]]:
    if control.shape != observed.shape or observed.shape != predicted.shape or control.ndim != 2:
        raise ValueError("control, observed, and predicted must have identical [cells, genes] shape.")
    rows: list[dict[str, float | int]] = []
    for index, (base, truth, guess) in enumerate(zip(control, observed, predicted)):
        observed_delta = truth - base
        predicted_delta = guess - base
        selected = np.argsort(np.abs(observed_delta))[-min(top_k, observed_delta.size):]
        rows.append({
            "cell_index": index,
            "expression_pearson": pearson(guess, truth),
            "expression_spearman": spearman(guess, truth),
            "delta_pearson": pearson(predicted_delta, observed_delta),
            "delta_spearman": spearman(predicted_delta, observed_delta),
            "topk_de_delta_pearson": pearson(predicted_delta[selected], observed_delta[selected]),
            "topk_de_delta_spearman": spearman(predicted_delta[selected], observed_delta[selected]),
        })
    metrics = {key: mean([float(row[key]) for row in rows]) for key in rows[0] if key != "cell_index"} if rows else {}
    return rows, {"num_cells": len(rows), "top_k": min(top_k, control.shape[1]), "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="NPZ with control, observed, and predicted arrays.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    with np.load(args.input) as arrays:
        required = ("control", "observed", "predicted")
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(f"Missing arrays: {', '.join(missing)}")
        rows, summary = evaluate(arrays["control"], arrays["observed"], arrays["predicted"], args.top_k)
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
