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
import json
from pathlib import Path

import numpy as np

from downstream.io import mean, write_json, write_rows


def c2s_genes(text: str) -> list[str]:
    return [token.strip(".,;:()[]") for token in str(text or "").replace("<\\ctrl100>", "").split() if token.strip(".,;:()[]")]


def control_sentence(instruction: str) -> str:
    marker = "Control cell sentence: "
    if marker not in instruction:
        return ""
    return instruction.split(marker, 1)[1].split(".\n\nPerturbed", 1)[0]


def c2s_vectors(prediction_path: str, train_path: str, top_genes: int, limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode existing C2S JSONL artifacts into a fixed rank-expression space."""
    counts: dict[str, int] = {}
    with Path(train_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = f"{record.get('output', '')} {control_sentence(record.get('instruction', ''))}"
            for gene in c2s_genes(text):
                counts[gene] = counts.get(gene, 0) + 1
    vocabulary = [gene for gene, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_genes]]
    if not vocabulary:
        raise ValueError("No genes were found while building the C2S training vocabulary.")
    gene_to_index = {gene: index for index, gene in enumerate(vocabulary)}

    def vectorize(text: str) -> np.ndarray:
        genes = c2s_genes(text)
        vector = np.zeros(len(vocabulary), dtype=float)
        for rank, gene in enumerate(genes):
            if gene in gene_to_index:
                vector[gene_to_index[gene]] = len(genes) - rank
        return vector

    control, observed, predicted = [], [], []
    with Path(prediction_path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            control_text = record.get("control_base", control_sentence(record.get("instruction", "")))
            target_text = record.get("ground_truth", record.get("output", ""))
            prediction_text = record.get("prediction", record.get("predicted_answer", ""))
            control.append(vectorize(control_text))
            observed.append(vectorize(target_text))
            predicted.append(vectorize(prediction_text))
            if limit is not None and len(control) >= limit:
                break
    if not control:
        raise ValueError("The C2S prediction JSONL contains no usable records.")
    return np.asarray(control), np.asarray(observed), np.asarray(predicted)


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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="NPZ with control, observed, and predicted arrays.")
    source.add_argument("--c2s-predictions", help="C2S prediction JSONL with control_base, ground_truth, prediction.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--c2s-train", help="C2S training JSONL used to build a fixed gene vocabulary.")
    parser.add_argument("--top-genes", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.input:
        with np.load(args.input) as arrays:
            required = ("control", "observed", "predicted")
            missing = [name for name in required if name not in arrays]
            if missing:
                raise ValueError(f"Missing arrays: {', '.join(missing)}")
            control, observed, predicted = arrays["control"], arrays["observed"], arrays["predicted"]
        input_mode = "matrix_npz"
    else:
        if not args.c2s_train:
            parser.error("--c2s-train is required with --c2s-predictions.")
        control, observed, predicted = c2s_vectors(args.c2s_predictions, args.c2s_train, args.top_genes, args.limit)
        input_mode = "c2s_rank_text"
    rows, summary = evaluate(control, observed, predicted, args.top_k)
    summary["input_mode"] = input_mode
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
