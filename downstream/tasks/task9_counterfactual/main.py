#!/usr/bin/env python3
"""Task IX: counterfactual pathway perturbation trajectory evaluation.

Consumes an NPZ with aligned ``control``, ``predicted``, and ``target`` latent
trajectories shaped ``[cases, steps, latent_dim]``. ``predicted`` is the model's
post-intervention trajectory and ``target`` is the observed/annotated
post-intervention trajectory. The evaluator measures the intervention effect,
not just closeness to the unperturbed state.

The current FrameworkA HNN has no intervention input ``u``. This module will
not misrepresent it as a counterfactual generator; a conditioned dynamics
model and paired perturbation data are required upstream.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from downstream.common.io import mean, write_json, write_rows
from downstream.tasks.task3_pcte import dtw_distance, remove_padding


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12)
    return float(np.dot(left, right) / denominator)


def evaluate(control: np.ndarray, predicted: np.ndarray, target: np.ndarray, max_length: int) -> tuple[list[dict[str, float | int]], dict[str, object]]:
    if control.shape != predicted.shape or predicted.shape != target.shape or control.ndim != 3:
        raise ValueError("control, predicted, and target must have identical [cases, steps, latent_dim] shape.")
    rows = []
    for index, (base, guess, truth) in enumerate(zip(control, predicted, target)):
        base, guess, truth = remove_padding(base), remove_padding(guess), remove_padding(truth)
        if not len(base) or not len(guess) or not len(truth):
            continue
        pcte, path_length = dtw_distance(guess, truth, "cosine", max_length)
        predicted_effect = guess[-1] - base[min(len(base), len(guess)) - 1]
        target_effect = truth[-1] - base[min(len(base), len(truth)) - 1]
        rows.append({
            "case_id": index,
            "counterfactual_pcte": pcte,
            "dtw_path_length": path_length,
            "endpoint_l2": float(np.linalg.norm(guess[-1] - truth[-1])),
            "effect_cosine": cosine(predicted_effect, target_effect),
            "predicted_effect_norm": float(np.linalg.norm(predicted_effect)),
            "target_effect_norm": float(np.linalg.norm(target_effect)),
        })
    return rows, {"num_cases": len(rows), "metrics": {
        key: mean([float(row[key]) for row in rows])
        for key in ("counterfactual_pcte", "endpoint_l2", "effect_cosine", "predicted_effect_norm", "target_effect_norm")
    }}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="NPZ with control, predicted, and target latent arrays.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dtw-max-length", type=int, default=256)
    args = parser.parse_args()
    with np.load(args.input) as arrays:
        required = ("control", "predicted", "target")
        missing = [name for name in required if name not in arrays]
        if missing:
            raise ValueError(f"Missing arrays: {', '.join(missing)}")
        rows, summary = evaluate(arrays["control"], arrays["predicted"], arrays["target"], args.dtw_max_length)
    summary["warning"] = "A reportable result needs paired intervention trajectories and an intervention-conditioned generator."
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
