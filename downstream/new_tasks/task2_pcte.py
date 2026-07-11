#!/usr/bin/env python3
"""Task 2: representation-level Physics-Consistency Trajectory Error (PCTE).

PCTE aligns predicted-answer and gold-answer latent trajectories with DTW.  It
does not run the HNN, and low PCTE is not by itself proof that two pathways are
biologically equivalent.  The manifest pins the held-out split and the exact
backbone/adapter/AE representation used for both sides. A causal-substep
granularity additionally requires independent ordering provenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from downstream.common.io import mean, write_json, write_rows
from downstream.tasks.task3_pcte import dtw_distance
from downstream.new_tasks.schemas import (
    SchemaError,
    load_json_object,
    require_choice,
    require_fields,
    require_integer,
    require_mapping,
    require_text,
    validate_finite_array,
)


def validate_manifest(value: dict[str, Any]) -> dict[str, Any]:
    require_fields(value, ("schema_version", "dataset_id", "split", "granularity", "representation"), "manifest")
    if require_integer(value["schema_version"], "manifest.schema_version") != 1:
        raise SchemaError("manifest.schema_version must be 1.")
    require_text(value["dataset_id"], "manifest.dataset_id")
    require_choice(value["split"], ("validation", "test"), "manifest.split")
    granularity = require_choice(
        value["granularity"], ("token", "graph_layer", "causal_substep"), "manifest.granularity"
    )
    if granularity == "causal_substep":
        ordering = require_mapping(value.get("ordering_provenance"), "manifest.ordering_provenance")
        require_fields(ordering, ("source", "version"), "manifest.ordering_provenance")
        require_text(ordering["source"], "manifest.ordering_provenance.source")
        require_text(ordering["version"], "manifest.ordering_provenance.version")
    representation = require_mapping(value["representation"], "manifest.representation")
    require_fields(
        representation,
        ("base_checkpoint", "adapter_checkpoint", "ae_checkpoint"),
        "manifest.representation",
    )
    require_text(representation["base_checkpoint"], "manifest.representation.base_checkpoint")
    require_text(representation["adapter_checkpoint"], "manifest.representation.adapter_checkpoint")
    require_text(representation["ae_checkpoint"], "manifest.representation.ae_checkpoint")
    return dict(value)


def _length_array(value: Any, name: str, samples: int, max_points: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (samples,) or not np.issubdtype(raw.dtype, np.integer):
        raise SchemaError(f"{name} must be an integer array shaped [{samples}].")
    result = raw.astype(int)
    if np.any(result < 1) or np.any(result > max_points):
        raise SchemaError(f"{name} values must lie in [1, {max_points}].")
    return result


def evaluate_arrays(
    predicted: np.ndarray,
    target: np.ndarray,
    predicted_lengths: np.ndarray,
    target_lengths: np.ndarray,
    *,
    metric: str = "cosine",
    max_length: int = 256,
    sample_ids: list[Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_finite_array(predicted, "predicted_latents", ndim=3)
    validate_finite_array(target, "target_latents", ndim=3)
    if predicted.shape[0] != target.shape[0] or predicted.shape[2] != target.shape[2]:
        raise SchemaError("predicted/target latents must match in sample count and latent dimension.")
    if metric not in {"cosine", "euclidean"}:
        raise SchemaError("metric must be 'cosine' or 'euclidean'.")
    if max_length < 1:
        raise SchemaError("max_length must be positive.")
    samples = predicted.shape[0]
    pred_lengths = _length_array(predicted_lengths, "predicted_lengths", samples, predicted.shape[1])
    gold_lengths = _length_array(target_lengths, "target_lengths", samples, target.shape[1])
    if sample_ids is not None and len(sample_ids) != samples:
        raise SchemaError("sample_ids length must match the latent sample count.")
    rows = []
    for index in range(samples):
        left = predicted[index, : pred_lengths[index]]
        right = target[index, : gold_lengths[index]]
        distance, path_length = dtw_distance(left, right, metric, max_length)
        rows.append({
            "sample_id": sample_ids[index] if sample_ids is not None else index,
            "predicted_points": int(pred_lengths[index]),
            "target_points": int(gold_lengths[index]),
            "pcte": distance,
            "dtw_path_length": path_length,
        })
    return rows, {
        "num_samples": len(rows),
        "metric": metric,
        "mean_pcte": mean([float(row["pcte"]) for row in rows]),
        "median_pcte": float(np.median([row["pcte"] for row in rows])) if rows else 0.0,
    }


def evaluate_npz(
    arrays: Any,
    manifest: dict[str, Any],
    *,
    metric: str = "cosine",
    max_length: int = 256,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata = validate_manifest(manifest)
    available = set(arrays.files if hasattr(arrays, "files") else arrays.keys())
    required = {"predicted_latents", "target_latents", "predicted_lengths", "target_lengths"}
    missing = sorted(required - available)
    if missing:
        raise SchemaError(f"PCTE NPZ is missing arrays: {', '.join(missing)}.")
    sample_ids = arrays["sample_ids"].tolist() if "sample_ids" in available else None
    rows, metrics = evaluate_arrays(
        arrays["predicted_latents"],
        arrays["target_latents"],
        arrays["predicted_lengths"],
        arrays["target_lengths"],
        metric=metric,
        max_length=max_length,
        sample_ids=sample_ids,
    )
    return rows, {
        "task": "task2_pcte",
        "manifest": metadata,
        "metrics": metrics,
        "interpretation_warning": (
            "PCTE measures agreement in one fixed learned representation. It is distinct from HNN rollout "
            "self-consistency and requires separate biological correctness metrics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="NPZ with paired latent trajectories and explicit lengths.")
    parser.add_argument("--manifest", required=True, help="Task 2 provenance JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metric", choices=("cosine", "euclidean"), default="cosine")
    parser.add_argument("--dtw-max-length", type=int, default=256)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as arrays:
        rows, summary = evaluate_npz(
            arrays,
            load_json_object(args.manifest),
            metric=args.metric,
            max_length=args.dtw_max_length,
        )
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
