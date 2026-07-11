#!/usr/bin/env python3
"""Task 5: perturbed-cell transfer in a fixed, aligned gene space.

The evaluator consumes paired control/observed/predicted matrices and a strict
manifest.  It checks cell order, gene order, normalization, perturbation IDs,
and held-out split provenance before reusing the maintained expression/delta
metrics.  This is a transfer task: a pathway-text checkpoint must first be
adapted to the cellular representation; the pathway model is not directly
compatible with AnnData or Cell2Sentence inputs without that adapter stage.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from downstream.common.io import mean, write_json, write_rows
from downstream.tasks.task6_perturbed_cell import evaluate as evaluate_expression
from downstream.new_tasks.schemas import (
    SchemaError,
    ensure_unique,
    load_json_object,
    require_choice,
    require_fields,
    require_integer,
    require_mapping,
    require_sequence,
    require_text,
    validate_finite_array,
)


def validate_manifest(value: dict[str, Any], cells: int, genes: int, has_baseline: bool) -> dict[str, Any]:
    require_fields(
        value,
        (
            "schema_version",
            "dataset_id",
            "dataset_version",
            "split",
            "representation",
            "normalization",
            "control_matching",
            "gene_ids",
            "cell_ids",
            "perturbation_ids",
            "model_provenance",
        ),
        "manifest",
    )
    if require_integer(value["schema_version"], "manifest.schema_version") != 1:
        raise SchemaError("manifest.schema_version must be 1.")
    require_text(value["dataset_id"], "manifest.dataset_id")
    require_text(value["dataset_version"], "manifest.dataset_version")
    require_choice(value["split"], ("test_seen_perturbation", "test_unseen_perturbation"), "manifest.split")
    require_choice(value["representation"], ("normalized_expression", "c2s_rank_score"), "manifest.representation")
    require_text(value["normalization"], "manifest.normalization")
    require_text(value["control_matching"], "manifest.control_matching")
    gene_ids = list(require_sequence(value["gene_ids"], "manifest.gene_ids", nonempty=True))
    cell_ids = list(require_sequence(value["cell_ids"], "manifest.cell_ids", nonempty=True))
    perturbation_ids = list(require_sequence(value["perturbation_ids"], "manifest.perturbation_ids", nonempty=True))
    if len(gene_ids) != genes:
        raise SchemaError(f"manifest.gene_ids has {len(gene_ids)} entries but matrices have {genes} genes.")
    if len(cell_ids) != cells or len(perturbation_ids) != cells:
        raise SchemaError("manifest.cell_ids and perturbation_ids must match the matrix row count.")
    ensure_unique(gene_ids, "manifest.gene_ids")
    ensure_unique(cell_ids, "manifest.cell_ids")
    if any(not str(value).strip() for value in perturbation_ids):
        raise SchemaError("manifest.perturbation_ids must not contain empty values.")
    model = require_mapping(value["model_provenance"], "manifest.model_provenance")
    require_fields(model, ("base_checkpoint", "task_adapter_checkpoint", "training_data_id"), "manifest.model_provenance")
    for key in ("base_checkpoint", "task_adapter_checkpoint", "training_data_id"):
        require_text(model[key], f"manifest.model_provenance.{key}")
    if has_baseline:
        require_text(value.get("controlled_ablation_id"), "manifest.controlled_ablation_id")
    return dict(value)


def _validate_matrices(arrays: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    available = set(arrays.files if hasattr(arrays, "files") else arrays.keys())
    required = {"control", "observed", "predicted"}
    missing = sorted(required - available)
    if missing:
        raise SchemaError(f"Task 5 NPZ is missing arrays: {', '.join(missing)}.")
    control = np.asarray(arrays["control"])
    observed = np.asarray(arrays["observed"])
    predicted = np.asarray(arrays["predicted"])
    for name, value in (("control", control), ("observed", observed), ("predicted", predicted)):
        validate_finite_array(value, name, ndim=2)
    if control.shape != observed.shape or observed.shape != predicted.shape:
        raise SchemaError("control, observed, and predicted must have identical [cells, genes] shape.")
    baseline = np.asarray(arrays["baseline_predicted"]) if "baseline_predicted" in available else None
    if baseline is not None:
        validate_finite_array(baseline, "baseline_predicted", ndim=2)
        if baseline.shape != control.shape:
            raise SchemaError("baseline_predicted must match the primary matrices.")
    return control, observed, predicted, baseline


def evaluate_npz(
    arrays: Any,
    manifest: dict[str, Any],
    *,
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if top_k < 1:
        raise SchemaError("top_k must be positive.")
    control, observed, predicted, baseline = _validate_matrices(arrays)
    metadata = validate_manifest(manifest, control.shape[0], control.shape[1], baseline is not None)
    rows, primary = evaluate_expression(control, observed, predicted, top_k)
    for index, row in enumerate(rows):
        row["cell_id"] = str(metadata["cell_ids"][index])
        row["perturbation_id"] = str(metadata["perturbation_ids"][index])

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["perturbation_id"])].append(row)
    group_rows = []
    metric_names = tuple(primary["metrics"])
    for perturbation_id, group in sorted(groups.items()):
        group_rows.append({
            "perturbation_id": perturbation_id,
            "num_cells": len(group),
            **{name: mean([float(row[name]) for row in group]) for name in metric_names},
        })

    comparison = {"available": baseline is not None}
    if baseline is not None:
        baseline_rows, baseline_summary = evaluate_expression(control, observed, baseline, top_k)
        comparison.update({
            "controlled_ablation_id": metadata["controlled_ablation_id"],
            "baseline_metrics": baseline_summary["metrics"],
            "mean_metric_difference": {
                name: float(primary["metrics"][name]) - float(baseline_summary["metrics"][name])
                for name in metric_names
            },
            "paired_win_rate": {
                name: mean([float(float(row[name]) > float(base[name])) for row, base in zip(rows, baseline_rows)])
                for name in metric_names
            },
        })
    return rows, group_rows, {
        "task": "task5_perturbed_cell_transfer",
        "manifest": metadata,
        "metrics": primary,
        "controlled_baseline": comparison,
        "compatibility_policy": (
            "Pathway SFT/HNN checkpoints require an explicit cell-representation adapter or post-training stage; "
            "this evaluator does not make the base pathway model AnnData-compatible."
        ),
        "claim_warning": (
            "Attributing transfer gains to the Hamiltonian prior requires a controlled baseline with the same cell data, "
            "split, adapter capacity, and optimization recipe."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="NPZ with control/observed/predicted matrices.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    with np.load(args.input, allow_pickle=False) as arrays:
        rows, group_rows, summary = evaluate_npz(arrays, load_json_object(args.manifest), top_k=args.top_k)
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "cell_metrics.csv", rows)
    write_rows(output_dir / "perturbation_metrics.csv", group_rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
