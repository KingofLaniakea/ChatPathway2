"""Run inference/evaluation for this experiment."""

from experiments._launch import (
    asset_path,
    controlled_inference_budget_args,
    run_module,
    seeded_asset_path,
)


if __name__ == "__main__":
    run_module(
        "method.inference.pathway",
        [
            *controlled_inference_budget_args(),
            "--base-model", asset_path("models/qwen3_8B"),
            "--adapter", seeded_asset_path("checkpoints/shared/pathway_sft/checkpoint_best"),
            "--require-complete", seeded_asset_path("checkpoints/shared/pathway_sft/run_complete.json"),
            "--input", asset_path("data/test_kegg_pathway_eval.csv"),
            "--output", seeded_asset_path("runs/experiments/exp000_sft_only_direct/direct.csv"),
        ],
    )
