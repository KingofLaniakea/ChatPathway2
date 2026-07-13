"""Direct greedy inference for the forced/damped HNN-trained adapter."""

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
            "--adapter", seeded_asset_path("checkpoints/experiments/exp002_forced_damped_hnn_reconae_joint_direct/final_lora/checkpoint_best"),
            "--require-complete", seeded_asset_path("checkpoints/experiments/exp002_forced_damped_hnn_reconae_joint_direct/final_lora/run_complete.json"),
            "--input", asset_path("data/pathway_v3_cap256/test_pathway_continuation_v3.csv"),
            "--output", seeded_asset_path("runs/experiments/exp002_forced_damped_hnn_reconae_joint_direct/direct.csv"),
            "--progress-output", seeded_asset_path("runs/experiments/exp002_forced_damped_hnn_reconae_joint_direct/direct.progress.jsonl"),
        ],
    )
