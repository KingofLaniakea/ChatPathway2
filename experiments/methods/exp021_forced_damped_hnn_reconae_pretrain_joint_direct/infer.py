"""Run direct generation from the validation-composite-selected E021 LoRA."""

from experiments._launch import asset_path, controlled_inference_budget_args, run_module, seeded_asset_path


if __name__ == "__main__":
    root = "checkpoints/experiments/exp021_forced_damped_hnn_reconae_pretrain_joint_direct/final_lora"
    run_module(
        "method.inference.pathway",
        [
            *controlled_inference_budget_args(),
            "--base-model", asset_path("models/qwen3_8B"),
            "--adapter", seeded_asset_path(f"{root}/checkpoint_best"),
            "--require-complete", seeded_asset_path(f"{root}/run_complete.json"),
            "--input", asset_path("data/pathway_v4_full/test_pathway_continuation_v4.csv"),
            "--output", seeded_asset_path("runs/experiments/exp021_forced_damped_hnn_reconae_pretrain_joint_direct/direct.csv"),
            "--progress-output", seeded_asset_path("runs/experiments/exp021_forced_damped_hnn_reconae_pretrain_joint_direct/direct.progress.jsonl"),
        ],
    )
