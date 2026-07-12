"""Train pure HNN regularization using the shared SFT and AE."""

from experiments._launch import (
    asset_path,
    controlled_training_budget_args,
    run_steps,
    seeded_asset_path,
)


if __name__ == "__main__":
    run_steps([
        (
            "method.training.framework_a",
            [
                *controlled_training_budget_args(),
                "--variant", "hnn",
                "--structure-mode", "orthogonal_poisson",
                "--damping-mode", "isotropic",
                "--base-model", asset_path("models/qwen3_8B"),
                "--sft-lora", seeded_asset_path("checkpoints/shared/pathway_sft/checkpoint_best"),
                "--ae-ckpt", seeded_asset_path("checkpoints/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt"),
                "--train", asset_path("data/train_kegg_pathway_record_balanced_0p1pct.csv"),
                "--save-dir", seeded_asset_path("checkpoints/experiments/exp001_hnn_reconae_joint_direct/final_lora"),
            ],
        ),
    ])
