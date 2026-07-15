"""Train forced/damped HNN regularization using the shared SFT and AE."""

import os

from experiments._launch import (
    asset_path,
    controlled_training_budget_args,
    run_torchrun_module,
    seeded_asset_path,
)


if __name__ == "__main__":
    os.environ.setdefault("CHATPATHWAY_NPROC_PER_NODE", "1")
    run_torchrun_module(
        "method.training.framework_a_ddp",
        [
            *controlled_training_budget_args(),
            "--variant", "forced_damped_hnn",
            "--structure-mode", "orthogonal_poisson",
            "--damping-mode", "isotropic",
            "--dynamics-resolution", "substep_multiscale",
            "--max-dynamics-steps", "512",
            "--lr", "1e-5",
            "--epochs", "3",
            "--kl-weight", "0.02",
            "--gradient-conflict-interval", "100",
            "--base-model", asset_path("models/qwen3_8B"),
            "--sft-lora", seeded_asset_path("checkpoints/shared/pathway_sft/checkpoint_best"),
            "--ae-ckpt", seeded_asset_path("checkpoints/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt"),
            "--train", asset_path("data/pathway_v4_full/train_pathway_continuation_v4.csv"),
            "--validation", asset_path("data/pathway_v4_full/validation_pathway_continuation_v4.csv"),
            "--save-dir", seeded_asset_path("checkpoints/experiments/exp002_forced_damped_hnn_reconae_joint_direct/final_lora"),
        ],
    )
