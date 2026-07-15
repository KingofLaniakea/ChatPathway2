"""Run the exact joint-stage pipeline with every dynamics loss disabled."""

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
            "--variant", "hnn",
            "--structure-mode", "orthogonal_poisson",
            "--damping-mode", "isotropic",
            "--lambda-align", "0",
            "--lambda-state", "0",
            "--lambda-latent-state", "0",
            "--lambda-structure", "0",
            "--dynamics-lr", "0",
            "--base-model", asset_path("models/qwen3_8B"),
            "--sft-lora", seeded_asset_path("checkpoints/shared/pathway_sft/checkpoint_best"),
            "--ae-ckpt", seeded_asset_path("checkpoints/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt"),
            "--train", asset_path("data/pathway_v4_full/train_pathway_continuation_v4.csv"),
            "--validation", asset_path("data/pathway_v4_full/validation_pathway_continuation_v4.csv"),
            "--save-dir", seeded_asset_path("checkpoints/experiments/exp003_stage2_sft_only_direct/final_lora"),
        ],
    )
