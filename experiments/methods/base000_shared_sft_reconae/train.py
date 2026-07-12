"""Train the shared SFT adapter and reconstruction AE once."""

from experiments._launch import (
    asset_path,
    controlled_training_budget_args,
    run_steps,
    seeded_asset_path,
)


if __name__ == "__main__":
    run_steps([
        (
            "torchrun:method.training.sft",
            [
                *controlled_training_budget_args(),
                "--base-model", asset_path("models/qwen3_8B"),
                "--train", asset_path("data/train_kegg_pathway_record_balanced_0p1pct.csv"),
                "--save-dir", seeded_asset_path("checkpoints/shared/pathway_sft"),
            ],
        ),
        (
            "method.training.latent_ae",
            [
                *controlled_training_budget_args(),
                "--base-model", asset_path("models/qwen3_8B"),
                "--sft-lora", seeded_asset_path("checkpoints/shared/pathway_sft/checkpoint_best"),
                "--train", asset_path("data/train_kegg_pathway_record_balanced_0p1pct.csv"),
                "--save-dir", seeded_asset_path("checkpoints/shared/pathway_reconstruction_ae"),
            ],
        ),
    ])
