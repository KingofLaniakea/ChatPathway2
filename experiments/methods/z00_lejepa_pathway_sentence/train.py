"""Train the pathway LeJEPA probe."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "method.training.lejepa_pathway",
        [
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--adapter",
            asset_path("checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"),
            "--train",
            asset_path("data/train_11_species_dataset.csv"),
            "--save",
            asset_path("checkpoints/pathway_lejepa_sentence"),
        ],
    )
