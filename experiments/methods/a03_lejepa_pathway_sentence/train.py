"""Train the pathway LeJEPA probe."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.training.lejepa_pathway",
        [
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--adapter",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5",
            "--train",
            "/root/autodl-tmp/data/train_11_species_dataset.csv",
            "--save",
            "/root/autodl-tmp/checkpoints/pathway_lejepa_sentence",
        ],
    )
