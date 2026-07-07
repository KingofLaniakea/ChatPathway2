"""Run the pathway LeJEPA probe."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "method.inference.lejepa_pathway",
        [
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--adapter",
            asset_path("checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5"),
            "--checkpoint",
            asset_path("checkpoints/pathway_lejepa_sentence/lejepa_epoch_3.pt"),
            "--input",
            asset_path("data/test_7_species_dataset.csv"),
            "--output",
            asset_path("runs/lejepa_pathway/test_7_species_lejepa_scores.jsonl"),
        ],
    )
