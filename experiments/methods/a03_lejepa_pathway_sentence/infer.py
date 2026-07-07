"""Run the pathway LeJEPA probe."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.inference.lejepa_pathway",
        [
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--adapter",
            "/root/autodl-tmp/checkpoints/legacy/qwen3_8b_sft/checkpoint_epoch_5",
            "--checkpoint",
            "/root/autodl-tmp/checkpoints/pathway_lejepa_sentence/lejepa_epoch_3.pt",
            "--input",
            "/root/autodl-tmp/data/test_7_species_dataset.csv",
            "--output",
            "/root/autodl-tmp/runs/lejepa_pathway/test_7_species_lejepa_scores.jsonl",
        ],
    )
