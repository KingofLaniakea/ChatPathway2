"""Run direct generation with the distributed SFT LoRA adapter."""

from experiments._launch import asset_path, run_module


if __name__ == "__main__":
    run_module(
        "method.inference.pathway",
        [
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--adapter",
            asset_path("checkpoints/qwen3_8b_sft_ddp/checkpoint_epoch_5"),
            "--input",
            asset_path("data/test_7_species_dataset.csv"),
            "--output",
            asset_path("runs/inference/sft_ddp/test_7_species_sft_ddp_epoch5.csv"),
        ],
    )
