"""Train the SFT LoRA baseline through torchrun/DDP."""

from experiments._launch import asset_path, run_torchrun_module


if __name__ == "__main__":
    run_torchrun_module(
        "method.training.sft",
        [
            "--base-model",
            asset_path("models/qwen3_8B"),
            "--train",
            asset_path("data/train_11_species_dataset.csv"),
            "--save-dir",
            asset_path("checkpoints/qwen3_8b_sft_ddp"),
        ],
    )
