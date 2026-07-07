"""Train the SFT LoRA baseline through torchrun/DDP."""

from experiments._launch import run_torchrun_module


if __name__ == "__main__":
    run_torchrun_module(
        "method.training.sft",
        [
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--train",
            "/root/autodl-tmp/data/train_11_species_dataset.csv",
            "--save-dir",
            "/root/autodl-tmp/checkpoints/qwen3_8b_sft_ddp",
        ],
    )
