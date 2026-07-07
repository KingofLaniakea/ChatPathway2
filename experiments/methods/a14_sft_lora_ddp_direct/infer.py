"""Run direct generation with the distributed SFT LoRA adapter."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.inference.pathway",
        [
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--adapter",
            "/root/autodl-tmp/checkpoints/qwen3_8b_sft_ddp/checkpoint_epoch_5",
            "--input",
            "/root/autodl-tmp/data/test_7_species_dataset.csv",
            "--output",
            "/root/autodl-tmp/runs/inference/sft_ddp/test_7_species_sft_ddp_epoch5.csv",
        ],
    )
