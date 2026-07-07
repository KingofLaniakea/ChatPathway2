"""Run direct LoRA inference for a FrameworkA HNN-regularized adapter."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.inference.pathway",
        [
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--adapter",
            "/root/autodl-tmp/checkpoints/qwen3_8b_FrameworkA_ae_cos/checkpoint_epoch_4",
            "--input",
            "/root/autodl-tmp/data/test_7_species_dataset.csv",
            "--output",
            "/root/autodl-tmp/runs/inference/frameworka_ae_cos/test_7_species_frameworka_ae_cos_epoch4.csv",
        ],
    )
