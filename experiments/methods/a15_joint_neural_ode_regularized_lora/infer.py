"""Run direct generation with the jointly trained Neural ODE-regularized adapter."""

from experiments._launch import run_module


if __name__ == "__main__":
    run_module(
        "method.inference.pathway",
        [
            "--base-model",
            "/root/autodl-tmp/models/qwen3_8B",
            "--adapter",
            "/root/autodl-tmp/checkpoints/joint_lora_dynamics/neural_ode/checkpoint_epoch_3",
            "--input",
            "/root/autodl-tmp/data/test_7_species_dataset.csv",
            "--output",
            "/root/autodl-tmp/runs/inference/joint_lora_dynamics/neural_ode_epoch3.csv",
        ],
    )
