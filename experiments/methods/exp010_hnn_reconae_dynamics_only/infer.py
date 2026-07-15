"""Verify the validation-selected HNN pretraining artifact."""

from experiments._launch import run_module, seeded_asset_path


if __name__ == "__main__":
    root = "checkpoints/experiments/exp010_hnn_reconae_dynamics_only/dynamics_pretrain"
    run_module(
        "experiments.artifact_check",
        [
            "--path", seeded_asset_path(f"{root}/checkpoint_best/hamiltonian_dynamics.pt"),
            "--path", seeded_asset_path(f"{root}/run_complete.json"),
        ],
    )
