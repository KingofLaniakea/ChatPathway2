"""Verify the shared SFT and AE artifacts."""

from experiments._launch import run_module, seeded_asset_path


if __name__ == "__main__":
    run_module(
        "experiments.artifact_check",
        [
            "--path", seeded_asset_path("checkpoints/shared/pathway_sft/checkpoint_best"),
            "--path", seeded_asset_path("checkpoints/shared/pathway_reconstruction_ae/checkpoint_best/ae_proj.pt"),
        ],
    )
