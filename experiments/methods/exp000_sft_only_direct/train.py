"""Verify the shared SFT checkpoint used by the direct baseline."""

from experiments._launch import run_module, seeded_asset_path


if __name__ == "__main__":
    run_module(
        "experiments.artifact_check",
        [
            "--path", seeded_asset_path("checkpoints/shared/pathway_sft/checkpoint_best"),
            "--path", seeded_asset_path("checkpoints/shared/pathway_sft/run_complete.json"),
        ],
    )
