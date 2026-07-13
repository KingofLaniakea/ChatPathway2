from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

from experiments._launch import (
    controlled_inference_budget_args,
    experiment_seed,
    seeded_asset_path,
    step_commands,
)
from experiments.check_runtime_assets import rewrite_asset_path


class LaunchTests(unittest.TestCase):
    def test_controlled_inference_budget_covers_core_gold_without_runaway_generation(self) -> None:
        args = controlled_inference_budget_args()
        self.assertEqual(args[args.index("--batch-size") + 1], "1")
        self.assertEqual(args[args.index("--max-length") + 1], "8192")
        self.assertEqual(args[args.index("--max-new-tokens") + 1], "1024")

    def test_run_steps_passthrough_is_appended_to_every_stage(self) -> None:
        commands = step_commands(
            [("stage.one", ["--fixed", "a"]), ("stage.two", ["--fixed", "b"])],
            ["--epochs", "1", "--limit", "2"],
        )
        self.assertEqual(
            commands,
            [
                [sys.executable, "-m", "stage.one", "--fixed", "a", "--epochs", "1", "--limit", "2"],
                [sys.executable, "-m", "stage.two", "--fixed", "b", "--epochs", "1", "--limit", "2"],
            ],
        )

    def test_seed_is_read_from_cli_and_scopes_mutable_assets(self) -> None:
        with patch.object(sys, "argv", ["wrapper", "--seed", "20260712"]):
            with patch.dict(os.environ, {"CHATPATHWAY_ASSET_ROOT": "/assets"}, clear=False):
                self.assertEqual(experiment_seed(), "20260712")
                self.assertEqual(
                    seeded_asset_path("checkpoints/shared/model"),
                    "/assets/checkpoints/seeds/20260712/shared/model",
                )

    def test_seed_environment_override_has_precedence(self) -> None:
        with patch.object(sys, "argv", ["wrapper", "--seed=20260712"]):
            with patch.dict(os.environ, {"CHATPATHWAY_EXPERIMENT_SEED": "20260713"}, clear=False):
                self.assertEqual(experiment_seed(), "20260713")

    def test_seeded_asset_path_rejects_immutable_asset_kinds(self) -> None:
        with self.assertRaises(ValueError):
            seeded_asset_path("data/train.csv")

    def test_runtime_manifest_default_seed_can_be_rewritten(self) -> None:
        with patch.dict(os.environ, {"CHATPATHWAY_EXPERIMENT_SEED": "20260713"}, clear=False):
            resolved = rewrite_asset_path(
                "/root/autodl-tmp/checkpoints/seeds/20260711/shared/model",
                "/root/autodl-tmp",
                "/assets",
            )
        self.assertEqual(str(resolved), "/assets/checkpoints/seeds/20260713/shared/model")


if __name__ == "__main__":
    unittest.main()
