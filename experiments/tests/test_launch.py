from __future__ import annotations

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from experiments import run_experiment
from experiments._launch import (
    controlled_inference_budget_args,
    controlled_training_budget_args,
    dataset_namespace,
    experiment_seed,
    seeded_asset_path,
    step_commands,
)
from experiments.check_runtime_assets import rewrite_asset_path


class LaunchTests(unittest.TestCase):
    def test_structured_release_entry_forwards_v31_build_controls(self) -> None:
        completed = SimpleNamespace(returncode=0)
        argv = [
            "run_experiment",
            "prepare-structured-data",
            "--workers",
            "3",
            "--worker-batch-size",
            "64",
            "--max-files",
            "2",
            "--overwrite",
        ]
        with patch.object(sys, "argv", argv), patch.object(
            run_experiment, "asset_path", side_effect=lambda value: f"/assets/{value}"
        ), patch.object(
            run_experiment.subprocess, "run", return_value=completed
        ) as run:
            with self.assertRaises(SystemExit) as exit_context:
                run_experiment.main()

        self.assertEqual(exit_context.exception.code, 0)
        command = run.call_args.args[0]
        expected = {
            "--processed-graph-root": "/assets/KEGG_all_new/processed_graph",
            "--processed-root": "/assets/KEGG_all_new/processed",
            "--max-length": "8192",
            "--evaluation-candidate-record-fraction": "1.0",
            "--seen-evaluation-candidate-record-fraction": "0.02",
            "--max-records-per-family": "256",
            "--maximum-train-records": "18000",
            "--target-train-input-tokens-per-epoch": "36000000",
            "--workers": "3",
            "--worker-batch-size": "64",
            "--max-files": "2",
        }
        for option, value in expected.items():
            self.assertEqual(command[command.index(option) + 1], value)

    def test_controlled_training_uses_one_prefix_per_record_per_epoch(self) -> None:
        args = controlled_training_budget_args()
        self.assertEqual(
            args[args.index("--prefix-sampling") + 1],
            "one_per_record",
        )

    def test_controlled_inference_budget_uses_three_strict_json_attempts(self) -> None:
        args = controlled_inference_budget_args()
        self.assertEqual(args[args.index("--batch-size") + 1], "1")
        self.assertEqual(args[args.index("--max-length") + 1], "8192")
        self.assertEqual(args[args.index("--max-new-tokens") + 1], "4096")
        self.assertEqual(args[args.index("--max-json-attempts") + 1], "3")
        self.assertEqual(args[args.index("--retry-max-new-tokens") + 1], "8192")

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
                    "/assets/checkpoints/datasets/pathway_v4_full/seeds/20260712/shared/model",
                )

    def test_dataset_namespace_scopes_mutable_assets(self) -> None:
        with patch.object(sys, "argv", ["wrapper", "--seed", "20260712"]):
            with patch.dict(
                os.environ,
                {
                    "CHATPATHWAY_ASSET_ROOT": "/assets",
                    "CHATPATHWAY_DATASET_NAMESPACE": "pathway_v4_full_deadbeef",
                },
                clear=False,
            ):
                self.assertEqual(dataset_namespace(), "pathway_v4_full_deadbeef")
                self.assertEqual(
                    seeded_asset_path("runs/experiments/example"),
                    "/assets/runs/datasets/pathway_v4_full_deadbeef/seeds/20260712/experiments/example",
                )

    def test_dataset_namespace_rejects_path_components(self) -> None:
        with patch.dict(
            os.environ,
            {"CHATPATHWAY_DATASET_NAMESPACE": "../wrong"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                dataset_namespace()

    def test_dataset_namespace_is_derived_from_manifest_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "data/pathway_v4_full"
            release.mkdir(parents=True)
            (release / "dataset_manifest.json").write_text(
                json.dumps({"dataset_build_id": "dataset:" + "b" * 24}),
                encoding="utf-8",
            )
            with patch.dict(
                os.environ,
                {"CHATPATHWAY_ASSET_ROOT": directory},
                clear=False,
            ):
                os.environ.pop("CHATPATHWAY_DATASET_NAMESPACE", None)
                self.assertEqual(
                    dataset_namespace(),
                    "pathway_v4_full_" + "b" * 24,
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
                "/root/autodl-tmp/checkpoints/datasets/pathway_v4_full/seeds/20260711/shared/model",
                "/root/autodl-tmp",
                "/assets",
            )
        self.assertEqual(
            str(resolved),
            "/assets/checkpoints/datasets/pathway_v4_full/seeds/20260713/shared/model",
        )


if __name__ == "__main__":
    unittest.main()
