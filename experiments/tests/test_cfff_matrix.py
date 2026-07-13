from __future__ import annotations

import sys
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from experiments.run_cfff_matrix import (
    Job,
    build_jobs,
    run_scheduler,
    select_baseline_inference_jobs,
    validate_inputs,
)


class CfffMatrixSchedulerTests(unittest.TestCase):
    def test_runtime_preflight_hashes_every_csv_and_record_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "models/qwen3_8B"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "chatpathway_download_manifest.json").write_text(
                "{}\n", encoding="utf-8"
            )
            data = root / "data/pathway_v3_cap256"
            data.mkdir(parents=True)
            csv_paths = {
                "train": data / "train_pathway_continuation_v3_cap256.csv",
                "validation": data / "validation_pathway_continuation_v3.csv",
                "test": data / "test_pathway_continuation_v3.csv",
            }
            record_paths = {
                split: data / f"{split}_pathway_records_v3.jsonl"
                for split in csv_paths
            }
            for split, path in csv_paths.items():
                path.write_text(f"{split}\n", encoding="utf-8")
                record_paths[split].write_text(f'{{"split":"{split}"}}\n', encoding="utf-8")
            manifest = data / "dataset_manifest.json"
            manifest.write_text("{}\n", encoding="utf-8")

            def digest(path: Path) -> str:
                return hashlib.sha256(path.read_bytes()).hexdigest()

            audit = {
                "status": "passed",
                "strict_failures": [],
                "manifest_sha256": digest(manifest),
                "splits": {
                    split: {
                        "sha256": digest(csv_paths[split]),
                        "record_jsonl": {"sha256": digest(record_paths[split])},
                    }
                    for split in csv_paths
                },
            }
            audit_path = data / "data_audit.json"
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            audit_path.chmod(0o444)
            validate_inputs(root)
            record_paths["train"].chmod(0o644)
            record_paths["train"].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train record JSONL changed"):
                validate_inputs(root)

    def test_job_graph_uses_four_gpu_sft_and_four_disjoint_inference_shards(self) -> None:
        root = Path("/assets")
        jobs = build_jobs([11], root, "/python")
        by_key = {job.key: job for job in jobs}

        self.assertEqual(len(jobs), 25)
        self.assertEqual(by_key["11:sft"].resources, 4)
        self.assertEqual(by_key["11:ae"].resources, 1)
        self.assertEqual(by_key["11:ae"].dependencies, ("11:sft",))
        fdhnn_train = by_key["11:exp002_forced_damped_hnn_reconae_joint_direct:train"]
        hnn_train = by_key["11:exp001_hnn_reconae_joint_direct:train"]
        stage2_sft_train = by_key["11:exp003_stage2_sft_only_direct:train"]
        self.assertEqual(fdhnn_train.resources, 4)
        self.assertEqual(hnn_train.resources, 2)
        self.assertEqual(stage2_sft_train.resources, 2)
        self.assertEqual(
            fdhnn_train.command[fdhnn_train.command.index("--gradient-accumulation-steps") + 1],
            "3",
        )
        for job in (hnn_train, stage2_sft_train):
            self.assertEqual(
                job.command[job.command.index("--gradient-accumulation-steps") + 1],
                "6",
            )
        inference_prefix = "11:exp001_hnn_reconae_joint_direct:infer"
        shard_keys = tuple(f"{inference_prefix}:shard{index}" for index in range(4))
        self.assertEqual(by_key[inference_prefix].dependencies, shard_keys)
        self.assertIn("method.inference.merge_pathway_shards", by_key[inference_prefix].command)
        for index, key in enumerate(shard_keys):
            shard = by_key[key]
            self.assertEqual(shard.resources, 1)
            self.assertEqual(shard.dependencies, ("11:exp001_hnn_reconae_joint_direct:train",))
            self.assertEqual(shard.command[shard.command.index("--shard-count") + 1], "4")
            self.assertEqual(shard.command[shard.command.index("--shard-index") + 1], str(index))
        self.assertIn("torch.distributed.run", by_key["11:sft"].command)
        sft_command = by_key["11:sft"].command
        self.assertEqual(sft_command[sft_command.index("--nproc_per_node") + 1], "4")
        self.assertEqual(sft_command[sft_command.index("--batch-size") + 1], "1")
        self.assertEqual(
            sft_command[sft_command.index("--gradient-accumulation-steps") + 1],
            "1",
        )
        self.assertIn(Path("/assets/checkpoints/seeds/11/shared/pathway_sft/run_complete.json"), by_key["11:sft"].outputs)
        self.assertIn(Path("/assets/checkpoints/seeds/11/shared/pathway_reconstruction_ae/run_complete.json"), by_key["11:ae"].outputs)
        self.assertIn(
            Path("/assets/checkpoints/seeds/11/experiments/exp001_hnn_reconae_joint_direct/final_lora/run_complete.json"),
            by_key["11:exp001_hnn_reconae_joint_direct:train"].outputs,
        )
        self.assertIn(
            Path("/assets/runs/seeds/11/experiments/exp000_sft_only_direct/direct.progress.jsonl"),
            by_key["11:exp000:infer"].outputs,
        )
        self.assertIn(
            Path(
                "/assets/runs/seeds/11/experiments/exp000_sft_only_direct/"
                "direct.progress.shard-00002-of-00004.jsonl"
            ),
            by_key["11:exp000:infer:shard2"].outputs,
        )
        self.assertEqual(
            by_key["11:exp000:infer:shard2"].skip_if_outputs,
            by_key["11:exp000:infer"].outputs,
        )
        for key in ("11:sft", "11:ae"):
            command = by_key[key].command
            self.assertEqual(command[command.index("--max-length") + 1], "8192")
            self.assertEqual(command[command.index("--batch-size") + 1], "1")
            self.assertEqual(
                command[command.index("--validation-group-column") + 1],
                "pathway_family_id",
            )

    def test_dry_run_does_not_require_runtime_assets(self) -> None:
        jobs = build_jobs([11], Path("/missing"), "/python")
        with tempfile.TemporaryDirectory() as directory:
            with redirect_stdout(StringIO()):
                result = run_scheduler(
                    jobs,
                    gpus=["0", "1", "2", "3"],
                    profile="cfff",
                    log_dir=Path(directory),
                    poll_seconds=0.01,
                    skip_existing=True,
                    dry_run=True,
                )
        self.assertEqual(result, 0)

    def test_baseline_only_keeps_sft_dependency_four_shards_and_merge(self) -> None:
        selected = select_baseline_inference_jobs(build_jobs([11], Path("/assets"), "/python"))
        keys = {job.key for job in selected}

        self.assertEqual(len(selected), 6)
        self.assertEqual(
            keys,
            {
                "11:sft",
                "11:exp000:infer",
                "11:exp000:infer:shard0",
                "11:exp000:infer:shard1",
                "11:exp000:infer:shard2",
                "11:exp000:infer:shard3",
            },
        )
        for job in selected:
            self.assertTrue(set(job.dependencies).issubset(keys))

    def test_scheduler_allocates_four_then_two_plus_two_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_output = root / "first.txt"
            left_output = root / "left.txt"
            right_output = root / "right.txt"

            def writer(path: Path) -> tuple[str, ...]:
                code = (
                    "import os,sys; "
                    "from pathlib import Path; "
                    "Path(sys.argv[1]).write_text(os.environ['CUDA_VISIBLE_DEVICES'])"
                )
                return (sys.executable, "-c", code, str(path))

            jobs = [
                Job("first", 1, 4, (), writer(first_output), (first_output,)),
                Job("left", 1, 2, ("first",), writer(left_output), (left_output,)),
                Job("right", 1, 2, ("first",), writer(right_output), (right_output,)),
            ]
            with redirect_stdout(StringIO()):
                result = run_scheduler(
                    jobs,
                    gpus=["0", "1", "2", "3"],
                    profile="cfff",
                    log_dir=root / "logs",
                    poll_seconds=0.01,
                    skip_existing=False,
                    dry_run=False,
                )

            self.assertEqual(result, 0)
            self.assertEqual(first_output.read_text(), "0,1,2,3")
            self.assertEqual(left_output.read_text(), "0,1")
            self.assertEqual(right_output.read_text(), "2,3")


if __name__ == "__main__":
    unittest.main()
