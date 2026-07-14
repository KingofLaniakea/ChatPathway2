from __future__ import annotations

import sys
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from dataprocess.release_contract import (
    AUDIT_SCHEMA_VERSION,
    OVERLAP_CONTRACT,
    PARTITIONS,
    PRIMARY_CSV_NAMES,
    PRIMARY_PROMPT_PROFILE,
    RECORD_JSONL_NAMES,
    RELEASE_SCHEMA_VERSION,
    SOURCE_GRAPH_HASHES_NAME,
)
from dataprocess.prompt_profiles import (
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
)
from experiments.run_cfff_matrix import (
    Job,
    build_jobs,
    run_scheduler,
    select_baseline_inference_jobs,
    validate_inputs,
)


class CfffMatrixSchedulerTests(unittest.TestCase):
    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _release_fixture(self, root: Path) -> tuple[dict[str, Path], dict[str, Path], Path]:
        model = root / "models/qwen3_8B"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "chatpathway_download_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        data = root / "data/pathway_v3_cap256"
        data.mkdir(parents=True)
        csv_paths = {
            split: data / PRIMARY_CSV_NAMES[split]
            for split in PARTITIONS
        }
        record_paths = {
            split: data / RECORD_JSONL_NAMES[split]
            for split in PARTITIONS
        }
        split_manifest = {}
        split_audit = {}
        for split in PARTITIONS:
            csv_paths[split].write_text(f"{split}\n", encoding="utf-8")
            record_paths[split].write_text(
                f'{{"split":"{split}"}}\n', encoding="utf-8"
            )
            csv_sha = self._digest(csv_paths[split])
            record_sha = self._digest(record_paths[split])
            split_manifest[split] = {
                "csv_sha256": csv_sha,
                "records_sha256": record_sha,
                "prompt_profile": PRIMARY_PROMPT_PROFILE,
                "prompt_profile_interface_applied": True,
                "prefix_horizon_interface_applied": True,
            }
            split_audit[split] = {
                "sha256": csv_sha,
                "rows": 1,
                "errors": [],
                "prompt_profiles": {PRIMARY_PROMPT_PROFILE: 1},
                "truncation_estimate": {
                    "max_length": 8192,
                    "accepted_rows_over_budget": 0,
                },
                "record_jsonl": {"sha256": record_sha},
            }
        graph_root = root / "KEGG_all_new/processed_graph"
        graph_root.mkdir(parents=True)
        graph = graph_root / "a.json"
        graph.write_text('{"graph":true}\n', encoding="utf-8")
        source_hashes = data / SOURCE_GRAPH_HASHES_NAME
        source_hashes.write_text(
            json.dumps(
                {
                    "source_graph_json": "a.json",
                    "bytes": graph.stat().st_size,
                    "sha256": self._digest(graph),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        source_sha = self._digest(source_hashes)
        control_files = {}
        control_reports = {}
        pair_checks = {}
        for profile in (
            NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
            SPECIES_NEUTRAL_IDS_NO_ORGANISM,
        ):
            control_files[profile] = {}
            for split in PARTITIONS:
                path = data / "prompt_controls" / profile / PRIMARY_CSV_NAMES[split]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{profile},{split}\n", encoding="utf-8")
                path_sha = self._digest(path)
                relative = path.relative_to(data).as_posix()
                control_files[profile][split] = {
                    "path": relative,
                    "sha256": path_sha,
                }
                control_reports[f"{profile}:{split}"] = {
                    "path": str(path),
                    "sha256": path_sha,
                    "rows": 1,
                    "errors": [],
                }
        for split in PARTITIONS:
            pair_checks[
                f"{split}:{PRIMARY_PROMPT_PROFILE}_vs_"
                f"{NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS}"
            ] = {
                "passed": True,
                "base_sample_policy": "exact_primary_set",
            }
            pair_checks[
                f"{split}:{PRIMARY_PROMPT_PROFILE}_vs_"
                f"{SPECIES_NEUTRAL_IDS_NO_ORGANISM}"
            ] = {
                "passed": True,
                "base_sample_policy": "strict_natural_neutral_subset",
            }
        manifest_value = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "max_length": 8192,
            "primary_prompt_profile": PRIMARY_PROMPT_PROFILE,
            "processed_graph_root": str(graph_root),
            "source_graph_hashes": {
                "path": source_hashes.name,
                "records": 1,
                "sha256": source_sha,
            },
            "paired_prompt_profiles": {
                "status": "published",
                "published": True,
                "files": control_files,
            },
            "prompt_controls": control_files,
            "splits": split_manifest,
        }
        manifest = data / "dataset_manifest.json"
        manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
        overlaps = {}
        for (left, right), contract in OVERLAP_CONTRACT.items():
            overlaps[f"{left}_vs_{right}"] = {
                "identity_contract": {
                    field: {"policy": "forbidden", "passed": True}
                    for field in (
                        "source_json",
                        "graph_id",
                        "view_id",
                        "record_id",
                        "base_sample_id",
                    )
                },
                "biological_contract": {
                    field: {"policy": policy, "passed": True}
                    for field, policy in contract.items()
                },
            }
        audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "release_schema_version": RELEASE_SCHEMA_VERSION,
            "status": "passed",
            "strict_failures": [],
            "max_length": 8192,
            "manifest_sha256": self._digest(manifest),
            "source_graph_hashes": {
                "status": "passed",
                "errors": [],
                "records": 1,
                "sha256": source_sha,
            },
            "paired_prompt_profiles": {
                "status": "passed",
                "manifest_published": True,
                "canonical_files_match_prompt_controls": True,
                "declared_file_reports": control_reports,
                "pair_checks": pair_checks,
            },
            "required_summary": {"strict_overlap": overlaps},
            "splits": split_audit,
        }
        audit_path = data / "data_audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        audit_path.chmod(0o444)
        return csv_paths, record_paths, audit_path

    def test_runtime_preflight_hashes_every_csv_and_record_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _csv_paths, record_paths, _audit_path = self._release_fixture(root)
            validate_inputs(root)
            record_paths["train"].chmod(0o644)
            record_paths["train"].write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "train record JSONL changed"):
                validate_inputs(root)

    def test_runtime_preflight_rejects_audited_rows_over_8192_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _csv_paths, _record_paths, audit_path = self._release_fixture(root)
            audit_path.chmod(0o644)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["splits"]["train"]["truncation_estimate"][
                "accepted_rows_over_budget"
            ] = 1
            audit_path.write_text(json.dumps(audit), encoding="utf-8")
            audit_path.chmod(0o444)
            with self.assertRaisesRegex(ValueError, "strict 8192-token"):
                validate_inputs(root)

    def test_runtime_preflight_requires_every_published_control_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _csv_paths, _record_paths, _audit_path = self._release_fixture(root)
            missing = (
                root
                / "data/pathway_v3_cap256/prompt_controls"
                / NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS
                / PRIMARY_CSV_NAMES["test_family_only"]
            )
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_inputs(root)

    def test_runtime_preflight_rehashes_referenced_source_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_fixture(root)
            (root / "KEGG_all_new/processed_graph/a.json").write_text(
                '{"graph":"changed"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "live source graph content hashes"):
                validate_inputs(root)

    def test_job_graph_uses_four_gpu_sft_and_four_disjoint_inference_shards(self) -> None:
        root = Path("/assets")
        jobs = build_jobs([11], root, "/python")
        by_key = {job.key: job for job in jobs}

        self.assertEqual(len(jobs), 65)
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
        family_shard = by_key["11:exp000:infer:test_family_only:shard2"]
        self.assertEqual(
            family_shard.command[family_shard.command.index("--input") + 1],
            "/assets/data/pathway_v3_cap256/test_family_only_pathway_continuation_v3.csv",
        )
        self.assertIn(
            Path(
                "/assets/runs/seeds/11/experiments/exp000_sft_only_direct/"
                "diagnostics/test_family_only/direct.csv"
            ),
            by_key["11:exp000:infer:test_family_only"].outputs,
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

        self.assertEqual(len(selected), 16)
        self.assertEqual(
            keys,
            {
                "11:sft",
                "11:exp000:infer",
                "11:exp000:infer:shard0",
                "11:exp000:infer:shard1",
                "11:exp000:infer:shard2",
                "11:exp000:infer:shard3",
                "11:exp000:infer:test_family_only",
                "11:exp000:infer:test_family_only:shard0",
                "11:exp000:infer:test_family_only:shard1",
                "11:exp000:infer:test_family_only:shard2",
                "11:exp000:infer:test_family_only:shard3",
                "11:exp000:infer:test_organism_only",
                "11:exp000:infer:test_organism_only:shard0",
                "11:exp000:infer:test_organism_only:shard1",
                "11:exp000:infer:test_organism_only:shard2",
                "11:exp000:infer:test_organism_only:shard3",
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
