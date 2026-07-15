from __future__ import annotations

import sys
import csv
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from dataprocess.release_contract_v4 import (
    ALL_SPLITS,
    AUDIT_NAME,
    AUDIT_SCHEMA_VERSION,
    CSV_NAMES,
    MANIFEST_NAME,
    PRIMARY_PROMPT_PROFILE,
    RECORD_NAMES,
    RELEASE_SCHEMA_VERSION,
    SOURCE_GRAPH_HASHES_NAME,
    SPLIT_ASSIGNMENTS_NAME,
)
from dataprocess.structured_schema import graph_id_for_source
from dataprocess.prompt_profiles import (
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
)
from experiments.run_cfff_matrix import (
    Job,
    build_jobs,
    outputs_complete,
    run_scheduler,
    select_baseline_inference_jobs,
    validate_inputs,
)


class CfffMatrixSchedulerTests(unittest.TestCase):
    @staticmethod
    def _digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_outputs_complete_requires_successful_terminal_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            marker = Path(temporary) / "run_complete.json"
            artifact = Path(temporary) / "checkpoint.pt"
            artifact.write_bytes(b"checkpoint")
            job = Job(
                key="pretrain",
                seed=11,
                resources=1,
                dependencies=(),
                command=("true",),
                outputs=(artifact, marker),
            )
            marker.write_text('{"status":"max_epochs_without_stability"}\n', encoding="utf-8")
            self.assertFalse(outputs_complete(job))
            marker.write_text('{"status":"completed"}\n', encoding="utf-8")
            self.assertTrue(outputs_complete(job))

    def _release_fixture(self, root: Path) -> tuple[dict[str, Path], dict[str, Path], Path]:
        model = root / "models/qwen3_8B"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "chatpathway_download_manifest.json").write_text(
            "{}\n", encoding="utf-8"
        )
        data = root / "data/pathway_v4_full"
        data.mkdir(parents=True)
        csv_paths = {
            split: data / CSV_NAMES[split] for split in ALL_SPLITS
        }
        record_paths = {
            split: data / RECORD_NAMES[split] for split in ALL_SPLITS
        }
        graph_root = root / "KEGG_all_new/processed_graph"
        graph_root.mkdir(parents=True)
        graph = graph_root / "org/a.json"
        graph.parent.mkdir(parents=True)
        graph.write_text('{"graph":true}\n', encoding="utf-8")
        graph_raw = graph.read_bytes()
        source_hashes = data / SOURCE_GRAPH_HASHES_NAME
        with source_hashes.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["source_graph_json", "graph_id", "sha256", "bytes", "status"])
            writer.writerow(
                [
                    "org/a.json",
                    graph_id_for_source("org/a.json", graph_raw),
                    hashlib.sha256(graph_raw).hexdigest(),
                    len(graph_raw),
                    "ok",
                ]
            )
        source_sha = self._digest(source_hashes)
        split_assignments = data / SPLIT_ASSIGNMENTS_NAME
        split_assignments.write_text("{}\n", encoding="utf-8")
        split_assignments.chmod(0o444)

        outputs = {}
        audit_outputs = {}
        for split in ALL_SPLITS:
            csv_paths[split].write_text(f"{split}\n", encoding="utf-8")
            record_paths[split].write_text(
                f'{{"split":"{split}"}}\n', encoding="utf-8"
            )
            csv_sha = self._digest(csv_paths[split])
            record_sha = self._digest(record_paths[split])
            control_files = {}
            for profile in (
                NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
                SPECIES_NEUTRAL_IDS_NO_ORGANISM,
            ):
                path = data / "prompt_controls" / profile / CSV_NAMES[split]
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{profile},{split}\n", encoding="utf-8")
                control_files[profile] = {
                    "path": path.relative_to(data).as_posix(),
                    "sha256": self._digest(path),
                    "bytes": path.stat().st_size,
                }
            audit_outputs[split] = {
                "rows": 1,
                "records": 1,
                "graphs": 1,
                "views": 1,
                "sources": 1,
                "organisms": 1,
                "families": 1,
                "input_tokens": 10,
                "token_length": {"min": 10, "mean": 10.0, "max": 10},
                "horizons": {"short_target": 1},
                "layer_length_distribution": {"2": 1},
                "substeps": {
                    "semantic_duplicates_within_layer": 0,
                    "duplicate_participant_canonical_ids": 0,
                },
                "prompt_controls": {
                    "p1_rows": 1,
                    "p2_rows": 1,
                    "p2_ineligible_records": 0,
                    "files": control_files,
                },
                "csv_sha256": csv_sha,
                "records_sha256": record_sha,
                "csv_bytes": csv_paths[split].stat().st_size,
                "records_bytes": record_paths[split].stat().st_size,
            }
            outputs[split] = {
                **audit_outputs[split],
                "csv_file": CSV_NAMES[split],
                "records_file": RECORD_NAMES[split],
            }
        manifest_value = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "max_length": 8192,
            "primary_prompt_profile": PRIMARY_PROMPT_PROFILE,
            "processed_graph_root": str(graph_root),
            "canonical_index": {
                "path": str(data / "canonical_index_v4.sqlite3"),
                "sha256": "canonical-sha",
                "processed_graph_root": str(graph_root),
                "summary": {"graphs": 1, "records": len(ALL_SPLITS)},
            },
            "source_graph_hashes": {
                "path": source_hashes.name,
                "sources": 1,
                "sha256": source_sha,
                "bytes": source_hashes.stat().st_size,
            },
            "outputs": outputs,
        }
        manifest = data / MANIFEST_NAME
        manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
        pinned_paths = [
            manifest,
            source_hashes,
            split_assignments,
            *csv_paths.values(),
            *record_paths.values(),
            *sorted(data.glob("prompt_controls/*/*.csv")),
        ]
        pinned_files = {
            path.relative_to(data).as_posix(): {
                "sha256": self._digest(path),
                "bytes": path.stat().st_size,
            }
            for path in pinned_paths
        }
        overlaps = {
            f"{left}__{right}": {
                key: 0
                for key in (
                    "samples",
                    "records",
                    "graphs",
                    "views",
                    "sources",
                    "families",
                    "organisms",
                )
            }
            for index, left in enumerate(ALL_SPLITS)
            for right in ALL_SPLITS[index + 1 :]
        }
        audit = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "release_schema_version": RELEASE_SCHEMA_VERSION,
            "status": "passed",
            "failures": [],
            "canonical_index": {
                "complete": True,
                "processed_graph_root": str(graph_root),
                "inventory": {"graph_files": 1},
                "summary": {"graphs": 1, "records": len(ALL_SPLITS)},
                "database_sha256": "canonical-sha",
            },
            "canonical_assignment_coverage": {
                "records": len(ALL_SPLITS),
                "assigned_records": len(ALL_SPLITS),
                "unassigned_records": 0,
            },
            "source_holdout": {
                "policy": "dataset_internal_coverage_quantile_stratified_source_holdout",
                "claims_phylogenetic_balance": False,
            },
            "materialized_splits": audit_outputs,
            "strict_overlap": overlaps,
            "duplicate_ids": {
                "canonical_record_ids": 0,
                "source_graph_paths": 0,
                "source_graph_ids": 0,
                "materialized_sample_ids_within_splits": 0,
                "materialized_sample_ids_across_splits": 0,
                "materialized_record_ids_across_splits": 0,
            },
            "horizon_balance": {
                split: {"actual_matches_theoretical_optimum": True}
                for split in ALL_SPLITS
            },
            "token_and_truncation": {"max_length": 8192},
            "graph_artifact_coverage": {
                "selected_source_hashes": {
                    "sources": 1,
                    "sha256": source_sha,
                    "bytes": source_hashes.stat().st_size,
                },
                "processed_text_counterparts": {
                    "status": "complete",
                    "checked_sources": 1,
                    "indexed_sources": 1,
                    "missing_counterparts": 0,
                    "visible_legacy_text_events": 1,
                    "exact_legacy_text_matches": 1,
                    "unmatched_legacy_text_events": 0,
                },
            },
            "hashes": {"files": pinned_files},
        }
        audit_path = data / AUDIT_NAME
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
            with self.assertRaisesRegex(ValueError, "release artifact changed after audit"):
                validate_inputs(root)

    def test_runtime_preflight_rejects_audited_rows_over_8192_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _csv_paths, _record_paths, audit_path = self._release_fixture(root)
            audit_path.chmod(0o644)
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit["materialized_splits"]["train"]["token_length"]["max"] = 8193
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
                / "data/pathway_v4_full/prompt_controls"
                / NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS
                / CSV_NAMES["test_strict"]
            )
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                validate_inputs(root)

    def test_runtime_preflight_rehashes_referenced_source_graphs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._release_fixture(root)
            (root / "KEGG_all_new/processed_graph/org/a.json").write_text(
                '{"graph":"changed"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "live source graph content hashes"):
                validate_inputs(root)

    def test_job_graph_uses_four_gpu_sft_and_four_disjoint_inference_shards(self) -> None:
        root = Path("/assets")
        jobs = build_jobs([11], root, "/python")
        by_key = {job.key: job for job in jobs}

        self.assertEqual(len(jobs), 99)
        self.assertEqual(by_key["11:sft"].resources, 4)
        self.assertEqual(by_key["11:ae"].resources, 1)
        self.assertEqual(by_key["11:ae"].dependencies, ("11:sft",))
        self.assertEqual(
            by_key["11:sft"].command[by_key["11:sft"].command.index("--epochs") + 1],
            "1",
        )
        self.assertEqual(
            by_key["11:ae"].command[by_key["11:ae"].command.index("--epochs") + 1],
            "3",
        )
        fdhnn_train = by_key["11:exp002_forced_damped_hnn_reconae_joint_direct:train"]
        hnn_train = by_key["11:exp001_hnn_reconae_joint_direct:train"]
        stage2_sft_train = by_key["11:exp003_stage2_sft_only_direct:train"]
        self.assertEqual(fdhnn_train.resources, 1)
        self.assertEqual(hnn_train.resources, 1)
        self.assertEqual(stage2_sft_train.resources, 1)
        for job in (hnn_train, stage2_sft_train):
            self.assertEqual(
                job.command[job.command.index("--gradient-accumulation-steps") + 1],
                "12",
            )
        hnn_pretrain = by_key["11:exp010_hnn_reconae_dynamics_only:train"]
        fdhnn_pretrain = by_key["11:exp020_forced_damped_hnn_reconae_dynamics_only:train"]
        staged_hnn = by_key["11:exp011_hnn_reconae_pretrain_joint_direct:train"]
        staged_fdhnn = by_key["11:exp021_forced_damped_hnn_reconae_pretrain_joint_direct:train"]
        self.assertEqual(hnn_pretrain.resources, 1)
        self.assertEqual(fdhnn_pretrain.resources, 1)
        self.assertEqual(staged_hnn.dependencies, (hnn_pretrain.key,))
        self.assertEqual(staged_fdhnn.dependencies, (fdhnn_pretrain.key,))
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
        family_shard = by_key["11:exp000:infer:test_strict:shard2"]
        self.assertEqual(
            family_shard.command[family_shard.command.index("--input") + 1],
            "/assets/data/pathway_v4_full/test_strict_pathway_continuation_v4.csv",
        )
        self.assertIn(
            Path(
                "/assets/runs/seeds/11/experiments/exp000_sft_only_direct/"
                "diagnostics/test_strict/direct.csv"
            ),
            by_key["11:exp000:infer:test_strict"].outputs,
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
                "11:exp000:infer:test_organism",
                "11:exp000:infer:test_organism:shard0",
                "11:exp000:infer:test_organism:shard1",
                "11:exp000:infer:test_organism:shard2",
                "11:exp000:infer:test_organism:shard3",
                "11:exp000:infer:test_strict",
                "11:exp000:infer:test_strict:shard0",
                "11:exp000:infer:test_strict:shard1",
                "11:exp000:infer:test_strict:shard2",
                "11:exp000:infer:test_strict:shard3",
            },
        )
        for job in selected:
            self.assertTrue(set(job.dependencies).issubset(keys))

    def test_scheduler_allocates_four_gpu_sft_then_single_gpu_tasks(self) -> None:
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
