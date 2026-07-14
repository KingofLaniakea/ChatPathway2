#!/usr/bin/env python3
"""Dependency-aware four-GPU scheduler for the maintained CFFF matrix.

Stage-1 SFT is a real four-process DDP job and the shared AE remains a single-GPU
prerequisite.  The primary forced/damped HNN then receives all four GPUs; after
it completes, pure HNN and the compute-matched stage-2 SFT control each receive
two GPUs concurrently.  Per-process gradient accumulation is adjusted so all
stage-2 arms retain an effective global batch of 12 examples per optimizer
update.  Each direct-inference job is split into deterministic strided shards,
followed by a verified merge that restores source order and rejects gaps,
duplicates, or provenance drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
from dataprocess.source_hashes import verify_source_graph_hashes
from experiments._launch import controlled_training_budget_args
from experiments.runtime_config import asset_root


STAGE2_IDS = (
    "exp002_forced_damped_hnn_reconae_joint_direct",
    "exp001_hnn_reconae_joint_direct",
    "exp003_stage2_sft_only_direct",
)
DATASET_DIRECTORY = Path("data/pathway_v3_cap256")
EVALUATION_PARTITIONS = ("test", "test_family_only", "test_organism_only")
CONTROLLED_MAX_LENGTH = 8192

STAGE2_RESOURCES = {
    "exp002_forced_damped_hnn_reconae_joint_direct": 4,
    "exp001_hnn_reconae_joint_direct": 2,
    "exp003_stage2_sft_only_direct": 2,
}

STAGE2_ACCUMULATION_STEPS = {
    experiment_id: 12 // resources
    for experiment_id, resources in STAGE2_RESOURCES.items()
}


@dataclass(frozen=True)
class Job:
    key: str
    seed: int
    resources: int
    dependencies: tuple[str, ...]
    command: tuple[str, ...]
    outputs: tuple[Path, ...]
    skip_if_outputs: tuple[Path, ...] = ()


@dataclass
class RunningJob:
    job: Job
    gpus: tuple[str, ...]
    process: subprocess.Popen[bytes]
    log_handle: object
    log_path: Path


def parse_csv_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected distinct comma-separated integers")
    return values


def parse_csv_strings(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("expected distinct comma-separated values")
    return values


def experiment_command(
    python: str,
    phase: str,
    experiment_id: str,
    seed: int,
    extra_args: Iterable[str] = (),
) -> tuple[str, ...]:
    return (
        python,
        "-m",
        "experiments.run_experiment",
        phase,
        experiment_id,
        "--",
        "--seed",
        str(seed),
        *extra_args,
    )


def shard_path(path: Path, shard_index: int, shard_count: int) -> Path:
    return path.with_name(
        f"{path.stem}.shard-{shard_index:05d}-of-{shard_count:05d}{path.suffix}"
    )


def build_inference_jobs(
    *,
    seed: int,
    experiment_id: str,
    key_prefix: str,
    dependency: str,
    test: Path,
    run_directory: Path,
    python: str,
    shard_count: int,
) -> list[Job]:
    output = run_directory / "direct.csv"
    progress = run_directory / "direct.progress.jsonl"
    final_outputs = (output, progress, output.with_suffix(".run.json"))
    shard_jobs: list[Job] = []
    shard_outputs: list[Path] = []
    shard_progress: list[Path] = []
    shard_keys: list[str] = []
    for shard_index in range(shard_count):
        shard_output = shard_path(output, shard_index, shard_count)
        shard_progress_path = shard_path(progress, shard_index, shard_count)
        shard_key = f"{key_prefix}:shard{shard_index}"
        shard_jobs.append(
            Job(
                key=shard_key,
                seed=seed,
                resources=1,
                dependencies=(dependency,),
                command=experiment_command(
                    python,
                    "infer",
                    experiment_id,
                    seed,
                    (
                        "--input", str(test),
                        "--shard-count", str(shard_count),
                        "--shard-index", str(shard_index),
                        "--output", str(shard_output),
                        "--progress-output", str(shard_progress_path),
                        "--overwrite",
                    ),
                ),
                outputs=(
                    shard_output,
                    shard_progress_path,
                    shard_output.with_suffix(".run.json"),
                ),
                skip_if_outputs=final_outputs,
            )
        )
        shard_outputs.append(shard_output)
        shard_progress.append(shard_progress_path)
        shard_keys.append(shard_key)
    merge_command = [
        python,
        "-m",
        "method.inference.merge_pathway_shards",
        "--input",
        str(test),
        "--output",
        str(output),
        "--progress-output",
        str(progress),
        "--overwrite",
    ]
    for shard_output, shard_progress_path in zip(shard_outputs, shard_progress):
        merge_command.extend(("--shard-output", str(shard_output)))
        merge_command.extend(("--shard-progress", str(shard_progress_path)))
    return [
        *shard_jobs,
        Job(
            key=key_prefix,
            seed=seed,
            resources=1,
            dependencies=tuple(shard_keys),
            command=tuple(merge_command),
            outputs=final_outputs,
        ),
    ]


def build_jobs(
    seeds: Iterable[int],
    root: Path,
    python: str,
    *,
    inference_shards: int = 4,
    evaluation_partitions: Iterable[str] = EVALUATION_PARTITIONS,
) -> list[Job]:
    evaluation_partitions = tuple(evaluation_partitions)
    if (
        not evaluation_partitions
        or len(evaluation_partitions) != len(set(evaluation_partitions))
        or not set(evaluation_partitions).issubset(EVALUATION_PARTITIONS)
    ):
        raise ValueError(
            "evaluation_partitions must be distinct values from "
            + ", ".join(EVALUATION_PARTITIONS)
        )
    jobs: list[Job] = []
    model = root / "models/qwen3_8B"
    dataset_root = root / DATASET_DIRECTORY
    csv_files = {
        partition: dataset_root / PRIMARY_CSV_NAMES[partition]
        for partition in PARTITIONS
    }
    train = csv_files["train"]
    validation = csv_files["validation"]

    def add_diagnostic_jobs(
        *,
        seed: int,
        experiment_id: str,
        key_prefix: str,
        dependency: str,
        run_directory: Path,
    ) -> None:
        for partition in evaluation_partitions:
            partition_key = key_prefix if partition == "test" else f"{key_prefix}:{partition}"
            partition_run_directory = (
                run_directory
                if partition == "test"
                else run_directory / "diagnostics" / partition
            )
            jobs.extend(
                build_inference_jobs(
                    seed=seed,
                    experiment_id=experiment_id,
                    key_prefix=partition_key,
                    dependency=dependency,
                    test=csv_files[partition],
                    run_directory=partition_run_directory,
                    python=python,
                    shard_count=inference_shards,
                )
            )

    for seed in seeds:
        seed_root = root / f"checkpoints/seeds/{seed}"
        run_root = root / f"runs/seeds/{seed}/experiments"
        sft_root = seed_root / "shared/pathway_sft"
        ae_root = seed_root / "shared/pathway_reconstruction_ae"
        sft_key = f"{seed}:sft"
        ae_key = f"{seed}:ae"
        jobs.append(
            Job(
                key=sft_key,
                seed=seed,
                resources=4,
                dependencies=(),
                command=(
                    python,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node",
                    "4",
                    "-m",
                    "method.training.sft",
                    *controlled_training_budget_args(),
                    "--gradient-accumulation-steps",
                    "1",
                    "--base-model",
                    str(model),
                    "--train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--save-dir",
                    str(sft_root),
                    "--seed",
                    str(seed),
                ),
                outputs=(
                    sft_root / "checkpoint_best/adapter_model.safetensors",
                    sft_root / "run_manifest.json",
                    sft_root / "run_complete.json",
                ),
            )
        )
        jobs.append(
            Job(
                key=ae_key,
                seed=seed,
                resources=1,
                dependencies=(sft_key,),
                command=(
                    python,
                    "-m",
                    "method.training.latent_ae",
                    *controlled_training_budget_args(),
                    "--base-model",
                    str(model),
                    "--sft-lora",
                    str(sft_root / "checkpoint_best"),
                    "--train",
                    str(train),
                    "--validation",
                    str(validation),
                    "--save-dir",
                    str(ae_root),
                    "--seed",
                    str(seed),
                ),
                outputs=(
                    ae_root / "checkpoint_best/ae_proj.pt",
                    ae_root / "run_manifest.json",
                    ae_root / "run_complete.json",
                ),
            )
        )

        add_diagnostic_jobs(
            seed=seed,
            experiment_id="exp000_sft_only_direct",
            key_prefix=f"{seed}:exp000:infer",
            dependency=sft_key,
            run_directory=run_root / "exp000_sft_only_direct",
        )

        for experiment_id in STAGE2_IDS:
            train_key = f"{seed}:{experiment_id}:train"
            infer_key = f"{seed}:{experiment_id}:infer"
            checkpoint = seed_root / f"experiments/{experiment_id}/final_lora/checkpoint_best"
            resources = STAGE2_RESOURCES[experiment_id]
            jobs.append(
                Job(
                    key=train_key,
                    seed=seed,
                    resources=resources,
                    dependencies=(ae_key,),
                    command=experiment_command(
                        python,
                        "train",
                        experiment_id,
                        seed,
                        (
                            "--gradient-accumulation-steps",
                            str(STAGE2_ACCUMULATION_STEPS[experiment_id]),
                        ),
                    ),
                    outputs=(
                        checkpoint / "adapter_model.safetensors",
                        checkpoint / "hamiltonian_dynamics.pt",
                        checkpoint.parent / "run_manifest.json",
                        checkpoint.parent / "run_complete.json",
                    ),
                )
            )
            add_diagnostic_jobs(
                seed=seed,
                experiment_id=experiment_id,
                key_prefix=infer_key,
                dependency=train_key,
                run_directory=run_root / experiment_id,
            )
    return jobs


def select_baseline_inference_jobs(jobs: Iterable[Job]) -> list[Job]:
    """Keep only SFT prerequisites and all requested exp000 diagnostics."""

    selected = [
        job
        for job in jobs
        if job.key.endswith(":sft") or ":exp000:infer" in job.key
    ]
    selected_keys = {job.key for job in selected}
    unresolved = {
        dependency
        for job in selected
        for dependency in job.dependencies
        if dependency not in selected_keys
    }
    if unresolved:
        raise ValueError(
            "baseline-only selection lost dependencies: " + ", ".join(sorted(unresolved))
        )
    return selected


def command_string(command: tuple[str, ...]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)


def outputs_complete(job: Job) -> bool:
    return (
        (bool(job.outputs) and all(path.exists() for path in job.outputs))
        or (bool(job.skip_if_outputs) and all(path.exists() for path in job.skip_if_outputs))
    )


def validate_inputs(root: Path) -> None:
    dataset_root = root / DATASET_DIRECTORY
    csv_files = {
        partition: dataset_root / PRIMARY_CSV_NAMES[partition]
        for partition in PARTITIONS
    }
    record_files = {
        partition: dataset_root / RECORD_JSONL_NAMES[partition]
        for partition in PARTITIONS
    }
    manifest = dataset_root / "dataset_manifest.json"
    audit_path = dataset_root / "data_audit.json"
    source_graph_hashes = dataset_root / SOURCE_GRAPH_HASHES_NAME
    required = (
        root / "models/qwen3_8B/config.json",
        root / "models/qwen3_8B/chatpathway_download_manifest.json",
        *csv_files.values(),
        manifest,
        audit_path,
        source_graph_hashes,
        *record_files.values(),
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing CFFF matrix input(s):\n" + "\n".join(missing))
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if manifest_value.get("schema_version") != RELEASE_SCHEMA_VERSION:
        raise ValueError("dataset manifest is not the structured release v3.1 contract")
    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError("data audit is not the v3.1 audit contract")
    if audit.get("release_schema_version") != RELEASE_SCHEMA_VERSION:
        raise ValueError("data audit was generated for a different release schema")
    if audit.get("status") != "passed" or audit.get("strict_failures"):
        raise ValueError(f"dataset release audit did not pass: {audit_path}")
    if stat.S_IMODE(audit_path.stat().st_mode) != 0o444:
        raise PermissionError(f"generated data audit must have mode 0444: {audit_path}")
    if manifest_value.get("max_length") != CONTROLLED_MAX_LENGTH or audit.get("max_length") != CONTROLLED_MAX_LENGTH:
        raise ValueError("v3.1 release and audit must enforce max_length=8192")
    if manifest_value.get("primary_prompt_profile") != PRIMARY_PROMPT_PROFILE:
        raise ValueError("matrix requires the explicit-organism source-native primary profile")
    if set(manifest_value.get("splits", {})) != set(PARTITIONS):
        raise ValueError("dataset manifest must declare exactly five primary partitions")
    if set(audit.get("splits", {})) != set(PARTITIONS):
        raise ValueError("data audit must cover exactly five primary partitions")

    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    if sha256(manifest) != audit.get("manifest_sha256"):
        raise ValueError("dataset manifest changed after data_audit.json was generated")
    for split in PARTITIONS:
        path = csv_files[split]
        split_manifest = manifest_value["splits"][split]
        split_audit = audit["splits"][split]
        if split_manifest.get("prompt_profile") != PRIMARY_PROMPT_PROFILE:
            raise ValueError(f"{split} manifest does not use the primary prompt profile")
        if split_manifest.get("prompt_profile_interface_applied") is not True:
            raise ValueError(f"{split} did not apply the prompt profile interface")
        if split_manifest.get("prefix_horizon_interface_applied") is not True:
            raise ValueError(f"{split} did not apply the prefix-horizon interface")
        if split_audit.get("errors"):
            raise ValueError(f"{split} has strict audit errors")
        rows = split_audit.get("rows")
        if not isinstance(rows, int) or rows < 1:
            raise ValueError(f"{split} must contain at least one accepted row")
        if split_audit.get("prompt_profiles") != {PRIMARY_PROMPT_PROFILE: rows}:
            raise ValueError(f"{split} contains a non-primary or unaudited prompt profile")
        truncation = split_audit.get("truncation_estimate", {})
        if (
            truncation.get("max_length") != CONTROLLED_MAX_LENGTH
            or truncation.get("accepted_rows_over_budget") != 0
        ):
            raise ValueError(f"{split} violates the strict 8192-token release budget")
        expected = split_audit.get("sha256")
        if not expected or sha256(path) != expected:
            raise ValueError(f"{split} CSV changed after data_audit.json was generated")
        if split_manifest.get("csv_sha256") != expected:
            raise ValueError(f"{split} CSV hash disagrees between manifest and audit")
        record_expected = (
            split_audit.get("record_jsonl", {}).get("sha256")
        )
        if not record_expected or sha256(record_files[split]) != record_expected:
            raise ValueError(f"{split} record JSONL changed after data_audit.json was generated")
        if split_manifest.get("records_sha256") != record_expected:
            raise ValueError(f"{split} record JSONL hash disagrees between manifest and audit")

    source_hash_report = audit.get("source_graph_hashes", {})
    if source_hash_report.get("status") != "passed" or source_hash_report.get("errors"):
        raise ValueError("source graph content-hash verification did not pass")
    if sha256(source_graph_hashes) != source_hash_report.get("sha256"):
        raise ValueError("source_graph_hashes.jsonl changed after the release audit")
    manifest_source_hashes = manifest_value.get("source_graph_hashes", {})
    if manifest_source_hashes.get("sha256") != source_hash_report.get("sha256"):
        raise ValueError("source graph inventory hash disagrees between manifest and audit")
    graph_root_value = manifest_value.get("processed_graph_root")
    if not graph_root_value:
        raise ValueError("dataset manifest does not declare processed_graph_root")
    graph_root = Path(str(graph_root_value))
    if not graph_root.is_dir():
        raise FileNotFoundError(graph_root)
    live_source_report = verify_source_graph_hashes(graph_root, source_graph_hashes)
    if live_source_report.get("errors"):
        raise ValueError("live source graph content hashes no longer match the release")
    if (
        live_source_report.get("sha256") != source_hash_report.get("sha256")
        or live_source_report.get("records") != source_hash_report.get("records")
    ):
        raise ValueError("live source graph hash inventory disagrees with data_audit.json")

    strict_overlap = audit.get("required_summary", {}).get("strict_overlap", {})
    for (left, right), contract in OVERLAP_CONTRACT.items():
        pair = f"{left}_vs_{right}"
        report = strict_overlap.get(pair, {})
        identity = report.get("identity_contract", {})
        if not identity or any(result.get("passed") is not True for result in identity.values()):
            raise ValueError(f"{pair} identity overlap contract did not pass")
        biological = report.get("biological_contract", {})
        for field, policy in contract.items():
            result = biological.get(field, {})
            if result.get("policy") != policy or result.get("passed") is not True:
                raise ValueError(f"{pair} {field} overlap contract did not pass")

    paired_manifest = manifest_value.get("paired_prompt_profiles", {})
    paired_files = paired_manifest.get("files", {})
    expected_controls = (
        NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
        SPECIES_NEUTRAL_IDS_NO_ORGANISM,
    )
    if (
        paired_manifest.get("status") != "published"
        or paired_manifest.get("published") is not True
        or set(paired_files) != set(expected_controls)
    ):
        raise ValueError("v3.1 release must publish canonical P1 and strict-natural P2 controls")
    paired_audit = audit.get("paired_prompt_profiles", {})
    if (
        paired_audit.get("status") != "passed"
        or paired_audit.get("manifest_published") is not True
        or paired_audit.get("canonical_files_match_prompt_controls") is not True
    ):
        raise ValueError("paired prompt-profile audit did not pass")
    reports = paired_audit.get("declared_file_reports", {})
    checks = paired_audit.get("pair_checks", {})
    for profile in expected_controls:
        partition_files = paired_files.get(profile, {})
        if set(partition_files) != set(PARTITIONS):
            raise ValueError(f"{profile} must publish all five primary partitions")
        for partition in PARTITIONS:
            metadata = partition_files[partition]
            if not isinstance(metadata, dict) or not metadata.get("path"):
                raise ValueError(f"{profile}:{partition} has no canonical file metadata")
            control_path = dataset_root / str(metadata["path"])
            if not control_path.is_file():
                raise FileNotFoundError(control_path)
            expected_hash = metadata.get("sha256") or metadata.get("csv_sha256")
            report = reports.get(f"{profile}:{partition}", {})
            actual_hash = sha256(control_path)
            if not expected_hash or actual_hash != expected_hash or report.get("sha256") != actual_hash:
                raise ValueError(f"{profile}:{partition} control CSV hash verification failed")
            if report.get("errors") or not isinstance(report.get("rows"), int) or report["rows"] < 1:
                raise ValueError(f"{profile}:{partition} control CSV did not pass row auditing")

    for partition in PARTITIONS:
        p1_key = (
            f"{partition}:{PRIMARY_PROMPT_PROFILE}_vs_"
            f"{NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS}"
        )
        p2_key = (
            f"{partition}:{PRIMARY_PROMPT_PROFILE}_vs_"
            f"{SPECIES_NEUTRAL_IDS_NO_ORGANISM}"
        )
        p1 = checks.get(p1_key, {})
        p2 = checks.get(p2_key, {})
        if p1.get("passed") is not True or p1.get("base_sample_policy") != "exact_primary_set":
            raise ValueError(f"{partition} P0/P1 exact pairing contract did not pass")
        if (
            p2.get("passed") is not True
            or p2.get("base_sample_policy") != "strict_natural_neutral_subset"
        ):
            raise ValueError(f"{partition} P2 strict-natural subset contract did not pass")


def run_scheduler(
    jobs: list[Job],
    *,
    gpus: list[str],
    profile: str,
    log_dir: Path,
    poll_seconds: float,
    skip_existing: bool,
    dry_run: bool,
) -> int:
    if len(gpus) != 4:
        raise ValueError("the maintained CFFF schedule requires exactly four GPU ids")
    if dry_run:
        for job in jobs:
            print(
                f"{job.key}\tgpus={job.resources}\tdepends={','.join(job.dependencies) or '-'}\t"
                f"{command_string(job.command)}"
            )
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    pending = {job.key: job for job in jobs}
    completed: set[str] = set()
    failed: set[str] = set()
    skipped: set[str] = set()
    running: dict[str, RunningJob] = {}

    if skip_existing:
        for key, job in list(pending.items()):
            if outputs_complete(job):
                completed.add(key)
                pending.pop(key)
                print(f"already_complete\t{key}")

    try:
        while pending or running:
            for key, active in list(running.items()):
                return_code = active.process.poll()
                if return_code is None:
                    continue
                active.log_handle.close()
                running.pop(key)
                if return_code == 0 and outputs_complete(active.job):
                    completed.add(key)
                    print(f"completed\t{key}\tlog={active.log_path}")
                else:
                    failed.add(key)
                    print(f"failed\t{key}\treturn_code={return_code}\tlog={active.log_path}")

            for key, job in list(pending.items()):
                if any(dependency in failed or dependency in skipped for dependency in job.dependencies):
                    skipped.add(key)
                    pending.pop(key)
                    print(f"blocked\t{key}\tfailed_dependency")

            used = {gpu for active in running.values() for gpu in active.gpus}
            free = [gpu for gpu in gpus if gpu not in used]
            ready = [
                job
                for job in pending.values()
                if set(job.dependencies).issubset(completed)
            ]
            ready.sort(key=lambda job: (-job.resources, job.key))
            launched = False
            for job in ready:
                if not 1 <= job.resources <= len(gpus):
                    raise ValueError(
                        f"job {job.key!r} requests {job.resources} GPUs; "
                        f"available scheduler width is {len(gpus)}"
                    )
                if len(free) < job.resources:
                    continue
                assigned = tuple(free[: job.resources])
                del free[: job.resources]

                job_log = log_dir / f"{job.key.replace(':', '__')}.log"
                handle = job_log.open("wb")
                env = os.environ.copy()
                env["CHATPATHWAY_PROFILE"] = profile
                env["CHATPATHWAY_NPROC_PER_NODE"] = str(job.resources)
                env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
                process = subprocess.Popen(
                    job.command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[job.key] = RunningJob(job, assigned, process, handle, job_log)
                pending.pop(job.key)
                print(f"started\t{job.key}\tgpus={','.join(assigned)}\tlog={job_log}")
                launched = True

            if pending and not running and not launched:
                unresolved = ", ".join(sorted(pending))
                raise RuntimeError(f"scheduler dependency deadlock: {unresolved}")
            if running:
                time.sleep(poll_seconds)
    except KeyboardInterrupt:
        for active in running.values():
            os.killpg(active.process.pid, signal.SIGTERM)
        raise
    finally:
        for active in running.values():
            active.log_handle.close()

    print(
        f"summary\tcompleted={len(completed)}\tfailed={len(failed)}\tblocked={len(skipped)}"
    )
    return 1 if failed or skipped else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("20260711,20260712,20260713"))
    parser.add_argument("--gpus", type=parse_csv_strings, default=parse_csv_strings("0,1,2,3"))
    parser.add_argument("--profile", default="cfff")
    parser.add_argument("--inference-shards", type=int, default=4)
    parser.add_argument(
        "--evaluation-partitions",
        type=parse_csv_strings,
        default=list(EVALUATION_PARTITIONS),
        help="Comma-separated diagnostic test partitions to schedule.",
    )
    parser.add_argument(
        "--only-baseline-inference",
        action="store_true",
        help="Run only completed/shared SFT prerequisites and exp000 inference shards/merge.",
    )
    parser.add_argument("--log-dir")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if not 1 <= args.inference_shards <= 4:
        parser.error("--inference-shards must be between 1 and 4")
    if not set(args.evaluation_partitions).issubset(EVALUATION_PARTITIONS):
        parser.error(
            "--evaluation-partitions must contain only "
            + ", ".join(EVALUATION_PARTITIONS)
        )
    return args


def main() -> None:
    args = parse_args()
    root = asset_root(args.profile)
    if not args.dry_run:
        validate_inputs(root)
    jobs = build_jobs(
        args.seeds,
        root,
        sys.executable,
        inference_shards=args.inference_shards,
        evaluation_partitions=args.evaluation_partitions,
    )
    if args.only_baseline_inference:
        jobs = select_baseline_inference_jobs(jobs)
    log_dir = Path(args.log_dir) if args.log_dir else root / "runs/cfff_matrix_scheduler"
    raise SystemExit(
        run_scheduler(
            jobs,
            gpus=args.gpus,
            profile=args.profile,
            log_dir=log_dir,
            poll_seconds=args.poll_seconds,
            skip_existing=args.skip_existing,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
