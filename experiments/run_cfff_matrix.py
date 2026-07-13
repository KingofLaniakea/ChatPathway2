#!/usr/bin/env python3
"""Dependency-aware four-GPU scheduler for the maintained CFFF matrix.

Stage-1 SFT is a real four-process DDP job.  AE, stage-2 arms, and inference
are intentionally single-GPU jobs; after the SFT prerequisites finish, this
scheduler fills the four GPUs with independent seed/arm jobs instead of
pretending that every small auxiliary module benefits from model parallelism.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from experiments._launch import controlled_training_budget_args
from experiments.runtime_config import asset_root


STAGE2_IDS = (
    "exp003_stage2_sft_only_direct",
    "exp001_hnn_reconae_joint_direct",
    "exp002_forced_damped_hnn_reconae_joint_direct",
)


@dataclass(frozen=True)
class Job:
    key: str
    seed: int
    resources: int
    dependencies: tuple[str, ...]
    command: tuple[str, ...]
    outputs: tuple[Path, ...]


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


def experiment_command(python: str, phase: str, experiment_id: str, seed: int) -> tuple[str, ...]:
    return (
        python,
        "-m",
        "experiments.run_experiment",
        phase,
        experiment_id,
        "--",
        "--seed",
        str(seed),
    )


def build_jobs(seeds: Iterable[int], root: Path, python: str) -> list[Job]:
    jobs: list[Job] = []
    model = root / "models/qwen3_8B"
    train = root / "data/train_kegg_pathway_record_balanced_0p1pct.csv"
    test = root / "data/test_kegg_pathway_eval.csv"

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
                    "--base-model",
                    str(model),
                    "--train",
                    str(train),
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

        baseline_key = f"{seed}:exp000:infer"
        jobs.append(
            Job(
                key=baseline_key,
                seed=seed,
                resources=1,
                dependencies=(sft_key,),
                command=experiment_command(python, "infer", "exp000_sft_only_direct", seed),
                outputs=(
                    run_root / "exp000_sft_only_direct/direct.csv",
                    run_root / "exp000_sft_only_direct/direct.progress.jsonl",
                    run_root / "exp000_sft_only_direct/direct.run.json",
                ),
            )
        )

        for experiment_id in STAGE2_IDS:
            train_key = f"{seed}:{experiment_id}:train"
            infer_key = f"{seed}:{experiment_id}:infer"
            checkpoint = seed_root / f"experiments/{experiment_id}/final_lora/checkpoint_best"
            jobs.append(
                Job(
                    key=train_key,
                    seed=seed,
                    resources=1,
                    dependencies=(ae_key,),
                    command=experiment_command(python, "train", experiment_id, seed),
                    outputs=(
                        checkpoint / "adapter_model.safetensors",
                        checkpoint / "hamiltonian_dynamics.pt",
                        checkpoint.parent / "run_manifest.json",
                        checkpoint.parent / "run_complete.json",
                    ),
                )
            )
            jobs.append(
                Job(
                    key=infer_key,
                    seed=seed,
                    resources=1,
                    dependencies=(train_key,),
                    command=experiment_command(python, "infer", experiment_id, seed),
                    outputs=(
                        run_root / f"{experiment_id}/direct.csv",
                        run_root / f"{experiment_id}/direct.progress.jsonl",
                        run_root / f"{experiment_id}/direct.run.json",
                    ),
                )
            )
    return jobs


def command_string(command: tuple[str, ...]) -> str:
    import shlex

    return " ".join(shlex.quote(part) for part in command)


def outputs_complete(job: Job) -> bool:
    return bool(job.outputs) and all(path.exists() for path in job.outputs)


def validate_inputs(root: Path) -> None:
    required = (
        root / "models/qwen3_8B/config.json",
        root / "models/qwen3_8B/chatpathway_download_manifest.json",
        root / "data/train_kegg_pathway_record_balanced_0p1pct.csv",
        root / "data/test_kegg_pathway_eval.csv",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing CFFF matrix input(s):\n" + "\n".join(missing))


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
                if job.resources == 4:
                    if running or len(free) < 4:
                        continue
                    assigned = tuple(gpus)
                else:
                    if not free:
                        break
                    assigned = (free.pop(0),)

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
                if job.resources == 4:
                    free.clear()
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
    parser.add_argument("--log-dir")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def main() -> None:
    args = parse_args()
    root = asset_root(args.profile)
    if not args.dry_run:
        validate_inputs(root)
    jobs = build_jobs(args.seeds, root, sys.executable)
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
