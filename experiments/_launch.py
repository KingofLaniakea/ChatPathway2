"""Small helpers for experiment train/inference wrappers."""

from __future__ import annotations

import runpy
import os
import shlex
import subprocess
import sys
from pathlib import Path

from experiments.runtime_config import active_profile, asset_path


DEFAULT_EXPERIMENT_SEED = "20260711"


def _command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def experiment_seed() -> str:
    """Return the seed that scopes experiment artifacts.

    The wrapper sees its CLI before composing inner commands, so one ``--seed``
    value can select both the RNG seed and an isolated artifact tree.  This
    prevents replicate runs from overwriting a different seed's shared SFT/AE.
    """

    env_seed = os.environ.get("CHATPATHWAY_EXPERIMENT_SEED")
    if env_seed:
        return env_seed
    args = sys.argv[1:]
    for index, value in enumerate(args):
        if value == "--seed" and index + 1 < len(args):
            return args[index + 1]
        if value.startswith("--seed="):
            return value.split("=", 1)[1]
    return DEFAULT_EXPERIMENT_SEED


def seeded_asset_path(relative_path: str) -> str:
    """Resolve a checkpoint/run path below ``<kind>/seeds/<seed>/``.

    Models and datasets are intentionally not accepted here: only mutable or
    derived experiment artifacts should vary by replicate seed.
    """

    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts:
        raise ValueError("seeded_asset_path expects a non-empty relative path")
    kind, *rest = relative.parts
    if kind not in {"checkpoints", "runs", "artifacts"}:
        raise ValueError(f"Seed-scoped assets must be checkpoints/runs/artifacts, got {kind!r}")
    return asset_path(str(Path(kind) / "seeds" / experiment_seed() / Path(*rest)))


def run_module(module: str, default_args: list[str] | None = None) -> None:
    """Run a module as ``__main__`` while preserving extra CLI args.

    Wrapper scripts use this so every experiment has a concrete train.py and
    infer.py without duplicating the underlying implementation.
    """

    if os.environ.get("CHATPATHWAY_LAUNCH_DRY_RUN") == "1":
        print(_command_string([sys.executable, "-m", module, *(default_args or []), *sys.argv[1:]]))
        return
    sys.argv = [module, *(default_args or []), *sys.argv[1:]]
    runpy.run_module(module, run_name="__main__")


def step_commands(
    steps: list[tuple[str, list[str]]],
    passthrough: list[str] | None = None,
) -> list[list[str]]:
    """Compose stage commands and append wrapper CLI arguments to every stage."""

    passthrough = list(passthrough or [])
    commands: list[list[str]] = []
    for module, args in steps:
        if module.startswith("torchrun:"):
            target = module.split(":", 1)[1]
            default_nproc = "4" if active_profile() == "cfff" else "2"
            nproc = os.environ.get("CHATPATHWAY_NPROC_PER_NODE", default_nproc)
            commands.append([
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                nproc,
                "-m",
                target,
                *args,
                *passthrough,
            ])
        else:
            commands.append([sys.executable, "-m", module, *args, *passthrough])
    return commands


def run_steps(steps: list[tuple[str, list[str]]]) -> None:
    """Run a sequence of module commands for a full experiment pipeline.

    Extra wrapper arguments are intentionally forwarded to every stage. Shared
    pipeline flags such as ``--epochs``, ``--limit``, and ``--seed`` therefore
    behave the same in dry-run and real execution.
    """

    dry_run = os.environ.get("CHATPATHWAY_LAUNCH_DRY_RUN") == "1" or "--dry-run" in sys.argv[1:]
    passthrough = [value for value in sys.argv[1:] if value != "--dry-run"]
    commands = step_commands(steps, passthrough)
    if dry_run:
        for command in commands:
            print(_command_string(command))
        return
    for command in commands:
        print(_command_string(command))
        subprocess.run(command, check=True)


def run_torchrun_module(module: str, default_args: list[str] | None = None) -> None:
    """Launch a module through ``torch.distributed.run``.

    Wrapper-level options:

    - ``--nproc-per-node N`` controls local process count.
    - ``--no-standalone`` omits the single-node rendezvous helper.
    - ``--dry-run`` prints the torchrun command without executing it.

    Remaining arguments are passed to the target training module.
    """

    nproc = os.environ.get("CHATPATHWAY_NPROC_PER_NODE", "2")
    standalone = True
    dry_run = os.environ.get("CHATPATHWAY_LAUNCH_DRY_RUN") == "1"
    passthrough: list[str] = []
    args = list(sys.argv[1:])
    i = 0
    while i < len(args):
        if args[i] == "--nproc-per-node":
            if i + 1 >= len(args):
                raise SystemExit("--nproc-per-node requires a value")
            nproc = args[i + 1]
            i += 2
        elif args[i].startswith("--nproc-per-node="):
            nproc = args[i].split("=", 1)[1]
            i += 1
        elif args[i] == "--no-standalone":
            standalone = False
            i += 1
        elif args[i] == "--dry-run":
            dry_run = True
            i += 1
        else:
            passthrough.append(args[i])
            i += 1

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        nproc,
    ]
    if standalone:
        command.append("--standalone")
    command.extend(["-m", module, *(default_args or []), *passthrough])
    print(_command_string(command))
    if not dry_run:
        subprocess.run(command, check=True)
