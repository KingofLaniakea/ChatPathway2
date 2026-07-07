"""Small helpers for experiment train/inference wrappers."""

from __future__ import annotations

import runpy
import os
import shlex
import subprocess
import sys

from experiments.runtime_config import asset_path


def _command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


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


def run_steps(steps: list[tuple[str, list[str]]]) -> None:
    """Run a sequence of module commands for a full experiment pipeline."""

    dry_run = os.environ.get("CHATPATHWAY_LAUNCH_DRY_RUN") == "1" or "--dry-run" in sys.argv[1:]
    commands = [[sys.executable, "-m", module, *args] for module, args in steps]
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
