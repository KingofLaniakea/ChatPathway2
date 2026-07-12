"""Audit experiment wrappers without importing heavy model dependencies.

This script runs every wrapper module under ``CHATPATHWAY_LAUNCH_DRY_RUN=1``.
The wrapper should print the inner command it would execute and then exit. This
checks that matrix modules are not only present, but also launchable.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


MATRIX_PATH = Path(__file__).with_name("matrix.json")


def load_rows() -> list[dict[str, Any]]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8")).get("implemented", [])


def module_command(module: str) -> list[str]:
    return [sys.executable, "-m", module]


def audit_module(
    module: str,
    phase: str,
    experiment_id: str,
    *,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env["CHATPATHWAY_LAUNCH_DRY_RUN"] = "1"
    env.update(env_overrides or {})
    result = subprocess.run(
        module_command(module),
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    ok = result.returncode == 0 and bool(stdout)
    return {
        "experiment_id": experiment_id,
        "phase": phase,
        "module": module,
        "ok": ok,
        "returncode": result.returncode,
        "inner_command": stdout,
        "stderr": stderr,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--phase", choices=("train", "infer", "both"), default="both")
    parser.add_argument("--jsonl", help="Optional JSONL output path.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    for row in load_rows():
        phases = []
        if args.phase in {"train", "both"}:
            phases.append(("train", row["train_module"]))
        if args.phase in {"infer", "both"}:
            phases.append(("infer", row["infer_module"]))
        for phase, module in phases:
            record = audit_module(module, phase, row["id"])
            records.append(record)
            if not args.quiet:
                status = "ok" if record["ok"] else "FAIL"
                print(f"{status}\t{row['id']}\t{phase}\t{record['inner_command']}")

    if args.jsonl:
        path = Path(args.jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    failures = [record for record in records if not record["ok"]]
    if failures:
        print(f"Wrapper audit failed for {len(failures)} entries.", file=sys.stderr)
        for record in failures:
            print(
                f"{record['experiment_id']} {record['phase']} returncode={record['returncode']} stderr={record['stderr']}",
                file=sys.stderr,
            )
        raise SystemExit(1)
    print(f"Audited {len(records)} wrapper entry points.")


if __name__ == "__main__":
    main()
