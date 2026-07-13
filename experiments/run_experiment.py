"""High-level wrapper for ChatPathway2 training/inference experiments."""

from __future__ import annotations

import argparse
import os
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.runtime_config import asset_path


MATRIX_PATH = Path(__file__).with_name("matrix.json")
RUNTIME_MANIFEST_PATH = Path(__file__).with_name("runtime_manifest.json")


def load_matrix() -> dict[str, Any]:
    with MATRIX_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_runtime_manifest() -> dict[str, Any]:
    with RUNTIME_MANIFEST_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def implemented_rows(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    return list(matrix.get("implemented", []))


def find_row(matrix: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    for row in implemented_rows(matrix):
        if row["id"] == experiment_id:
            return row
    available = ", ".join(row["id"] for row in implemented_rows(matrix))
    raise SystemExit(f"Unknown experiment '{experiment_id}'. Available: {available}")


def module_command(module: str, passthrough: list[str]) -> list[str]:
    return [sys.executable, "-m", module, *passthrough]


def command_string(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def split_passthrough(raw_args: list[str], dry_run: bool) -> tuple[list[str], bool]:
    passthrough = []
    for value in raw_args:
        if value == "--dry-run":
            dry_run = True
        else:
            passthrough.append(value)
    if passthrough[:1] == ["--"]:
        passthrough = passthrough[1:]
    return passthrough, dry_run


def commands_for(row: dict[str, Any], phase: str, passthrough: list[str]) -> list[tuple[str, list[str]]]:
    commands: list[tuple[str, list[str]]] = []
    if phase in {"train", "pipeline"}:
        commands.append(("train", module_command(row["train_module"], passthrough)))
    if phase in {"infer", "pipeline"}:
        commands.append(("infer", module_command(row["infer_module"], passthrough)))
    return commands


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_log(log_path: str | None, record: dict[str, Any]) -> None:
    if not log_path:
        return
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_command(
    command: list[str],
    dry_run: bool,
    *,
    experiment_id: str,
    phase: str,
    log_jsonl: str | None = None,
    continue_on_error: bool = False,
) -> None:
    rendered = command_string(command)
    print(rendered)
    record = {
        "time": utc_now(),
        "experiment_id": experiment_id,
        "phase": phase,
        "dry_run": dry_run,
        "command": command,
        "command_string": rendered,
        "status": "dry_run" if dry_run else "started",
    }
    write_log(log_jsonl, record)
    if not dry_run:
        completed = subprocess.run(command, check=False)
        status = "ok" if completed.returncode == 0 else "failed"
        write_log(log_jsonl, {**record, "time": utc_now(), "status": status, "returncode": completed.returncode})
        if completed.returncode != 0 and not continue_on_error:
            raise SystemExit(completed.returncode)


def print_rows(matrix: dict[str, Any]) -> None:
    print(f"Experiment matrix version: {matrix.get('version')}")
    for row in implemented_rows(matrix):
        print(f"{row['id']}\t{row['middle_network']}\t{row['inference_mode']}")


def print_candidates(matrix: dict[str, Any]) -> None:
    for axis, values in matrix.get("candidate_axes", {}).items():
        print(f"\n[{axis}]")
        for value in values:
            print(f"- {value}")
    combinations = matrix.get("combinations", [])
    if combinations:
        print("\n[post_current_research_and_deferred_combinations]")
        for value in combinations:
            print(f"- {value['id']}: {value['title']} ({value['status']})")


def parse_id_list(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def select_rows(
    rows: list[dict[str, Any]],
    *,
    ids: str | None = None,
    exclude: str | None = None,
    start_at: str | None = None,
    stop_after: str | None = None,
    contains: str | None = None,
) -> list[dict[str, Any]]:
    include_ids = parse_id_list(ids)
    exclude_ids = parse_id_list(exclude) or set()
    selected: list[dict[str, Any]] = []
    active = start_at is None
    for row in rows:
        row_id = row["id"]
        if row_id == start_at:
            active = True
        if not active:
            continue
        if include_ids is not None and row_id not in include_ids:
            if row_id == stop_after:
                break
            continue
        if row_id in exclude_ids:
            if row_id == stop_after:
                break
            continue
        if contains:
            haystack = " ".join(str(row.get(field, "")) for field in ("title", "granularity", "middle_network", "training_coupling", "inference_mode", "notes"))
            if contains.lower() not in haystack.lower():
                if row_id == stop_after:
                    break
                continue
        selected.append(row)
        if row_id == stop_after:
            break
    known_ids = {row["id"] for row in rows}
    requested = (include_ids or set()) | (parse_id_list(exclude) or set())
    for row_id in requested:
        if row_id not in known_ids:
            raise SystemExit(f"Unknown experiment id in selection: {row_id}")
    for row_id in (start_at, stop_after):
        if row_id is not None and row_id not in known_ids:
            raise SystemExit(f"Unknown experiment id in range selection: {row_id}")
    return selected


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ids", help="Comma-separated experiment ids to include.")
    parser.add_argument("--exclude", help="Comma-separated experiment ids to skip.")
    parser.add_argument("--start-at", help="Start at this experiment id in matrix order.")
    parser.add_argument("--stop-after", help="Stop after this experiment id in matrix order.")
    parser.add_argument("--contains", help="Keep rows whose descriptive fields contain this text.")


def render_plan(rows: list[dict[str, Any]], phase: str, passthrough: list[str], output_format: str) -> str:
    entries: list[dict[str, Any]] = []
    for row in rows:
        for command_phase, command in commands_for(row, phase, passthrough):
            entries.append({
                "experiment_id": row["id"],
                "phase": command_phase,
                "title": row["title"],
                "command": command,
                "command_string": command_string(command),
            })
    if output_format == "jsonl":
        return "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries)
    if output_format == "tsv":
        header = "experiment_id\tphase\ttitle\tcommand"
        lines = [header]
        lines.extend(f"{entry['experiment_id']}\t{entry['phase']}\t{entry['title']}\t{entry['command_string']}" for entry in entries)
        return "\n".join(lines)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    lines.extend(entry["command_string"] for entry in entries)
    return "\n".join(lines)


def emit_text(text: str, output: str | None) -> None:
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + ("\n" if text else ""), encoding="utf-8")
        print(f"Wrote plan: {path}")
    else:
        print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List implemented experiment rows.")
    sub.add_parser("axes", help="List candidate experimental axes not all implemented yet.")

    show = sub.add_parser("show", help="Print one experiment row.")
    show.add_argument("experiment_id")

    runtime = sub.add_parser("runtime", help="Print runtime requirements and expected outputs for one experiment row.")
    runtime.add_argument("experiment_id")

    for name in ("train", "infer", "pipeline"):
        command_parser = sub.add_parser(name, help=f"Run {name} for one experiment.")
        command_parser.add_argument("experiment_id")
        command_parser.add_argument("--dry-run", action="store_true")
        command_parser.add_argument("--log-jsonl", help="Append command status records to this JSONL file.")
        command_parser.add_argument("--continue-on-error", action="store_true", help="Return only after logging a failed command.")
        command_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed after '--' to the experiment wrapper.")

    run_all = sub.add_parser("run-all", help="Run every implemented experiment for one phase.")
    run_all.add_argument("--phase", choices=("train", "infer", "pipeline"), required=True)
    run_all.add_argument("--dry-run", action="store_true")
    run_all.add_argument("--log-jsonl", help="Append command status records to this JSONL file.")
    run_all.add_argument("--continue-on-error", action="store_true", help="Continue through later experiments after a failed command.")
    add_selection_args(run_all)
    run_all.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed after '--' to every selected experiment wrapper.")

    plan = sub.add_parser("plan", help="Emit a reproducible command plan without running it.")
    plan.add_argument("--phase", choices=("train", "infer", "pipeline"), required=True)
    plan.add_argument("--format", choices=("shell", "jsonl", "tsv"), default="shell")
    plan.add_argument("--output", help="Optional output file for the rendered plan.")
    add_selection_args(plan)
    plan.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed after '--' to every selected experiment wrapper.")

    audit = sub.add_parser("audit", help="Audit wrapper modules by expanding their inner dry-run commands.")
    audit.add_argument("--phase", choices=("train", "infer", "both"), default="both")
    audit.add_argument("--jsonl", help="Optional JSONL output path.")
    audit.add_argument("--quiet", action="store_true")

    consistency = sub.add_parser("consistency", help="Audit wrapper dry-run commands against runtime_manifest.json paths.")
    consistency.add_argument("--phase", choices=("train", "infer", "both"), default="both")
    consistency.add_argument("--ids", help="Comma-separated experiment ids to audit.")
    consistency.add_argument("--asset-root", help="Temporarily override the runtime asset root.")
    consistency.add_argument("--profile", help="Runtime profile name from chatpathway.config.json.")
    consistency.add_argument("--jsonl", help="Optional JSONL output path.")
    consistency.add_argument("--quiet", action="store_true")

    check_assets = sub.add_parser("check-assets", help="Check runtime data/checkpoint paths from runtime_manifest.json.")
    check_assets.add_argument("--phase", choices=("train", "infer", "both"), default="both")
    check_assets.add_argument("--asset-root", help="Temporarily override the asset root from chatpathway.config.json.")
    check_assets.add_argument("--profile", help="Runtime profile name from chatpathway.config.json.")
    check_assets.add_argument("--seed", type=int, default=20260711, help="Seed-scoped checkpoint/run tree to inspect.")
    check_assets.add_argument("--check-outputs", action="store_true", help="Also require expected result artifacts to exist.")
    check_assets.add_argument("--create-output-dirs", action="store_true", help="Create output parent directories if missing.")
    check_assets.add_argument("--jsonl", help="Optional JSONL output path.")
    check_assets.add_argument("--strict", action="store_true", help="Exit non-zero if any required path is missing.")
    check_assets.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    add_selection_args(check_assets)

    prepare_smoke = sub.add_parser("prepare-smoke", help="Create tiny CSV/JSONL inputs for AutoDL smoke runs.")
    prepare_smoke.add_argument("--rows", type=int, default=2)
    prepare_smoke.add_argument("--overwrite", action="store_true")
    prepare_smoke.add_argument("--skip-missing", action="store_true")

    prepare_data = sub.add_parser(
        "prepare-data",
        help="Build the record-balanced 0.1% first-round training set and evaluation CSVs from the full KEGG CSVs.",
    )
    prepare_data.add_argument("--record-fraction", type=float, default=0.001)
    prepare_data.add_argument("--phenotype-record-fraction", type=float, default=1.0)
    prepare_data.add_argument("--pathway-family-holdout-fraction", type=float, default=0.1)
    prepare_data.add_argument("--pathway-family-holdout-seed", type=int)
    prepare_data.add_argument("--max-prefixes-per-record", type=int, default=3)
    prepare_data.add_argument("--max-test-prefixes-per-record", type=int, default=1)
    prepare_data.add_argument("--max-multistep-prefixes-per-record", type=int, default=3)
    prepare_data.add_argument("--seed", type=int, default=20260711)
    prepare_data.add_argument("--overwrite", action="store_true")

    prepare_coverage = sub.add_parser(
        "prepare-coverage",
        help="Create a family-capped, organism/length-diverse optimization set and fixed family validation set.",
    )
    prepare_coverage.add_argument("--max-records-per-family", type=int, default=32)
    prepare_coverage.add_argument("--validation-fraction", type=float, default=0.05)
    prepare_coverage.add_argument("--seed", type=int, default=20260711)
    prepare_coverage.add_argument("--overwrite", action="store_true")

    prepare_structured = sub.add_parser(
        "prepare-structured-data",
        help="Build and strictly audit the canonical processed_graph pathway-continuation v3 release.",
    )
    prepare_structured.add_argument("--processed-graph-root")
    prepare_structured.add_argument("--output-dir")
    prepare_structured.add_argument("--max-length", type=int, default=8192)
    prepare_structured.add_argument("--test-organisms", default="tru,xtr,dre,gga,dmk,dme,cel")
    prepare_structured.add_argument("--test-family-fraction", type=float, default=0.05)
    prepare_structured.add_argument("--validation-family-fraction", type=float, default=0.05)
    prepare_structured.add_argument("--train-candidate-record-fraction", type=float, default=0.003)
    prepare_structured.add_argument("--max-records-per-family", type=int, default=256)
    prepare_structured.add_argument("--max-prefixes-per-train-record", type=int, default=3)
    prepare_structured.add_argument("--minimum-train-records", type=int, default=12000)
    prepare_structured.add_argument("--seed", type=int, default=20260711)
    prepare_structured.add_argument("--progress-every", type=int, default=1000)
    prepare_structured.add_argument("--overwrite", action="store_true")

    download_model = sub.add_parser("download-model", help="Download and verify the pinned Qwen3-8B snapshot.")
    download_model.add_argument("--revision", default="b968826")
    download_model.add_argument("--endpoint")
    download_model.add_argument("--verify-only", action="store_true")

    args = parser.parse_args()
    matrix = load_matrix()

    if args.command == "list":
        print_rows(matrix)
        return
    if args.command == "axes":
        print_candidates(matrix)
        return
    if args.command == "show":
        print(json.dumps(find_row(matrix, args.experiment_id), indent=2, ensure_ascii=False))
        return
    if args.command == "runtime":
        row = find_row(matrix, args.experiment_id)
        manifest = load_runtime_manifest()
        runtime_entry = manifest.get("rows", {}).get(args.experiment_id)
        if runtime_entry is None:
            raise SystemExit(f"No runtime manifest entry for {args.experiment_id}")
        print(json.dumps({"matrix": row, "runtime": runtime_entry}, indent=2, ensure_ascii=False))
        return

    if args.command in {"train", "infer", "pipeline"}:
        row = find_row(matrix, args.experiment_id)
        passthrough, dry_run = split_passthrough(args.args, args.dry_run)
        for command_phase, command in commands_for(row, args.command, passthrough):
            run_command(
                command,
                dry_run,
                experiment_id=row["id"],
                phase=command_phase,
                log_jsonl=args.log_jsonl,
                continue_on_error=args.continue_on_error,
            )
        return

    if args.command == "run-all":
        passthrough, dry_run = split_passthrough(args.args, args.dry_run)
        rows = select_rows(
            implemented_rows(matrix),
            ids=args.ids,
            exclude=args.exclude,
            start_at=args.start_at,
            stop_after=args.stop_after,
            contains=args.contains,
        )
        for row in rows:
            for command_phase, command in commands_for(row, args.phase, passthrough):
                run_command(
                    command,
                    dry_run,
                    experiment_id=row["id"],
                    phase=command_phase,
                    log_jsonl=args.log_jsonl,
                    continue_on_error=args.continue_on_error,
                )
        return

    if args.command == "plan":
        passthrough, _ = split_passthrough(args.args, False)
        rows = select_rows(
            implemented_rows(matrix),
            ids=args.ids,
            exclude=args.exclude,
            start_at=args.start_at,
            stop_after=args.stop_after,
            contains=args.contains,
        )
        emit_text(render_plan(rows, args.phase, passthrough, args.format), args.output)
        return

    if args.command == "audit":
        command = [sys.executable, "-m", "experiments.audit_wrappers", "--phase", args.phase]
        if args.jsonl:
            command.extend(["--jsonl", args.jsonl])
        if args.quiet:
            command.append("--quiet")
        env = os.environ.copy()
        env["CHATPATHWAY_LAUNCH_DRY_RUN"] = "1"
        raise SystemExit(subprocess.run(command, env=env, check=False).returncode)

    if args.command == "consistency":
        command = [sys.executable, "-m", "experiments.audit_matrix_consistency", "--phase", args.phase]
        if args.ids:
            command.extend(["--ids", args.ids])
        if args.asset_root:
            command.extend(["--asset-root", args.asset_root])
        if args.profile:
            command.extend(["--profile", args.profile])
        if args.jsonl:
            command.extend(["--jsonl", args.jsonl])
        if args.quiet:
            command.append("--quiet")
        raise SystemExit(subprocess.run(command, check=False).returncode)

    if args.command == "check-assets":
        rows = select_rows(
            implemented_rows(matrix),
            ids=args.ids,
            exclude=args.exclude,
            start_at=args.start_at,
            stop_after=args.stop_after,
            contains=args.contains,
        )
        command = [
            sys.executable,
            "-m",
            "experiments.check_runtime_assets",
            "--phase",
            args.phase,
            "--ids",
            ",".join(row["id"] for row in rows),
        ]
        if args.asset_root:
            command.extend(["--asset-root", args.asset_root])
        if args.profile:
            command.extend(["--profile", args.profile])
        if args.check_outputs:
            command.append("--check-outputs")
        if args.create_output_dirs:
            command.append("--create-output-dirs")
        if args.jsonl:
            command.extend(["--jsonl", args.jsonl])
        if args.strict:
            command.append("--strict")
        if args.quiet:
            command.append("--quiet")
        env = os.environ.copy()
        env["CHATPATHWAY_EXPERIMENT_SEED"] = str(args.seed)
        raise SystemExit(subprocess.run(command, env=env, check=False).returncode)

    if args.command == "prepare-smoke":
        command = [sys.executable, "-m", "experiments.prepare_smoke_inputs", "--rows", str(args.rows)]
        if args.overwrite:
            command.append("--overwrite")
        if args.skip_missing:
            command.append("--skip-missing")
        raise SystemExit(subprocess.run(command, check=False).returncode)

    if args.command == "prepare-data":
        command = [
            sys.executable,
            "-m",
            "dataprocess.prepare_experiment_data",
            "--train-input",
            asset_path("data/train_kegg_pathway_dataset.csv"),
            "--test-input",
            asset_path("data/test_kegg_pathway_dataset.csv"),
            "--train-output",
            asset_path("data/train_kegg_pathway_record_balanced_0p1pct.csv"),
            "--test-output",
            asset_path("data/test_kegg_pathway_eval.csv"),
            "--multistep-test-output",
            asset_path("data/test_kegg_pathway_multistep_eval.csv"),
            "--organism-test-output",
            asset_path("data/test_kegg_pathway_organism_eval.csv"),
            "--organism-multistep-test-output",
            asset_path("data/test_kegg_pathway_organism_multistep_eval.csv"),
            "--record-fraction",
            str(args.record_fraction),
            "--phenotype-record-fraction",
            str(args.phenotype_record_fraction),
            "--pathway-family-holdout-fraction",
            str(args.pathway_family_holdout_fraction),
            "--max-prefixes-per-record",
            str(args.max_prefixes_per_record),
            "--max-test-prefixes-per-record",
            str(args.max_test_prefixes_per_record),
            "--max-multistep-prefixes-per-record",
            str(args.max_multistep_prefixes_per_record),
            "--seed",
            str(args.seed),
            "--report",
            asset_path("artifacts/dataset/record_balanced_0p1pct_family_disjoint_v2.json"),
        ]
        if args.pathway_family_holdout_seed is not None:
            command.extend(
                ["--pathway-family-holdout-seed", str(args.pathway_family_holdout_seed)]
            )
        if args.overwrite:
            command.append("--overwrite")
        raise SystemExit(subprocess.run(command, check=False).returncode)

    if args.command == "prepare-coverage":
        command = [
            sys.executable,
            "-m",
            "dataprocess.select_training_coverage",
            "--input",
            asset_path("data/train_kegg_pathway_record_balanced_0p1pct.csv"),
            "--train-output",
            asset_path(
                f"data/train_kegg_pathway_coverage_cap{args.max_records_per_family}.csv"
            ),
            "--validation-output",
            asset_path("data/validation_kegg_pathway_family.csv"),
            "--report",
            asset_path(
                "artifacts/dataset/"
                f"family_capped_organism_length_diverse_cap{args.max_records_per_family}_v1.json"
            ),
            "--max-records-per-family",
            str(args.max_records_per_family),
            "--validation-fraction",
            str(args.validation_fraction),
            "--seed",
            str(args.seed),
        ]
        if args.overwrite:
            command.append("--overwrite")
        raise SystemExit(subprocess.run(command, check=False).returncode)

    if args.command == "prepare-structured-data":
        command = [
            sys.executable,
            "-m",
            "dataprocess.build_structured_dataset",
            "--processed-graph-root",
            args.processed_graph_root or asset_path("KEGG_all_new/processed_graph"),
            "--output-dir",
            args.output_dir
            or asset_path(f"data/pathway_v3_cap{args.max_records_per_family}"),
            "--tokenizer",
            asset_path("models/qwen3_8B"),
            "--max-length",
            str(args.max_length),
            "--test-organisms",
            args.test_organisms,
            "--test-family-fraction",
            str(args.test_family_fraction),
            "--validation-family-fraction",
            str(args.validation_family_fraction),
            "--train-candidate-record-fraction",
            str(args.train_candidate_record_fraction),
            "--max-records-per-family",
            str(args.max_records_per_family),
            "--max-prefixes-per-train-record",
            str(args.max_prefixes_per_train_record),
            "--minimum-train-records",
            str(args.minimum_train_records),
            "--seed",
            str(args.seed),
            "--progress-every",
            str(args.progress_every),
        ]
        if args.overwrite:
            command.append("--overwrite")
        raise SystemExit(subprocess.run(command, check=False).returncode)

    if args.command == "download-model":
        command = [
            sys.executable,
            "-m",
            "scripts.model.download_qwen3_8B",
            "--revision",
            args.revision,
            "--target",
            asset_path("models/qwen3_8B"),
        ]
        if args.endpoint:
            command.extend(["--endpoint", args.endpoint])
        if args.verify_only:
            command.append("--verify-only")
        raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
