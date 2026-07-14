"""Audit consistency between experiment wrappers and runtime_manifest.json.

This is stricter than ``validate_matrix``: it expands each wrapper under
``CHATPATHWAY_LAUNCH_DRY_RUN=1``, checks that the dry-run command mentions the
runtime paths declared in the manifest, and verifies that wrapper-passed CLI
options and literal ``choices`` values are declared by the target module's
``argparse`` parser.
"""

from __future__ import annotations

import argparse
import ast
import json
import shlex
import sys
from pathlib import Path
from typing import Any

from experiments.audit_wrappers import audit_module
from experiments.check_runtime_assets import rewrite_asset_path
from experiments.runtime_config import asset_root as configured_asset_root


MATRIX_PATH = Path(__file__).with_name("matrix.json")
RUNTIME_MANIFEST_PATH = Path(__file__).with_name("runtime_manifest.json")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def command_tokens(command: str) -> set[str]:
    return set(shlex.split(command))


def command_parts(command: str) -> tuple[str | None, list[str]]:
    tokens = shlex.split(command)
    module_positions = [index for index, token in enumerate(tokens) if token == "-m" and index + 1 < len(tokens)]
    if not module_positions:
        return None, []
    index = module_positions[-1]
    return tokens[index + 1], tokens[index + 2 :]


def module_to_file(module: str) -> Path:
    return REPOSITORY_ROOT / Path(*module.split(".")).with_suffix(".py")


def literal_string_set(node: ast.AST) -> set[str] | None:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    values: set[str] = set()
    for item in node.elts:
        if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
            return None
        values.add(item.value)
    return values


def declared_cli_options(
    module: str,
    *,
    _visited: set[str] | None = None,
) -> dict[str, set[str] | None]:
    """Collect local options plus options from an explicitly imported parser.

    Thin distributed entry points intentionally reuse the canonical
    ``parse_args`` function rather than copying dozens of CLI declarations.
    Following only explicit ``from ... import parse_args`` imports keeps this
    static audit strict while supporting that single-source-of-truth pattern.
    """

    visited = set(_visited or ())
    if module in visited:
        return {}
    visited.add(module)
    path = module_to_file(module)
    if not path.is_file():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    options: dict[str, set[str] | None] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
            continue
        choices = None
        for keyword in node.keywords:
            if keyword.arg == "choices":
                choices = literal_string_set(keyword.value)
                break
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                options[arg.value.split("=", 1)[0]] = choices
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if any(alias.name == "parse_args" for alias in node.names):
            options.update(declared_cli_options(node.module, _visited=visited))
    return options


def command_cli_options(args: list[str]) -> dict[str, str | None]:
    options: dict[str, str | None] = {}
    i = 0
    while i < len(args):
        value = args[i]
        if not value.startswith("--"):
            i += 1
            continue
        if "=" in value:
            option, option_value = value.split("=", 1)
            options[option] = option_value
            i += 1
            continue
        option = value
        option_value = None
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            option_value = args[i + 1]
            i += 2
        else:
            i += 1
        options[option] = option_value
    return options


def parse_id_list(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def selected_rows(matrix: dict[str, Any], raw_ids: str | None) -> list[dict[str, Any]]:
    rows = list(matrix.get("implemented", []))
    requested = parse_id_list(raw_ids)
    if requested is None:
        return rows
    known = {row["id"] for row in rows}
    unknown = sorted(requested - known)
    if unknown:
        raise SystemExit(f"Unknown experiment id(s): {', '.join(unknown)}")
    return [row for row in rows if row["id"] in requested]


def expand_requirement_paths(
    requirement: str,
    *,
    manifest_rows: dict[str, Any],
    row_ids: set[str],
) -> list[str]:
    if requirement not in row_ids:
        return [requirement]
    dependency = manifest_rows[requirement]
    return list(dependency.get("infer_artifacts", dependency.get("train_outputs", [])))


def phase_requirements(entry: dict[str, Any], phase: str) -> list[str]:
    return list(entry.get(f"{phase}_requires", entry.get("requires", [])))


PATH_FIELDS = {
    "requires",
    "train_requires",
    "infer_requires",
    "train_outputs",
    "infer_artifacts",
    "infer_outputs",
}


def remap_manifest_rows(
    rows: dict[str, Any],
    *,
    manifest_root: str,
    target_root: str,
) -> dict[str, Any]:
    """Resolve static manifest paths into the profile used by the wrappers.

    ``runtime_manifest.json`` is committed with the canonical AutoDL root, but
    wrapper dry-runs resolve paths from the active runtime profile.  Auditing a
    CFFF wrapper against unreplaced AutoDL strings produces false failures.
    Dependency row ids must remain ids, while declared filesystem paths are
    rewritten exactly as the runtime asset checker rewrites them.
    """

    row_ids = set(rows)
    remapped: dict[str, Any] = {}
    for row_id, entry in rows.items():
        output = dict(entry)
        for field in PATH_FIELDS:
            values = entry.get(field)
            if values is None:
                continue
            output[field] = [
                value
                if value in row_ids
                else str(rewrite_asset_path(value, manifest_root, target_root))
                for value in values
            ]
        remapped[row_id] = output
    return remapped


def train_output_anchors(path: str, asset_root: str) -> list[str]:
    """Return acceptable command anchors for a train output path.

    Training wrappers usually receive a save root rather than the final
    epoch-specific checkpoint file, so an ancestor directory is acceptable if it
    is still more specific than the global ``checkpoints`` directory.
    """

    raw = Path(path)
    root = Path(asset_root)
    candidates = [str(raw)]
    candidates.extend(str(parent) for parent in raw.parents)
    filtered = []
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path == root:
            continue
        try:
            rel = candidate_path.relative_to(root)
        except ValueError:
            filtered.append(candidate)
            continue
        parts = rel.parts
        if len(parts) >= 2:
            filtered.append(candidate)
    return filtered


def missing_exact_paths(tokens: set[str], paths: list[str]) -> list[str]:
    return [path for path in paths if path not in tokens]


def check_train_command(
    *,
    experiment_id: str,
    tokens: set[str],
    entry: dict[str, Any],
    manifest_rows: dict[str, Any],
    row_ids: set[str],
    asset_root: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    required_paths: list[str] = []
    for requirement in phase_requirements(entry, "train"):
        required_paths.extend(
            expand_requirement_paths(requirement, manifest_rows=manifest_rows, row_ids=row_ids)
        )
    for path in sorted(set(required_paths)):
        records.append({
            "experiment_id": experiment_id,
            "phase": "train",
            "kind": "train_requirement_in_command",
            "path": path,
            "ok": path in tokens,
        })

    for output in entry.get("train_outputs", []):
        anchors = train_output_anchors(output, asset_root)
        records.append({
            "experiment_id": experiment_id,
            "phase": "train",
            "kind": "train_output_anchor_in_command",
            "path": output,
            "ok": any(anchor in tokens for anchor in anchors),
            "accepted_anchors": anchors,
        })
    return records


def check_infer_command(
    *,
    experiment_id: str,
    tokens: set[str],
    entry: dict[str, Any],
    manifest_rows: dict[str, Any],
    row_ids: set[str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    paths: list[tuple[str, str]] = []
    for requirement in phase_requirements(entry, "infer"):
        for path in expand_requirement_paths(requirement, manifest_rows=manifest_rows, row_ids=row_ids):
            paths.append(("infer_requirement_in_command", path))
    for path in entry.get("infer_artifacts", entry.get("train_outputs", [])):
        paths.append(("infer_artifact_in_command", path))
    for path in entry.get("infer_outputs", []):
        paths.append(("infer_output_in_command", path))

    seen: set[tuple[str, str]] = set()
    for kind, path in paths:
        key = (kind, path)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "experiment_id": experiment_id,
            "phase": "infer",
            "kind": kind,
            "path": path,
            "ok": path in tokens,
        })
    return records


def audit_row(
    row: dict[str, Any],
    *,
    phase: str,
    manifest_rows: dict[str, Any],
    row_ids: set[str],
    asset_root: str,
    env_overrides: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    entry = manifest_rows.get(row["id"])
    if entry is None:
        return [{
            "experiment_id": row["id"],
            "phase": phase,
            "kind": "manifest_entry",
            "path": "",
            "ok": False,
            "note": "missing runtime manifest row",
        }]

    phases: list[tuple[str, str]] = []
    if phase in {"train", "both"}:
        phases.append(("train", row["train_module"]))
    if phase in {"infer", "both"}:
        phases.append(("infer", row["infer_module"]))

    for concrete_phase, module in phases:
        wrapper_record = audit_module(
            module,
            concrete_phase,
            row["id"],
            env_overrides=env_overrides,
        )
        tokens = command_tokens(wrapper_record["inner_command"])
        target_module, target_args = command_parts(wrapper_record["inner_command"])
        records.append({
            "experiment_id": row["id"],
            "phase": concrete_phase,
            "kind": "wrapper_dry_run",
            "path": module,
            "ok": bool(wrapper_record["ok"]),
            "inner_command": wrapper_record["inner_command"],
            "stderr": wrapper_record["stderr"],
        })
        if not wrapper_record["ok"]:
            continue
        if target_module is None:
            records.append({
                "experiment_id": row["id"],
                "phase": concrete_phase,
                "kind": "target_module_detected",
                "path": "",
                "ok": False,
                "note": "could not find -m target module in wrapper command",
            })
            continue
        declared_options = declared_cli_options(target_module)
        if not declared_options:
            records.append({
                "experiment_id": row["id"],
                "phase": concrete_phase,
                "kind": "target_module_argparse_detected",
                "path": target_module,
                "ok": False,
                "note": "target source missing or no argparse add_argument options detected",
            })
            continue
        for option, option_value in sorted(command_cli_options(target_args).items()):
            choices = declared_options.get(option)
            records.append({
                "experiment_id": row["id"],
                "phase": concrete_phase,
                "kind": "target_cli_option_declared",
                "path": f"{target_module} {option}",
                "ok": option in declared_options,
            })
            if option in declared_options and choices is not None and option_value is not None:
                records.append({
                    "experiment_id": row["id"],
                    "phase": concrete_phase,
                    "kind": "target_cli_choice_allowed",
                    "path": f"{target_module} {option}={option_value}",
                    "ok": option_value in choices,
                    "allowed": sorted(choices),
                })
        if concrete_phase == "train":
            records.extend(
                check_train_command(
                    experiment_id=row["id"],
                    tokens=tokens,
                    entry=entry,
                    manifest_rows=manifest_rows,
                    row_ids=row_ids,
                    asset_root=asset_root,
                )
            )
        else:
            records.extend(
                check_infer_command(
                    experiment_id=row["id"],
                    tokens=tokens,
                    entry=entry,
                    manifest_rows=manifest_rows,
                    row_ids=row_ids,
                )
            )
    return records


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--phase", choices=("train", "infer", "both"), default="both")
    parser.add_argument("--ids", help="Comma-separated experiment ids to audit.")
    parser.add_argument("--asset-root", help="Temporarily override the runtime asset root.")
    parser.add_argument("--profile", help="Runtime profile name from chatpathway.config.json.")
    parser.add_argument("--jsonl", help="Optional JSONL output path.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_json(MATRIX_PATH)
    manifest = load_json(RUNTIME_MANIFEST_PATH)
    manifest_root = manifest.get("asset_root", "/root/autodl-tmp")
    target_root = args.asset_root or str(configured_asset_root(args.profile))
    manifest_rows = remap_manifest_rows(
        manifest.get("rows", {}),
        manifest_root=manifest_root,
        target_root=target_root,
    )
    row_ids = set(manifest_rows)
    env_overrides: dict[str, str] = {}
    if args.profile:
        env_overrides["CHATPATHWAY_PROFILE"] = args.profile
    if args.asset_root:
        env_overrides["CHATPATHWAY_ASSET_ROOT"] = args.asset_root

    records: list[dict[str, Any]] = []
    for row in selected_rows(matrix, args.ids):
        records.extend(
            audit_row(
                row,
                phase=args.phase,
                manifest_rows=manifest_rows,
                row_ids=row_ids,
                asset_root=target_root,
                env_overrides=env_overrides,
            )
        )

    if args.jsonl:
        write_jsonl(args.jsonl, records)

    failures = [record for record in records if not record.get("ok")]
    if not args.quiet:
        for record in records:
            status = "ok" if record.get("ok") else "FAIL"
            print(f"{status}\t{record['experiment_id']}\t{record['phase']}\t{record['kind']}\t{record.get('path', '')}")
    if failures:
        print(f"Matrix consistency audit failed for {len(failures)} records.", file=sys.stderr)
        raise SystemExit(1)
    print(f"Matrix consistency audit passed for {len(records)} records.")


if __name__ == "__main__":
    main()
