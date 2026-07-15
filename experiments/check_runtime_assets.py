"""Check runtime data, checkpoint, and output paths for experiment rows.

This checker is intentionally lightweight: it only reads the experiment matrix
and runtime manifest, so it can run before installing or importing torch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

from experiments.runtime_config import asset_root as configured_asset_root


MATRIX_PATH = Path(__file__).with_name("matrix.json")
RUNTIME_MANIFEST_PATH = Path(__file__).with_name("runtime_manifest.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_id_list(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    values = {item.strip() for item in raw.split(",") if item.strip()}
    return values or None


def implemented_ids(matrix: dict[str, Any]) -> list[str]:
    return [row["id"] for row in matrix.get("implemented", [])]


def select_ids(all_ids: list[str], raw_ids: str | None) -> list[str]:
    requested = parse_id_list(raw_ids)
    if requested is None:
        return all_ids
    unknown = sorted(requested - set(all_ids))
    if unknown:
        raise SystemExit(f"Unknown experiment id(s): {', '.join(unknown)}")
    return [experiment_id for experiment_id in all_ids if experiment_id in requested]


def rewrite_asset_path(raw_path: str, manifest_root: str, asset_root: str | None) -> Path:
    expanded = os.path.expanduser(raw_path)
    dataset_namespace = os.environ.get("CHATPATHWAY_DATASET_NAMESPACE")
    if dataset_namespace:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_namespace):
            raise ValueError("CHATPATHWAY_DATASET_NAMESPACE is not filesystem-safe")
        expanded = re.sub(
            r"/(checkpoints|runs|artifacts)/datasets/[^/]+/seeds/",
            rf"/\1/datasets/{dataset_namespace}/seeds/",
            expanded,
        )
    seed = os.environ.get("CHATPATHWAY_EXPERIMENT_SEED")
    if seed:
        expanded = expanded.replace("/seeds/20260711/", f"/seeds/{seed}/")
    if asset_root and expanded.startswith(manifest_root.rstrip("/") + "/"):
        suffix = expanded[len(manifest_root.rstrip("/")) + 1 :]
        return Path(asset_root) / suffix
    if asset_root and expanded == manifest_root:
        return Path(asset_root)
    return Path(expanded)


def path_record(
    *,
    experiment_id: str,
    phase: str,
    kind: str,
    raw_path: str,
    path: Path,
    required: bool,
    note: str = "",
) -> dict[str, Any]:
    exists = path.exists()
    ok = exists or not required
    status = "ok" if exists else ("missing" if required else "absent")
    return {
        "experiment_id": experiment_id,
        "phase": phase,
        "kind": kind,
        "raw_path": raw_path,
        "path": str(path),
        "exists": exists,
        "required": required,
        "ok": ok,
        "status": status,
        "note": note,
    }


def parent_record(
    *,
    experiment_id: str,
    phase: str,
    kind: str,
    raw_path: str,
    path: Path,
    create: bool,
) -> dict[str, Any]:
    parent = path.parent
    created = False
    if create and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
        created = True
    exists = parent.exists()
    writable = os.access(parent, os.W_OK) if exists else False
    ok = exists and writable
    if ok:
        status = "created" if created else "ok"
    elif exists:
        status = "not_writable"
    else:
        status = "parent_missing"
    return {
        "experiment_id": experiment_id,
        "phase": phase,
        "kind": kind,
        "raw_path": raw_path,
        "path": str(parent),
        "output_path": str(path),
        "exists": exists,
        "writable": writable,
        "created": created,
        "required": True,
        "ok": ok,
        "status": status,
        "note": "output parent directory",
    }


def dependency_records(
    *,
    experiment_id: str,
    phase: str,
    dependency_id: str,
    manifest_rows: dict[str, Any],
    manifest_root: str,
    asset_root: str | None,
) -> list[dict[str, Any]]:
    dependency = manifest_rows.get(dependency_id)
    if dependency is None:
        return [
            {
                "experiment_id": experiment_id,
                "phase": phase,
                "kind": "row_dependency",
                "dependency_id": dependency_id,
                "required": True,
                "ok": False,
                "status": "unknown_dependency",
                "note": "dependency id is not present in runtime manifest",
            }
        ]

    records: list[dict[str, Any]] = []
    for raw_path in dependency.get("train_outputs", []):
        path = rewrite_asset_path(raw_path, manifest_root, asset_root)
        records.append(
            path_record(
                experiment_id=experiment_id,
                phase=phase,
                kind="row_dependency_train_output",
                raw_path=raw_path,
                path=path,
                required=True,
                note=f"requires {dependency_id}",
            )
        )
    return records


def phase_requirements(entry: dict[str, Any], phase: str) -> list[str]:
    phase_key = f"{phase}_requires"
    value = entry.get(phase_key)
    if value is not None:
        return list(value)
    return list(entry.get("requires", []))


def check_requirements(
    *,
    experiment_id: str,
    phase: str,
    entry: dict[str, Any],
    manifest_rows: dict[str, Any],
    row_ids: set[str],
    manifest_root: str,
    asset_root: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for requirement in phase_requirements(entry, phase):
        if requirement in row_ids:
            records.extend(
                dependency_records(
                    experiment_id=experiment_id,
                    phase=phase,
                    dependency_id=requirement,
                    manifest_rows=manifest_rows,
                    manifest_root=manifest_root,
                    asset_root=asset_root,
                )
            )
            continue
        path = rewrite_asset_path(requirement, manifest_root, asset_root)
        records.append(
            path_record(
                experiment_id=experiment_id,
                phase=phase,
                kind="requirement",
                raw_path=requirement,
                path=path,
                required=True,
            )
        )
    return records


def check_trained_artifacts(
    *,
    experiment_id: str,
    phase: str,
    entry: dict[str, Any],
    manifest_root: str,
    asset_root: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    artifact_paths = entry.get("infer_artifacts", entry.get("train_outputs", []))
    for raw_path in artifact_paths:
        path = rewrite_asset_path(raw_path, manifest_root, asset_root)
        records.append(
            path_record(
                experiment_id=experiment_id,
                phase=phase,
                kind="trained_artifact_for_inference",
                raw_path=raw_path,
                path=path,
                required=True,
                note="current row train output",
            )
        )
    return records


def check_output_parents(
    *,
    experiment_id: str,
    phase: str,
    entry: dict[str, Any],
    manifest_root: str,
    asset_root: str | None,
    create_output_dirs: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    field = "train_outputs" if phase == "train" else "infer_outputs"
    output_paths = [
        rewrite_asset_path(raw_path, manifest_root, asset_root)
        for raw_path in entry.get(field, [])
    ]
    output_path_set = set(output_paths)
    for raw_path, path in zip(entry.get(field, []), output_paths):
        if path.parent in output_path_set:
            continue
        records.append(
            parent_record(
                experiment_id=experiment_id,
                phase=phase,
                kind=f"{field}_parent",
                raw_path=raw_path,
                path=path,
                create=create_output_dirs,
            )
        )
    return records


def check_expected_outputs(
    *,
    experiment_id: str,
    phase: str,
    entry: dict[str, Any],
    manifest_root: str,
    asset_root: str | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    field = "train_outputs" if phase == "train" else "infer_outputs"
    for raw_path in entry.get(field, []):
        path = rewrite_asset_path(raw_path, manifest_root, asset_root)
        records.append(
            path_record(
                experiment_id=experiment_id,
                phase=phase,
                kind=f"expected_{field}",
                raw_path=raw_path,
                path=path,
                required=True,
                note="expected result artifact",
            )
        )
    return records


def iter_phases(phase: str) -> Iterable[str]:
    if phase == "both":
        return ("train", "infer")
    return (phase,)


def check_rows(
    *,
    ids: list[str],
    phase: str,
    manifest: dict[str, Any],
    check_outputs: bool,
    create_output_dirs: bool,
    asset_root: str | None,
) -> list[dict[str, Any]]:
    manifest_rows = manifest.get("rows", {})
    row_ids = set(manifest_rows)
    manifest_root = manifest.get("asset_root", "/root/autodl-tmp")
    records: list[dict[str, Any]] = []

    for experiment_id in ids:
        entry = manifest_rows.get(experiment_id)
        if entry is None:
            records.append(
                {
                    "experiment_id": experiment_id,
                    "phase": phase,
                    "kind": "manifest_entry",
                    "required": True,
                    "ok": False,
                    "status": "missing_manifest_entry",
                    "note": "row is missing from runtime_manifest.json",
                }
            )
            continue

        for concrete_phase in iter_phases(phase):
            records.extend(
                check_requirements(
                    experiment_id=experiment_id,
                    phase=concrete_phase,
                    entry=entry,
                    manifest_rows=manifest_rows,
                    row_ids=row_ids,
                    manifest_root=manifest_root,
                    asset_root=asset_root,
                )
            )
            if concrete_phase == "infer":
                records.extend(
                    check_trained_artifacts(
                        experiment_id=experiment_id,
                        phase=concrete_phase,
                        entry=entry,
                        manifest_root=manifest_root,
                        asset_root=asset_root,
                    )
                )
            records.extend(
                check_output_parents(
                    experiment_id=experiment_id,
                    phase=concrete_phase,
                    entry=entry,
                    manifest_root=manifest_root,
                    asset_root=asset_root,
                    create_output_dirs=create_output_dirs,
                )
            )
            if check_outputs:
                records.extend(
                    check_expected_outputs(
                        experiment_id=experiment_id,
                        phase=concrete_phase,
                        entry=entry,
                        manifest_root=manifest_root,
                        asset_root=asset_root,
                    )
                )
    return records


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_records(records: list[dict[str, Any]], quiet: bool) -> None:
    if not quiet:
        for record in records:
            status = "ok" if record["ok"] else "MISSING"
            path = record.get("path") or record.get("dependency_id") or ""
            note = f"\t{record['note']}" if record.get("note") else ""
            print(
                f"{status}\t{record['experiment_id']}\t{record['phase']}\t"
                f"{record['kind']}\t{record['status']}\t{path}{note}"
            )

    missing_required = [record for record in records if record.get("required") and not record.get("ok")]
    print(f"Checked {len(records)} runtime asset records; missing/invalid required records: {len(missing_required)}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ids", help="Comma-separated experiment ids to check. Defaults to all implemented rows.")
    parser.add_argument("--phase", choices=("train", "infer", "both"), default="both")
    parser.add_argument(
        "--asset-root",
        help="Temporarily override the asset root from chatpathway.config.json.",
    )
    parser.add_argument("--profile", help="Runtime profile name from chatpathway.config.json.")
    parser.add_argument("--check-outputs", action="store_true", help="Also require expected result artifacts to exist.")
    parser.add_argument(
        "--create-output-dirs",
        action="store_true",
        help="Create parent directories for expected train/infer outputs before checking writability.",
    )
    parser.add_argument("--jsonl", help="Optional JSONL output path for machine-readable records.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any required record is missing or invalid.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final summary.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix = load_json(MATRIX_PATH)
    manifest = load_json(RUNTIME_MANIFEST_PATH)
    ids = select_ids(implemented_ids(matrix), args.ids)
    asset_root = args.asset_root or str(configured_asset_root(args.profile))
    records = check_rows(
        ids=ids,
        phase=args.phase,
        manifest=manifest,
        check_outputs=args.check_outputs,
        create_output_dirs=args.create_output_dirs,
        asset_root=asset_root,
    )
    if args.jsonl:
        write_jsonl(args.jsonl, records)
    print_records(records, args.quiet)

    if args.strict and any(record.get("required") and not record.get("ok") for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
