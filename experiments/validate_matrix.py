"""Validate that every implemented experiment row has runnable entry modules."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

from dataprocess.release_contract_v4 import (
    ALL_SPLITS,
    AUDIT_SCHEMA_VERSION,
    CSV_NAMES,
    PRIMARY_PROMPT_PROFILE,
    RECORD_NAMES,
    RELEASE_SCHEMA_VERSION,
    SOURCE_GRAPH_HASHES_NAME,
)

MATRIX_PATH = Path(__file__).with_name("matrix.json")
RUNTIME_MANIFEST_PATH = Path(__file__).with_name("runtime_manifest.json")
METHODS_DIR = Path(__file__).parent / "methods"
REQUIRED_FIELDS = (
    "id",
    "title",
    "granularity",
    "middle_network",
    "training_coupling",
    "inference_mode",
    "train_module",
    "infer_module",
    "notes",
)
POST_CURRENT_GENERATION_IDS = {
    "plan011_graph_layer_boundary_stepwise",
    "plan012_token_resolution_stepwise",
    "plan013_multiscale_hybrid_generation",
}
EXPECTED_DIAGNOSTIC_PARTITIONS = ("test", "test_organism", "test_strict")


def validate_dataset_profile(matrix: dict[str, Any]) -> list[str]:
    """Keep the human-readable matrix aligned with the enforced v4 release."""

    errors: list[str] = []
    profile = matrix.get("dataset_profile", {})
    expected_root = "data/pathway_v4_full"
    expected_paths = {
        "train": f"{expected_root}/{CSV_NAMES['train']}",
        "validation": f"{expected_root}/{CSV_NAMES['validation']}",
    }
    for name, expected in expected_paths.items():
        if profile.get(name) != expected:
            errors.append(f"dataset_profile.{name} must be {expected!r}")
    diagnostics = profile.get("diagnostic_tests", {})
    if set(diagnostics) != set(EXPECTED_DIAGNOSTIC_PARTITIONS):
        errors.append("dataset_profile.diagnostic_tests must declare all three test partitions")
    else:
        for partition in EXPECTED_DIAGNOSTIC_PARTITIONS:
            expected = f"{expected_root}/{CSV_NAMES[partition]}"
            if not str(diagnostics[partition]).startswith(expected):
                errors.append(
                    f"dataset_profile.diagnostic_tests.{partition} must start with {expected!r}"
                )
    records = profile.get("record_jsonl", {})
    expected_records = {
        partition: f"{expected_root}/{RECORD_NAMES[partition]}"
        for partition in ALL_SPLITS
    }
    if records != expected_records:
        errors.append("dataset_profile.record_jsonl does not match the five v4 record files")
    checks = {
        "release_schema": RELEASE_SCHEMA_VERSION,
        "audit_schema": AUDIT_SCHEMA_VERSION,
        "primary_prompt_profile": PRIMARY_PROMPT_PROFILE,
        "release_manifest": f"{expected_root}/dataset_manifest.json",
        "immutable_audit": f"{expected_root}/data_audit.json",
        "source_graph_hashes": f"{expected_root}/{SOURCE_GRAPH_HASHES_NAME}",
    }
    for field, expected in checks.items():
        if profile.get(field) != expected:
            errors.append(f"dataset_profile.{field} must be {expected!r}")
    if matrix.get("sequence_budget", {}).get("max_length") != 8192:
        errors.append("sequence_budget.max_length must remain 8192 for release v4")
    return errors


def module_to_file(module: str) -> Path:
    return Path(*module.split(".")).with_suffix(".py")


def validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if not str(row.get(field, "")).strip():
            errors.append(f"{row.get('id', '<missing id>')}: missing or empty {field}")
    experiment_id = row.get("id")
    if not experiment_id:
        return ["row is missing id"]

    expected_dir = METHODS_DIR / str(experiment_id)
    if not expected_dir.is_dir():
        errors.append(f"{experiment_id}: missing directory {expected_dir}")
    settings_path = expected_dir / "settings.json"
    if not settings_path.is_file():
        errors.append(f"{experiment_id}: missing settings file {settings_path}")
    else:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        for field in ("a_dynamics", "b_ae", "d_training_schedule", "c_inference", "status"):
            if settings.get(field) != row.get(field):
                errors.append(
                    f"{experiment_id}: settings.{field}={settings.get(field)!r} "
                    f"does not match matrix value {row.get(field)!r}"
                )

    for key, filename in (("train_module", "train.py"), ("infer_module", "infer.py")):
        module = row.get(key)
        if not module:
            errors.append(f"{experiment_id}: missing {key}")
            continue
        if importlib.util.find_spec(module) is None:
            errors.append(f"{experiment_id}: cannot resolve module {module}")
        expected_file = expected_dir / filename
        if not expected_file.is_file():
            errors.append(f"{experiment_id}: missing file {expected_file}")
        module_file = Path.cwd() / module_to_file(module)
        if module_file.exists() and module_file.resolve() != expected_file.resolve():
            errors.append(f"{experiment_id}: {key} points to {module_file}, expected {expected_file}")
    return errors


def validate_runtime_manifest(row_ids: set[str]) -> list[str]:
    errors: list[str] = []
    if not RUNTIME_MANIFEST_PATH.is_file():
        return [f"missing runtime manifest: {RUNTIME_MANIFEST_PATH}"]
    manifest = json.loads(RUNTIME_MANIFEST_PATH.read_text(encoding="utf-8"))
    rows = manifest.get("rows", {})
    manifest_ids = set(rows)
    missing = sorted(row_ids - manifest_ids)
    extra = sorted(manifest_ids - row_ids)
    if missing:
        errors.append(f"runtime_manifest missing rows: {', '.join(missing)}")
    if extra:
        errors.append(f"runtime_manifest has unknown rows: {', '.join(extra)}")

    required_fields = ("requires", "train_outputs", "infer_outputs", "runtime_notes")
    for row_id, entry in rows.items():
        for field in required_fields:
            if field not in entry:
                errors.append(f"{row_id}: runtime_manifest missing {field}")
        for field in ("requires", "train_outputs", "infer_outputs"):
            value = entry.get(field)
            if not isinstance(value, list):
                errors.append(f"{row_id}: runtime_manifest {field} must be a list")
            elif field != "requires" and not value:
                errors.append(f"{row_id}: runtime_manifest {field} must not be empty")
            elif any(not isinstance(item, str) or not item.strip() for item in value):
                errors.append(f"{row_id}: runtime_manifest {field} has blank/non-string entries")
        for field in ("train_requires", "infer_requires", "infer_artifacts"):
            if field not in entry:
                continue
            value = entry.get(field)
            if not isinstance(value, list):
                errors.append(f"{row_id}: runtime_manifest {field} must be a list")
            elif any(not isinstance(item, str) or not item.strip() for item in value):
                errors.append(f"{row_id}: runtime_manifest {field} has blank/non-string entries")
        notes = entry.get("runtime_notes")
        if not isinstance(notes, str) or not notes.strip():
            errors.append(f"{row_id}: runtime_manifest runtime_notes must be a non-empty string")
        for field in ("requires", "train_requires", "infer_requires"):
            for requirement in entry.get(field, []):
                if isinstance(requirement, str) and requirement.startswith("a") and requirement not in row_ids and not requirement.startswith("/"):
                    errors.append(f"{row_id}: runtime_manifest {field} references unknown row dependency {requirement}")
    return errors


def validate_research_plan(matrix: dict[str, Any]) -> list[str]:
    """Keep deferred generation questions explicit without making them runnable."""

    errors: list[str] = []
    for row in matrix.get("implemented", []):
        inference = row.get("c_inference")
        if inference not in {"artifact_check", "c0_direct_lora"}:
            errors.append(
                f"{row.get('id', '<missing id>')}: current matrix must use direct inference; got {inference!r}"
            )
    combinations = matrix.get("combinations", [])
    by_id = {item.get("id"): item for item in combinations}
    missing = sorted(POST_CURRENT_GENERATION_IDS - set(by_id))
    if missing:
        errors.append(f"missing post-current generation studies: {', '.join(missing)}")
        return errors
    for experiment_id in POST_CURRENT_GENERATION_IDS:
        item = by_id[experiment_id]
        if item.get("status") != "deferred_until_after_current_matrix":
            errors.append(f"{experiment_id}: must remain deferred until the current matrix completes")
        if item.get("research_phase") != "post_current_hamiltonian_matrix":
            errors.append(f"{experiment_id}: missing post-current research phase")
        if item.get("not_removed") is not True:
            errors.append(f"{experiment_id}: must be explicitly marked not_removed")
    token_item = by_id["plan012_token_resolution_stepwise"]
    if token_item.get("requires_separately_trained_token_resolution_dynamics") is not True:
        errors.append("plan012_token_resolution_stepwise: token-resolution dynamics must be trained separately")
    if token_item.get("forbids_graph_layer_checkpoint_per_token") is not True:
        errors.append("plan012_token_resolution_stepwise: graph-layer checkpoint reuse per token must be forbidden")
    return errors


def main() -> None:
    matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
    rows = matrix.get("implemented", [])
    seen: set[str] = set()
    errors: list[str] = []
    for row in rows:
        experiment_id = row.get("id")
        if experiment_id in seen:
            errors.append(f"duplicate id: {experiment_id}")
        seen.add(experiment_id)
        errors.extend(validate_row(row))
    errors.extend(validate_runtime_manifest(seen))
    errors.extend(validate_dataset_profile(matrix))
    errors.extend(validate_research_plan(matrix))

    if errors:
        for error in errors:
            print(error)
        raise SystemExit(1)
    print(f"Validated {len(rows)} implemented experiment rows.")


if __name__ == "__main__":
    main()
