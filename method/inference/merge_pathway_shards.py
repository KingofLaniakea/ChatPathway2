#!/usr/bin/env python3
"""Verify and merge deterministic pathway-inference shards in input order."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from method.inference.csv_io import read_csv_text_rows


PREDICTION_FIELDS = (
    "predicted_answer",
    "generated_token_count",
    "total_generated_token_count",
    "finish_reason",
    "generation_attempts",
    "prediction_json_valid",
    "prediction_schema_valid",
)
SHARED_RUN_FIELDS = (
    "base_model_id",
    "trained_lora_path",
    "test_data_path",
    "batch_size",
    "max_length",
    "max_new_tokens",
    "max_json_attempts",
    "retry_max_new_tokens",
    "limit",
    "shard_count",
    "seed",
    "completion_marker",
    "git_commit",
    "input_sha256",
    "completion_marker_sha256",
    "input_rows",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_progress(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"progress record must be an object: {path}:{line_number}")
            rows.append(value)
    return rows


def _index(value: Any, *, context: str) -> int:
    try:
        index = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid dataset index in {context}: {value!r}") from exc
    if index < 0:
        raise ValueError(f"negative dataset index in {context}: {index}")
    return index


def _truth(value: Any) -> bool:
    return value is True or str(value).strip().casefold() in {"1", "true", "yes"}


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            quoting=csv.QUOTE_MINIMAL,
            escapechar="\\",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge_shards(
    *,
    input_path: str | Path,
    shard_outputs: Sequence[str | Path],
    shard_progress: Sequence[str | Path],
    output_path: str | Path,
    progress_output_path: str | Path,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    source_path = Path(input_path)
    outputs = [Path(path) for path in shard_outputs]
    progress_paths = [Path(path) for path in shard_progress]
    destination = Path(output_path)
    progress_destination = Path(progress_output_path)
    metadata_destination = destination.with_suffix(".run.json")
    if not outputs or len(outputs) != len(progress_paths):
        raise ValueError("equal non-empty --shard-output and --shard-progress lists are required")
    if len(set(outputs)) != len(outputs) or len(set(progress_paths)) != len(progress_paths):
        raise ValueError("shard artifact paths must be unique")
    for path in (destination, progress_destination, metadata_destination):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing merged artifact: {path}")
    for path in (*outputs, *progress_paths):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_fields, source_rows = read_csv_text_rows(source_path, limit=limit)
    expected_indices = set(range(len(source_rows)))
    output_by_index: dict[int, dict[str, str]] = {}
    progress_by_index: dict[int, dict[str, Any]] = {}
    run_manifests: list[tuple[Path, dict[str, Any]]] = []
    output_fields: list[str] | None = None

    for shard_output, shard_progress_path in zip(outputs, progress_paths):
        fields, rows = read_csv_text_rows(shard_output)
        required = {"dataset_index", *source_fields, *PREDICTION_FIELDS}
        missing = sorted(required - set(fields))
        if missing:
            raise ValueError(f"{shard_output} is missing fields: {', '.join(missing)}")
        if output_fields is None:
            output_fields = fields
        elif fields != output_fields:
            raise ValueError("all shard CSV headers must be identical")
        for local_index, row in enumerate(rows):
            index = _index(row["dataset_index"], context=f"{shard_output}:row {local_index + 2}")
            if index not in expected_indices:
                raise ValueError(f"out-of-range dataset index {index} in {shard_output}")
            if index in output_by_index:
                raise ValueError(f"duplicate dataset index {index} across shard CSVs")
            mismatches = [field for field in source_fields if row[field] != source_rows[index][field]]
            if mismatches:
                raise ValueError(
                    f"source provenance mismatch at dataset index {index}: {', '.join(mismatches[:5])}"
                )
            output_by_index[index] = row

        for local_index, row in enumerate(_read_progress(shard_progress_path)):
            index = _index(row.get("sample_index"), context=f"{shard_progress_path}:record {local_index + 1}")
            if index not in expected_indices:
                raise ValueError(f"out-of-range sample_index {index} in {shard_progress_path}")
            if index in progress_by_index:
                raise ValueError(f"duplicate sample_index {index} across shard progress files")
            progress_by_index[index] = row

        run_path = shard_output.with_suffix(".run.json")
        if not run_path.is_file():
            raise FileNotFoundError(run_path)
        manifest = _read_json(run_path)
        if manifest.get("progress_output_sha256") != file_sha256(shard_progress_path):
            raise ValueError(f"progress hash mismatch for {shard_progress_path}")
        if int(manifest.get("evaluated_rows", -1)) != len(rows):
            raise ValueError(f"evaluated_rows mismatch in {run_path}")
        run_manifests.append((run_path, manifest))

    missing_csv = sorted(expected_indices - output_by_index.keys())
    missing_progress = sorted(expected_indices - progress_by_index.keys())
    if missing_csv or missing_progress:
        raise ValueError(
            f"incomplete shard coverage: missing_csv={missing_csv[:10]} "
            f"missing_progress={missing_progress[:10]}"
        )
    if output_by_index.keys() != progress_by_index.keys():
        raise ValueError("shard CSV and progress index sets differ")
    provenance_fields = {
        "sample_id": "sample_id",
        "base_sample_id": "base_sample_id",
        "record_id": "record_id",
        "graph_id": "graph_id",
        "view_id": "view_id",
        "organism": "organism",
        "pathway_family_id": "pathway_family_id",
        "source_graph_json": "source_graph_json",
        "prompt_profile": "prompt_profile",
        "organism_conditioning": "organism_conditioning",
        "entity_id_space": "entity_id_space",
        "entity_mapping_status": "entity_mapping_status",
        "prefix_horizon": "prefix_horizon",
        "gold_answer": "answer",
    }
    for index in range(len(source_rows)):
        source_row = source_rows[index]
        output_row = output_by_index[index]
        progress_row = progress_by_index[index]
        for progress_field, source_field in provenance_fields.items():
            if source_field not in source_fields:
                continue
            expected = source_row.get(source_field, "")
            if str(progress_row.get(progress_field, "")) != expected:
                raise ValueError(
                    f"progress provenance mismatch at dataset index {index}: {progress_field}"
                )
        for field in ("predicted_answer", "finish_reason"):
            if str(progress_row.get(field, "")) != output_row[field]:
                raise ValueError(f"CSV/progress mismatch at dataset index {index}: {field}")
        for field in (
            "generated_token_count",
            "total_generated_token_count",
            "generation_attempts",
        ):
            if _index(
                progress_row.get(field),
                context=f"progress {field} at dataset index {index}",
            ) != _index(
                output_row[field],
                context=f"CSV {field} at dataset index {index}",
            ):
                raise ValueError(f"CSV/progress mismatch at dataset index {index}: {field}")
        attempts = _index(
            output_row["generation_attempts"],
            context=f"generation_attempts at dataset index {index}",
        )
        if not 1 <= attempts <= 3:
            raise ValueError(
                f"generation_attempts must be in [1, 3] at dataset index {index}"
            )
        for field in ("prediction_json_valid", "prediction_schema_valid"):
            if _truth(progress_row.get(field)) != _truth(output_row[field]):
                raise ValueError(f"CSV/progress mismatch at dataset index {index}: {field}")
        if not _truth(output_row["prediction_json_valid"]) or not _truth(
            output_row["prediction_schema_valid"]
        ):
            raise ValueError(f"strict-invalid prediction reached final shard at dataset index {index}")
        if progress_row.get("status") != "completed":
            raise ValueError(f"progress record is not completed at dataset index {index}")
    first_manifest = run_manifests[0][1]
    for field in SHARED_RUN_FIELDS:
        values = [manifest.get(field) for _, manifest in run_manifests]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"shard run manifests disagree on {field!r}: {values!r}")
    shard_count = int(first_manifest.get("shard_count", -1))
    shard_indices = sorted(int(manifest.get("shard_index", -1)) for _, manifest in run_manifests)
    if shard_count != len(outputs) or shard_indices != list(range(shard_count)):
        raise ValueError(
            f"shard manifest coordinates are incomplete: count={shard_count}, indices={shard_indices}"
        )
    if int(first_manifest.get("input_rows", -1)) != len(source_rows):
        raise ValueError("shard input_rows does not match merge input")

    merged_rows = [output_by_index[index] for index in range(len(source_rows))]
    merged_progress = [progress_by_index[index] for index in range(len(source_rows))]
    assert output_fields is not None
    destination.parent.mkdir(parents=True, exist_ok=True)
    progress_destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(destination, output_fields, merged_rows)
    _atomic_jsonl(progress_destination, merged_progress)

    finish_counts = Counter(row["finish_reason"] for row in merged_rows)
    attempt_counts = Counter(row["generation_attempts"] for row in merged_rows)
    metadata = {
        "merge_schema_version": 1,
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
        "input": str(source_path),
        "input_sha256": file_sha256(source_path),
        "input_rows": len(source_rows),
        "output": str(destination),
        "output_sha256": file_sha256(destination),
        "progress_output": str(progress_destination),
        "progress_output_sha256": file_sha256(progress_destination),
        "shard_count": shard_count,
        "shared_run_config": {field: first_manifest.get(field) for field in SHARED_RUN_FIELDS},
        "shards": [
            {
                "output": str(shard_output),
                "output_sha256": file_sha256(shard_output),
                "progress": str(shard_progress_path),
                "progress_sha256": file_sha256(shard_progress_path),
                "run_manifest": str(run_path),
                "run_manifest_sha256": file_sha256(run_path),
            }
            for shard_output, shard_progress_path, (run_path, _) in zip(
                outputs, progress_paths, run_manifests
            )
        ],
        "finish_reason_counts": dict(sorted(finish_counts.items())),
        "generation_attempt_counts": dict(sorted(attempt_counts.items())),
        "total_generated_tokens_including_repairs": sum(
            int(row["total_generated_token_count"]) for row in merged_rows
        ),
        "prediction_json_valid_count": sum(_truth(row["prediction_json_valid"]) for row in merged_rows),
        "prediction_schema_valid_count": sum(_truth(row["prediction_schema_valid"]) for row in merged_rows),
    }
    _atomic_json(metadata_destination, metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--shard-output", action="append", required=True)
    parser.add_argument("--shard-progress", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    metadata = merge_shards(
        input_path=args.input,
        shard_outputs=args.shard_output,
        shard_progress=args.shard_progress,
        output_path=args.output,
        progress_output_path=args.progress_output,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
