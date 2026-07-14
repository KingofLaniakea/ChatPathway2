"""Deterministic per-source hashes for a structured dataset release."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Mapping


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_hash_record(graph_root: Path, relative_path: str) -> dict[str, object]:
    normalized = Path(relative_path).as_posix()
    if normalized.startswith("/") or ".." in Path(normalized).parts:
        raise ValueError(f"source graph path must be relative and contained: {relative_path!r}")
    artifact = graph_root / normalized
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    return {
        "source_graph_json": normalized,
        "bytes": artifact.stat().st_size,
        "sha256": file_sha256(artifact),
    }


def write_source_graph_hashes(
    graph_root: Path,
    sources: Iterable[str],
    output_path: Path,
    *,
    overwrite: bool,
) -> dict[str, object]:
    """Write an atomic, sorted JSONL hash inventory for referenced graphs."""

    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    normalized_sources = sorted({Path(value).as_posix() for value in sources})
    if not normalized_sources:
        raise ValueError("source graph hash inventory cannot be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for source in normalized_sources:
            handle.write(
                json.dumps(
                    source_hash_record(graph_root, source),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    temporary.replace(output_path)
    return {
        "path": output_path.name,
        "records": len(normalized_sources),
        "sha256": file_sha256(output_path),
    }


def verify_source_graph_hashes(
    graph_root: Path,
    inventory_path: Path,
    *,
    expected_sources: Iterable[str] | None = None,
) -> dict[str, object]:
    """Verify inventory syntax, uniqueness, content hashes, and optional coverage."""

    observed: set[str] = set()
    errors: list[str] = []
    with inventory_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, Mapping) or set(value) != {
                    "source_graph_json",
                    "bytes",
                    "sha256",
                }:
                    raise ValueError("record must have source_graph_json, bytes, sha256")
                source = str(value["source_graph_json"])
                if source in observed:
                    raise ValueError(f"duplicate source_graph_json {source!r}")
                expected = source_hash_record(graph_root, source)
                if dict(value) != expected:
                    raise ValueError(f"source hash or size mismatch for {source!r}")
                observed.add(source)
            except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
                errors.append(f"line {line_number}: {exc}")
    expected = (
        {Path(value).as_posix() for value in expected_sources}
        if expected_sources is not None
        else None
    )
    if expected is not None and observed != expected:
        errors.append(
            "source inventory coverage mismatch: "
            f"missing={len(expected - observed)} extra={len(observed - expected)}"
        )
    return {
        "path": str(inventory_path),
        "records": len(observed),
        "sha256": file_sha256(inventory_path),
        "errors": errors,
    }


__all__ = [
    "file_sha256",
    "source_hash_record",
    "verify_source_graph_hashes",
    "write_source_graph_hashes",
]
