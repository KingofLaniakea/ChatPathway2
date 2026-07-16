#!/usr/bin/env python3
"""Build or resume the full compressed v4 canonical record index.

Every ``processed_graph`` JSON is scanned exactly once for one generator
contract.  No train/test decision and no sampling fraction is applied here.
Changing a later token budget therefore never requires reparsing the 204 GB
source corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import sys
import zlib
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from dataprocess.event_text import (
    TEXT_SOURCE_CANONICAL_FALLBACK,
    TEMPLATE_ASSET,
    template_provenance,
)
from dataprocess.schemas import canonical_pathway_family_id
from dataprocess.split_policy_v4 import stable_rank
from dataprocess.structured_schema import (
    DATASET_SCHEMA_VERSION,
    compact_json,
    graph_events,
    graph_id_for_source,
    record_from_object,
)
from dataprocess.structured_views import build_structured_records


INDEX_SCHEMA_VERSION = "chatpathway_canonical_index_v4.0"
DEFAULT_WORKERS = 64
DEFAULT_BATCH_SIZE = 8
CONTRACT_FILES = (
    Path(__file__).resolve(),
    Path(__file__).resolve().parent / "structured_schema.py",
    Path(__file__).resolve().parent / "structured_views.py",
    Path(__file__).resolve().parent / "event_text.py",
    Path(__file__).resolve().parent / "entity_projection.py",
    Path(__file__).resolve().parent / "prompt_profiles.py",
    TEMPLATE_ASSET,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def generator_contract() -> dict[str, object]:
    files = {
        path.name: file_sha256(path)
        for path in CONTRACT_FILES
    }
    return {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "record_schema_version": DATASET_SCHEMA_VERSION,
        "files": files,
        "step12_template_provenance": template_provenance(),
        "contract_sha256": hashlib.sha256(
            compact_json(files).encode("utf-8")
        ).hexdigest(),
    }


def iter_graph_files(root: Path) -> Iterator[Path]:
    for directory, directory_names, file_names in os.walk(root):
        directory_names.sort()
        for file_name in sorted(file_names):
            if file_name.endswith(".json"):
                yield Path(directory) / file_name


def source_identity(path: Path, root: Path) -> tuple[str, str, str]:
    relative = path.relative_to(root).as_posix()
    organism = relative.split("/", 1)[0] if "/" in relative else ""
    family = canonical_pathway_family_id(path.stem)
    return relative, organism, family


@dataclass(frozen=True)
class ScanTask:
    path: str
    processed_path: str | None
    relative: str
    organism: str
    family: str
    seed: int


@dataclass(frozen=True)
class ScanBatch:
    tasks: tuple[ScanTask, ...]


@dataclass(frozen=True)
class ScanResult:
    graph_row: tuple[object, ...]
    record_rows: tuple[tuple[object, ...], ...]


@dataclass(frozen=True)
class IndexedSource:
    graph_size: int
    graph_sha256: str
    processed_size: int
    processed_sha256: str
    processed_text_status: str


def _stable_error(exc: BaseException) -> str:
    if isinstance(exc, OSError):
        detail = exc.strerror or "I/O error"
    else:
        detail = str(exc)
    return f"{type(exc).__name__}: {detail}"[:2000]


def _graph_row(
    task: ScanTask,
    *,
    graph_id: str,
    content_sha256: str,
    file_size: int,
    status: str,
    error_label: str = "",
    error_message: str = "",
    raw_event_count: int = 0,
    record_count: int = 0,
    short_record_count: int = 0,
    semantic_event_count: int = 0,
    merged_occurrence_count: int = 0,
    alias_count: int = 0,
    fallback_text_count: int = 0,
    processed_text_status: str = "not_evaluable",
    processed_content_sha256: str = "",
    processed_file_size: int = 0,
    visible_legacy_text_count: int = 0,
    visible_legacy_text_match_count: int = 0,
) -> tuple[object, ...]:
    return (
        task.relative,
        task.organism,
        task.family,
        graph_id,
        content_sha256,
        file_size,
        status,
        error_label,
        error_message,
        raw_event_count,
        record_count,
        short_record_count,
        semantic_event_count,
        merged_occurrence_count,
        alias_count,
        fallback_text_count,
        # Keep the canonical database content-derived and reproducible across
        # clean builds and interrupted/resumed builds.  Wall-clock provenance
        # belongs in index_status.json, not in one row per source artifact.
        "",
        processed_text_status,
        processed_content_sha256,
        processed_file_size,
        visible_legacy_text_count,
        visible_legacy_text_match_count,
    )


def _string_values(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _string_values(item)
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _string_values(value[key])


def audit_processed_text(
    task: ScanTask,
    records: Sequence[Any],
) -> tuple[str, str, int, int, int]:
    """Match event-level legacy text against the archived paragraph view.

    ``processed`` lost event IDs and deduplicated text after layer assignment,
    so it cannot be the structural source of truth.  Exact substring matching
    still proves that each visible producer event's reconstructed legacy text
    occurs in the historical paragraph artifact.
    """

    legacy_by_producer: dict[str, str] = {}
    for record in records:
        for layer in record.layers:
            for event in layer.events:
                if event.legacy_text is None:
                    continue
                overrides = dict(event.legacy_text_overrides)
                for producer_event_id in event.producer_renderable_event_ids:
                    producer_legacy_text = overrides.get(
                        producer_event_id, event.legacy_text
                    )
                    previous = legacy_by_producer.get(producer_event_id)
                    if previous is not None and previous != producer_legacy_text:
                        raise ValueError(
                            f"producer event {producer_event_id!r} has conflicting legacy text"
                        )
                    legacy_by_producer[producer_event_id] = producer_legacy_text
    available = len(legacy_by_producer)
    if task.processed_path is None:
        return "not_requested", "", 0, available, 0
    path = Path(task.processed_path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return "missing", "", 0, available, 0
    except OSError:
        return "read_error", "", 0, available, 0
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return "invalid_json", digest, len(raw), available, 0
    if not isinstance(payload, dict):
        return "invalid_shape", digest, len(raw), available, 0
    corpus = "\n".join(_string_values(payload))
    matched = sum(text in corpus for text in legacy_by_producer.values())
    status = "complete" if matched == available else "legacy_text_mismatch"
    return status, digest, len(raw), available, matched


def scan_graph(task: ScanTask) -> ScanResult:
    try:
        raw = Path(task.path).read_bytes()
    except OSError as exc:
        return ScanResult(
            _graph_row(
                task,
                graph_id="",
                content_sha256="",
                file_size=0,
                status="invalid",
                error_label="read_error",
                error_message=_stable_error(exc),
            ),
            (),
        )
    content_sha256 = hashlib.sha256(raw).hexdigest()
    graph_id = graph_id_for_source(task.relative, raw)
    try:
        graph = json.loads(raw)
    except json.JSONDecodeError as exc:
        return ScanResult(
            _graph_row(
                task,
                graph_id=graph_id,
                content_sha256=content_sha256,
                file_size=len(raw),
                status="invalid",
                error_label="invalid_json",
                error_message=_stable_error(exc),
            ),
            (),
        )
    try:
        events, rejected = graph_events(graph)
    except (KeyError, TypeError, ValueError) as exc:
        return ScanResult(
            _graph_row(
                task,
                graph_id=graph_id,
                content_sha256=content_sha256,
                file_size=len(raw),
                status="invalid",
                error_label="invalid_structural_graph",
                error_message=_stable_error(exc),
            ),
            (),
        )
    raw_event_count = len(events) + rejected
    if rejected:
        return ScanResult(
            _graph_row(
                task,
                graph_id=graph_id,
                content_sha256=content_sha256,
                file_size=len(raw),
                status="quarantined",
                error_label="rejected_structural_events",
                error_message=f"rejected_events={rejected}",
                raw_event_count=raw_event_count,
            ),
            (),
        )
    try:
        records = build_structured_records(
            graph,
            graph_id=graph_id,
            source_graph_json=task.relative,
            parsed_events=(events, rejected),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return ScanResult(
            _graph_row(
                task,
                graph_id=graph_id,
                content_sha256=content_sha256,
                file_size=len(raw),
                status="invalid",
                error_label="view_build_error",
                error_message=_stable_error(exc),
                raw_event_count=raw_event_count,
            ),
            (),
        )

    try:
        (
            processed_text_status,
            processed_content_sha256,
            processed_file_size,
            visible_legacy_text_count,
            visible_legacy_text_match_count,
        ) = audit_processed_text(task, records)
    except ValueError as exc:
        return ScanResult(
            _graph_row(
                task,
                graph_id=graph_id,
                content_sha256=content_sha256,
                file_size=len(raw),
                status="invalid",
                error_label="legacy_text_validation_error",
                error_message=_stable_error(exc),
                raw_event_count=raw_event_count,
            ),
            (),
        )
    if task.processed_path is not None and processed_text_status != "complete":
        # The structured graph remains inventoried, but a source whose visible
        # legacy event text cannot be reconciled to the historical paragraph
        # artifact is not allowed to contribute any canonical training record.
        return ScanResult(
            _graph_row(
                task,
                graph_id=graph_id,
                content_sha256=content_sha256,
                file_size=len(raw),
                status="quarantined",
                error_label=f"processed_text_{processed_text_status}",
                error_message=(
                    "historical processed text reconciliation did not complete: "
                    f"matched={visible_legacy_text_match_count}/"
                    f"{visible_legacy_text_count}"
                ),
                raw_event_count=raw_event_count,
                processed_text_status=processed_text_status,
                processed_content_sha256=processed_content_sha256,
                processed_file_size=processed_file_size,
                visible_legacy_text_count=visible_legacy_text_count,
                visible_legacy_text_match_count=visible_legacy_text_match_count,
            ),
            (),
        )

    rows: list[tuple[object, ...]] = []
    short_record_count = 0
    semantic_event_count = 0
    merged_occurrence_count = 0
    alias_count = 0
    fallback_text_count = 0
    try:
        for record in records:
            if record.organism != task.organism or record.family != task.family:
                raise ValueError("record organism/family disagrees with source path")
            payload = record.record_object()
            if record_from_object(payload).record_object() != payload:
                raise ValueError("canonical record round-trip changed payload")
            if len(record.layers) < 2:
                short_record_count += 1
                continue
            visible_events = [
                event for layer in record.layers for event in layer.events
            ]
            record_semantic_events = len(visible_events)
            record_producer_events = sum(
                len(event.producer_event_ids) for event in visible_events
            )
            record_aliases = sum(
                len(entity["aliases"])
                for event in visible_events
                for side in (event.source, event.mediator, event.target)
                for entity in side
            )
            record_fallbacks = sum(
                event.text_source == TEXT_SOURCE_CANONICAL_FALLBACK
                for event in visible_events
            )
            semantic_event_count += record_semantic_events
            merged_occurrence_count += record_producer_events - record_semantic_events
            alias_count += record_aliases
            fallback_text_count += record_fallbacks
            serialized = compact_json(payload).encode("utf-8")
            rows.append(
                (
                    record.record_id,
                    record.graph_id,
                    record.view_id,
                    record.source_graph_json,
                    record.organism,
                    record.family,
                    record.pathway_id,
                    len(record.layers),
                    record_semantic_events,
                    record_producer_events,
                    record_aliases,
                    record_fallbacks,
                    stable_rank(record.record_id, task.seed, "canonical_record"),
                    # Return ordinary bytes across the process boundary.
                    # sqlite3.Binary creates a memoryview, which cannot be
                    # pickled by ProcessPoolExecutor on Linux.
                    zlib.compress(serialized, level=3),
                    "",
                )
            )
    except (KeyError, TypeError, ValueError) as exc:
        return ScanResult(
            _graph_row(
                task,
                graph_id=graph_id,
                content_sha256=content_sha256,
                file_size=len(raw),
                status="invalid",
                error_label="record_validation_error",
                error_message=_stable_error(exc),
                raw_event_count=raw_event_count,
            ),
            (),
        )
    status = "ok" if rows else "no_trainable_views"
    return ScanResult(
        _graph_row(
            task,
            graph_id=graph_id,
            content_sha256=content_sha256,
            file_size=len(raw),
            status=status,
            raw_event_count=raw_event_count,
            record_count=len(rows),
            short_record_count=short_record_count,
            semantic_event_count=semantic_event_count,
            merged_occurrence_count=merged_occurrence_count,
            alias_count=alias_count,
            fallback_text_count=fallback_text_count,
            processed_text_status=processed_text_status,
            processed_content_sha256=processed_content_sha256,
            processed_file_size=processed_file_size,
            visible_legacy_text_count=visible_legacy_text_count,
            visible_legacy_text_match_count=visible_legacy_text_match_count,
        ),
        tuple(rows),
    )


def scan_batch(batch: ScanBatch) -> tuple[ScanResult, ...]:
    return tuple(scan_graph(task) for task in batch.tasks)


SECONDARY_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS records_family_rank ON records(family, rank);
CREATE INDEX IF NOT EXISTS records_organism_rank ON records(organism, rank);
CREATE INDEX IF NOT EXISTS records_graph_rank ON records(graph_id, rank);
CREATE INDEX IF NOT EXISTS records_split_rank ON records(split, rank);
"""


def create_secondary_indexes(connection: sqlite3.Connection) -> None:
    """Create query indexes after bulk ingestion has finished.

    Maintaining these four B-trees while millions of rows are inserted is
    needlessly expensive, especially on a network filesystem.  The legacy
    single-database entrypoint keeps its old behaviour; the sharded builder
    defers this call until its deterministic local merge.
    """

    connection.executescript(SECONDARY_INDEX_SQL)
    connection.commit()


def initialize_database(
    path: Path,
    contract: dict[str, object],
    *,
    create_query_indexes: bool = True,
) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-524288")
    connection.execute("PRAGMA mmap_size=2147483648")
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS graphs (
            source_graph_json TEXT PRIMARY KEY,
            organism TEXT NOT NULL,
            family TEXT NOT NULL,
            graph_id TEXT NOT NULL,
            content_sha256 TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            status TEXT NOT NULL,
            error_label TEXT NOT NULL,
            error_message TEXT NOT NULL,
            raw_event_count INTEGER NOT NULL,
            record_count INTEGER NOT NULL,
            short_record_count INTEGER NOT NULL,
            semantic_event_count INTEGER NOT NULL,
            merged_occurrence_count INTEGER NOT NULL,
            alias_count INTEGER NOT NULL,
            fallback_text_count INTEGER NOT NULL,
            processed_at_utc TEXT NOT NULL,
            processed_text_status TEXT NOT NULL,
            processed_content_sha256 TEXT NOT NULL,
            processed_file_size INTEGER NOT NULL,
            visible_legacy_text_count INTEGER NOT NULL,
            visible_legacy_text_match_count INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS records (
            record_id TEXT PRIMARY KEY,
            graph_id TEXT NOT NULL,
            view_id TEXT NOT NULL,
            source_graph_json TEXT NOT NULL,
            organism TEXT NOT NULL,
            family TEXT NOT NULL,
            pathway_id TEXT NOT NULL,
            layer_count INTEGER NOT NULL,
            semantic_event_count INTEGER NOT NULL,
            producer_event_count INTEGER NOT NULL,
            alias_count INTEGER NOT NULL,
            fallback_text_count INTEGER NOT NULL,
            rank TEXT NOT NULL,
            record_zlib BLOB NOT NULL,
            split TEXT NOT NULL DEFAULT '',
            FOREIGN KEY(source_graph_json) REFERENCES graphs(source_graph_json)
        );
        """
    )
    if create_query_indexes:
        create_secondary_indexes(connection)
    existing = connection.execute(
        "SELECT value_json FROM meta WHERE key='generator_contract'"
    ).fetchone()
    serialized = compact_json(contract)
    if existing is not None and existing[0] != serialized:
        raise ValueError(
            "canonical index generator contract changed; use a new output directory "
            "instead of mixing record schemas"
        )
    connection.execute(
        "INSERT OR IGNORE INTO meta(key, value_json) VALUES('generator_contract', ?)",
        (serialized,),
    )
    connection.commit()
    return connection


def iter_batches(
    root: Path,
    *,
    processed_root: Path | None,
    already_scanned: dict[str, IndexedSource],
    seed: int,
    batch_size: int,
    max_files: int,
    inventory: Counter[str],
    inventory_digest: Any,
) -> Iterator[ScanBatch]:
    batch: list[ScanTask] = []
    for position, path in enumerate(iter_graph_files(root), start=1):
        if max_files and position > max_files:
            break
        relative, organism, family = source_identity(path, root)
        size = path.stat().st_size
        inventory["graph_files"] += 1
        inventory["graph_bytes"] += size
        inventory_digest.update(f"{relative}\t{size}\n".encode("utf-8"))
        if relative in already_scanned:
            prior = already_scanned[relative]
            if prior.graph_size != size:
                raise ValueError(
                    "source graph changed size after it was indexed; use a new "
                    f"canonical index directory: {relative} "
                    f"indexed={prior.graph_size} live={size}"
                )
            if not prior.graph_sha256 or file_sha256(path) != prior.graph_sha256:
                raise ValueError(
                    "source graph content changed after it was indexed; use a new "
                    f"canonical index directory: {relative}"
                )
            if processed_root is not None:
                counterpart = processed_root / relative
                if prior.processed_sha256:
                    if not counterpart.is_file():
                        raise ValueError(
                            "historical processed counterpart disappeared after indexing: "
                            f"{relative}"
                        )
                    live_processed_size = counterpart.stat().st_size
                    if (
                        live_processed_size != prior.processed_size
                        or file_sha256(counterpart) != prior.processed_sha256
                    ):
                        raise ValueError(
                            "historical processed counterpart changed after indexing; use "
                            f"a new canonical index directory: {relative}"
                        )
                elif prior.processed_text_status == "missing":
                    if counterpart.exists():
                        raise ValueError(
                            "historical processed counterpart appeared after indexing; use "
                            f"a new canonical index directory: {relative}"
                        )
                else:
                    raise ValueError(
                        "previously indexed source lacks a verifiable processed hash; use "
                        f"a new canonical index directory: {relative}"
                    )
            inventory["graph_files_already_indexed"] += 1
            continue
        processed_path = (
            str(processed_root / relative) if processed_root is not None else None
        )
        batch.append(
            ScanTask(str(path), processed_path, relative, organism, family, seed)
        )
        if len(batch) == batch_size:
            yield ScanBatch(tuple(batch))
            batch = []
    if batch:
        yield ScanBatch(tuple(batch))


def bounded_ordered_map(
    executor: ProcessPoolExecutor,
    batches: Iterable[ScanBatch],
    *,
    maximum_pending: int,
) -> Iterator[tuple[ScanResult, ...]]:
    """Map batches without submitting the 1.36M-file corpus all at once.

    Results are yielded in source-inventory order so SQLite insertion order,
    the canonical database hash, and every downstream build identity remain
    reproducible.  A small pending window still keeps all workers busy while
    bounding coordinator memory.
    """

    if maximum_pending < 1:
        raise ValueError("maximum_pending must be positive")
    iterator = iter(batches)
    pending: deque[Any] = deque()
    for _ in range(maximum_pending):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        pending.append(executor.submit(scan_batch, batch))
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            batch = next(iterator)
        except StopIteration:
            continue
        pending.append(executor.submit(scan_batch, batch))


def consume_results(
    results: Iterable[tuple[ScanResult, ...]],
    *,
    connection: sqlite3.Connection,
    progress_every: int,
) -> int:
    processed = 0
    indexed_graphs, canonical_records = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(record_count),0) FROM graphs"
    ).fetchone()
    graph_sql = "INSERT INTO graphs VALUES (" + ",".join("?" for _ in range(22)) + ")"
    record_sql = "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    for batch in results:
        for result in batch:
            connection.execute(graph_sql, result.graph_row)
            if result.record_rows:
                connection.executemany(record_sql, result.record_rows)
            processed += 1
            indexed_graphs += 1
            canonical_records += int(result.graph_row[9])
            if processed % 250 == 0:
                connection.commit()
            if progress_every and processed % progress_every == 0:
                print(
                    f"new_graphs={processed} indexed_graphs={indexed_graphs} "
                    f"canonical_records={canonical_records}",
                    file=sys.stderr,
                    flush=True,
                )
    connection.commit()
    return processed


def summarize_database(connection: sqlite3.Connection) -> dict[str, object]:
    status_counts = dict(
        connection.execute(
            "SELECT status, COUNT(*) FROM graphs GROUP BY status ORDER BY status"
        )
    )
    totals = connection.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(file_size),0), COALESCE(SUM(record_count),0),
               COALESCE(SUM(raw_event_count),0),
               COALESCE(SUM(semantic_event_count),0),
               COALESCE(SUM(merged_occurrence_count),0),
               COALESCE(SUM(alias_count),0), COALESCE(SUM(fallback_text_count),0)
        FROM graphs
        """
    ).fetchone()
    record_counts = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT graph_id), COUNT(DISTINCT organism), "
        "COUNT(DISTINCT family), MIN(layer_count), MAX(layer_count) FROM records"
    ).fetchone()
    processed_text_status = dict(
        connection.execute(
            "SELECT processed_text_status, COUNT(*) FROM graphs "
            "GROUP BY processed_text_status ORDER BY processed_text_status"
        )
    )
    processed_text_totals = connection.execute(
        "SELECT COALESCE(SUM(processed_file_size),0), "
        "COALESCE(SUM(visible_legacy_text_count),0), "
        "COALESCE(SUM(visible_legacy_text_match_count),0) FROM graphs"
    ).fetchone()
    return {
        "graphs": totals[0],
        "graph_bytes": totals[1],
        "records_declared_by_graphs": totals[2],
        "raw_events": totals[3],
        "semantic_events_across_views": totals[4],
        "merged_occurrences_across_views": totals[5],
        "aliases_across_views": totals[6],
        "fallback_text_events_across_views": totals[7],
        "graph_status": status_counts,
        "processed_text_status": processed_text_status,
        "processed_text_bytes": processed_text_totals[0],
        "visible_legacy_text_events": processed_text_totals[1],
        "visible_legacy_text_matches": processed_text_totals[2],
        "records": record_counts[0],
        "record_graphs": record_counts[1],
        "organisms": record_counts[2],
        "families": record_counts[3],
        "minimum_layers": record_counts[4],
        "maximum_layers": record_counts[5],
    }


def write_status(
    path: Path,
    *,
    graph_root: Path,
    processed_root: Path | None,
    database_path: Path,
    contract: dict[str, object],
    inventory: Counter[str],
    inventory_sha256: str,
    summary: dict[str, object],
    complete: bool,
) -> None:
    payload = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "do_not_edit": "Regenerate by rerunning the indexer; do not hand edit.",
        "generated_at_utc": utc_now(),
        "complete": complete,
        "processed_graph_root": str(graph_root),
        "processed_root": str(processed_root) if processed_root is not None else None,
        "database": database_path.name,
        "database_bytes": database_path.stat().st_size,
        "database_sha256": file_sha256(database_path) if complete else None,
        "inventory": {
            **dict(inventory),
            "path_size_inventory_sha256": inventory_sha256,
        },
        "summary": summary,
        "generator_contract": contract,
    }
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH if complete else 0o644)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--processed-graph-root", required=True)
    parser.add_argument(
        "--processed-root",
        help="Matching historical paragraph root for exact legacy-text auditing.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_files < 0:
        parser.error("--max-files cannot be negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    graph_root = Path(args.processed_graph_root).expanduser().resolve()
    processed_root = (
        Path(args.processed_root).expanduser().resolve()
        if args.processed_root
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not graph_root.is_dir():
        raise FileNotFoundError(f"processed_graph root does not exist: {graph_root}")
    if processed_root is not None and not processed_root.is_dir():
        raise FileNotFoundError(f"processed root does not exist: {processed_root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "canonical_index_v4.sqlite3"
    status_path = output_dir / "index_status.json"
    if status_path.exists():
        status_path.chmod(0o644)
    contract = generator_contract()
    connection = initialize_database(database_path, contract)
    try:
        input_roots = compact_json(
            {
                "processed_graph_root": str(graph_root),
                "processed_root": str(processed_root) if processed_root is not None else None,
            }
        )
        existing_roots = connection.execute(
            "SELECT value_json FROM meta WHERE key='input_roots'"
        ).fetchone()
        if existing_roots is not None and existing_roots[0] != input_roots:
            raise ValueError(
                "canonical index input roots changed; use a new output directory"
            )
        connection.execute(
            "INSERT OR IGNORE INTO meta(key, value_json) VALUES('input_roots', ?)",
            (input_roots,),
        )
        connection.commit()
        already_scanned = {
            str(row[0]): IndexedSource(
                graph_size=int(row[1]),
                graph_sha256=str(row[2]),
                processed_size=int(row[3]),
                processed_sha256=str(row[4]),
                processed_text_status=str(row[5]),
            )
            for row in connection.execute(
                "SELECT source_graph_json, file_size, content_sha256, "
                "processed_file_size, processed_content_sha256, "
                "processed_text_status FROM graphs"
            )
        }
        inventory: Counter[str] = Counter()
        digest = hashlib.sha256()
        batches = iter_batches(
            graph_root,
            processed_root=processed_root,
            already_scanned=already_scanned,
            seed=args.seed,
            batch_size=args.batch_size,
            max_files=args.max_files,
            inventory=inventory,
            inventory_digest=digest,
        )
        print(
            f"canonical_index_workers={args.workers} batch_size={args.batch_size} "
            f"resume_graphs={len(already_scanned)}",
            file=sys.stderr,
            flush=True,
        )
        if args.workers == 1:
            processed = consume_results(
                map(scan_batch, batches),
                connection=connection,
                progress_every=args.progress_every,
            )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                processed = consume_results(
                    bounded_ordered_map(
                        executor,
                        batches,
                        maximum_pending=max(args.workers * 4, 8),
                    ),
                    connection=connection,
                    progress_every=args.progress_every,
                )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        summary = summarize_database(connection)
        expected_graphs = inventory["graph_files"]
        complete = summary["graphs"] == expected_graphs
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value_json) VALUES('source_inventory', ?)",
            (
                compact_json(
                    {
                        "graph_files": expected_graphs,
                        "graph_bytes": inventory["graph_bytes"],
                        "path_size_inventory_sha256": digest.hexdigest(),
                    }
                ),
            ),
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        write_status(
            status_path,
            graph_root=graph_root,
            processed_root=processed_root,
            database_path=database_path,
            contract=contract,
            inventory=inventory,
            inventory_sha256=digest.hexdigest(),
            summary=summary,
            complete=complete,
        )
        if not complete:
            raise ValueError(
                f"canonical index incomplete: indexed={summary['graphs']} "
                f"inventory={expected_graphs}"
            )
        print(
            json.dumps(
                {"new_graphs": processed, "complete": complete, "summary": summary},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
