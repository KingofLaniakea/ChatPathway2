#!/usr/bin/env python3
"""Build a capped, prefix-only v3 dataset directly from processed_graph JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from dataprocess.prompt_profiles import (
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
    forbidden_model_metadata_markers,
)
from dataprocess.release_contract import (
    MANIFEST_NAME,
    PRIMARY_CSV_NAMES,
    PRIMARY_PROMPT_PROFILE,
    RECORD_JSONL_NAMES,
    RELEASE_SCHEMA_VERSION,
    SOURCE_GRAPH_HASHES_NAME,
)
from dataprocess.schemas import canonical_pathway_family_id
from dataprocess.source_hashes import write_source_graph_hashes
from dataprocess.structured_schema import (
    SUBSTEP_SOURCE,
    V3_CSV_FIELDNAMES,
    compact_json,
    csv_row,
    graph_events,
    graph_id_for_source,
    record_from_object,
    total_training_tokens,
)
from dataprocess.structured_views import build_structured_records


DEFAULT_TEST_ORGANISMS = "tru,xtr,dre,gga,dmk,dme,cel"
MAX_AUTO_SCAN_WORKERS = 32
DEFAULT_WORKER_BATCH_SIZE = 128
LENGTH_BINS = ("02-04", "05-08", "09-16", "17-32", "33+")
SPLITS = (
    "train",
    "validation",
    "test",
    "test_family_only",
    "test_organism_only",
)
OUTPUT_NAMES = {
    **{f"{split}_csv": name for split, name in PRIMARY_CSV_NAMES.items() if split != "train"},
    **{f"{split}_records": name for split, name in RECORD_JSONL_NAMES.items()},
    "source_graph_hashes": SOURCE_GRAPH_HASHES_NAME,
    "manifest": MANIFEST_NAME,
    "audit": "data_audit.json",
}


@dataclass(frozen=True)
class PrefixHorizon:
    prefix_len: int
    horizon: str


@dataclass
class MaterializedCandidate:
    record: Any
    rows: tuple[tuple[PrefixHorizon, dict[str, object], int], ...]


@dataclass(frozen=True)
class GraphScanTask:
    path: str
    relative: str
    organism: str
    family: str
    split: str | None
    force_candidate: bool = False


@dataclass(frozen=True)
class GraphScanConfig:
    train_candidate_fraction: float
    evaluation_candidate_fraction: float
    seen_evaluation_candidate_fraction: float
    seed: int


@dataclass(frozen=True)
class GraphScanBatch:
    tasks: tuple[GraphScanTask, ...]
    config: GraphScanConfig


@dataclass(frozen=True)
class GraphScanResult:
    relative: str
    stats: tuple[tuple[str, int], ...]
    candidates: tuple[tuple[object, ...], ...] = ()
    error_label: str = ""
    error_message: str = ""


csv.field_size_limit(sys.maxsize)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(value: str, seed: int, namespace: str) -> float:
    digest = hashlib.sha256(f"{namespace}:{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def stable_rank(value: str, seed: int, namespace: str = "coverage") -> str:
    return hashlib.sha256(f"{namespace}:{seed}:{value}".encode("utf-8")).hexdigest()


def prefix_horizons(layer_count: int, maximum: int = 3) -> tuple[PrefixHorizon, ...]:
    """Return unique long/middle/short continuation choices.

    ``prefix_len`` counts observed layers, so a small prefix is a long target.
    Very short records cannot expose three distinct horizons; those cases stay
    explicit instead of duplicating the same CSV sample under several labels.
    """

    if maximum < 1 or maximum > 3:
        raise ValueError("maximum horizon count must be in [1, 3]")
    if layer_count < 2:
        return ()
    candidates = tuple(range(1, layer_count))
    if len(candidates) == 1:
        choices = (PrefixHorizon(candidates[0], "degenerate_target"),)
    elif len(candidates) == 2:
        choices = (
            PrefixHorizon(candidates[0], "long_target"),
            PrefixHorizon(candidates[-1], "short_target"),
        )
    else:
        choices = (
            PrefixHorizon(candidates[0], "long_target"),
            PrefixHorizon(candidates[len(candidates) // 2], "middle_target"),
            PrefixHorizon(candidates[-1], "short_target"),
        )
    if len(choices) <= maximum:
        return choices
    if maximum == 1:
        middle = next(
            (choice for choice in choices if choice.horizon == "middle_target"),
            None,
        )
        return (middle or choices[len(choices) // 2],)
    if maximum == 2:
        return (choices[0], choices[-1])
    return choices[:maximum]


def assign_balanced_validation_horizons(
    eligible: Mapping[str, Sequence[PrefixHorizon]],
    *,
    seed: int,
) -> dict[str, PrefixHorizon]:
    """Choose one fixed validation horizon per record with global balance.

    Records are visited in a stable hashed order.  At each record the least
    represented available horizon wins; a per-record stable tie-break prevents
    input-order dependence.  When every record has all three horizons the
    resulting counts differ by at most one.
    """

    counts: Counter[str] = Counter()
    assignments: dict[str, PrefixHorizon] = {}
    ordered_ids = sorted(
        eligible,
        key=lambda record_id: (
            stable_rank(record_id, seed, "validation_record"),
            record_id,
        ),
    )
    for record_id in ordered_ids:
        choices = tuple(eligible[record_id])
        if not choices:
            raise ValueError(f"validation record {record_id!r} has no eligible horizon")
        tie_order = sorted(
            choices,
            key=lambda choice: (
                counts[choice.horizon],
                stable_rank(
                    f"{record_id}:{choice.horizon}",
                    seed,
                    "validation_horizon",
                ),
                choice.prefix_len,
            ),
        )
        selected = tie_order[0]
        assignments[record_id] = selected
        counts[selected.horizon] += 1
    return assignments


def parse_organisms(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def iter_graph_files(root: Path) -> Iterator[Path]:
    for directory, directory_names, file_names in os.walk(root):
        directory_names.sort()
        for file_name in sorted(file_names):
            if file_name.endswith(".json"):
                yield Path(directory) / file_name


def source_identity(path: Path, root: Path) -> tuple[str, str, str]:
    relative = path.relative_to(root).as_posix()
    parts = relative.split("/")
    organism = parts[0] if len(parts) > 1 else ""
    pathway_id = path.stem
    family = canonical_pathway_family_id(pathway_id)
    return relative, organism, family


def choose_family_fraction(
    families: set[str],
    *,
    fraction: float,
    seed: int,
    namespace: str,
) -> set[str]:
    if not families:
        raise ValueError(f"no families are available for {namespace}")
    if not 0 < fraction < 1:
        raise ValueError(f"{namespace} fraction must be in (0, 1)")
    count = min(max(1, round(len(families) * fraction)), max(1, len(families) - 1))
    ranked = sorted(
        families,
        key=lambda family: (
            stable_fraction(family, seed, namespace),
            family,
        ),
    )
    return set(ranked[:count])


def choose_family_splits(
    *,
    test_available_families: set[str],
    non_test_available_families: set[str],
    test_fraction: float,
    validation_fraction: float,
    seed: int,
) -> tuple[set[str], set[str], set[str]]:
    """Choose strict-test, validation, and train family sets.

    Strict families must occur in both organism domains so the dual holdout and
    family-only diagnostic are defined on the same family identities.
    """

    strict_candidates = test_available_families & non_test_available_families
    strict_families = choose_family_fraction(
        strict_candidates,
        fraction=test_fraction,
        seed=seed,
        namespace="strict_test_family",
    )
    validation_candidates = non_test_available_families - strict_families
    validation_families = choose_family_fraction(
        validation_candidates,
        fraction=validation_fraction,
        seed=seed,
        namespace="validation_family",
    )
    train_families = (
        non_test_available_families - strict_families - validation_families
    )
    if not train_families:
        raise ValueError(
            "strict family selection left no train families; lower the test or "
            "validation family fraction"
        )
    return strict_families, validation_families, train_families


def discover_inventory(
    root: Path,
    test_organisms: set[str],
) -> tuple[set[str], set[str], set[str], dict[str, object]]:
    all_families: set[str] = set()
    test_organism_families: set[str] = set()
    non_test_organism_families: set[str] = set()
    organisms: set[str] = set()
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    for path in iter_graph_files(root):
        relative, organism, family = source_identity(path, root)
        size = path.stat().st_size
        digest.update(f"{relative}\t{size}\n".encode("utf-8"))
        count += 1
        total_bytes += size
        organisms.add(organism)
        all_families.add(family)
        if organism in test_organisms:
            test_organism_families.add(family)
        else:
            non_test_organism_families.add(family)
    if not count:
        raise ValueError(f"processed_graph root has no JSON files: {root}")
    return all_families, test_organism_families, non_test_organism_families, {
        "graph_files": count,
        "graph_bytes": total_bytes,
        "organisms": len(organisms),
        "families": len(all_families),
        "path_size_inventory_sha256": digest.hexdigest(),
    }


def assigned_split(
    organism: str,
    family: str,
    *,
    test_organisms: set[str],
    test_families: set[str],
    validation_families: set[str],
    train_families: set[str],
) -> str | None:
    """Assign one source graph to the five-way biological split contract."""

    if organism in test_organisms:
        if family in test_families:
            return "test"
        if family in train_families:
            return "test_organism_only"
        return None
    if family in test_families:
        return "test_family_only"
    if family in validation_families:
        return "validation"
    if family in train_families:
        return "train"
    return None


def initialize_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute(
        """
        CREATE TABLE candidates (
            record_id TEXT PRIMARY KEY,
            split TEXT NOT NULL,
            family TEXT NOT NULL,
            organism TEXT NOT NULL,
            graph_id TEXT NOT NULL,
            source_graph_json TEXT NOT NULL,
            layer_count INTEGER NOT NULL,
            rank TEXT NOT NULL,
            record_json TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX candidates_split_family ON candidates(split, family)")
    connection.execute(
        "CREATE INDEX candidates_split_family_rank "
        "ON candidates(split, family, rank)"
    )
    return connection


def resolve_scan_workers(requested: int) -> int:
    """Resolve ``0`` to a conservative automatic CPU count."""

    if requested < 0:
        raise ValueError("workers must be zero (automatic) or positive")
    if requested:
        return requested
    return max(1, min(os.cpu_count() or 1, MAX_AUTO_SCAN_WORKERS))


def _stable_scan_error(exc: BaseException) -> str:
    """Render worker errors without embedding an absolute source path."""

    if isinstance(exc, OSError):
        detail = exc.strerror or "I/O error"
    else:
        detail = str(exc)
    return f"{type(exc).__name__}: {detail}"


def _scan_graph_task(task: GraphScanTask, config: GraphScanConfig) -> GraphScanResult:
    """Read and transform one graph without touching SQLite or stderr."""

    stats: Counter[str] = Counter(graph_files_scanned=1)
    if task.split is None:
        stats["graph_files_excluded_by_split"] += 1
        return GraphScanResult(task.relative, tuple(sorted(stats.items())))

    try:
        raw = Path(task.path).read_bytes()
        graph = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        stats["invalid_graph_files"] += 1
        return GraphScanResult(
            task.relative,
            tuple(sorted(stats.items())),
            error_label="invalid_graph",
            error_message=_stable_scan_error(exc),
        )

    graph_id = graph_id_for_source(task.relative, raw)
    try:
        structural_events, missing_endpoints = graph_events(graph)
    except (KeyError, TypeError, ValueError) as exc:
        stats["invalid_structural_graph_files"] += 1
        return GraphScanResult(
            task.relative,
            tuple(sorted(stats.items())),
            error_label="invalid_structural_graph",
            error_message=_stable_scan_error(exc),
        )

    stats["graph_events_total"] += len(structural_events) + missing_endpoints
    stats["graph_missing_endpoint_events"] += missing_endpoints
    if missing_endpoints:
        stats["graphs_excluded_missing_endpoint_events"] += 1
        return GraphScanResult(task.relative, tuple(sorted(stats.items())))

    if task.split == "train":
        fraction = config.train_candidate_fraction
    elif task.split in {"validation", "test_family_only"}:
        fraction = config.seen_evaluation_candidate_fraction
    else:
        fraction = config.evaluation_candidate_fraction
    if task.force_candidate:
        stats["graphs_forced_coverage_train"] += 1
    # This single content-bound decision applies to every sink view emitted by
    # the graph and therefore cannot favour multi-sink graphs.
    if (
        not task.force_candidate
        and
        stable_fraction(
            graph_id,
            config.seed,
            f"candidate_graph:{task.split}",
        )
        >= fraction
    ):
        stats[f"graphs_fraction_excluded_{task.split}"] += 1
        return GraphScanResult(task.relative, tuple(sorted(stats.items())))

    try:
        records = build_structured_records(
            graph,
            graph_id=graph_id,
            source_graph_json=task.relative,
            parsed_events=(structural_events, missing_endpoints),
        )
    except (KeyError, TypeError, ValueError) as exc:
        stats["invalid_structural_graph_files"] += 1
        return GraphScanResult(
            task.relative,
            tuple(sorted(stats.items())),
            error_label="invalid_structural_graph",
            error_message=_stable_scan_error(exc),
        )

    stats["graphs_parsed"] += 1
    stats["views_built"] += len(records)
    if not structural_events:
        stats["graphs_without_structural_events"] += 1
    try:
        record_payloads = [
            (record, record.record_object())
            for record in records
        ]
        for _record, payload in record_payloads:
            rebuilt = record_from_object(payload)
            if rebuilt.record_object() != payload:
                raise ValueError("structured record round-trip changed canonical payload")
    except (KeyError, TypeError, ValueError) as exc:
        stats["graphs_failed_record_roundtrip"] += 1
        return GraphScanResult(
            task.relative,
            tuple(sorted(stats.items())),
            error_label="invalid_record_roundtrip",
            error_message=_stable_scan_error(exc),
        )
    candidates: list[tuple[object, ...]] = []
    for record, record_payload in record_payloads:
        if record.organism != task.organism or record.family != task.family:
            stats["views_excluded_source_identity_mismatch"] += 1
            continue
        if len(record.layers) < 2:
            stats["views_skipped_short"] += 1
            continue
        candidates.append(
            (
                record.record_id,
                task.split,
                record.family,
                record.organism,
                record.graph_id,
                record.source_graph_json,
                len(record.layers),
                stable_rank(record.record_id, config.seed),
                compact_json(record_payload),
            )
        )
        stats["candidate_records"] += 1
        stats[f"candidate_records_{task.split}"] += 1
    return GraphScanResult(
        task.relative,
        tuple(sorted(stats.items())),
        tuple(candidates),
    )


def _scan_graph_batch(batch: GraphScanBatch) -> tuple[GraphScanResult, ...]:
    """Process a bounded batch; one executor Future never means one file."""

    return tuple(_scan_graph_task(task, batch.config) for task in batch.tasks)


def _iter_graph_scan_batches(
    graph_root: Path,
    *,
    test_organisms: set[str],
    test_families: set[str],
    validation_families: set[str],
    train_families: set[str],
    config: GraphScanConfig,
    max_files: int,
    worker_batch_size: int,
    coverage_graphs_per_train_organism: int,
) -> Iterator[GraphScanBatch]:
    batch: list[GraphScanTask] = []
    train_graphs_seen_by_organism: Counter[str] = Counter()
    for file_index, path in enumerate(iter_graph_files(graph_root), start=1):
        if max_files and file_index > max_files:
            break
        relative, organism, family = source_identity(path, graph_root)
        split = assigned_split(
            organism,
            family,
            test_organisms=test_organisms,
            test_families=test_families,
            validation_families=validation_families,
            train_families=train_families,
        )
        force_candidate = False
        if split == "train":
            train_graphs_seen_by_organism[organism] += 1
            force_candidate = (
                train_graphs_seen_by_organism[organism]
                <= coverage_graphs_per_train_organism
            )
        batch.append(
            GraphScanTask(
                path=str(path),
                relative=relative,
                organism=organism,
                family=family,
                split=split,
                force_candidate=force_candidate,
            )
        )
        if len(batch) >= worker_batch_size:
            yield GraphScanBatch(tuple(batch), config)
            batch = []
    if batch:
        yield GraphScanBatch(tuple(batch), config)


def _consume_graph_scan_batches(
    batch_results: Iterable[tuple[GraphScanResult, ...]],
    *,
    connection: sqlite3.Connection,
    progress_every: int,
) -> dict[str, int]:
    """Aggregate ordered worker results and remain the sole SQLite writer."""

    stats: Counter[str] = Counter()
    processed_files = 0
    for batch in batch_results:
        for result in batch:
            processed_files += 1
            stats.update(dict(result.stats))
            if result.error_label:
                # executor.map preserves the sorted input order, so failures
                # are always reported against relative paths in the same order.
                print(
                    f"{result.error_label}={result.relative} "
                    f"error={result.error_message}",
                    file=sys.stderr,
                )
            if result.candidates:
                connection.executemany(
                    "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    result.candidates,
                )
            if progress_every and processed_files % progress_every == 0:
                print(
                    f"scanned_graph_files={processed_files} "
                    f"candidate_records={stats['candidate_records']}",
                    file=sys.stderr,
                    flush=True,
                )
            if processed_files % 1000 == 0:
                connection.commit()
    connection.commit()
    return dict(stats)


def scan_candidates(
    graph_root: Path,
    connection: sqlite3.Connection,
    *,
    test_organisms: set[str],
    test_families: set[str],
    validation_families: set[str],
    train_families: set[str],
    train_candidate_fraction: float,
    evaluation_candidate_fraction: float,
    seen_evaluation_candidate_fraction: float = 0.02,
    seed: int,
    max_files: int,
    progress_every: int,
    workers: int = 1,
    worker_batch_size: int = DEFAULT_WORKER_BATCH_SIZE,
    coverage_graphs_per_train_organism: int = 5,
) -> dict[str, int]:
    """Build candidates serially or with deterministic ordered worker batches."""

    resolved_workers = resolve_scan_workers(workers)
    if worker_batch_size < 1:
        raise ValueError("worker_batch_size must be positive")
    if coverage_graphs_per_train_organism < 0:
        raise ValueError("coverage_graphs_per_train_organism cannot be negative")
    config = GraphScanConfig(
        train_candidate_fraction=train_candidate_fraction,
        evaluation_candidate_fraction=evaluation_candidate_fraction,
        seen_evaluation_candidate_fraction=seen_evaluation_candidate_fraction,
        seed=seed,
    )
    batches = _iter_graph_scan_batches(
        graph_root,
        test_organisms=test_organisms,
        test_families=test_families,
        validation_families=validation_families,
        train_families=train_families,
        config=config,
        max_files=max_files,
        worker_batch_size=worker_batch_size,
        coverage_graphs_per_train_organism=coverage_graphs_per_train_organism,
    )
    print(
        f"graph_scan_workers={resolved_workers} "
        f"worker_batch_size={worker_batch_size}",
        file=sys.stderr,
        flush=True,
    )
    if resolved_workers == 1:
        return _consume_graph_scan_batches(
            map(_scan_graph_batch, batches),
            connection=connection,
            progress_every=progress_every,
        )
    # A Future covers ``worker_batch_size`` files.  Ordered ``map`` makes
    # SQLite inserts, diagnostics, max-files handling, and statistics invariant
    # to worker scheduling while avoiding one Future per corpus file.
    with ProcessPoolExecutor(max_workers=resolved_workers) as executor:
        batch_results = executor.map(_scan_graph_batch, batches, chunksize=1)
        return _consume_graph_scan_batches(
            batch_results,
            connection=connection,
            progress_every=progress_every,
        )


def length_bin(layer_count: int) -> str:
    if layer_count <= 4:
        return "02-04"
    if layer_count <= 8:
        return "05-08"
    if layer_count <= 16:
        return "09-16"
    if layer_count <= 32:
        return "17-32"
    return "33+"


def _round_robin_length(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[length_bin(int(row["layer_count"]))].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda row: (row["rank"], row["record_id"]))
    ordered: list[dict[str, Any]] = []
    position = 0
    while True:
        added = False
        for name in LENGTH_BINS:
            bucket = buckets.get(name, ())
            if position < len(bucket):
                ordered.append(bucket[position])
                added = True
        if not added:
            break
        position += 1
    return ordered


def diversity_order(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_organism: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_organism[row["organism"]].append(row)
    primary: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for organism in sorted(by_organism):
        organism_rows = by_organism[organism]
        ordered = sorted(
            organism_rows,
            key=lambda row: (row["rank"], row["record_id"]),
        )
        primary.append(ordered[0])
        remaining.extend(ordered[1:])
    return _round_robin_length(primary) + _round_robin_length(remaining)


def graph_round_robin_order(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order all first sink views before any graph contributes a second view."""

    by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        graph_id = str(row.get("graph_id", "")).strip()
        if not graph_id:
            raise ValueError("candidate row lacks graph_id")
        by_graph[graph_id].append(row)
    ordered_by_graph = {
        graph_id: sorted(
            graph_rows,
            key=lambda row: (row["rank"], row["record_id"]),
        )
        for graph_id, graph_rows in by_graph.items()
    }
    output: list[dict[str, Any]] = []
    view_index = 0
    while True:
        one_view_per_graph = [
            ordered_by_graph[graph_id][view_index]
            for graph_id in sorted(ordered_by_graph)
            if view_index < len(ordered_by_graph[graph_id])
        ]
        if not one_view_per_graph:
            return output
        output.extend(diversity_order(one_view_per_graph))
        view_index += 1


def organism_round_robin_order(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Offer one record per organism before any organism offers a second."""

    by_organism: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        organism = str(row.get("organism") or "").strip()
        if not organism:
            raise ValueError("candidate row lacks organism")
        by_organism[organism].append(row)
    queues = {
        organism: graph_round_robin_order(organism_rows)
        for organism, organism_rows in by_organism.items()
    }
    output: list[dict[str, Any]] = []
    position = 0
    while True:
        round_rows = [
            queues[organism][position]
            for organism in sorted(queues)
            if position < len(queues[organism])
        ]
        if not round_rows:
            return output
        output.extend(_round_robin_length(round_rows))
        position += 1


def select_records(
    connection: sqlite3.Connection,
    split: str,
) -> list[dict[str, Any]]:
    """Return every candidate in deterministic family/graph-diverse order.

    The family cap is deliberately *not* applied here.  Materialization first
    rejects over-budget records and then consumes later candidates until the
    accepted-record cap is reached.
    """

    families = [
        row[0]
        for row in connection.execute(
            "SELECT DISTINCT family FROM candidates WHERE split=? ORDER BY family",
            (split,),
        )
    ]
    selected: list[dict[str, Any]] = []
    for family in families:
        rows = [
            {
                "record_id": record_id,
                "family": family,
                "organism": organism,
                "graph_id": graph_id,
                "source_graph_json": source_graph_json,
                "layer_count": layer_count,
                "rank": rank,
                "record_json": record_json,
            }
            for (
                record_id,
                organism,
                graph_id,
                source_graph_json,
                layer_count,
                rank,
                record_json,
            ) in connection.execute(
                """
                SELECT record_id, organism, graph_id, source_graph_json,
                       layer_count, rank, record_json
                FROM candidates WHERE split=? AND family=? ORDER BY rank
                """,
                (split, family),
            )
        ]
        selected.extend(graph_round_robin_order(rows))
    return organism_round_robin_order(selected)


def output_paths(output_dir: Path, maximum_per_family: int) -> dict[str, Path]:
    names = {
        **OUTPUT_NAMES,
        "train_csv": f"train_pathway_continuation_v3_cap{maximum_per_family}.csv",
    }
    return {key: output_dir / name for key, name in names.items()}


def validate_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    for path in paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)


def control_output_paths(
    output_dir: Path,
    maximum_per_family: int,
) -> dict[str, dict[str, Path]]:
    """Return CSV-only paired-control paths for P1 and strict-natural P2."""

    output: dict[str, dict[str, Path]] = {}
    for profile in (
        NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
        SPECIES_NEUTRAL_IDS_NO_ORGANISM,
    ):
        profile_dir = output_dir / "prompt_controls" / profile
        output[profile] = {
            split: profile_dir
            / (
                f"train_pathway_continuation_v3_cap{maximum_per_family}.csv"
                if split == "train"
                else PRIMARY_CSV_NAMES[split]
            )
            for split in SPLITS
        }
    return output


def validate_control_outputs(
    paths: Mapping[str, Mapping[str, Path]],
    overwrite: bool,
) -> None:
    for profile_paths in paths.values():
        for path in profile_paths.values():
            if path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
            path.parent.mkdir(parents=True, exist_ok=True)


def _csv_row_with_policy(
    record: Any,
    prefix: PrefixHorizon,
    *,
    split: str,
    prompt_profile: str = PRIMARY_PROMPT_PROFILE,
) -> tuple[dict[str, object], bool, bool]:
    """Call the evolving structured-row interface through one narrow seam.

    TODO(dataset prompt integration): ``structured_schema.csv_row`` is being
    extended independently with ``prompt_profile`` and ``prefix_horizon``.
    Passing those parameters here as soon as they appear avoids freezing this
    builder to the legacy implicit-no-organism prompt during the parallel edit.
    """

    parameters = inspect.signature(csv_row).parameters
    kwargs: dict[str, object] = {}
    profile_applied = False
    horizon_applied = False
    for name in ("prompt_profile", "profile"):
        if name in parameters:
            kwargs[name] = prompt_profile
            profile_applied = True
            break
    for name in ("prefix_horizon", "horizon"):
        if name in parameters:
            kwargs[name] = prefix.horizon
            horizon_applied = True
            break
    if "split" in parameters:
        kwargs["split"] = split
    row = dict(csv_row(record, prefix.prefix_len, **kwargs))
    # Once the maintained structured CSV header grows these provenance fields,
    # keep them explicit even if csv_row delegates only prompt rendering.
    if "prefix_horizon" in V3_CSV_FIELDNAMES:
        row["prefix_horizon"] = prefix.horizon
        horizon_applied = True
    if "prompt_profile" in V3_CSV_FIELDNAMES:
        row["prompt_profile"] = prompt_profile
        profile_applied = True
    return row, profile_applied, horizon_applied


def _selected_rows_for_split(
    packages: Sequence[MaterializedCandidate],
    *,
    split: str,
    seed: int,
) -> dict[str, tuple[tuple[PrefixHorizon, dict[str, object], int], ...]]:
    if split != "validation":
        return {
            package.record.record_id: package.rows
            for package in packages
        }
    assignments = assign_balanced_validation_horizons(
        {
            package.record.record_id: tuple(item[0] for item in package.rows)
            for package in packages
        },
        seed=seed,
    )
    selected: dict[str, tuple[tuple[PrefixHorizon, dict[str, object], int], ...]] = {}
    for package in packages:
        choice = assignments[package.record.record_id]
        selected[package.record.record_id] = tuple(
            item for item in package.rows if item[0] == choice
        )
    return selected


def write_selected_split(
    selected: Sequence[dict[str, Any]],
    *,
    split: str,
    csv_path: Path,
    record_path: Path,
    tokenizer: Any,
    max_length: int,
    max_prefixes_per_train_record: int,
    max_records_per_family: int,
    seed: int,
    maximum_records: int = 0,
    target_input_tokens_per_epoch: int = 0,
    progress_every: int = 0,
) -> dict[str, Any]:
    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    records_temporary = record_path.with_suffix(record_path.suffix + ".tmp")
    stats: Counter[str] = Counter()
    token_lengths: list[int] = []
    accepted_families: set[str] = set()
    accepted_organisms: set[str] = set()
    accepted_sources: set[str] = set()
    accepted_graphs: Counter[str] = Counter()
    accepted_per_family: Counter[str] = Counter()
    horizon_counts: Counter[str] = Counter()
    packages: list[MaterializedCandidate] = []
    estimated_epoch_input_tokens = 0.0
    profile_support = True
    horizon_support = True

    # Candidates arrive in family order and graph-round-robin order.  A record
    # consumes a family slot only after at least one complete row passes the
    # token budget, so an over-budget record is backfilled by the next view.
    for candidate_index, selected_row in enumerate(selected, start=1):
        if progress_every and candidate_index % progress_every == 0:
            print(
                f"materializing_split={split} "
                f"candidate_records_considered={candidate_index} "
                f"accepted_records={len(packages)}",
                file=sys.stderr,
                flush=True,
            )
        if maximum_records and len(packages) >= maximum_records:
            stats["candidate_records_skipped_global_cap"] += 1
            continue
        value = json.loads(selected_row["record_json"])
        record = record_from_object(value)
        if accepted_per_family[record.family] >= max_records_per_family:
            stats["candidate_records_skipped_family_cap"] += 1
            continue
        maximum = max_prefixes_per_train_record if split == "train" else 3
        accepted_rows: list[tuple[PrefixHorizon, dict[str, object], int]] = []
        for prefix in prefix_horizons(len(record.layers), maximum):
            row, row_profile_support, row_horizon_support = _csv_row_with_policy(
                record,
                prefix,
                split=split,
            )
            profile_support = profile_support and row_profile_support
            horizon_support = horizon_support and row_horizon_support
            # Explicit organism conditioning is allowed.  Pathway identity and
            # other answer-revealing provenance remain forbidden model input.
            question = str(row["question"])
            leaked_markers = forbidden_model_metadata_markers(question)
            if leaked_markers:
                raise ValueError(
                    f"model-visible metadata leaked for {row['sample_id']}: "
                    + ", ".join(leaked_markers)
                )
            json.loads(str(row["answer"]))
            token_count = total_training_tokens(tokenizer, row)
            if token_count > max_length:
                stats["rows_dropped_token_budget"] += 1
                continue
            accepted_rows.append((prefix, row, token_count))
        if not accepted_rows:
            stats["records_dropped_no_complete_json_sample"] += 1
            continue
        record_epoch_tokens = sum(item[2] for item in accepted_rows) / len(
            accepted_rows
        )
        if (
            target_input_tokens_per_epoch
            and estimated_epoch_input_tokens + record_epoch_tokens
            > target_input_tokens_per_epoch
        ):
            stats["candidate_records_skipped_epoch_token_budget"] += 1
            continue
        accepted_per_family[record.family] += 1
        estimated_epoch_input_tokens += record_epoch_tokens
        accepted_graphs[record.graph_id] += 1
        packages.append(MaterializedCandidate(record, tuple(accepted_rows)))

    rows_by_record = _selected_rows_for_split(packages, split=split, seed=seed)
    with csv_temporary.open("w", encoding="utf-8", newline="") as csv_handle, records_temporary.open(
        "w", encoding="utf-8"
    ) as record_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=V3_CSV_FIELDNAMES)
        writer.writeheader()
        for package in packages:
            record = package.record
            accepted_rows = rows_by_record[record.record_id]
            record_handle.write(compact_json(record.record_object()) + "\n")
            stats["records"] += 1
            accepted_families.add(record.family)
            accepted_organisms.add(record.organism)
            accepted_sources.add(record.source_graph_json)
            for prefix, row, token_count in accepted_rows:
                writer.writerow(row)
                stats["rows"] += 1
                token_lengths.append(token_count)
                horizon_counts[prefix.horizon] += 1
    if not stats["rows"]:
        csv_temporary.unlink(missing_ok=True)
        records_temporary.unlink(missing_ok=True)
        raise ValueError(f"no {split} samples survived strict token materialization")
    csv_temporary.replace(csv_path)
    records_temporary.replace(record_path)
    return {
        **dict(stats),
        "sources": len(accepted_sources),
        "families": len(accepted_families),
        "organisms": len(accepted_organisms),
        "graphs": len(accepted_graphs),
        "maximum_views_per_graph": max(accepted_graphs.values(), default=0),
        "maximum_records_in_one_family": max(accepted_per_family.values(), default=0),
        "prefix_horizons": dict(sorted(horizon_counts.items())),
        "prompt_profile": PRIMARY_PROMPT_PROFILE,
        "prompt_profile_interface_applied": profile_support,
        "prefix_horizon_interface_applied": horizon_support,
        "token_length": {
            "min": min(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
            "max": max(token_lengths),
        },
        "estimated_input_tokens_per_epoch": round(
            estimated_epoch_input_tokens
        ),
        "csv_sha256": file_sha256(csv_path),
        "records_sha256": file_sha256(record_path),
    }


def write_profile_control_csv(
    *,
    primary_csv_path: Path,
    record_path: Path,
    output_path: Path,
    prompt_profile: str,
    split: str,
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    """Materialize one paired prompt condition from accepted primary records.

    P1 must contain exactly the P0 ``base_sample_id`` set.  P2 is a documented
    strict-natural subset: the complete observed+target pair must already use
    species-neutral identifiers and names, otherwise the base sample is
    excluded rather than rewritten by prefix stripping.
    """

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    stats: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    record_ids: set[str] = set()
    base_sample_ids: set[str] = set()
    token_lengths: list[int] = []
    with primary_csv_path.open("r", encoding="utf-8", newline="") as primary_handle, record_path.open(
        "r", encoding="utf-8"
    ) as record_handle, temporary.open("w", encoding="utf-8", newline="") as output_handle:
        reader = csv.DictReader(primary_handle)
        if reader.fieldnames != V3_CSV_FIELDNAMES:
            raise ValueError(
                f"primary v3 header mismatch for {primary_csv_path}: {reader.fieldnames}"
            )
        writer = csv.DictWriter(output_handle, fieldnames=V3_CSV_FIELDNAMES)
        writer.writeheader()
        row_iterator = iter(reader)
        current = next(row_iterator, None)
        for line_number, line in enumerate(record_handle, 1):
            if not line.strip():
                continue
            record = record_from_object(json.loads(line))
            if current is not None and current.get("record_id") != record.record_id:
                raise ValueError(
                    f"primary CSV/record JSONL order mismatch at record line {line_number}: "
                    f"csv={current.get('record_id')} records={record.record_id}"
                )
            while current is not None and current.get("record_id") == record.record_id:
                stats["primary_rows_considered"] += 1
                try:
                    prefix_len = int(current["prefix_step_count"])
                    candidate = csv_row(
                        record,
                        prefix_len,
                        prompt_profile=prompt_profile,
                        prefix_horizon=str(current["prefix_horizon"]),
                        split=split,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    if prompt_profile != SPECIES_NEUTRAL_IDS_NO_ORGANISM:
                        raise
                    reason_text = str(exc).split(": ", 1)[-1]
                    for item in reason_text.split(","):
                        reason = item.split("=", 1)[0].strip() or "projection_ineligible"
                        rejection_reasons[reason] += 1
                    stats["rows_excluded_profile_ineligible"] += 1
                    current = next(row_iterator, None)
                    continue
                if candidate["base_sample_id"] != current["base_sample_id"]:
                    raise ValueError("paired prompt profile changed base_sample_id")
                if candidate["answer"] != current["answer"]:
                    raise ValueError("paired prompt profile changed the supervised answer")
                token_count = total_training_tokens(tokenizer, candidate)
                if token_count > max_length:
                    if prompt_profile != SPECIES_NEUTRAL_IDS_NO_ORGANISM:
                        raise ValueError(
                            f"P1 paired row exceeds P0 token budget: {candidate['base_sample_id']}"
                        )
                    stats["rows_excluded_token_budget"] += 1
                    current = next(row_iterator, None)
                    continue
                base_sample_id = str(candidate["base_sample_id"])
                if base_sample_id in base_sample_ids:
                    raise ValueError(f"duplicate paired base_sample_id={base_sample_id}")
                base_sample_ids.add(base_sample_id)
                record_ids.add(record.record_id)
                token_lengths.append(token_count)
                writer.writerow(candidate)
                stats["rows"] += 1
                current = next(row_iterator, None)
        if current is not None:
            raise ValueError(
                f"primary CSV contains record_id absent from {record_path}: {current.get('record_id')}"
            )
    temporary.replace(output_path)
    output_sha256 = file_sha256(output_path)
    return {
        **dict(stats),
        "records": len(record_ids),
        "prompt_profile": prompt_profile,
        "base_sample_id_sha256": hashlib.sha256(
            "\n".join(sorted(base_sample_ids)).encode("utf-8")
        ).hexdigest(),
        "profile_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "token_length": {
            "min": min(token_lengths, default=0),
            "mean": (sum(token_lengths) / len(token_lengths)) if token_lengths else 0,
            "max": max(token_lengths, default=0),
        },
        "sha256": output_sha256,
        "csv_sha256": output_sha256,
    }


def referenced_sources(record_paths: Iterable[Path]) -> set[str]:
    sources: set[str] = set()
    for path in record_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                source = str(value.get("source_graph_json") or "").strip()
                if not source:
                    raise ValueError(f"record in {path} lacks source_graph_json")
                sources.add(source)
    return sources


def record_partition_identities(path: Path) -> dict[str, set[str]]:
    identities = {
        "families": set(),
        "organisms": set(),
        "sources": set(),
        "graphs": set(),
        "views": set(),
        "records": set(),
    }
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            identities["families"].add(str(value["pathway_family_id"]))
            identities["organisms"].add(str(value["organism"]))
            identities["sources"].add(str(value["source_graph_json"]))
            identities["graphs"].add(str(value["graph_id"]))
            identities["views"].add(str(value["view_id"]))
            identities["records"].add(str(value["record_id"]))
    return identities


def processed_artifact_coverage(
    processed_root: Path | None,
    sources: Iterable[str],
) -> dict[str, Any]:
    if processed_root is None:
        return {
            "status": "not_requested",
            "note": "processed text is not a structural truth source",
        }
    missing = sorted(
        source for source in sources if not (processed_root / source).is_file()
    )
    return {
        "status": "complete" if not missing else "incomplete",
        "root": str(processed_root),
        "referenced_sources": len(set(sources)),
        "missing_counterparts": len(missing),
        "missing_examples": missing[:20],
        "role": "historical text/path reconciliation only; never relation truth",
    }


def _generate_release_audit_compat(
    *,
    paths: Mapping[str, Path],
    graph_root: Path,
    tokenizer: Any,
    max_length: int,
    overwrite: bool,
) -> None:
    """Call the current or forthcoming five-partition audit interface."""

    from dataprocess.audit_dataset_release import generate_release_audit

    parameters = inspect.signature(generate_release_audit).parameters
    base_kwargs: dict[str, Any] = {
        "train_path": paths["train_csv"],
        "validation_path": paths["validation_csv"],
        "test_path": paths["test_csv"],
        "graph_root": graph_root,
        "manifest_path": paths["manifest"],
        "tokenizer": tokenizer,
        "max_length": max_length,
        "output_path": paths["audit"],
        "overwrite": overwrite,
    }
    kwargs = {
        name: value
        for name, value in base_kwargs.items()
        if name in parameters
    }
    optional_paths = {
        "test_family_only_path": paths["test_family_only_csv"],
        "test_organism_only_path": paths["test_organism_only_csv"],
    }
    kwargs.update(
        {
            name: value
            for name, value in optional_paths.items()
            if name in parameters
        }
    )
    split_paths = {
        split: paths[f"{split}_csv"]
        for split in SPLITS
    }
    if "split_paths" in parameters:
        kwargs["split_paths"] = split_paths
    if "partition_paths" in parameters:
        kwargs["partition_paths"] = split_paths
    generate_release_audit(**kwargs)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed-graph-root", required=True)
    parser.add_argument(
        "--processed-root",
        help=(
            "Optional matching processed-text root used only to audit source-path "
            "coverage; relation and reaction truth always comes from processed_graph."
        ),
    )
    parser.add_argument(
        "--coverage-graphs-per-train-organism",
        type=int,
        default=5,
        help=(
            "Always inspect this many deterministic train-assigned graphs per "
            "organism before applying the global train candidate fraction."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--test-organisms", default=DEFAULT_TEST_ORGANISMS)
    parser.add_argument("--test-family-fraction", type=float, default=0.05)
    parser.add_argument("--validation-family-fraction", type=float, default=0.05)
    parser.add_argument("--train-candidate-record-fraction", type=float, default=0.02)
    parser.add_argument("--evaluation-candidate-record-fraction", type=float, default=1.0)
    parser.add_argument(
        "--seen-evaluation-candidate-record-fraction",
        type=float,
        default=0.02,
        help=(
            "Graph-level fraction for validation and family-only evaluation across "
            "the >10k seen-organism directories; held-out-organism tests use the "
            "separate evaluation fraction."
        ),
    )
    parser.add_argument("--max-records-per-family", type=int, default=256)
    parser.add_argument(
        "--maximum-train-records",
        type=int,
        default=18000,
        help="Secondary hard cap after organism-first ordering and token filtering.",
    )
    parser.add_argument(
        "--target-train-input-tokens-per-epoch",
        type=int,
        default=36000000,
        help=(
            "Approximate one-prefix-per-record input-token budget per SFT epoch; "
            "records are backfilled until this budget or the record cap is reached."
        ),
    )
    parser.add_argument("--max-prefixes-per-train-record", type=int, default=3)
    parser.add_argument(
        "--minimum-train-records",
        type=int,
        default=12000,
        help="Fail instead of releasing a training set too small for the planned full run.",
    )
    parser.add_argument(
        "--reference-input-tokens-per-second",
        type=float,
        default=2418.9274035045514,
        help="Measured four-A100 SFT input-token throughput from the cap32 timing run.",
    )
    parser.add_argument("--planned-max-sft-epochs", type=int, default=12)
    parser.add_argument(
        "--reference-validation-train-time-ratio",
        type=float,
        default=0.2170806685965555,
        help="Measured validation/train wall-time ratio from the cap32 timing run.",
    )
    parser.add_argument("--maximum-estimated-sft-hours", type=float, default=72.0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Graph-scan worker processes; 0 selects min(available CPUs, "
            f"{MAX_AUTO_SCAN_WORKERS}). Use 1 for the deterministic in-process path."
        ),
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=DEFAULT_WORKER_BATCH_SIZE,
        help="Graph files per process-pool task; bounds Future count and IPC overhead.",
    )
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if not 0 < args.train_candidate_record_fraction <= 1:
        parser.error("--train-candidate-record-fraction must be in (0, 1]")
    if not 0 < args.evaluation_candidate_record_fraction <= 1:
        parser.error("--evaluation-candidate-record-fraction must be in (0, 1]")
    if not 0 < args.seen_evaluation_candidate_record_fraction <= 1:
        parser.error("--seen-evaluation-candidate-record-fraction must be in (0, 1]")
    if args.max_records_per_family < 1:
        parser.error("--max-records-per-family must be positive")
    if args.maximum_train_records < 1:
        parser.error("--maximum-train-records must be positive")
    if args.minimum_train_records > args.maximum_train_records:
        parser.error("--minimum-train-records cannot exceed --maximum-train-records")
    if args.target_train_input_tokens_per_epoch < 1:
        parser.error("--target-train-input-tokens-per-epoch must be positive")
    if args.reference_input_tokens_per_second <= 0:
        parser.error("--reference-input-tokens-per-second must be positive")
    if args.planned_max_sft_epochs < 1:
        parser.error("--planned-max-sft-epochs must be positive")
    if args.reference_validation_train_time_ratio < 0:
        parser.error("--reference-validation-train-time-ratio cannot be negative")
    if args.maximum_estimated_sft_hours <= 0:
        parser.error("--maximum-estimated-sft-hours must be positive")
    if not 1 <= args.max_prefixes_per_train_record <= 3:
        parser.error("--max-prefixes-per-train-record must be in [1, 3]")
    if args.workers < 0:
        parser.error("--workers must be zero (automatic) or positive")
    if args.worker_batch_size < 1:
        parser.error("--worker-batch-size must be positive")
    if args.coverage_graphs_per_train_organism < 0:
        parser.error("--coverage-graphs-per-train-organism cannot be negative")
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
    paths = output_paths(output_dir, args.max_records_per_family)
    control_paths = control_output_paths(output_dir, args.max_records_per_family)
    validate_outputs(paths, args.overwrite)
    validate_control_outputs(control_paths, args.overwrite)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    test_organisms = parse_organisms(args.test_organisms)
    (
        _all_families,
        test_available_families,
        non_test_available_families,
        inventory,
    ) = discover_inventory(
        graph_root,
        test_organisms,
    )
    test_families, validation_families, train_families = choose_family_splits(
        test_available_families=test_available_families,
        non_test_available_families=non_test_available_families,
        test_fraction=args.test_family_fraction,
        validation_fraction=args.validation_family_fraction,
        seed=args.seed,
    )

    database_path = output_dir / ".structured_candidates.sqlite3"
    database_path.unlink(missing_ok=True)
    connection = initialize_database(database_path)
    try:
        scan_stats = scan_candidates(
            graph_root,
            connection,
            test_organisms=test_organisms,
            test_families=test_families,
            validation_families=validation_families,
            train_families=train_families,
            train_candidate_fraction=args.train_candidate_record_fraction,
            evaluation_candidate_fraction=args.evaluation_candidate_record_fraction,
            seen_evaluation_candidate_fraction=(
                args.seen_evaluation_candidate_record_fraction
            ),
            seed=args.seed,
            max_files=args.max_files,
            progress_every=args.progress_every,
            workers=args.workers,
            worker_batch_size=args.worker_batch_size,
            coverage_graphs_per_train_organism=(
                args.coverage_graphs_per_train_organism
            ),
        )
        if scan_stats.get("graphs_failed_record_roundtrip", 0):
            raise ValueError(
                "strict build found canonical record round-trip failures: "
                f"{scan_stats['graphs_failed_record_roundtrip']} graphs"
            )
        selected = {
            split: select_records(connection, split)
            for split in SPLITS
        }
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        database_path.with_suffix(".sqlite3-wal").unlink(missing_ok=True)
        database_path.with_suffix(".sqlite3-shm").unlink(missing_ok=True)

    candidate_strict_families = (
        {str(row["family"]) for row in selected["test"]}
        & {str(row["family"]) for row in selected["test_family_only"]}
    )
    if not candidate_strict_families:
        raise ValueError(
            "no strict family has structurally valid candidates in both held-out "
            "and seen-organism evaluation partitions"
        )
    selected["test"] = [
        row for row in selected["test"]
        if str(row["family"]) in candidate_strict_families
    ]
    selected["test_family_only"] = [
        row for row in selected["test_family_only"]
        if str(row["family"]) in candidate_strict_families
    ]

    # Individual malformed source artifacts are quarantined and counted.  The
    # strict guarantee is that no partially accepted graph contributes a row;
    # one bad file must not make 1.3 million independent source graphs unusable.
    if scan_stats.get("views_excluded_source_identity_mismatch", 0):
        raise ValueError(
            "strict build found processed_graph metadata/path identity mismatches: "
            f"{scan_stats['views_excluded_source_identity_mismatch']} views"
        )

    split_outputs = {}
    for split in SPLITS:
        split_outputs[split] = write_selected_split(
            selected[split],
            split=split,
            csv_path=paths[f"{split}_csv"],
            record_path=paths[f"{split}_records"],
            tokenizer=tokenizer,
            max_length=args.max_length,
            max_prefixes_per_train_record=args.max_prefixes_per_train_record,
            max_records_per_family=args.max_records_per_family,
            seed=args.seed,
            maximum_records=(args.maximum_train_records if split == "train" else 0),
            target_input_tokens_per_epoch=(
                args.target_train_input_tokens_per_epoch if split == "train" else 0
            ),
            progress_every=args.progress_every,
        )
    partition_identities = {
        split: record_partition_identities(paths[f"{split}_records"])
        for split in SPLITS
    }
    if (
        partition_identities["test"]["families"]
        != partition_identities["test_family_only"]["families"]
    ):
        raise ValueError(
            "strict and family-only evaluation lost different families after token filtering"
        )
    if partition_identities["test"]["organisms"] != test_organisms:
        raise ValueError(
            "strict test does not cover the complete declared held-out organism set"
        )
    if partition_identities["test_organism_only"]["organisms"] != test_organisms:
        raise ValueError(
            "organism-only test does not cover the complete declared held-out organism set"
        )
    if not (
        partition_identities["train"]["families"]
        & partition_identities["test_organism_only"]["families"]
    ):
        raise ValueError("organism-only test has no train-family overlap")
    if (
        partition_identities["train"]["families"]
        & partition_identities["validation"]["families"]
        or partition_identities["train"]["families"]
        & partition_identities["test"]["families"]
        or partition_identities["validation"]["families"]
        & partition_identities["test"]["families"]
    ):
        raise ValueError("train/validation/strict-test pathway families are not disjoint")
    if (
        partition_identities["train"]["organisms"] & test_organisms
        or partition_identities["validation"]["organisms"] & test_organisms
        or partition_identities["test_family_only"]["organisms"] & test_organisms
    ):
        raise ValueError("held-out organisms leaked into a seen-organism partition")
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1 :]:
            for identity in ("sources", "graphs", "views", "records"):
                overlap = (
                    partition_identities[left][identity]
                    & partition_identities[right][identity]
                )
                if overlap:
                    raise ValueError(
                        f"{identity} overlap between {left} and {right}: "
                        f"{sorted(overlap)[:3]}"
                    )
    prompt_control_outputs: dict[str, dict[str, Any]] = {}
    for profile, profile_paths in control_paths.items():
        prompt_control_outputs[profile] = {}
        for split in SPLITS:
            result = write_profile_control_csv(
                primary_csv_path=paths[f"{split}_csv"],
                record_path=paths[f"{split}_records"],
                output_path=profile_paths[split],
                prompt_profile=profile,
                split=split,
                tokenizer=tokenizer,
                max_length=args.max_length,
            )
            if profile == NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS:
                if result["rows"] != split_outputs[split]["rows"]:
                    raise ValueError(
                        f"P0/P1 paired row count mismatch for {split}: "
                        f"{split_outputs[split]['rows']} != {result['rows']}"
                    )
            prompt_control_outputs[profile][split] = {
                **result,
                "path": profile_paths[split].relative_to(output_dir).as_posix(),
            }
    p2_empty_partitions = [
        split
        for split, result in prompt_control_outputs[
            SPECIES_NEUTRAL_IDS_NO_ORGANISM
        ].items()
        if not result["rows"]
    ]
    if p2_empty_partitions:
        raise ValueError(
            "strict species-neutral control has no eligible rows for partitions: "
            + ",".join(p2_empty_partitions)
        )
    if split_outputs["train"]["records"] < args.minimum_train_records:
        raise ValueError(
            f"accepted train records={split_outputs['train']['records']} is below "
            f"--minimum-train-records={args.minimum_train_records} after strict token filtering; "
            "increase --coverage-graphs-per-train-organism and/or "
            "--train-candidate-record-fraction, then rebuild"
        )

    sources = referenced_sources(paths[f"{split}_records"] for split in SPLITS)
    source_hash_info = write_source_graph_hashes(
        graph_root,
        sources,
        paths["source_graph_hashes"],
        overwrite=args.overwrite,
    )
    processed_coverage = processed_artifact_coverage(processed_root, sources)
    if processed_coverage.get("status") == "incomplete":
        raise ValueError(
            "processed/processed_graph source-path coverage is incomplete: "
            f"missing={processed_coverage['missing_counterparts']}"
        )

    estimated_train_hours_per_epoch = (
        split_outputs["train"]["estimated_input_tokens_per_epoch"]
        / args.reference_input_tokens_per_second
        / 3600.0
    )
    estimated_hours = (
        estimated_train_hours_per_epoch
        * (1.0 + args.reference_validation_train_time_ratio)
        * args.planned_max_sft_epochs
    )
    if estimated_hours > args.maximum_estimated_sft_hours:
        raise ValueError(
            f"estimated maximum SFT wall time={estimated_hours:.1f}h exceeds "
            f"budget={args.maximum_estimated_sft_hours:.1f}h; lower the train token budget"
        )
    build_identity_payload = {
        "schema_version": "structured_pathway_record_v3",
        "inventory_sha256": inventory["path_size_inventory_sha256"],
        "seed": args.seed,
        "test_organisms": sorted(test_organisms),
        "test_families": sorted(partition_identities["test"]["families"]),
        "validation_families": sorted(
            partition_identities["validation"]["families"]
        ),
        "train_families": sorted(partition_identities["train"]["families"]),
        "max_length": args.max_length,
        "max_records_per_family": args.max_records_per_family,
        "maximum_train_records": args.maximum_train_records,
        "target_train_input_tokens_per_epoch": (
            args.target_train_input_tokens_per_epoch
        ),
        "max_prefixes_per_train_record": args.max_prefixes_per_train_record,
        "coverage_graphs_per_train_organism": (
            args.coverage_graphs_per_train_organism
        ),
        "split_hashes": {
            split: {
                "csv_sha256": split_outputs[split]["csv_sha256"],
                "records_sha256": split_outputs[split]["records_sha256"],
            }
            for split in SPLITS
        },
        "source_graph_hashes_sha256": source_hash_info["sha256"],
        "prompt_control_hashes": {
            profile: {
                split: result["csv_sha256"]
                for split, result in split_results.items()
            }
            for profile, split_results in prompt_control_outputs.items()
        },
    }
    dataset_build_id = "dataset:" + hashlib.sha256(
        compact_json(build_identity_payload).encode("utf-8")
    ).hexdigest()[:24]
    manifest = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "dataset_build_id": dataset_build_id,
        "generated_at_utc": utc_now(),
        "generator": "dataprocess/build_structured_dataset.py",
        "do_not_edit": "Regenerate this file together with the dataset; do not hand edit.",
        "outputs": {
            key: path.name
            for key, path in paths.items()
            if key not in {"manifest", "audit"}
        },
        "processed_graph_root": str(graph_root),
        "processed_root": str(processed_root) if processed_root is not None else None,
        "processed_artifact_coverage": processed_coverage,
        "inventory": inventory,
        "seed": args.seed,
        "test_organisms": sorted(test_organisms),
        "strict_test_families": sorted(
            partition_identities["test"]["families"]
        ),
        "validation_families": sorted(
            partition_identities["validation"]["families"]
        ),
        "train_families": sorted(partition_identities["train"]["families"]),
        "split_policy": (
            "five-way: train=seen organism/train family; validation=seen organism/"
            "held-out validation family; test=held-out organism/strict family; "
            "test_family_only=seen organism/strict family; "
            "test_organism_only=held-out organism/train family"
        ),
        "prompt_policy": (
            "prefix-only primary profile with explicit known organism; no pathway "
            "name, class, id, title, block, or phenotype in model-visible prompt"
        ),
        "csv_header": V3_CSV_FIELDNAMES,
        "primary_prompt_profile": PRIMARY_PROMPT_PROFILE,
        "paired_prompt_profiles": {
            "status": "published",
            "published": True,
            "files": prompt_control_outputs,
            "profile_contracts": {
                NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS: {
                    "published": True,
                    "base_sample_contract": "exact_primary_set",
                    "answer_contract": "exact_primary_answer",
                    "species_claim": "no_explicit_name_only_native_ids_can_leak_species",
                },
                SPECIES_NEUTRAL_IDS_NO_ORGANISM: {
                    "published": True,
                    "base_sample_contract": "strict_natural_neutral_subset",
                    "answer_contract": "exact_primary_answer_on_shared_base_samples",
                    "mapping_contract": "no_prefix_stripping_or_synthetic_mapping",
                },
            },
        },
        "prompt_controls": prompt_control_outputs,
        "prompt_control_policy": {
            NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS: (
                "exact P0 base-sample set and target; native IDs may still reveal species"
            ),
            SPECIES_NEUTRAL_IDS_NO_ORGANISM: (
                "strict natural-neutral subset only; no prefix stripping or synthetic mapping"
            ),
        },
        "phenotype_policy": "not_annotated metadata only; absent from model input and target",
        "parser_source": SUBSTEP_SOURCE,
        "max_length": args.max_length,
        "train_candidate_record_fraction": args.train_candidate_record_fraction,
        "coverage_graphs_per_train_organism": (
            args.coverage_graphs_per_train_organism
        ),
        "evaluation_candidate_record_fraction": args.evaluation_candidate_record_fraction,
        "seen_evaluation_candidate_record_fraction": (
            args.seen_evaluation_candidate_record_fraction
        ),
        "max_records_per_family": args.max_records_per_family,
        "maximum_train_records": args.maximum_train_records,
        "target_train_input_tokens_per_epoch": (
            args.target_train_input_tokens_per_epoch
        ),
        "max_prefixes_per_train_record": args.max_prefixes_per_train_record,
        "scan": scan_stats,
        "splits": split_outputs,
        "source_graph_hashes": source_hash_info,
        "runtime_estimate": {
            "reference_run": "cap32_oneprefix_sft_20260713_epoch1_four_a100",
            "reference_input_tokens_per_second": args.reference_input_tokens_per_second,
            "reference_validation_train_time_ratio": (
                args.reference_validation_train_time_ratio
            ),
            "planned_max_sft_epochs": args.planned_max_sft_epochs,
            "estimated_train_hours_per_epoch": estimated_train_hours_per_epoch,
            "estimated_maximum_sft_hours": estimated_hours,
            "maximum_estimated_sft_hours": args.maximum_estimated_sft_hours,
            "warning": (
                "Estimate uses measured four-A100 throughput; the first v3 epoch log "
                "must replace it with observed throughput."
            ),
        },
    }
    temporary_manifest = paths["manifest"].with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(paths["manifest"])

    _generate_release_audit_compat(
        paths=paths,
        graph_root=graph_root,
        tokenizer=tokenizer,
        max_length=args.max_length,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
