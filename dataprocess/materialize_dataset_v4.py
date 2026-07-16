#!/usr/bin/env python3
"""Split the full v4 index and materialize a token-budgeted formal release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import sqlite3
import stat
import sys
import zlib
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dataprocess.event_text import TEMPLATE_ASSET, template_provenance
from dataprocess.prompt_profiles import (
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
    forbidden_model_metadata_markers,
)
from dataprocess.release_contract_v4 import (
    ALL_SPLITS,
    AUDIT_NAME,
    AUDIT_SCHEMA_VERSION,
    CSV_NAMES,
    MANIFEST_NAME,
    MATERIALIZATION_DATABASE_NAME,
    PRIMARY_PROMPT_PROFILE,
    RECORD_NAMES,
    RELEASE_SCHEMA_VERSION,
    SOURCE_GRAPH_HASHES_NAME,
    SPLIT_ASSIGNMENTS_NAME,
)
from dataprocess.split_policy_v4 import (
    FamilyWeight,
    PrefixChoice,
    SourceCoverage,
    assign_exact_horizons,
    assign_family_splits,
    choose_coverage_holdout_sources,
    prefix_choices,
)
from dataprocess.structured_schema import (
    DATASET_SCHEMA_VERSION,
    SUBSTEP_SOURCE,
    V4_CSV_FIELDNAMES,
    compact_json,
    csv_row,
    record_from_object,
    total_training_tokens,
)


DEFAULT_TRAIN_TOKEN_BUDGET = 515_000_000
DEFAULT_EVALUATION_RECORDS = 20_000
DEFAULT_MAX_LENGTH = 8192
REFERENCE_TOKENS_PER_SECOND = 2418.9274035045514
REFERENCE_VALIDATION_TRAIN_RATIO = 0.2170806685965555
DEFAULT_TOKEN_WORKERS = 32
DEFAULT_TOKEN_WORKER_BATCH_SIZE = 8

_TOKEN_WORKER_TOKENIZER: Any | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object], *, readonly: bool) -> None:
    if path.exists():
        path.chmod(0o644)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    if readonly:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def load_record(blob: bytes) -> Any:
    payload = json.loads(zlib.decompress(blob))
    return record_from_object(payload)


def initialize_token_worker(tokenizer_path: str) -> None:
    global _TOKEN_WORKER_TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer

    _TOKEN_WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def tokenize_record_batch(
    task: tuple[str, int, tuple[tuple[str, int, bytes], ...]],
) -> tuple[tuple[str, int, int, tuple[tuple[str, int, int], ...]], ...]:
    """Tokenize one ordered record batch inside an isolated worker process."""

    split, max_length, values = task
    if _TOKEN_WORKER_TOKENIZER is None:
        raise RuntimeError("token worker was not initialized")
    output: list[tuple[str, int, int, tuple[tuple[str, int, int], ...]]] = []
    for record_id, priority, blob in values:
        record = load_record(blob)
        accepted: list[tuple[str, int, int]] = []
        candidate_rows = 0
        for choice in prefix_choices(len(record.layers)):
            materialized = csv_row(
                record,
                choice.prefix_len,
                prompt_profile=PRIMARY_PROMPT_PROFILE,
                prefix_horizon=choice.horizon,
                split=split,
            )
            token_count = total_training_tokens(
                _TOKEN_WORKER_TOKENIZER, materialized
            )
            candidate_rows += 1
            if token_count <= max_length:
                accepted.append((choice.horizon, choice.prefix_len, token_count))
        output.append((record_id, priority, candidate_rows, tuple(accepted)))
    return tuple(output)


def bounded_ordered_executor_map(
    executor: ProcessPoolExecutor,
    tasks: Iterable[tuple[str, int, tuple[tuple[str, int, bytes], ...]]],
    *,
    maximum_pending: int,
) -> Iterable[tuple[tuple[str, int, int, tuple[tuple[str, int, int], ...]], ...]]:
    """Yield parallel results in input order with bounded coordinator memory."""

    iterator = iter(tasks)
    pending: deque[Any] = deque()
    for _ in range(maximum_pending):
        try:
            pending.append(executor.submit(tokenize_record_batch, next(iterator)))
        except StopIteration:
            break
    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            pending.append(executor.submit(tokenize_record_batch, next(iterator)))
        except StopIteration:
            continue


def open_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-524288")
    connection.execute("PRAGMA mmap_size=2147483648")
    contract = connection.execute(
        "SELECT value_json FROM meta WHERE key='generator_contract'"
    ).fetchone()
    if contract is None:
        raise ValueError("canonical index lacks generator contract")
    value = json.loads(contract[0])
    if value.get("record_schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("canonical index record schema is not v4")
    return connection


def verify_complete_index(index_dir: Path, connection: sqlite3.Connection) -> dict[str, Any]:
    status_path = index_dir / "index_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True:
        raise ValueError("canonical index status is not complete")
    indexed = connection.execute("SELECT COUNT(*) FROM graphs").fetchone()[0]
    inventory = int(status["inventory"]["graph_files"])
    if indexed != inventory:
        raise ValueError(
            f"canonical index graph count={indexed} differs from inventory={inventory}"
        )
    return status


def apply_split_policy(
    connection: sqlite3.Connection,
    *,
    source_holdout_fraction: float,
    seed: int,
    protected_sources: Sequence[str],
) -> dict[str, Any]:
    source_coverage = {
        str(source): SourceCoverage(
            records=int(records),
            graphs=int(graphs),
            families=int(families),
            layer_total=int(layer_total),
            semantic_event_total=int(semantic_event_total),
        )
        for source, records, graphs, families, layer_total, semantic_event_total in connection.execute(
            """
            SELECT organism, COUNT(*), COUNT(DISTINCT graph_id), COUNT(DISTINCT family),
                   SUM(layer_count), SUM(semantic_event_count)
            FROM records GROUP BY organism ORDER BY organism
            """
        )
    }
    heldout, holdout_report = choose_coverage_holdout_sources(
        source_coverage,
        fraction=source_holdout_fraction,
        seed=seed,
        protected_sources=protected_sources,
    )
    connection.execute("DROP TABLE IF EXISTS temp.heldout_sources")
    connection.execute(
        "CREATE TEMP TABLE heldout_sources(source_code TEXT PRIMARY KEY)"
    )
    connection.executemany(
        "INSERT INTO heldout_sources VALUES (?)",
        ((source_code,) for source_code in sorted(heldout)),
    )
    family_organism_counts: dict[str, dict[str, int]] = defaultdict(dict)
    for family, organism, count in connection.execute(
        """
        SELECT r.family, r.organism, COUNT(*)
        FROM records r LEFT JOIN heldout_sources h ON h.source_code=r.organism
        WHERE h.source_code IS NULL
        GROUP BY r.family, r.organism
        """
    ):
        family_organism_counts[family][organism] = int(count)
    family_weights = {
        family: FamilyWeight(sum(counts.values()), counts)
        for family, counts in family_organism_counts.items()
        if sum(counts.values()) > 0
    }
    family_assignment, family_report = assign_family_splits(
        family_weights, seed=seed
    )
    all_families = {
        str(row[0]) for row in connection.execute("SELECT DISTINCT family FROM records")
    }
    heldout_only_families = sorted(all_families - set(family_assignment))
    # A family observed only in held-out sources has no seen-source weight to
    # optimize. It is necessarily unseen during training, so retain it in the
    # strict source-plus-family test instead of silently dropping it.
    family_assignment.update({family: "test" for family in heldout_only_families})
    family_report["heldout_source_only_families"] = len(heldout_only_families)
    family_report["heldout_source_only_family_examples"] = heldout_only_families[:20]
    connection.execute("DROP TABLE IF EXISTS temp.family_assignments")
    connection.execute(
        "CREATE TEMP TABLE family_assignments(family TEXT PRIMARY KEY, main_split TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO family_assignments VALUES (?, ?)",
        sorted(family_assignment.items()),
    )
    connection.execute("DROP TABLE IF EXISTS temp.record_splits")
    connection.execute(
        "CREATE TEMP TABLE record_splits(record_id TEXT PRIMARY KEY, split TEXT NOT NULL)"
    )
    connection.execute(
        """
        INSERT INTO record_splits(record_id, split)
        SELECT r.record_id,
               CASE
                   WHEN h.source_code IS NULL THEN f.main_split
                   WHEN f.main_split='train' THEN 'test_organism'
                   ELSE 'test_strict'
               END
        FROM records r
        JOIN family_assignments f ON f.family=r.family
        LEFT JOIN heldout_sources h ON h.source_code=r.organism
        """
    )
    canonical_counts = {
        split: {
            "records": count,
            "graphs": graphs,
            "sources": sources,
            "organisms": organisms_count,
            "families": families,
        }
        for split, count, graphs, sources, organisms_count, families in connection.execute(
            """
            SELECT s.split, COUNT(*), COUNT(DISTINCT r.graph_id),
                   COUNT(DISTINCT r.source_graph_json), COUNT(DISTINCT r.organism),
                   COUNT(DISTINCT r.family)
            FROM records r JOIN record_splits s USING(record_id)
            GROUP BY s.split ORDER BY s.split
            """
        )
    }
    assigned_records = sum(int(values["records"]) for values in canonical_counts.values())
    canonical_records = int(
        connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    )
    if assigned_records != canonical_records:
        raise ValueError(
            "canonical split assignment dropped records: "
            f"assigned={assigned_records} canonical={canonical_records}"
        )
    return {
        "source_holdout": holdout_report,
        "heldout_sources": sorted(heldout),
        "protected_sources": sorted(set(protected_sources) & set(source_coverage)),
        "family_assignment": dict(sorted(family_assignment.items())),
        "family_optimizer": family_report,
        "canonical_split_counts": canonical_counts,
        "canonical_assignment_coverage": {
            "records": canonical_records,
            "assigned_records": assigned_records,
            "unassigned_records": canonical_records - assigned_records,
        },
    }


def candidate_order(
    connection: sqlite3.Connection,
    split: str,
    *,
    priority_organism: str,
) -> list[str]:
    rows = list(
        connection.execute(
            "SELECT r.record_id, r.graph_id, r.organism, r.rank "
            "FROM records r JOIN record_splits s USING(record_id) "
            "WHERE s.split=? ORDER BY r.rank",
            (split,),
        )
    )
    if not rows:
        return []
    first_per_organism: dict[str, sqlite3.Row] = {}
    first_per_graph: dict[str, sqlite3.Row] = {}
    for row in rows:
        first_per_organism.setdefault(row["organism"], row)
        first_per_graph.setdefault(row["graph_id"], row)
    ordered: list[str] = []
    seen: set[str] = set()

    def add(values: Iterable[sqlite3.Row]) -> None:
        for row in values:
            record_id = str(row["record_id"])
            if record_id not in seen:
                seen.add(record_id)
                ordered.append(record_id)

    if split == "train":
        add(sorted(first_per_organism.values(), key=lambda row: row["rank"]))
        add(row for row in rows if row["organism"] == priority_organism)
    add(sorted(first_per_graph.values(), key=lambda row: row["rank"]))
    add(rows)
    return ordered


def initialize_work_database(path: Path, *, overwrite: bool) -> sqlite3.Connection:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"work database already exists: {path}")
        path.unlink()
    if overwrite:
        # An interrupted SQLite process may leave sidecars behind.  They must
        # never be replayed into a new formal materialization database.
        for suffix in ("-wal", "-shm"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE choices (
            split TEXT NOT NULL,
            record_id TEXT NOT NULL,
            prefix_len INTEGER NOT NULL,
            horizon TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            PRIMARY KEY(split, record_id, horizon)
        );
        CREATE TABLE assignments (
            split TEXT NOT NULL,
            record_id TEXT NOT NULL,
            prefix_len INTEGER NOT NULL,
            horizon TEXT NOT NULL,
            token_count INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            PRIMARY KEY(split, record_id)
        );
        """
    )
    return connection


def collect_eligible_choices(
    index: sqlite3.Connection,
    work: sqlite3.Connection,
    tokenizer: Any,
    *,
    split: str,
    record_ids: Sequence[str],
    max_length: int,
    train_token_budget: int,
    maximum_train_records: int,
    maximum_evaluation_records: int,
    progress_every: int,
) -> dict[str, int]:
    stats: Counter[str] = Counter()
    conservative_tokens = 0
    for priority, record_id in enumerate(record_ids):
        stats["records_considered"] = priority + 1
        if split != "train" and stats["selected_records"] >= maximum_evaluation_records:
            break
        if split == "train" and maximum_train_records and stats["selected_records"] >= maximum_train_records:
            break
        row = index.execute(
            "SELECT r.record_zlib FROM records r JOIN record_splits s USING(record_id) "
            "WHERE r.record_id=? AND s.split=?",
            (record_id, split),
        ).fetchone()
        if row is None:
            raise ValueError(f"candidate record disappeared from split {split}: {record_id}")
        record = load_record(row[0])
        accepted: list[tuple[str, int, int]] = []
        for choice in prefix_choices(len(record.layers)):
            materialized = csv_row(
                record,
                choice.prefix_len,
                prompt_profile=PRIMARY_PROMPT_PROFILE,
                prefix_horizon=choice.horizon,
                split=split,
            )
            token_count = total_training_tokens(tokenizer, materialized)
            stats["candidate_rows"] += 1
            if token_count > max_length:
                stats["rows_excluded_over_max_length"] += 1
                continue
            accepted.append((choice.horizon, choice.prefix_len, token_count))
        if not accepted:
            stats["records_excluded_no_complete_in_budget_choice"] += 1
            continue
        record_cost = max(value[2] for value in accepted)
        if split == "train" and conservative_tokens + record_cost > train_token_budget:
            stats["records_excluded_train_token_budget"] += 1
            continue
        conservative_tokens += record_cost
        work.executemany(
            "INSERT INTO choices VALUES (?, ?, ?, ?, ?, ?)",
            (
                (split, record_id, prefix_len, horizon, token_count, priority)
                for horizon, prefix_len, token_count in accepted
            ),
        )
        stats["selected_records"] += 1
        if stats["selected_records"] % 500 == 0:
            work.commit()
        if progress_every and (priority + 1) % progress_every == 0:
            print(
                f"tokenizing_split={split} considered={priority + 1} "
                f"selected={stats['selected_records']} conservative_tokens={conservative_tokens}",
                file=sys.stderr,
                flush=True,
            )
    work.commit()
    stats["conservative_token_budget_used"] = conservative_tokens
    return dict(stats)


def tokenization_tasks(
    index: sqlite3.Connection,
    *,
    split: str,
    record_ids: Sequence[str],
    max_length: int,
    batch_size: int,
) -> Iterable[tuple[str, int, tuple[tuple[str, int, bytes], ...]]]:
    batch: list[tuple[str, int, bytes]] = []
    for priority, record_id in enumerate(record_ids):
        row = index.execute(
            "SELECT r.record_zlib FROM records r JOIN record_splits s USING(record_id) "
            "WHERE r.record_id=? AND s.split=?",
            (record_id, split),
        ).fetchone()
        if row is None:
            raise ValueError(
                f"candidate record disappeared from split {split}: {record_id}"
            )
        batch.append((record_id, priority, bytes(row[0])))
        if len(batch) == batch_size:
            yield split, max_length, tuple(batch)
            batch = []
    if batch:
        yield split, max_length, tuple(batch)


def collect_eligible_choices_parallel(
    index: sqlite3.Connection,
    work: sqlite3.Connection,
    executor: ProcessPoolExecutor,
    *,
    split: str,
    record_ids: Sequence[str],
    max_length: int,
    train_token_budget: int,
    maximum_train_records: int,
    maximum_evaluation_records: int,
    progress_every: int,
    batch_size: int,
    maximum_pending: int,
) -> dict[str, int]:
    """Parallel equivalent of ``collect_eligible_choices`` with stable order."""

    stats: Counter[str] = Counter()
    conservative_tokens = 0
    stop = False
    tasks = tokenization_tasks(
        index,
        split=split,
        record_ids=record_ids,
        max_length=max_length,
        batch_size=batch_size,
    )
    for batch in bounded_ordered_executor_map(
        executor, tasks, maximum_pending=maximum_pending
    ):
        for record_id, priority, candidate_rows, accepted_values in batch:
            if (
                split != "train"
                and stats["selected_records"] >= maximum_evaluation_records
            ) or (
                split == "train"
                and maximum_train_records
                and stats["selected_records"] >= maximum_train_records
            ):
                stop = True
                break
            stats["records_considered"] = priority + 1
            stats["candidate_rows"] += candidate_rows
            stats["rows_excluded_over_max_length"] += (
                candidate_rows - len(accepted_values)
            )
            if not accepted_values:
                stats["records_excluded_no_complete_in_budget_choice"] += 1
                continue
            record_cost = max(value[2] for value in accepted_values)
            if split == "train" and conservative_tokens + record_cost > train_token_budget:
                stats["records_excluded_train_token_budget"] += 1
                continue
            conservative_tokens += record_cost
            work.executemany(
                "INSERT INTO choices VALUES (?, ?, ?, ?, ?, ?)",
                (
                    (split, record_id, prefix_len, horizon, token_count, priority)
                    for horizon, prefix_len, token_count in accepted_values
                ),
            )
            stats["selected_records"] += 1
            if stats["selected_records"] % 500 == 0:
                work.commit()
            considered = priority + 1
            if progress_every and considered % progress_every == 0:
                print(
                    f"tokenizing_split={split} considered={considered} "
                    f"selected={stats['selected_records']} "
                    f"conservative_tokens={conservative_tokens}",
                    file=sys.stderr,
                    flush=True,
                )
        if stop:
            break
    work.commit()
    stats["conservative_token_budget_used"] = conservative_tokens
    return dict(stats)


def assign_split_horizons(
    work: sqlite3.Connection,
    *,
    split: str,
    seed: int,
) -> dict[str, object]:
    eligible: dict[str, list[PrefixChoice]] = defaultdict(list)
    token_lookup: dict[tuple[str, str], int] = {}
    priorities: dict[str, int] = {}
    for record_id, prefix_len, horizon, token_count, priority in work.execute(
        "SELECT record_id, prefix_len, horizon, token_count, priority "
        "FROM choices WHERE split=? ORDER BY priority, record_id, horizon",
        (split,),
    ):
        eligible[record_id].append(PrefixChoice(prefix_len, horizon))
        token_lookup[(record_id, horizon)] = token_count
        priorities[record_id] = priority
    assignments, report = assign_exact_horizons(
        eligible, seed=seed + ALL_SPLITS.index(split)
    )
    work.executemany(
        "INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?)",
        (
            (
                split,
                record_id,
                choice.prefix_len,
                choice.horizon,
                token_lookup[(record_id, choice.horizon)],
                priorities[record_id],
            )
            for record_id, choice in assignments.items()
        ),
    )
    work.commit()
    report["assigned_input_tokens"] = sum(
        token_lookup[(record_id, choice.horizon)]
        for record_id, choice in assignments.items()
    )
    return report


def output_paths(output_dir: Path) -> dict[str, Path]:
    output: dict[str, Path] = {
        **{f"{split}_csv": output_dir / CSV_NAMES[split] for split in ALL_SPLITS},
        **{f"{split}_records": output_dir / RECORD_NAMES[split] for split in ALL_SPLITS},
        "manifest": output_dir / MANIFEST_NAME,
        "audit": output_dir / AUDIT_NAME,
        "source_hashes": output_dir / SOURCE_GRAPH_HASHES_NAME,
        "split_assignments": output_dir / SPLIT_ASSIGNMENTS_NAME,
    }
    return output


def prepare_outputs(paths: Mapping[str, Path], output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    control_paths = [
        output_dir / "prompt_controls" / profile / name
        for profile in (
            NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
            SPECIES_NEUTRAL_IDS_NO_ORGANISM,
        )
        for name in CSV_NAMES.values()
    ]
    existing = [path for path in (*paths.values(), *control_paths) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {existing[0]}; pass --overwrite"
        )
    # Remove the old audit before any other release artifact.  This is the
    # fail-closed boundary that prevents a scheduler from accepting a stale
    # passed audit while new CSV/JSONL files are being written.
    audit_path = paths["audit"]
    if audit_path.exists():
        audit_path.chmod(0o644)
        audit_path.unlink()
    for key, path in paths.items():
        if key != "audit" and path.exists():
            path.chmod(0o644)
            path.unlink()
    for profile in (
        NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
        SPECIES_NEUTRAL_IDS_NO_ORGANISM,
    ):
        directory = output_dir / "prompt_controls" / profile
        directory.mkdir(parents=True, exist_ok=True)
    for path in control_paths:
        if path.exists():
            path.unlink()


def _record_statistics(record: Any, counters: Counter[str]) -> None:
    counters["records"] += 1
    counters[f"layers:{len(record.layers)}"] += 1
    for layer in record.layers:
        semantic_ids = [event.semantic_event_id for event in layer.events]
        counters["layers"] += 1
        counters["semantic_events"] += len(layer.events)
        counters["semantic_duplicates_within_layer"] += len(semantic_ids) - len(set(semantic_ids))
        for event in layer.events:
            counters[f"event_type:{event.event_type}"] += 1
            counters["producer_events"] += len(event.producer_event_ids)
            counters["merged_producer_occurrences"] += len(event.producer_event_ids) - 1
            counters["aliases"] += sum(
                len(entity["aliases"])
                for side in (event.source, event.mediator, event.target)
                for entity in side
            )
            for side in (event.source, event.mediator, event.target):
                ids = [entity["canonical_id"] for entity in side]
                counters["duplicate_participant_canonical_ids"] += len(ids) - len(set(ids))
            if event.legacy_text is None:
                counters["legacy_text_unavailable"] += 1
            else:
                counters["legacy_text_available"] += 1


def write_materialized_outputs(
    index: sqlite3.Connection,
    work: sqlite3.Connection,
    tokenizer: Any,
    *,
    split: str,
    output_dir: Path,
    csv_path: Path,
    record_path: Path,
    max_length: int,
) -> tuple[dict[str, object], dict[str, set[str]]]:
    csv_tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    record_tmp = record_path.with_suffix(record_path.suffix + ".tmp")
    control_handles: dict[str, tuple[Any, csv.DictWriter, Path, Path]] = {}
    for profile in (
        NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
        SPECIES_NEUTRAL_IDS_NO_ORGANISM,
    ):
        final = output_dir / "prompt_controls" / profile / CSV_NAMES[split]
        temporary = final.with_suffix(final.suffix + ".tmp")
        handle = temporary.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=V4_CSV_FIELDNAMES)
        writer.writeheader()
        control_handles[profile] = (handle, writer, temporary, final)

    stats: Counter[str] = Counter()
    token_lengths: list[int] = []
    sample_ids: set[str] = set()
    record_ids: set[str] = set()
    graph_ids: set[str] = set()
    view_ids: set[str] = set()
    sources: set[str] = set()
    organisms: set[str] = set()
    families: set[str] = set()
    try:
        with csv_tmp.open("w", encoding="utf-8", newline="") as csv_handle, record_tmp.open(
            "w", encoding="utf-8"
        ) as records_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=V4_CSV_FIELDNAMES)
            writer.writeheader()
            assignments = work.execute(
                "SELECT record_id, prefix_len, horizon, token_count FROM assignments "
                "WHERE split=? ORDER BY priority, record_id",
                (split,),
            )
            for record_id, prefix_len, horizon, declared_tokens in assignments:
                source = index.execute(
                    "SELECT r.record_zlib FROM records r JOIN record_splits s USING(record_id) "
                    "WHERE r.record_id=? AND s.split=?",
                    (record_id, split),
                ).fetchone()
                if source is None:
                    raise ValueError(f"assigned record disappeared: {record_id}")
                record = load_record(source[0])
                row = csv_row(
                    record,
                    prefix_len,
                    prompt_profile=PRIMARY_PROMPT_PROFILE,
                    prefix_horizon=horizon,
                    split=split,
                )
                tokens = total_training_tokens(tokenizer, row)
                if tokens != declared_tokens or tokens > max_length:
                    raise ValueError(
                        f"token count changed for {row['sample_id']}: {declared_tokens} -> {tokens}"
                    )
                answer = json.loads(str(row["answer"]))
                if answer.get("schema_version") != "pathway_continuation_v4":
                    raise ValueError("materialized answer is not v4")
                question = str(row["question"])
                leaked_markers = forbidden_model_metadata_markers(question)
                if leaked_markers:
                    raise ValueError(
                        f"model-visible metadata leaked for {row['sample_id']}: "
                        + ", ".join(leaked_markers)
                    )
                if row["sample_id"] in sample_ids or record.record_id in record_ids:
                    raise ValueError("duplicate materialized sample or record identity")
                sample_ids.add(str(row["sample_id"]))
                record_ids.add(record.record_id)
                graph_ids.add(record.graph_id)
                view_ids.add(record.view_id)
                sources.add(record.source_graph_json)
                organisms.add(record.organism)
                families.add(record.family)
                writer.writerow(row)
                records_handle.write(compact_json(record.record_object()) + "\n")
                token_lengths.append(tokens)
                stats["rows"] += 1
                stats[f"horizon:{horizon}"] += 1
                _record_statistics(record, stats)

                p1 = csv_row(
                    record,
                    prefix_len,
                    prompt_profile=NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
                    prefix_horizon=horizon,
                    split=split,
                )
                if p1["answer"] != row["answer"]:
                    raise ValueError("P1 target differs from primary target")
                control_handles[NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS][1].writerow(p1)
                stats["p1_rows"] += 1
                try:
                    p2 = csv_row(
                        record,
                        prefix_len,
                        prompt_profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM,
                        prefix_horizon=horizon,
                        split=split,
                    )
                except ValueError:
                    stats["p2_ineligible_records"] += 1
                else:
                    if total_training_tokens(tokenizer, p2) <= max_length:
                        control_handles[SPECIES_NEUTRAL_IDS_NO_ORGANISM][1].writerow(p2)
                        stats["p2_rows"] += 1
                    else:
                        stats["p2_rows_over_max_length"] += 1
        csv_tmp.replace(csv_path)
        record_tmp.replace(record_path)
        for handle, _writer, temporary, final in control_handles.values():
            handle.close()
            temporary.replace(final)
    except BaseException:
        csv_tmp.unlink(missing_ok=True)
        record_tmp.unlink(missing_ok=True)
        for handle, _writer, temporary, _final in control_handles.values():
            handle.close()
            temporary.unlink(missing_ok=True)
        raise
    if not stats["rows"]:
        raise ValueError(f"split {split} materialized no records")
    layer_distribution = {
        key.split(":", 1)[1]: count
        for key, count in stats.items()
        if key.startswith("layers:")
    }
    control_file_stats = {
        profile: {
            "path": final.relative_to(output_dir).as_posix(),
            "sha256": file_sha256(final),
            "bytes": final.stat().st_size,
        }
        for profile, (_handle, _writer, _temporary, final) in control_handles.items()
    }
    public = {
        "rows": stats["rows"],
        "records": stats["records"],
        "graphs": len(graph_ids),
        "views": len(view_ids),
        "sources": len(sources),
        "organisms": len(organisms),
        "families": len(families),
        "input_tokens": sum(token_lengths),
        "token_length": {
            "min": min(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
            "max": max(token_lengths),
        },
        "horizons": {
            key.split(":", 1)[1]: count
            for key, count in stats.items()
            if key.startswith("horizon:")
        },
        "layer_length_distribution": dict(
            sorted(layer_distribution.items(), key=lambda item: int(item[0]))
        ),
        "substeps": {
            "layers": stats["layers"],
            "semantic_events": stats["semantic_events"],
            "producer_events": stats["producer_events"],
            "merged_producer_occurrences": stats["merged_producer_occurrences"],
            "relation_events": stats["event_type:relation"],
            "reaction_events": stats["event_type:reaction"],
            "semantic_duplicates_within_layer": stats["semantic_duplicates_within_layer"],
            "duplicate_participant_canonical_ids": stats["duplicate_participant_canonical_ids"],
            "aliases": stats["aliases"],
            "legacy_text_available": stats["legacy_text_available"],
            "legacy_text_unavailable": stats["legacy_text_unavailable"],
        },
        "prompt_controls": {
            "p1_rows": stats["p1_rows"],
            "p2_rows": stats["p2_rows"],
            "p2_ineligible_records": stats["p2_ineligible_records"],
            "files": control_file_stats,
        },
        "csv_sha256": file_sha256(csv_path),
        "records_sha256": file_sha256(record_path),
        "csv_bytes": csv_path.stat().st_size,
        "records_bytes": record_path.stat().st_size,
    }
    identities = {
        "samples": sample_ids,
        "records": record_ids,
        "graphs": graph_ids,
        "views": view_ids,
        "sources": sources,
        "families": families,
        "organisms": organisms,
    }
    return public, identities


def write_source_hashes(
    index: sqlite3.Connection,
    sources: set[str],
    output_path: Path,
) -> dict[str, object]:
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ["source_graph_json", "graph_id", "sha256", "bytes", "status"]
        )
        for source in sorted(sources):
            row = index.execute(
                "SELECT graph_id, content_sha256, file_size, status FROM graphs "
                "WHERE source_graph_json=?",
                (source,),
            ).fetchone()
            if row is None:
                raise ValueError(f"materialized source missing from full index: {source}")
            if row[3] != "ok":
                raise ValueError(
                    f"materialized source has non-eligible graph status {row[3]!r}: {source}"
                )
            writer.writerow([source, row[0], row[1], row[2], row[3]])
    temporary.replace(output_path)
    return {
        "sources": len(sources),
        "sha256": file_sha256(output_path),
        "bytes": output_path.stat().st_size,
    }


def build_audit(
    *,
    index: sqlite3.Connection,
    index_status: Mapping[str, Any],
    split_policy: Mapping[str, Any],
    collection: Mapping[str, Mapping[str, int]],
    horizons: Mapping[str, Mapping[str, object]],
    outputs: Mapping[str, Mapping[str, object]],
    identities: Mapping[str, Mapping[str, set[str]]],
    paths: Mapping[str, Path],
    source_hashes: Mapping[str, object],
    processed_root: Path | None,
    max_length: int,
    train_token_budget: int,
    minimum_train_records: int,
) -> dict[str, object]:
    failures: list[str] = []
    overlap: dict[str, dict[str, int]] = {}
    for left_index, left in enumerate(ALL_SPLITS):
        for right in ALL_SPLITS[left_index + 1 :]:
            key = f"{left}__{right}"
            overlap[key] = {
                identity: len(identities[left][identity] & identities[right][identity])
                for identity in (
                    "samples",
                    "records",
                    "graphs",
                    "views",
                    "sources",
                    "families",
                    "organisms",
                )
            }
            for identity in ("samples", "records", "graphs", "views", "sources"):
                if overlap[key][identity]:
                    failures.append(f"strict_identity_overlap:{key}:{identity}")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if identities[left]["families"] & identities[right]["families"]:
            failures.append(f"primary_family_overlap:{left}:{right}")
    heldout = set(split_policy["heldout_sources"])
    for split in ("train", "validation", "test"):
        if identities[split]["organisms"] & heldout:
            failures.append(f"heldout_organism_leak:{split}")
    if not identities["test_organism"]["organisms"].issubset(heldout):
        failures.append("test_organism_contains_seen_organism")
    if not identities["test_strict"]["organisms"].issubset(heldout):
        failures.append("test_strict_contains_seen_organism")
    if not (identities["test_organism"]["families"] & identities["train"]["families"]):
        failures.append("test_organism_has_no_train_family_overlap")
    if identities["test_strict"]["families"] & identities["train"]["families"]:
        failures.append("test_strict_family_leaks_train")
    if split_policy["canonical_assignment_coverage"]["unassigned_records"]:
        failures.append("canonical_records_missing_split_assignment")
    if outputs["train"]["records"] < minimum_train_records:
        failures.append("train_records_below_minimum")
    if outputs["train"]["input_tokens"] > train_token_budget:
        failures.append("train_token_budget_exceeded")
    for split in ALL_SPLITS:
        if outputs[split]["token_length"]["max"] > max_length:
            failures.append(f"max_length_exceeded:{split}")
        if outputs[split]["substeps"]["semantic_duplicates_within_layer"]:
            failures.append(f"semantic_duplicates_within_layer:{split}")
        if outputs[split]["substeps"]["duplicate_participant_canonical_ids"]:
            failures.append(f"duplicate_participant_ids:{split}")
        if not horizons[split]["actual_matches_theoretical_optimum"]:
            failures.append(f"horizon_not_globally_optimal:{split}")
    duplicate_record_ids = int(
        index.execute(
            "SELECT COUNT(*)-COUNT(DISTINCT record_id) FROM records"
        ).fetchone()[0]
    )
    duplicate_source_paths = int(
        index.execute(
            "SELECT COUNT(*)-COUNT(DISTINCT source_graph_json) FROM graphs"
        ).fetchone()[0]
    )
    duplicate_graph_ids = int(
        index.execute(
            "SELECT COUNT(*)-COUNT(DISTINCT graph_id) FROM graphs"
        ).fetchone()[0]
    )
    if duplicate_record_ids:
        failures.append("duplicate_canonical_record_ids")
    if duplicate_source_paths:
        failures.append("duplicate_source_graph_paths")
    if duplicate_graph_ids:
        failures.append("duplicate_source_graph_ids")
    all_sources = set().union(*(value["sources"] for value in identities.values()))
    processed_coverage: dict[str, object]
    if processed_root is None:
        processed_coverage = {"status": "not_requested"}
    else:
        index.execute("DROP TABLE IF EXISTS temp.selected_processed_sources")
        index.execute(
            "CREATE TEMP TABLE selected_processed_sources "
            "(source_graph_json TEXT PRIMARY KEY)"
        )
        index.executemany(
            "INSERT INTO selected_processed_sources VALUES (?)",
            ((source,) for source in sorted(all_sources)),
        )
        indexed_sources = int(
            index.execute(
                "SELECT COUNT(*) FROM selected_processed_sources s "
                "JOIN graphs g USING(source_graph_json)"
            ).fetchone()[0]
        )
        status_counts = dict(
            index.execute(
                "SELECT g.processed_text_status, COUNT(*) "
                "FROM selected_processed_sources s "
                "JOIN graphs g USING(source_graph_json) "
                "GROUP BY g.processed_text_status ORDER BY g.processed_text_status"
            )
        )
        text_totals = index.execute(
            "SELECT COALESCE(SUM(g.processed_file_size),0), "
            "COALESCE(SUM(g.visible_legacy_text_count),0), "
            "COALESCE(SUM(g.visible_legacy_text_match_count),0), "
            "SUM(CASE WHEN g.processed_content_sha256='' THEN 1 ELSE 0 END) "
            "FROM selected_processed_sources s JOIN graphs g USING(source_graph_json)"
        ).fetchone()
        missing = [
            source
            for source in sorted(all_sources)
            if not (processed_root / source).is_file()
        ]
        noncomplete = sum(
            int(count)
            for status, count in status_counts.items()
            if status != "complete"
        )
        unmatched = int(text_totals[1]) - int(text_totals[2])
        hashless = int(text_totals[3] or 0)
        complete = (
            indexed_sources == len(all_sources)
            and not missing
            and noncomplete == 0
            and unmatched == 0
            and hashless == 0
        )
        processed_coverage = {
            "status": "complete" if complete else "incomplete",
            "checked_sources": len(all_sources),
            "indexed_sources": indexed_sources,
            "indexed_status": status_counts,
            "missing_counterparts": len(missing),
            "missing_examples": missing[:20],
            "processed_bytes": int(text_totals[0]),
            "visible_legacy_text_events": int(text_totals[1]),
            "exact_legacy_text_matches": int(text_totals[2]),
            "unmatched_legacy_text_events": unmatched,
            "sources_without_processed_sha256": hashless,
            "policy": (
                "every selected producer event legacy_text must occur exactly in "
                "its matching historical processed paragraph artifact"
            ),
        }
        if not complete:
            failures.append("processed_counterpart_coverage_incomplete")
    all_file_hashes = {
        path.relative_to(paths["manifest"].parent).as_posix(): {
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for key, path in paths.items()
        if key != "audit" and path.is_file()
    }
    for control_path in sorted(
        paths["manifest"].parent.glob("prompt_controls/*/*.csv")
    ):
        all_file_hashes[control_path.relative_to(paths["manifest"].parent).as_posix()] = {
            "sha256": file_sha256(control_path),
            "bytes": control_path.stat().st_size,
        }
    canonical_summary = index_status["summary"]
    quarantine = canonical_summary["graph_status"]
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "do_not_edit": "Generated from immutable release artifacts; do not hand edit.",
        "generated_at_utc": utc_now(),
        "status": "passed" if not failures else "failed",
        "failures": sorted(set(failures)),
        "canonical_index": {
            "complete": index_status["complete"],
            "processed_graph_root": index_status["processed_graph_root"],
            "inventory": index_status["inventory"],
            "summary": canonical_summary,
            "database_sha256": index_status["database_sha256"],
            "quarantined_or_invalid_graphs": sum(
                int(quarantine.get(name, 0)) for name in ("quarantined", "invalid")
            ),
        },
        "canonical_splits": split_policy["canonical_split_counts"],
        "canonical_assignment_coverage": split_policy[
            "canonical_assignment_coverage"
        ],
        "family_split_optimizer": split_policy["family_optimizer"],
        "source_holdout": split_policy["source_holdout"],
        "materialized_splits": outputs,
        "candidate_filtering": collection,
        "horizon_balance": horizons,
        "strict_overlap": overlap,
        "organism_overlap": {
            key: values["organisms"] for key, values in overlap.items()
        },
        "duplicate_ids": {
            "canonical_record_ids": duplicate_record_ids,
            "source_graph_paths": duplicate_source_paths,
            "source_graph_ids": duplicate_graph_ids,
            "materialized_sample_ids_within_splits": 0,
            "materialized_sample_ids_across_splits": sum(
                values["samples"] for values in overlap.values()
            ),
            "materialized_record_ids_across_splits": sum(
                values["records"] for values in overlap.values()
            ),
        },
        "phenotype_status": {
            "policy": "not_annotated",
            "negative_inference_from_missing": False,
            "model_visible": False,
        },
        "parser_source": SUBSTEP_SOURCE,
        "substep_coverage": {
            split: outputs[split]["substeps"] for split in ALL_SPLITS
        },
        "layer_length_distribution": {
            split: outputs[split]["layer_length_distribution"] for split in ALL_SPLITS
        },
        "token_and_truncation": {
            "max_length": max_length,
            "policy": "exclude_complete_sample_before_materialization_never_truncate_json",
            "train_token_budget": train_token_budget,
            "splits": {
                split: {
                    "tokens": outputs[split]["input_tokens"],
                    "length": outputs[split]["token_length"],
                    "rows_excluded_over_max_length": collection[split].get(
                        "rows_excluded_over_max_length", 0
                    ),
                }
                for split in ALL_SPLITS
            },
        },
        "graph_artifact_coverage": {
            "processed_graph_inventory_indexed": index_status["complete"],
            "selected_source_hashes": source_hashes,
            "processed_text_counterparts": processed_coverage,
        },
        "hashes": {
            "files": all_file_hashes,
            "event_template_sha256": file_sha256(TEMPLATE_ASSET),
            "event_template_provenance": template_provenance(),
        },
    }
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--processed-root")
    parser.add_argument("--source-holdout-fraction", type=float, default=0.10)
    parser.add_argument(
        "--protected-sources",
        default="hsa,ko,ec",
        help="Comma-separated source codes never assigned to the unseen-source tests.",
    )
    parser.add_argument("--train-token-budget", type=int, default=DEFAULT_TRAIN_TOKEN_BUDGET)
    parser.add_argument("--maximum-train-records", type=int, default=0)
    parser.add_argument("--maximum-evaluation-records", type=int, default=DEFAULT_EVALUATION_RECORDS)
    parser.add_argument("--minimum-train-records", type=int, default=12_000)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--token-workers", type=int, default=DEFAULT_TOKEN_WORKERS)
    parser.add_argument(
        "--token-worker-batch-size",
        type=int,
        default=DEFAULT_TOKEN_WORKER_BATCH_SIZE,
    )
    parser.add_argument("--priority-organism", default="hsa")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not 0 < args.source_holdout_fraction < 1:
        parser.error("--source-holdout-fraction must be in (0, 1)")
    args.protected_sources = tuple(
        value.strip() for value in args.protected_sources.split(",") if value.strip()
    )
    if not args.protected_sources or len(set(args.protected_sources)) != len(args.protected_sources):
        parser.error("--protected-sources must contain distinct non-empty codes")
    if args.train_token_budget < 1 or args.maximum_evaluation_records < 1:
        parser.error("token and evaluation budgets must be positive")
    if args.maximum_train_records < 0 or args.minimum_train_records < 1:
        parser.error("record bounds are invalid")
    if args.max_length < 2:
        parser.error("--max-length must be at least 2")
    if args.token_workers < 1 or args.token_worker_batch_size < 1:
        parser.error("token worker counts and batch size must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    processed_root = (
        Path(args.processed_root).expanduser().resolve() if args.processed_root else None
    )
    index_path = index_dir / "canonical_index_v4.sqlite3"
    if not index_path.is_file():
        raise FileNotFoundError("canonical v4 index is missing")
    paths = output_paths(output_dir)
    prepare_outputs(paths, output_dir, overwrite=args.overwrite)
    index = open_index(index_path)
    work_path = output_dir / MATERIALIZATION_DATABASE_NAME
    work = initialize_work_database(work_path, overwrite=args.overwrite)
    try:
        index_status = verify_complete_index(index_dir, index)
        split_policy = apply_split_policy(
            index,
            source_holdout_fraction=args.source_holdout_fraction,
            seed=args.seed,
            protected_sources=args.protected_sources,
        )
        atomic_json(paths["split_assignments"], split_policy, readonly=True)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer,
            trust_remote_code=True,
            local_files_only=True,
        )
        collection: dict[str, dict[str, int]] = {}
        horizons: dict[str, dict[str, object]] = {}
        token_executor: ProcessPoolExecutor | None = None
        if args.token_workers > 1:
            token_executor = ProcessPoolExecutor(
                max_workers=args.token_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_token_worker,
                initargs=(str(Path(args.tokenizer).expanduser().resolve()),),
            )
        try:
            for split in ALL_SPLITS:
                order = candidate_order(
                    index, split, priority_organism=args.priority_organism
                )
                if not order:
                    raise ValueError(f"canonical split {split} has no records")
                common = {
                    "split": split,
                    "record_ids": order,
                    "max_length": args.max_length,
                    "train_token_budget": args.train_token_budget,
                    "maximum_train_records": args.maximum_train_records,
                    "maximum_evaluation_records": args.maximum_evaluation_records,
                    "progress_every": args.progress_every,
                }
                if token_executor is None:
                    collection[split] = collect_eligible_choices(
                        index, work, tokenizer, **common
                    )
                else:
                    collection[split] = collect_eligible_choices_parallel(
                        index,
                        work,
                        token_executor,
                        batch_size=args.token_worker_batch_size,
                        maximum_pending=max(args.token_workers * 4, 8),
                        **common,
                    )
                horizons[split] = assign_split_horizons(
                    work, split=split, seed=args.seed
                )
        finally:
            if token_executor is not None:
                token_executor.shutdown(wait=True, cancel_futures=True)

        outputs: dict[str, dict[str, object]] = {}
        materialized_identities: dict[str, dict[str, set[str]]] = {}
        for split in ALL_SPLITS:
            outputs[split], materialized_identities[split] = write_materialized_outputs(
                index,
                work,
                tokenizer,
                split=split,
                output_dir=output_dir,
                csv_path=paths[f"{split}_csv"],
                record_path=paths[f"{split}_records"],
                max_length=args.max_length,
            )
        source_hashes = write_source_hashes(
            index,
            set().union(
                *(value["sources"] for value in materialized_identities.values())
            ),
            paths["source_hashes"],
        )
        runtime_hours = (
            outputs["train"]["input_tokens"]
            / REFERENCE_TOKENS_PER_SECOND
            / 3600.0
            * (1.0 + REFERENCE_VALIDATION_TRAIN_RATIO)
        )
        manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "dataset_build_id": "dataset:"
            + hashlib.sha256(
                compact_json(
                    {
                        "index_sha256": index_status["database_sha256"],
                        "source_holdout": split_policy["heldout_sources"],
                        "seed": args.seed,
                        "outputs": {
                            split: outputs[split]["csv_sha256"] for split in ALL_SPLITS
                        },
                    }
                ).encode("utf-8")
            ).hexdigest()[:24],
            "generated_at_utc": utc_now(),
            "generator": "dataprocess/materialize_dataset_v4.py",
            "do_not_edit": "Regenerate from the canonical index; do not hand edit.",
            "canonical_index": {
                "path": str(index_path),
                "sha256": index_status["database_sha256"],
                "processed_graph_root": index_status["processed_graph_root"],
                "summary": index_status["summary"],
            },
            "processed_graph_root": index_status["processed_graph_root"],
            "record_schema_version": DATASET_SCHEMA_VERSION,
            "answer_schema_version": "pathway_continuation_v4",
            "primary_prompt_profile": PRIMARY_PROMPT_PROFILE,
            "prompt_policy": (
                "explicit known organism/source code in P0 question; no pathway provenance "
                "fields, headers, category, block, or phenotype in model-visible input or "
                "answer; biological entity/event text is retained even when its surface "
                "phrase coincides with a pathway title; complete rich action JSON"
            ),
            "entity_policy": (
                "one non-group KGML entry is one canonical participant; additional resolved IDs "
                "are aliases; groups expand to resolved members"
            ),
            "text_policy": (
                "deterministic event text from pinned Step12 templates with audited corrections; "
                "legacy_text retained in canonical records and exact-matched against each "
                "selected historical processed paragraph artifact"
            ),
            "split_policy": {
                "primary": "family-indivisible optimized 70/20/10 on seen organisms",
                "source_holdout": (
                    "dataset-internal coverage-quantile-stratified source-code holdout; "
                    "does not claim phylogenetic balance"
                ),
                "test_organism": "held-out source codes with train families",
                "test_strict": "held-out source codes with any non-train family",
                "details": paths["split_assignments"].name,
            },
            "phenotype_policy": "not_annotated; absent from model input and answer",
            "max_length": args.max_length,
            "train_token_budget": args.train_token_budget,
            "maximum_evaluation_records": args.maximum_evaluation_records,
            "recommended_sft_epochs": 1,
            "exploratory_max_sft_epochs": 5,
            "runtime_estimate": {
                "reference_four_a100_tokens_per_second": REFERENCE_TOKENS_PER_SECOND,
                "conservative_validation_train_ratio": REFERENCE_VALIDATION_TRAIN_RATIO,
                "estimated_one_epoch_hours": runtime_hours,
                "note": "replace with observed v4 packed-training throughput after the first run",
            },
            "csv_header": V4_CSV_FIELDNAMES,
            "outputs": {
                split: {
                    **outputs[split],
                    "csv_file": paths[f"{split}_csv"].name,
                    "records_file": paths[f"{split}_records"].name,
                }
                for split in ALL_SPLITS
            },
            "source_graph_hashes": source_hashes,
            "event_template": {
                "path": TEMPLATE_ASSET.relative_to(Path(__file__).resolve().parents[1]).as_posix(),
                "sha256": file_sha256(TEMPLATE_ASSET),
                "provenance": template_provenance(),
            },
        }
        atomic_json(paths["manifest"], manifest, readonly=True)
        audit = build_audit(
            index=index,
            index_status=index_status,
            split_policy=split_policy,
            collection=collection,
            horizons=horizons,
            outputs=outputs,
            identities=materialized_identities,
            paths=paths,
            source_hashes=source_hashes,
            processed_root=processed_root,
            max_length=args.max_length,
            train_token_budget=args.train_token_budget,
            minimum_train_records=args.minimum_train_records,
        )
        atomic_json(paths["audit"], audit, readonly=True)
        if audit["status"] != "passed":
            raise ValueError("v4 release audit failed: " + ", ".join(audit["failures"]))
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        work.close()
        index.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
