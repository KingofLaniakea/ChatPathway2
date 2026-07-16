#!/usr/bin/env python3
"""Build the v4 canonical index with deterministic local SQLite shards.

The source corpus is inventoried once.  Source paths are assigned to stable
hash shards, each shard is parsed and written independently, and the verified
shards are merged on local storage before query indexes are created.  This
keeps the canonical single-database contract used by materialization while
removing the network-filesystem single-writer bottleneck from ingestion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import stat
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from dataprocess.index_structured_graphs_v4 import (
    INDEX_SCHEMA_VERSION,
    ScanTask,
    create_secondary_indexes,
    file_sha256,
    generator_contract,
    initialize_database,
    iter_graph_files,
    scan_graph,
    source_identity,
    summarize_database,
    write_status,
)
from dataprocess.structured_schema import compact_json


DEFAULT_SHARDS = 64
DEFAULT_WORKERS = 64
DEFAULT_COMMIT_EVERY = 5000
BUILD_SCHEMA_VERSION = "chatpathway_canonical_sharded_build_v4.1"
GRAPH_INSERT_SQL = "INSERT INTO graphs VALUES (" + ",".join("?" for _ in range(22)) + ")"
RECORD_INSERT_SQL = (
    "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


def atomic_json(path: Path, payload: dict[str, object], *, readonly: bool = False) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    path.chmod(0o444 if readonly else 0o644)


def stable_bucket(namespace: str, value: str, modulus: int) -> int:
    digest = hashlib.sha256(f"{namespace}\0{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def remove_sqlite_family(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
        if candidate.exists():
            candidate.unlink()


def check_free_space(path: Path, minimum_free_gb: float) -> int:
    free = shutil.disk_usage(path).free
    required = int(minimum_free_gb * 1024**3)
    if free < required:
        raise OSError(
            f"insufficient local scratch: free={free / 1024**3:.1f} GiB "
            f"required={minimum_free_gb:.1f} GiB path={path}"
        )
    return free


@dataclass(frozen=True)
class Inventory:
    total_graphs: int
    selected_graphs: int
    selected_path_manifest_sha256: str
    manifest_sha256: tuple[str, ...]
    manifest_graphs: tuple[int, ...]

    def object(self) -> dict[str, object]:
        return {
            "total_graphs": self.total_graphs,
            "selected_graphs": self.selected_graphs,
            "selected_path_manifest_sha256": self.selected_path_manifest_sha256,
            "manifest_sha256": list(self.manifest_sha256),
            "manifest_graphs": list(self.manifest_graphs),
        }

    @classmethod
    def from_object(cls, value: dict[str, object]) -> "Inventory":
        return cls(
            total_graphs=int(value["total_graphs"]),
            selected_graphs=int(value["selected_graphs"]),
            selected_path_manifest_sha256=str(
                value["selected_path_manifest_sha256"]
            ),
            manifest_sha256=tuple(str(item) for item in value["manifest_sha256"]),
            manifest_graphs=tuple(int(item) for item in value["manifest_graphs"]),
        )


def manifest_paths(manifest_dir: Path, shards: int) -> tuple[Path, ...]:
    return tuple(manifest_dir / f"shard_{index:03d}.txt" for index in range(shards))


def build_inventory(
    graph_root: Path,
    manifest_dir: Path,
    *,
    shards: int,
    sample_denominator: int,
    sample_residue: int,
    rebuild: bool,
) -> Inventory:
    status_path = manifest_dir / "inventory.json"
    paths = manifest_paths(manifest_dir, shards)
    if status_path.is_file() and not rebuild:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status.get("complete") is True
            and status.get("graph_root") == str(graph_root)
            and int(status.get("shards", 0)) == shards
            and int(status.get("sample_denominator", 0)) == sample_denominator
            and int(status.get("sample_residue", -1)) == sample_residue
            and all(path.is_file() for path in paths)
        ):
            inventory = Inventory.from_object(status["inventory"])
            observed_hashes = tuple(file_sha256(path) for path in paths)
            if observed_hashes != inventory.manifest_sha256:
                raise ValueError("saved shard manifests no longer match inventory.json")
            return inventory
        raise ValueError(
            "existing inventory belongs to another build; use a new work directory "
            "or pass --rebuild-inventory"
        )

    manifest_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = tuple(path.with_suffix(".txt.tmp") for path in paths)
    for path in temporary_paths:
        if path.exists():
            path.unlink()
    handles = [path.open("w", encoding="utf-8") for path in temporary_paths]
    total_graphs = 0
    selected_graphs = 0
    selected_path_digest = hashlib.sha256()
    counts = [0] * shards
    try:
        for path in iter_graph_files(graph_root):
            relative, _, _ = source_identity(path, graph_root)
            total_graphs += 1
            if stable_bucket("sample", relative, sample_denominator) != sample_residue:
                if total_graphs % 100000 == 0:
                    print(
                        f"stage=inventory graph_paths={total_graphs} "
                        f"selected={selected_graphs}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            shard = stable_bucket("shard", relative, shards)
            handles[shard].write(relative + "\n")
            counts[shard] += 1
            selected_graphs += 1
            selected_path_digest.update(f"{relative}\n".encode("utf-8"))
            if total_graphs % 100000 == 0:
                print(
                    f"stage=inventory graph_paths={total_graphs} "
                    f"selected={selected_graphs}",
                    file=sys.stderr,
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.close()
    if selected_graphs < 1:
        raise ValueError("stable sample selected no graph files")
    for temporary, final in zip(temporary_paths, paths):
        temporary.replace(final)
    inventory = Inventory(
        total_graphs=total_graphs,
        selected_graphs=selected_graphs,
        selected_path_manifest_sha256=selected_path_digest.hexdigest(),
        manifest_sha256=tuple(file_sha256(path) for path in paths),
        manifest_graphs=tuple(counts),
    )
    atomic_json(
        status_path,
        {
            "schema_version": BUILD_SCHEMA_VERSION,
            "complete": True,
            "graph_root": str(graph_root),
            "shards": shards,
            "sample_denominator": sample_denominator,
            "sample_residue": sample_residue,
            "inventory": inventory.object(),
        },
        readonly=True,
    )
    return inventory


def iter_manifest(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            relative = line.rstrip("\n")
            if relative:
                yield relative


@dataclass(frozen=True)
class ShardJob:
    shard_id: int
    manifest_path: str
    manifest_sha256: str
    expected_graphs: int
    graph_root: str
    processed_root: str | None
    shard_dir: str
    seed: int
    contract: dict[str, object]
    commit_every: int
    progress_every: int


def shard_paths(shard_dir: Path, shard_id: int) -> tuple[Path, Path]:
    stem = f"canonical_shard_{shard_id:03d}"
    return shard_dir / f"{stem}.sqlite3", shard_dir / f"{stem}.json"


def completed_shard(job: ShardJob) -> dict[str, object] | None:
    database, status_path = shard_paths(Path(job.shard_dir), job.shard_id)
    if not database.is_file() or not status_path.is_file():
        return None
    status = json.loads(status_path.read_text(encoding="utf-8"))
    expected = {
        "complete": True,
        "shard_id": job.shard_id,
        "manifest_sha256": job.manifest_sha256,
        "expected_graphs": job.expected_graphs,
        "contract_sha256": job.contract["contract_sha256"],
    }
    if any(status.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"completed shard {job.shard_id} does not match the current build contract"
        )
    if database.stat().st_size != int(status["database_bytes"]):
        raise ValueError(f"completed shard {job.shard_id} changed size")
    return status


def build_shard(job: ShardJob) -> dict[str, object]:
    prior = completed_shard(job)
    if prior is not None:
        return prior
    shard_dir = Path(job.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    database, status_path = shard_paths(shard_dir, job.shard_id)
    partial = database.with_suffix(".sqlite3.partial")
    remove_sqlite_family(partial)
    graph_root = Path(job.graph_root)
    processed_root = Path(job.processed_root) if job.processed_root else None
    connection = initialize_database(
        partial,
        job.contract,
        create_query_indexes=False,
    )
    started = time.monotonic()
    graphs = 0
    records = 0
    try:
        connection.execute("PRAGMA wal_autocheckpoint=100000")
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value_json) VALUES('shard_contract', ?)",
            (
                compact_json(
                    {
                        "shard_id": job.shard_id,
                        "manifest_sha256": job.manifest_sha256,
                        "expected_graphs": job.expected_graphs,
                    }
                ),
            ),
        )
        for relative in iter_manifest(Path(job.manifest_path)):
            path = graph_root / relative
            relative_check, organism, family = source_identity(path, graph_root)
            if relative_check != relative:
                raise ValueError(f"manifest path escaped graph root: {relative}")
            task = ScanTask(
                str(path),
                str(processed_root / relative) if processed_root is not None else None,
                relative,
                organism,
                family,
                job.seed,
            )
            result = scan_graph(task)
            connection.execute(GRAPH_INSERT_SQL, result.graph_row)
            if result.record_rows:
                connection.executemany(RECORD_INSERT_SQL, result.record_rows)
            graphs += 1
            records += len(result.record_rows)
            if graphs % job.commit_every == 0:
                connection.commit()
            if job.progress_every and graphs % job.progress_every == 0:
                elapsed = max(time.monotonic() - started, 1e-9)
                print(
                    f"shard={job.shard_id:03d} graphs={graphs}/{job.expected_graphs} "
                    f"records={records} graphs_per_second={graphs / elapsed:.2f}",
                    file=sys.stderr,
                    flush=True,
                )
        connection.commit()
        if graphs != job.expected_graphs:
            raise ValueError(
                f"shard {job.shard_id} graph mismatch: actual={graphs} "
                f"expected={job.expected_graphs}"
            )
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        foreign_key_failures = connection.execute("PRAGMA foreign_key_check").fetchall()
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if foreign_key_failures or quick_check != "ok":
            raise ValueError(
                f"shard {job.shard_id} integrity failure: "
                f"foreign_keys={len(foreign_key_failures)} quick_check={quick_check}"
            )
        summary = summarize_database(connection)
    finally:
        connection.close()
    partial.replace(database)
    elapsed_seconds = time.monotonic() - started
    status: dict[str, object] = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "complete": True,
        "shard_id": job.shard_id,
        "manifest_sha256": job.manifest_sha256,
        "expected_graphs": job.expected_graphs,
        "contract_sha256": job.contract["contract_sha256"],
        "database": database.name,
        "database_bytes": database.stat().st_size,
        "database_sha256": file_sha256(database),
        "elapsed_seconds": elapsed_seconds,
        "graphs_per_second": graphs / max(elapsed_seconds, 1e-9),
        "summary": summary,
    }
    atomic_json(status_path, status, readonly=True)
    return status


def validate_shard_hashes(
    shard_dir: Path, jobs: Sequence[ShardJob]
) -> tuple[dict[str, object], ...]:
    statuses = []
    for job in jobs:
        status = completed_shard(job)
        if status is None:
            raise ValueError(f"shard {job.shard_id} is incomplete")
        database, _ = shard_paths(shard_dir, job.shard_id)
        observed = file_sha256(database)
        if observed != status["database_sha256"]:
            raise ValueError(f"shard {job.shard_id} database hash mismatch")
        statuses.append(status)
    return tuple(statuses)


def attach_insert(
    connection: sqlite3.Connection,
    shard_path: Path,
    *,
    table: str,
    order_by: str,
) -> None:
    connection.execute("ATTACH DATABASE ? AS sharddb", (str(shard_path),))
    try:
        connection.execute(
            f"INSERT INTO main.{table} SELECT * FROM sharddb.{table} ORDER BY {order_by}"
        )
        connection.commit()
    finally:
        connection.execute("DETACH DATABASE sharddb")


def merged_path_size_inventory(
    connection: sqlite3.Connection,
) -> tuple[int, str]:
    """Derive the strict source inventory after each source was actually read."""

    graph_bytes = 0
    digest = hashlib.sha256()
    for relative, file_size in connection.execute(
        "SELECT source_graph_json, file_size FROM graphs ORDER BY source_graph_json"
    ):
        size = int(file_size)
        graph_bytes += size
        digest.update(f"{relative}\t{size}\n".encode("utf-8"))
    return graph_bytes, digest.hexdigest()


def merge_shards(
    *,
    graph_root: Path,
    processed_root: Path | None,
    output_dir: Path,
    shard_dir: Path,
    jobs: Sequence[ShardJob],
    inventory: Inventory,
    contract: dict[str, object],
    minimum_free_gb: float,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    database = output_dir / "canonical_index_v4.sqlite3"
    status_path = output_dir / "index_status.json"
    if database.is_file() and status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status.get("complete") is True
            and status.get("generator_contract", {}).get("contract_sha256")
            == contract["contract_sha256"]
            and status.get("inventory", {}).get("graph_files")
            == inventory.selected_graphs
            and database.stat().st_size == int(status["database_bytes"])
            and file_sha256(database) == status["database_sha256"]
        ):
            return status
        raise ValueError("existing merged index is not the current complete build")
    check_free_space(output_dir, minimum_free_gb)
    validate_shard_hashes(shard_dir, jobs)
    partial = output_dir / "canonical_index_v4.sqlite3.partial"
    remove_sqlite_family(partial)
    connection = initialize_database(partial, contract, create_query_indexes=False)
    started = time.monotonic()
    try:
        input_roots = compact_json(
            {
                "processed_graph_root": str(graph_root),
                "processed_root": str(processed_root) if processed_root is not None else None,
            }
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value_json) VALUES('input_roots', ?)",
            (input_roots,),
        )
        for job in jobs:
            shard, _ = shard_paths(shard_dir, job.shard_id)
            attach_insert(
                connection,
                shard,
                table="graphs",
                order_by="source_graph_json",
            )
        for job in jobs:
            shard, _ = shard_paths(shard_dir, job.shard_id)
            attach_insert(
                connection,
                shard,
                table="records",
                order_by="source_graph_json, record_id",
            )
        merged_graph_bytes, merged_inventory_sha256 = merged_path_size_inventory(
            connection
        )
        connection.execute(
            "INSERT OR REPLACE INTO meta(key, value_json) VALUES('source_inventory', ?)",
            (
                compact_json(
                    {
                        "graph_files": inventory.selected_graphs,
                        "graph_bytes": merged_graph_bytes,
                        "path_size_inventory_sha256": merged_inventory_sha256,
                    }
                ),
            ),
        )
        connection.commit()
        create_secondary_indexes(connection)
        connection.execute("ANALYZE")
        connection.commit()
        foreign_key_failures = connection.execute("PRAGMA foreign_key_check").fetchall()
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        summary = summarize_database(connection)
        if foreign_key_failures or quick_check != "ok":
            raise ValueError(
                f"merged integrity failure: foreign_keys={len(foreign_key_failures)} "
                f"quick_check={quick_check}"
            )
        if summary["graphs"] != inventory.selected_graphs:
            raise ValueError(
                f"merged graph mismatch: actual={summary['graphs']} "
                f"expected={inventory.selected_graphs}"
            )
        if summary["records"] != summary["records_declared_by_graphs"]:
            raise ValueError("merged record count disagrees with graph declarations")
        if summary["graph_bytes"] != merged_graph_bytes:
            raise ValueError("merged graph bytes disagree with source inventory")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        connection.close()
    partial.replace(database)
    inventory_counter: Counter[str] = Counter(
        {
            "graph_files": inventory.selected_graphs,
            "graph_bytes": merged_graph_bytes,
            "graph_files_total_before_sampling": inventory.total_graphs,
        }
    )
    write_status(
        status_path,
        graph_root=graph_root,
        processed_root=processed_root,
        database_path=database,
        contract=contract,
        inventory=inventory_counter,
        inventory_sha256=merged_inventory_sha256,
        summary=summary,
        complete=True,
    )
    database.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status["sharded_build"] = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "shards": len(jobs),
        "merge_elapsed_seconds": time.monotonic() - started,
        "manifest_sha256": list(inventory.manifest_sha256),
    }
    status_path.chmod(0o644)
    atomic_json(status_path, status, readonly=True)
    return status


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--processed-graph-root", required=True)
    parser.add_argument("--processed-root")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shards", type=int, default=DEFAULT_SHARDS)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--commit-every", type=int, default=DEFAULT_COMMIT_EVERY)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--minimum-free-gb", type=float, default=150.0)
    parser.add_argument("--sample-denominator", type=int, default=1)
    parser.add_argument("--sample-residue", type=int, default=0)
    parser.add_argument("--rebuild-inventory", action="store_true")
    args = parser.parse_args(argv)
    if args.shards < 1 or args.workers < 1:
        parser.error("--shards and --workers must be positive")
    if args.commit_every < 1 or args.progress_every < 0:
        parser.error("commit/progress intervals are invalid")
    if args.minimum_free_gb <= 0:
        parser.error("--minimum-free-gb must be positive")
    if args.sample_denominator < 1:
        parser.error("--sample-denominator must be positive")
    if not 0 <= args.sample_residue < args.sample_denominator:
        parser.error("--sample-residue must be within the sample denominator")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    graph_root = Path(args.processed_graph_root).expanduser().resolve()
    processed_root = (
        Path(args.processed_root).expanduser().resolve() if args.processed_root else None
    )
    work_dir = Path(args.work_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not graph_root.is_dir():
        raise FileNotFoundError(f"processed_graph root does not exist: {graph_root}")
    if processed_root is not None and not processed_root.is_dir():
        raise FileNotFoundError(f"processed root does not exist: {processed_root}")
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    free_at_start = check_free_space(work_dir, args.minimum_free_gb)
    contract = generator_contract()
    build_contract = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "processed_graph_root": str(graph_root),
        "processed_root": str(processed_root) if processed_root is not None else None,
        "shards": args.shards,
        "seed": args.seed,
        "sample_denominator": args.sample_denominator,
        "sample_residue": args.sample_residue,
        "generator_contract_sha256": contract["contract_sha256"],
    }
    contract_path = work_dir / "build_contract.json"
    if contract_path.is_file():
        observed = json.loads(contract_path.read_text(encoding="utf-8"))
        if observed != build_contract:
            raise ValueError("work directory belongs to a different sharded build")
    else:
        atomic_json(contract_path, build_contract, readonly=True)

    started = time.monotonic()
    inventory = build_inventory(
        graph_root,
        work_dir / "manifests",
        shards=args.shards,
        sample_denominator=args.sample_denominator,
        sample_residue=args.sample_residue,
        rebuild=args.rebuild_inventory,
    )
    inventory_elapsed = time.monotonic() - started
    shard_dir = work_dir / "shards"
    jobs = tuple(
        ShardJob(
            shard_id=index,
            manifest_path=str(manifest_paths(work_dir / "manifests", args.shards)[index]),
            manifest_sha256=inventory.manifest_sha256[index],
            expected_graphs=inventory.manifest_graphs[index],
            graph_root=str(graph_root),
            processed_root=str(processed_root) if processed_root is not None else None,
            shard_dir=str(shard_dir),
            seed=args.seed,
            contract=contract,
            commit_every=args.commit_every,
            progress_every=args.progress_every,
        )
        for index in range(args.shards)
    )
    print(
        f"stage=sharded_index inventory_graphs={inventory.selected_graphs} "
        f"shards={args.shards} workers={args.workers} "
        f"local_free_gib={free_at_start / 1024**3:.1f}",
        file=sys.stderr,
        flush=True,
    )
    shard_started = time.monotonic()
    complete = 0
    complete_graphs = 0
    if args.workers == 1:
        completed = ((job, build_shard(job)) for job in jobs)
        for job, status in completed:
            complete += 1
            complete_graphs += int(status["summary"]["graphs"])
            print(
                f"shard_complete={complete}/{args.shards} shard={job.shard_id:03d} "
                f"graphs_complete={complete_graphs}/{inventory.selected_graphs}",
                file=sys.stderr,
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, args.shards)) as executor:
            futures = {executor.submit(build_shard, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                status = future.result()
                complete += 1
                complete_graphs += int(status["summary"]["graphs"])
                print(
                    f"shard_complete={complete}/{args.shards} shard={job.shard_id:03d} "
                    f"graphs_complete={complete_graphs}/{inventory.selected_graphs}",
                    file=sys.stderr,
                    flush=True,
                )
    shard_elapsed = time.monotonic() - shard_started
    check_free_space(output_dir, args.minimum_free_gb)
    status = merge_shards(
        graph_root=graph_root,
        processed_root=processed_root,
        output_dir=output_dir,
        shard_dir=shard_dir,
        jobs=jobs,
        inventory=inventory,
        contract=contract,
        minimum_free_gb=args.minimum_free_gb,
    )
    total_elapsed = time.monotonic() - started
    metrics = {
        "schema_version": BUILD_SCHEMA_VERSION,
        "complete": True,
        "inventory": inventory.object(),
        "inventory_elapsed_seconds": inventory_elapsed,
        "shard_elapsed_seconds": shard_elapsed,
        "merge_elapsed_seconds": status["sharded_build"]["merge_elapsed_seconds"],
        "total_elapsed_seconds": total_elapsed,
        "selected_graphs_per_second_excluding_inventory": inventory.selected_graphs
        / max(shard_elapsed, 1e-9),
        "free_gib_at_start": free_at_start / 1024**3,
        "index_status_sha256": file_sha256(output_dir / "index_status.json"),
    }
    atomic_json(output_dir / "sharded_build_metrics.json", metrics, readonly=True)
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
