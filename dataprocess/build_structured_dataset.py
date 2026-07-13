#!/usr/bin/env python3
"""Build a capped, prefix-only v3 dataset directly from processed_graph JSON."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import stat
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from dataprocess.schemas import CSV_FIELDNAMES, canonical_pathway_family_id
from dataprocess.structured_schema import (
    SUBSTEP_SOURCE,
    chat_prompt,
    compact_json,
    csv_row,
    graph_events,
    graph_id_for_source,
    record_from_object,
    selected_prefix_lengths,
    total_training_tokens,
)
from dataprocess.structured_views import build_structured_records


DEFAULT_TEST_ORGANISMS = "tru,xtr,dre,gga,dmk,dme,cel"
LENGTH_BINS = ("02-04", "05-08", "09-16", "17-32", "33+")
SPLITS = ("train", "validation", "test")
OUTPUT_NAMES = {
    "validation_csv": "validation_pathway_continuation_v3.csv",
    "test_csv": "test_pathway_continuation_v3.csv",
    "train_records": "train_pathway_records_v3.jsonl",
    "validation_records": "validation_pathway_records_v3.jsonl",
    "test_records": "test_pathway_records_v3.jsonl",
    "manifest": "dataset_manifest.json",
    "audit": "data_audit.json",
}

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


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"coverage:{seed}:{value}".encode("utf-8")).hexdigest()


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
) -> str | None:
    if organism in test_organisms:
        return "test" if family in test_families else None
    if family in test_families:
        return None
    if family in validation_families:
        return "validation"
    return "train"


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
            layer_count INTEGER NOT NULL,
            rank TEXT NOT NULL,
            record_json TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX candidates_split_family ON candidates(split, family)")
    return connection


def scan_candidates(
    graph_root: Path,
    connection: sqlite3.Connection,
    *,
    test_organisms: set[str],
    test_families: set[str],
    validation_families: set[str],
    train_candidate_fraction: float,
    evaluation_candidate_fraction: float,
    seed: int,
    max_files: int,
    progress_every: int,
) -> dict[str, int]:
    stats: Counter[str] = Counter()
    for file_index, path in enumerate(iter_graph_files(graph_root), start=1):
        if max_files and file_index > max_files:
            break
        if progress_every and file_index % progress_every == 0:
            print(
                f"scanned_graph_files={file_index} candidate_records={stats['candidate_records']}",
                file=sys.stderr,
                flush=True,
            )
        relative, organism, family = source_identity(path, graph_root)
        split = assigned_split(
            organism,
            family,
            test_organisms=test_organisms,
            test_families=test_families,
            validation_families=validation_families,
        )
        stats["graph_files_scanned"] += 1
        if split is None:
            stats["graph_files_excluded_by_split"] += 1
            continue
        try:
            raw = path.read_bytes()
            graph = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            stats["invalid_graph_files"] += 1
            print(f"invalid_graph={relative} error={exc}", file=sys.stderr)
            continue
        graph_id = graph_id_for_source(relative, raw)
        try:
            structural_events, missing_endpoints = graph_events(graph)
        except (KeyError, TypeError, ValueError) as exc:
            stats["invalid_structural_graph_files"] += 1
            print(f"invalid_structural_graph={relative} error={exc}", file=sys.stderr)
            continue
        stats["graph_events_total"] += len(structural_events) + missing_endpoints
        stats["graph_missing_endpoint_events"] += missing_endpoints
        if missing_endpoints:
            stats["graphs_excluded_missing_endpoint_events"] += 1
            continue
        records = build_structured_records(
            graph,
            graph_id=graph_id,
            source_graph_json=relative,
            parsed_events=(structural_events, missing_endpoints),
        )
        stats["graphs_parsed"] += 1
        stats["views_built"] += len(records)
        if not structural_events:
            stats["graphs_without_structural_events"] += 1
        fraction = (
            train_candidate_fraction
            if split == "train"
            else evaluation_candidate_fraction
        )
        for record in records:
            if record.organism != organism or record.family != family:
                stats["views_excluded_source_identity_mismatch"] += 1
                continue
            if len(record.layers) < 2:
                stats["views_skipped_short"] += 1
                continue
            if stable_fraction(record.record_id, seed, f"candidate:{split}") >= fraction:
                stats[f"views_fraction_excluded_{split}"] += 1
                continue
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    record.record_id,
                    split,
                    record.family,
                    record.organism,
                    len(record.layers),
                    stable_rank(record.record_id, seed),
                    compact_json(record.record_object()),
                ),
            )
            stats["candidate_records"] += 1
            stats[f"candidate_records_{split}"] += 1
        if file_index % 1000 == 0:
            connection.commit()
    connection.commit()
    return dict(stats)


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
    for organism_rows in by_organism.values():
        ordered = sorted(
            organism_rows,
            key=lambda row: (row["rank"], row["record_id"]),
        )
        primary.append(ordered[0])
        remaining.extend(ordered[1:])
    return _round_robin_length(primary) + _round_robin_length(remaining)


def select_records(
    connection: sqlite3.Connection,
    split: str,
    maximum_per_family: int,
) -> list[dict[str, Any]]:
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
                "organism": organism,
                "layer_count": layer_count,
                "rank": rank,
                "record_json": record_json,
            }
            for record_id, organism, layer_count, rank, record_json in connection.execute(
                """
                SELECT record_id, organism, layer_count, rank, record_json
                FROM candidates WHERE split=? AND family=? ORDER BY rank
                """,
                (split, family),
            )
        ]
        selected.extend(diversity_order(rows)[:maximum_per_family])
    return sorted(selected, key=lambda row: row["record_id"])


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


def write_selected_split(
    selected: Sequence[dict[str, Any]],
    *,
    split: str,
    csv_path: Path,
    record_path: Path,
    tokenizer: Any,
    max_length: int,
    max_prefixes_per_train_record: int,
) -> dict[str, Any]:
    csv_temporary = csv_path.with_suffix(csv_path.suffix + ".tmp")
    records_temporary = record_path.with_suffix(record_path.suffix + ".tmp")
    stats: Counter[str] = Counter()
    token_lengths: list[int] = []
    accepted_families: set[str] = set()
    accepted_organisms: set[str] = set()
    accepted_sources: set[str] = set()
    with csv_temporary.open("w", encoding="utf-8", newline="") as csv_handle, records_temporary.open(
        "w", encoding="utf-8"
    ) as record_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for selected_row in selected:
            value = json.loads(selected_row["record_json"])
            record = record_from_object(value)
            maximum = max_prefixes_per_train_record if split == "train" else 1
            accepted_rows: list[dict[str, object]] = []
            for prefix_len in selected_prefix_lengths(len(record.layers), maximum):
                row = csv_row(record, prefix_len)
                # Generation code never sees these metadata columns; explicitly
                # reject accidental prompt leakage during materialization.
                question = str(row["question"])
                if any(
                    marker in question
                    for marker in (
                        "Organism:",
                        "KEGG pathway ID:",
                        "Pathway title:",
                        "Pathway block:",
                    )
                ):
                    raise ValueError(f"model-visible metadata leaked for {row['sample_id']}")
                json.loads(str(row["answer"]))
                token_count = total_training_tokens(tokenizer, row)
                if token_count > max_length:
                    stats["rows_dropped_token_budget"] += 1
                    continue
                accepted_rows.append(row)
                token_lengths.append(token_count)
            if not accepted_rows:
                stats["records_dropped_no_complete_json_sample"] += 1
                continue
            record_handle.write(compact_json(record.record_object()) + "\n")
            stats["records"] += 1
            accepted_families.add(record.family)
            accepted_organisms.add(record.organism)
            accepted_sources.add(record.source_graph_json)
            for row in accepted_rows:
                writer.writerow(row)
                stats["rows"] += 1
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
        "token_length": {
            "min": min(token_lengths),
            "mean": sum(token_lengths) / len(token_lengths),
            "max": max(token_lengths),
        },
        "csv_sha256": file_sha256(csv_path),
        "records_sha256": file_sha256(record_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--processed-graph-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--test-organisms", default=DEFAULT_TEST_ORGANISMS)
    parser.add_argument("--test-family-fraction", type=float, default=0.05)
    parser.add_argument("--validation-family-fraction", type=float, default=0.05)
    parser.add_argument("--train-candidate-record-fraction", type=float, default=0.003)
    parser.add_argument("--evaluation-candidate-record-fraction", type=float, default=1.0)
    parser.add_argument("--max-records-per-family", type=int, default=256)
    parser.add_argument("--max-prefixes-per-train-record", type=int, default=3)
    parser.add_argument(
        "--minimum-train-records",
        type=int,
        default=12000,
        help="Fail instead of releasing a training set too small for the planned full run.",
    )
    parser.add_argument("--reference-train-records", type=int, default=4609)
    parser.add_argument("--reference-run-hours", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260711)
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
    if args.max_records_per_family < 1:
        parser.error("--max-records-per-family must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    graph_root = Path(args.processed_graph_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not graph_root.is_dir():
        raise FileNotFoundError(f"processed_graph root does not exist: {graph_root}")
    paths = output_paths(output_dir, args.max_records_per_family)
    validate_outputs(paths, args.overwrite)

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
    test_families = choose_family_fraction(
        test_available_families,
        fraction=args.test_family_fraction,
        seed=args.seed,
        namespace="strict_test_family",
    )
    validation_candidates = non_test_available_families - test_families
    validation_families = choose_family_fraction(
        validation_candidates,
        fraction=args.validation_family_fraction,
        seed=args.seed,
        namespace="validation_family",
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
            train_candidate_fraction=args.train_candidate_record_fraction,
            evaluation_candidate_fraction=args.evaluation_candidate_record_fraction,
            seed=args.seed,
            max_files=args.max_files,
            progress_every=args.progress_every,
        )
        selected = {
            split: select_records(connection, split, args.max_records_per_family)
            for split in SPLITS
        }
    finally:
        connection.close()
        database_path.unlink(missing_ok=True)
        database_path.with_suffix(".sqlite3-wal").unlink(missing_ok=True)
        database_path.with_suffix(".sqlite3-shm").unlink(missing_ok=True)

    if scan_stats.get("invalid_graph_files", 0):
        raise ValueError(
            f"strict build rejected {scan_stats['invalid_graph_files']} invalid processed_graph files"
        )
    if scan_stats.get("invalid_structural_graph_files", 0):
        raise ValueError(
            "strict build rejected structurally invalid processed_graph files: "
            f"{scan_stats['invalid_structural_graph_files']}"
        )
    if scan_stats.get("views_excluded_source_identity_mismatch", 0):
        raise ValueError(
            "strict build found processed_graph metadata/path identity mismatches: "
            f"{scan_stats['views_excluded_source_identity_mismatch']} views"
        )

    if len(selected["train"]) < args.minimum_train_records:
        raise ValueError(
            f"selected train records={len(selected['train'])} is below "
            f"--minimum-train-records={args.minimum_train_records}; increase "
            "--train-candidate-record-fraction and rebuild"
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
        )
    if split_outputs["train"]["records"] < args.minimum_train_records:
        raise ValueError(
            f"accepted train records={split_outputs['train']['records']} is below "
            f"--minimum-train-records={args.minimum_train_records} after strict token filtering; "
            "increase --train-candidate-record-fraction and rebuild"
        )

    estimated_hours = (
        split_outputs["train"]["records"]
        / max(args.reference_train_records, 1)
        * args.reference_run_hours
    )
    build_identity_payload = {
        "schema_version": "structured_pathway_record_v3",
        "inventory_sha256": inventory["path_size_inventory_sha256"],
        "seed": args.seed,
        "test_organisms": sorted(test_organisms),
        "test_families": sorted(test_families),
        "validation_families": sorted(validation_families),
        "max_length": args.max_length,
        "max_records_per_family": args.max_records_per_family,
        "max_prefixes_per_train_record": args.max_prefixes_per_train_record,
        "split_hashes": {
            split: {
                "csv_sha256": split_outputs[split]["csv_sha256"],
                "records_sha256": split_outputs[split]["records_sha256"],
            }
            for split in SPLITS
        },
    }
    dataset_build_id = "dataset:" + hashlib.sha256(
        compact_json(build_identity_payload).encode("utf-8")
    ).hexdigest()[:24]
    manifest = {
        "schema_version": "chatpathway_dataset_manifest_v3",
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
        "inventory": inventory,
        "seed": args.seed,
        "test_organisms": sorted(test_organisms),
        "strict_test_families": sorted(test_families),
        "validation_families": sorted(validation_families),
        "split_policy": "strict organism-disjoint and pathway-family-disjoint test; family-disjoint validation",
        "prompt_policy": "prefix-only; no pathway name, class, id, title, block, or organism in model-visible prompt",
        "phenotype_policy": "not_annotated metadata only; absent from model input and target",
        "parser_source": SUBSTEP_SOURCE,
        "max_length": args.max_length,
        "train_candidate_record_fraction": args.train_candidate_record_fraction,
        "evaluation_candidate_record_fraction": args.evaluation_candidate_record_fraction,
        "max_records_per_family": args.max_records_per_family,
        "max_prefixes_per_train_record": args.max_prefixes_per_train_record,
        "scan": scan_stats,
        "splits": split_outputs,
        "runtime_estimate": {
            "reference_records": args.reference_train_records,
            "reference_run_hours": args.reference_run_hours,
            "estimated_run_hours_by_record_ratio": estimated_hours,
            "warning": "Linear record-count estimate only; confirm with measured supervised-token throughput.",
        },
    }
    temporary_manifest = paths["manifest"].with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(paths["manifest"])

    from dataprocess.audit_dataset_release import generate_release_audit

    generate_release_audit(
        train_path=paths["train_csv"],
        validation_path=paths["validation_csv"],
        test_path=paths["test_csv"],
        graph_root=graph_root,
        manifest_path=paths["manifest"],
        tokenizer=tokenizer,
        max_length=args.max_length,
        output_path=paths["audit"],
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
