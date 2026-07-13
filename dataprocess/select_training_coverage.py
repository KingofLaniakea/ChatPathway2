#!/usr/bin/env python3
"""Build a family-capped, organism/length-diverse optimization subset.

The input is the small record-balanced eligible-prefix table, not the 171 GiB
full prefix expansion. Entire pathway-family groups are first reserved for a
fixed validation file. Within each remaining family, records receive a stable
order that prioritizes distinct organisms and round-robins trajectory-length
bins. Increasing ``--max-records-per-family`` therefore expands a deterministic
family-balanced training set without changing the validation groups.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    from dataprocess.audit_pathway_csv import audit_file
except ImportError:  # Allows direct script execution from dataprocess/.
    from audit_pathway_csv import audit_file  # type: ignore


DEFAULT_INPUT = "../data/train_kegg_pathway_record_balanced_0p1pct.csv"
DEFAULT_TRAIN_OUTPUT = "../data/train_kegg_pathway_coverage_cap32.csv"
DEFAULT_VALIDATION_OUTPUT = "../data/validation_kegg_pathway_family.csv"
LENGTH_BINS = ("02-04", "05-08", "09-16", "17-32", "33+")

csv.field_size_limit(sys.maxsize)


@dataclass(frozen=True)
class RecordRows:
    record_id: str
    family: str
    organism: str
    total_steps: int
    rows: tuple[dict[str, str], ...]
    source_order: int

    @property
    def length_bin(self) -> str:
        value = self.total_steps
        if value <= 4:
            return "02-04"
        if value <= 8:
            return "05-08"
        if value <= 16:
            return "09-16"
        if value <= 32:
            return "17-32"
        return "33+"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root() / path).resolve()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def stable_rank(record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:coverage:{record_id}".encode("utf-8")).hexdigest()


def _consistent(group: Sequence[dict[str, str]], field: str, record_id: str) -> str:
    values = {str(row.get(field, "")).strip() for row in group}
    if len(values) != 1 or not next(iter(values), ""):
        raise ValueError(f"record {record_id!r} has inconsistent or empty {field}")
    return next(iter(values))


def read_record_groups(path: Path) -> tuple[list[str], list[RecordRows]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    source_order: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        required = {
            "record_id",
            "sample_id",
            "pathway_family_id",
            "organism",
            "total_step",
            "prefix_step_count",
        }
        missing = required - set(fieldnames)
        if missing:
            raise ValueError(f"input CSV missing fields: {', '.join(sorted(missing))}")
        for row_index, row in enumerate(reader):
            record_id = str(row.get("record_id", "")).strip()
            if not record_id:
                raise ValueError(f"row {row_index + 2} has empty record_id")
            if record_id not in grouped:
                source_order[record_id] = len(source_order)
            grouped.setdefault(record_id, []).append(dict(row))

    records: list[RecordRows] = []
    for record_id, rows in grouped.items():
        family = _consistent(rows, "pathway_family_id", record_id)
        organism = _consistent(rows, "organism", record_id)
        total_step_text = _consistent(rows, "total_step", record_id)
        try:
            total_steps = int(total_step_text)
        except ValueError as exc:
            raise ValueError(
                f"record {record_id!r} has invalid total_step={total_step_text!r}"
            ) from exc
        if total_steps < 0:
            raise ValueError(f"record {record_id!r} has negative total_step")
        prefixes = [int(row["prefix_step_count"]) for row in rows]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError(f"record {record_id!r} has duplicate prefix rows")
        ordered_rows = tuple(
            sorted(rows, key=lambda row: (int(row["prefix_step_count"]), row["sample_id"]))
        )
        records.append(
            RecordRows(
                record_id=record_id,
                family=family,
                organism=organism,
                total_steps=total_steps,
                rows=ordered_rows,
                source_order=source_order[record_id],
            )
        )
    if not records:
        raise ValueError("input CSV has no records")
    return fieldnames, records


def validation_families(
    families: set[str],
    *,
    fraction: float,
    seed: int,
) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError("validation fraction must be in (0, 1)")
    if len(families) < 2:
        raise ValueError("validation split needs at least two pathway families")
    selected = {
        family for family in families if stable_fraction(family, seed) < fraction
    }
    if not selected or selected == families:
        ranked = sorted(families, key=lambda family: (stable_fraction(family, seed), family))
        count = min(max(1, round(len(ranked) * fraction)), len(ranked) - 1)
        selected = set(ranked[:count])
    return selected


def _round_robin_length(records: Iterable[RecordRows], seed: int) -> list[RecordRows]:
    buckets: dict[str, list[RecordRows]] = defaultdict(list)
    for record in records:
        buckets[record.length_bin].append(record)
    for bucket in buckets.values():
        bucket.sort(key=lambda record: (stable_rank(record.record_id, seed), record.record_id))

    ordered: list[RecordRows] = []
    position = 0
    while True:
        added = False
        for length_bin in LENGTH_BINS:
            bucket = buckets.get(length_bin, ())
            if position < len(bucket):
                ordered.append(bucket[position])
                added = True
        if not added:
            break
        position += 1
    return ordered


def family_diversity_order(records: Sequence[RecordRows], seed: int) -> list[RecordRows]:
    """Return a cap-independent order: unique organisms first, length-balanced."""

    by_organism: dict[str, list[RecordRows]] = defaultdict(list)
    for record in records:
        by_organism[record.organism].append(record)
    primary: list[RecordRows] = []
    remaining: list[RecordRows] = []
    for organism_records in by_organism.values():
        ranked = sorted(
            organism_records,
            key=lambda record: (stable_rank(record.record_id, seed), record.record_id),
        )
        primary.append(ranked[0])
        remaining.extend(ranked[1:])
    return _round_robin_length(primary, seed) + _round_robin_length(remaining, seed)


def select_optimization_records(
    records: Sequence[RecordRows],
    *,
    maximum_per_family: int,
    seed: int,
) -> list[RecordRows]:
    if maximum_per_family < 1:
        raise ValueError("maximum_per_family must be positive")
    by_family: dict[str, list[RecordRows]] = defaultdict(list)
    for record in records:
        by_family[record.family].append(record)
    selected: list[RecordRows] = []
    for family in sorted(by_family):
        selected.extend(
            family_diversity_order(by_family[family], seed)[:maximum_per_family]
        )
    return sorted(selected, key=lambda record: record.source_order)


def write_records(
    path: Path,
    fieldnames: Sequence[str],
    records: Sequence[RecordRows],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerows(record.rows)
    temporary.replace(path)


def summary(records: Sequence[RecordRows]) -> dict[str, object]:
    family_counts = Counter(record.family for record in records)
    length_counts = Counter(record.length_bin for record in records)
    return {
        "records": len(records),
        "rows": sum(len(record.rows) for record in records),
        "families": len(family_counts),
        "organisms": len({record.organism for record in records}),
        "family_record_min": min(family_counts.values()) if family_counts else 0,
        "family_record_max": max(family_counts.values()) if family_counts else 0,
        "length_bins": {key: length_counts.get(key, 0) for key in LENGTH_BINS},
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--validation-output", default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--report")
    parser.add_argument("--max-records-per-family", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.max_records_per_family < 1:
        parser.error("--max-records-per-family must be positive")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be in (0, 1)")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = resolve(args.input)
    train_output = resolve(args.train_output)
    validation_output = resolve(args.validation_output)
    report_output = resolve(args.report) if args.report else train_output.with_suffix(".report.json")
    for path in (train_output, validation_output, report_output):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames, records = read_record_groups(source)
    heldout_families = validation_families(
        {record.family for record in records},
        fraction=args.validation_fraction,
        seed=args.seed,
    )
    validation_records = [
        record for record in records if record.family in heldout_families
    ]
    optimization_candidates = [
        record for record in records if record.family not in heldout_families
    ]
    selected_records = select_optimization_records(
        optimization_candidates,
        maximum_per_family=args.max_records_per_family,
        seed=args.seed,
    )
    write_records(train_output, fieldnames, selected_records)
    write_records(validation_output, fieldnames, validation_records)

    train_audit = audit_file(train_output, max_errors=100)
    validation_audit = audit_file(validation_output, max_errors=100)
    family_overlap = train_audit.pathway_families & validation_audit.pathway_families
    errors = train_audit.errors + validation_audit.errors
    if family_overlap:
        errors.append(f"train/validation family overlap: {sorted(family_overlap)[:20]}")
    report = {
        "format_version": 1,
        "profile": "family_capped_organism_length_diverse_v1",
        "input": str(source),
        "input_sha256": file_sha256(source),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "validation_families": sorted(heldout_families),
        "max_records_per_family": args.max_records_per_family,
        "input_summary": summary(records),
        "optimization_candidate_summary": summary(optimization_candidates),
        "train_summary": summary(selected_records),
        "validation_summary": summary(validation_records),
        "train_output": str(train_output),
        "train_sha256": file_sha256(train_output),
        "validation_output": str(validation_output),
        "validation_sha256": file_sha256(validation_output),
        "audit_errors": errors,
        "train_record_ids": [record.record_id for record in selected_records],
    }
    temporary_report = report_output.with_suffix(report_output.suffix + ".tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_report.replace(report_output)
    if errors:
        raise SystemExit(f"coverage-selected outputs failed audit; inspect {report_output}")
    console_report = {
        key: value for key, value in report.items() if key != "train_record_ids"
    }
    console_report["train_record_id_count"] = len(report["train_record_ids"])
    print(json.dumps(console_report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
