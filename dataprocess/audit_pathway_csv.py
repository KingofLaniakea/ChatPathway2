#!/usr/bin/env python3
"""Stream-audit ChatPathway train/test CSVs before model training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from dataprocess.schemas import (
        CSV_FIELDNAMES,
        QUESTION_TYPE,
        canonical_pathway_family_id,
    )
except ImportError:  # Allows: python dataprocess/audit_pathway_csv.py
    from schemas import (  # type: ignore
        CSV_FIELDNAMES,
        QUESTION_TYPE,
        canonical_pathway_family_id,
    )


NULL_PHENOTYPE_STATUSES = {
    "not_annotated",
    "ambiguous_file_level",
    "conflict",
    "source_error",
}

csv.field_size_limit(sys.maxsize)


@dataclass
class FileAudit:
    path: str
    rows: int = 0
    sources: set[str] = field(default_factory=set, repr=False)
    records: set[str] = field(default_factory=set, repr=False)
    samples: set[str] = field(default_factory=set, repr=False)
    pathway_families: set[str] = field(default_factory=set, repr=False)
    organisms: Counter[str] = field(default_factory=Counter)
    phenotype_statuses: Counter[str] = field(default_factory=Counter)
    target_step_counts: Counter[int] = field(default_factory=Counter)
    question_char_lengths: Counter[int] = field(default_factory=Counter, repr=False)
    answer_char_lengths: Counter[int] = field(default_factory=Counter, repr=False)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str, max_errors: int) -> None:
        if len(self.errors) < max_errors:
            self.errors.append(message)

    def report(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "rows": self.rows,
            "unique_sources": len(self.sources),
            "unique_records": len(self.records),
            "unique_samples": len(self.samples),
            "unique_pathway_families": len(self.pathway_families),
            "organisms": dict(self.organisms.most_common()),
            "phenotype_statuses": dict(self.phenotype_statuses.most_common()),
            "target_step_counts": {str(key): value for key, value in sorted(self.target_step_counts.items())},
            "question_char_lengths": length_summary(self.question_char_lengths),
            "answer_char_lengths": length_summary(self.answer_char_lengths),
            "errors": self.errors,
            "warnings": self.warnings,
        }


def length_summary(histogram: Counter[int]) -> dict[str, float | int]:
    count = sum(histogram.values())
    if not count:
        return {"count": 0, "min": 0, "mean": 0.0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(histogram.items())

    def percentile(fraction: float) -> int:
        threshold = max(1, round(count * fraction))
        cumulative = 0
        for length, frequency in ordered:
            cumulative += frequency
            if cumulative >= threshold:
                return length
        return ordered[-1][0]

    total = sum(length * frequency for length, frequency in ordered)
    return {
        "count": count,
        "min": ordered[0][0],
        "mean": total / count,
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1][0],
    }


def expected_record_id(row: dict[str, str]) -> str:
    identity = "\n".join(
        (
            row.get("organism", ""),
            row.get("source_json", ""),
            row.get("pathway_id", ""),
            row.get("pathway_block", ""),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def parse_int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is not an integer: {value!r}") from exc


def validate_answer(row: dict[str, str]) -> tuple[int, str]:
    answer = json.loads(row.get("answer", ""))
    if not isinstance(answer, dict) or set(answer) != {"remaining_steps", "predicted_phenotype"}:
        raise ValueError("answer must contain exactly remaining_steps and predicted_phenotype")
    steps = answer["remaining_steps"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("remaining_steps must be a non-empty list")
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ValueError(f"remaining_steps[{index}] is not an object")
        if not isinstance(step.get("step"), int):
            raise ValueError(f"remaining_steps[{index}].step is not an integer")
        if not isinstance(step.get("layer"), str) or not step["layer"].strip():
            raise ValueError(f"remaining_steps[{index}].layer is empty")
        substeps = step.get("substeps")
        if not isinstance(substeps, list) or not substeps:
            raise ValueError(f"remaining_steps[{index}].substeps must be a non-empty list")
        for substep_index, substep in enumerate(substeps):
            if not isinstance(substep, dict):
                raise ValueError(f"remaining_steps[{index}].substeps[{substep_index}] is not an object")
            if not isinstance(substep.get("substep"), int):
                raise ValueError(f"remaining_steps[{index}].substeps[{substep_index}].substep is not an integer")
            if not isinstance(substep.get("text"), str) or not substep["text"].strip():
                raise ValueError(f"remaining_steps[{index}].substeps[{substep_index}].text is empty")

    status = row.get("phenotype_status", "")
    phenotype_column = row.get("phenotype", "").strip()
    phenotype = answer["predicted_phenotype"]
    if status == "available":
        if not phenotype_column:
            raise ValueError("available phenotype has an empty phenotype column")
        if not isinstance(phenotype, dict) or not str(phenotype.get("text", "")).strip():
            raise ValueError("available phenotype is missing from answer JSON")
    elif status in NULL_PHENOTYPE_STATUSES:
        if phenotype is not None:
            raise ValueError(f"phenotype_status={status} requires predicted_phenotype=null")
        if phenotype_column:
            raise ValueError(f"phenotype_status={status} requires an empty phenotype column")
    else:
        raise ValueError(f"unknown phenotype_status={status!r}")
    return len(steps), status


def audit_file(path: Path, max_errors: int) -> FileAudit:
    audit = FileAudit(path=str(path.resolve()))
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in CSV_FIELDNAMES if field not in (reader.fieldnames or [])]
        if missing:
            audit.add_error(f"missing columns: {', '.join(missing)}", max_errors)
            return audit

        for line_number, row in enumerate(reader, start=2):
            audit.rows += 1
            prefix = f"line {line_number}"
            try:
                if row.get("question_type") != QUESTION_TYPE:
                    raise ValueError(f"question_type must be {QUESTION_TYPE!r}")
                if row.get("substep_schema_version") != "layer_set_v1":
                    raise ValueError("substep_schema_version must be 'layer_set_v1'")
                if not row.get("substep_source", "").strip():
                    raise ValueError("substep_source must be non-empty")
                source = row.get("source_json", "").strip()
                record = row.get("record_id", "").strip()
                sample = row.get("sample_id", "").strip()
                if not source or not record or not sample:
                    raise ValueError("source_json, record_id, and sample_id must be non-empty")
                if record != expected_record_id(row):
                    raise ValueError("record_id does not match its source identity fields")
                prefix_count = parse_int(row.get("prefix_step_count", ""), "prefix_step_count")
                if sample != f"{record}:prefix={prefix_count}":
                    raise ValueError("sample_id does not match record_id and prefix_step_count")
                pathway_family = row.get("pathway_family_id", "").strip()
                expected_family = canonical_pathway_family_id(row.get("pathway_id", ""))
                if pathway_family != expected_family:
                    raise ValueError(
                        "pathway_family_id does not match the canonical KEGG pathway family"
                    )
                if sample in audit.samples:
                    raise ValueError(f"duplicate sample_id {sample}")
                target_steps, status = validate_answer(row)
                declared_target = parse_int(row.get("target_step_count", ""), "target_step_count")
                if target_steps != declared_target:
                    raise ValueError("target_step_count does not match answer JSON")
                total_step = parse_int(row.get("total_step", ""), "total_step")
                if target_steps + prefix_count != total_step + 1:
                    raise ValueError("prefix + target steps does not reconstruct the full record")

                audit.sources.add(source)
                audit.records.add(record)
                audit.samples.add(sample)
                audit.pathway_families.add(pathway_family)
                audit.organisms[row.get("organism", "").strip() or "<missing>"] += 1
                audit.phenotype_statuses[status] += 1
                audit.target_step_counts[target_steps] += 1
                audit.question_char_lengths[len(row.get("question", ""))] += 1
                audit.answer_char_lengths[len(row.get("answer", ""))] += 1
            except (ValueError, json.JSONDecodeError) as exc:
                audit.add_error(f"{prefix}: {exc}", max_errors)

    if audit.rows == 0:
        audit.add_error("CSV contains no data rows", max_errors)
    if audit.phenotype_statuses.get("available", 0) == 0:
        audit.warnings.append(
            "No available phenotype targets were found; this is allowed, but phenotype accuracy will have no eligible denominator."
        )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--report")
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument(
        "--allow-pathway-family-overlap",
        action="store_true",
        help="Allow a cross-species transfer evaluation to reuse pathway families from train.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = audit_file(Path(args.train), args.max_errors)
    test = audit_file(Path(args.test), args.max_errors)
    source_overlap = sorted(train.sources & test.sources)
    record_overlap = sorted(train.records & test.records)
    sample_overlap = sorted(train.samples & test.samples)
    pathway_family_overlap = sorted(train.pathway_families & test.pathway_families)
    cross_split = {
        "source_overlap_count": len(source_overlap),
        "record_overlap_count": len(record_overlap),
        "sample_overlap_count": len(sample_overlap),
        "pathway_family_overlap_count": len(pathway_family_overlap),
        "source_overlap_examples": source_overlap[:20],
        "record_overlap_examples": record_overlap[:20],
        "sample_overlap_examples": sample_overlap[:20],
        "pathway_family_overlap_examples": pathway_family_overlap[:20],
    }
    report = {
        "format_version": 1,
        "train": train.report(),
        "test": test.report(),
        "cross_split": cross_split,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote dataset audit: {destination}")
    else:
        print(rendered)

    failed = bool(
        train.errors
        or test.errors
        or cross_split["source_overlap_count"]
        or cross_split["record_overlap_count"]
        or cross_split["sample_overlap_count"]
        or (
            cross_split["pathway_family_overlap_count"]
            and not args.allow_pathway_family_overlap
        )
    )
    if args.strict and failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
