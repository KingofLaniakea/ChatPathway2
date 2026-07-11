#!/usr/bin/env python3
"""Create reproducible, record-balanced train/eval CSVs from the full corpus.

The full generated training CSV contains tens of millions of prefix rows and
therefore overweights long pathway records. The default pilot keeps a stable
hash-selected 0.1% of pathway records and at most three evenly spaced prefixes
per record (long-continuation, middle, and next-step views). It also upgrades
older generated CSVs with stable record/sample identities and explicit
``not_annotated`` phenotype status.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from dataprocess.audit_pathway_csv import audit_file
    from dataprocess.schemas import CSV_FIELDNAMES
    from dataprocess.substeps import parse_substeps
except ImportError:  # Allows: python dataprocess/prepare_experiment_data.py
    from audit_pathway_csv import audit_file  # type: ignore
    from schemas import CSV_FIELDNAMES  # type: ignore
    from substeps import parse_substeps  # type: ignore


DEFAULT_TRAIN_INPUT = "../data/train_kegg_pathway_dataset.csv"
DEFAULT_TEST_INPUT = "../data/test_kegg_pathway_dataset.csv"
DEFAULT_TRAIN_OUTPUT = "../data/train_kegg_pathway_pilot.csv"
DEFAULT_TEST_OUTPUT = "../data/test_kegg_pathway_eval.csv"
DEFAULT_MULTISTEP_TEST_OUTPUT = "../data/test_kegg_pathway_multistep_eval.csv"

csv.field_size_limit(sys.maxsize)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (repo_root() / path).resolve()


def stable_fraction(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(16**16)


def record_key(row: dict[str, str]) -> str:
    source = row.get("source_json", "").strip()
    block = row.get("pathway_block", "").strip()
    if not source or not block:
        raise ValueError("source_json and pathway_block are required for record-balanced sampling")
    return f"{source}\n{block}"


def record_id(row: dict[str, str]) -> str:
    identity = "\n".join(
        (
            row.get("organism", "").strip(),
            row.get("source_json", "").strip(),
            row.get("pathway_id", "").strip(),
            row.get("pathway_block", "").strip(),
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def normalize_row(row: dict[str, str]) -> dict[str, Any]:
    output: dict[str, Any] = {field: row.get(field, "") for field in CSV_FIELDNAMES}
    identity = record_id(row)
    try:
        prefix = int(row.get("prefix_step_count", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid prefix_step_count={row.get('prefix_step_count')!r}") from exc
    output["record_id"] = identity
    output["sample_id"] = f"{identity}:prefix={prefix}"
    output["substep_schema_version"] = "layer_set_v1"
    output["substep_source"] = row.get("substep_source", "").strip() or "sentence_parser_v1"
    question = str(output.get("question", ""))
    question = question.replace(
        "- A Step may summarize multiple reaction or relation events that occur in the same graph layer.",
        "- Each Step contains one or more substeps/events at the same graph depth; do not invent an order among same-depth events.",
    )
    question = question.replace(
        '- Return valid JSON only, with keys "remaining_steps" and "predicted_phenotype".',
        '- Return valid JSON only, with keys "remaining_steps" and "predicted_phenotype"; each remaining Step must contain "step", "layer", and a "substeps" list.',
    )
    output["question"] = question
    try:
        answer = json.loads(str(output.get("answer", "")))
        if isinstance(answer, dict) and isinstance(answer.get("remaining_steps"), list):
            for step in answer["remaining_steps"]:
                if not isinstance(step, dict) or isinstance(step.get("substeps"), list):
                    continue
                text = str(step.get("text", "")).strip()
                step["substeps"] = [event.as_dict() for event in parse_substeps(text)]
                step.pop("text", None)
            output["answer"] = json.dumps(
                answer,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
    except json.JSONDecodeError:
        pass
    if str(output.get("phenotype_status", "")).strip() in {"", "missing"}:
        output["phenotype_status"] = "not_annotated"
    if output["phenotype_status"] != "available":
        output["phenotype"] = ""
        try:
            answer = json.loads(str(output.get("answer", "")))
            if isinstance(answer, dict):
                answer["predicted_phenotype"] = None
                output["answer"] = json.dumps(
                    answer,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
        except json.JSONDecodeError:
            pass
    return output


def selected_prefixes(rows: list[dict[str, str]], maximum: int) -> list[dict[str, str]]:
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    if maximum == 1:
        return [rows[-1]]
    indices = {
        round(position * (len(rows) - 1) / (maximum - 1))
        for position in range(maximum)
    }
    return [rows[index] for index in sorted(indices)]


def validate_output_paths(paths: Iterable[Path], overwrite: bool) -> None:
    for path in paths:
        if path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite {path}; pass --overwrite")
        path.parent.mkdir(parents=True, exist_ok=True)


def prepare_train(
    source: Path,
    destination: Path,
    *,
    record_fraction: float,
    max_prefixes_per_record: int,
    seed: int,
    phenotype_record_fraction: float,
) -> dict[str, int | float | str]:
    input_rows = input_records = selected_records = selected_phenotype_records = output_rows = 0
    last_key: str | None = None
    buffer: list[dict[str, str]] = []

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open(newline="", encoding="utf-8-sig") as input_handle, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        def flush() -> None:
            nonlocal input_records, selected_records, selected_phenotype_records, output_rows, buffer, last_key
            if not buffer or last_key is None:
                return
            input_records += 1
            has_phenotype = any(row.get("phenotype_status") == "available" for row in buffer)
            selected = stable_fraction(last_key, seed) < record_fraction
            if has_phenotype and stable_fraction(f"phenotype:{last_key}", seed) < phenotype_record_fraction:
                selected = True
            if selected:
                selected_records += 1
                if has_phenotype:
                    selected_phenotype_records += 1
                for raw_row in selected_prefixes(buffer, max_prefixes_per_record):
                    writer.writerow(normalize_row(raw_row))
                    output_rows += 1
            buffer = []

        for row in reader:
            input_rows += 1
            key = record_key(row)
            if last_key is None:
                last_key = key
            elif key != last_key:
                flush()
                last_key = key
            buffer.append(row)
        flush()

    if not output_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError("record_fraction selected no training rows; increase the fraction")
    temporary.replace(destination)
    return {
        "input": str(source),
        "output": str(destination),
        "input_rows": input_rows,
        "input_records": input_records,
        "selected_records": selected_records,
        "selected_phenotype_records": selected_phenotype_records,
        "output_rows": output_rows,
        "record_fraction": record_fraction,
        "max_prefixes_per_record": max_prefixes_per_record,
        "seed": seed,
        "phenotype_record_fraction": phenotype_record_fraction,
    }


def prepare_test(
    source: Path,
    destination: Path,
    *,
    max_prefixes_per_record: int,
) -> dict[str, int | str]:
    input_rows = input_records = output_rows = 0
    last_key: str | None = None
    buffer: list[dict[str, str]] = []
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open(newline="", encoding="utf-8-sig") as input_handle, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        def flush() -> None:
            nonlocal input_records, output_rows, buffer
            if not buffer:
                return
            input_records += 1
            for raw_row in selected_prefixes(buffer, max_prefixes_per_record):
                writer.writerow(normalize_row(raw_row))
                output_rows += 1
            buffer = []

        for row in reader:
            input_rows += 1
            key = record_key(row)
            if last_key is None:
                last_key = key
            elif key != last_key:
                flush()
                last_key = key
            buffer.append(row)
        flush()
    if not output_rows:
        temporary.unlink(missing_ok=True)
        raise ValueError("test CSV contains no rows")
    temporary.replace(destination)
    return {
        "input": str(source),
        "output": str(destination),
        "input_rows": input_rows,
        "input_records": input_records,
        "output_rows": output_rows,
        "max_prefixes_per_record": max_prefixes_per_record,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-input", default=DEFAULT_TRAIN_INPUT)
    parser.add_argument("--test-input", default=DEFAULT_TEST_INPUT)
    parser.add_argument("--train-output", default=DEFAULT_TRAIN_OUTPUT)
    parser.add_argument("--test-output", default=DEFAULT_TEST_OUTPUT)
    parser.add_argument("--multistep-test-output", default=DEFAULT_MULTISTEP_TEST_OUTPUT)
    parser.add_argument("--record-fraction", type=float, default=0.001)
    parser.add_argument("--phenotype-record-fraction", type=float, default=1.0)
    parser.add_argument("--max-prefixes-per-record", type=int, default=3)
    parser.add_argument("--max-test-prefixes-per-record", type=int, default=1)
    parser.add_argument("--max-multistep-prefixes-per-record", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--report")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 < args.record_fraction <= 1:
        parser.error("--record-fraction must be in (0, 1]")
    if not 0 < args.phenotype_record_fraction <= 1:
        parser.error("--phenotype-record-fraction must be in (0, 1]")
    if args.max_prefixes_per_record < 1:
        parser.error("--max-prefixes-per-record must be positive")
    if args.max_test_prefixes_per_record < 1:
        parser.error("--max-test-prefixes-per-record must be positive")
    if args.max_multistep_prefixes_per_record < 1:
        parser.error("--max-multistep-prefixes-per-record must be positive")
    return args


def main() -> None:
    args = parse_args()
    train_input = resolve(args.train_input)
    test_input = resolve(args.test_input)
    train_output = resolve(args.train_output)
    test_output = resolve(args.test_output)
    multistep_test_output = resolve(args.multistep_test_output)
    validate_output_paths((train_output, test_output, multistep_test_output), args.overwrite)
    train_stats = prepare_train(
        train_input,
        train_output,
        record_fraction=args.record_fraction,
        max_prefixes_per_record=args.max_prefixes_per_record,
        seed=args.seed,
        phenotype_record_fraction=args.phenotype_record_fraction,
    )
    test_stats = prepare_test(
        test_input,
        test_output,
        max_prefixes_per_record=args.max_test_prefixes_per_record,
    )
    multistep_test_stats = prepare_test(
        test_input,
        multistep_test_output,
        max_prefixes_per_record=args.max_multistep_prefixes_per_record,
    )
    train_audit = audit_file(train_output, max_errors=100)
    test_audit = audit_file(test_output, max_errors=100)
    multistep_test_audit = audit_file(multistep_test_output, max_errors=100)
    overlap = train_audit.sources & test_audit.sources
    report = {
        "format_version": 1,
        "profile": "pilot_record_balanced_v1",
        "train": train_stats,
        "test": test_stats,
        "multistep_test": multistep_test_stats,
        "audit": {
            "train_errors": train_audit.errors,
            "test_errors": test_audit.errors,
            "multistep_test_errors": multistep_test_audit.errors,
            "source_overlap_count": len(overlap),
            "source_overlap_examples": sorted(overlap)[:20],
            "train_phenotype_statuses": dict(train_audit.phenotype_statuses),
            "test_phenotype_statuses": dict(test_audit.phenotype_statuses),
            "multistep_test_phenotype_statuses": dict(multistep_test_audit.phenotype_statuses),
        },
    }
    report_path = resolve(args.report) if args.report else train_output.with_suffix(".prepare.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if train_audit.errors or test_audit.errors or multistep_test_audit.errors or overlap:
        raise SystemExit(f"Prepared files failed audit; inspect {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
