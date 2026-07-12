#!/usr/bin/env python3
"""Create reproducible, record-balanced train/eval CSVs from the full corpus.

The full generated training CSV contains tens of millions of prefix rows and
therefore overweights long pathway records. The default first-round training set keeps a stable
hash-selected 0.1% of eligible pathway records and at most three evenly spaced
prefixes per record (long-continuation, middle, and next-step views). Pathway
families reserved from the held-out organisms are removed from training and
form the strict core evaluation. A separate organism-transfer evaluation keeps
all held-out-organism families and reports their overlap. Older generated CSVs
are upgraded with stable sample/record/family identities and explicit
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
    from dataprocess.schemas import CSV_FIELDNAMES, canonical_pathway_family_id
    from dataprocess.substeps import parse_substeps
except ImportError:  # Allows: python dataprocess/prepare_experiment_data.py
    from audit_pathway_csv import audit_file  # type: ignore
    from schemas import CSV_FIELDNAMES, canonical_pathway_family_id  # type: ignore
    from substeps import parse_substeps  # type: ignore


DEFAULT_TRAIN_INPUT = "../data/train_kegg_pathway_dataset.csv"
DEFAULT_TEST_INPUT = "../data/test_kegg_pathway_dataset.csv"
DEFAULT_TRAIN_OUTPUT = "../data/train_kegg_pathway_record_balanced_0p1pct.csv"
DEFAULT_TEST_OUTPUT = "../data/test_kegg_pathway_eval.csv"
DEFAULT_MULTISTEP_TEST_OUTPUT = "../data/test_kegg_pathway_multistep_eval.csv"
DEFAULT_ORGANISM_TEST_OUTPUT = "../data/test_kegg_pathway_organism_eval.csv"
DEFAULT_ORGANISM_MULTISTEP_TEST_OUTPUT = "../data/test_kegg_pathway_organism_multistep_eval.csv"

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


def record_family(row: dict[str, str]) -> str:
    return canonical_pathway_family_id(row.get("pathway_id", ""))


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
    output["pathway_family_id"] = record_family(row)
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


def collect_pathway_families(source: Path) -> set[str]:
    families: set[str] = set()
    with source.open(newline="", encoding="utf-8-sig") as input_handle:
        reader = csv.DictReader(input_handle)
        for row in reader:
            families.add(record_family(row))
    if not families:
        raise ValueError(f"no pathway families found in {source}")
    return families


def select_holdout_families(
    families: set[str],
    *,
    fraction: float,
    seed: int,
) -> set[str]:
    if not 0 < fraction < 1:
        raise ValueError("family holdout fraction must be in (0, 1)")
    if len(families) < 2:
        raise ValueError("family holdout requires at least two pathway families")
    count = max(1, min(len(families) - 1, round(len(families) * fraction)))
    ranked = sorted(families, key=lambda family: (stable_fraction(f"family:{family}", seed), family))
    return set(ranked[:count])


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
    excluded_pathway_families: set[str] | None = None,
) -> dict[str, int | float | str]:
    input_rows = input_records = selected_records = selected_phenotype_records = output_rows = 0
    excluded_family_records = 0
    last_key: str | None = None
    buffer: list[dict[str, str]] = []
    excluded_pathway_families = set(excluded_pathway_families or ())

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open(newline="", encoding="utf-8-sig") as input_handle, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        def flush() -> None:
            nonlocal input_records, selected_records, selected_phenotype_records, output_rows, buffer, last_key
            nonlocal excluded_family_records
            if not buffer or last_key is None:
                return
            input_records += 1
            if record_family(buffer[0]) in excluded_pathway_families:
                excluded_family_records += 1
                buffer = []
                return
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
        "excluded_family_records": excluded_family_records,
        "excluded_pathway_family_count": len(excluded_pathway_families),
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
    included_pathway_families: set[str] | None = None,
) -> dict[str, int | str]:
    input_rows = input_records = selected_records = excluded_family_records = output_rows = 0
    last_key: str | None = None
    buffer: list[dict[str, str]] = []
    included_pathway_families = (
        None if included_pathway_families is None else set(included_pathway_families)
    )
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with source.open(newline="", encoding="utf-8-sig") as input_handle, temporary.open(
        "w", newline="", encoding="utf-8"
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        def flush() -> None:
            nonlocal input_records, selected_records, excluded_family_records, output_rows, buffer
            if not buffer:
                return
            input_records += 1
            if (
                included_pathway_families is not None
                and record_family(buffer[0]) not in included_pathway_families
            ):
                excluded_family_records += 1
                buffer = []
                return
            selected_records += 1
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
        "selected_records": selected_records,
        "excluded_family_records": excluded_family_records,
        "included_pathway_family_count": (
            len(included_pathway_families) if included_pathway_families is not None else 0
        ),
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
    parser.add_argument("--organism-test-output", default=DEFAULT_ORGANISM_TEST_OUTPUT)
    parser.add_argument(
        "--organism-multistep-test-output",
        default=DEFAULT_ORGANISM_MULTISTEP_TEST_OUTPUT,
    )
    parser.add_argument("--record-fraction", type=float, default=0.001)
    parser.add_argument("--phenotype-record-fraction", type=float, default=1.0)
    parser.add_argument(
        "--pathway-family-holdout-fraction",
        type=float,
        default=0.1,
        help=(
            "Fraction of pathway families present in the held-out organisms to reserve "
            "from training and use for the strict core evaluation."
        ),
    )
    parser.add_argument(
        "--pathway-family-holdout-seed",
        type=int,
        help="Family-ranking seed; defaults to --seed.",
    )
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
    if not 0 < args.pathway_family_holdout_fraction < 1:
        parser.error("--pathway-family-holdout-fraction must be in (0, 1)")
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
    organism_test_output = resolve(args.organism_test_output)
    organism_multistep_test_output = resolve(args.organism_multistep_test_output)
    validate_output_paths(
        (
            train_output,
            test_output,
            multistep_test_output,
            organism_test_output,
            organism_multistep_test_output,
        ),
        args.overwrite,
    )

    family_seed = (
        args.pathway_family_holdout_seed
        if args.pathway_family_holdout_seed is not None
        else args.seed
    )
    test_families = collect_pathway_families(test_input)
    heldout_families = select_holdout_families(
        test_families,
        fraction=args.pathway_family_holdout_fraction,
        seed=family_seed,
    )
    train_stats = prepare_train(
        train_input,
        train_output,
        record_fraction=args.record_fraction,
        max_prefixes_per_record=args.max_prefixes_per_record,
        seed=args.seed,
        phenotype_record_fraction=args.phenotype_record_fraction,
        excluded_pathway_families=heldout_families,
    )
    test_stats = prepare_test(
        test_input,
        test_output,
        max_prefixes_per_record=args.max_test_prefixes_per_record,
        included_pathway_families=heldout_families,
    )
    multistep_test_stats = prepare_test(
        test_input,
        multistep_test_output,
        max_prefixes_per_record=args.max_multistep_prefixes_per_record,
        included_pathway_families=heldout_families,
    )
    organism_test_stats = prepare_test(
        test_input,
        organism_test_output,
        max_prefixes_per_record=args.max_test_prefixes_per_record,
    )
    organism_multistep_test_stats = prepare_test(
        test_input,
        organism_multistep_test_output,
        max_prefixes_per_record=args.max_multistep_prefixes_per_record,
    )
    train_audit = audit_file(train_output, max_errors=100)
    test_audit = audit_file(test_output, max_errors=100)
    multistep_test_audit = audit_file(multistep_test_output, max_errors=100)
    organism_test_audit = audit_file(organism_test_output, max_errors=100)
    organism_multistep_test_audit = audit_file(
        organism_multistep_test_output,
        max_errors=100,
    )
    source_overlap = train_audit.sources & test_audit.sources
    record_overlap = train_audit.records & test_audit.records
    sample_overlap = train_audit.samples & test_audit.samples
    family_overlap = train_audit.pathway_families & test_audit.pathway_families
    organism_family_overlap = (
        train_audit.pathway_families & organism_test_audit.pathway_families
    )
    report = {
        "format_version": 1,
        "profile": "record_balanced_0p1pct_family_disjoint_v2",
        "pathway_family_holdout": {
            "definition": "terminal five-digit KEGG pathway reference-map id",
            "selection": "lowest deterministic SHA-256 family ranks among held-out organisms",
            "fraction": args.pathway_family_holdout_fraction,
            "seed": family_seed,
            "available_test_family_count": len(test_families),
            "heldout_family_count": len(heldout_families),
            "heldout_families": sorted(heldout_families),
        },
        "train": train_stats,
        "strict_family_disjoint_test": test_stats,
        "strict_family_disjoint_multistep_test": multistep_test_stats,
        "organism_heldout_test": organism_test_stats,
        "organism_heldout_multistep_test": organism_multistep_test_stats,
        "audit": {
            "train_errors": train_audit.errors,
            "strict_test_errors": test_audit.errors,
            "strict_multistep_test_errors": multistep_test_audit.errors,
            "organism_test_errors": organism_test_audit.errors,
            "organism_multistep_test_errors": organism_multistep_test_audit.errors,
            "strict_source_overlap_count": len(source_overlap),
            "strict_record_overlap_count": len(record_overlap),
            "strict_sample_overlap_count": len(sample_overlap),
            "strict_pathway_family_overlap_count": len(family_overlap),
            "strict_pathway_family_overlap_examples": sorted(family_overlap)[:20],
            "organism_eval_pathway_family_overlap_count": len(organism_family_overlap),
            "organism_eval_pathway_family_overlap_examples": sorted(organism_family_overlap)[:20],
            "train_phenotype_statuses": dict(train_audit.phenotype_statuses),
            "strict_test_phenotype_statuses": dict(test_audit.phenotype_statuses),
            "strict_multistep_test_phenotype_statuses": dict(multistep_test_audit.phenotype_statuses),
            "organism_test_phenotype_statuses": dict(organism_test_audit.phenotype_statuses),
            "organism_multistep_test_phenotype_statuses": dict(
                organism_multistep_test_audit.phenotype_statuses
            ),
        },
    }
    report_path = resolve(args.report) if args.report else train_output.with_suffix(".prepare.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = bool(
        train_audit.errors
        or test_audit.errors
        or multistep_test_audit.errors
        or organism_test_audit.errors
        or organism_multistep_test_audit.errors
        or source_overlap
        or record_overlap
        or sample_overlap
        or family_overlap
    )
    if failed:
        raise SystemExit(f"Prepared files failed audit; inspect {report_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
