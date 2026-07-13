"""Measure how much pathway supervision survives the training token budget."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, TextIO

from method.training.sequence import encode_supervised


def safe_fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def audit_rows(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    answer_budget_fraction: float,
    text_column: str = "answer",
    progress_every: int = 0,
    progress_stream: TextIO | None = None,
) -> dict[str, Any]:
    counters = {
        "rows": 0,
        "rows_prompt_truncated": 0,
        "rows_answer_truncated": 0,
        "rows_with_semantic_steps": 0,
        "rows_full_semantic_step_retention": 0,
        "rows_partial_semantic_step_retention": 0,
        "rows_zero_retained_semantic_steps": 0,
        "prompt_tokens_dropped": 0,
        "answer_tokens_dropped": 0,
        "semantic_steps_total": 0,
        "semantic_steps_retained": 0,
        "substeps_total": 0,
        "substeps_retained": 0,
    }
    stream = progress_stream or sys.stderr
    for row in rows:
        question = str(row.get("question", ""))
        answer = str(
            row.get(text_column)
            or row.get("answer")
            or row.get("formatted_answer_no_phenotype")
            or ""
        )
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
        encoded = encode_supervised(
            tokenizer,
            prompt,
            answer,
            max_length=max_length,
            answer_budget_fraction=answer_budget_fraction,
            truncation_policy="measure",
        )
        counters["rows"] += 1
        counters["rows_prompt_truncated"] += int(encoded.prompt_tokens_dropped > 0)
        counters["rows_answer_truncated"] += int(encoded.answer_tokens_dropped > 0)
        counters["prompt_tokens_dropped"] += encoded.prompt_tokens_dropped
        counters["answer_tokens_dropped"] += encoded.answer_tokens_dropped
        counters["semantic_steps_total"] += encoded.semantic_steps_total
        counters["semantic_steps_retained"] += encoded.semantic_steps_retained
        counters["substeps_total"] += encoded.substeps_total
        counters["substeps_retained"] += encoded.substeps_retained
        if encoded.semantic_steps_total:
            counters["rows_with_semantic_steps"] += 1
            if encoded.semantic_steps_retained == encoded.semantic_steps_total:
                counters["rows_full_semantic_step_retention"] += 1
            elif encoded.semantic_steps_retained == 0:
                counters["rows_zero_retained_semantic_steps"] += 1
            else:
                counters["rows_partial_semantic_step_retention"] += 1
        if progress_every and counters["rows"] % progress_every == 0:
            print(f"token_budget_audit rows={counters['rows']}", file=stream, flush=True)

    rows_count = counters["rows"]
    semantic_rows = counters["rows_with_semantic_steps"]
    return {
        "format_version": 1,
        "max_length": max_length,
        "answer_budget_fraction": answer_budget_fraction,
        **counters,
        "semantic_steps_dropped": (
            counters["semantic_steps_total"] - counters["semantic_steps_retained"]
        ),
        "substeps_dropped": counters["substeps_total"] - counters["substeps_retained"],
        "row_prompt_truncation_fraction": safe_fraction(
            counters["rows_prompt_truncated"], rows_count
        ),
        "row_answer_truncation_fraction": safe_fraction(
            counters["rows_answer_truncated"], rows_count
        ),
        "row_zero_retained_semantic_fraction": safe_fraction(
            counters["rows_zero_retained_semantic_steps"], semantic_rows
        ),
        "semantic_step_retention_fraction": safe_fraction(
            counters["semantic_steps_retained"], counters["semantic_steps_total"]
        ),
        "substep_retention_fraction": safe_fraction(
            counters["substeps_retained"], counters["substeps_total"]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--output")
    parser.add_argument("--text-column", default="answer")
    parser.add_argument("--max-length", type=int, default=1072)
    parser.add_argument("--answer-budget-fraction", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length < 2:
        raise SystemExit("--max-length must be at least 2")
    if not 0 < args.answer_budget_fraction < 1:
        raise SystemExit("--answer-budget-fraction must be between 0 and 1")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    input_path = Path(args.input)
    with input_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {input_path}")
        rows: Iterable[Mapping[str, Any]] = reader
        if args.limit is not None:
            import itertools

            rows = itertools.islice(rows, args.limit)
        report = audit_rows(
            rows,
            tokenizer,
            max_length=args.max_length,
            answer_budget_fraction=args.answer_budget_fraction,
            text_column=args.text_column,
            progress_every=args.progress_every,
        )
    report["input"] = str(input_path.resolve())
    report["input_size_bytes"] = input_path.stat().st_size
    report["base_model"] = str(Path(args.base_model).resolve())
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(output)
    print(rendered, end="")


if __name__ == "__main__":
    main()
