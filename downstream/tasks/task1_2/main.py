#!/usr/bin/env python3
"""Tasks I and II: PCER retrieval and biochemical entity consistency.

The evaluator accepts the inference CSV produced by ``method/inference/pathway.py``.
For a scientifically valid PCER result supply a reference mapping built from an
external pathway database. When no reference is supplied, it constructs a
closed-corpus reference from gold answers only for pipeline debugging and marks
that limitation in ``summary_metrics.json``.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from downstream.common.entities import extract_entities, load_synonyms, precision_recall_f1
from downstream.common.io import load_records, mean, write_json, write_rows
from downstream.common.pathway_json import parse_pathway_payload, record_id


def pathway_step_text(value: Any) -> str:
    parsed = parse_pathway_payload(value)
    return parsed.step_text or str(value or "")


def pathway_name(record: dict[str, Any], column: str) -> str:
    if record.get(column):
        return str(record[column]).strip()
    for key in ("pathway_name", "pathway_id"):
        if record.get(key):
            return str(record[key]).strip()
    question = str(record.get("question", ""))
    matched = re.search(r"pathway about\s+(.+?)(?:\s*\(|\.|\n|$)", question, re.I)
    return matched.group(1).strip() if matched else "UNKNOWN_PATHWAY"


def load_reference(path: str, pathway_column: str, gene_column: str, synonyms: dict[str, str]) -> dict[str, set[str]]:
    source = Path(path)
    if source.suffix.lower() == ".csv":
        mapping: dict[str, set[str]] = defaultdict(set)
        for row in load_records(source):
            pathway, gene = row.get(pathway_column), row.get(gene_column)
            if pathway and gene:
                mapping[str(pathway)].update(extract_entities(f"gene {gene}", synonyms))
        return dict(mapping)
    import json

    with source.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    raw = raw.get("pathways", raw) if isinstance(raw, dict) else raw
    if not isinstance(raw, dict):
        raise ValueError("Reference JSON must map pathway names to entity lists.")
    return {
        str(name): set().union(*(extract_entities(f"gene {gene}", synonyms) for gene in genes))
        for name, genes in raw.items()
    }


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(overlap: int, query_size: int, pathway_size: int, universe_size: int) -> float:
    """P(X >= overlap) without requiring scipy."""
    if overlap <= 0 or query_size <= 0 or pathway_size <= 0:
        return 1.0
    upper = min(query_size, pathway_size)
    lower = max(overlap, query_size - (universe_size - pathway_size), 0)
    log_denominator = log_comb(universe_size, query_size)
    logs = [log_comb(pathway_size, value) + log_comb(universe_size - pathway_size, query_size - value) - log_denominator
            for value in range(lower, upper + 1)]
    max_log = max(logs)
    return min(1.0, math.exp(max_log) * sum(math.exp(value - max_log) for value in logs))


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    count = len(pvalues)
    order = sorted(range(count), key=pvalues.__getitem__)
    result = [1.0] * count
    running = 1.0
    for reverse_index in range(count - 1, -1, -1):
        index = order[reverse_index]
        rank = reverse_index + 1
        running = min(running, pvalues[index] * count / rank)
        result[index] = min(1.0, running)
    return result


def evaluate(records: list[dict[str, Any]], reference: dict[str, set[str]], *, predicted_column: str,
             target_column: str, pathway_column: str, synonyms: dict[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    universe = set().union(*reference.values()) if reference else set()
    if not universe:
        raise ValueError("Reference library contains no entities after normalization.")
    reference_names = list(reference)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        predicted = extract_entities(pathway_step_text(record.get(predicted_column, "")), synonyms)
        target = extract_entities(pathway_step_text(record.get(target_column, "")), synonyms)
        metrics = precision_recall_f1(predicted, target)
        truth = pathway_name(record, pathway_column)
        pvalues = [hypergeom_sf(len(predicted & reference[name]), len(predicted), len(reference[name]), len(universe))
                   for name in reference_names]
        qvalues = benjamini_hochberg(pvalues)
        ranked = sorted(
            zip(reference_names, pvalues, qvalues),
            key=lambda item: (item[2], item[1], item[0]),
        )
        rank = next((rank for rank, (name, _, _) in enumerate(ranked, 1) if name == truth), None)
        row = {
            "id": record_id(record, index),
            "true_pathway": truth,
            "predicted_pathway": ranked[0][0] if ranked else "",
            "rank_true": rank or "",
            "hit_at_1": float(rank == 1),
            "hit_at_3": float(rank is not None and rank <= 3),
            "hit_at_5": float(rank is not None and rank <= 5),
            "mrr": 1.0 / rank if rank else 0.0,
            "predicted_entity_count": len(predicted),
            "target_entity_count": len(target),
            "predicted_entities": ";".join(sorted(predicted)),
            "target_entities": ";".join(sorted(target)),
            **metrics,
        }
        rows.append(row)
    summary = {
        "num_records": len(rows),
        "num_reference_pathways": len(reference),
        "num_universe_entities": len(universe),
        "pcer": {key: mean([float(row[key]) for row in rows]) for key in ("hit_at_1", "hit_at_3", "hit_at_5", "mrr")},
        "entity_consistency": {key: mean([float(row[key]) for row in rows]) for key in ("precision", "recall", "f1", "exact_match")},
        "entity_error_totals": {key: sum(int(row[key]) for row in rows) for key in ("tp", "fp", "fn")},
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True, help="Prediction CSV/JSON/JSONL.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference", help="External pathway -> entity JSON or pathway/entity CSV.")
    parser.add_argument("--pathway-column", default="pathway_id")
    parser.add_argument("--gene-column", default="gene")
    parser.add_argument("--predicted-column", default="predicted_answer")
    parser.add_argument("--target-column", default="answer")
    parser.add_argument("--synonyms", help="Optional JSON alias -> canonical entity mapping.")
    args = parser.parse_args()
    records = load_records(args.input)
    synonyms = load_synonyms(args.synonyms)
    if args.reference:
        reference = load_reference(args.reference, args.pathway_column, args.gene_column, synonyms)
        reference_mode = "external_reference"
    else:
        reference = defaultdict(set)
        for record in records:
            reference[pathway_name(record, args.pathway_column)].update(
                extract_entities(pathway_step_text(record.get(args.target_column, "")), synonyms)
            )
        reference = dict(reference)
        reference_mode = "closed_corpus_debug_only"
    rows, summary = evaluate(
        records, reference, predicted_column=args.predicted_column, target_column=args.target_column,
        pathway_column=args.pathway_column, synonyms=synonyms,
    )
    summary["reference_mode"] = reference_mode
    if reference_mode != "external_reference":
        summary["warning"] = "PCER used gold-derived reference sets; do not report it as external enrichment retrieval."
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
