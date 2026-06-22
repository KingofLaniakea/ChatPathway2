#!/usr/bin/env python3
"""Task IV: Conditional Step Prediction (CSP) evaluation.

It evaluates the *generated continuation* against the gold continuation. The
input is expected to contain ``answer`` and ``predicted_answer`` columns, as
produced by the inference script. A step can be JSON, ``entity | relation |
entity``, or a natural-language ``gene A activates gene B`` sentence.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from downstream.entities import normalize_entity
from downstream.io import load_records, mean, write_json, write_rows


RELATIONS = (
    "activates", "inhibits", "binds", "regulates", "phosphorylates", "dephosphorylates",
    "ubiquitinates", "methylates", "acetylates", "degrades", "cleaves", "forms", "produces",
    "converts", "catalyzes", "induces", "represses", "expresses", "transports", "associates",
    "dissociates",
)
RELATION_RE = re.compile(r"\b(" + "|".join(RELATIONS) + r")\b", re.IGNORECASE)
NATURAL_STEP_RE = re.compile(
    r"(?:gene|protein|metabolite|compound|component)?\s*([A-Za-z0-9._/\- ]+?)\s+"
    r"(" + "|".join(RELATIONS) + r")\s+(?:gene|protein|metabolite|compound|component)?\s*"
    r"([A-Za-z0-9._/\- ]+?)(?:[.;]|$)",
    re.IGNORECASE,
)


def clean_line(line: str) -> str:
    return re.sub(r"^\s*(?:step\s*)?\d+\s*[:.)-]\s*", "", line.strip(), flags=re.I)


def as_step(value: Any) -> tuple[str, str, str] | None:
    if isinstance(value, dict):
        left = value.get("reactant") or value.get("source") or value.get("e1")
        relation = value.get("relation") or value.get("reaction") or value.get("r")
        right = value.get("product") or value.get("target") or value.get("e2")
        if left and relation and right:
            return normalize_entity(str(left)), str(relation).lower().strip(), normalize_entity(str(right))
        return None
    line = clean_line(str(value))
    if "|" in line:
        parts = [part.strip() for part in line.split("|")]
        if len(parts) == 3:
            return normalize_entity(parts[0]), parts[1].lower(), normalize_entity(parts[2])
    if "->" in line:
        parts = [part.strip() for part in line.split("->")]
        if len(parts) == 3:
            return normalize_entity(parts[0]), parts[1].lower(), normalize_entity(parts[2])
    matched = NATURAL_STEP_RE.search(line)
    if matched:
        return tuple(normalize_entity(part) if index != 1 else part.lower() for index, part in enumerate(matched.groups()))  # type: ignore[return-value]
    return None


def parse_steps(text: Any) -> list[tuple[str, str, str] | None]:
    if isinstance(text, list):
        return [as_step(item) for item in text]
    value = str(text or "").strip()
    if not value:
        return []
    try:
        loaded = json.loads(value)
        if isinstance(loaded, dict):
            loaded = loaded.get("steps", loaded.get("pathway", []))
        if isinstance(loaded, list):
            return [as_step(item) for item in loaded]
    except json.JSONDecodeError:
        pass
    return [as_step(line) for line in value.splitlines() if clean_line(line)]


def evaluate_pair(target: list[tuple[str, str, str] | None], predicted: list[tuple[str, str, str] | None]) -> dict[str, float | int]:
    denominator = max(len(target), len(predicted), 1)
    overlap = min(len(target), len(predicted))
    exact = reactant = reaction = valid = 0
    for expected, actual in zip(target, predicted):
        if actual is not None:
            valid += 1
        if expected is None or actual is None:
            continue
        if (expected[0], expected[2]) == (actual[0], actual[2]):  # source and target
            reactant += 1
        if expected[1] == actual[1]:
            reaction += 1
        if expected == actual:
            exact += 1
    return {
        "target_steps": len(target), "predicted_steps": len(predicted), "paired_steps": overlap,
        "exact_match": exact / denominator, "reactant_match": reactant / denominator,
        "reaction_match": reaction / denominator, "parse_validity": valid / max(len(predicted), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-column", default="answer")
    parser.add_argument("--predicted-column", default="predicted_answer")
    args = parser.parse_args()
    rows = []
    for index, record in enumerate(load_records(args.input)):
        row = {"id": record.get("id", record.get("entry_id", index))}
        row.update(evaluate_pair(parse_steps(record.get(args.target_column)), parse_steps(record.get(args.predicted_column))))
        rows.append(row)
    summary = {"num_records": len(rows), "metrics": {
        key: mean([float(row[key]) for row in rows])
        for key in ("exact_match", "reactant_match", "reaction_match", "parse_validity")
    }}
    output_dir = Path(args.output_dir)
    write_rows(output_dir / "sample_metrics.csv", rows)
    write_json(output_dir / "summary_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
