"""Small, dependency-light I/O helpers shared by downstream evaluators."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a list of records from CSV, JSON, or JSONL."""
    source = Path(path)
    if source.suffix.lower() == ".csv":
        with source.open(newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if source.suffix.lower() == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, list):
        return value
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        return value["records"]
    raise ValueError(f"{source} must contain a list or an object with a 'records' list.")


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def as_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
