"""Lossless CSV loading helpers for inference provenance fields."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TypeVar


Row = TypeVar("Row")


def read_csv_text_rows(
    path: str | Path,
    *,
    limit: int | None = None,
) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV cells as exact text instead of inferring numeric/NA types.

    Identifiers such as the KEGG family ``00051`` are semantic strings.  A
    dataframe parser that infers integers silently rewrites that value to
    ``51`` on output, while default NA parsing can turn an intentionally empty
    phenotype into a floating-point missing value.  The standard-library CSV
    reader preserves both contracts before the rows enter a dataframe.
    """

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    csv.field_size_limit(max(csv.field_size_limit(), 16 * 1024 * 1024))
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        fieldnames = list(reader.fieldnames)
        rows: list[dict[str, str]] = []
        if limit == 0:
            return fieldnames, rows
        for raw in reader:
            if None in raw:
                raise ValueError(f"CSV row has more cells than its header: {path}")
            rows.append(
                {
                    field: "" if raw.get(field) is None else str(raw[field])
                    for field in fieldnames
                }
            )
            if limit is not None and len(rows) >= limit:
                break
    return fieldnames, rows


def select_strided_shard(
    rows: list[Row],
    *,
    shard_index: int,
    shard_count: int,
) -> list[tuple[int, Row]]:
    """Return one deterministic shard while preserving global row indices.

    Striding balances long and short generations better than contiguous slices.
    The original index is carried into every progress record and shard CSV so a
    merger can prove exact, duplicate-free coverage and restore input order.
    """

    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [
        (index, row)
        for index, row in enumerate(rows)
        if index % shard_count == shard_index
    ]


__all__ = ["read_csv_text_rows", "select_strided_shard"]
