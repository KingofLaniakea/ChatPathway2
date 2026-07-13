"""Strict CSV loading shared by pathway trainers."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


def read_training_csv(path: str | Path) -> pd.DataFrame:
    """Read every field as text so identity hashes preserve leading zeros."""

    return pd.read_csv(
        path,
        engine="c",
        quoting=csv.QUOTE_MINIMAL,
        on_bad_lines="error",
        dtype=str,
        keep_default_na=False,
    )


__all__ = ["read_training_csv"]
