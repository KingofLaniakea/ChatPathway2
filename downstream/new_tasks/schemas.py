"""Shared validation helpers for the revised downstream task contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class SchemaError(ValueError):
    """Raised when an evaluation artifact violates a published task schema."""


def require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaError(f"{path} must be an object.")
    return value


def require_sequence(value: Any, path: str, *, nonempty: bool = False) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaError(f"{path} must be an array.")
    if nonempty and not value:
        raise SchemaError(f"{path} must not be empty.")
    return value


def require_text(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise SchemaError(f"{path} must be a string.")
    result = value.strip()
    if nonempty and not result:
        raise SchemaError(f"{path} must not be empty.")
    return result


def require_number(value: Any, path: str, *, finite: bool = True) -> float:
    if isinstance(value, bool):
        raise SchemaError(f"{path} must be numeric, not boolean.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"{path} must be numeric.") from exc
    if finite and not math.isfinite(result):
        raise SchemaError(f"{path} must be finite.")
    return result


def require_integer(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise SchemaError(f"{path} must be an integer, not boolean.")
    try:
        result = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaError(f"{path} must be an integer.") from exc
    if not math.isfinite(numeric) or numeric != result:
        raise SchemaError(f"{path} must be an integer.")
    if minimum is not None and result < minimum:
        raise SchemaError(f"{path} must be at least {minimum}.")
    return result


def require_probability(value: Any, path: str) -> float:
    result = require_number(value, path)
    if not 0.0 <= result <= 1.0:
        raise SchemaError(f"{path} must be in [0, 1].")
    return result


def require_binary(value: Any, path: str) -> int:
    if isinstance(value, bool):
        return int(value)
    result = require_number(value, path)
    if result not in (0.0, 1.0):
        raise SchemaError(f"{path} must be 0 or 1.")
    return int(result)


def require_choice(value: Any, choices: Iterable[str], path: str) -> str:
    result = require_text(value, path)
    allowed = tuple(choices)
    if result not in allowed:
        raise SchemaError(f"{path} must be one of {allowed}; got {result!r}.")
    return result


def require_fields(value: Mapping[str, Any], fields: Iterable[str], path: str) -> None:
    missing = [field for field in fields if field not in value]
    if missing:
        raise SchemaError(f"{path} is missing required fields: {', '.join(missing)}.")


def ensure_unique(values: Sequence[Any], path: str) -> None:
    normalized = [str(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise SchemaError(f"{path} must contain unique values.")


def load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise SchemaError(f"{source} must contain a JSON object.")
    return value


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON or JSONL records without accepting ambiguous scalar payloads."""

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with source.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SchemaError(f"{source}:{line_number} contains invalid JSON: {exc.msg}.") from exc
                rows.append(dict(require_mapping(value, f"record[{line_number - 1}]")))
        return rows
    with source.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if isinstance(value, dict) and isinstance(value.get("records"), list):
        value = value["records"]
    rows = require_sequence(value, str(source))
    return [dict(require_mapping(record, f"record[{index}]")) for index, record in enumerate(rows)]


def validate_finite_array(array: Any, path: str, *, ndim: int | None = None) -> None:
    """Validate a NumPy-like array without importing NumPy at module import time."""

    import numpy as np

    value = np.asarray(array)
    if ndim is not None and value.ndim != ndim:
        raise SchemaError(f"{path} must have {ndim} dimensions; got shape {value.shape}.")
    if value.size == 0:
        raise SchemaError(f"{path} must not be empty.")
    if not np.issubdtype(value.dtype, np.number) or not np.all(np.isfinite(value)):
        raise SchemaError(f"{path} must contain only finite numeric values.")
