"""Deterministic record-level prefix selection for pathway training.

One biological pathway record can have several highly overlapping prefix rows.
This module keeps all eligible rows available while exposing one row per record
in each epoch.  The selected horizon rotates deterministically across epochs,
so training cost follows the number of records rather than the number of
materialized prefixes.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any


PREFIX_SAMPLING_MODES = ("all_rows", "one_per_record")
PREFIX_POLICIES: dict[str, tuple[str, ...]] = {
    # Half of the record-epochs emphasize next-step/short continuation while
    # retaining systematic middle and long-continuation supervision.
    "sft_cycle": ("short", "short", "middle", "long"),
    # Latent dynamics benefits from longer target trajectories more often.
    "dynamics_cycle": ("long", "middle", "long", "short"),
    "balanced_cycle": ("long", "middle", "short"),
}

PREFIX_HORIZON_VALUES = (
    "long_target",
    "middle_target",
    "short_target",
    "degenerate_target",
)
_SCHEDULE_TO_HORIZON = {
    "long": "long_target",
    "middle": "middle_target",
    "short": "short_target",
}


def _stable_offset(seed: int, record_id: str, modulus: int) -> int:
    digest = hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulus


def _prefix_value(row: Mapping[str, Any], column: str) -> int:
    raw = row.get(column)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {column}={raw!r}") from exc
    if value < 0:
        raise ValueError(f"{column} must be non-negative, got {value}")
    return value


class EpochPrefixView:
    """Expose all rows or one deterministic prefix row per biological record."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        sampling_mode: str,
        policy: str,
        seed: int,
        record_column: str = "record_id",
        prefix_column: str = "prefix_step_count",
        horizon_column: str = "prefix_horizon",
    ) -> None:
        if sampling_mode not in PREFIX_SAMPLING_MODES:
            raise ValueError(
                f"unknown prefix sampling mode {sampling_mode!r}; "
                f"expected one of {PREFIX_SAMPLING_MODES}"
            )
        if policy not in PREFIX_POLICIES:
            raise ValueError(
                f"unknown prefix policy {policy!r}; expected one of {tuple(PREFIX_POLICIES)}"
            )
        if not rows:
            raise ValueError("prefix sampling requires at least one row")

        self.rows = [dict(row) for row in rows]
        self.sampling_mode = sampling_mode
        self.policy = policy
        self.seed = int(seed)
        self.record_column = record_column
        self.prefix_column = prefix_column
        self.horizon_column = horizon_column
        self.epoch = 1

        if sampling_mode == "all_rows":
            self._record_ids = [str(index) for index in range(len(self.rows))]
            self._groups = [(index,) for index in range(len(self.rows))]
            return

        grouped: dict[str, list[int]] = {}
        for index, row in enumerate(self.rows):
            record_id = str(row.get(record_column, "")).strip()
            if not record_id:
                raise ValueError(
                    f"one_per_record prefix sampling requires non-empty {record_column!r}"
                )
            grouped.setdefault(record_id, []).append(index)

        self._record_ids = list(grouped)
        groups: list[tuple[int, ...]] = []
        horizons: list[dict[str, int]] = []
        for record_id in self._record_ids:
            indices = sorted(
                grouped[record_id],
                key=lambda index: (
                    _prefix_value(self.rows[index], prefix_column),
                    str(self.rows[index].get("sample_id", "")),
                ),
            )
            prefix_values = [
                _prefix_value(self.rows[index], prefix_column) for index in indices
            ]
            if len(prefix_values) != len(set(prefix_values)):
                raise ValueError(
                    f"record {record_id!r} has duplicate {prefix_column} values"
                )
            groups.append(tuple(indices))
            explicit_horizons: dict[str, int] = {}
            for row_index in indices:
                raw_horizon = str(self.rows[row_index].get(horizon_column, "")).strip()
                if not raw_horizon:
                    continue
                if raw_horizon not in PREFIX_HORIZON_VALUES:
                    raise ValueError(
                        f"record {record_id!r} has invalid {horizon_column}={raw_horizon!r}"
                    )
                if raw_horizon in explicit_horizons:
                    raise ValueError(
                        f"record {record_id!r} has duplicate {horizon_column}={raw_horizon!r}"
                    )
                explicit_horizons[raw_horizon] = row_index
            if explicit_horizons and len(explicit_horizons) != len(indices):
                raise ValueError(
                    f"record {record_id!r} mixes explicit and missing {horizon_column} values"
                )
            horizons.append(explicit_horizons)
        self._groups = groups
        self._horizons = horizons

    def __len__(self) -> int:
        return len(self._groups)

    @property
    def eligible_row_count(self) -> int:
        return len(self.rows)

    @property
    def record_count(self) -> int:
        return len(self._groups)

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("epoch must be at least 1")
        self.epoch = int(epoch)

    def _selection(self, index: int) -> tuple[int, str]:
        group = self._groups[index]
        if self.sampling_mode == "all_rows":
            return group[0], "all"
        if len(group) == 1:
            return group[0], "only"

        record_id = self._record_ids[index]
        schedule = PREFIX_POLICIES[self.policy]
        offset = _stable_offset(self.seed, record_id, len(schedule))
        horizon = schedule[(self.epoch - 1 + offset) % len(schedule)]
        explicit_horizons = self._horizons[index]
        if explicit_horizons:
            requested = _SCHEDULE_TO_HORIZON[horizon]
            selected = explicit_horizons.get(requested)
            if selected is not None:
                return selected, horizon
        positions = {
            "long": 0,
            "middle": (len(group) - 1) // 2,
            "short": len(group) - 1,
        }
        return group[positions[horizon]], horizon

    def row(self, index: int) -> dict[str, Any]:
        row_index, _ = self._selection(index)
        return self.rows[row_index]

    def selected_row_indices(self) -> tuple[int, ...]:
        return tuple(self._selection(index)[0] for index in range(len(self)))

    def selection_summary(self) -> dict[str, int]:
        counts = Counter(
            self._selection(index)[1]
            for index in range(len(self))
        )
        return dict(sorted(counts.items()))


__all__ = [
    "EpochPrefixView",
    "PREFIX_HORIZON_VALUES",
    "PREFIX_POLICIES",
    "PREFIX_SAMPLING_MODES",
]
