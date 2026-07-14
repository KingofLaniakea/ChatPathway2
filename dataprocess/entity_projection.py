"""Strict, auditable entity projection for pathway prompt profiles.

The two source-native prompt profiles preserve entity identifiers exactly.  The
species-neutral profile is deliberately fail-closed: without a reviewed
gene-to-KO mapping, it accepts only identifiers that are *already* in a small
allowlist of species-neutral KEGG namespaces.  It never creates a mapping by
stripping an organism prefix.

Canonical records may retain provenance such as ``organism`` outside the model
payload.  This module checks every entity under every event in a record; the
prompt renderer remains responsible for keeping provenance metadata out of the
model-visible question and answer.
"""

from __future__ import annotations

import copy
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from dataprocess.prompt_profiles import (
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    PROMPT_PROFILE_NAMES,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
)


# This is intentionally a conservative allowlist, not a catalogue of every
# KEGG namespace.  ``gl`` is KEGG's native glycan prefix; ``glycan`` is kept as
# the explicit canonical spelling used by some processed records.
SPECIES_NEUTRAL_KEGG_NAMESPACES = frozenset(
    {"ko", "cpd", "gl", "glycan", "rn", "ec"}
)

_NEUTRAL_IDENTIFIER_PATTERNS: dict[str, re.Pattern[str]] = {
    "ko": re.compile(r"K\d{5}", re.IGNORECASE),
    "cpd": re.compile(r"C\d{5}", re.IGNORECASE),
    "gl": re.compile(r"G\d{5}", re.IGNORECASE),
    "glycan": re.compile(r"G\d{5}", re.IGNORECASE),
    "rn": re.compile(r"R\d{5}", re.IGNORECASE),
    # EC components can contain a dash or a provisional n-number.
    "ec": re.compile(
        r"(?:\d+|-|n\d+)(?:\.(?:\d+|-|n\d+)){3}", re.IGNORECASE
    ),
}

_SOURCE_NATIVE_NAMESPACES = frozenset(
    {"gene", "genes", "protein", "proteins", "ortholog"}
)
_PATHWAY_NAMESPACES = frozenset({"path", "pathway"})
_INTERNAL_NAMESPACES = frozenset(
    {
        "entry",
        "group",
        "internal",
        "node",
        "raw",
        "source",
        "undefined",
        "unknown",
    }
)

_NAMESPACED_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_])([A-Za-z][A-Za-z0-9_-]*):"
    r"([^\s,;()\[\]{}<>]+)"
)


@dataclass(frozen=True)
class Eligibility:
    """Profile eligibility plus deterministic, machine-readable reject counts."""

    eligible: bool
    rejection_reason_counts: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {
            str(reason): int(count)
            for reason, count in sorted(self.rejection_reason_counts.items())
            if int(count) > 0
        }
        if self.eligible == bool(normalized):
            raise ValueError(
                "eligible must be true exactly when rejection_reason_counts is empty"
            )
        object.__setattr__(
            self, "rejection_reason_counts", MappingProxyType(normalized)
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit representation."""

        return {
            "eligible": self.eligible,
            "rejection_reason_counts": dict(self.rejection_reason_counts),
        }


@dataclass(frozen=True)
class ProjectionResult:
    """A projected deep copy, or ``None`` when the whole input is ineligible."""

    profile: str
    projected: dict[str, Any] | None
    eligibility: Eligibility

    @property
    def eligible(self) -> bool:
        return self.eligibility.eligible

    @property
    def rejection_reason_counts(self) -> Mapping[str, int]:
        return self.eligibility.rejection_reason_counts


def _result(
    value: Mapping[str, Any], profile: str, reasons: Counter[str]
) -> ProjectionResult:
    reason_counts = dict(sorted(reasons.items()))
    eligible = not reason_counts
    return ProjectionResult(
        profile=profile,
        projected=copy.deepcopy(dict(value)) if eligible else None,
        eligibility=Eligibility(
            eligible=eligible,
            rejection_reason_counts=reason_counts,
        ),
    )


def _validate_profile(profile: str) -> None:
    if profile not in PROMPT_PROFILE_NAMES:
        raise ValueError(
            f"unknown prompt profile {profile!r}; expected one of "
            f"{PROMPT_PROFILE_NAMES}"
        )


def _normalize_organism(organism: object) -> str:
    value = str(organism).strip() if organism is not None else ""
    if not value:
        raise ValueError("organism must be non-empty provenance")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", value):
        raise ValueError("organism must be a KEGG-style organism code")
    return value.casefold()


def _assess_canonical_id(
    canonical_id: object,
    organism: str,
    reasons: Counter[str],
) -> None:
    if not isinstance(canonical_id, str) or not canonical_id.strip():
        reasons["missing_canonical_id"] += 1
        return
    value = canonical_id.strip()
    if value != canonical_id or any(character.isspace() for character in value):
        reasons["invalid_canonical_id_format"] += 1
        return
    if ":" not in value:
        # A removed prefix is not evidence of a real cross-species mapping.
        reasons["missing_namespace"] += 1
        return
    namespace, identifier = value.split(":", 1)
    namespace = namespace.casefold()
    if not namespace or not identifier or ":" in identifier:
        reasons["invalid_canonical_id_format"] += 1
        return
    if namespace == organism:
        reasons["organism_specific_namespace"] += 1
        return
    if namespace in _SOURCE_NATIVE_NAMESPACES:
        reasons["source_native_namespace"] += 1
        return
    if namespace in _PATHWAY_NAMESPACES:
        reasons["pathway_namespace"] += 1
        return
    if namespace in _INTERNAL_NAMESPACES:
        reasons["internal_namespace"] += 1
        return
    if namespace not in SPECIES_NEUTRAL_KEGG_NAMESPACES:
        reasons["unknown_namespace"] += 1
        return
    if _NEUTRAL_IDENTIFIER_PATTERNS[namespace].fullmatch(identifier) is None:
        reasons["invalid_neutral_identifier"] += 1


def _assess_name(name: object, organism: str, reasons: Counter[str]) -> None:
    if not isinstance(name, str) or not name.strip():
        reasons["missing_entity_name"] += 1
        return
    value = name.strip()
    if value != name:
        reasons["invalid_entity_name_format"] += 1

    # Match a bare organism code as well as compact forms such as hsa207.
    organism_token = re.compile(
        rf"(?<![A-Za-z0-9]){re.escape(organism)}(?![A-Za-z])",
        re.IGNORECASE,
    )
    if organism_token.search(value):
        reasons["name_contains_organism_code"] += 1

    for match in _NAMESPACED_TOKEN.finditer(value):
        namespace = match.group(1).casefold()
        identifier = match.group(2)
        if namespace not in SPECIES_NEUTRAL_KEGG_NAMESPACES:
            reasons["name_contains_source_native_id"] += 1
            continue
        pattern = _NEUTRAL_IDENTIFIER_PATTERNS[namespace]
        if pattern.fullmatch(identifier) is None:
            reasons["name_contains_invalid_neutral_id"] += 1


def _assess_entity(
    entity: object,
    organism: str,
    profile: str,
    reasons: Counter[str],
) -> None:
    if not isinstance(entity, Mapping):
        reasons["malformed_entity"] += 1
        return
    canonical_id = entity.get("canonical_id")
    name = entity.get("name")
    if profile == SPECIES_NEUTRAL_IDS_NO_ORGANISM:
        _assess_canonical_id(canonical_id, organism, reasons)
        _assess_name(name, organism, reasons)
        return
    if not isinstance(canonical_id, str) or not canonical_id.strip():
        reasons["missing_canonical_id"] += 1
    if not isinstance(name, str) or not name.strip():
        reasons["missing_entity_name"] += 1


def _assess_entity_side(
    event: Mapping[str, Any],
    side: str,
    organism: str,
    profile: str,
    reasons: Counter[str],
) -> None:
    entities = event.get(side)
    if not isinstance(entities, Sequence) or isinstance(
        entities, (str, bytes, bytearray)
    ):
        reasons[f"malformed_{side}_entities"] += 1
        return
    if not entities:
        reasons[f"empty_{side}_entities"] += 1
        return
    for entity in entities:
        _assess_entity(entity, organism, profile, reasons)


def _assess_event(
    event: object,
    organism: str,
    profile: str,
    reasons: Counter[str],
) -> None:
    if not isinstance(event, Mapping):
        reasons["malformed_event"] += 1
        return
    _assess_entity_side(event, "source", organism, profile, reasons)
    _assess_entity_side(event, "target", organism, profile, reasons)


def _collect_events(
    value: object,
    events: list[object],
    reasons: Counter[str],
) -> None:
    """Collect event candidates recursively without counting any event twice."""

    if isinstance(value, Mapping):
        if "source" in value or "target" in value:
            events.append(value)
            return
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            if key == "events":
                if not isinstance(item, Sequence) or isinstance(
                    item, (str, bytes, bytearray)
                ):
                    reasons["malformed_events_collection"] += 1
                elif not item:
                    reasons["empty_events_collection"] += 1
                else:
                    events.extend(item)
                continue
            _collect_events(item, events, reasons)
        return
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for item in value:
            _collect_events(item, events, reasons)


def project_event(
    event: Mapping[str, Any],
    *,
    organism: str,
    profile: str,
) -> ProjectionResult:
    """Project one event for a prompt profile without mutating ``event``.

    P0/P1 perform only structural eligibility checks and preserve source-native
    IDs and names byte-for-byte.  P2 applies the strict neutral allowlist.
    """

    _validate_profile(profile)
    organism_code = _normalize_organism(organism)
    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    reasons: Counter[str] = Counter()
    _assess_event(event, organism_code, profile, reasons)
    return _result(event, profile, reasons)


def project_record(
    record: Mapping[str, Any],
    *,
    profile: str,
    organism: str | None = None,
) -> ProjectionResult:
    """Project all events in a record, rejecting the entire record on one leak.

    The traversal supports canonical ``layers`` records and model payloads with
    ``observed_layers`` and/or ``remaining_layers``.  If ``organism`` is not
    supplied, a top-level ``record[\"organism\"]`` value is used as provenance.
    """

    _validate_profile(profile)
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    organism_code = _normalize_organism(
        organism if organism is not None else record.get("organism")
    )
    reasons: Counter[str] = Counter()
    events: list[object] = []
    _collect_events(record, events, reasons)
    if not events:
        reasons["missing_events"] += 1
    for event in events:
        _assess_event(event, organism_code, profile, reasons)
    return _result(record, profile, reasons)


__all__ = [
    "Eligibility",
    "ProjectionResult",
    "SPECIES_NEUTRAL_KEGG_NAMESPACES",
    "project_event",
    "project_record",
]
