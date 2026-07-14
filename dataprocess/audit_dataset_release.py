#!/usr/bin/env python3
"""Generate the immutable, strict audit for a structured dataset release.

The v3.1 release contract is intentionally stronger than the legacy three-way
audit.  It verifies five biological partitions, exact prompt profiles, model
payloads, record reconstruction, and all content hashes.  Legacy keyword paths
remain accepted only so older fixtures and already-built v3 releases can still
be inspected; they are never silently represented as a v3.1 manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import stat
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import dataprocess.structured_schema as structured_schema
from dataprocess.entity_projection import project_record
from dataprocess.prompt_profiles import (
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    PROMPT_PROFILE_METADATA,
    PROMPT_PROFILE_NAMES,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
    render_pathway_question,
)
from dataprocess.release_contract import (
    AUDIT_SCHEMA_VERSION,
    OVERLAP_CONTRACT,
    PAIRED_PROMPT_PROFILES,
    PARTITIONS,
    PREFIX_HORIZONS,
    PRIMARY_PROMPT_PROFILE,
    RECORD_JSONL_NAMES,
    RELEASE_SCHEMA_VERSION,
    SOURCE_GRAPH_HASHES_NAME,
    normalized_pair,
)
from dataprocess.schemas import CSV_FIELDNAMES, canonical_pathway_family_id
from dataprocess.source_hashes import verify_source_graph_hashes
from dataprocess.structured_schema import (
    ANSWER_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    QUESTION_TYPE,
    SUBSTEP_SCHEMA_VERSION,
    SUBSTEP_SOURCE,
    chat_prompt,
    compact_json,
    graph_id_for_source,
    record_from_object,
)
from dataprocess.structured_views import build_structured_records


# ``structured_schema`` owns the v3.1 ordered header.  The fallback keeps this
# auditor importable while the schema extension lands independently; a v3.1
# manifest must still declare the exact header that was audited.
V3_CSV_FIELDNAMES = list(
    getattr(structured_schema, "V3_CSV_FIELDNAMES", CSV_FIELDNAMES)
)
LEGACY_PARTITIONS = ("train", "validation", "test")
IDENTITY_FIELDS = (
    "source_json",
    "graph_id",
    "view_id",
    "record_id",
    "base_sample_id",
)
FORBIDDEN_PROMPT_LABELS = (
    "KEGG pathway ID:",
    "Pathway title:",
    "Pathway class:",
    "Pathway block:",
    "Phenotype:",
)
FORBIDDEN_PROMPT_KEYS = (
    '"pathway_id"',
    '"pathway_family_id"',
    '"pathway_title"',
    '"pathway_class"',
    '"pathway_block"',
    '"phenotype"',
    '"phenotype_status"',
    '"source_json"',
    '"source_graph_json"',
)

csv.field_size_limit(sys.maxsize)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_sha256(path: Path) -> str:
    try:
        return file_sha256(path)
    except OSError:
        return ""


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def csv_contract_sha256(
    row: Mapping[str, object],
    fieldnames: Sequence[str] | None = None,
) -> str:
    """Hash every ordered CSV field after writer-compatible conversion."""

    maintained = list(fieldnames or V3_CSV_FIELDNAMES)
    payload = [str(row.get(name, "")) for name in maintained]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def integer(value: object, name: str) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an integer: {value!r}") from exc


def histogram_summary(histogram: Counter[int]) -> dict[str, int | float]:
    count = sum(histogram.values())
    if not count:
        return {
            "count": 0,
            "min": 0,
            "mean": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
            "max": 0,
        }
    ordered = sorted(histogram.items())

    def percentile(fraction: float) -> int:
        threshold = max(1, int((count * fraction) + 0.999999))
        cumulative = 0
        for value, frequency in ordered:
            cumulative += frequency
            if cumulative >= threshold:
                return value
        return ordered[-1][0]

    weighted = sum(value * frequency for value, frequency in ordered)
    return {
        "count": count,
        "min": ordered[0][0],
        "mean": weighted / count,
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1][0],
    }


def _entity_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"canonical_id", "name"}
        and isinstance(value.get("canonical_id"), str)
        and bool(value["canonical_id"].strip())
        and isinstance(value.get("name"), str)
        and bool(value["name"].strip())
    )


def _validate_model_event(event: object, location: str) -> tuple[int, str]:
    if not isinstance(event, dict) or set(event) != {
        "source",
        "relation",
        "target",
        "text",
    }:
        raise ValueError(f"{location} does not exactly match the model event schema")
    sources = event.get("source")
    targets = event.get("target")
    if not isinstance(sources, list) or not sources or not all(
        _entity_valid(item) for item in sources
    ):
        raise ValueError(f"{location}.source is invalid")
    if not isinstance(targets, list) or not targets or not all(
        _entity_valid(item) for item in targets
    ):
        raise ValueError(f"{location}.target is invalid")
    relation = event.get("relation")
    if not isinstance(relation, str) or not relation.strip():
        raise ValueError(f"{location}.relation is empty")
    if not isinstance(event.get("text"), str) or not event["text"].strip():
        raise ValueError(f"{location}.text is empty")
    return len(sources) + len(targets), relation


def validate_v3_answer(
    answer_text: str,
    *,
    expected_first_layer: int,
) -> tuple[int, int, int, Counter[int]]:
    """Return layer/event/entity counts after exact model-schema validation."""

    payload = json.loads(answer_text)
    if not isinstance(payload, dict):
        raise ValueError("answer is not a JSON object")
    if set(payload) != {"schema_version", "remaining_layers"}:
        raise ValueError("answer top-level keys do not exactly match v3")
    if payload.get("schema_version") != ANSWER_SCHEMA_VERSION:
        raise ValueError("answer schema_version is not pathway_continuation_v3")
    layers = payload.get("remaining_layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("remaining_layers must be a non-empty list")

    event_count = 0
    entity_count = 0
    events_per_layer: Counter[int] = Counter()
    expected_index = expected_first_layer
    for position, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {"layer_index", "events"}:
            raise ValueError(f"remaining_layers[{position}] does not exactly match v3")
        layer_index = layer.get("layer_index")
        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise ValueError(f"remaining_layers[{position}].layer_index is not an integer")
        if layer_index != expected_index:
            raise ValueError(
                f"remaining_layers[{position}].layer_index={layer_index} "
                f"does not equal {expected_index}"
            )
        expected_index += 1
        events = layer.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"remaining_layers[{position}].events must be non-empty")
        events_per_layer[len(events)] += 1
        for event_position, event in enumerate(events):
            entities, _relation = _validate_model_event(
                event,
                f"remaining_layers[{position}].events[{event_position}]",
            )
            event_count += 1
            entity_count += entities
    return len(layers), event_count, entity_count, events_per_layer


def _observed_payload_from_question(question: str) -> dict[str, Any]:
    marker = "Observed prefix:"
    if question.count(marker) != 1:
        raise ValueError("question must contain exactly one Observed prefix marker")
    encoded = question.split(marker, 1)[1].strip()
    payload = json.loads(encoded)
    if not isinstance(payload, dict) or set(payload) != {"observed_layers"}:
        raise ValueError("Observed prefix must be exactly an observed_layers JSON object")
    layers = payload.get("observed_layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("observed_layers must be a non-empty list")
    for expected_index, layer in enumerate(layers):
        if not isinstance(layer, dict) or set(layer) != {"layer_index", "events"}:
            raise ValueError(f"observed_layers[{expected_index}] does not exactly match v3")
        if layer.get("layer_index") != expected_index:
            raise ValueError("observed layer indices must begin at zero and be consecutive")
        events = layer.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"observed_layers[{expected_index}].events must be non-empty")
        for event_index, event in enumerate(events):
            _validate_model_event(
                event,
                f"observed_layers[{expected_index}].events[{event_index}]",
            )
    return payload


def _expected_horizon(prefix_count: int, total_layers: int) -> str:
    if total_layers < 2 or not 0 < prefix_count < total_layers:
        raise ValueError("invalid prefix/record length for horizon")
    if total_layers == 2:
        return "degenerate_target"
    if prefix_count == 1:
        return "long_target"
    if prefix_count == total_layers - 1:
        return "short_target"
    if prefix_count == (total_layers + 1) // 2:
        return "middle_target"
    raise ValueError("prefix is not one of the fixed long/middle/short horizons")


def _eligible_horizons(total_layers: int) -> set[str]:
    if total_layers == 2:
        return {"degenerate_target"}
    if total_layers == 3:
        return {"long_target", "short_target"}
    return set(PREFIX_HORIZONS) - {"degenerate_target"}


@dataclass(frozen=True)
class RowSpec:
    sample_id: str
    base_sample_id: str
    record_id: str
    prefix_count: int
    profile: str
    horizon: str
    csv_contract_sha256: str


@dataclass
class SplitAudit:
    split: str
    path: Path
    graph_root: Path
    tokenizer: Any
    max_length: int
    fieldnames: Sequence[str]
    strict_v31: bool
    expected_profile: str | None = None
    max_errors: int = 100
    rows: int = 0
    header: list[str] = field(default_factory=list)
    sources: set[str] = field(default_factory=set)
    graphs: set[str] = field(default_factory=set)
    views: set[str] = field(default_factory=set)
    records: set[str] = field(default_factory=set)
    samples: set[str] = field(default_factory=set)
    base_samples: set[str] = field(default_factory=set)
    families: set[str] = field(default_factory=set)
    organisms: set[str] = field(default_factory=set)
    organism_row_counts: Counter[str] = field(default_factory=Counter)
    organism_record_ids: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    sample_counts: Counter[str] = field(default_factory=Counter)
    record_counts: Counter[str] = field(default_factory=Counter)
    record_identities: dict[str, tuple[str, str, str]] = field(default_factory=dict)
    record_identity_collisions: set[str] = field(default_factory=set)
    prefixes_by_record: dict[str, set[int]] = field(default_factory=dict)
    horizons_by_record: dict[str, set[str]] = field(default_factory=dict)
    horizon_counts: Counter[str] = field(default_factory=Counter)
    record_layer_lengths: dict[str, int] = field(default_factory=dict)
    record_family: dict[str, str] = field(default_factory=dict)
    row_specs_by_record: dict[str, list[RowSpec]] = field(default_factory=dict)
    row_source_by_record: dict[str, str] = field(default_factory=dict)
    row_view_by_record: dict[str, str] = field(default_factory=dict)
    row_graph_by_record: dict[str, str] = field(default_factory=dict)
    profile_contracts: dict[str, dict[str, str]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    profile_identity_contracts: dict[str, dict[str, str]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    phenotype_statuses: Counter[str] = field(default_factory=Counter)
    parser_sources: Counter[str] = field(default_factory=Counter)
    substep_schema_versions: Counter[str] = field(default_factory=Counter)
    total_token_lengths: Counter[int] = field(default_factory=Counter)
    prompt_token_lengths: Counter[int] = field(default_factory=Counter)
    answer_token_lengths: Counter[int] = field(default_factory=Counter)
    target_layer_lengths: Counter[int] = field(default_factory=Counter)
    events_per_layer: Counter[int] = field(default_factory=Counter)
    target_relations: Counter[str] = field(default_factory=Counter)
    target_events: int = 0
    target_entity_references: int = 0
    rows_complete_substep_schema: int = 0
    accepted_rows_over_budget: int = 0
    prompt_metadata_leak_rows: int = 0
    prompt_template_valid_rows: int = 0
    graph_artifacts_present: set[str] = field(default_factory=set)
    graph_artifacts_missing: set[str] = field(default_factory=set)
    p2_eligible_base_samples: set[str] = field(default_factory=set)
    p2_rejection_reasons: Counter[str] = field(default_factory=Counter)
    p2_rejected_row_reasons: Counter[str] = field(default_factory=Counter)
    relation_event_types: Counter[str] = field(default_factory=Counter)
    reaction_types: Counter[str] = field(default_factory=Counter)
    topology_roles: Counter[str] = field(default_factory=Counter)
    records_per_graph: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    def error(self, line_number: int, message: str) -> None:
        if len(self.errors) < self.max_errors:
            self.errors.append(f"line {line_number}: {message}")

    def process(self) -> None:
        try:
            handle = self.path.open("r", encoding="utf-8-sig", newline="")
        except OSError as exc:
            self.error(1, f"CSV cannot be opened: {exc}")
            return
        with handle:
            reader = csv.DictReader(handle)
            self.header = list(reader.fieldnames or ())
            if (
                not self.strict_v31
                and self.header in (list(CSV_FIELDNAMES), list(V3_CSV_FIELDNAMES))
            ):
                # Read-only compatibility with both generations of legacy v3
                # assets; only a v3.1 manifest is bound to one exact header.
                self.fieldnames = list(self.header)
            elif self.header != list(self.fieldnames):
                self.error(
                    1,
                    "CSV header does not exactly match the maintained ordered fields",
                )
                return
            for line_number, row in enumerate(reader, start=2):
                self.rows += 1
                try:
                    self._process_row(row)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self.error(line_number, str(exc))
        if not self.rows:
            self.error(1, "CSV has no data rows")

    def _process_row(self, row: Mapping[str, str]) -> None:
        sample_id = row.get("sample_id", "").strip()
        record_id = row.get("record_id", "").strip()
        source = row.get("source_json", "").strip()
        source_graph = row.get("source_graph_json", "").strip()
        family = row.get("pathway_family_id", "").strip()
        organism = row.get("organism", "").strip()
        view_id = (row.get("view_id", "") or row.get("pathway_block", "")).strip()
        graph_id = row.get("graph_id", "").strip()
        if not all((sample_id, record_id, source, source_graph, family, organism, view_id)):
            raise ValueError("identity/source fields must all be non-empty")
        if source != source_graph:
            raise ValueError("source_json and source_graph_json must identify the same graph")
        if self.strict_v31:
            if not graph_id:
                raise ValueError("v3.1 graph_id must be explicit and non-empty")
            if row.get("view_id", "").strip() != row.get("pathway_block", "").strip():
                raise ValueError("view_id and pathway_block must identify the same view")
        if row.get("question_type") != QUESTION_TYPE:
            raise ValueError(f"question_type must be {QUESTION_TYPE}")
        if row.get("substep_schema_version") != SUBSTEP_SCHEMA_VERSION:
            raise ValueError(f"substep_schema_version must be {SUBSTEP_SCHEMA_VERSION}")
        if row.get("substep_source") != SUBSTEP_SOURCE:
            raise ValueError(f"substep_source must be {SUBSTEP_SOURCE}")
        if family != canonical_pathway_family_id(row.get("pathway_id", "")):
            raise ValueError("pathway_family_id does not match pathway_id")
        if row.get("phenotype_status") != "not_annotated":
            raise ValueError("core release phenotype_status must be not_annotated")
        if row.get("phenotype", "").strip() or row.get("phenotype_source", "").strip():
            raise ValueError("core release phenotype columns must be empty")
        if integer(row.get("has_empty_prefix", ""), "has_empty_prefix") != 0:
            raise ValueError("release does not permit an empty observed prefix")

        prefix_count = integer(row.get("prefix_step_count", ""), "prefix_step_count")
        target_count = integer(row.get("target_step_count", ""), "target_step_count")
        total_step = integer(row.get("total_step", ""), "total_step")
        if prefix_count < 1 or target_count < 1:
            raise ValueError("prefix and target must each contain at least one layer")
        if integer(row.get("given_step", ""), "given_step") != prefix_count - 1:
            raise ValueError("given_step does not match prefix_step_count")
        declared_base_sample_id = row.get("base_sample_id", "").strip()
        if self.strict_v31 and not declared_base_sample_id:
            raise ValueError("v3.1 base_sample_id must be explicit and non-empty")
        base_sample_id = declared_base_sample_id or (
            f"{record_id}:prefix={prefix_count}"
        )
        if base_sample_id != f"{record_id}:prefix={prefix_count}":
            raise ValueError("base_sample_id does not match record_id and prefix length")
        if sample_id != base_sample_id and not sample_id.startswith(base_sample_id + ":"):
            raise ValueError("sample_id is not bound to base_sample_id")

        declared_profile = row.get("prompt_profile", "").strip()
        if self.strict_v31 and not declared_profile:
            raise ValueError("v3.1 prompt_profile must be explicit and non-empty")
        profile = declared_profile or (
            self.expected_profile or PRIMARY_PROMPT_PROFILE
        )
        if profile not in PROMPT_PROFILE_NAMES:
            raise ValueError(f"unknown prompt_profile {profile!r}")
        if self.expected_profile is not None and profile != self.expected_profile:
            raise ValueError(
                f"row prompt_profile={profile!r} does not equal declared "
                f"profile={self.expected_profile!r}"
            )
        if self.strict_v31:
            if sample_id != f"{base_sample_id}:profile={profile}":
                raise ValueError("v3.1 sample_id must bind base_sample_id and prompt_profile")
            if row.get("split", "").strip() != self.split:
                raise ValueError("row split does not match its partition")
            profile_metadata = PROMPT_PROFILE_METADATA[profile]
            for key in (
                "organism_conditioning",
                "entity_id_space",
                "entity_mapping_status",
            ):
                if row.get(key, "").strip() != profile_metadata[key]:
                    raise ValueError(f"{key} does not match prompt_profile metadata")

        question = row.get("question", "")
        answer = row.get("answer", "")
        layers, events, entities, per_layer = validate_v3_answer(
            answer,
            expected_first_layer=prefix_count,
        )
        answer_payload = json.loads(answer)
        if layers != target_count:
            raise ValueError("target_step_count does not match remaining_layers")
        record_layers = prefix_count + target_count
        if total_step + 1 != record_layers:
            raise ValueError("prefix and target counts do not reconstruct the record")
        previous_length = self.record_layer_lengths.setdefault(record_id, record_layers)
        if previous_length != record_layers:
            raise ValueError("one record_id has inconsistent total layer counts")

        if self.strict_v31:
            observed_payload = _observed_payload_from_question(question)
            if len(observed_payload["observed_layers"]) != prefix_count:
                raise ValueError("observed layer count does not match prefix_step_count")
            expected_question = render_pathway_question(
                observed_payload,
                prefix_count,
                organism,
                profile,
            )
            if question != expected_question:
                raise ValueError("question does not exactly match its declared prompt profile")
            if any(marker in question for marker in FORBIDDEN_PROMPT_LABELS + FORBIDDEN_PROMPT_KEYS):
                self.prompt_metadata_leak_rows += 1
                raise ValueError("model-visible prompt contains forbidden provenance metadata")
            explicit_line = f"Organism (KEGG code): {organism}"
            organism_lines = [
                line
                for line in question.splitlines()
                if line.startswith("Organism (KEGG code):")
            ]
            if profile == EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS:
                if organism_lines != [explicit_line]:
                    raise ValueError("P0 must show the exact organism line exactly once")
            elif organism_lines:
                raise ValueError("P1/P2 must not show an explicit organism line")
            model_pair = {
                "observed_layers": observed_payload["observed_layers"],
                "remaining_layers": answer_payload["remaining_layers"],
            }
            projection = project_record(
                model_pair,
                organism=organism,
                profile=profile,
            )
            if not projection.eligible:
                details = ",".join(
                    f"{key}={value}"
                    for key, value in projection.rejection_reason_counts.items()
                )
                raise ValueError(f"profile entity projection is ineligible: {details}")
            p2_projection = project_record(
                model_pair,
                organism=organism,
                profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM,
            )
            if p2_projection.eligible:
                self.p2_eligible_base_samples.add(base_sample_id)
            else:
                self.p2_rejection_reasons.update(p2_projection.rejection_reason_counts)
                self.p2_rejected_row_reasons.update(
                    p2_projection.rejection_reason_counts.keys()
                )
            self.prompt_template_valid_rows += 1
        else:
            legacy_markers = FORBIDDEN_PROMPT_LABELS + FORBIDDEN_PROMPT_KEYS + (
                "Organism:",
                '"organism"',
            )
            if any(marker in question for marker in legacy_markers):
                self.prompt_metadata_leak_rows += 1
                raise ValueError("legacy model prompt contains provenance metadata")

        prompt_ids = self.tokenizer.encode(chat_prompt(question), add_special_tokens=False)
        answer_ids = self.tokenizer.encode(f"{answer}<|im_end|>", add_special_tokens=False)
        total_tokens = len(prompt_ids) + len(answer_ids)
        self.prompt_token_lengths[len(prompt_ids)] += 1
        self.answer_token_lengths[len(answer_ids)] += 1
        self.total_token_lengths[total_tokens] += 1
        if total_tokens > self.max_length:
            self.accepted_rows_over_budget += 1
            raise ValueError(
                f"complete prompt+answer uses {total_tokens} tokens, "
                f"above max_length={self.max_length}"
            )

        expected_horizon = _expected_horizon(prefix_count, record_layers)
        declared_horizon = row.get("prefix_horizon", "").strip()
        if self.strict_v31 and not declared_horizon:
            raise ValueError("v3.1 prefix_horizon must be explicit and non-empty")
        horizon = declared_horizon or expected_horizon
        if horizon not in PREFIX_HORIZONS or horizon != expected_horizon:
            raise ValueError("prefix_horizon does not match the fixed horizon contract")

        identity = (source_graph, graph_id, view_id)
        previous_identity = self.record_identities.setdefault(record_id, identity)
        if previous_identity != identity:
            self.record_identity_collisions.add(record_id)
            raise ValueError("record_id maps to multiple graph/view identities")
        self.sample_counts[sample_id] += 1
        if self.sample_counts[sample_id] > 1:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        self.record_counts[record_id] += 1
        self.prefixes_by_record.setdefault(record_id, set()).add(prefix_count)
        self.horizons_by_record.setdefault(record_id, set()).add(horizon)
        self.record_family[record_id] = family
        self.row_source_by_record[record_id] = source_graph
        self.row_view_by_record[record_id] = view_id
        if graph_id:
            self.row_graph_by_record[record_id] = graph_id
            self.graphs.add(graph_id)
        spec = RowSpec(
            sample_id=sample_id,
            base_sample_id=base_sample_id,
            record_id=record_id,
            prefix_count=prefix_count,
            profile=profile,
            horizon=horizon,
            csv_contract_sha256=csv_contract_sha256(row, self.fieldnames),
        )
        self.row_specs_by_record.setdefault(record_id, []).append(spec)
        answer_contract = hashlib.sha256(compact_json(answer_payload).encode("utf-8")).hexdigest()
        previous_answer = self.profile_contracts[profile].setdefault(
            base_sample_id,
            answer_contract,
        )
        if previous_answer != answer_contract:
            raise ValueError("one profile/base_sample_id has conflicting answers")
        identity_contract = hashlib.sha256(
            compact_json(
                {
                    "record_id": record_id,
                    "source_graph_json": source_graph,
                    "graph_id": graph_id,
                    "view_id": view_id,
                    "pathway_family_id": family,
                    "organism": organism,
                    "prefix_step_count": prefix_count,
                    "prefix_horizon": horizon,
                    "split": self.split,
                }
            ).encode("utf-8")
        ).hexdigest()
        previous_identity_contract = self.profile_identity_contracts[profile].setdefault(
            base_sample_id,
            identity_contract,
        )
        if previous_identity_contract != identity_contract:
            raise ValueError("one profile/base_sample_id has conflicting identity metadata")

        self.sources.add(source_graph)
        self.views.add(view_id)
        self.records.add(record_id)
        self.samples.add(sample_id)
        self.base_samples.add(base_sample_id)
        self.families.add(family)
        self.organisms.add(organism)
        self.organism_row_counts[organism] += 1
        self.organism_record_ids[organism].add(record_id)
        self.phenotype_statuses[row.get("phenotype_status", "")] += 1
        self.parser_sources[row.get("substep_source", "")] += 1
        self.substep_schema_versions[row.get("substep_schema_version", "")] += 1
        self.target_layer_lengths[layers] += 1
        self.events_per_layer.update(per_layer)
        self.horizon_counts[horizon] += 1
        self.target_events += events
        self.target_entity_references += entities
        self.rows_complete_substep_schema += 1
        for layer in answer_payload["remaining_layers"]:
            for event in layer["events"]:
                self.target_relations[event["relation"]] += 1
        artifact = self.graph_root / source_graph
        if artifact.is_file():
            self.graph_artifacts_present.add(source_graph)
        else:
            self.graph_artifacts_missing.add(source_graph)

    def finalize_policy(
        self,
        manifest_split: Mapping[str, Any],
        family_cap: int | None,
    ) -> None:
        if self.strict_v31 and self.split == "validation":
            counts = [
                self.horizon_counts[name]
                for name in PREFIX_HORIZONS
                if name != "degenerate_target"
            ]
            if counts and max(counts) - min(counts) > 1:
                self.errors.append("validation prefix horizons are not globally balanced")
            if any(len(values) != 1 for values in self.horizons_by_record.values()):
                self.errors.append("validation must select exactly one horizon per record")
        if self.strict_v31 and self.split.startswith("test"):
            for record_id, observed in self.horizons_by_record.items():
                expected = _eligible_horizons(self.record_layer_lengths[record_id])
                if observed != expected:
                    self.errors.append(
                        f"test record {record_id} does not contain every eligible horizon"
                    )
                    break
        declared_horizons = manifest_split.get("prefix_horizons")
        if isinstance(declared_horizons, Mapping):
            declared = {str(key): int(value) for key, value in declared_horizons.items()}
            if declared != dict(sorted(self.horizon_counts.items())):
                self.errors.append("manifest prefix_horizons do not match CSV")
        if family_cap:
            counts = Counter(self.record_family.values())
            if counts and max(counts.values()) > family_cap:
                self.errors.append("accepted records exceed the manifest family cap")
        observed_family_max = max(Counter(self.record_family.values()).values(), default=0)
        declared_family_max = manifest_split.get("maximum_records_in_one_family")
        if declared_family_max is not None and int(declared_family_max) != observed_family_max:
            self.errors.append("manifest maximum_records_in_one_family does not match CSV")
        observed_view_max = max(self.records_per_graph.values(), default=0)
        declared_view_max = manifest_split.get("maximum_views_per_graph")
        if declared_view_max is not None and int(declared_view_max) != observed_view_max:
            self.errors.append("manifest maximum_views_per_graph does not match record JSONL")

    def report(self, manifest_split: Mapping[str, Any]) -> dict[str, Any]:
        duplicate_samples = sorted(
            sample_id for sample_id, count in self.sample_counts.items() if count > 1
        )
        repeated_records = sorted(
            record_id for record_id, count in self.record_counts.items() if count > 1
        )
        dropped = int(manifest_split.get("rows_dropped_token_budget", 0) or 0)
        denominator = self.rows + dropped
        actual_sha = _safe_sha256(self.path)
        expected_sha = str(manifest_split.get("csv_sha256", ""))
        if expected_sha and expected_sha != actual_sha:
            self.errors.append("CSV SHA-256 does not match dataset_manifest.json")
        actual_counts = {
            "rows": self.rows,
            "records": len(self.records),
            "sources": len(self.sources),
            "graphs": len(self.graphs),
            "families": len(self.families),
            "organisms": len(self.organisms),
        }
        for name, actual in actual_counts.items():
            expected = manifest_split.get(name)
            if expected is not None and int(expected) != actual:
                self.errors.append(
                    f"manifest {name}={expected} does not match audited value={actual}"
                )
        family_record_counts = Counter(self.record_family.values())
        return {
            "path": str(self.path),
            "sha256": actual_sha,
            "header": self.header,
            "rows": self.rows,
            "records": len(self.records),
            "source_json": len(self.sources),
            "graphs": len(self.graphs),
            "views": len(self.views),
            "families": len(self.families),
            "family_values": sorted(self.families),
            "organisms": len(self.organisms),
            "organism_values": sorted(self.organisms),
            "organism_distribution": {
                organism: {
                    "rows": self.organism_row_counts[organism],
                    "records": len(self.organism_record_ids[organism]),
                }
                for organism in sorted(self.organisms)
            },
            "phenotype_status": dict(sorted(self.phenotype_statuses.items())),
            "parser_source": dict(sorted(self.parser_sources.items())),
            "substep_schema_version": dict(sorted(self.substep_schema_versions.items())),
            "prompt_profiles": {
                profile: len(values)
                for profile, values in sorted(self.profile_contracts.items())
            },
            "prompt_template_valid_rows": self.prompt_template_valid_rows,
            "prefix_horizons": dict(sorted(self.horizon_counts.items())),
            "duplicate_ids": {
                "sample_id_duplicate_count": len(duplicate_samples),
                "sample_id_duplicate_examples": duplicate_samples[:20],
                "record_id_repeated_for_multiple_prefixes_count": len(repeated_records),
                "record_id_repeated_for_multiple_prefixes_examples": repeated_records[:20],
                "record_id_identity_collision_count": len(self.record_identity_collisions),
                "record_id_identity_collision_examples": sorted(self.record_identity_collisions)[:20],
            },
            "coverage_policy": {
                "maximum_records_in_one_family": max(family_record_counts.values(), default=0),
                "maximum_views_per_graph": max(self.records_per_graph.values(), default=0),
                "records_per_graph_distribution": histogram_summary(
                    Counter(self.records_per_graph.values())
                ),
            },
            "substep_coverage": {
                "rows_with_complete_structured_events": self.rows_complete_substep_schema,
                "row_fraction": (
                    self.rows_complete_substep_schema / self.rows if self.rows else None
                ),
                "target_layers": sum(self.target_layer_lengths.elements()),
                "target_events": self.target_events,
                "target_entity_references": self.target_entity_references,
            },
            "layer_length_distribution": {
                "record_total_layers": histogram_summary(
                    Counter(self.record_layer_lengths.values())
                ),
                "row_target_layers": histogram_summary(self.target_layer_lengths),
                "events_per_target_layer": histogram_summary(self.events_per_layer),
            },
            "event_distribution": {
                "target_relation_labels": dict(sorted(self.target_relations.items())),
                "record_event_types": dict(sorted(self.relation_event_types.items())),
                "record_reaction_types": dict(sorted(self.reaction_types.items())),
                "record_topology_roles": dict(sorted(self.topology_roles.items())),
            },
            "token_length_distribution": {
                "prompt": histogram_summary(self.prompt_token_lengths),
                "closed_answer_including_end_token": histogram_summary(self.answer_token_lengths),
                "complete_prompt_and_answer": histogram_summary(self.total_token_lengths),
            },
            "truncation_estimate": {
                "max_length": self.max_length,
                "accepted_rows_over_budget": self.accepted_rows_over_budget,
                "rows_dropped_during_materialization": dropped,
                "estimated_pre_filter_row_fraction": (
                    dropped / denominator if denominator else None
                ),
                "policy": "drop the sample before training; never truncate assistant JSON",
            },
            "graph_artifact_coverage": {
                "referenced_unique_source_json": len(self.sources),
                "present": len(self.graph_artifacts_present),
                "missing": len(self.graph_artifacts_missing),
                "missing_examples": sorted(self.graph_artifacts_missing)[:20],
                "coverage_fraction": (
                    len(self.graph_artifacts_present) / len(self.sources)
                    if self.sources
                    else None
                ),
            },
            "species_neutral_projection_coverage": {
                "eligible_base_samples": len(self.p2_eligible_base_samples),
                "total_base_samples": len(self.base_samples),
                "rejection_reason_counts": dict(sorted(self.p2_rejection_reasons.items())),
                "rejected_row_reason_counts": dict(
                    sorted(self.p2_rejected_row_reasons.items())
                ),
            },
            "prompt_metadata_leak_rows": self.prompt_metadata_leak_rows,
            "errors": self.errors,
        }


def _expected_csv_row(record: Any, spec: RowSpec, audit: SplitAudit) -> dict[str, object]:
    parameters = inspect.signature(structured_schema.csv_row).parameters
    kwargs: dict[str, object] = {}
    if audit.strict_v31:
        for name in ("prompt_profile", "profile"):
            if name in parameters:
                kwargs[name] = spec.profile
                break
        for name in ("prefix_horizon", "horizon"):
            if name in parameters:
                kwargs[name] = spec.horizon
                break
        if "split" in parameters:
            kwargs["split"] = audit.split
    row = dict(structured_schema.csv_row(record, spec.prefix_count, **kwargs))
    if audit.strict_v31:
        observed = structured_schema.observed_payload(record.layers[: spec.prefix_count])
        row["question"] = render_pathway_question(
            observed,
            spec.prefix_count,
            record.organism,
            spec.profile,
        )
    additions: dict[str, object] = {
        "sample_id": spec.sample_id,
        "base_sample_id": spec.base_sample_id,
        "prompt_profile": spec.profile,
        "graph_id": record.graph_id,
        "view_id": record.view_id,
    }
    if audit.strict_v31:
        additions["prefix_horizon"] = spec.horizon
    for key, value in additions.items():
        if key in audit.fieldnames:
            row[key] = value
    return {field: row.get(field, "") for field in audit.fieldnames}


def audit_record_jsonl(
    path: Path,
    csv_audit: SplitAudit,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    errors: list[str] = []
    if not path.is_file():
        return {
            "path": str(path),
            "records": 0,
            "sha256": "",
            "errors": ["record JSONL is missing"],
        }
    record_ids: set[str] = set()
    sources: set[str] = set()
    families: set[str] = set()
    total_layers = 0
    total_events = 0
    excluded_events = 0
    graph_identity_cache: dict[str, str] = {}
    canonical_record_hash_cache: dict[str, dict[str, str]] = {}
    csv_record_contract_mismatches = 0
    graph_identity_mismatches = 0
    canonical_record_mismatches = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict) or value.get("schema_version") != DATASET_SCHEMA_VERSION:
                    raise ValueError("unsupported record schema_version")
                record = record_from_object(value)
                if record.record_id in record_ids:
                    raise ValueError(f"duplicate record_id {record.record_id}")
                if len(record.layers) < 2:
                    raise ValueError("record must contain at least two layers")
                if record.graph_missing_endpoint_event_count:
                    raise ValueError("record contains rejected graph events")
                artifact = csv_audit.graph_root / record.source_graph_json
                if artifact.is_file():
                    expected_graph_id = graph_identity_cache.get(record.source_graph_json)
                    if expected_graph_id is None:
                        raw_graph_json = artifact.read_bytes()
                        expected_graph_id = graph_id_for_source(
                            record.source_graph_json,
                            raw_graph_json,
                        )
                        graph_identity_cache[record.source_graph_json] = expected_graph_id
                        graph_value = json.loads(raw_graph_json)
                        canonical_record_hash_cache[record.source_graph_json] = {
                            candidate.record_id: hashlib.sha256(
                                compact_json(candidate.record_object()).encode("utf-8")
                            ).hexdigest()
                            for candidate in build_structured_records(
                                graph_value,
                                graph_id=expected_graph_id,
                                source_graph_json=record.source_graph_json,
                            )
                        }
                    if record.graph_id != expected_graph_id:
                        graph_identity_mismatches += 1
                        raise ValueError("record graph_id does not match source path and content")
                    canonical_hash = canonical_record_hash_cache[record.source_graph_json].get(
                        record.record_id
                    )
                    actual_hash = hashlib.sha256(
                        compact_json(record.record_object()).encode("utf-8")
                    ).hexdigest()
                    if canonical_hash != actual_hash:
                        canonical_record_mismatches += 1
                        raise ValueError(
                            "record does not exactly match the canonical view rebuilt from source"
                        )
                if csv_audit.row_source_by_record.get(record.record_id) != record.source_graph_json:
                    raise ValueError("CSV and record JSONL source identity disagree")
                if csv_audit.row_view_by_record.get(record.record_id) != record.view_id:
                    raise ValueError("CSV and record JSONL view identity disagree")
                row_graph = csv_audit.row_graph_by_record.get(record.record_id)
                if row_graph and row_graph != record.graph_id:
                    raise ValueError("CSV and record JSONL graph identity disagree")
                if csv_audit.record_family.get(record.record_id) != record.family:
                    raise ValueError("CSV and record JSONL family disagree")

                event_ids: set[str] = set()
                for expected_layer, layer in enumerate(record.layers):
                    if layer.layer_index != expected_layer or not layer.events:
                        raise ValueError("record layers must be consecutive and contain events")
                    for event in layer.events:
                        if event.event_id in event_ids:
                            raise ValueError(f"duplicate event_id inside view: {event.event_id}")
                        event_ids.add(event.event_id)
                        total_events += 1
                        csv_audit.relation_event_types[event.event_type] += 1
                        if event.event_type == "reaction":
                            csv_audit.reaction_types[event.raw_reaction_type] += 1
                        csv_audit.topology_roles[event.topology_role] += 1
                for event in record.excluded_events:
                    if event.event_id in event_ids:
                        raise ValueError("excluded event duplicates a core event")
                    event_ids.add(event.event_id)
                    excluded_events += 1
                    csv_audit.relation_event_types[f"excluded:{event.event_type}"] += 1
                    csv_audit.topology_roles[event.topology_role] += 1

                for spec in csv_audit.row_specs_by_record.get(record.record_id, []):
                    expected_row = _expected_csv_row(record, spec, csv_audit)
                    if csv_contract_sha256(expected_row, csv_audit.fieldnames) != spec.csv_contract_sha256:
                        csv_record_contract_mismatches += 1
                        raise ValueError(
                            f"CSV sample {spec.sample_id} does not reconstruct from record JSONL"
                        )
                record_ids.add(record.record_id)
                sources.add(record.source_graph_json)
                families.add(record.family)
                csv_audit.graphs.add(record.graph_id)
                csv_audit.views.add(record.view_id)
                csv_audit.records_per_graph[record.graph_id] += 1
                total_layers += len(record.layers)
            except (json.JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
                if len(errors) < 100:
                    errors.append(f"line {line_number}: {exc}")
    actual_sha = _safe_sha256(path)
    if expected_sha256 and expected_sha256 != actual_sha:
        errors.append("record JSONL SHA-256 does not match dataset_manifest.json")
    if record_ids != csv_audit.records:
        errors.append("record JSONL record_id set does not match CSV")
    if sources != csv_audit.sources:
        errors.append("record JSONL source_json set does not match CSV")
    if families != csv_audit.families:
        errors.append("record JSONL family set does not match CSV")
    return {
        "path": str(path),
        "sha256": actual_sha,
        "records": len(record_ids),
        "source_json": len(sources),
        "families": len(families),
        "layers": total_layers,
        "events": total_events,
        "excluded_events": excluded_events,
        "csv_record_contract_mismatches": csv_record_contract_mismatches,
        "graph_identity_mismatches": graph_identity_mismatches,
        "canonical_record_mismatches": canonical_record_mismatches,
        "errors": errors,
    }


def overlap_report(left: SplitAudit, right: SplitAudit) -> dict[str, Any]:
    fields = {
        "source_json": left.sources & right.sources,
        "graph_id": left.graphs & right.graphs,
        "view_id": left.views & right.views,
        "record_id": left.records & right.records,
        "base_sample_id": left.base_samples & right.base_samples,
        "family": left.families & right.families,
        "organism": left.organisms & right.organisms,
    }
    return {
        name: {"count": len(values), "examples": sorted(values)[:20]}
        for name, values in fields.items()
    }


def _set_contract_result(
    policy: str,
    left_values: set[str],
    right_values: set[str],
) -> dict[str, Any]:
    intersection = left_values & right_values
    if policy == "allowed":
        passed = True
    elif policy == "forbidden":
        passed = not intersection
    elif policy == "required":
        passed = bool(intersection)
    elif policy == "required_equal":
        passed = bool(left_values) and left_values == right_values
    else:
        raise ValueError(f"unknown overlap policy {policy!r}")
    return {
        "policy": policy,
        "passed": passed,
        "left_count": len(left_values),
        "right_count": len(right_values),
        "intersection_count": len(intersection),
        "intersection_examples": sorted(intersection)[:20],
        "sets_equal": left_values == right_values,
    }


def _overlap_contract_reports(
    audits: Mapping[str, SplitAudit],
) -> tuple[dict[str, Any], list[str]]:
    reports: dict[str, Any] = {}
    failures: list[str] = []
    ordered = [name for name in PARTITIONS if name in audits]
    for left_index, left in enumerate(ordered):
        for right in ordered[left_index + 1 :]:
            pair = normalized_pair(left, right)
            name = f"{pair[0]}_vs_{pair[1]}"
            overlap = overlap_report(audits[pair[0]], audits[pair[1]])
            identity_results = {
                field: {
                    "policy": "forbidden",
                    "passed": overlap[field]["count"] == 0,
                    **overlap[field],
                }
                for field in IDENTITY_FIELDS
            }
            for field, result in identity_results.items():
                if not result["passed"]:
                    failures.append(f"{name}:{field}_overlap")
            biological: dict[str, Any] = {}
            contract = OVERLAP_CONTRACT[pair]
            for field, policy in contract.items():
                attribute = "families" if field == "family" else "organisms"
                left_values = getattr(audits[pair[0]], attribute)
                right_values = getattr(audits[pair[1]], attribute)
                result = _set_contract_result(policy, left_values, right_values)
                biological[field] = result
                if not result["passed"]:
                    failures.append(f"{name}:{field}_{policy}_contract_failed")
            reports[name] = {
                **overlap,
                "identity_contract": identity_results,
                "biological_contract": biological,
            }
    return reports, failures


def _declared_file_path(value: object, parent: Path) -> tuple[Path | None, str]:
    expected_sha = ""
    if isinstance(value, Mapping):
        raw_path = value.get("path") or value.get("file")
        expected_sha = str(value.get("csv_sha256") or value.get("sha256") or "")
    else:
        raw_path = value
    if not raw_path:
        return None, expected_sha
    path = Path(str(raw_path))
    return (path if path.is_absolute() else parent / path), expected_sha


def _paired_files(section: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("files", "profiles", "profile_partition_files"):
        value = section.get(key)
        if isinstance(value, Mapping):
            return value
    direct = {
        profile: section[profile]
        for profile in PAIRED_PROMPT_PROFILES
        if isinstance(section.get(profile), Mapping)
    }
    return direct


def _profile_partition_map(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    partitions = value.get("partitions")
    return partitions if isinstance(partitions, Mapping) else value


def _base_sample_set_sha256(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()


EXPECTED_PAIRED_PROFILE_CONTRACTS: dict[str, dict[str, Any]] = {
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS: {
        "published": True,
        "base_sample_contract": "exact_primary_set",
        "answer_contract": "exact_primary_answer",
        "species_claim": "no_explicit_name_only_native_ids_can_leak_species",
    },
    SPECIES_NEUTRAL_IDS_NO_ORGANISM: {
        "published": True,
        "base_sample_contract": "strict_natural_neutral_subset",
        "answer_contract": "exact_primary_answer_on_shared_base_samples",
        "mapping_contract": "no_prefix_stripping_or_synthetic_mapping",
    },
}


def _audit_paired_profiles(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    primary_audits: Mapping[str, SplitAudit],
    *,
    graph_root: Path,
    tokenizer: Any,
    max_length: int,
    fieldnames: Sequence[str],
    strict_v31: bool,
) -> tuple[dict[str, Any], list[str]]:
    raw_section = manifest.get("paired_prompt_profiles")
    section = raw_section if isinstance(raw_section, Mapping) else {}
    status_text = str(section.get("status", "")).strip().casefold()
    published = section.get("published") is True or status_text == "published"
    reasons: list[str] = []
    raw_reasons = section.get("reasons")
    if isinstance(raw_reasons, list):
        reasons.extend(str(value) for value in raw_reasons if str(value).strip())
    elif section.get("reason"):
        reasons.append(str(section["reason"]))
    if not published and not reasons:
        reasons.extend(
            (
                "paired_profile_assets_not_declared_as_complete",
                "species_neutral_profile_requires_complete_natural_neutral_entities_or_reviewed_mapping",
            )
        )

    failures: list[str] = []
    canonical_files = _paired_files(section)
    compatibility_value = manifest.get("prompt_controls")
    compatibility_files = (
        compatibility_value if isinstance(compatibility_value, Mapping) else {}
    )
    compatibility_match: bool | None = None
    if canonical_files and compatibility_files:
        compatibility_match = canonical_files == compatibility_files
        if not compatibility_match:
            failures.append("paired_profiles:prompt_controls_manifest_mismatch")
    elif canonical_files or compatibility_files:
        compatibility_match = False
        if strict_v31 and published:
            failures.append("paired_profiles:canonical_or_compatibility_files_missing")
    files = canonical_files or compatibility_files

    declared_profile_contracts = section.get("profile_contracts")
    declared_profile_contracts = (
        declared_profile_contracts
        if isinstance(declared_profile_contracts, Mapping)
        else {}
    )
    profile_contract_report: dict[str, Any] = {}
    for profile, expected in EXPECTED_PAIRED_PROFILE_CONTRACTS.items():
        actual = declared_profile_contracts.get(profile)
        matches = isinstance(actual, Mapping) and dict(actual) == expected
        profile_contract_report[profile] = {
            "passed": matches,
            "expected": expected,
            "manifest_declared": dict(actual) if isinstance(actual, Mapping) else actual,
        }
        if strict_v31 and published and not matches:
            failures.append(f"paired_profiles:{profile}:profile_contract_mismatch")
    if strict_v31 and published:
        if status_text != "published" or section.get("published") is not True:
            failures.append("paired_profiles:published_status_fields_mismatch")
        expected_file_profiles = {
            NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
            SPECIES_NEUTRAL_IDS_NO_ORGANISM,
        }
        if set(files) != expected_file_profiles:
            failures.append("paired_profiles:published_files_profile_set_mismatch")

    extra_reports: dict[str, dict[str, Any]] = {}
    contracts: dict[str, dict[str, dict[str, str]]] = {
        partition: {
            profile: dict(audit.profile_contracts.get(profile, {}))
            for profile in PAIRED_PROMPT_PROFILES
            if audit.profile_contracts.get(profile)
        }
        for partition, audit in primary_audits.items()
    }
    identity_contracts: dict[str, dict[str, dict[str, str]]] = {
        partition: {
            profile: dict(audit.profile_identity_contracts.get(profile, {}))
            for profile in PAIRED_PROMPT_PROFILES
            if audit.profile_identity_contracts.get(profile)
        }
        for partition, audit in primary_audits.items()
    }
    for profile, raw_profile_paths in files.items():
        profile_name = str(profile)
        if profile_name not in PAIRED_PROMPT_PROFILES:
            failures.append(f"paired_profiles:unknown_profile:{profile_name}")
            continue
        partition_map = _profile_partition_map(raw_profile_paths)
        if published and set(partition_map) != set(primary_audits):
            failures.append(
                f"paired_profiles:{profile_name}:published_partition_set_mismatch"
            )
        for partition, raw_file in partition_map.items():
            if partition not in primary_audits:
                failures.append(f"paired_profiles:{profile_name}:unknown_partition:{partition}")
                continue
            path, expected_sha = _declared_file_path(raw_file, manifest_path.parent)
            if path is None:
                failures.append(f"paired_profiles:{profile_name}:{partition}:missing_path")
                continue
            if path == primary_audits[partition].path:
                failures.append(
                    f"paired_profiles:{profile_name}:{partition}:reuses_primary_csv"
                )
                continue
            audit = SplitAudit(
                partition,
                path,
                graph_root,
                tokenizer,
                max_length,
                fieldnames,
                strict_v31,
                expected_profile=profile_name,
            )
            audit.process()
            actual_sha = _safe_sha256(path)
            if expected_sha and expected_sha != actual_sha:
                audit.errors.append("paired CSV SHA-256 does not match manifest")
            if strict_v31 and not _is_sha256(expected_sha):
                audit.errors.append("paired CSV SHA-256 is missing or invalid")
            if isinstance(raw_file, Mapping):
                expected_values = {
                    "prompt_profile": profile_name,
                    "rows": audit.rows,
                    "records": len(audit.records),
                    "base_sample_id_sha256": _base_sample_set_sha256(
                        audit.base_samples
                    ),
                }
                for field_name, expected_value in expected_values.items():
                    if raw_file.get(field_name) != expected_value:
                        audit.errors.append(
                            f"paired manifest {field_name} does not match CSV"
                        )
                primary_rows_considered = raw_file.get("primary_rows_considered")
                if (
                    primary_rows_considered is not None
                    and int(primary_rows_considered) != primary_audits[partition].rows
                ):
                    audit.errors.append(
                        "paired manifest primary_rows_considered does not match primary CSV"
                    )
                declared_rejections = raw_file.get("profile_rejection_reasons")
                if isinstance(declared_rejections, Mapping):
                    expected_rejections = (
                        dict(
                            sorted(
                                primary_audits[
                                    partition
                                ].p2_rejected_row_reasons.items()
                            )
                        )
                        if profile_name == SPECIES_NEUTRAL_IDS_NO_ORGANISM
                        else {}
                    )
                    normalized_rejections = {
                        str(name): int(count)
                        for name, count in sorted(declared_rejections.items())
                    }
                    if normalized_rejections != expected_rejections:
                        audit.errors.append(
                            "paired manifest profile_rejection_reasons do not match primary eligibility audit"
                        )
            key = f"{profile_name}:{partition}"
            extra_reports[key] = {
                "path": str(path),
                "sha256": actual_sha,
                "rows": audit.rows,
                "records": len(audit.records),
                "base_samples": len(audit.base_samples),
                "base_sample_id_sha256": _base_sample_set_sha256(
                    audit.base_samples
                ),
                "organism_distribution": {
                    organism: {
                        "rows": audit.organism_row_counts[organism],
                        "records": len(audit.organism_record_ids[organism]),
                    }
                    for organism in sorted(audit.organisms)
                },
                "errors": audit.errors,
            }
            if audit.errors:
                failures.append(f"paired_profiles:{profile_name}:{partition}:row_validation_failed")
            contracts.setdefault(partition, {})[profile_name] = dict(
                audit.profile_contracts.get(profile_name, {})
            )
            identity_contracts.setdefault(partition, {})[profile_name] = dict(
                audit.profile_identity_contracts.get(profile_name, {})
            )

    pair_checks: dict[str, Any] = {}
    for partition, per_profile in contracts.items():
        available = [profile for profile in PAIRED_PROMPT_PROFILES if profile in per_profile]
        for left_index, left in enumerate(available):
            for right in available[left_index + 1 :]:
                left_values = per_profile[left]
                right_values = per_profile[right]
                left_ids = set(left_values)
                right_ids = set(right_values)
                shared = left_ids & right_ids
                answer_mismatches = sorted(
                    base for base in shared if left_values[base] != right_values[base]
                )
                left_identity_values = identity_contracts.get(partition, {}).get(left, {})
                right_identity_values = identity_contracts.get(partition, {}).get(right, {})
                identity_mismatches = sorted(
                    base
                    for base in shared
                    if left_identity_values.get(base) != right_identity_values.get(base)
                )
                if SPECIES_NEUTRAL_IDS_NO_ORGANISM in (left, right):
                    neutral_ids = (
                        left_ids
                        if left == SPECIES_NEUTRAL_IDS_NO_ORGANISM
                        else right_ids
                    )
                    native_ids = right_ids if left == SPECIES_NEUTRAL_IDS_NO_ORGANISM else left_ids
                    set_contract_passed = neutral_ids.issubset(native_ids)
                    set_policy = "strict_natural_neutral_subset"
                else:
                    set_contract_passed = left_ids == right_ids
                    set_policy = "exact_primary_set"
                passed = (
                    set_contract_passed
                    and not answer_mismatches
                    and not identity_mismatches
                )
                key = f"{partition}:{left}_vs_{right}"
                pair_checks[key] = {
                    "passed": passed,
                    "base_sample_policy": set_policy,
                    "left_base_samples": len(left_ids),
                    "right_base_samples": len(right_ids),
                    "missing_from_left": len(right_ids - left_ids),
                    "missing_from_right": len(left_ids - right_ids),
                    "answer_contract_mismatches": len(answer_mismatches),
                    "answer_contract_mismatch_examples": answer_mismatches[:20],
                    "identity_contract_mismatches": len(identity_mismatches),
                    "identity_contract_mismatch_examples": identity_mismatches[:20],
                }
                if not passed:
                    failures.append(f"paired_profiles:{key}:base_or_answer_mismatch")

        p0_ids = set(per_profile.get(EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS, {}))
        p1_ids = set(per_profile.get(NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS, {}))
        p2_ids = set(per_profile.get(SPECIES_NEUTRAL_IDS_NO_ORGANISM, {}))
        eligible_p2_ids = set(primary_audits[partition].p2_eligible_base_samples)
        if p1_ids and p1_ids != p0_ids:
            failures.append(f"paired_profiles:{partition}:P1_not_exact_primary_set")
        if p2_ids and p2_ids != eligible_p2_ids:
            failures.append(
                f"paired_profiles:{partition}:P2_not_exact_eligible_neutral_subset"
            )
        if published and not p2_ids:
            failures.append(f"paired_profiles:{partition}:P2_empty_published_partition")

    if published:
        for partition in primary_audits:
            per_profile = contracts.get(partition, {})
            for profile in PAIRED_PROMPT_PROFILES:
                if not per_profile.get(profile):
                    failures.append(f"paired_profiles:{partition}:{profile}:missing_coverage")
        paired_status = "failed" if failures else "passed"
    else:
        paired_status = "not_published"

    p2_eligibility = {
        partition: {
            "eligible_base_samples": len(audit.p2_eligible_base_samples),
            "total_base_samples": len(audit.base_samples),
            "rejection_reason_counts": dict(sorted(audit.p2_rejection_reasons.items())),
            "rejected_row_reason_counts": dict(
                sorted(audit.p2_rejected_row_reasons.items())
            ),
        }
        for partition, audit in primary_audits.items()
    }
    return (
        {
            "status": paired_status,
            "manifest_published": published,
            "not_published_reasons": reasons if not published else [],
            "required_profiles": list(PAIRED_PROMPT_PROFILES),
            "canonical_files_match_prompt_controls": compatibility_match,
            "profile_contracts": profile_contract_report,
            "declared_file_reports": extra_reports,
            "pair_checks": pair_checks,
            "species_neutral_eligibility_from_primary": p2_eligibility,
        },
        failures,
    )


def _manifest_set_assertions(
    manifest: Mapping[str, Any], audits: Mapping[str, SplitAudit]
) -> tuple[dict[str, Any], list[str]]:
    if set(audits) != set(PARTITIONS):
        return {}, []
    declarations = {
        "train_families": (
            set(map(str, manifest.get("train_families", []))),
            audits["train"].families,
        ),
        "validation_families": (
            set(map(str, manifest.get("validation_families", []))),
            audits["validation"].families,
        ),
        "strict_test_families_vs_test": (
            set(map(str, manifest.get("strict_test_families", []))),
            audits["test"].families,
        ),
        "strict_test_families_vs_family_only": (
            set(map(str, manifest.get("strict_test_families", []))),
            audits["test_family_only"].families,
        ),
        "test_organisms_vs_test": (
            set(map(str, manifest.get("test_organisms", []))),
            audits["test"].organisms,
        ),
        "test_organisms_vs_organism_only": (
            set(map(str, manifest.get("test_organisms", []))),
            audits["test_organism_only"].organisms,
        ),
    }
    report: dict[str, Any] = {}
    failures: list[str] = []
    for name, (declared, observed) in declarations.items():
        passed = bool(declared) and declared == observed
        report[name] = {
            "passed": passed,
            "declared_count": len(declared),
            "observed_count": len(observed),
            "missing": sorted(observed - declared)[:20],
            "extra": sorted(declared - observed)[:20],
        }
        if not passed:
            failures.append(f"manifest_set_assertion:{name}")
    return report, failures


def _source_hash_report(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    graph_root: Path,
    expected_sources: Iterable[str],
    *,
    required: bool,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    metadata_errors: list[str] = []
    metadata = manifest.get("source_graph_hashes")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    outputs = manifest.get("outputs")
    outputs = outputs if isinstance(outputs, Mapping) else {}
    declared = metadata.get("path") or outputs.get("source_graph_hashes")
    if required:
        required_metadata = {"path", "records", "sha256"}
        if not required_metadata.issubset(metadata):
            failures.append("manifest:source_graph_hash_metadata_incomplete")
            metadata_errors.append("manifest source hash metadata is incomplete")
        if not _is_sha256(metadata.get("sha256")):
            failures.append("manifest:source_graph_hash_inventory_sha256_invalid")
            metadata_errors.append("manifest source hash inventory SHA-256 is invalid")
        output_declaration = outputs.get("source_graph_hashes")
        if output_declaration and metadata.get("path") != output_declaration:
            failures.append("manifest:source_graph_hash_inventory_path_mismatch")
            metadata_errors.append("manifest source hash inventory paths disagree")
    if not declared and not required:
        candidate = manifest_path.parent / SOURCE_GRAPH_HASHES_NAME
        if not candidate.is_file():
            return {
                "status": "not_declared",
                "path": str(candidate),
                "records": 0,
                "sha256": "",
                "errors": [],
            }, []
        declared = SOURCE_GRAPH_HASHES_NAME
    if not declared:
        failures.append("manifest:source_graph_hash_inventory_not_declared")
        return {
            "status": "failed",
            "path": "",
            "records": 0,
            "sha256": "",
            "errors": ["source graph hash inventory is not declared"],
        }, failures
    inventory_path = Path(str(declared))
    if not inventory_path.is_absolute():
        inventory_path = manifest_path.parent / inventory_path
    try:
        report = verify_source_graph_hashes(
            graph_root,
            inventory_path,
            expected_sources=expected_sources,
        )
    except OSError as exc:
        report = {
            "path": str(inventory_path),
            "records": 0,
            "sha256": "",
            "errors": [str(exc)],
        }
    report["errors"].extend(metadata_errors)
    declared_sha = str(metadata.get("sha256", ""))
    if declared_sha and declared_sha != report.get("sha256"):
        report["errors"].append("source hash inventory SHA-256 does not match manifest")
    declared_records = metadata.get("records")
    if declared_records is not None and int(declared_records) != int(report.get("records", 0)):
        report["errors"].append("source hash inventory record count does not match manifest")
    report["status"] = "failed" if report["errors"] else "passed"
    if report["errors"]:
        failures.append("source_graph_hashes:verification_failed")
    return report, failures


def _write_read_only_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite immutable audit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    temporary.replace(path)


def _partition_paths(
    partition_paths: Mapping[str, Path] | None,
    *,
    train_path: Path | None,
    validation_path: Path | None,
    test_path: Path | None,
    test_family_only_path: Path | None,
    test_organism_only_path: Path | None,
) -> dict[str, Path]:
    if partition_paths is not None:
        return {str(name): Path(path) for name, path in partition_paths.items()}
    values = {
        "train": train_path,
        "validation": validation_path,
        "test": test_path,
        "test_family_only": test_family_only_path,
        "test_organism_only": test_organism_only_path,
    }
    return {name: Path(path) for name, path in values.items() if path is not None}


def generate_release_audit(
    *,
    graph_root: Path,
    manifest_path: Path,
    tokenizer: Any,
    max_length: int,
    output_path: Path,
    overwrite: bool,
    partition_paths: Mapping[str, Path] | None = None,
    train_path: Path | None = None,
    validation_path: Path | None = None,
    test_path: Path | None = None,
    test_family_only_path: Path | None = None,
    test_organism_only_path: Path | None = None,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    graph_root = Path(graph_root)
    output_path = Path(output_path)
    failures: list[str] = []
    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw_manifest, dict):
            raise ValueError("manifest is not a JSON object")
        manifest: dict[str, Any] = raw_manifest
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        manifest = {}
        failures.append(f"manifest:cannot_load:{exc}")

    strict_v31 = manifest.get("schema_version") == RELEASE_SCHEMA_VERSION
    dataset_build_id = str(manifest.get("dataset_build_id", ""))
    if not dataset_build_id.startswith("dataset:") or len(dataset_build_id) != 32:
        failures.append("manifest:missing_or_invalid_dataset_build_id")
    paths = _partition_paths(
        partition_paths,
        train_path=train_path,
        validation_path=validation_path,
        test_path=test_path,
        test_family_only_path=test_family_only_path,
        test_organism_only_path=test_organism_only_path,
    )
    unknown_partitions = set(paths) - set(PARTITIONS)
    if unknown_partitions:
        failures.append("paths:unknown_partitions")
        paths = {name: path for name, path in paths.items() if name in PARTITIONS}
    if strict_v31:
        if set(paths) != set(PARTITIONS):
            failures.append("paths:v3.1_requires_exactly_five_partitions")
        if max_length != 8192 or int(manifest.get("max_length", 0) or 0) != 8192:
            failures.append("manifest:v3.1_requires_max_length_8192")
        declared_header = manifest.get("csv_header", manifest.get("csv_fieldnames"))
        if declared_header != V3_CSV_FIELDNAMES:
            failures.append("manifest:csv_header_not_exact_v3.1_header")
        if manifest.get("primary_prompt_profile") != PRIMARY_PROMPT_PROFILE:
            failures.append("manifest:primary_prompt_profile_mismatch")
        if set(manifest.get("splits", {})) != set(PARTITIONS):
            failures.append("manifest:v3.1_splits_are_not_exactly_five_partitions")
    elif not set(LEGACY_PARTITIONS).issubset(paths):
        failures.append("paths:legacy_audit_requires_train_validation_test")

    fieldnames = V3_CSV_FIELDNAMES if strict_v31 else list(CSV_FIELDNAMES)
    manifest_splits = manifest.get("splits")
    manifest_splits = manifest_splits if isinstance(manifest_splits, Mapping) else {}
    audits: dict[str, SplitAudit] = {}
    for partition in PARTITIONS:
        if partition not in paths:
            continue
        split_manifest = manifest_splits.get(partition)
        split_manifest = split_manifest if isinstance(split_manifest, Mapping) else {}
        expected_profile = str(
            split_manifest.get(
                "prompt_profile",
                manifest.get("primary_prompt_profile", PRIMARY_PROMPT_PROFILE),
            )
        )
        if strict_v31:
            if expected_profile != PRIMARY_PROMPT_PROFILE:
                failures.append(f"manifest:{partition}:prompt_profile_mismatch")
            if split_manifest.get("prompt_profile_interface_applied") is not True:
                failures.append(f"manifest:{partition}:prompt_profile_interface_not_applied")
            if split_manifest.get("prefix_horizon_interface_applied") is not True:
                failures.append(f"manifest:{partition}:prefix_horizon_interface_not_applied")
            for hash_field in ("csv_sha256", "records_sha256"):
                if not _is_sha256(split_manifest.get(hash_field)):
                    failures.append(f"manifest:{partition}:{hash_field}_invalid_or_missing")
        audit = SplitAudit(
            partition,
            paths[partition],
            graph_root,
            tokenizer,
            max_length,
            fieldnames,
            strict_v31,
            expected_profile=expected_profile if strict_v31 else None,
        )
        audit.process()
        audits[partition] = audit

    family_cap_value = manifest.get("max_records_per_family")
    try:
        family_cap = int(family_cap_value) if family_cap_value is not None else None
    except (TypeError, ValueError):
        family_cap = None
        failures.append("manifest:invalid_family_cap")
    if strict_v31 and (family_cap is None or family_cap < 1):
        failures.append("manifest:missing_positive_family_cap")

    manifest_outputs = manifest.get("outputs")
    manifest_outputs = manifest_outputs if isinstance(manifest_outputs, Mapping) else {}
    record_reports: dict[str, dict[str, Any]] = {}
    for partition, audit in audits.items():
        split_manifest = manifest_splits.get(partition)
        split_manifest = split_manifest if isinstance(split_manifest, Mapping) else {}
        record_name = manifest_outputs.get(
            f"{partition}_records",
            RECORD_JSONL_NAMES.get(partition, f"{partition}_pathway_records_v3.jsonl"),
        )
        record_path = Path(str(record_name))
        if not record_path.is_absolute():
            record_path = manifest_path.parent / record_path
        record_report = audit_record_jsonl(
            record_path,
            audit,
            expected_sha256=str(split_manifest.get("records_sha256", "")),
        )
        audit.errors.extend(record_report["errors"])
        audit.finalize_policy(split_manifest, family_cap)
        record_reports[partition] = record_report

    split_reports: dict[str, dict[str, Any]] = {}
    for partition, audit in audits.items():
        split_manifest = manifest_splits.get(partition)
        split_manifest = split_manifest if isinstance(split_manifest, Mapping) else {}
        report = audit.report(split_manifest)
        report["record_jsonl"] = record_reports[partition]
        split_reports[partition] = report
        if report["errors"]:
            failures.append(f"{partition}:row_or_file_validation_failed")
        if report["duplicate_ids"]["sample_id_duplicate_count"]:
            failures.append(f"{partition}:duplicate_sample_id")
        if report["duplicate_ids"]["record_id_identity_collision_count"]:
            failures.append(f"{partition}:record_id_identity_collision")
        if report["truncation_estimate"]["accepted_rows_over_budget"]:
            failures.append(f"{partition}:accepted_over_token_budget")
        if report["graph_artifact_coverage"]["missing"]:
            failures.append(f"{partition}:missing_graph_artifact")
        if report["prompt_metadata_leak_rows"]:
            failures.append(f"{partition}:model_metadata_leak")

    overlaps, overlap_failures = _overlap_contract_reports(audits)
    failures.extend(overlap_failures)
    manifest_sets, manifest_set_failures = _manifest_set_assertions(manifest, audits)
    if strict_v31:
        failures.extend(manifest_set_failures)

    paired_report, paired_failures = _audit_paired_profiles(
        manifest,
        manifest_path,
        audits,
        graph_root=graph_root,
        tokenizer=tokenizer,
        max_length=max_length,
        fieldnames=fieldnames,
        strict_v31=strict_v31,
    )
    failures.extend(paired_failures)

    expected_sources = set().union(*(audit.sources for audit in audits.values())) if audits else set()
    source_hash_report, source_hash_failures = _source_hash_report(
        manifest,
        manifest_path,
        graph_root,
        expected_sources,
        required=strict_v31,
    )
    failures.extend(source_hash_failures)

    inventory_count = int(manifest.get("inventory", {}).get("graph_files", 0) or 0) if isinstance(
        manifest.get("inventory"), Mapping
    ) else 0
    required_summary = {
        "train_test_row_counts": {
            partition: report["rows"] for partition, report in split_reports.items()
        },
        "record_counts": {
            partition: report["records"] for partition, report in split_reports.items()
        },
        "source_json_counts": {
            partition: report["source_json"] for partition, report in split_reports.items()
        },
        "family_counts": {
            partition: report["families"] for partition, report in split_reports.items()
        },
        "strict_overlap": overlaps,
        "organism_overlap": {
            pair: value["organism"] for pair, value in overlaps.items()
        },
        "duplicate_record_sample_ids": {
            partition: report["duplicate_ids"] for partition, report in split_reports.items()
        },
        "phenotype_status": {
            partition: report["phenotype_status"] for partition, report in split_reports.items()
        },
        "parser_source": {
            partition: report["parser_source"] for partition, report in split_reports.items()
        },
        "substep_coverage": {
            partition: report["substep_coverage"] for partition, report in split_reports.items()
        },
        "layer_length_distribution": {
            partition: report["layer_length_distribution"]
            for partition, report in split_reports.items()
        },
        "truncation_estimate": {
            partition: report["truncation_estimate"] for partition, report in split_reports.items()
        },
        "graph_artifact_coverage": {
            "inventory_graph_files": inventory_count,
            "splits": {
                partition: report["graph_artifact_coverage"]
                for partition, report in split_reports.items()
            },
        },
    }
    unique_failures = sorted(set(failures))
    report = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "release_schema_version": manifest.get("schema_version", "legacy_or_missing"),
        "generated_at_utc": utc_now(),
        "generator": "dataprocess/audit_dataset_release.py",
        "do_not_edit": "This file is generated and read-only. Regenerate the dataset to change it.",
        "status": "failed" if unique_failures else "passed",
        "strict_failures": unique_failures,
        "manifest": str(manifest_path),
        "manifest_sha256": _safe_sha256(manifest_path),
        "dataset_build_id": dataset_build_id,
        "max_length": max_length,
        "csv_header_contract": {
            "expected": list(fieldnames),
            "manifest_declared": manifest.get("csv_header", manifest.get("csv_fieldnames")),
        },
        "manifest_set_assertions": manifest_sets,
        "paired_prompt_profiles": paired_report,
        "source_graph_hashes": source_hash_report,
        "required_summary": required_summary,
        "splits": split_reports,
    }
    _write_read_only_json(output_path, report, overwrite=overwrite)
    if unique_failures and raise_on_failure:
        raise ValueError(
            f"strict dataset audit failed; see {output_path}: "
            + ", ".join(unique_failures)
        )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--test-family-only")
    parser.add_argument("--test-organism-only")
    parser.add_argument("--processed-graph-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    generate_release_audit(
        train_path=Path(args.train),
        validation_path=Path(args.validation),
        test_path=Path(args.test),
        test_family_only_path=(
            Path(args.test_family_only) if args.test_family_only else None
        ),
        test_organism_only_path=(
            Path(args.test_organism_only) if args.test_organism_only else None
        ),
        graph_root=Path(args.processed_graph_root),
        manifest_path=Path(args.manifest),
        tokenizer=tokenizer,
        max_length=args.max_length,
        output_path=Path(args.output),
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
