#!/usr/bin/env python3
"""Generate the immutable, strict audit for a v3 dataset release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import stat
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dataprocess.schemas import CSV_FIELDNAMES, canonical_pathway_family_id
from dataprocess.structured_schema import (
    ANSWER_SCHEMA_VERSION,
    DATASET_SCHEMA_VERSION,
    QUESTION_TYPE,
    SUBSTEP_SCHEMA_VERSION,
    SUBSTEP_SOURCE,
    chat_prompt,
    compact_json,
    csv_row,
    graph_id_for_source,
    record_from_object,
)
from dataprocess.structured_views import build_structured_records


SPLITS = ("train", "validation", "test")
PAIR_NAMES = (
    ("train", "validation"),
    ("train", "test"),
    ("validation", "test"),
)
MODEL_METADATA_KEYS = (
    '"organism"',
    '"pathway_id"',
    '"pathway_title"',
    '"pathway_class"',
    '"pathway_block"',
)
MODEL_METADATA_LABELS = (
    "Organism:",
    "KEGG pathway ID:",
    "Pathway title:",
    "Pathway class:",
    "Pathway block:",
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


def csv_contract_sha256(row: Mapping[str, object]) -> str:
    """Hash every maintained CSV field after writer-compatible string conversion."""

    payload = [str(row.get(field, "")) for field in CSV_FIELDNAMES]
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


def validate_v3_answer(
    answer_text: str,
    *,
    expected_first_layer: int,
) -> tuple[int, int, int, Counter[int]]:
    """Return layer/event/entity counts after exact v3 schema validation."""

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
                f"remaining_layers[{position}].layer_index={layer_index} does not equal {expected_index}"
            )
        expected_index += 1
        events = layer.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError(f"remaining_layers[{position}].events must be non-empty")
        events_per_layer[len(events)] += 1
        for event_position, event in enumerate(events):
            if not isinstance(event, dict) or set(event) != {
                "source",
                "relation",
                "target",
                "text",
            }:
                raise ValueError(
                    f"remaining_layers[{position}].events[{event_position}] does not exactly match v3"
                )
            sources = event.get("source")
            targets = event.get("target")
            if not isinstance(sources, list) or not sources or not all(
                _entity_valid(item) for item in sources
            ):
                raise ValueError(
                    f"remaining_layers[{position}].events[{event_position}].source is invalid"
                )
            if not isinstance(targets, list) or not targets or not all(
                _entity_valid(item) for item in targets
            ):
                raise ValueError(
                    f"remaining_layers[{position}].events[{event_position}].target is invalid"
                )
            if not isinstance(event.get("relation"), str) or not event["relation"].strip():
                raise ValueError(
                    f"remaining_layers[{position}].events[{event_position}].relation is empty"
                )
            if not isinstance(event.get("text"), str) or not event["text"].strip():
                raise ValueError(
                    f"remaining_layers[{position}].events[{event_position}].text is empty"
                )
            event_count += 1
            entity_count += len(sources) + len(targets)
    return len(layers), event_count, entity_count, events_per_layer


@dataclass
class SplitAudit:
    split: str
    path: Path
    graph_root: Path
    tokenizer: Any
    max_length: int
    max_errors: int = 100
    rows: int = 0
    sources: set[str] = field(default_factory=set)
    records: set[str] = field(default_factory=set)
    samples: set[str] = field(default_factory=set)
    families: set[str] = field(default_factory=set)
    organisms: set[str] = field(default_factory=set)
    sample_counts: Counter[str] = field(default_factory=Counter)
    record_counts: Counter[str] = field(default_factory=Counter)
    record_identities: dict[str, tuple[str, str]] = field(default_factory=dict)
    record_identity_collisions: set[str] = field(default_factory=set)
    prefixes_by_record: dict[str, set[int]] = field(default_factory=dict)
    csv_contract_by_sample: dict[str, str] = field(default_factory=dict)
    phenotype_statuses: Counter[str] = field(default_factory=Counter)
    parser_sources: Counter[str] = field(default_factory=Counter)
    substep_schema_versions: Counter[str] = field(default_factory=Counter)
    total_token_lengths: Counter[int] = field(default_factory=Counter)
    prompt_token_lengths: Counter[int] = field(default_factory=Counter)
    answer_token_lengths: Counter[int] = field(default_factory=Counter)
    target_layer_lengths: Counter[int] = field(default_factory=Counter)
    record_layer_lengths: dict[str, int] = field(default_factory=dict)
    events_per_layer: Counter[int] = field(default_factory=Counter)
    target_events: int = 0
    target_entity_references: int = 0
    rows_complete_substep_schema: int = 0
    accepted_rows_over_budget: int = 0
    prompt_metadata_leak_rows: int = 0
    graph_artifacts_present: set[str] = field(default_factory=set)
    graph_artifacts_missing: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)

    def error(self, line_number: int, message: str) -> None:
        if len(self.errors) < self.max_errors:
            self.errors.append(f"line {line_number}: {message}")

    def process(self) -> None:
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if list(reader.fieldnames or ()) != CSV_FIELDNAMES:
                self.error(
                    1,
                    "CSV header does not exactly match the maintained ordered v3 fields",
                )
                return
            for line_number, row in enumerate(reader, start=2):
                self.rows += 1
                try:
                    self._process_row(row)
                except (ValueError, json.JSONDecodeError) as exc:
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
        view_id = row.get("pathway_block", "").strip()
        if not all((sample_id, record_id, source, source_graph, family, organism, view_id)):
            raise ValueError("identity/source fields must all be non-empty")
        if source != source_graph:
            raise ValueError("source_json and source_graph_json must identify the same graph")
        if row.get("question_type") != QUESTION_TYPE:
            raise ValueError(f"question_type must be {QUESTION_TYPE}")
        if row.get("substep_schema_version") != SUBSTEP_SCHEMA_VERSION:
            raise ValueError(f"substep_schema_version must be {SUBSTEP_SCHEMA_VERSION}")
        if row.get("substep_source") != SUBSTEP_SOURCE:
            raise ValueError(f"substep_source must be {SUBSTEP_SOURCE}")
        if family != canonical_pathway_family_id(row.get("pathway_id", "")):
            raise ValueError("pathway_family_id does not match pathway_id")
        if row.get("phenotype_status") != "not_annotated":
            raise ValueError("core v3 phenotype_status must be not_annotated")
        if row.get("phenotype", "").strip() or row.get("phenotype_source", "").strip():
            raise ValueError("core v3 phenotype columns must be empty")
        if integer(row.get("has_empty_prefix", ""), "has_empty_prefix") != 0:
            raise ValueError("v3 does not permit an empty observed prefix")

        prefix_count = integer(row.get("prefix_step_count", ""), "prefix_step_count")
        target_count = integer(row.get("target_step_count", ""), "target_step_count")
        total_step = integer(row.get("total_step", ""), "total_step")
        if prefix_count < 1 or target_count < 1:
            raise ValueError("prefix and target must each contain at least one layer")
        if sample_id != f"{record_id}:prefix={prefix_count}":
            raise ValueError("sample_id does not match record_id and prefix length")

        question = row.get("question", "")
        leaked = any(marker in question for marker in MODEL_METADATA_LABELS + MODEL_METADATA_KEYS)
        if leaked:
            self.prompt_metadata_leak_rows += 1
            raise ValueError("model-visible prompt contains forbidden pathway/organism metadata")
        answer = row.get("answer", "")
        layers, events, entities, per_layer = validate_v3_answer(
            answer,
            expected_first_layer=prefix_count,
        )
        if layers != target_count:
            raise ValueError("target_step_count does not match remaining_layers")
        record_layers = prefix_count + target_count
        if total_step + 1 != record_layers:
            raise ValueError("prefix and target counts do not reconstruct the record")
        previous_length = self.record_layer_lengths.setdefault(record_id, record_layers)
        if previous_length != record_layers:
            raise ValueError("one record_id has inconsistent total layer counts")

        prompt_ids = self.tokenizer.encode(chat_prompt(question), add_special_tokens=False)
        answer_ids = self.tokenizer.encode(f"{answer}<|im_end|>", add_special_tokens=False)
        total_tokens = len(prompt_ids) + len(answer_ids)
        self.prompt_token_lengths[len(prompt_ids)] += 1
        self.answer_token_lengths[len(answer_ids)] += 1
        self.total_token_lengths[total_tokens] += 1
        if total_tokens > self.max_length:
            self.accepted_rows_over_budget += 1
            raise ValueError(
                f"complete prompt+answer uses {total_tokens} tokens, above max_length={self.max_length}"
            )

        identity = (source_graph, view_id)
        previous_identity = self.record_identities.setdefault(record_id, identity)
        if previous_identity != identity:
            self.record_identity_collisions.add(record_id)
            raise ValueError("record_id maps to multiple graph/view identities")
        self.sample_counts[sample_id] += 1
        if self.sample_counts[sample_id] > 1:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        self.record_counts[record_id] += 1
        self.prefixes_by_record.setdefault(record_id, set()).add(prefix_count)
        self.csv_contract_by_sample[sample_id] = csv_contract_sha256(row)
        self.sources.add(source_graph)
        self.records.add(record_id)
        self.samples.add(sample_id)
        self.families.add(family)
        self.organisms.add(organism)
        self.phenotype_statuses[row.get("phenotype_status", "")] += 1
        self.parser_sources[row.get("substep_source", "")] += 1
        self.substep_schema_versions[row.get("substep_schema_version", "")] += 1
        self.target_layer_lengths[layers] += 1
        self.events_per_layer.update(per_layer)
        self.target_events += events
        self.target_entity_references += entities
        self.rows_complete_substep_schema += 1
        artifact = self.graph_root / source_graph
        if artifact.is_file():
            self.graph_artifacts_present.add(source_graph)
        else:
            self.graph_artifacts_missing.add(source_graph)

    def report(self, manifest_split: Mapping[str, Any]) -> dict[str, Any]:
        duplicate_samples = sorted(
            sample_id for sample_id, count in self.sample_counts.items() if count > 1
        )
        repeated_records = sorted(
            record_id for record_id, count in self.record_counts.items() if count > 1
        )
        dropped = int(manifest_split.get("rows_dropped_token_budget", 0) or 0)
        denominator = self.rows + dropped
        expected_sha = str(manifest_split.get("csv_sha256", ""))
        actual_sha = file_sha256(self.path)
        if expected_sha and expected_sha != actual_sha:
            self.errors.append("CSV SHA-256 does not match dataset_manifest.json")
        actual_counts = {
            "rows": self.rows,
            "records": len(self.records),
            "sources": len(self.sources),
            "families": len(self.families),
            "organisms": len(self.organisms),
        }
        for name, actual in actual_counts.items():
            expected = manifest_split.get(name)
            if expected is not None and int(expected) != actual:
                self.errors.append(
                    f"manifest {name}={expected} does not match audited value={actual}"
                )
        return {
            "path": str(self.path),
            "sha256": actual_sha,
            "rows": self.rows,
            "records": len(self.records),
            "source_json": len(self.sources),
            "families": len(self.families),
            "organisms": len(self.organisms),
            "organism_values": sorted(self.organisms),
            "phenotype_status": dict(sorted(self.phenotype_statuses.items())),
            "parser_source": dict(sorted(self.parser_sources.items())),
            "substep_schema_version": dict(sorted(self.substep_schema_versions.items())),
            "duplicate_ids": {
                "sample_id_duplicate_count": len(duplicate_samples),
                "sample_id_duplicate_examples": duplicate_samples[:20],
                "record_id_repeated_for_multiple_prefixes_count": len(repeated_records),
                "record_id_repeated_for_multiple_prefixes_examples": repeated_records[:20],
                "record_id_identity_collision_count": len(self.record_identity_collisions),
                "record_id_identity_collision_examples": sorted(self.record_identity_collisions)[:20],
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
                "record_total_layers": histogram_summary(Counter(self.record_layer_lengths.values())),
                "row_target_layers": histogram_summary(self.target_layer_lengths),
                "events_per_target_layer": histogram_summary(self.events_per_layer),
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
                "estimated_pre_filter_row_fraction": dropped / denominator if denominator else None,
                "policy": "drop the sample before training; never truncate assistant JSON",
            },
            "graph_artifact_coverage": {
                "referenced_unique_source_json": len(self.sources),
                "present": len(self.graph_artifacts_present),
                "missing": len(self.graph_artifacts_missing),
                "missing_examples": sorted(self.graph_artifacts_missing)[:20],
                "coverage_fraction": (
                    len(self.graph_artifacts_present) / len(self.sources) if self.sources else None
                ),
            },
            "prompt_metadata_leak_rows": self.prompt_metadata_leak_rows,
            "errors": self.errors,
        }


def overlap_report(left: SplitAudit, right: SplitAudit) -> dict[str, Any]:
    fields = {
        "source_json": left.sources & right.sources,
        "record_id": left.records & right.records,
        "sample_id": left.samples & right.samples,
        "family": left.families & right.families,
        "organism": left.organisms & right.organisms,
    }
    return {
        name: {
            "count": len(values),
            "examples": sorted(values)[:20],
        }
        for name, values in fields.items()
    }


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
                    raise ValueError("record contains missing-endpoint graph events")
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
                            "record does not exactly match the canonical sink-SCC view rebuilt from source"
                        )
                event_ids: set[str] = set()
                for expected_layer, layer in enumerate(record.layers):
                    if layer.layer_index != expected_layer or not layer.events:
                        raise ValueError("record layers must be consecutive and contain events")
                    for event in layer.events:
                        if event.event_id in event_ids:
                            raise ValueError(f"duplicate event_id inside view: {event.event_id}")
                        event_ids.add(event.event_id)
                        total_events += 1
                prefixes = csv_audit.prefixes_by_record.get(record.record_id, set())
                for prefix_length in sorted(prefixes):
                    expected_row = csv_row(record, prefix_length)
                    sample_id = str(expected_row["sample_id"])
                    if csv_contract_sha256(expected_row) != csv_audit.csv_contract_by_sample.get(
                        sample_id
                    ):
                        csv_record_contract_mismatches += 1
                        raise ValueError(
                            f"CSV sample {sample_id} does not exactly reconstruct from record JSONL"
                        )
                record_ids.add(record.record_id)
                sources.add(record.source_graph_json)
                families.add(record.family)
                total_layers += len(record.layers)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                if len(errors) < 100:
                    errors.append(f"line {line_number}: {exc}")
    actual_sha = file_sha256(path)
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
        "csv_record_contract_mismatches": csv_record_contract_mismatches,
        "graph_identity_mismatches": graph_identity_mismatches,
        "canonical_record_mismatches": canonical_record_mismatches,
        "errors": errors,
    }
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


def generate_release_audit(
    *,
    train_path: Path,
    validation_path: Path,
    test_path: Path,
    graph_root: Path,
    manifest_path: Path,
    tokenizer: Any,
    max_length: int,
    output_path: Path,
    overwrite: bool,
    raise_on_failure: bool = True,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_failures: list[str] = []
    dataset_build_id = str(manifest.get("dataset_build_id", ""))
    if not dataset_build_id.startswith("dataset:") or len(dataset_build_id) != 32:
        manifest_failures.append("manifest:missing_or_invalid_dataset_build_id")
    paths = {
        "train": Path(train_path),
        "validation": Path(validation_path),
        "test": Path(test_path),
    }
    audits = {
        split: SplitAudit(split, paths[split], Path(graph_root), tokenizer, max_length)
        for split in SPLITS
    }
    for audit in audits.values():
        audit.process()
    overlaps = {
        f"{left}_vs_{right}": overlap_report(audits[left], audits[right])
        for left, right in PAIR_NAMES
    }
    split_reports: dict[str, dict[str, Any]] = {}
    manifest_outputs = manifest.get("outputs", {})
    for split in SPLITS:
        manifest_split = manifest.get("splits", {}).get(split, {})
        report = audits[split].report(manifest_split)
        record_name = manifest_outputs.get(
            f"{split}_records",
            f"{split}_pathway_records_v3.jsonl",
        )
        record_report = audit_record_jsonl(
            manifest_path.parent / str(record_name),
            audits[split],
            expected_sha256=str(manifest_split.get("records_sha256", "")),
        )
        report["record_jsonl"] = record_report
        report["errors"].extend(record_report["errors"])
        split_reports[split] = report

    failures: list[str] = list(manifest_failures)
    for split, report in split_reports.items():
        if report["errors"]:
            failures.append(f"{split}:row_or_file_validation_failed")
        if report["duplicate_ids"]["sample_id_duplicate_count"]:
            failures.append(f"{split}:duplicate_sample_id")
        if report["duplicate_ids"]["record_id_identity_collision_count"]:
            failures.append(f"{split}:record_id_identity_collision")
        if report["truncation_estimate"]["accepted_rows_over_budget"]:
            failures.append(f"{split}:accepted_over_token_budget")
        if report["graph_artifact_coverage"]["missing"]:
            failures.append(f"{split}:missing_graph_artifact")
        if report["prompt_metadata_leak_rows"]:
            failures.append(f"{split}:model_metadata_leak")

    for pair_name, pair in overlaps.items():
        for key in ("source_json", "record_id", "sample_id", "family"):
            if pair[key]["count"]:
                failures.append(f"{pair_name}:{key}_overlap")
    for pair_name in ("train_vs_test", "validation_vs_test"):
        if overlaps[pair_name]["organism"]["count"]:
            failures.append(f"{pair_name}:organism_overlap")

    inventory_count = int(manifest.get("inventory", {}).get("graph_files", 0) or 0)
    required_summary = {
        "train_test_row_counts": {
            split: split_reports[split]["rows"] for split in SPLITS
        },
        "record_counts": {
            split: split_reports[split]["records"] for split in SPLITS
        },
        "source_json_counts": {
            split: split_reports[split]["source_json"] for split in SPLITS
        },
        "family_counts": {
            split: split_reports[split]["families"] for split in SPLITS
        },
        "strict_overlap": overlaps,
        "organism_overlap": {
            pair: value["organism"] for pair, value in overlaps.items()
        },
        "duplicate_record_sample_ids": {
            split: split_reports[split]["duplicate_ids"] for split in SPLITS
        },
        "phenotype_status": {
            split: split_reports[split]["phenotype_status"] for split in SPLITS
        },
        "parser_source": {
            split: split_reports[split]["parser_source"] for split in SPLITS
        },
        "substep_coverage": {
            split: split_reports[split]["substep_coverage"] for split in SPLITS
        },
        "layer_length_distribution": {
            split: split_reports[split]["layer_length_distribution"] for split in SPLITS
        },
        "truncation_estimate": {
            split: split_reports[split]["truncation_estimate"] for split in SPLITS
        },
        "graph_artifact_coverage": {
            "inventory_graph_files": inventory_count,
            "splits": {
                split: split_reports[split]["graph_artifact_coverage"]
                for split in SPLITS
            },
        },
    }
    report = {
        "schema_version": "chatpathway_data_audit_v3",
        "generated_at_utc": utc_now(),
        "generator": "dataprocess/audit_dataset_release.py",
        "do_not_edit": "This file is generated and read-only. Regenerate the dataset to change it.",
        "status": "failed" if failures else "passed",
        "strict_failures": sorted(set(failures)),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "dataset_build_id": dataset_build_id,
        "max_length": max_length,
        "required_summary": required_summary,
        "splits": split_reports,
    }
    _write_read_only_json(output_path, report, overwrite=overwrite)
    if failures and raise_on_failure:
        raise ValueError(
            f"strict dataset audit failed; see {output_path}: {', '.join(sorted(set(failures)))}"
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
