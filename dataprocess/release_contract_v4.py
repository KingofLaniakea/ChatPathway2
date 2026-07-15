"""Single source of truth for the formal structured pathway v4 release."""

from __future__ import annotations

from dataprocess.prompt_profiles import EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS


RELEASE_SCHEMA_VERSION = "chatpathway_structured_release_v4.0"
AUDIT_SCHEMA_VERSION = "chatpathway_data_audit_v4.0"
PRIMARY_PROMPT_PROFILE = EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS

PRIMARY_SPLITS = ("train", "validation", "test")
DIAGNOSTIC_SPLITS = ("test_organism", "test_strict")
ALL_SPLITS = PRIMARY_SPLITS + DIAGNOSTIC_SPLITS

CSV_NAMES = {
    split: f"{split}_pathway_continuation_v4.csv" for split in ALL_SPLITS
}
RECORD_NAMES = {
    split: f"{split}_pathway_records_v4.jsonl" for split in ALL_SPLITS
}

MANIFEST_NAME = "dataset_manifest.json"
AUDIT_NAME = "data_audit.json"
SOURCE_GRAPH_HASHES_NAME = "source_graph_hashes.tsv"
SPLIT_ASSIGNMENTS_NAME = "split_assignments.json"
MATERIALIZATION_DATABASE_NAME = ".materialization_v4.sqlite3"


__all__ = [
    "ALL_SPLITS",
    "AUDIT_NAME",
    "AUDIT_SCHEMA_VERSION",
    "CSV_NAMES",
    "DIAGNOSTIC_SPLITS",
    "MANIFEST_NAME",
    "MATERIALIZATION_DATABASE_NAME",
    "PRIMARY_PROMPT_PROFILE",
    "PRIMARY_SPLITS",
    "RECORD_NAMES",
    "RELEASE_SCHEMA_VERSION",
    "SOURCE_GRAPH_HASHES_NAME",
    "SPLIT_ASSIGNMENTS_NAME",
]
