"""Central, versioned contract for the structured pathway v3 release."""

from __future__ import annotations

from dataprocess.prompt_profiles import (
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
)


RELEASE_SCHEMA_VERSION = "chatpathway_structured_release_v3.1"
AUDIT_SCHEMA_VERSION = "chatpathway_data_audit_v3.1"

PARTITIONS = (
    "train",
    "validation",
    "test",
    "test_family_only",
    "test_organism_only",
)

PRIMARY_PROMPT_PROFILE = EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS
PAIRED_PROMPT_PROFILES = (
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
)

PREFIX_HORIZONS = (
    "long_target",
    "middle_target",
    "short_target",
    "degenerate_target",
)

# Source/graph/view/record/sample identity is always disjoint.  Family and
# organism overlap below is the scientific purpose of each diagnostic split,
# not a manifest option that a release can silently relax.
OVERLAP_CONTRACT: dict[tuple[str, str], dict[str, str]] = {
    ("train", "validation"): {"family": "forbidden", "organism": "allowed"},
    ("train", "test"): {"family": "forbidden", "organism": "forbidden"},
    ("train", "test_family_only"): {"family": "forbidden", "organism": "allowed"},
    ("train", "test_organism_only"): {"family": "required", "organism": "forbidden"},
    ("validation", "test"): {"family": "forbidden", "organism": "forbidden"},
    ("validation", "test_family_only"): {"family": "forbidden", "organism": "allowed"},
    ("validation", "test_organism_only"): {"family": "forbidden", "organism": "forbidden"},
    ("test", "test_family_only"): {"family": "required_equal", "organism": "forbidden"},
    ("test", "test_organism_only"): {"family": "forbidden", "organism": "required_equal"},
    ("test_family_only", "test_organism_only"): {
        "family": "forbidden",
        "organism": "forbidden",
    },
}

PRIMARY_CSV_NAMES = {
    "train": "train_pathway_continuation_v3_cap256.csv",
    "validation": "validation_pathway_continuation_v3.csv",
    "test": "test_pathway_continuation_v3.csv",
    "test_family_only": "test_family_only_pathway_continuation_v3.csv",
    "test_organism_only": "test_organism_only_pathway_continuation_v3.csv",
}

RECORD_JSONL_NAMES = {
    partition: f"{partition}_pathway_records_v3.jsonl"
    for partition in PARTITIONS
}

SOURCE_GRAPH_HASHES_NAME = "source_graph_hashes.jsonl"
MANIFEST_NAME = "dataset_manifest.json"
AUDIT_NAME = "data_audit.json"


def normalized_pair(left: str, right: str) -> tuple[str, str]:
    """Return a partition pair in canonical release order."""

    indices = {name: index for index, name in enumerate(PARTITIONS)}
    if left not in indices or right not in indices or left == right:
        raise ValueError(f"invalid distinct partition pair: {left!r}, {right!r}")
    return (left, right) if indices[left] < indices[right] else (right, left)


__all__ = [
    "AUDIT_NAME",
    "AUDIT_SCHEMA_VERSION",
    "MANIFEST_NAME",
    "OVERLAP_CONTRACT",
    "PAIRED_PROMPT_PROFILES",
    "PARTITIONS",
    "PREFIX_HORIZONS",
    "PRIMARY_CSV_NAMES",
    "PRIMARY_PROMPT_PROFILE",
    "RECORD_JSONL_NAMES",
    "RELEASE_SCHEMA_VERSION",
    "SOURCE_GRAPH_HASHES_NAME",
    "normalized_pair",
]
