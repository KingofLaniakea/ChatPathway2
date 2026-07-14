"""Versioned prompt profiles for structured pathway continuation.

The canonical record keeps organism and source provenance outside the model
payload.  These profiles control only what is rendered into the user message.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS = "explicit_organism_source_native_ids"
NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS = (
    "no_explicit_organism_source_native_ids"
)
SPECIES_NEUTRAL_IDS_NO_ORGANISM = "species_neutral_ids_no_organism"

PROMPT_PROFILE_NAMES = (
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
)

PROMPT_PROFILE_METADATA: dict[str, dict[str, str]] = {
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS: {
        "organism_conditioning": "explicit",
        "entity_id_space": "source_native",
        "entity_mapping_status": "not_applicable",
    },
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS: {
        "organism_conditioning": "implicit_in_source_native_ids",
        "entity_id_space": "source_native",
        "entity_mapping_status": "not_applicable",
    },
    SPECIES_NEUTRAL_IDS_NO_ORGANISM: {
        "organism_conditioning": "absent_after_neutralization",
        "entity_id_space": "species_neutral_kegg",
        "entity_mapping_status": "complete",
    },
}

PATHWAY_CONTINUATION_SCHEMA_VERSION = "pathway_continuation_v3"

_FORBIDDEN_MODEL_METADATA_KEYS = frozenset(
    {
        "organism",
        "pathway_id",
        "pathway_family_id",
        "pathway_title",
        "pathway_class",
        "pathway_block",
        "phenotype",
        "phenotype_status",
        "phenotype_source",
        "source_json",
        "source_graph_json",
    }
)


def _forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold()
            if normalized in _FORBIDDEN_MODEL_METADATA_KEYS:
                found.add(normalized)
            found.update(_forbidden_keys(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.update(_forbidden_keys(item))
    return found


def _canonical_ids(value: object) -> list[str]:
    output: list[str] = []
    if isinstance(value, Mapping):
        canonical_id = value.get("canonical_id")
        if isinstance(canonical_id, str) and canonical_id.strip():
            output.append(canonical_id.strip())
        for item in value.values():
            output.extend(_canonical_ids(item))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            output.extend(_canonical_ids(item))
    return output


def _output_skeleton(next_layer_index: int) -> str:
    skeleton = {
        "schema_version": PATHWAY_CONTINUATION_SCHEMA_VERSION,
        "remaining_layers": [
            {
                "layer_index": next_layer_index,
                "events": [
                    {
                        "source": [
                            {
                                "canonical_id": "<source canonical ID>",
                                "name": "<source name>",
                            }
                        ],
                        "relation": "<controlled relation label>",
                        "target": [
                            {
                                "canonical_id": "<target canonical ID>",
                                "name": "<target name>",
                            }
                        ],
                        "text": "<biological relation sentence>",
                    }
                ],
            }
        ],
    }
    return json.dumps(skeleton, ensure_ascii=False, indent=2)


def render_pathway_question(
    observed_payload: Mapping[str, Any],
    next_layer_index: int,
    organism: str,
    profile: str,
) -> str:
    """Render one model-visible continuation question for ``profile``.

    ``observed_payload`` must contain only ``observed_layers`` at the top level.
    Provenance metadata is deliberately rejected instead of silently leaking
    into a prompt.  The species-neutral profile also rejects source-native IDs
    carrying the supplied organism prefix; full name/ID normalization remains
    the responsibility of the dataset projection that calls this renderer.
    """

    if profile not in PROMPT_PROFILE_METADATA:
        raise ValueError(
            f"unknown prompt profile {profile!r}; expected one of {PROMPT_PROFILE_NAMES}"
        )
    if isinstance(next_layer_index, bool) or not isinstance(next_layer_index, int):
        raise TypeError("next_layer_index must be an integer")
    if next_layer_index < 0:
        raise ValueError("next_layer_index must be non-negative")
    organism_value = str(organism).strip()
    if not organism_value:
        raise ValueError("organism must be non-empty provenance")
    if not isinstance(observed_payload, Mapping):
        raise TypeError("observed_payload must be a mapping")
    if set(observed_payload) != {"observed_layers"}:
        raise ValueError("observed_payload must contain exactly observed_layers")
    observed_layers = observed_payload.get("observed_layers")
    if not isinstance(observed_layers, list) or not observed_layers:
        raise ValueError("observed_layers must be a non-empty list")
    forbidden = _forbidden_keys(observed_payload)
    if forbidden:
        raise ValueError(
            "observed_payload contains forbidden model metadata keys: "
            + ", ".join(sorted(forbidden))
        )
    if profile == SPECIES_NEUTRAL_IDS_NO_ORGANISM:
        prefix = f"{organism_value.casefold()}:"
        leaked_ids = [
            value
            for value in _canonical_ids(observed_payload)
            if value.casefold().startswith(prefix)
        ]
        if leaked_ids:
            raise ValueError(
                "species-neutral profile contains source-native organism-prefixed IDs"
            )

    lines = ["Continue the biological mechanism from the observed upstream layers."]
    if profile == EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS:
        lines.extend(("", f"Organism (KEGG code): {organism_value}"))
    lines.extend(
        (
            "",
            "Return exactly one complete JSON object. Do not use Markdown or commentary. "
            "Do not add extra keys or repeat observed layers.",
            f"The first remaining layer must use layer_index {next_layer_index}; "
            "each later layer must increase it by 1.",
            "Use the exact key structure below, replacing placeholder strings and "
            "including as many layers, events, and entities as required.",
            "",
            "Required output JSON format:",
            _output_skeleton(next_layer_index),
            "",
            "Observed prefix:",
            json.dumps(
                observed_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
    )
    return "\n".join(lines)


__all__ = [
    "EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS",
    "NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS",
    "PATHWAY_CONTINUATION_SCHEMA_VERSION",
    "PROMPT_PROFILE_METADATA",
    "PROMPT_PROFILE_NAMES",
    "SPECIES_NEUTRAL_IDS_NO_ORGANISM",
    "render_pathway_question",
]
