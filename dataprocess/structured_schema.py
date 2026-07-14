"""Structured graph-event records and the prefix-only v3 model contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from dataprocess.entity_projection import project_record
from dataprocess.prompt_profiles import (
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    PROMPT_PROFILE_METADATA,
    PROMPT_PROFILE_NAMES,
    render_pathway_question,
)
from dataprocess.schemas import CSV_FIELDNAMES, canonical_pathway_family_id


DATASET_SCHEMA_VERSION = "structured_pathway_record_v3"
ANSWER_SCHEMA_VERSION = "pathway_continuation_v3"
QUESTION_TYPE = "pathway_continuation_v3"
SUBSTEP_SCHEMA_VERSION = "graph_event_set_v3"
SUBSTEP_SOURCE = "processed_graph_structured_event_v3"

# Keep the legacy text-dataset header unchanged.  The structured v3 release
# carries the identities and conditioning metadata needed to audit paired
# prompt profiles and to merge inference shards without losing provenance.
V3_CSV_FIELDNAMES = CSV_FIELDNAMES + [
    "base_sample_id",
    "graph_id",
    "view_id",
    "prompt_profile",
    "organism_conditioning",
    "entity_id_space",
    "entity_mapping_status",
    "prefix_horizon",
    "split",
]

RELATION_TYPES = frozenset({"ECrel", "PPrel", "GErel", "PCrel", "maplink"})
REACTION_TYPES = frozenset({"reversible", "irreversible"})

# The KGML manual explicitly assigns entry1 -> entry2 direction when a
# direction-bearing subtype is present.  The remaining official subtypes are
# retained as context, but they do not manufacture pathway order by themselves.
DIRECTIONAL_RELATION_SUBTYPES = frozenset(
    {"activation", "inhibition", "expression", "repression", "indirect effect"}
)
CONTEXT_RELATION_SUBTYPES = frozenset(
    {
        "binding/association",
        "dissociation",
        "state change",
        "compound",
        "hidden compound",
        "phosphorylation",
        "dephosphorylation",
        "glycosylation",
        "ubiquitination",
        "methylation",
    }
)
MISSING_RELATION_SUBTYPE = "missing interaction"

TOPOLOGY_BACKBONE = "backbone"
TOPOLOGY_CONTEXT = "context"
TOPOLOGY_CONTEXT_CROSS_LAYER = "context_cross_layer"
TOPOLOGY_EXCLUDED = "excluded"
TOPOLOGY_ROLES = frozenset(
    {
        TOPOLOGY_BACKBONE,
        TOPOLOGY_CONTEXT,
        TOPOLOGY_CONTEXT_CROSS_LAYER,
        TOPOLOGY_EXCLUDED,
    }
)


class EventValidationError(ValueError):
    """A structural event makes its whole processed_graph artifact ineligible."""


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def stable_id(namespace: str, *parts: object, length: int = 24) -> str:
    material = "\n".join(str(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest[:length]}"


def graph_id_for_source(source_graph_json: str, raw_graph_json: bytes) -> str:
    """Return a provenance-stable graph identity.

    Content alone is not a sufficient sample identity: two source artifacts can
    legitimately contain byte-identical graphs.  Binding the relative source
    path to the content digest prevents those artifacts from collapsing onto
    one ``record_id`` while still making any content change visible.
    """

    content_sha256 = hashlib.sha256(raw_graph_json).hexdigest()
    return stable_id("graph", source_graph_json, content_sha256)


def normalized_pathway_id(value: object) -> str:
    text = str(value or "").strip()
    return text.split(":", 1)[1] if text.startswith("path:") else text


def _strict_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise EventValidationError(f"{field} must be a list")
    output = tuple(str(item).strip() for item in value)
    if any(not item for item in output):
        raise EventValidationError(f"{field} contains an empty value")
    if len(set(output)) != len(output):
        raise EventValidationError(f"{field} contains duplicate values")
    return output


def _strict_integer_list(value: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise EventValidationError(f"{field} must be a list")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise EventValidationError(f"{field} contains a boolean")
        try:
            parsed = int(item)
        except (TypeError, ValueError) as exc:
            raise EventValidationError(f"{field} contains an invalid integer") from exc
        if parsed <= 0:
            raise EventValidationError(f"{field} must contain positive integers")
        output.append(parsed)
    if len(set(output)) != len(output):
        raise EventValidationError(f"{field} contains duplicate node IDs")
    return tuple(output)


def _strict_event_id(record: dict[str, Any], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool):
        raise EventValidationError(f"{key} must be a non-negative integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EventValidationError(f"{key} must be a non-negative integer") from exc
    if parsed < 0:
        raise EventValidationError(f"{key} must be a non-negative integer")
    return parsed


def _strict_renderable(record: dict[str, Any]) -> bool:
    value = record.get("renderable")
    if not isinstance(value, bool):
        raise EventValidationError("renderable must be boolean")
    return value


def _node_projection(
    node_id: int,
    node_lookup: dict[int, dict[str, Any]],
    cache: dict[int, tuple[tuple[dict[str, str], ...], dict[str, object]]],
    stack: tuple[int, ...] = (),
) -> tuple[tuple[dict[str, str], ...], dict[str, object]]:
    """Return all trusted identifiers for one graph node plus raw provenance.

    A KGML group has no canonical identifier of its own.  Such a node is
    projected to the complete, recursively validated component entity set.
    Multi-ID entries likewise remain multiple model entities instead of being
    silently collapsed to the producer's first token.
    """

    if node_id in cache:
        return cache[node_id]
    if node_id in stack:
        raise EventValidationError("group component graph contains a cycle")
    node = node_lookup.get(node_id)
    if node is None:
        raise EventValidationError(f"event endpoint node_id={node_id} is missing")
    if node.get("resolved") is not True:
        raise EventValidationError(f"event endpoint node_id={node_id} is unresolved")
    unresolved_tokens = node.get("unresolved_tokens")
    if not isinstance(unresolved_tokens, list) or unresolved_tokens:
        raise EventValidationError(
            f"event endpoint node_id={node_id} has unresolved tokens"
        )
    display_name = str(node.get("display_name") or "").strip()
    if not display_name:
        raise EventValidationError(f"event endpoint node_id={node_id} lacks display_name")

    entity_type = str(node.get("entity_type") or "").strip()
    raw_resolved_ids = node.get("resolved_ids")
    raw_component_ids = node.get("component_entry_ids")
    if entity_type == "group":
        component_ids = _strict_integer_list(
            raw_component_ids,
            field=f"node[{node_id}].component_entry_ids",
        )
        if not component_ids:
            raise EventValidationError(f"group node_id={node_id} has no components")
        model_entities: list[dict[str, str]] = []
        effective_ids: list[str] = []
        for component_id in component_ids:
            component_entities, _ = _node_projection(
                component_id,
                node_lookup,
                cache,
                stack + (node_id,),
            )
            for entity in component_entities:
                if entity["canonical_id"] in effective_ids:
                    continue
                effective_ids.append(entity["canonical_id"])
                model_entities.append(dict(entity))
        if not model_entities:
            raise EventValidationError(f"group node_id={node_id} has no resolved components")
        resolved_ids = tuple(
            str(item).strip()
            for item in raw_resolved_ids
        ) if isinstance(raw_resolved_ids, list) else ()
    else:
        resolved_ids = _strict_strings(
            raw_resolved_ids,
            field=f"node[{node_id}].resolved_ids",
        )
        if not resolved_ids:
            raise EventValidationError(f"event endpoint node_id={node_id} has no resolved_ids")
        canonical_id = node.get("canonical_id")
        if not isinstance(canonical_id, str) or canonical_id.strip() not in resolved_ids:
            raise EventValidationError(
                f"event endpoint node_id={node_id} canonical_id is not a resolved ID"
            )
        component_ids = _strict_integer_list(
            raw_component_ids,
            field=f"node[{node_id}].component_entry_ids",
        ) if raw_component_ids else ()
        effective_ids = list(resolved_ids)
        model_entities = [
            {"canonical_id": canonical_id, "name": display_name}
            for canonical_id in effective_ids
        ]

    aliases = node.get("aliases")
    if not isinstance(aliases, list) or any(not isinstance(item, str) for item in aliases):
        raise EventValidationError(f"node[{node_id}].aliases must be a string list")
    provenance: dict[str, object] = {
        "node_id": node_id,
        "entity_type": entity_type,
        "canonical_id": node.get("canonical_id"),
        "resolved_ids": list(resolved_ids),
        "effective_resolved_ids": list(effective_ids),
        "raw_name": str(node.get("raw_name") or ""),
        "display_name": display_name,
        "aliases": list(aliases),
        "unresolved_tokens": [],
        "component_entry_ids": list(component_ids),
        "resolved": True,
    }
    output = (tuple(model_entities), provenance)
    cache[node_id] = output
    return output


def node_value(node: dict[str, Any]) -> dict[str, str]:
    """Compatibility helper for a single, already trusted non-group node."""

    resolved_ids = _strict_strings(node.get("resolved_ids"), field="resolved_ids")
    canonical_id = node.get("canonical_id")
    if (
        node.get("resolved") is not True
        or node.get("unresolved_tokens") != []
        or not isinstance(canonical_id, str)
        or canonical_id.strip() not in resolved_ids
    ):
        raise EventValidationError("node identity is not strictly resolved")
    name = str(node.get("display_name") or "").strip()
    if not name:
        raise EventValidationError("node display_name is empty")
    return {"canonical_id": canonical_id.strip(), "name": name}


def _integer_values(record: dict[str, Any], key: str) -> tuple[int, ...]:
    values = record.get(key, ())
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        values = (values,)
    output: list[int] = []
    for value in values:
        try:
            output.append(int(value))
        except (TypeError, ValueError):
            return ()
    return tuple(output)


def _integer_value(record: dict[str, Any], key: str) -> int | None:
    try:
        value = record.get(key)
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _joined_names(values: Sequence[dict[str, str]]) -> str:
    return " and ".join(value["name"] for value in values)


_RELATION_PHRASES = {
    "activation": "activates",
    "inhibition": "inhibits",
    "expression": "increases the expression of",
    "repression": "represses",
    "phosphorylation": "phosphorylates",
    "dephosphorylation": "dephosphorylates",
    "glycosylation": "glycosylates",
    "ubiquitination": "ubiquitinates",
    "methylation": "methylates",
    "binding/association": "binds or associates with",
    "dissociation": "dissociates from",
    "indirect effect": "indirectly affects",
    "state change": "changes the state of",
    "compound": "is connected through a compound-mediated relation to",
    "hidden compound": "is connected through a hidden compound-mediated relation to",
}

_SUBTYPE_LABELS = {
    "binding/association": "binding_association",
    "indirect effect": "indirect_effect",
    "state change": "state_change",
    "hidden compound": "hidden_compound_mediated",
    "compound": "compound_mediated",
    "missing interaction": "missing_interaction",
}


def _normalized_subtype_label(value: str) -> str:
    return _SUBTYPE_LABELS.get(value, value.replace(" ", "_"))


def _relation_subtypes(
    record: dict[str, Any],
) -> tuple[tuple[dict[str, str], ...], tuple[str, ...], tuple[int, ...]]:
    raw_subtypes = record.get("subtypes")
    if not isinstance(raw_subtypes, list):
        raise EventValidationError("relation subtypes must be a list")
    parsed: list[dict[str, str]] = []
    names: list[str] = []
    mediator_ids: list[int] = []
    visible_compound_ids: list[int] = []
    for position, subtype in enumerate(raw_subtypes):
        if not isinstance(subtype, dict) or set(subtype) != {"name", "value"}:
            raise EventValidationError(
                f"relation subtype[{position}] must contain exactly name and value"
            )
        name = str(subtype.get("name") or "").strip().casefold()
        value = str(subtype.get("value") or "").strip()
        if not name:
            raise EventValidationError(f"relation subtype[{position}] has an empty name")
        parsed.append({"name": name, "value": value})
        names.append(name)
        if name in {"compound", "hidden compound"}:
            if not value:
                raise EventValidationError(f"relation subtype {name} lacks a mediator ID")
            try:
                mediator_id = int(value)
            except (TypeError, ValueError) as exc:
                raise EventValidationError(
                    f"relation subtype {name} has an invalid mediator ID"
                ) from exc
            if mediator_id <= 0:
                raise EventValidationError(
                    f"relation subtype {name} has a non-positive mediator ID"
                )
            mediator_ids.append(mediator_id)
            if name == "compound":
                visible_compound_ids.append(mediator_id)

    subtype_names = record.get("subtype_names")
    semantic_tags = record.get("semantic_tags")
    if not isinstance(subtype_names, list) or not isinstance(semantic_tags, list):
        raise EventValidationError(
            "relation subtype_names and semantic_tags must both be lists"
        )
    normalized_subtype_names = tuple(str(item).strip().casefold() for item in subtype_names)
    normalized_semantic_tags = tuple(str(item).strip().casefold() for item in semantic_tags)
    if tuple(names) != normalized_subtype_names or tuple(names) != normalized_semantic_tags:
        raise EventValidationError(
            "relation subtypes, subtype_names, and semantic_tags disagree"
        )
    expected_missing = MISSING_RELATION_SUBTYPE in names
    if record.get("has_missing_interaction") is not expected_missing:
        raise EventValidationError("relation has_missing_interaction disagrees with subtypes")

    producer_mediator = record.get("mediator_entry_id")
    if producer_mediator is not None:
        if isinstance(producer_mediator, bool):
            raise EventValidationError("mediator_entry_id must be an integer or null")
        try:
            producer_mediator = int(producer_mediator)
        except (TypeError, ValueError) as exc:
            raise EventValidationError("mediator_entry_id is invalid") from exc
    if visible_compound_ids:
        if producer_mediator != visible_compound_ids[-1]:
            raise EventValidationError(
                "mediator_entry_id disagrees with compound subtype value"
            )
    elif producer_mediator is not None:
        raise EventValidationError(
            "mediator_entry_id exists without a visible compound subtype"
        )
    return tuple(parsed), tuple(names), tuple(dict.fromkeys(mediator_ids))


def relation_label(record: dict[str, Any]) -> str:
    _, tags, _ = _relation_subtypes(record)
    if MISSING_RELATION_SUBTYPE in tags:
        return "missing_interaction"
    return "+".join(_normalized_subtype_label(tag) for tag in tags) or (
        f"{str(record.get('relation_type') or 'relation').casefold()}_relation"
    )


def relation_text(
    record: dict[str, Any],
    sources: Sequence[dict[str, str]],
    targets: Sequence[dict[str, str]],
    mediators: Sequence[dict[str, str]] = (),
) -> str:
    source_text = _joined_names(sources)
    target_text = _joined_names(targets)
    _, raw_tags, _ = _relation_subtypes(record)
    if MISSING_RELATION_SUBTYPE in raw_tags:
        return f"KEGG marks the interaction between {source_text} and {target_text} as missing in this pathway."
    tags = tuple(raw_tags)
    phrase = _RELATION_PHRASES.get(tags[0]) if len(tags) == 1 else None
    if phrase and not mediators:
        return f"{source_text} {phrase} {target_text}."
    relation_type = str(record.get("relation_type") or "unspecified")
    subtype = ", ".join(tags) if tags else "unspecified subtype"
    mediator_text = (
        f" mediated by {_joined_names(mediators)}" if mediators else ""
    )
    return (
        f"KEGG records a {relation_type} relation from {source_text} to {target_text}"
        f"{mediator_text} with subtypes {subtype}."
    )


def reaction_text(
    record: dict[str, Any],
    sources: Sequence[dict[str, str]],
    targets: Sequence[dict[str, str]],
) -> str:
    source_text = _joined_names(sources)
    target_text = _joined_names(targets)
    reaction_type = str(record.get("reaction_type") or "unspecified")
    verb = "is" if len(sources) == 1 else "are"
    if reaction_type == "reversible":
        return (
            f"{source_text} {verb} shown as reversibly connected to {target_text} "
            "in the KEGG pathway map."
        )
    return f"{source_text} {verb} converted to {target_text}."


@dataclass(frozen=True)
class StructuredEvent:
    event_id: str
    event_type: str
    source_node_ids: tuple[int, ...]
    target_node_ids: tuple[int, ...]
    source: tuple[dict[str, str], ...]
    relation: str
    target: tuple[dict[str, str], ...]
    text: str
    producer_renderable: bool
    source_entity_provenance: tuple[dict[str, object], ...] = ()
    target_entity_provenance: tuple[dict[str, object], ...] = ()
    mediator_node_ids: tuple[int, ...] = ()
    mediator: tuple[dict[str, str], ...] = ()
    mediator_entity_provenance: tuple[dict[str, object], ...] = ()
    raw_relation_type: str = ""
    raw_reaction_name: str = ""
    raw_reaction_type: str = ""
    raw_subtypes: tuple[dict[str, str], ...] = ()
    topology_role: str = TOPOLOGY_BACKBONE
    topology_arcs: tuple[tuple[int, int], ...] = ()
    core_included: bool = True
    exclusion_reason: str = ""

    def model_object(self) -> dict[str, object]:
        return {
            "source": list(self.source),
            "relation": self.relation,
            "target": list(self.target),
            "text": self.text,
        }

    def record_object(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source_node_ids": list(self.source_node_ids),
            "target_node_ids": list(self.target_node_ids),
            "source": list(self.source),
            "relation": self.relation,
            "target": list(self.target),
            "text": self.text,
            "producer_renderable": self.producer_renderable,
            "source_entity_provenance": [dict(value) for value in self.source_entity_provenance],
            "target_entity_provenance": [dict(value) for value in self.target_entity_provenance],
            "mediator_node_ids": list(self.mediator_node_ids),
            "mediator": list(self.mediator),
            "mediator_entity_provenance": [
                dict(value) for value in self.mediator_entity_provenance
            ],
            "raw_relation_type": self.raw_relation_type,
            "raw_reaction_name": self.raw_reaction_name,
            "raw_reaction_type": self.raw_reaction_type,
            "raw_subtypes": [dict(value) for value in self.raw_subtypes],
            "topology_role": self.topology_role,
            "topology_arcs": [list(arc) for arc in self.topology_arcs],
            "core_included": self.core_included,
            "exclusion_reason": self.exclusion_reason,
        }


@dataclass(frozen=True)
class StructuredLayer:
    layer_index: int
    distance_to_sink: int
    events: tuple[StructuredEvent, ...]

    def model_object(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "events": [event.model_object() for event in self.events],
        }

    def record_object(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "distance_to_sink": self.distance_to_sink,
            "events": [event.record_object() for event in self.events],
        }


@dataclass(frozen=True)
class StructuredRecord:
    graph_id: str
    view_id: str
    record_id: str
    source_graph_json: str
    organism: str
    pathway_id: str
    pathway_title: str
    sink_node_ids: tuple[int, ...]
    layers: tuple[StructuredLayer, ...]
    graph_event_count: int
    graph_missing_endpoint_event_count: int
    excluded_events: tuple[StructuredEvent, ...] = ()

    @property
    def family(self) -> str:
        return canonical_pathway_family_id(self.pathway_id)

    def record_object(self) -> dict[str, object]:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "graph_id": self.graph_id,
            "view_id": self.view_id,
            "record_id": self.record_id,
            "source_graph_json": self.source_graph_json,
            "organism": self.organism,
            "pathway_id": self.pathway_id,
            "pathway_family_id": self.family,
            "pathway_title": self.pathway_title,
            "sink_node_ids": list(self.sink_node_ids),
            "layers": [layer.record_object() for layer in self.layers],
            "graph_event_count": self.graph_event_count,
            "graph_missing_endpoint_event_count": self.graph_missing_endpoint_event_count,
            "graph_excluded_noncore_event_count": len(self.excluded_events),
            "excluded_events": [event.record_object() for event in self.excluded_events],
            "phenotype_status": "not_annotated",
            "parser_source": SUBSTEP_SOURCE,
        }


def _project_nodes(
    node_ids: Sequence[int],
    node_lookup: dict[int, dict[str, Any]],
    cache: dict[int, tuple[tuple[dict[str, str], ...], dict[str, object]]],
) -> tuple[tuple[dict[str, str], ...], tuple[dict[str, object], ...]]:
    model_entities: list[dict[str, str]] = []
    provenance: list[dict[str, object]] = []
    for node_id in node_ids:
        node_entities, node_provenance = _node_projection(node_id, node_lookup, cache)
        model_entities.extend(dict(entity) for entity in node_entities)
        provenance.append(dict(node_provenance))
    if not model_entities:
        raise EventValidationError("event endpoint projection is empty")
    return tuple(model_entities), tuple(provenance)


def event_from_relation(
    record: dict[str, Any],
    node_lookup: dict[int, dict[str, Any]],
    projection_cache: dict[
        int, tuple[tuple[dict[str, str], ...], dict[str, object]]
    ] | None = None,
) -> StructuredEvent:
    relation_id = _strict_event_id(record, "relation_id")
    source_ids = _strict_integer_list(
        [record.get("entry1_id")], field="relation.entry1_id"
    )
    target_ids = _strict_integer_list(
        [record.get("entry2_id")], field="relation.entry2_id"
    )
    relation_type = str(record.get("relation_type") or "").strip()
    if relation_type not in RELATION_TYPES:
        raise EventValidationError(f"unsupported relation_type={relation_type!r}")
    renderable = _strict_renderable(record)
    raw_subtypes, subtype_names, mediator_ids = _relation_subtypes(record)
    cache = projection_cache if projection_cache is not None else {}
    sources, source_provenance = _project_nodes(source_ids, node_lookup, cache)
    targets, target_provenance = _project_nodes(target_ids, node_lookup, cache)
    if mediator_ids:
        mediators, mediator_provenance = _project_nodes(
            mediator_ids, node_lookup, cache
        )
    else:
        mediators, mediator_provenance = (), ()

    unknown_subtypes = sorted(
        set(subtype_names)
        - DIRECTIONAL_RELATION_SUBTYPES
        - CONTEXT_RELATION_SUBTYPES
        - {MISSING_RELATION_SUBTYPE}
    )
    if MISSING_RELATION_SUBTYPE in subtype_names:
        topology_role = TOPOLOGY_EXCLUDED
        topology_arcs: tuple[tuple[int, int], ...] = ()
        core_included = False
        exclusion_reason = "missing_interaction"
    elif unknown_subtypes:
        topology_role = TOPOLOGY_EXCLUDED
        topology_arcs = ()
        core_included = False
        exclusion_reason = "unknown_subtypes:" + ",".join(unknown_subtypes)
    elif DIRECTIONAL_RELATION_SUBTYPES.intersection(subtype_names):
        topology_role = TOPOLOGY_BACKBONE
        topology_arcs = ((source_ids[0], target_ids[0]),)
        core_included = True
        exclusion_reason = ""
    else:
        topology_role = TOPOLOGY_CONTEXT
        topology_arcs = ()
        core_included = True
        exclusion_reason = ""

    return StructuredEvent(
        event_id=f"relation:{relation_id}",
        event_type="relation",
        source_node_ids=source_ids,
        target_node_ids=target_ids,
        source=sources,
        relation=relation_label(record),
        target=targets,
        text=relation_text(record, sources, targets, mediators),
        producer_renderable=renderable,
        source_entity_provenance=source_provenance,
        target_entity_provenance=target_provenance,
        mediator_node_ids=mediator_ids,
        mediator=mediators,
        mediator_entity_provenance=mediator_provenance,
        raw_relation_type=relation_type,
        raw_subtypes=raw_subtypes,
        topology_role=topology_role,
        topology_arcs=topology_arcs,
        core_included=core_included,
        exclusion_reason=exclusion_reason,
    )


def event_from_reaction(
    record: dict[str, Any],
    node_lookup: dict[int, dict[str, Any]],
    projection_cache: dict[
        int, tuple[tuple[dict[str, str], ...], dict[str, object]]
    ] | None = None,
) -> StructuredEvent:
    reaction_id = _strict_event_id(record, "reaction_id")
    source_ids = _strict_integer_list(
        record.get("substrate_entry_ids"), field="reaction.substrate_entry_ids"
    )
    target_ids = _strict_integer_list(
        record.get("product_entry_ids"), field="reaction.product_entry_ids"
    )
    if not source_ids or not target_ids:
        raise EventValidationError("reaction must have substrates and products")
    reaction_name = str(record.get("reaction_name") or "").strip()
    if not reaction_name:
        raise EventValidationError("reaction_name must be non-empty")
    reaction_type = str(record.get("reaction_type") or "").strip().casefold()
    if reaction_type not in REACTION_TYPES:
        raise EventValidationError(f"unsupported reaction_type={reaction_type!r}")
    renderable = _strict_renderable(record)
    cache = projection_cache if projection_cache is not None else {}
    sources, source_provenance = _project_nodes(source_ids, node_lookup, cache)
    targets, target_provenance = _project_nodes(target_ids, node_lookup, cache)

    arcs = {
        (source_id, target_id)
        for source_id in source_ids
        for target_id in target_ids
    }
    if reaction_type == "reversible":
        arcs.update((target_id, source_id) for source_id, target_id in tuple(arcs))
    relation = (
        "reversible_conversion"
        if reaction_type == "reversible"
        else "irreversible_conversion"
    )
    return StructuredEvent(
        event_id=f"reaction:{reaction_id}",
        event_type="reaction",
        source_node_ids=source_ids,
        target_node_ids=target_ids,
        source=sources,
        relation=relation,
        target=targets,
        text=reaction_text(record, sources, targets),
        producer_renderable=renderable,
        source_entity_provenance=source_provenance,
        target_entity_provenance=target_provenance,
        raw_reaction_name=reaction_name,
        raw_reaction_type=reaction_type,
        topology_role=TOPOLOGY_BACKBONE,
        topology_arcs=tuple(sorted(arcs)),
        core_included=True,
    )


def graph_events(graph: dict[str, Any]) -> tuple[tuple[StructuredEvent, ...], int]:
    raw_nodes = graph.get("nodes")
    raw_relations = graph.get("relations")
    raw_reactions = graph.get("reactions")
    if not isinstance(raw_nodes, list):
        raise ValueError("processed_graph nodes must be a list")
    if not isinstance(raw_relations, list):
        raise ValueError("processed_graph relations must be a list")
    if not isinstance(raw_reactions, list):
        raise ValueError("processed_graph reactions must be a list")
    node_lookup: dict[int, dict[str, Any]] = {}
    for position, node in enumerate(raw_nodes):
        if not isinstance(node, dict) or node.get("node_id") is None:
            raise ValueError(f"processed_graph node[{position}] lacks a node_id")
        if isinstance(node["node_id"], bool):
            raise ValueError(f"processed_graph node[{position}] has an invalid node_id")
        try:
            node_id = int(node["node_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"processed_graph node[{position}] has an invalid node_id"
            ) from exc
        if node_id <= 0:
            raise ValueError(f"processed_graph node[{position}] has a non-positive node_id")
        if node_id in node_lookup:
            raise ValueError(f"processed_graph contains duplicate node_id={node_id}")
        node_lookup[node_id] = node
    events: list[StructuredEvent] = []
    rejected_event_count = 0
    projection_cache: dict[
        int, tuple[tuple[dict[str, str], ...], dict[str, object]]
    ] = {}
    for position, relation in enumerate(raw_relations):
        if not isinstance(relation, dict):
            raise ValueError(f"processed_graph relation[{position}] is not an object")
        try:
            event = event_from_relation(relation, node_lookup, projection_cache)
        except EventValidationError:
            rejected_event_count += 1
            continue
        events.append(event)
    for position, reaction in enumerate(raw_reactions):
        if not isinstance(reaction, dict):
            raise ValueError(f"processed_graph reaction[{position}] is not an object")
        try:
            event = event_from_reaction(reaction, node_lookup, projection_cache)
        except EventValidationError:
            rejected_event_count += 1
            continue
        events.append(event)
    seen_event_ids: set[str] = set()
    duplicate_event_ids: set[str] = set()
    for event in events:
        if event.event_id in seen_event_ids:
            duplicate_event_ids.add(event.event_id)
        seen_event_ids.add(event.event_id)
    if duplicate_event_ids:
        raise ValueError(
            "processed_graph contains duplicate structural event IDs: "
            + ", ".join(sorted(duplicate_event_ids)[:10])
        )
    # The caller rejects the whole graph whenever this count is non-zero.  No
    # partially validated event set may be used to recompute SCCs or layers.
    return tuple(events), rejected_event_count


def observed_payload(layers: Sequence[StructuredLayer]) -> dict[str, object]:
    return {
        "observed_layers": [layer.model_object() for layer in layers],
    }


def answer_payload(layers: Sequence[StructuredLayer]) -> dict[str, object]:
    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "remaining_layers": [layer.model_object() for layer in layers],
    }


def prefix_only_question(
    layers: Sequence[StructuredLayer],
    *,
    organism: str,
    prompt_profile: str = EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
) -> str:
    """Render the full v3 output contract plus the observed graph prefix."""

    return render_pathway_question(
        observed_payload(layers),
        next_layer_index=len(layers),
        organism=organism,
        profile=prompt_profile,
    )


def answer_json(layers: Sequence[StructuredLayer]) -> str:
    return compact_json(answer_payload(layers))


def selected_prefix_lengths(layer_count: int, maximum: int) -> tuple[int, ...]:
    candidates = list(range(1, layer_count))
    if maximum <= 0 or len(candidates) <= maximum:
        return tuple(candidates)
    if maximum == 1:
        return (candidates[len(candidates) // 2],)
    indices = {
        round(position * (len(candidates) - 1) / (maximum - 1))
        for position in range(maximum)
    }
    return tuple(candidates[index] for index in sorted(indices))


def csv_row(
    record: StructuredRecord,
    prefix_len: int,
    *,
    prompt_profile: str = EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    prefix_horizon: str = "",
    split: str = "",
) -> dict[str, object]:
    if not 0 < prefix_len < len(record.layers):
        raise ValueError("prefix_len must leave at least one observed and one target layer")
    if prompt_profile not in PROMPT_PROFILE_NAMES:
        raise ValueError(f"unknown prompt_profile={prompt_profile!r}")

    # Projection is applied to the complete supervised pair, not only the
    # visible prefix.  Therefore one non-neutral target entity makes the whole
    # P2 base sample ineligible instead of silently changing its answer.
    pair_payload = {
        "observed_layers": [
            layer.model_object() for layer in record.layers[:prefix_len]
        ],
        "remaining_layers": [
            layer.model_object() for layer in record.layers[prefix_len:]
        ],
    }
    projection = project_record(
        pair_payload,
        profile=prompt_profile,
        organism=record.organism,
    )
    if not projection.eligible or projection.projected is None:
        reasons = ",".join(
            f"{key}={value}"
            for key, value in projection.rejection_reason_counts.items()
        )
        raise ValueError(
            f"record {record.record_id} is ineligible for {prompt_profile}: {reasons}"
        )
    projected_observed = {
        "observed_layers": projection.projected["observed_layers"]
    }
    projected_answer = {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "remaining_layers": projection.projected["remaining_layers"],
    }
    base_sample_id = f"{record.record_id}:prefix={prefix_len}"
    sample_id = f"{base_sample_id}:profile={prompt_profile}"
    profile_metadata = PROMPT_PROFILE_METADATA[prompt_profile]
    row: dict[str, object] = {field: "" for field in V3_CSV_FIELDNAMES}
    row.update(
        {
            "sample_id": sample_id,
            "base_sample_id": base_sample_id,
            "record_id": record.record_id,
            "graph_id": record.graph_id,
            "view_id": record.view_id,
            "question": render_pathway_question(
                projected_observed,
                next_layer_index=prefix_len,
                organism=record.organism,
                profile=prompt_profile,
            ),
            "answer": compact_json(projected_answer),
            "question_type": QUESTION_TYPE,
            "given_step": prefix_len - 1,
            "total_step": len(record.layers) - 1,
            "pathway_id": record.pathway_id,
            "pathway_family_id": record.family,
            "entry_id": stable_id("sink", *record.sink_node_ids),
            "phenotype": "",
            "phenotype_status": "not_annotated",
            "phenotype_source": "",
            "organism": record.organism,
            "pathway_block": record.view_id,
            "pathway_title": record.pathway_title,
            "source_json": record.source_graph_json,
            "source_graph_json": record.source_graph_json,
            "prefix_step_count": prefix_len,
            "target_step_count": len(record.layers) - prefix_len,
            "has_empty_prefix": 0,
            "substep_schema_version": SUBSTEP_SCHEMA_VERSION,
            "substep_source": SUBSTEP_SOURCE,
            "prompt_profile": prompt_profile,
            "organism_conditioning": profile_metadata["organism_conditioning"],
            "entity_id_space": profile_metadata["entity_id_space"],
            "entity_mapping_status": profile_metadata["entity_mapping_status"],
            "prefix_horizon": prefix_horizon,
            "split": split,
        }
    )
    return row


def chat_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def total_training_tokens(tokenizer: Any, row: dict[str, object]) -> int:
    prompt_ids = tokenizer.encode(chat_prompt(str(row["question"])), add_special_tokens=False)
    answer_ids = tokenizer.encode(
        f"{row['answer']}<|im_end|>",
        add_special_tokens=False,
    )
    return len(prompt_ids) + len(answer_ids)


def records_jsonl(records: Iterable[StructuredRecord]) -> Iterable[str]:
    for record in records:
        yield compact_json(record.record_object())


_EVENT_RECORD_KEYS = {
    "event_id",
    "event_type",
    "source_node_ids",
    "target_node_ids",
    "source",
    "relation",
    "target",
    "text",
    "producer_renderable",
    "source_entity_provenance",
    "target_entity_provenance",
    "mediator_node_ids",
    "mediator",
    "mediator_entity_provenance",
    "raw_relation_type",
    "raw_reaction_name",
    "raw_reaction_type",
    "raw_subtypes",
    "topology_role",
    "topology_arcs",
    "core_included",
    "exclusion_reason",
}

_ENTITY_PROVENANCE_KEYS = {
    "node_id",
    "entity_type",
    "canonical_id",
    "resolved_ids",
    "effective_resolved_ids",
    "raw_name",
    "display_name",
    "aliases",
    "unresolved_tokens",
    "component_entry_ids",
    "resolved",
}


def _record_entity_provenance_valid(value: object) -> bool:
    if not isinstance(value, dict) or set(value) != _ENTITY_PROVENANCE_KEYS:
        return False
    if not isinstance(value.get("node_id"), int) or isinstance(value.get("node_id"), bool):
        return False
    if value["node_id"] <= 0 or value.get("resolved") is not True:
        return False
    if value.get("unresolved_tokens") != []:
        return False
    if not isinstance(value.get("resolved_ids"), list):
        return False
    effective_ids = value.get("effective_resolved_ids")
    if (
        not isinstance(effective_ids, list)
        or not effective_ids
        or not all(isinstance(item, str) and item.strip() for item in effective_ids)
        or len(set(effective_ids)) != len(effective_ids)
    ):
        return False
    if not isinstance(value.get("component_entry_ids"), list):
        return False
    if not isinstance(value.get("aliases"), list):
        return False
    if not isinstance(value.get("display_name"), str) or not value["display_name"].strip():
        return False
    canonical_id = value.get("canonical_id")
    if value.get("entity_type") == "group":
        return canonical_id in (None, "") and bool(value["component_entry_ids"])
    return (
        isinstance(canonical_id, str)
        and canonical_id in value["resolved_ids"]
        and set(effective_ids) == set(value["resolved_ids"])
    )


def _record_event_from_object(value: object) -> StructuredEvent:
    if not isinstance(value, dict) or set(value) != _EVENT_RECORD_KEYS:
        raise ValueError("structured record event does not exactly match v3")
    raw_source = value.get("source")
    raw_target = value.get("target")
    raw_mediator = value.get("mediator")
    if (
        not isinstance(raw_source, list)
        or not raw_source
        or not all(_record_entity_valid(item) for item in raw_source)
        or not isinstance(raw_target, list)
        or not raw_target
        or not all(_record_entity_valid(item) for item in raw_target)
        or not isinstance(raw_mediator, list)
        or not all(_record_entity_valid(item) for item in raw_mediator)
    ):
        raise ValueError("structured record event entities are invalid")
    source_node_ids = _strict_integer_list(
        value.get("source_node_ids"), field="record.source_node_ids"
    )
    target_node_ids = _strict_integer_list(
        value.get("target_node_ids"), field="record.target_node_ids"
    )
    mediator_node_ids = _strict_integer_list(
        value.get("mediator_node_ids"), field="record.mediator_node_ids"
    )
    source_provenance = value.get("source_entity_provenance")
    target_provenance = value.get("target_entity_provenance")
    mediator_provenance = value.get("mediator_entity_provenance")
    if (
        not isinstance(source_provenance, list)
        or len(source_provenance) != len(source_node_ids)
        or not all(_record_entity_provenance_valid(item) for item in source_provenance)
        or not isinstance(target_provenance, list)
        or len(target_provenance) != len(target_node_ids)
        or not all(_record_entity_provenance_valid(item) for item in target_provenance)
        or not isinstance(mediator_provenance, list)
        or len(mediator_provenance) != len(mediator_node_ids)
        or not all(_record_entity_provenance_valid(item) for item in mediator_provenance)
    ):
        raise ValueError("structured record event entity provenance is invalid")

    def projected_ids(provenance: Sequence[dict[str, object]]) -> list[str]:
        return [
            str(canonical_id)
            for item in provenance
            for canonical_id in item["effective_resolved_ids"]
        ]

    if [item["canonical_id"] for item in raw_source] != projected_ids(source_provenance):
        raise ValueError("structured record source projection loses resolved IDs")
    if [item["canonical_id"] for item in raw_target] != projected_ids(target_provenance):
        raise ValueError("structured record target projection loses resolved IDs")
    if [item["canonical_id"] for item in raw_mediator] != projected_ids(mediator_provenance):
        raise ValueError("structured record mediator projection loses resolved IDs")
    if not str(value.get("event_id") or "").strip():
        raise ValueError("structured record event_id is empty")
    event_type = value.get("event_type")
    if event_type not in {"relation", "reaction"}:
        raise ValueError("structured record event_type is invalid")
    if not isinstance(value.get("relation"), str) or not value["relation"].strip():
        raise ValueError("structured record relation is empty")
    if not isinstance(value.get("text"), str) or not value["text"].strip():
        raise ValueError("structured record text is empty")
    if not isinstance(value.get("producer_renderable"), bool):
        raise ValueError("structured record producer_renderable must be boolean")
    topology_role = value.get("topology_role")
    if topology_role not in TOPOLOGY_ROLES:
        raise ValueError("structured record topology_role is invalid")
    raw_arcs = value.get("topology_arcs")
    if not isinstance(raw_arcs, list):
        raise ValueError("structured record topology_arcs must be a list")
    topology_arcs: list[tuple[int, int]] = []
    for raw_arc in raw_arcs:
        arc = _strict_integer_list(raw_arc, field="record.topology_arc")
        if len(arc) != 2:
            raise ValueError("structured record topology_arc must contain two nodes")
        topology_arcs.append((arc[0], arc[1]))
    if len(set(topology_arcs)) != len(topology_arcs):
        raise ValueError("structured record topology_arcs contain duplicates")
    core_included = value.get("core_included")
    if not isinstance(core_included, bool):
        raise ValueError("structured record core_included must be boolean")
    if topology_role == TOPOLOGY_BACKBONE and (not core_included or not topology_arcs):
        raise ValueError("backbone event must be included and have topology arcs")
    if topology_role != TOPOLOGY_BACKBONE and topology_arcs:
        raise ValueError("non-backbone event cannot have topology arcs")
    if topology_role == TOPOLOGY_EXCLUDED and core_included:
        raise ValueError("excluded event cannot be core included")
    raw_subtypes = value.get("raw_subtypes")
    if not isinstance(raw_subtypes, list) or not all(
        isinstance(item, dict) and set(item) == {"name", "value"}
        for item in raw_subtypes
    ):
        raise ValueError("structured record raw_subtypes are invalid")
    if event_type == "reaction":
        if value.get("raw_reaction_type") not in REACTION_TYPES:
            raise ValueError("reaction raw_reaction_type is invalid")
        expected_relation = (
            "reversible_conversion"
            if value["raw_reaction_type"] == "reversible"
            else "irreversible_conversion"
        )
        if value.get("relation") != expected_relation:
            raise ValueError("reaction relation label is not normalized")
        if value.get("raw_relation_type") or raw_subtypes or mediator_node_ids:
            raise ValueError("reaction contains relation-only provenance")
    else:
        if value.get("raw_relation_type") not in RELATION_TYPES:
            raise ValueError("relation raw_relation_type is invalid")
        if value.get("raw_reaction_name") or value.get("raw_reaction_type"):
            raise ValueError("relation contains reaction-only provenance")
    return StructuredEvent(
        event_id=str(value["event_id"]),
        event_type=str(event_type),
        source_node_ids=source_node_ids,
        target_node_ids=target_node_ids,
        source=tuple(dict(item) for item in raw_source),
        relation=str(value["relation"]),
        target=tuple(dict(item) for item in raw_target),
        text=str(value["text"]),
        producer_renderable=value["producer_renderable"],
        source_entity_provenance=tuple(dict(item) for item in source_provenance),
        target_entity_provenance=tuple(dict(item) for item in target_provenance),
        mediator_node_ids=mediator_node_ids,
        mediator=tuple(dict(item) for item in raw_mediator),
        mediator_entity_provenance=tuple(dict(item) for item in mediator_provenance),
        raw_relation_type=str(value.get("raw_relation_type") or ""),
        raw_reaction_name=str(value.get("raw_reaction_name") or ""),
        raw_reaction_type=str(value.get("raw_reaction_type") or ""),
        raw_subtypes=tuple(dict(item) for item in raw_subtypes),
        topology_role=str(topology_role),
        topology_arcs=tuple(topology_arcs),
        core_included=core_included,
        exclusion_reason=str(value.get("exclusion_reason") or ""),
    )


def record_from_object(value: dict[str, Any]) -> StructuredRecord:
    expected_record_keys = {
        "schema_version",
        "graph_id",
        "view_id",
        "record_id",
        "source_graph_json",
        "organism",
        "pathway_id",
        "pathway_family_id",
        "pathway_title",
        "sink_node_ids",
        "layers",
        "graph_event_count",
        "graph_missing_endpoint_event_count",
        "graph_excluded_noncore_event_count",
        "excluded_events",
        "phenotype_status",
        "parser_source",
    }
    if set(value) != expected_record_keys:
        raise ValueError("structured record top-level keys do not exactly match v3")
    if value.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("unsupported structured record schema_version")
    if value.get("phenotype_status") != "not_annotated":
        raise ValueError("structured record phenotype_status must be not_annotated")
    if value.get("parser_source") != SUBSTEP_SOURCE:
        raise ValueError("structured record parser_source does not match v3")
    raw_layers = value.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("structured record layers must be a non-empty list")
    layers: list[StructuredLayer] = []
    seen_event_ids: set[str] = set()
    previous_distance: int | None = None
    for expected_layer_index, layer_value in enumerate(raw_layers):
        if not isinstance(layer_value, dict) or set(layer_value) != {
            "layer_index",
            "distance_to_sink",
            "events",
        }:
            raise ValueError("structured record layer does not exactly match v3")
        if layer_value.get("layer_index") != expected_layer_index:
            raise ValueError("structured record layer indices must be consecutive")
        distance = layer_value.get("distance_to_sink")
        if not isinstance(distance, int) or isinstance(distance, bool) or distance < 0:
            raise ValueError("structured record distance_to_sink is invalid")
        if previous_distance is not None and distance >= previous_distance:
            raise ValueError("structured record longest distances must decrease by layer")
        previous_distance = distance
        raw_events = layer_value.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("structured record layers must contain events")
        events = tuple(_record_event_from_object(item) for item in raw_events)
        if any(not event.core_included for event in events):
            raise ValueError("structured record layer contains a non-core event")
        for event in events:
            if event.event_id in seen_event_ids:
                raise ValueError("structured record repeats an event within one view")
            seen_event_ids.add(event.event_id)
        layers.append(
            StructuredLayer(
                layer_index=expected_layer_index,
                distance_to_sink=distance,
                events=events,
            )
        )
    raw_excluded_events = value.get("excluded_events")
    if not isinstance(raw_excluded_events, list):
        raise ValueError("structured record excluded_events must be a list")
    excluded_events = tuple(
        _record_event_from_object(item) for item in raw_excluded_events
    )
    if any(event.core_included or event.topology_role != TOPOLOGY_EXCLUDED for event in excluded_events):
        raise ValueError("structured record excluded_events contains a core event")
    if int(value.get("graph_excluded_noncore_event_count", -1)) != len(excluded_events):
        raise ValueError("structured record excluded event count disagrees")
    for event in excluded_events:
        if event.event_id in seen_event_ids:
            raise ValueError("structured record event exists in core and excluded sets")
        seen_event_ids.add(event.event_id)

    sink_node_ids = _strict_integer_list(
        value.get("sink_node_ids"), field="record.sink_node_ids"
    )
    if not sink_node_ids:
        raise ValueError("structured record sink_node_ids must be non-empty")
    graph_id = str(value["graph_id"])
    view_id = str(value["view_id"])
    record_id = str(value["record_id"])
    expected_view_id = stable_id("view", graph_id, ",".join(map(str, sink_node_ids)))
    if view_id != expected_view_id:
        raise ValueError("structured record view_id does not match graph and sink")
    if record_id != stable_id("record", graph_id, view_id):
        raise ValueError("structured record record_id does not match graph and view")
    pathway_id = str(value["pathway_id"])
    if value.get("pathway_family_id") != canonical_pathway_family_id(pathway_id):
        raise ValueError("structured record pathway_family_id does not match pathway_id")
    graph_event_count = int(value["graph_event_count"])
    missing_count = int(value["graph_missing_endpoint_event_count"])
    if missing_count != 0:
        raise ValueError("materialized structured record contains rejected graph events")
    if graph_event_count < len(seen_event_ids):
        raise ValueError("structured record graph_event_count is too small")
    record = StructuredRecord(
        graph_id=graph_id,
        view_id=view_id,
        record_id=record_id,
        source_graph_json=str(value["source_graph_json"]),
        organism=str(value["organism"]),
        pathway_id=pathway_id,
        pathway_title=str(value.get("pathway_title", "")),
        sink_node_ids=sink_node_ids,
        layers=tuple(layers),
        graph_event_count=graph_event_count,
        graph_missing_endpoint_event_count=missing_count,
        excluded_events=excluded_events,
    )
    if not record.graph_id or not record.view_id or not record.record_id:
        raise ValueError("structured record stable identities must be non-empty")
    if not record.source_graph_json or not record.organism or not record.pathway_id:
        raise ValueError("structured record provenance fields must be non-empty")
    return record


def _record_entity_valid(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == {"canonical_id", "name"}
        and isinstance(value.get("canonical_id"), str)
        and bool(value["canonical_id"].strip())
        and isinstance(value.get("name"), str)
        and bool(value["name"].strip())
    )


__all__ = [
    "ANSWER_SCHEMA_VERSION",
    "CONTEXT_RELATION_SUBTYPES",
    "DATASET_SCHEMA_VERSION",
    "DIRECTIONAL_RELATION_SUBTYPES",
    "EventValidationError",
    "QUESTION_TYPE",
    "REACTION_TYPES",
    "RELATION_TYPES",
    "SUBSTEP_SCHEMA_VERSION",
    "SUBSTEP_SOURCE",
    "V3_CSV_FIELDNAMES",
    "StructuredEvent",
    "StructuredLayer",
    "StructuredRecord",
    "answer_json",
    "chat_prompt",
    "compact_json",
    "csv_row",
    "event_from_reaction",
    "event_from_relation",
    "graph_events",
    "graph_id_for_source",
    "normalized_pathway_id",
    "prefix_only_question",
    "records_jsonl",
    "record_from_object",
    "selected_prefix_lengths",
    "stable_id",
    "total_training_tokens",
    "TOPOLOGY_BACKBONE",
    "TOPOLOGY_CONTEXT",
    "TOPOLOGY_CONTEXT_CROSS_LAYER",
    "TOPOLOGY_EXCLUDED",
]
