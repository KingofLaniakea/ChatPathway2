"""Structured graph-event records and the prefix-only v3 model contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from dataprocess.schemas import CSV_FIELDNAMES, canonical_pathway_family_id


DATASET_SCHEMA_VERSION = "structured_pathway_record_v3"
ANSWER_SCHEMA_VERSION = "pathway_continuation_v3"
QUESTION_TYPE = "pathway_continuation_v3"
SUBSTEP_SCHEMA_VERSION = "graph_event_set_v3"
SUBSTEP_SOURCE = "processed_graph_structured_event_v3"


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


def node_value(node: dict[str, Any]) -> dict[str, str]:
    canonical_id = str(
        node.get("canonical_id")
        or next(iter(node.get("resolved_ids") or ()), "")
        or node.get("raw_name")
        or f"node:{node.get('node_id', '')}"
    ).strip()
    name = str(
        node.get("display_name")
        or node.get("raw_name")
        or canonical_id
    ).strip()
    return {"canonical_id": canonical_id, "name": name}


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


def relation_label(record: dict[str, Any]) -> str:
    tags = [
        str(value).strip()
        for value in record.get("semantic_tags", record.get("subtype_names", ()))
        if str(value).strip()
    ]
    if record.get("has_missing_interaction"):
        return "missing_interaction"
    return "+".join(tags) or str(record.get("relation_type") or "relation")


def relation_text(
    record: dict[str, Any],
    sources: Sequence[dict[str, str]],
    targets: Sequence[dict[str, str]],
) -> str:
    source_text = _joined_names(sources)
    target_text = _joined_names(targets)
    if record.get("has_missing_interaction"):
        return f"KEGG marks the interaction between {source_text} and {target_text} as missing in this pathway."
    tags = [
        str(value).strip().casefold()
        for value in record.get("semantic_tags", record.get("subtype_names", ()))
        if str(value).strip()
    ]
    phrase = next((_RELATION_PHRASES[tag] for tag in tags if tag in _RELATION_PHRASES), None)
    if phrase:
        return f"{source_text} {phrase} {target_text}."
    relation_type = str(record.get("relation_type") or "unspecified")
    subtype = ", ".join(tags) if tags else "unspecified subtype"
    return f"{source_text} has a KEGG {relation_type} relation ({subtype}) to {target_text}."


def reaction_text(
    record: dict[str, Any],
    sources: Sequence[dict[str, str]],
    targets: Sequence[dict[str, str]],
) -> str:
    source_text = _joined_names(sources)
    target_text = _joined_names(targets)
    reaction_type = str(record.get("reaction_type") or "unspecified")
    verb = "is" if len(sources) == 1 else "are"
    return f"{source_text} {verb} converted to {target_text} in a {reaction_type} way."


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
            "phenotype_status": "not_annotated",
            "parser_source": SUBSTEP_SOURCE,
        }


def event_from_relation(
    record: dict[str, Any],
    node_lookup: dict[int, dict[str, Any]],
) -> StructuredEvent | None:
    entry1_id = _integer_value(record, "entry1_id")
    entry2_id = _integer_value(record, "entry2_id")
    if entry1_id is None or entry2_id is None:
        return None
    source_ids = [entry1_id]
    mediator = _integer_value(record, "mediator_entry_id")
    if mediator is not None:
        source_ids.append(mediator)
    target_ids = [entry2_id]
    if any(node_id not in node_lookup for node_id in source_ids + target_ids):
        return None
    sources = tuple(node_value(node_lookup[node_id]) for node_id in source_ids)
    targets = tuple(node_value(node_lookup[node_id]) for node_id in target_ids)
    return StructuredEvent(
        event_id=f"relation:{int(record.get('relation_id', 0))}",
        event_type="relation",
        source_node_ids=tuple(source_ids),
        target_node_ids=tuple(target_ids),
        source=sources,
        relation=relation_label(record),
        target=targets,
        text=relation_text(record, sources, targets),
        producer_renderable=bool(record.get("renderable", False)),
    )


def event_from_reaction(
    record: dict[str, Any],
    node_lookup: dict[int, dict[str, Any]],
) -> StructuredEvent | None:
    source_ids = _integer_values(record, "substrate_entry_ids")
    target_ids = _integer_values(record, "product_entry_ids")
    if not source_ids or not target_ids:
        return None
    if any(node_id not in node_lookup for node_id in source_ids + target_ids):
        return None
    sources = tuple(node_value(node_lookup[node_id]) for node_id in source_ids)
    targets = tuple(node_value(node_lookup[node_id]) for node_id in target_ids)
    reaction_name = str(record.get("reaction_name") or "reaction")
    reaction_type = str(record.get("reaction_type") or "unspecified")
    return StructuredEvent(
        event_id=f"reaction:{int(record.get('reaction_id', 0))}",
        event_type="reaction",
        source_node_ids=source_ids,
        target_node_ids=target_ids,
        source=sources,
        relation=f"{reaction_name}:{reaction_type}",
        target=targets,
        text=reaction_text(record, sources, targets),
        producer_renderable=bool(record.get("renderable", False)),
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
        try:
            node_id = int(node["node_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"processed_graph node[{position}] has an invalid node_id"
            ) from exc
        if node_id in node_lookup:
            raise ValueError(f"processed_graph contains duplicate node_id={node_id}")
        node_lookup[node_id] = node
    events: list[StructuredEvent] = []
    missing_endpoints = 0
    for position, relation in enumerate(raw_relations):
        if not isinstance(relation, dict):
            raise ValueError(f"processed_graph relation[{position}] is not an object")
        event = event_from_relation(relation, node_lookup)
        if event is None:
            missing_endpoints += 1
        else:
            events.append(event)
    for position, reaction in enumerate(raw_reactions):
        if not isinstance(reaction, dict):
            raise ValueError(f"processed_graph reaction[{position}] is not an object")
        event = event_from_reaction(reaction, node_lookup)
        if event is None:
            missing_endpoints += 1
        else:
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
    return tuple(events), missing_endpoints


def observed_payload(layers: Sequence[StructuredLayer]) -> dict[str, object]:
    return {
        "observed_layers": [layer.model_object() for layer in layers],
    }


def answer_payload(layers: Sequence[StructuredLayer]) -> dict[str, object]:
    return {
        "schema_version": ANSWER_SCHEMA_VERSION,
        "remaining_layers": [layer.model_object() for layer in layers],
    }


def prefix_only_question(layers: Sequence[StructuredLayer]) -> str:
    skeleton = (
        '{"schema_version":"pathway_continuation_v3",'
        '"remaining_layers":[{"layer_index":<integer>,'
        '"events":[{"source":[{"canonical_id":"<string>","name":"<string>"}],'
        '"relation":"<string>",'
        '"target":[{"canonical_id":"<string>","name":"<string>"}],'
        '"text":"<string>"}]}]}'
    )
    return "\n".join(
        (
            "Continue the biological pathway from the observed upstream layers.",
            "Layers are ordered from upstream to downstream. Events inside one layer are an unordered set.",
            "Return only one complete JSON object. Do not repeat observed layers and do not add phenotype fields.",
            f"Required JSON shape: {skeleton}",
            f"Observed prefix: {compact_json(observed_payload(layers))}",
        )
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


def csv_row(record: StructuredRecord, prefix_len: int) -> dict[str, object]:
    if not 0 < prefix_len < len(record.layers):
        raise ValueError("prefix_len must leave at least one observed and one target layer")
    remaining = record.layers[prefix_len:]
    sample_id = f"{record.record_id}:prefix={prefix_len}"
    row: dict[str, object] = {field: "" for field in CSV_FIELDNAMES}
    row.update(
        {
            "sample_id": sample_id,
            "record_id": record.record_id,
            "question": prefix_only_question(record.layers[:prefix_len]),
            "answer": answer_json(remaining),
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
            "target_step_count": len(remaining),
            "has_empty_prefix": 0,
            "substep_schema_version": SUBSTEP_SCHEMA_VERSION,
            "substep_source": SUBSTEP_SOURCE,
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
    layers: list[StructuredLayer] = []
    raw_layers = value.get("layers")
    if not isinstance(raw_layers, list) or not raw_layers:
        raise ValueError("structured record layers must be a non-empty list")
    for expected_layer_index, layer_value in enumerate(raw_layers):
        if not isinstance(layer_value, dict) or set(layer_value) != {
            "layer_index",
            "distance_to_sink",
            "events",
        }:
            raise ValueError("structured record layer does not exactly match v3")
        if layer_value.get("layer_index") != expected_layer_index:
            raise ValueError("structured record layer indices must be consecutive")
        raw_events = layer_value.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            raise ValueError("structured record layers must contain events")
        events: list[StructuredEvent] = []
        for event_value in raw_events:
            if not isinstance(event_value, dict) or set(event_value) != {
                "event_id",
                "event_type",
                "source_node_ids",
                "target_node_ids",
                "source",
                "relation",
                "target",
                "text",
                "producer_renderable",
            }:
                raise ValueError("structured record event does not exactly match v3")
            raw_source = event_value.get("source")
            raw_target = event_value.get("target")
            if (
                not isinstance(raw_source, list)
                or not raw_source
                or not all(_record_entity_valid(item) for item in raw_source)
                or not isinstance(raw_target, list)
                or not raw_target
                or not all(_record_entity_valid(item) for item in raw_target)
            ):
                raise ValueError("structured record event entities are invalid")
            if not str(event_value.get("event_id", "")).strip():
                raise ValueError("structured record event_id is empty")
            if event_value.get("event_type") not in {"relation", "reaction"}:
                raise ValueError("structured record event_type is invalid")
            if not str(event_value.get("relation", "")).strip():
                raise ValueError("structured record relation is empty")
            if not str(event_value.get("text", "")).strip():
                raise ValueError("structured record text is empty")
            if not isinstance(event_value.get("producer_renderable"), bool):
                raise ValueError("structured record producer_renderable must be boolean")
            source_node_ids = tuple(int(item) for item in event_value["source_node_ids"])
            target_node_ids = tuple(int(item) for item in event_value["target_node_ids"])
            if not source_node_ids or len(source_node_ids) != len(raw_source):
                raise ValueError("structured record source node/entity counts disagree")
            if not target_node_ids or len(target_node_ids) != len(raw_target):
                raise ValueError("structured record target node/entity counts disagree")
            events.append(
                StructuredEvent(
                    event_id=str(event_value["event_id"]),
                    event_type=str(event_value["event_type"]),
                    source_node_ids=source_node_ids,
                    target_node_ids=target_node_ids,
                    source=tuple(dict(item) for item in raw_source),
                    relation=str(event_value["relation"]),
                    target=tuple(dict(item) for item in raw_target),
                    text=str(event_value["text"]),
                    producer_renderable=event_value["producer_renderable"],
                )
            )
        layers.append(
            StructuredLayer(
                layer_index=int(layer_value["layer_index"]),
                distance_to_sink=int(layer_value["distance_to_sink"]),
                events=tuple(events),
            )
        )
    sink_node_ids = tuple(int(item) for item in value["sink_node_ids"])
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
        graph_event_count=int(value["graph_event_count"]),
        graph_missing_endpoint_event_count=int(value["graph_missing_endpoint_event_count"]),
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
    "DATASET_SCHEMA_VERSION",
    "QUESTION_TYPE",
    "SUBSTEP_SCHEMA_VERSION",
    "SUBSTEP_SOURCE",
    "StructuredEvent",
    "StructuredLayer",
    "StructuredRecord",
    "answer_json",
    "chat_prompt",
    "compact_json",
    "csv_row",
    "graph_events",
    "graph_id_for_source",
    "normalized_pathway_id",
    "prefix_only_question",
    "records_jsonl",
    "record_from_object",
    "selected_prefix_lengths",
    "stable_id",
    "total_training_tokens",
]
