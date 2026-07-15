"""Build stable sink-SCC views directly from canonical processed_graph events."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import replace
from typing import Any, Iterable

from dataprocess.structured_schema import (
    StructuredEvent,
    StructuredLayer,
    StructuredRecord,
    TOPOLOGY_BACKBONE,
    TOPOLOGY_CONTEXT,
    TOPOLOGY_CONTEXT_CROSS_LAYER,
    TOPOLOGY_EXCLUDED,
    graph_events,
    normalized_pathway_id,
    stable_id,
)


def tarjan_scc(adjacency: dict[int, set[int]]) -> tuple[tuple[tuple[int, ...], ...], dict[int, int]]:
    """Deterministic iterative Tarjan SCC decomposition.

    Some KEGG maps contain enough nodes to exceed Python's recursion limit, so
    the full-corpus builder must not use a recursive DFS.
    """

    index = 0
    stack: list[int] = []
    on_stack: set[int] = set()
    indices: dict[int, int] = {}
    lowlinks: dict[int, int] = {}
    components: list[tuple[int, ...]] = []

    for root in sorted(adjacency):
        if root in indices:
            continue
        parents: dict[int, int] = {}
        frames: list[tuple[int, Any]] = []

        def push(node: int) -> None:
            nonlocal index
            indices[node] = index
            lowlinks[node] = index
            index += 1
            stack.append(node)
            on_stack.add(node)
            frames.append((node, iter(sorted(adjacency[node]))))

        push(root)
        while frames:
            node, neighbors = frames[-1]
            try:
                neighbor = next(neighbors)
            except StopIteration:
                frames.pop()
                if lowlinks[node] == indices[node]:
                    members: list[int] = []
                    while stack:
                        member = stack.pop()
                        on_stack.remove(member)
                        members.append(member)
                        if member == node:
                            break
                    components.append(tuple(sorted(members)))
                if node in parents:
                    parent = parents[node]
                    lowlinks[parent] = min(lowlinks[parent], lowlinks[node])
                continue
            if neighbor not in indices:
                parents[neighbor] = node
                push(neighbor)
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

    ordered = tuple(sorted(components, key=lambda values: (min(values), values)))
    node_to_component = {
        node: component_index
        for component_index, component in enumerate(ordered)
        for node in component
    }
    return ordered, node_to_component


def _event_adjacency(events: Iterable[StructuredEvent]) -> dict[int, set[int]]:
    adjacency: dict[int, set[int]] = defaultdict(set)
    for event in events:
        if event.topology_role != TOPOLOGY_BACKBONE or not event.core_included:
            continue
        for source, target in event.topology_arcs:
            adjacency[source]
            adjacency[target]
            adjacency[source].add(target)
    return {node: set(neighbors) for node, neighbors in adjacency.items()}


def _component_graph(
    adjacency: dict[int, set[int]],
    component_count: int,
    node_to_component: dict[int, int],
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    forward: dict[int, set[int]] = defaultdict(set)
    reverse: dict[int, set[int]] = defaultdict(set)
    for source, targets in adjacency.items():
        source_component = node_to_component[source]
        for target in targets:
            target_component = node_to_component[target]
            if source_component == target_component:
                continue
            forward[source_component].add(target_component)
            reverse[target_component].add(source_component)
    for component_index in range(component_count):
        forward[component_index]
        reverse[component_index]
    return dict(forward), dict(reverse)


def _ancestors_of_sink(
    sink_component: int,
    reverse_graph: dict[int, set[int]],
) -> set[int]:
    ancestors = {sink_component}
    queue: deque[int] = deque((sink_component,))
    while queue:
        component = queue.popleft()
        for parent in sorted(reverse_graph[component]):
            if parent in ancestors:
                continue
            ancestors.add(parent)
            queue.append(parent)
    return ancestors


def _longest_distances_to_sink(
    sink_component: int,
    component_graph: dict[int, set[int]],
    reverse_graph: dict[int, set[int]],
) -> dict[int, int]:
    """Longest DAG distance guarantees monotone upstream/downstream layers."""

    ancestors = _ancestors_of_sink(sink_component, reverse_graph)
    remaining_children = {
        component: len(component_graph[component].intersection(ancestors))
        for component in ancestors
    }
    distances = {component: 0 for component in ancestors}
    queue: deque[int] = deque(
        sorted(component for component, count in remaining_children.items() if count == 0)
    )
    processed = 0
    while queue:
        component = queue.popleft()
        processed += 1
        for parent in sorted(reverse_graph[component].intersection(ancestors)):
            distances[parent] = max(distances[parent], distances[component] + 1)
            remaining_children[parent] -= 1
            if remaining_children[parent] == 0:
                queue.append(parent)
    if processed != len(ancestors):
        raise ValueError("condensed component graph is not acyclic")
    return distances


def _event_sort_key(event: StructuredEvent) -> tuple[int, int, str]:
    return (
        min(event.target_node_ids),
        min(event.source_node_ids),
        event.event_id,
    )


def _merge_endpoint_provenance(
    events: Iterable[StructuredEvent],
    *,
    node_ids_attribute: str,
    provenance_attribute: str,
) -> tuple[tuple[int, ...], tuple[dict[str, object], ...]]:
    """Union occurrence-node provenance without duplicating model entities."""

    by_node_id: dict[int, dict[str, object]] = {}
    for event in events:
        node_ids = getattr(event, node_ids_attribute)
        provenance = getattr(event, provenance_attribute)
        if len(node_ids) != len(provenance):
            raise ValueError("event endpoint IDs and provenance lengths disagree")
        for node_id, value in zip(node_ids, provenance):
            materialized = dict(value)
            prior = by_node_id.get(node_id)
            if prior is not None and prior != materialized:
                raise ValueError(
                    f"node_id={node_id} has conflicting event provenance"
                )
            by_node_id[node_id] = materialized
    ordered_ids = tuple(sorted(by_node_id))
    return ordered_ids, tuple(by_node_id[node_id] for node_id in ordered_ids)


def _merge_semantic_events(events: Iterable[StructuredEvent]) -> StructuredEvent:
    """Merge model-identical raw occurrences assigned to the same layer.

    Occurrence nodes may be duplicated on a KEGG map even when their resolved
    participants and biological action are identical.  They must remain
    separate until SCC/layer assignment, because occurrence topology differs;
    only then can the model-visible duplicate be collapsed.  All producer IDs
    and occurrence-node provenance remain in the canonical record.
    """

    ordered = tuple(sorted(events, key=_event_sort_key))
    if not ordered:
        raise ValueError("cannot merge an empty semantic-event group")
    first = ordered[0]
    if any(event.semantic_event_id != first.semantic_event_id for event in ordered):
        raise ValueError("semantic-event merge group contains different identities")
    if any(event.model_object() != first.model_object() for event in ordered):
        raise ValueError("semantic-event identity collision changes model-visible content")
    if any(not event.core_included for event in ordered):
        raise ValueError("excluded events cannot be merged into a model-visible layer")

    source_node_ids, source_provenance = _merge_endpoint_provenance(
        ordered,
        node_ids_attribute="source_node_ids",
        provenance_attribute="source_entity_provenance",
    )
    target_node_ids, target_provenance = _merge_endpoint_provenance(
        ordered,
        node_ids_attribute="target_node_ids",
        provenance_attribute="target_entity_provenance",
    )
    mediator_node_ids, mediator_provenance = _merge_endpoint_provenance(
        ordered,
        node_ids_attribute="mediator_node_ids",
        provenance_attribute="mediator_entity_provenance",
    )

    producer_event_ids: list[str] = []
    producer_renderable_event_ids: list[str] = []
    reaction_names: list[str] = []
    raw_subtypes: list[dict[str, str]] = []
    seen_raw_subtypes: set[tuple[str, str]] = set()
    for event in ordered:
        for producer_event_id in event.producer_event_ids:
            if producer_event_id in producer_event_ids:
                raise ValueError(
                    f"duplicate producer event ID {producer_event_id!r} during merge"
                )
            producer_event_ids.append(producer_event_id)
        for producer_event_id in event.producer_renderable_event_ids:
            if producer_event_id in producer_renderable_event_ids:
                raise ValueError(
                    f"duplicate renderable producer ID {producer_event_id!r} during merge"
                )
            producer_renderable_event_ids.append(producer_event_id)
        for reaction_name in event.raw_reaction_names:
            if reaction_name not in reaction_names:
                reaction_names.append(reaction_name)
        for subtype in event.raw_subtypes:
            key = (str(subtype["name"]), str(subtype["value"]))
            if key not in seen_raw_subtypes:
                seen_raw_subtypes.add(key)
                raw_subtypes.append(dict(subtype))

    legacy_texts = {
        event.legacy_text for event in ordered if event.legacy_text is not None
    }
    if len(legacy_texts) > 1:
        raise ValueError("semantic duplicates disagree on legacy event text")
    legacy_text = next(iter(legacy_texts), None)
    if legacy_text is not None:
        text_source = next(
            event.text_source for event in ordered if event.legacy_text is not None
        )
    else:
        text_source = first.text_source

    roles = {event.topology_role for event in ordered}
    if TOPOLOGY_BACKBONE in roles:
        topology_role = TOPOLOGY_BACKBONE
    elif TOPOLOGY_CONTEXT_CROSS_LAYER in roles:
        topology_role = TOPOLOGY_CONTEXT_CROSS_LAYER
    else:
        topology_role = TOPOLOGY_CONTEXT
    topology_arcs = tuple(
        sorted({arc for event in ordered for arc in event.topology_arcs})
    )
    if topology_role != TOPOLOGY_BACKBONE:
        topology_arcs = ()

    return replace(
        first,
        event_id=producer_event_ids[0],
        producer_event_ids=tuple(producer_event_ids),
        producer_renderable_event_ids=tuple(producer_renderable_event_ids),
        source_node_ids=source_node_ids,
        target_node_ids=target_node_ids,
        source_entity_provenance=source_provenance,
        target_entity_provenance=target_provenance,
        mediator_node_ids=mediator_node_ids,
        mediator_entity_provenance=mediator_provenance,
        legacy_text=legacy_text,
        text_source=text_source,
        producer_renderable_count=len(producer_renderable_event_ids),
        raw_reaction_names=tuple(reaction_names),
        raw_subtypes=tuple(raw_subtypes),
        topology_role=topology_role,
        topology_arcs=topology_arcs,
        core_included=True,
        exclusion_reason="",
    )


def _deduplicate_layer_events(
    events: Iterable[StructuredEvent],
) -> tuple[StructuredEvent, ...]:
    grouped: dict[str, list[StructuredEvent]] = defaultdict(list)
    for event in events:
        grouped[event.semantic_event_id].append(event)
    merged = (_merge_semantic_events(group) for group in grouped.values())
    return tuple(sorted(merged, key=_event_sort_key))


def build_structured_records(
    graph: dict[str, Any],
    *,
    graph_id: str,
    source_graph_json: str,
    parsed_events: tuple[tuple[StructuredEvent, ...], int] | None = None,
) -> tuple[StructuredRecord, ...]:
    """Create one record per sink SCC from the strict ordering backbone.

    Direction-supported relations and reaction projections alone define SCCs,
    sinks, and longest-distance layers.  Non-directional relations are attached
    afterwards and never expand a sink view.  A graph with any rejected event
    is rejected as a whole rather than rebuilt after silently dropping an edge.
    """

    events, missing_endpoints = parsed_events if parsed_events is not None else graph_events(graph)
    if missing_endpoints:
        return ()
    if not events:
        return ()
    backbone_events = tuple(
        event
        for event in events
        if event.core_included and event.topology_role == TOPOLOGY_BACKBONE
    )
    context_events = tuple(
        event
        for event in events
        if event.core_included and event.topology_role == TOPOLOGY_CONTEXT
    )
    globally_excluded_events = tuple(
        event
        for event in events
        if not event.core_included or event.topology_role == TOPOLOGY_EXCLUDED
    )
    adjacency = _event_adjacency(backbone_events)
    if not adjacency:
        return ()
    components, node_to_component = tarjan_scc(adjacency)
    component_graph, reverse_component_graph = _component_graph(
        adjacency,
        len(components),
        node_to_component,
    )
    sinks = sorted(
        (
            component_index
            for component_index in range(len(components))
            if not component_graph[component_index]
        ),
        key=lambda component_index: components[component_index],
    )

    metadata = graph.get("metadata") if isinstance(graph.get("metadata"), dict) else {}
    organism = str(metadata.get("organism") or source_graph_json.split("/", 1)[0]).strip()
    pathway_id = normalized_pathway_id(metadata.get("pathway_id") or source_graph_json.rsplit("/", 1)[-1].removesuffix(".json"))
    pathway_title = str(metadata.get("title") or "").strip()

    output: list[StructuredRecord] = []
    for sink_component in sinks:
        distances = _longest_distances_to_sink(
            sink_component,
            component_graph,
            reverse_component_graph,
        )
        max_distance = max(distances.values())
        raw_layers: dict[int, list[StructuredEvent]] = defaultdict(list)
        raw_distances: dict[int, int] = {}
        for event in backbone_events:
            target_distances = [
                distances[node_to_component[target]]
                for target in event.target_node_ids
                if node_to_component[target] in distances
            ]
            if not target_distances:
                continue
            distance = min(target_distances)
            raw_index = max_distance - distance
            raw_layers[raw_index].append(event)
            raw_distances[raw_index] = distance

        view_excluded_events: list[StructuredEvent] = list(globally_excluded_events)
        for event in context_events:
            endpoint_ids = event.source_node_ids + event.target_node_ids
            if any(node_id not in node_to_component for node_id in endpoint_ids):
                view_excluded_events.append(
                    replace(
                        event,
                        topology_role=TOPOLOGY_EXCLUDED,
                        core_included=False,
                        exclusion_reason="context_not_attached_to_view",
                    )
                )
                continue
            endpoint_components = [node_to_component[node_id] for node_id in endpoint_ids]
            if any(component not in distances for component in endpoint_components):
                view_excluded_events.append(
                    replace(
                        event,
                        topology_role=TOPOLOGY_EXCLUDED,
                        core_included=False,
                        exclusion_reason="context_not_attached_to_view",
                    )
                )
                continue
            endpoint_indices = [
                max_distance - distances[component]
                for component in endpoint_components
            ]
            raw_index = max(endpoint_indices)
            attached_event = event
            if len(set(endpoint_indices)) > 1:
                attached_event = replace(
                    event,
                    topology_role=TOPOLOGY_CONTEXT_CROSS_LAYER,
                )
            raw_layers[raw_index].append(attached_event)
            raw_distances[raw_index] = max_distance - raw_index

        layers: list[StructuredLayer] = []
        for normalized_index, raw_index in enumerate(sorted(raw_layers)):
            layers.append(
                StructuredLayer(
                    layer_index=normalized_index,
                    distance_to_sink=raw_distances[raw_index],
                    events=_deduplicate_layer_events(raw_layers[raw_index]),
                )
            )
        if not layers:
            continue
        sink_node_ids = components[sink_component]
        view_id = stable_id("view", graph_id, ",".join(map(str, sink_node_ids)))
        record_id = stable_id("record", graph_id, view_id)
        output.append(
            StructuredRecord(
                graph_id=graph_id,
                view_id=view_id,
                record_id=record_id,
                source_graph_json=source_graph_json,
                organism=organism,
                pathway_id=pathway_id,
                pathway_title=pathway_title,
                sink_node_ids=sink_node_ids,
                layers=tuple(layers),
                graph_event_count=len(events) + missing_endpoints,
                graph_missing_endpoint_event_count=missing_endpoints,
                excluded_events=tuple(
                    sorted(view_excluded_events, key=_event_sort_key)
                ),
            )
        )
    return tuple(output)


__all__ = ["build_structured_records", "tarjan_scc"]
