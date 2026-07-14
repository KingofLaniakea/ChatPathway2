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
                    events=tuple(sorted(raw_layers[raw_index], key=_event_sort_key)),
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
