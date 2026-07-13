"""Build stable sink-SCC views directly from canonical processed_graph events."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Iterable

from dataprocess.structured_schema import (
    StructuredEvent,
    StructuredLayer,
    StructuredRecord,
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
        for source in event.source_node_ids:
            adjacency[source]
        for target in event.target_node_ids:
            adjacency[target]
        for source in event.source_node_ids:
            adjacency[source].update(event.target_node_ids)
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


def _distances_to_sink(
    sink_component: int,
    reverse_graph: dict[int, set[int]],
) -> dict[int, int]:
    distances = {sink_component: 0}
    queue: deque[int] = deque((sink_component,))
    while queue:
        component = queue.popleft()
        for parent in sorted(reverse_graph[component]):
            if parent in distances:
                continue
            distances[parent] = distances[component] + 1
            queue.append(parent)
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
    """Create one record per sink SCC without text deduplication.

    Topology is built from every relation/reaction whose endpoints exist,
    including records the historical producer marked ``renderable=false``.
    The generic v3 renderer then gives every retained structural event a
    deterministic text value while preserving the producer flag for audits.
    """

    events, missing_endpoints = parsed_events if parsed_events is not None else graph_events(graph)
    if not events:
        return ()
    adjacency = _event_adjacency(events)
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
        distances = _distances_to_sink(sink_component, reverse_component_graph)
        max_distance = max(distances.values())
        raw_layers: dict[int, list[StructuredEvent]] = defaultdict(list)
        raw_distances: dict[int, int] = {}
        for event in events:
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
            )
        )
    return tuple(output)


__all__ = ["build_structured_records", "tarjan_scc"]
