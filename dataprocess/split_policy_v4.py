"""Data-internal source/family splits and exact horizon balancing for v4."""

from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


MAIN_SPLITS = ("train", "validation", "test")
MAIN_RATIOS = {"train": 0.70, "validation": 0.20, "test": 0.10}
HORIZONS = ("long_target", "middle_target", "short_target")


def stable_rank(value: str, seed: int, namespace: str) -> str:
    return hashlib.sha256(
        f"{namespace}:{seed}:{value}".encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class SourceCoverage:
    records: int
    graphs: int
    families: int
    layer_total: int
    semantic_event_total: int

    def __post_init__(self) -> None:
        if min(
            self.records,
            self.graphs,
            self.families,
            self.layer_total,
            self.semantic_event_total,
        ) <= 0:
            raise ValueError("source coverage values must all be positive")


def choose_coverage_holdout_sources(
    coverage: Mapping[str, SourceCoverage],
    *,
    fraction: float,
    seed: int,
    protected_sources: Iterable[str] = ("hsa", "ko", "ec"),
    strata_count: int = 10,
) -> tuple[set[str], dict[str, object]]:
    """Hold out source codes using only statistics present in the indexed corpus.

    Sources are ordered by record/family/graph/trajectory coverage and divided
    into equal-sized quantile strata.  Seeded selection inside each stratum
    keeps small and large organisms represented without depending on a mutable
    external taxonomy snapshot.  ``ko`` and ``ec`` are species-neutral
    reference sources, and ``hsa`` is the explicitly protected primary source.
    """

    if not 0 < fraction < 1:
        raise ValueError("source holdout fraction must be in (0, 1)")
    if strata_count < 1:
        raise ValueError("strata_count must be positive")
    if not coverage:
        raise ValueError("source coverage cannot be empty")
    if any(not str(source).strip() for source in coverage):
        raise ValueError("source coverage contains an empty source code")
    protected = set(protected_sources) & set(coverage)
    candidates = sorted(
        set(coverage) - protected,
        key=lambda source: (
            coverage[source].records,
            coverage[source].families,
            coverage[source].graphs,
            coverage[source].layer_total / coverage[source].records,
            coverage[source].semantic_event_total / coverage[source].records,
            source,
        ),
    )
    if len(candidates) < 2:
        raise ValueError("at least two non-protected source codes are required")
    # Keep at least two candidates in each stratum so even tiny audited test
    # fixtures can reserve a source while retaining a seen counterpart.
    effective_strata = min(strata_count, max(1, len(candidates) // 2))
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for position, source in enumerate(candidates):
        stratum_index = min(
            effective_strata - 1,
            position * effective_strata // len(candidates),
        )
        by_stratum[f"coverage_quantile_{stratum_index:02d}"].append(source)

    heldout: set[str] = set()
    strata_report: dict[str, dict[str, int]] = {}
    for stratum, values in sorted(by_stratum.items()):
        ranked = sorted(
            values,
            key=lambda source: (
                stable_rank(source, seed, f"coverage_holdout:{stratum}"),
                source,
            ),
        )
        if len(ranked) <= 1:
            count = 0
        else:
            count = min(len(ranked) - 1, max(1, round(len(ranked) * fraction)))
        heldout.update(ranked[:count])
        strata_report[stratum] = {"sources": len(ranked), "heldout": count}

    def totals(sources: Iterable[str]) -> dict[str, int]:
        values = list(sources)
        return {
            "sources": len(values),
            "records": sum(coverage[source].records for source in values),
            "graphs": sum(coverage[source].graphs for source in values),
            "families_sum": sum(coverage[source].families for source in values),
            "layer_total": sum(coverage[source].layer_total for source in values),
            "semantic_event_total": sum(
                coverage[source].semantic_event_total for source in values
            ),
        }

    return heldout, {
        "fraction": fraction,
        "policy": "dataset_internal_coverage_quantile_stratified_source_holdout",
        "claims_phylogenetic_balance": False,
        "present_sources": len(coverage),
        "heldout_sources": len(heldout),
        "protected_sources": sorted(protected),
        "all_coverage": totals(coverage),
        "heldout_coverage": totals(heldout),
        "seen_coverage": totals(set(coverage) - heldout),
        "strata": strata_report,
    }


@dataclass(frozen=True)
class FamilyWeight:
    total: int
    by_organism: Mapping[str, int]


def _squared_error(value: float, target: float, scale: float) -> float:
    return (value - target) ** 2 / max(1.0, scale)


def assign_family_splits(
    weights: Mapping[str, FamilyWeight],
    *,
    seed: int,
    ratios: Mapping[str, float] = MAIN_RATIOS,
) -> tuple[dict[str, str], dict[str, object]]:
    """Assign each complete KEGG family to one main split.

    The deterministic optimizer minimizes weighted global and per-organism
    ratio error while treating a family as indivisible.  This preserves strict
    family disjointness; exact row ratios are not falsely promised when one
    large family makes them mathematically impossible.
    """

    if set(ratios) != set(MAIN_SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("family split ratios must define train/validation/test and sum to 1")
    if not weights or len(weights) < len(MAIN_SPLITS):
        raise ValueError("at least three non-empty families are required")
    if any(weight.total <= 0 for weight in weights.values()):
        raise ValueError("family weights must be positive")

    overall_total = sum(weight.total for weight in weights.values())
    organism_totals: dict[str, int] = defaultdict(int)
    for weight in weights.values():
        for organism, count in weight.by_organism.items():
            organism_totals[organism] += int(count)
    overall_counts = {split: 0 for split in MAIN_SPLITS}
    organism_counts: dict[str, dict[str, int]] = {
        organism: {split: 0 for split in MAIN_SPLITS}
        for organism in organism_totals
    }

    def assignment_delta(family: str, old: str | None, new: str) -> float:
        weight = weights[family]
        delta = 0.0
        for split in MAIN_SPLITS:
            before = overall_counts[split]
            after = before
            if old == split:
                after -= weight.total
            if new == split:
                after += weight.total
            target = ratios[split] * overall_total
            delta += 12.0 * (
                _squared_error(after, target, overall_total)
                - _squared_error(before, target, overall_total)
            )
        for organism, count in weight.by_organism.items():
            total = organism_totals[organism]
            for split in MAIN_SPLITS:
                before = organism_counts[organism][split]
                after = before
                if old == split:
                    after -= count
                if new == split:
                    after += count
                target = ratios[split] * total
                delta += (
                    _squared_error(after, target, total)
                    - _squared_error(before, target, total)
                )
        return delta

    assignment: dict[str, str] = {}
    ordered_families = sorted(
        weights,
        key=lambda family: (
            -weights[family].total,
            stable_rank(family, seed, "family_split_order"),
            family,
        ),
    )
    for family in ordered_families:
        split = min(
            MAIN_SPLITS,
            key=lambda candidate: (
                assignment_delta(family, None, candidate),
                stable_rank(f"{family}:{candidate}", seed, "family_split_tie"),
                candidate,
            ),
        )
        assignment[family] = split
        weight = weights[family]
        overall_counts[split] += weight.total
        for organism, count in weight.by_organism.items():
            organism_counts[organism][split] += int(count)

    # A small synthetic corpus or a very heavy train target can otherwise make
    # the unconstrained greedy pass leave a split empty.  Seed every missing
    # split with the least-cost whole-family move before local refinement.
    for missing_split in MAIN_SPLITS:
        if missing_split in assignment.values():
            continue
        candidates = [
            family
            for family, old_split in assignment.items()
            if sum(value == old_split for value in assignment.values()) > 1
        ]
        if not candidates:
            raise ValueError("family optimizer cannot seed every main split")
        family = min(
            candidates,
            key=lambda value: (
                assignment_delta(value, assignment[value], missing_split),
                value,
            ),
        )
        old = assignment[family]
        weight = weights[family]
        overall_counts[old] -= weight.total
        overall_counts[missing_split] += weight.total
        for organism, count in weight.by_organism.items():
            organism_counts[organism][old] -= int(count)
            organism_counts[organism][missing_split] += int(count)
        assignment[family] = missing_split

    # Deterministic single-family local improvement.  Moving an indivisible
    # family is the correct operation; moving rows would reintroduce leakage.
    for _pass in range(50):
        improved = False
        for family in sorted(assignment):
            old = assignment[family]
            if sum(value == old for value in assignment.values()) <= 1:
                continue
            candidates = [split for split in MAIN_SPLITS if split != old]
            best = min(
                candidates,
                key=lambda candidate: (
                    assignment_delta(family, old, candidate),
                    candidate,
                ),
            )
            delta = assignment_delta(family, old, best)
            if delta >= -1e-12:
                continue
            weight = weights[family]
            overall_counts[old] -= weight.total
            overall_counts[best] += weight.total
            for organism, count in weight.by_organism.items():
                organism_counts[organism][old] -= int(count)
                organism_counts[organism][best] += int(count)
            assignment[family] = best
            improved = True
        if not improved:
            break

    if set(assignment.values()) != set(MAIN_SPLITS):
        raise ValueError("family optimizer left an empty main split")
    report = {
        "ratios_requested": dict(ratios),
        "record_counts": dict(overall_counts),
        "record_ratios": {
            split: overall_counts[split] / overall_total for split in MAIN_SPLITS
        },
        "family_counts": {
            split: sum(value == split for value in assignment.values())
            for split in MAIN_SPLITS
        },
        "overall_records": overall_total,
        "organisms": len(organism_totals),
        "maximum_family_weight": max(weight.total for weight in weights.values()),
        "constraint": "family_indivisible_and_disjoint",
    }
    return assignment, report


@dataclass(frozen=True)
class PrefixChoice:
    prefix_len: int
    horizon: str


def prefix_choices(layer_count: int) -> tuple[PrefixChoice, ...]:
    if layer_count < 2:
        return ()
    candidates = tuple(range(1, layer_count))
    if len(candidates) == 1:
        return (PrefixChoice(candidates[0], "short_target"),)
    output = [
        PrefixChoice(candidates[0], "long_target"),
        PrefixChoice(candidates[-1], "short_target"),
    ]
    if len(candidates) >= 3:
        output.insert(1, PrefixChoice(candidates[len(candidates) // 2], "middle_target"))
    return tuple(output)


class _Edge:
    __slots__ = ("to", "reverse", "capacity", "initial_capacity")

    def __init__(self, to: int, reverse: int, capacity: int) -> None:
        self.to = to
        self.reverse = reverse
        self.capacity = capacity
        self.initial_capacity = capacity


class _Dinic:
    def __init__(self, nodes: int) -> None:
        self.graph: list[list[_Edge]] = [[] for _ in range(nodes)]

    def add(self, source: int, target: int, capacity: int) -> _Edge:
        forward = _Edge(target, len(self.graph[target]), capacity)
        reverse = _Edge(source, len(self.graph[source]), 0)
        self.graph[source].append(forward)
        self.graph[target].append(reverse)
        return forward

    def flow(self, source: int, target: int) -> int:
        total = 0
        while True:
            level = [-1] * len(self.graph)
            level[source] = 0
            queue: deque[int] = deque((source,))
            while queue:
                node = queue.popleft()
                for edge in self.graph[node]:
                    if edge.capacity > 0 and level[edge.to] < 0:
                        level[edge.to] = level[node] + 1
                        queue.append(edge.to)
            if level[target] < 0:
                return total
            cursor = [0] * len(self.graph)

            def send(node: int, amount: int) -> int:
                if node == target:
                    return amount
                while cursor[node] < len(self.graph[node]):
                    edge = self.graph[node][cursor[node]]
                    if edge.capacity > 0 and level[edge.to] == level[node] + 1:
                        pushed = send(edge.to, min(amount, edge.capacity))
                        if pushed:
                            edge.capacity -= pushed
                            self.graph[edge.to][edge.reverse].capacity += pushed
                            return pushed
                    cursor[node] += 1
                return 0

            while True:
                pushed = send(source, 10**18)
                if not pushed:
                    break
                total += pushed


def _bounded_horizon_flow(
    groups: Mapping[tuple[str, ...], int],
    *,
    lower: int,
    upper: int,
) -> tuple[bool, dict[tuple[str, ...], dict[str, int]]]:
    group_keys = sorted(groups)
    group_nodes = {key: index for index, key in enumerate(group_keys)}
    horizon_nodes = {
        horizon: len(group_nodes) + index for index, horizon in enumerate(HORIZONS)
    }
    source = len(group_nodes) + len(HORIZONS)
    sink = source + 1
    super_source = sink + 1
    super_sink = super_source + 1
    network = _Dinic(super_sink + 1)
    demand = [0] * (super_sink + 1)
    references: dict[tuple[tuple[str, ...], str], _Edge] = {}

    def bounded_edge(left: int, right: int, minimum: int, maximum: int) -> _Edge:
        if not 0 <= minimum <= maximum:
            raise ValueError("invalid bounded-flow capacity")
        edge = network.add(left, right, maximum - minimum)
        demand[left] -= minimum
        demand[right] += minimum
        return edge

    total = sum(groups.values())
    for key in group_keys:
        count = groups[key]
        bounded_edge(source, group_nodes[key], count, count)
        for horizon in key:
            references[(key, horizon)] = bounded_edge(
                group_nodes[key], horizon_nodes[horizon], 0, count
            )
    for horizon in HORIZONS:
        bounded_edge(horizon_nodes[horizon], sink, lower, upper)
    bounded_edge(sink, source, total, total)
    required = 0
    for node, value in enumerate(demand[: super_source]):
        if value > 0:
            network.add(super_source, node, value)
            required += value
        elif value < 0:
            network.add(node, super_sink, -value)
    feasible = network.flow(super_source, super_sink) == required
    if not feasible:
        return False, {}
    allocation: dict[tuple[str, ...], dict[str, int]] = {}
    for key in group_keys:
        allocation[key] = {}
        for horizon in key:
            edge = references[(key, horizon)]
            allocation[key][horizon] = edge.initial_capacity - edge.capacity
    return True, allocation


def assign_exact_horizons(
    eligible: Mapping[str, Sequence[PrefixChoice]],
    *,
    seed: int,
) -> tuple[dict[str, PrefixChoice], dict[str, object]]:
    """Return a globally feasible assignment with a proven balance tolerance."""

    if not eligible:
        raise ValueError("cannot balance an empty horizon population")
    by_group: dict[tuple[str, ...], list[str]] = defaultdict(list)
    choice_lookup: dict[tuple[str, str], PrefixChoice] = {}
    for record_id, choices in eligible.items():
        unique: dict[str, PrefixChoice] = {}
        for choice in choices:
            if choice.horizon not in HORIZONS:
                raise ValueError(f"unknown horizon {choice.horizon!r}")
            prior = unique.get(choice.horizon)
            if prior is not None and prior.prefix_len != choice.prefix_len:
                raise ValueError("one record has two prefix lengths for one horizon")
            unique[choice.horizon] = choice
            choice_lookup[(record_id, choice.horizon)] = choice
        if not unique:
            raise ValueError(f"record {record_id!r} has no eligible horizon")
        by_group[tuple(sorted(unique))].append(record_id)
    group_counts = {key: len(values) for key, values in by_group.items()}
    total = len(eligible)
    floor_target = total // len(HORIZONS)
    ceil_target = (total + len(HORIZONS) - 1) // len(HORIZONS)
    low, high = 0, total
    best_allocation: dict[tuple[str, ...], dict[str, int]] = {}
    while low < high:
        middle = (low + high) // 2
        feasible, allocation = _bounded_horizon_flow(
            group_counts,
            lower=max(0, floor_target - middle),
            upper=min(total, ceil_target + middle),
        )
        if feasible:
            high = middle
            best_allocation = allocation
        else:
            low = middle + 1
    tolerance = low
    feasible, best_allocation = _bounded_horizon_flow(
        group_counts,
        lower=max(0, floor_target - tolerance),
        upper=min(total, ceil_target + tolerance),
    )
    if not feasible:
        raise RuntimeError("internal bounded-flow solver failed at proven tolerance")

    assignments: dict[str, PrefixChoice] = {}
    for key in sorted(by_group):
        records = sorted(
            by_group[key],
            key=lambda record_id: (
                stable_rank(record_id, seed, f"horizon_group:{','.join(key)}"),
                record_id,
            ),
        )
        cursor = 0
        for horizon in HORIZONS:
            count = best_allocation[key].get(horizon, 0)
            for record_id in records[cursor : cursor + count]:
                assignments[record_id] = choice_lookup[(record_id, horizon)]
            cursor += count
        if cursor != len(records):
            raise RuntimeError("horizon group allocation does not consume every record")
    counts = {
        horizon: sum(choice.horizon == horizon for choice in assignments.values())
        for horizon in HORIZONS
    }
    return assignments, {
        "records": total,
        "counts": counts,
        "max_minus_min": max(counts.values()) - min(counts.values()),
        "ideal_floor": floor_target,
        "ideal_ceil": ceil_target,
        "theoretical_optimal_tolerance": tolerance,
        "actual_matches_theoretical_optimum": True,
        "eligibility_groups": {
            "+".join(key): count for key, count in sorted(group_counts.items())
        },
    }


__all__ = [
    "FamilyWeight",
    "HORIZONS",
    "MAIN_RATIOS",
    "MAIN_SPLITS",
    "SourceCoverage",
    "PrefixChoice",
    "assign_exact_horizons",
    "assign_family_splits",
    "choose_coverage_holdout_sources",
    "prefix_choices",
    "stable_rank",
]
