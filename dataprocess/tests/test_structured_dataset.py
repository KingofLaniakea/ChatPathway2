from __future__ import annotations

import csv
import json
import stat
import tempfile
import unittest
from pathlib import Path

from dataprocess.audit_dataset_release import file_sha256, generate_release_audit
from dataprocess.structured_schema import (
    StructuredEvent,
    StructuredRecord,
    TOPOLOGY_BACKBONE,
    TOPOLOGY_CONTEXT,
    TOPOLOGY_CONTEXT_CROSS_LAYER,
    TOPOLOGY_EXCLUDED,
    V3_CSV_FIELDNAMES,
    csv_row,
    event_from_reaction,
    event_from_relation,
    graph_events,
    graph_id_for_source,
    record_from_object,
)
from dataprocess.structured_views import build_structured_records, tarjan_scc


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(value) for value in text]


def node(node_id: int, name: str) -> dict[str, object]:
    return {
        "node_id": node_id,
        "entry_id": node_id,
        "node_kind": "entry",
        "entity_type": "ortholog",
        "canonical_id": f"ko:K{node_id:05d}",
        "display_name": name,
        "resolved_ids": [f"ko:K{node_id:05d}"],
        "raw_name": f"ko:K{node_id:05d}",
        "aliases": [],
        "unresolved_tokens": [],
        "component_entry_ids": [],
        "resolved": True,
    }


def relation(relation_id: int, source: int, target: int, *, renderable: bool) -> dict[str, object]:
    return {
        "relation_id": relation_id,
        "entry1_id": source,
        "entry2_id": target,
        "relation_type": "PPrel",
        "subtypes": [{"name": "activation", "value": "-->"}],
        "subtype_names": ["activation"],
        "semantic_tags": ["activation"],
        "mediator_entry_id": None,
        "has_missing_interaction": False,
        "renderable": renderable,
    }


def relation_with_subtypes(
    relation_id: int,
    source: int,
    target: int,
    *subtypes: tuple[str, str],
) -> dict[str, object]:
    names = [name for name, _ in subtypes]
    visible_compounds = [
        int(value) for name, value in subtypes if name == "compound"
    ]
    return {
        "relation_id": relation_id,
        "entry1_id": source,
        "entry2_id": target,
        "relation_type": "PPrel",
        "subtypes": [
            {"name": name, "value": value} for name, value in subtypes
        ],
        "subtype_names": names,
        "semantic_tags": names,
        "mediator_entry_id": visible_compounds[-1] if visible_compounds else None,
        "has_missing_interaction": "missing interaction" in names,
        "renderable": True,
    }


def reaction(
    reaction_id: int,
    substrates: list[int],
    products: list[int],
    *,
    reaction_type: str,
) -> dict[str, object]:
    return {
        "reaction_id": reaction_id,
        "reaction_name": f"rn:R{reaction_id:05d}",
        "reaction_type": reaction_type,
        "substrate_entry_ids": substrates,
        "product_entry_ids": products,
        "renderable": True,
    }


class StructuredViewTests(unittest.TestCase):
    def test_duplicate_node_and_event_identities_fail_closed(self) -> None:
        duplicate_node_graph = {
            "metadata": {"organism": "aaa", "pathway_id": "aaa00010"},
            "nodes": [node(1, "A"), node(1, "A duplicate"), node(2, "B")],
            "relations": [relation(0, 1, 2, renderable=True)],
            "reactions": [],
        }
        with self.assertRaisesRegex(ValueError, "duplicate node_id"):
            build_structured_records(
                duplicate_node_graph,
                graph_id="graph:test",
                source_graph_json="aaa/aaa00010.json",
            )

        duplicate_event_graph = {
            "metadata": {"organism": "aaa", "pathway_id": "aaa00010"},
            "nodes": [node(1, "A"), node(2, "B"), node(3, "C")],
            "relations": [
                relation(0, 1, 2, renderable=True),
                relation(0, 2, 3, renderable=True),
            ],
            "reactions": [],
        }
        with self.assertRaisesRegex(ValueError, "duplicate structural event IDs"):
            build_structured_records(
                duplicate_event_graph,
                graph_id="graph:test",
                source_graph_json="aaa/aaa00010.json",
            )

    def test_scc_builder_handles_graphs_beyond_python_recursion_depth(self) -> None:
        adjacency = {index: {index + 1} for index in range(5000)}
        adjacency[5000] = set()
        components, node_to_component = tarjan_scc(adjacency)
        self.assertEqual(len(components), 5001)
        self.assertEqual(len(node_to_component), 5001)

    def test_all_structural_events_define_scc_layers_without_text_dedup(self) -> None:
        graph = {
            "metadata": {
                "organism": "aaa",
                "pathway_id": "path:aaa00010",
                "title": "must stay metadata only",
            },
            "nodes": [node(1, "A"), node(2, "B"), node(3, "C")],
            "relations": [
                relation(0, 1, 2, renderable=True),
                relation(1, 2, 1, renderable=False),
                relation(2, 2, 3, renderable=True),
            ],
            "reactions": [],
        }
        records = build_structured_records(
            graph,
            graph_id="graph:test",
            source_graph_json="aaa/aaa00010.json",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual([len(layer.events) for layer in records[0].layers], [2, 1])
        self.assertEqual(
            [event.event_id for event in records[0].layers[0].events],
            ["relation:1", "relation:0"],
        )
        self.assertFalse(records[0].layers[0].events[0].producer_renderable)
        row = csv_row(records[0], 1)
        self.assertNotIn("must stay metadata only", row["question"])
        self.assertNotIn("aaa00010", row["question"])
        answer = json.loads(str(row["answer"]))
        self.assertEqual(set(answer), {"schema_version", "remaining_layers"})


class StrictGraphSemanticsTests(unittest.TestCase):
    def graph(
        self,
        *,
        nodes: list[dict[str, object]],
        relations: list[dict[str, object]] | None = None,
        reactions: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return {
            "metadata": {"organism": "aaa", "pathway_id": "aaa00010"},
            "nodes": nodes,
            "relations": relations or [],
            "reactions": reactions or [],
        }

    def test_reactions_use_fixed_labels_and_explicit_direction_arcs(self) -> None:
        nodes = [node(index, chr(64 + index)) for index in range(1, 5)]
        irreversible = event_from_reaction(
            reaction(0, [1, 2], [3, 4], reaction_type="irreversible"),
            {int(value["node_id"]): value for value in nodes},
        )
        self.assertEqual(irreversible.relation, "irreversible_conversion")
        self.assertEqual(
            irreversible.topology_arcs,
            ((1, 3), (1, 4), (2, 3), (2, 4)),
        )
        reversible = event_from_reaction(
            reaction(1, [1], [2], reaction_type="reversible"),
            {int(value["node_id"]): value for value in nodes},
        )
        self.assertEqual(reversible.relation, "reversible_conversion")
        self.assertEqual(reversible.topology_arcs, ((1, 2), (2, 1)))
        self.assertEqual(reversible.topology_role, TOPOLOGY_BACKBONE)

    def test_self_loop_topology_arc_survives_record_round_trip(self) -> None:
        graph = self.graph(
            nodes=[node(1, "A"), node(2, "B")],
            relations=[
                relation(0, 1, 1, renderable=True),
                relation(1, 1, 2, renderable=True),
            ],
        )
        records = build_structured_records(
            graph,
            graph_id="graph:self-loop",
            source_graph_json="aaa/aaa00010.json",
        )
        self.assertEqual(len(records), 1)
        events = {
            event.event_id: event
            for layer in records[0].layers
            for event in layer.events
        }
        self.assertEqual(events["relation:0"].topology_arcs, ((1, 1),))
        payload = records[0].record_object()
        rebuilt = record_from_object(payload)
        self.assertEqual(rebuilt.record_object(), payload)

        self_loop = next(
            event
            for layer in payload["layers"]
            for event in layer["events"]
            if event["event_id"] == "relation:0"
        )
        self_loop["topology_arcs"].append([1, 1])
        with self.assertRaisesRegex(ValueError, "topology_arcs contain duplicates"):
            record_from_object(payload)

    def test_relation_direction_context_and_missing_are_separate(self) -> None:
        nodes = [node(index, chr(64 + index)) for index in range(1, 4)]
        lookup = {int(value["node_id"]): value for value in nodes}
        directional = event_from_relation(
            relation_with_subtypes(0, 1, 2, ("activation", "-->")),
            lookup,
        )
        self.assertEqual(directional.topology_role, TOPOLOGY_BACKBONE)
        self.assertEqual(directional.topology_arcs, ((1, 2),))

        context = event_from_relation(
            relation_with_subtypes(1, 1, 2, ("binding/association", "---")),
            lookup,
        )
        self.assertEqual(context.topology_role, TOPOLOGY_CONTEXT)
        self.assertEqual(context.topology_arcs, ())
        self.assertTrue(context.core_included)

        missing = event_from_relation(
            relation_with_subtypes(2, 1, 2, ("missing interaction", "--")),
            lookup,
        )
        self.assertEqual(missing.topology_role, TOPOLOGY_EXCLUDED)
        self.assertFalse(missing.core_included)
        self.assertEqual(missing.exclusion_reason, "missing_interaction")

    def test_compound_mediator_is_provenance_not_source(self) -> None:
        nodes = [node(index, chr(64 + index)) for index in range(1, 4)]
        event = event_from_relation(
            relation_with_subtypes(
                0,
                1,
                2,
                ("activation", "-->"),
                ("compound", "3"),
            ),
            {int(value["node_id"]): value for value in nodes},
        )
        self.assertEqual(event.source_node_ids, (1,))
        self.assertEqual(event.mediator_node_ids, (3,))
        self.assertEqual([value["canonical_id"] for value in event.source], ["ko:K00001"])
        self.assertEqual([value["canonical_id"] for value in event.mediator], ["ko:K00003"])
        self.assertNotIn("ko:K00003", [value["canonical_id"] for value in event.source])
        self.assertEqual(set(event.model_object()), {"source", "relation", "target", "text"})

        hidden = event_from_relation(
            relation_with_subtypes(1, 1, 2, ("hidden compound", "3")),
            {int(value["node_id"]): value for value in nodes},
        )
        self.assertEqual(hidden.mediator_node_ids, (3,))
        self.assertEqual(hidden.topology_role, TOPOLOGY_CONTEXT)

    def test_unknown_relation_extension_is_explicitly_excluded(self) -> None:
        nodes = [node(1, "A"), node(2, "B")]
        extension = event_from_relation(
            relation_with_subtypes(0, 1, 2, ("vendor extension", "?")),
            {int(value["node_id"]): value for value in nodes},
        )
        self.assertEqual(extension.topology_role, TOPOLOGY_EXCLUDED)
        self.assertFalse(extension.core_included)
        self.assertEqual(extension.exclusion_reason, "unknown_subtypes:vendor extension")

    def test_invalid_subtype_or_identity_rejects_the_whole_graph(self) -> None:
        relations = [
            relation(0, 1, 2, renderable=True),
            relation(1, 2, 3, renderable=True),
        ]
        relations[1]["semantic_tags"] = ["inhibition"]
        graph = self.graph(
            nodes=[node(1, "A"), node(2, "B"), node(3, "C")],
            relations=relations,
        )
        valid_events, rejected = graph_events(graph)
        self.assertEqual(len(valid_events), 1)
        self.assertEqual(rejected, 1)
        self.assertEqual(
            build_structured_records(
                graph,
                graph_id="graph:test",
                source_graph_json="aaa/aaa00010.json",
            ),
            (),
        )

        bad_nodes = [node(1, "A"), node(2, "B")]
        bad_nodes[1]["canonical_id"] = "ko:NOT_RESOLVED"
        bad_graph = self.graph(
            nodes=bad_nodes,
            relations=[relation(0, 1, 2, renderable=True)],
        )
        self.assertEqual(
            build_structured_records(
                bad_graph,
                graph_id="graph:test",
                source_graph_json="aaa/aaa00010.json",
            ),
            (),
        )

    def test_group_and_multi_id_endpoints_are_lossless_and_round_trip(self) -> None:
        first = node(1, "A")
        first["resolved_ids"] = ["ko:K00001", "ko:K10001"]
        group = node(3, "A and B")
        group.update(
            {
                "entity_type": "group",
                "canonical_id": None,
                "resolved_ids": [],
                "raw_name": "group:1+2",
                "component_entry_ids": [1, 2],
            }
        )
        graph = self.graph(
            nodes=[first, node(2, "B"), group, node(4, "C")],
            relations=[relation(0, 3, 4, renderable=True)],
        )
        records = build_structured_records(
            graph,
            graph_id="graph:test",
            source_graph_json="aaa/aaa00010.json",
        )
        self.assertEqual(len(records), 1)
        event = records[0].layers[0].events[0]
        self.assertEqual(
            [value["canonical_id"] for value in event.source],
            ["ko:K00001", "ko:K10001", "ko:K00002"],
        )
        self.assertEqual(event.source_entity_provenance[0]["component_entry_ids"], [1, 2])
        rebuilt = record_from_object(records[0].record_object())
        self.assertEqual(rebuilt.record_object(), records[0].record_object())

    def test_longest_distance_orders_branches_and_context_never_expands_view(self) -> None:
        graph = self.graph(
            nodes=[node(index, chr(64 + index)) for index in range(1, 6)],
            relations=[
                relation(0, 4, 1, renderable=True),
                relation(1, 1, 2, renderable=True),
                relation(2, 1, 3, renderable=True),
                relation(3, 2, 3, renderable=True),
                relation_with_subtypes(4, 1, 3, ("binding/association", "---")),
                relation_with_subtypes(5, 3, 5, ("binding/association", "---")),
            ],
        )
        records = build_structured_records(
            graph,
            graph_id="graph:test",
            source_graph_json="aaa/aaa00010.json",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(
            [layer.distance_to_sink for layer in records[0].layers],
            [2, 1, 0],
        )
        by_id = {
            event.event_id: event
            for layer in records[0].layers
            for event in layer.events
        }
        self.assertEqual(by_id["relation:4"].topology_role, TOPOLOGY_CONTEXT_CROSS_LAYER)
        self.assertNotIn("relation:5", by_id)
        excluded = {event.event_id: event for event in records[0].excluded_events}
        self.assertEqual(excluded["relation:5"].exclusion_reason, "context_not_attached_to_view")
        self.assertEqual(records[0].sink_node_ids, (3,))


class ReleaseAuditTests(unittest.TestCase):
    def event(self, suffix: str) -> StructuredEvent:
        return StructuredEvent(
            event_id=f"relation:{suffix}",
            event_type="relation",
            source_node_ids=(1,),
            target_node_ids=(2,),
            source=({"canonical_id": f"ko:S{suffix}", "name": f"S{suffix}"},),
            relation="activation",
            target=({"canonical_id": f"ko:T{suffix}", "name": f"T{suffix}"},),
            text=f"S{suffix} activates T{suffix}.",
            producer_renderable=True,
        )

    def record(
        self,
        *,
        organism: str,
        pathway_id: str,
        source: str,
        suffix: str,
    ) -> tuple[StructuredRecord, bytes]:
        graph = {
            "metadata": {
                "organism": organism,
                "pathway_id": pathway_id,
                "title": "metadata title",
            },
            "nodes": [node(index, f"{suffix}{index}") for index in range(1, 5)],
            "relations": [
                relation(index, index + 1, index + 2, renderable=True)
                for index in range(3)
            ],
            "reactions": [],
        }
        raw = (
            json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        records = build_structured_records(
            graph,
            graph_id=graph_id_for_source(source, raw),
            source_graph_json=source,
        )
        self.assertEqual(len(records), 1)
        return records[0], raw

    def write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=V3_CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def test_generated_audit_contains_all_required_sections_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_root = root / "processed_graph"
            record_artifacts = {
                "train": self.record(
                    organism="aaa", pathway_id="aaa00010", source="aaa/aaa00010.json", suffix="tr"
                ),
                "validation": self.record(
                    organism="aaa", pathway_id="aaa00020", source="aaa/aaa00020.json", suffix="va"
                ),
                "test": self.record(
                    organism="bbb", pathway_id="bbb00030", source="bbb/bbb00030.json", suffix="te"
                ),
            }
            records = {split: value[0] for split, value in record_artifacts.items()}
            for split, record in records.items():
                artifact = graph_root / record.source_graph_json
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_bytes(record_artifacts[split][1])

            csv_paths = {split: root / f"{split}.csv" for split in records}
            self.write_csv(
                csv_paths["train"],
                [csv_row(records["train"], 1), csv_row(records["train"], 2)],
            )
            self.write_csv(csv_paths["validation"], [csv_row(records["validation"], 1)])
            self.write_csv(csv_paths["test"], [csv_row(records["test"], 1)])
            record_paths = {
                split: root / f"{split}_pathway_records_v3.jsonl"
                for split in records
            }
            for split, record in records.items():
                record_paths[split].write_text(
                    json.dumps(record.record_object(), sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            manifest = {
                "dataset_build_id": "dataset:0123456789abcdef01234567",
                "inventory": {"graph_files": 3},
                "outputs": {
                    f"{split}_records": path.name
                    for split, path in record_paths.items()
                },
                "splits": {
                    split: {
                        "csv_sha256": file_sha256(path),
                        "records_sha256": file_sha256(record_paths[split]),
                        "rows_dropped_token_budget": 0,
                    }
                    for split, path in csv_paths.items()
                },
            }
            manifest_path = root / "dataset_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            output_path = root / "data_audit.json"
            report = generate_release_audit(
                train_path=csv_paths["train"],
                validation_path=csv_paths["validation"],
                test_path=csv_paths["test"],
                graph_root=graph_root,
                manifest_path=manifest_path,
                tokenizer=CharacterTokenizer(),
                max_length=100000,
                output_path=output_path,
                overwrite=False,
            )
            self.assertEqual(report["status"], "passed")
            required = report["required_summary"]
            self.assertEqual(
                set(required),
                {
                    "train_test_row_counts",
                    "record_counts",
                    "source_json_counts",
                    "family_counts",
                    "strict_overlap",
                    "organism_overlap",
                    "duplicate_record_sample_ids",
                    "phenotype_status",
                    "parser_source",
                    "substep_coverage",
                    "layer_length_distribution",
                    "truncation_estimate",
                    "graph_artifact_coverage",
                },
            )
            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode), 0o444)


if __name__ == "__main__":
    unittest.main()
