from __future__ import annotations

import csv
import json
import stat
import tempfile
import unittest
from pathlib import Path

from dataprocess.audit_dataset_release import file_sha256, generate_release_audit
from dataprocess.schemas import CSV_FIELDNAMES
from dataprocess.structured_schema import (
    StructuredEvent,
    StructuredRecord,
    csv_row,
    graph_id_for_source,
)
from dataprocess.structured_views import build_structured_records, tarjan_scc


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(value) for value in text]


def node(node_id: int, name: str) -> dict[str, object]:
    return {
        "node_id": node_id,
        "canonical_id": f"ko:K{node_id:05d}",
        "display_name": name,
        "resolved_ids": [f"ko:K{node_id:05d}"],
        "raw_name": f"ko:K{node_id:05d}",
    }


def relation(relation_id: int, source: int, target: int, *, renderable: bool) -> dict[str, object]:
    return {
        "relation_id": relation_id,
        "entry1_id": source,
        "entry2_id": target,
        "relation_type": "PPrel",
        "subtype_names": ["activation"],
        "semantic_tags": ["activation"],
        "mediator_entry_id": None,
        "has_missing_interaction": False,
        "renderable": renderable,
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
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDNAMES)
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
