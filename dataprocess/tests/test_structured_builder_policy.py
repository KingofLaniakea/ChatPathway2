from __future__ import annotations

import csv
import contextlib
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from dataprocess import build_structured_dataset as builder


class FakeRecord:
    def __init__(
        self,
        record_id: str,
        *,
        family: str = "00010",
        organism: str = "seen",
        graph_id: str | None = None,
        layer_count: int = 4,
    ) -> None:
        self.record_id = record_id
        self.family = family
        self.organism = organism
        self.graph_id = graph_id or f"graph:{record_id}"
        self.source_graph_json = f"{organism}/{record_id}.json"
        self.layers = tuple(range(layer_count))

    def record_object(self) -> dict[str, str]:
        return {"id": self.record_id}


def candidate(record: FakeRecord, rank: str = "0") -> dict[str, object]:
    return {
        "record_id": record.record_id,
        "organism": record.organism,
        "graph_id": record.graph_id,
        "source_graph_json": record.source_graph_json,
        "layer_count": len(record.layers),
        "rank": rank,
        "record_json": json.dumps(record.record_object()),
    }


def graph_node(node_id: int) -> dict[str, object]:
    return {
        "node_id": node_id,
        "entry_id": node_id,
        "node_kind": "entry",
        "entity_type": "ortholog",
        "canonical_id": f"ko:K{node_id:05d}",
        "display_name": f"node-{node_id}",
        "resolved_ids": [f"ko:K{node_id:05d}"],
        "raw_name": f"ko:K{node_id:05d}",
        "aliases": [],
        "unresolved_tokens": [],
        "component_entry_ids": [],
        "resolved": True,
    }


def graph_relation(relation_id: int, source: int, target: int) -> dict[str, object]:
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
        "renderable": True,
    }


def write_graph(path: Path) -> None:
    pathway_id = path.stem
    organism = pathway_id[:-5]
    graph = {
        "metadata": {"organism": organism, "pathway_id": pathway_id},
        "nodes": [graph_node(index) for index in range(1, 4)],
        "relations": [
            graph_relation(0, 1, 2),
            graph_relation(1, 2, 3),
        ],
        "reactions": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(graph, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


class FiveWaySplitPolicyTests(unittest.TestCase):
    def test_truth_table(self) -> None:
        kwargs = {
            "test_organisms": {"held"},
            "test_families": {"00020"},
            "validation_families": {"00030"},
            "train_families": {"00010"},
        }
        expected = {
            ("seen", "00010"): "train",
            ("seen", "00030"): "validation",
            ("held", "00020"): "test",
            ("seen", "00020"): "test_family_only",
            ("held", "00010"): "test_organism_only",
            ("held", "00030"): None,
            ("held", "99999"): None,
            ("seen", "99999"): None,
        }
        for (organism, family), split in expected.items():
            self.assertEqual(
                builder.assigned_split(organism, family, **kwargs),
                split,
            )

    def test_strict_families_come_from_seen_holdout_intersection(self) -> None:
        strict, validation, train = builder.choose_family_splits(
            test_available_families={"00010", "00020", "90000"},
            non_test_available_families={
                "00010",
                "00020",
                "00030",
                "00040",
                "00050",
            },
            test_fraction=0.5,
            validation_fraction=0.25,
            seed=17,
        )
        self.assertTrue(strict)
        self.assertLessEqual(strict, {"00010", "00020"})
        self.assertFalse(strict & validation)
        self.assertFalse(strict & train)
        self.assertFalse(validation & train)
        self.assertEqual(
            strict | validation | train,
            {"00010", "00020", "00030", "00040", "00050"},
        )


class HorizonPolicyTests(unittest.TestCase):
    def test_unique_horizons_for_short_and_long_records(self) -> None:
        self.assertEqual(
            builder.prefix_horizons(2),
            (builder.PrefixHorizon(1, "degenerate_target"),),
        )
        self.assertEqual(
            builder.prefix_horizons(3),
            (
                builder.PrefixHorizon(1, "long_target"),
                builder.PrefixHorizon(2, "short_target"),
            ),
        )
        self.assertEqual(
            builder.prefix_horizons(8),
            (
                builder.PrefixHorizon(1, "long_target"),
                builder.PrefixHorizon(4, "middle_target"),
                builder.PrefixHorizon(7, "short_target"),
            ),
        )
        with self.assertRaisesRegex(ValueError, r"\[1, 3\]"):
            builder.prefix_horizons(8, maximum=0)

    def test_validation_assignment_is_deterministic_and_globally_balanced(self) -> None:
        choices = builder.prefix_horizons(8)
        eligible = {f"record-{index:02d}": choices for index in range(30)}
        forward = builder.assign_balanced_validation_horizons(eligible, seed=23)
        reverse = builder.assign_balanced_validation_horizons(
            dict(reversed(tuple(eligible.items()))),
            seed=23,
        )
        self.assertEqual(forward, reverse)
        counts = Counter(choice.horizon for choice in forward.values())
        self.assertEqual(
            counts,
            {"long_target": 10, "middle_target": 10, "short_target": 10},
        )


class GraphSelectionTests(unittest.TestCase):
    @staticmethod
    def row(graph: str, record: str, rank: str) -> dict[str, object]:
        return {
            "graph_id": graph,
            "record_id": record,
            "organism": "seen",
            "layer_count": 4,
            "rank": rank,
        }

    def test_every_graph_contributes_first_view_before_any_second_view(self) -> None:
        ordered = builder.graph_round_robin_order(
            (
                self.row("g1", "g1-v1", "01"),
                self.row("g1", "g1-v2", "02"),
                self.row("g2", "g2-v1", "03"),
                self.row("g3", "g3-v1", "04"),
                self.row("g3", "g3-v2", "05"),
            )
        )
        self.assertEqual({row["graph_id"] for row in ordered[:3]}, {"g1", "g2", "g3"})
        self.assertEqual(len(ordered), 5)

    def test_every_organism_contributes_before_any_second_record(self) -> None:
        rows = (
            self.row("g-a1", "a-1", "01") | {"organism": "a"},
            self.row("g-a2", "a-2", "02") | {"organism": "a"},
            self.row("g-b1", "b-1", "03") | {"organism": "b"},
            self.row("g-c1", "c-1", "04") | {"organism": "c"},
            self.row("g-c2", "c-2", "05") | {"organism": "c"},
        )
        ordered = builder.organism_round_robin_order(rows)
        self.assertEqual(
            {str(row["organism"]) for row in ordered[:3]},
            {"a", "b", "c"},
        )

    def test_candidate_fraction_is_evaluated_once_per_graph(self) -> None:
        records = (
            FakeRecord("view-1", graph_id="graph:test"),
            FakeRecord("view-2", graph_id="graph:test"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "seen" / "seen00010.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}", encoding="utf-8")
            database = builder.initialize_database(root / "candidates.sqlite3")
            try:
                with (
                    mock.patch.object(builder, "graph_id_for_source", return_value="graph:test"),
                    mock.patch.object(builder, "graph_events", return_value=((object(),), 0)),
                    mock.patch.object(builder, "build_structured_records", return_value=records),
                    mock.patch.object(
                        builder,
                        "record_from_object",
                        side_effect=lambda value: next(
                            record for record in records if record.record_id == value["id"]
                        ),
                    ),
                    mock.patch.object(builder, "stable_fraction", return_value=0.0) as fraction,
                ):
                    stats = builder.scan_candidates(
                        root,
                        database,
                        test_organisms={"held"},
                        test_families={"00020"},
                        validation_families={"00030"},
                        train_families={"00010"},
                        train_candidate_fraction=0.5,
                        evaluation_candidate_fraction=1.0,
                        seed=29,
                        max_files=0,
                        progress_every=0,
                        coverage_graphs_per_train_organism=0,
                    )
                self.assertEqual(fraction.call_count, 1)
                self.assertEqual(stats["candidate_records"], 2)
                graph_ids = {
                    row[0] for row in database.execute("SELECT graph_id FROM candidates")
                }
                self.assertEqual(graph_ids, {"graph:test"})
            finally:
                database.close()

    def test_candidate_database_index_satisfies_family_rank_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = builder.initialize_database(
                Path(directory) / "candidates.sqlite3"
            )
            try:
                plan = "\n".join(
                    str(row[-1])
                    for row in database.execute(
                        "EXPLAIN QUERY PLAN "
                        "SELECT record_id, organism, graph_id, source_graph_json, "
                        "layer_count, rank, record_json FROM candidates "
                        "WHERE split=? AND family=? ORDER BY rank",
                        ("train", "00010"),
                    )
                )
                self.assertIn("candidates_split_family_rank", plan)
                self.assertNotIn("USE TEMP B-TREE", plan)
            finally:
                database.close()


class ParallelGraphScanTests(unittest.TestCase):
    def scan_snapshot(
        self,
        root: Path,
        *,
        workers: int,
        max_files: int = 0,
    ) -> tuple[dict[str, int], list[tuple[object, ...]], dict[str, list[str]]]:
        database_path = root / f"candidates-w{workers}-m{max_files}.sqlite3"
        database = builder.initialize_database(database_path)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                stats = builder.scan_candidates(
                    root / "graphs",
                    database,
                    test_organisms={"bbb"},
                    test_families={"00030"},
                    validation_families={"00020"},
                    train_families={"00010", "00011", "00012"},
                    train_candidate_fraction=1.0,
                    evaluation_candidate_fraction=1.0,
                    seed=41,
                    max_files=max_files,
                    progress_every=1,
                    workers=workers,
                    worker_batch_size=2,
                )
            database_rows = list(
                database.execute(
                    "SELECT record_id, split, family, organism, graph_id, "
                    "source_graph_json, layer_count, rank, record_json "
                    "FROM candidates ORDER BY record_id"
                )
            )
            selected = {
                split: [
                    str(row["record_id"])
                    for row in builder.select_records(database, split)
                ]
                for split in builder.SPLITS
            }
            return stats, database_rows, selected
        finally:
            database.close()

    def test_serial_and_parallel_scans_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_graph(root / "graphs" / "aaa" / "aaa00010.json")
            write_graph(root / "graphs" / "aaa" / "aaa00011.json")
            serial = self.scan_snapshot(root, workers=1)
            parallel = self.scan_snapshot(root, workers=2)
            self.assertEqual(serial, parallel)
            self.assertEqual(serial[0]["graph_files_scanned"], 2)
            self.assertEqual(serial[0]["candidate_records"], 2)

    def test_max_files_and_invalid_graphs_are_worker_invariant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_graph(root / "graphs" / "aaa" / "aaa00010.json")
            invalid = root / "graphs" / "aaa" / "aaa00011.json"
            invalid.write_text("{not-json", encoding="utf-8")
            write_graph(root / "graphs" / "aaa" / "aaa00012.json")
            serial = self.scan_snapshot(root, workers=1, max_files=2)
            parallel = self.scan_snapshot(root, workers=2, max_files=2)
            self.assertEqual(serial, parallel)
            stats, rows, _selected = serial
            self.assertEqual(stats["graph_files_scanned"], 2)
            self.assertEqual(stats["invalid_graph_files"], 1)
            self.assertEqual(stats["candidate_records"], 1)
            self.assertEqual(len(rows), 1)


class MaterializationPolicyTests(unittest.TestCase):
    def row_for(
        self,
        record: FakeRecord,
        prefix: builder.PrefixHorizon,
        **_kwargs,
    ):
        row = {field: "" for field in builder.V3_CSV_FIELDNAMES}
        row.update(
            {
                "sample_id": f"{record.record_id}:prefix={prefix.prefix_len}",
                "record_id": record.record_id,
                "question": "Question",
                "answer": "{}",
            }
        )
        return row, True, True

    def test_token_rejection_is_backfilled_before_record_cap(self) -> None:
        records = {
            item.record_id: item
            for item in (
                FakeRecord("too-long"),
                FakeRecord("accepted"),
                FakeRecord("after-cap"),
            )
        }
        selected = [candidate(record, rank=str(index)) for index, record in enumerate(records.values())]

        def token_count(_tokenizer, row):
            return 11 if str(row["sample_id"]).startswith("too-long:") else 5

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                builder,
                "record_from_object",
                side_effect=lambda value: records[value["id"]],
            ), mock.patch.object(
                builder,
                "_csv_row_with_policy",
                side_effect=self.row_for,
            ), mock.patch.object(
                builder,
                "total_training_tokens",
                side_effect=token_count,
            ):
                root = Path(directory)
                result = builder.write_selected_split(
                    selected,
                    split="train",
                    csv_path=root / "train.csv",
                    record_path=root / "train.jsonl",
                    tokenizer=object(),
                    max_length=10,
                    max_prefixes_per_train_record=3,
                    max_records_per_family=1,
                    seed=31,
                )
                self.assertEqual(result["records"], 1)
                self.assertEqual(result["maximum_records_in_one_family"], 1)
                self.assertEqual(result["records_dropped_no_complete_json_sample"], 1)
                self.assertEqual(result["candidate_records_skipped_family_cap"], 1)
                payload = json.loads(
                    (root / "train.jsonl").read_text(encoding="utf-8")
                )
                self.assertEqual(payload, {"id": "accepted"})

    def test_all_evaluation_horizons_are_materialized(self) -> None:
        record = FakeRecord("evaluation", layer_count=8)
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                builder,
                "record_from_object",
                return_value=record,
            ), mock.patch.object(
                builder,
                "_csv_row_with_policy",
                side_effect=self.row_for,
            ), mock.patch.object(
                builder,
                "total_training_tokens",
                return_value=5,
            ):
                root = Path(directory)
                result = builder.write_selected_split(
                    [candidate(record)],
                    split="test",
                    csv_path=root / "test.csv",
                    record_path=root / "test.jsonl",
                    tokenizer=object(),
                    max_length=10,
                    max_prefixes_per_train_record=3,
                    max_records_per_family=256,
                    seed=37,
                )
                self.assertEqual(result["rows"], 3)
                self.assertEqual(
                    result["prefix_horizons"],
                    {"long_target": 1, "middle_target": 1, "short_target": 1},
                )
                with (root / "test.csv").open(
                    encoding="utf-8",
                    newline="",
                ) as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_train_epoch_token_budget_keeps_whole_records(self) -> None:
        records = {
            record.record_id: record
            for record in (
                FakeRecord("first", organism="a"),
                FakeRecord("second", organism="b"),
                FakeRecord("third", organism="c"),
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                builder,
                "record_from_object",
                side_effect=lambda value: records[value["id"]],
            ), mock.patch.object(
                builder,
                "_csv_row_with_policy",
                side_effect=self.row_for,
            ), mock.patch.object(
                builder,
                "total_training_tokens",
                return_value=5,
            ):
                root = Path(directory)
                result = builder.write_selected_split(
                    [candidate(record) for record in records.values()],
                    split="train",
                    csv_path=root / "train.csv",
                    record_path=root / "train.jsonl",
                    tokenizer=object(),
                    max_length=10,
                    max_prefixes_per_train_record=3,
                    max_records_per_family=256,
                    seed=41,
                    maximum_records=10,
                    target_input_tokens_per_epoch=11,
                )
                self.assertEqual(result["records"], 2)
                self.assertEqual(result["estimated_input_tokens_per_epoch"], 10)
                self.assertEqual(
                    result["candidate_records_skipped_epoch_token_budget"],
                    1,
                )


if __name__ == "__main__":
    unittest.main()
