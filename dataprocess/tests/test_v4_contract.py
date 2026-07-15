from __future__ import annotations

import json
import tempfile
import unittest
import zlib
import copy
from pathlib import Path

from dataprocess.index_structured_graphs_v4 import ScanTask, main as index_main, scan_graph
from dataprocess.materialize_dataset_v4 import (
    apply_split_policy,
    assign_split_horizons,
    build_audit,
    candidate_order,
    collect_eligible_choices,
    initialize_work_database,
    open_index,
    prepare_outputs,
    output_paths,
    write_materialized_outputs,
    write_source_hashes,
)
from dataprocess.release_contract_v4 import ALL_SPLITS
from dataprocess.split_policy_v4 import (
    FamilyWeight,
    SourceCoverage,
    assign_exact_horizons,
    assign_family_splits,
    choose_coverage_holdout_sources,
    prefix_choices,
)
from dataprocess.structured_schema import csv_row, record_from_object
from dataprocess.structured_views import build_structured_records
from downstream.common.pathway_json import parse_pathway_payload
from downstream.new_tasks.task1_substep_csp import parse_substeps


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(value) for value in text]


def node(
    node_id: int,
    name: str,
    resolved_ids: list[str] | None = None,
) -> dict[str, object]:
    identifiers = resolved_ids or [f"ko:K{node_id:05d}"]
    return {
        "node_id": node_id,
        "entry_id": node_id,
        "node_kind": "entry",
        "entity_type": "ortholog",
        "canonical_id": identifiers[0],
        "display_name": name,
        "resolved_ids": identifiers,
        "raw_name": " ".join(identifiers),
        "aliases": identifiers[1:],
        "unresolved_tokens": [],
        "component_entry_ids": [],
        "resolved": True,
    }


def relation(relation_id: int, source: int, target: int) -> dict[str, object]:
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


def reaction(
    reaction_id: int,
    sources: list[int],
    targets: list[int],
) -> dict[str, object]:
    return {
        "reaction_id": reaction_id,
        "reaction_name": f"rn:R{reaction_id:05d}",
        "reaction_type": "irreversible",
        "substrate_entry_ids": sources,
        "product_entry_ids": targets,
        "renderable": True,
    }


def graph() -> dict[str, object]:
    return {
        "metadata": {
            "organism": "aaa",
            "pathway_id": "path:aaa00010",
            "title": "must remain provenance only",
        },
        "nodes": [
            node(1, "A", ["ko:K00001", "ec:1.1.1.1"]),
            node(2, "B"),
            node(3, "C"),
        ],
        "relations": [
            relation(0, 1, 2),
            relation(1, 1, 2),
            relation(2, 2, 3),
        ],
        "reactions": [],
    }


class V4EventContractTests(unittest.TestCase):
    def test_alias_is_not_a_second_participant_and_layer_duplicate_is_merged(self) -> None:
        records = build_structured_records(
            graph(),
            graph_id="graph:test",
            source_graph_json="aaa/aaa00010.json",
        )
        self.assertEqual(len(records), 1)
        first = records[0].layers[0].events[0]
        self.assertEqual(first.producer_event_ids, ("relation:0", "relation:1"))
        self.assertEqual(
            first.source,
            (
                {
                    "canonical_id": "ko:K00001",
                    "aliases": ["ec:1.1.1.1"],
                    "name": "A",
                },
            ),
        )
        self.assertEqual(first.action.model_object()["subtypes"], ["activation"])
        payload = records[0].record_object()
        self.assertEqual(record_from_object(payload).record_object(), payload)
        row = csv_row(records[0], 1)
        answer = json.loads(str(row["answer"]))
        self.assertEqual(answer["schema_version"], "pathway_continuation_v4")
        event = answer["remaining_layers"][0]["events"][0]
        self.assertEqual(
            set(event),
            {"event_type", "source", "action", "mediators", "target", "text"},
        )
        self.assertNotIn("aaa00010", row["question"])
        self.assertNotIn("must remain provenance only", row["question"])
        self.assertTrue(parse_pathway_payload(answer).schema_valid)

    def test_task1_consumes_full_action_and_mediator_contract(self) -> None:
        records = build_structured_records(
            graph(),
            graph_id="graph:test",
            source_graph_json="aaa/aaa00010.json",
        )
        answer = json.loads(str(csv_row(records[0], 1)["answer"]))
        parsed = parse_substeps(answer)
        self.assertTrue(parsed.strict_schema_valid)
        self.assertEqual(parsed.substeps[0].relation, "pprel:activation")
        self.assertEqual(parsed.substeps[0].mediators, ())

    def test_endpoint_order_survives_layer_merge_and_round_trip(self) -> None:
        payload = graph()
        payload["relations"] = []
        payload["reactions"] = [reaction(1, [2, 1], [3])]
        records = build_structured_records(
            payload,
            graph_id="graph:ordered",
            source_graph_json="aaa/aaa00010.json",
        )
        self.assertEqual(len(records), 1)
        event = records[0].layers[0].events[0]
        self.assertEqual(event.source_node_ids, (2, 1))
        self.assertEqual(
            [value["canonical_id"] for value in event.source],
            ["ko:K00002", "ko:K00001"],
        )
        rebuilt = record_from_object(records[0].record_object())
        self.assertEqual(rebuilt.record_object(), records[0].record_object())

    def test_duplicate_occurrence_entity_is_deduplicated_but_auditable(self) -> None:
        payload = graph()
        payload["relations"] = []
        duplicate = node(4, "A second occurrence", ["ko:K00001", "ec:2.2.2.2"])
        payload["nodes"].append(duplicate)
        payload["reactions"] = [reaction(1, [1, 4], [3])]
        records = build_structured_records(
            payload,
            graph_id="graph:duplicate-occurrence",
            source_graph_json="aaa/aaa00010.json",
        )
        self.assertEqual(len(records), 1)
        event = records[0].layers[0].events[0]
        self.assertEqual(event.source_node_ids, (1, 4))
        self.assertEqual(len(event.source_entity_provenance), 2)
        self.assertEqual(
            event.source,
            (
                {
                    "canonical_id": "ko:K00001",
                    "aliases": ["ec:1.1.1.1", "ec:2.2.2.2"],
                    "name": "A",
                },
            ),
        )
        self.assertIn("A is irreversibly converted", event.text)
        self.assertIn("A and A second occurrence are converted", event.legacy_text)
        rebuilt = record_from_object(records[0].record_object())
        self.assertEqual(rebuilt.record_object(), records[0].record_object())

    def test_semantic_merge_retains_producer_specific_legacy_text(self) -> None:
        payload = graph()
        duplicate_b = node(5, "B")
        duplicate_b["canonical_id"] = "ko:K00002"
        duplicate_b["resolved_ids"] = ["ko:K00002"]
        group_b = node(4, "B and B")
        group_b.update(
            {
                "entity_type": "group",
                "canonical_id": None,
                "resolved_ids": [],
                "raw_name": "group:2+5",
                "aliases": [],
                "component_entry_ids": [2, 5],
            }
        )
        payload["nodes"].extend((group_b, duplicate_b))
        payload["relations"] = [
            relation(0, 1, 2),
            relation(1, 1, 4),
            relation(2, 2, 3),
            relation(3, 4, 3),
        ]
        records = build_structured_records(
            payload,
            graph_id="graph:legacy-overrides",
            source_graph_json="aaa/aaa00010.json",
        )
        self.assertEqual(len(records), 1)
        merged = next(
            event
            for layer in records[0].layers
            for event in layer.events
            if event.producer_event_ids == ("relation:0", "relation:1")
        )
        self.assertEqual(
            merged.legacy_text,
            "A activates B.",
        )
        self.assertEqual(
            dict(merged.legacy_text_overrides),
            {"relation:1": "A activates B and B."},
        )
        rebuilt = record_from_object(records[0].record_object())
        self.assertEqual(rebuilt.record_object(), records[0].record_object())


class V4PolicyTests(unittest.TestCase):
    def test_horizon_solver_proves_exact_balance_when_feasible(self) -> None:
        eligible = {f"record:{index}": prefix_choices(5) for index in range(101)}
        assignments, report = assign_exact_horizons(eligible, seed=7)
        self.assertEqual(len(assignments), 101)
        self.assertEqual(report["theoretical_optimal_tolerance"], 0)
        self.assertLessEqual(report["max_minus_min"], 1)

    def test_family_optimizer_keeps_families_indivisible(self) -> None:
        assignment, report = assign_family_splits(
            {
                "00010": FamilyWeight(70, {"hsa": 40, "mmu": 30}),
                "00020": FamilyWeight(20, {"hsa": 10, "mmu": 10}),
                "00030": FamilyWeight(10, {"hsa": 5, "mmu": 5}),
                "00040": FamilyWeight(5, {"hsa": 5}),
            },
            seed=7,
        )
        self.assertEqual(set(assignment), {"00010", "00020", "00030", "00040"})
        self.assertEqual(set(assignment.values()), {"train", "validation", "test"})
        self.assertEqual(report["constraint"], "family_indivisible_and_disjoint")

    def test_data_internal_holdout_is_coverage_stratified_and_protects_human(self) -> None:
        coverage = {
            code: SourceCoverage(
                records=index * 10,
                graphs=index * 5,
                families=index,
                layer_total=index * 20,
                semantic_event_total=index * 30,
            )
            for index, code in enumerate(("hsa", "mmu", "rno", "eco", "sty"), 1)
        }
        heldout, report = choose_coverage_holdout_sources(
            coverage, fraction=0.5, seed=7, strata_count=2
        )
        self.assertNotIn("hsa", heldout)
        self.assertFalse(report["claims_phylogenetic_balance"])
        self.assertEqual(report["policy"], "dataset_internal_coverage_quantile_stratified_source_holdout")
        self.assertGreaterEqual(len(heldout), 2)


class V4IndexerTests(unittest.TestCase):
    def test_index_worker_emits_valid_compressed_roundtrip_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aaa00010.json"
            path.write_text(json.dumps(graph()), encoding="utf-8")
            result = scan_graph(
                ScanTask(
                    path=str(path),
                    processed_path=None,
                    relative="aaa/aaa00010.json",
                    organism="aaa",
                    family="00010",
                    seed=7,
                )
            )
        self.assertEqual(result.graph_row[6], "ok")
        self.assertEqual(len(result.record_rows), 1)
        payload = json.loads(zlib.decompress(result.record_rows[0][13]))
        self.assertEqual(record_from_object(payload).record_object(), payload)

    def test_index_worker_exactly_reconciles_historical_event_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_path = root / "aaa00010.json"
            graph_path.write_text(json.dumps(graph()), encoding="utf-8")
            base_task = ScanTask(
                path=str(graph_path),
                processed_path=None,
                relative="aaa/aaa00010.json",
                organism="aaa",
                family="00010",
                seed=7,
            )
            first = scan_graph(base_task)
            record = record_from_object(
                json.loads(zlib.decompress(first.record_rows[0][13]))
            )
            historical_text = " ".join(
                event.legacy_text or ""
                for layer in record.layers
                for event in layer.events
            )
            processed_path = root / "processed.json"
            processed_path.write_text(
                json.dumps({"pathway": [historical_text]}), encoding="utf-8"
            )
            matched = scan_graph(
                ScanTask(
                    path=str(graph_path),
                    processed_path=str(processed_path),
                    relative="aaa/aaa00010.json",
                    organism="aaa",
                    family="00010",
                    seed=7,
                )
            )
            processed_path.write_text(
                json.dumps({"pathway": ["different text"]}), encoding="utf-8"
            )
            mismatched = scan_graph(
                ScanTask(
                    path=str(graph_path),
                    processed_path=str(processed_path),
                    relative="aaa/aaa00010.json",
                    organism="aaa",
                    family="00010",
                    seed=7,
                )
            )
        self.assertEqual(matched.graph_row[17], "complete")
        self.assertEqual(matched.graph_row[6], "ok")
        self.assertEqual(matched.graph_row[20], matched.graph_row[21])
        self.assertGreater(matched.graph_row[20], 0)
        self.assertEqual(mismatched.graph_row[17], "legacy_text_mismatch")
        self.assertEqual(mismatched.graph_row[6], "quarantined")
        self.assertEqual(mismatched.record_rows, ())
        self.assertEqual(mismatched.graph_row[21], 0)

    def test_index_main_is_complete_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_dir = root / "graphs" / "aaa"
            graph_dir.mkdir(parents=True)
            (graph_dir / "aaa00010.json").write_text(
                json.dumps(graph()), encoding="utf-8"
            )
            processed_dir = root / "processed" / "aaa"
            processed_dir.mkdir(parents=True)
            historical_text = " ".join(
                event.legacy_text or ""
                for record in build_structured_records(
                    graph(),
                    graph_id="graph:test",
                    source_graph_json="aaa/aaa00010.json",
                )
                for layer in record.layers
                for event in layer.events
            )
            (processed_dir / "aaa00010.json").write_text(
                json.dumps({"pathway 0": {"layer 0": [historical_text]}}),
                encoding="utf-8",
            )
            output = root / "index"
            arguments = [
                "--processed-graph-root",
                str(root / "graphs"),
                "--processed-root",
                str(root / "processed"),
                "--output-dir",
                str(output),
                "--workers",
                "1",
                "--batch-size",
                "1",
                "--progress-every",
                "0",
            ]
            self.assertEqual(index_main(arguments), 0)
            first = json.loads((output / "index_status.json").read_text())
            self.assertTrue(first["complete"])
            self.assertEqual(first["summary"]["graphs"], 1)
            self.assertEqual(first["summary"]["records"], 1)
            self.assertEqual(index_main(arguments), 0)
            second = json.loads((output / "index_status.json").read_text())
            self.assertEqual(second["inventory"]["graph_files_already_indexed"], 1)

    def test_parallel_index_worker_results_cross_process_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_dir = root / "graphs" / "aaa"
            graph_dir.mkdir(parents=True)
            (graph_dir / "aaa00010.json").write_text(
                json.dumps(graph()), encoding="utf-8"
            )
            output = root / "index"
            self.assertEqual(
                index_main(
                    [
                        "--processed-graph-root",
                        str(root / "graphs"),
                        "--output-dir",
                        str(output),
                        "--workers",
                        "2",
                        "--batch-size",
                        "1",
                        "--progress-every",
                        "0",
                    ]
                ),
                0,
            )
            status = json.loads((output / "index_status.json").read_text())
            self.assertEqual(status["summary"]["records"], 1)


class V4MaterializationTests(unittest.TestCase):
    def test_split_token_filter_horizon_and_output_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_root = root / "graphs"
            organisms = ("aa1", "aa2", "aa3", "bb1", "bb2", "bb3")
            families = ("00010", "00020", "00030", "00040")
            for organism in organisms:
                organism_dir = graph_root / organism
                organism_dir.mkdir(parents=True)
                organism_families = list(families)
                if organism == "bb1":
                    organism_families.append("99101")
                elif organism == "bb2":
                    organism_families.append("99102")
                for family in organism_families:
                    payload = copy.deepcopy(graph())
                    payload["metadata"]["organism"] = organism
                    payload["metadata"]["pathway_id"] = f"path:{organism}{family}"
                    (organism_dir / f"{organism}{family}.json").write_text(
                        json.dumps(payload), encoding="utf-8"
                    )
            index_dir = root / "index"
            self.assertEqual(
                index_main(
                    [
                        "--processed-graph-root",
                        str(graph_root),
                        "--output-dir",
                        str(index_dir),
                        "--workers",
                        "1",
                        "--progress-every",
                        "0",
                    ]
                ),
                0,
            )
            index = open_index(index_dir / "canonical_index_v4.sqlite3")
            release = root / "release"
            paths = output_paths(release)
            prepare_outputs(paths, release, overwrite=True)
            work = initialize_work_database(
                release / ".materialization_v4.sqlite3", overwrite=True
            )
            try:
                policy = apply_split_policy(
                    index,
                    source_holdout_fraction=0.34,
                    seed=7,
                    protected_sources=("hsa", "ko", "ec"),
                )
                self.assertNotIn("", policy["canonical_split_counts"])
                self.assertNotIn(
                    "heldout_validation_family", policy["canonical_split_counts"]
                )
                self.assertEqual(
                    policy["canonical_assignment_coverage"]["unassigned_records"], 0
                )
                self.assertEqual(
                    policy["canonical_assignment_coverage"]["assigned_records"], 26
                )
                self.assertGreaterEqual(
                    policy["family_optimizer"]["heldout_source_only_families"], 1
                )
                tokenizer = CharacterTokenizer()
                collection_by_split = {}
                horizons_by_split = {}
                outputs_by_split = {}
                identities_by_split = {}
                for split in ALL_SPLITS:
                    order = candidate_order(index, split, priority_organism="hsa")
                    self.assertTrue(order)
                    stats = collect_eligible_choices(
                        index,
                        work,
                        tokenizer,
                        split=split,
                        record_ids=order,
                        max_length=20_000,
                        train_token_budget=1_000_000,
                        maximum_train_records=0,
                        maximum_evaluation_records=100,
                        progress_every=0,
                    )
                    collection_by_split[split] = stats
                    self.assertGreater(stats["selected_records"], 0)
                    balance = assign_split_horizons(work, split=split, seed=7)
                    horizons_by_split[split] = balance
                    self.assertTrue(balance["actual_matches_theoretical_optimum"])
                    public, identities = write_materialized_outputs(
                        index,
                        work,
                        tokenizer,
                        split=split,
                        output_dir=release,
                        csv_path=paths[f"{split}_csv"],
                        record_path=paths[f"{split}_records"],
                        max_length=20_000,
                    )
                    self.assertEqual(public["rows"], public["records"])
                    self.assertEqual(public["records"], len(identities["records"]))
                    outputs_by_split[split] = public
                    identities_by_split[split] = identities
                source_hashes = write_source_hashes(
                    index,
                    set().union(
                        *(value["sources"] for value in identities_by_split.values())
                    ),
                    paths["source_hashes"],
                )
                paths["manifest"].write_text("{}\n", encoding="utf-8")
                audit = build_audit(
                    index=index,
                    index_status=json.loads(
                        (index_dir / "index_status.json").read_text(encoding="utf-8")
                    ),
                    split_policy=policy,
                    collection=collection_by_split,
                    horizons=horizons_by_split,
                    outputs=outputs_by_split,
                    identities=identities_by_split,
                    paths=paths,
                    source_hashes=source_hashes,
                    processed_root=None,
                    max_length=20_000,
                    train_token_budget=1_000_000,
                    minimum_train_records=1,
                )
                self.assertEqual(audit["status"], "passed", audit["failures"])
                processed_root = root / "processed"
                selected_sources = set().union(
                    *(value["sources"] for value in identities_by_split.values())
                )
                for source in selected_sources:
                    counterpart = processed_root / source
                    counterpart.parent.mkdir(parents=True, exist_ok=True)
                    counterpart.write_text("{}\n", encoding="utf-8")
                index.execute(
                    "UPDATE graphs SET processed_text_status='complete', "
                    "processed_content_sha256=?, processed_file_size=3, "
                    "visible_legacy_text_count=3, visible_legacy_text_match_count=3",
                    ("a" * 64,),
                )
                strict_audit = build_audit(
                    index=index,
                    index_status=json.loads(
                        (index_dir / "index_status.json").read_text(encoding="utf-8")
                    ),
                    split_policy=policy,
                    collection=collection_by_split,
                    horizons=horizons_by_split,
                    outputs=outputs_by_split,
                    identities=identities_by_split,
                    paths=paths,
                    source_hashes=source_hashes,
                    processed_root=processed_root,
                    max_length=20_000,
                    train_token_budget=1_000_000,
                    minimum_train_records=1,
                )
                self.assertEqual(strict_audit["status"], "passed")
                self.assertEqual(
                    strict_audit["graph_artifact_coverage"]
                    ["processed_text_counterparts"]["unmatched_legacy_text_events"],
                    0,
                )
                index.execute(
                    "UPDATE graphs SET processed_text_status='legacy_text_mismatch' "
                    "WHERE source_graph_json=?",
                    (next(iter(selected_sources)),),
                )
                failing_audit = build_audit(
                    index=index,
                    index_status=json.loads(
                        (index_dir / "index_status.json").read_text(encoding="utf-8")
                    ),
                    split_policy=policy,
                    collection=collection_by_split,
                    horizons=horizons_by_split,
                    outputs=outputs_by_split,
                    identities=identities_by_split,
                    paths=paths,
                    source_hashes=source_hashes,
                    processed_root=processed_root,
                    max_length=20_000,
                    train_token_budget=1_000_000,
                    minimum_train_records=1,
                )
                self.assertIn(
                    "processed_counterpart_coverage_incomplete",
                    failing_audit["failures"],
                )
            finally:
                work.close()
                index.close()


if __name__ == "__main__":
    unittest.main()
