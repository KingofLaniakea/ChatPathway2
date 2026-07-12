from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from dataprocess.audit_pathway_csv import audit_file
from dataprocess.build_pathway_csv import extract_graph_fields, main as build_main
from dataprocess.prepare_experiment_data import (
    prepare_test,
    prepare_train,
    select_holdout_families,
)
from dataprocess.schemas import canonical_pathway_family_id


class DatasetPipelineTests(unittest.TestCase):
    def test_cross_organism_pathway_family_is_canonical(self) -> None:
        self.assertEqual(canonical_pathway_family_id("hsa04010"), "04010")
        self.assertEqual(canonical_pathway_family_id("mmu04010"), "04010")
        heldout = select_holdout_families(
            {"00010", "00020", "04010", "05200"},
            fraction=0.25,
            seed=20260711,
        )
        self.assertEqual(len(heldout), 1)

    def test_file_level_phenotype_is_not_copied_across_blocks(self) -> None:
        graph = {
            "phenotype": "file-wide label",
            "pathway 1": {"phenotype": "block one label"},
            "pathway 2": {},
        }
        _, _, _, block_one = extract_graph_fields(
            graph,
            block_name="pathway 1",
            allow_file_level_phenotype=False,
        )
        _, _, _, block_two = extract_graph_fields(
            graph,
            block_name="pathway 2",
            allow_file_level_phenotype=False,
        )
        self.assertEqual(block_one.text, "block one label")
        self.assertEqual(block_one.status, "available")
        self.assertIsNone(block_two.text)
        self.assertEqual(block_two.status, "ambiguous_file_level")

    def test_single_block_can_use_file_level_phenotype(self) -> None:
        graph = {"phenotype": "single-record phenotype", "pathway 0": {}}
        _, _, _, phenotype = extract_graph_fields(
            graph,
            block_name="pathway 0",
            allow_file_level_phenotype=True,
        )
        self.assertEqual(phenotype.text, "single-record phenotype")
        self.assertEqual(phenotype.status, "available")

    def test_build_prepare_and_audit_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed"
            graphs = root / "processed_graph"
            data = root / "data"
            for organism in ("hsa", "xtr"):
                (processed / organism).mkdir(parents=True)
                (graphs / organism).mkdir(parents=True)
            data.mkdir()

            (processed / "hsa" / "multi.json").write_text(
                json.dumps(
                    {
                        "pathway 1": {
                            "layer 0": ["A activates B.", "C inhibits D."],
                            "layer 1": ["B activates E."],
                            "layer 2": ["E activates F."],
                        },
                        "pathway 2": {
                            "layer 0": ["G binds H."],
                            "layer 1": ["H activates I."],
                            "layer 2": ["I inhibits J."],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (graphs / "hsa" / "multi.json").write_text(
                json.dumps(
                    {
                        "metadata": {"organism": "hsa", "pathway_id": "hsa00001"},
                        "phenotype": "ambiguous file label",
                        "pathway 1": {"phenotype": "block-specific disease"},
                        "pathway 2": {},
                    }
                ),
                encoding="utf-8",
            )
            (processed / "xtr" / "single.json").write_text(
                json.dumps(
                    {
                        "layer 0": ["K activates L."],
                        "layer 1": ["L activates M."],
                        "layer 2": ["M produces N."],
                    }
                ),
                encoding="utf-8",
            )
            (graphs / "xtr" / "single.json").write_text(
                json.dumps(
                    {
                        "metadata": {"organism": "xtr", "pathway_id": "xtr00001"},
                        "phenotype": "single-record phenotype",
                    }
                ),
                encoding="utf-8",
            )

            full_train = data / "train.csv"
            full_test = data / "test.csv"
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = build_main(
                    [
                        "--processed-root",
                        str(processed),
                        "--processed-graph-root",
                        str(graphs),
                        "--train-output",
                        str(full_train),
                        "--test-output",
                        str(full_test),
                        "--test-organisms",
                        "xtr",
                        "--progress-every",
                        "0",
                    ]
                )
            self.assertEqual(exit_code, 0)

            with full_train.open(newline="", encoding="utf-8") as handle:
                train_rows = list(csv.DictReader(handle))
            self.assertEqual(len(train_rows), 4)
            self.assertEqual(
                {row["pathway_family_id"] for row in train_rows},
                {"00001"},
            )
            statuses_by_block = {
                block: {row["phenotype_status"] for row in train_rows if row["pathway_block"] == block}
                for block in {row["pathway_block"] for row in train_rows}
            }
            self.assertEqual(statuses_by_block["pathway 1"], {"available"})
            self.assertEqual(statuses_by_block["pathway 2"], {"ambiguous_file_level"})
            first_answer = json.loads(train_rows[0]["answer"])
            self.assertEqual(
                [item["text"] for item in first_answer["remaining_steps"][0]["substeps"]],
                ["B activates E."],
            )

            sampled_train = data / "record_balanced_train.csv"
            core_eval = data / "core_eval.csv"
            multistep_eval = data / "multistep_eval.csv"
            train_stats = prepare_train(
                full_train,
                sampled_train,
                record_fraction=1.0,
                max_prefixes_per_record=2,
                seed=20260711,
                phenotype_record_fraction=1.0,
            )
            test_stats = prepare_test(full_test, core_eval, max_prefixes_per_record=1)
            prepare_test(full_test, multistep_eval, max_prefixes_per_record=2)
            self.assertEqual(train_stats["input_records"], 2)
            self.assertEqual(test_stats["input_records"], 1)

            train_audit = audit_file(sampled_train, max_errors=20)
            test_audit = audit_file(core_eval, max_errors=20)
            multistep_audit = audit_file(multistep_eval, max_errors=20)
            self.assertEqual(train_audit.errors, [])
            self.assertEqual(test_audit.errors, [])
            self.assertEqual(multistep_audit.errors, [])
            self.assertFalse(train_audit.sources & test_audit.sources)
            self.assertEqual(len(test_audit.records), 1)
            self.assertEqual(test_audit.rows, 1)
            self.assertEqual(multistep_audit.rows, 2)
            self.assertEqual(train_audit.pathway_families, {"00001"})
            self.assertEqual(test_audit.pathway_families, {"00001"})

            excluded_train = data / "excluded_record_balanced_train.csv"
            excluded_eval = data / "excluded_eval.csv"
            with self.assertRaises(ValueError):
                prepare_train(
                    full_train,
                    excluded_train,
                    record_fraction=1.0,
                    max_prefixes_per_record=2,
                    seed=20260711,
                    phenotype_record_fraction=1.0,
                    excluded_pathway_families={"00001"},
                )
            include_stats = prepare_test(
                full_test,
                excluded_eval,
                max_prefixes_per_record=1,
                included_pathway_families={"00001"},
            )
            self.assertEqual(include_stats["selected_records"], 1)


if __name__ == "__main__":
    unittest.main()
