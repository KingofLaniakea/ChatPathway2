"""Deterministic, no-network tests for Task 0-6 schemas and metrics."""

from __future__ import annotations

import unittest

import numpy as np

from downstream.new_tasks.schemas import SchemaError
from downstream.new_tasks.task0_self_consistency import evaluate_npz as evaluate_task0
from downstream.new_tasks.task1_substep_csp import (
    evaluate_pair as evaluate_task1_pair,
    evaluate_records as evaluate_task1_records,
)
from downstream.new_tasks.task2_pcte import evaluate_npz as evaluate_task2
from downstream.new_tasks.task3_causal_reranking import evaluate_cases as evaluate_task3
from downstream.new_tasks.task4_knockout_rescue import evaluate_cases as evaluate_task4
from downstream.new_tasks.task5_perturbed_cell_transfer import evaluate_npz as evaluate_task5
from downstream.new_tasks.task6_biomaze import evaluate_records as evaluate_task6


class RevisedDownstreamSmokeTests(unittest.TestCase):
    def test_task0_perfect_reconstruction_and_rollout(self) -> None:
        hidden = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
        latent = np.array([[[1.0, 0.0], [0.8, 0.2], [0.5, 0.5]]])
        summary, tables = evaluate_task0({
            "hidden_states": hidden,
            "reconstructed_states": hidden.copy(),
            "observed_latents": latent,
            "rollout_latents": latent.copy(),
            "lengths": np.array([3]),
        }, {
            "schema_version": 1,
            "dataset_id": "synthetic",
            "split": "test",
            "granularity": "graph_layer",
            "point_construction_version": "atomic-parser-v1",
            "dynamics_dt": 1.0,
            "checkpoints": {"base": "base@sha", "sft": "sft@sha", "ae": "ae@sha", "dynamics": "hnn@sha"},
        }, horizons=(1, 2))
        self.assertTrue(summary["complete_self_consistency"])
        self.assertEqual(summary["ae_reconstruction"]["mse"], 0.0)
        self.assertEqual(summary["dynamics_rollout"]["mse"], 0.0)
        self.assertEqual(len(tables["rollout_points"]), 3)

    def test_task1_structured_and_conservative_parser(self) -> None:
        target = {
            "remaining_substeps": [
                {"step": 1, "substep": 0, "source": ["AKT1"], "relation": "activates", "target": ["BAD"]},
                {"step": 1, "substep": 1, "source": ["BAD"], "relation": "inhibits", "target": ["CASP3"]},
            ],
            "predicted_phenotype": None,
        }
        prediction = {
            "remaining_steps": [
                {"step": 1, "text": "Gene AKT1 activates gene BAD. Gene BAD inhibits gene CASP3."}
            ],
            "predicted_phenotype": None,
        }
        metrics = evaluate_task1_pair(target, prediction)
        self.assertEqual(metrics["eligible"], 1)
        self.assertEqual(metrics["next_layer_event_set_exact"], 1.0)
        self.assertEqual(metrics["atomic_event_multiset_f1"], 1.0)
        self.assertIsNone(metrics["ordered_substep_f1"])
        self.assertEqual(metrics["prediction_strict_schema_validity"], 0.0)

        explicit_without_punctuation = {
            "remaining_steps": [
                {
                    "step": 1,
                    "layer": "layer 1",
                    "substeps": [
                        {"substep": 0, "text": "Gene AKT1 activates gene BAD"},
                        {"substep": 1, "text": "Gene BAD inhibits gene CASP3"},
                    ],
                }
            ],
            "predicted_phenotype": None,
        }
        exact_boundary_metrics = evaluate_task1_pair(target, explicit_without_punctuation)
        self.assertEqual(exact_boundary_metrics["eligible"], 1)
        self.assertEqual(exact_boundary_metrics["atomic_event_multiset_f1"], 1.0)

        causal = evaluate_task1_pair(target, target, ordering_mode="causal_substep_sequence")
        self.assertEqual(causal["next_substep_exact"], 1.0)
        self.assertEqual(causal["ordered_substep_f1"], 1.0)

        rows, summary = evaluate_task1_records(
            [{"sample_id": "s1", "answer": target, "predicted_answer": prediction}],
            {
                "schema_version": 1,
                "dataset_id": "synthetic",
                "split": "test",
                "model_checkpoint": "model@sha",
                "parser_version": "atomic_relation_v2",
                "ordering_mode": "layer_set",
            },
        )
        self.assertEqual(rows[0]["eligible"], 1)
        self.assertIsNone(summary["metrics"]["ordered_substep_f1"])

        ambiguous = {
            "remaining_steps": [
                {"step": 1, "text": "Gene A activates gene B and gene C inhibits gene D"}
            ],
            "predicted_phenotype": None,
        }
        rejected = evaluate_task1_pair(target, ambiguous)
        self.assertEqual(rejected["eligible"], 0)
        self.assertEqual(rejected["predicted_unparsed_clauses"], 1)

    def test_task2_identity_pcte(self) -> None:
        trajectory = np.array([[[1.0, 0.0], [0.0, 1.0]]])
        manifest = {
            "schema_version": 1,
            "dataset_id": "synthetic",
            "split": "test",
            "granularity": "graph_layer",
            "representation": {"base_checkpoint": "base@sha", "adapter_checkpoint": "sft@sha", "ae_checkpoint": "ae@sha"},
        }
        rows, summary = evaluate_task2({
            "predicted_latents": trajectory,
            "target_latents": trajectory.copy(),
            "predicted_lengths": np.array([2]),
            "target_lengths": np.array([2]),
        }, manifest)
        self.assertEqual(rows[0]["pcte"], 0.0)
        self.assertEqual(summary["metrics"]["mean_pcte"], 0.0)

    def test_task3_three_negative_types_and_calibration(self) -> None:
        diagnostics = lambda value: {"hnn_rollout_error": value}
        records = [{
            "id": "case-1",
            "question": "Continue the pathway",
            "expert_validated": True,
            "annotation_provenance": {
                "annotation_id": "synthetic-annotation-v1",
                "protocol_version": "direction-protocol-v1",
                "source_dataset_id": "synthetic-heldout",
            },
            "candidates": [
                {"id": "gold", "text": "A activates B", "label": "positive", "llm_score": -0.1, "hnn_diagnostics": diagnostics(0.1)},
                {
                    "id": "reverse", "text": "B activates A", "label": "negative",
                    "negative_type": "direction_reversal", "llm_score": -0.6,
                    "hnn_diagnostics": diagnostics(0.7),
                    "negative_provenance": {
                        "construction_method": "structured edge reversal",
                        "validation_protocol": "expert protocol v1",
                        "reversed_relation_ids": ["r1"],
                    },
                },
                {
                    "id": "shuffle", "text": "C then A then B", "label": "negative",
                    "negative_type": "step_shuffle", "llm_score": -0.7,
                    "hnn_diagnostics": diagnostics(0.8),
                    "negative_provenance": {
                        "construction_method": "graph-layer permutation",
                        "validation_protocol": "expert protocol v1",
                        "shuffle_unit": "graph_layer",
                        "original_order": ["l1", "l2", "l3"],
                        "shuffled_order": ["l3", "l1", "l2"],
                    },
                },
                {
                    "id": "unrelated", "text": "X binds Y", "label": "negative",
                    "negative_type": "unrelated_pathway", "llm_score": -1.0,
                    "hnn_diagnostics": diagnostics(1.0),
                    "negative_provenance": {
                        "construction_method": "matched hard-negative retrieval",
                        "validation_protocol": "expert protocol v1",
                        "source_pathway_id": "pathway-x",
                        "matching_protocol": "organism and length matched",
                    },
                },
            ],
        }]
        calibration = {
            "schema_version": 1,
            "calibration_id": "validation-v1",
            "fit_split": "validation",
            "features": {
                "llm_score": {"mean": -0.5, "scale": 0.5, "weight": 1.0},
                "hnn_rollout_error": {"mean": 0.5, "scale": 0.5, "weight": -0.25},
            },
        }
        candidates, rows, summary = evaluate_task3(records, calibration=calibration)
        self.assertEqual(len(candidates), 4)
        self.assertEqual(rows[0]["ranking_top1"], 1.0)
        self.assertEqual(summary["negative_type_counts"]["direction_reversal"], 1)

        invalid = [{**records[0], "candidates": [
            {**records[0]["candidates"][0], "hnn_diagnostics": {"energy_delta": -1.0}},
            records[0]["candidates"][1],
        ]}]
        with self.assertRaises(SchemaError):
            evaluate_task3(invalid)

    def test_task4_missing_phenotype_and_rescue(self) -> None:
        cases = [{
            "case_id": "unannotated",
            "phenotype_available": False,
        }, {
            "case_id": "annotated",
            "phenotype_available": True,
            "dataset_provenance": {
                "dataset_id": "perturb-db",
                "dataset_version": "v1",
                "split": "test",
                "evidence_source": "curated experiment accession",
            },
            "phenotype": {"phenotype_id": "survival", "positive_definition": "viable at assay endpoint"},
            "model_provenance": {
                "base_checkpoint": "base@sha",
                "adapter_checkpoint": "adapter@sha",
                "dynamics_checkpoint": "dynamics@sha",
                "prompt_template_version": "ko-rescue-prompt-v1",
            },
            "phenotype_scorer": {
                "scorer_id": "survival-scorer",
                "calibration_id": "validation-calibration-v1",
                "calibration_split": "validation",
                "threshold": 0.5,
            },
            "dynamics_conditioning": "prompt_initial_condition",
            "states": [
                {"state_id": "wt", "role": "wt", "interventions": [], "gold_positive": 1, "predicted_probability": 0.9},
                {"state_id": "ko-b", "role": "ko", "interventions": [{"kind": "knockout", "target": "B"}], "gold_positive": 0, "predicted_probability": 0.1},
                {"state_id": "rescue-c", "role": "rescue", "parent_ko": "ko-b", "interventions": [{"kind": "knockout", "target": "B"}, {"kind": "overexpression", "target": "C"}], "gold_positive": 1, "predicted_probability": 0.8},
                {"state_id": "nonrescue-x", "role": "rescue", "parent_ko": "ko-b", "interventions": [{"kind": "knockout", "target": "B"}, {"kind": "overexpression", "target": "X"}], "gold_positive": 0, "predicted_probability": 0.2},
            ],
        }]
        endpoints, case_rows, summary = evaluate_task4(cases)
        self.assertEqual(len(endpoints), 4)
        self.assertEqual(len(case_rows), 1)
        self.assertEqual(summary["phenotype_missing_cases_excluded"], 1)
        self.assertEqual(summary["metrics"]["rescue_hit_at_1"], 1.0)

        invalid = dict(cases[1])
        invalid["dynamics_conditioning"] = "F(t)"
        with self.assertRaises(SchemaError):
            evaluate_task4([invalid])

    def test_task5_transfer_and_controlled_baseline(self) -> None:
        control = np.zeros((2, 3), dtype=float)
        observed = np.array([[0.1, 0.5, 1.0], [1.0, 0.5, 0.1]], dtype=float)
        predicted = observed.copy()
        baseline = observed[:, ::-1].copy()
        manifest = {
            "schema_version": 1,
            "dataset_id": "cell-perturb",
            "dataset_version": "v1",
            "split": "test_unseen_perturbation",
            "representation": "normalized_expression",
            "normalization": "log1p counts with frozen train statistics",
            "control_matching": "same batch and cell type",
            "gene_ids": ["G1", "G2", "G3"],
            "cell_ids": ["C1", "C2"],
            "perturbation_ids": ["KO_A", "KO_B"],
            "model_provenance": {
                "base_checkpoint": "base@sha",
                "task_adapter_checkpoint": "cell-adapter@sha",
                "training_data_id": "cell-train-v1",
            },
            "controlled_ablation_id": "same-adapter-no-hnn-v1",
        }
        rows, groups, summary = evaluate_task5({
            "control": control,
            "observed": observed,
            "predicted": predicted,
            "baseline_predicted": baseline,
        }, manifest, top_k=2)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(groups), 2)
        self.assertEqual(summary["metrics"]["metrics"]["expression_pearson"], 1.0)
        self.assertTrue(summary["controlled_baseline"]["available"])

        invalid_manifest = {**manifest, "gene_ids": ["G1", "G2"]}
        with self.assertRaises(SchemaError):
            evaluate_task5({"control": control, "observed": observed, "predicted": predicted}, invalid_manifest)

    def test_task6_biomaze_choices(self) -> None:
        manifest = {
            "schema_version": 1,
            "dataset_id": "BioMaze",
            "dataset_version": "frozen-v1",
            "split": "test",
            "source": "official release",
            "license": "recorded in artifact",
            "evaluation_protocol": "single deterministic choice",
            "contamination_audit": {"status": "unknown", "method": "checkpoint chronology pending"},
            "model_checkpoint": "checkpoint@sha",
        }
        rows, summary = evaluate_task6([{
            "id": "q1",
            "question": "Which option follows?",
            "choices": {"A": "first", "B": "second"},
            "gold_option": "B",
            "predicted_option": "B",
            "tags": ["multihop"],
        }, {
            "id": "q2",
            "question": "Which option follows?",
            "choices": ["first", "second"],
            "gold_option": "A",
            "predicted_option": "Z",
        }], manifest)
        self.assertEqual(rows[0]["correct"], 1.0)
        self.assertEqual(rows[1]["answer_valid"], 0.0)
        self.assertEqual(summary["metrics"]["accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
