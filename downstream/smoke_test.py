#!/usr/bin/env python3
"""Deterministic, no-network smoke tests for downstream task metrics."""

from __future__ import annotations

import random

import numpy as np

from downstream.task1_2 import evaluate as evaluate_pcer
from downstream.task3_pcte import evaluate as evaluate_pcte
from downstream.task4_csp import evaluate_pair, parse_steps
from downstream.task5_cki import evaluate as evaluate_cki
from downstream.task6_perturbed_cell import evaluate as evaluate_cells
from downstream.task7_step_shuffling import rank_summary, shuffled_candidates
from downstream.task8_directional_reranking import evaluate_case as evaluate_directional
from downstream.task9_counterfactual import evaluate as evaluate_counterfactual
from downstream.task10_biosafety import evaluate as evaluate_biosafety


def main() -> None:
    records = [{
        "id": "a", "pathway_id": "Pathway-A", "question": "This is a pathway about Pathway-A.",
        "answer": "Gene TP53 activates gene MDM2.", "predicted_answer": "Gene TP53 activates gene MDM2.",
    }, {
        "id": "b", "pathway_id": "Pathway-B", "question": "This is a pathway about Pathway-B.",
        "answer": "Gene AKT1 inhibits gene BAD.", "predicted_answer": "Gene AKT1 inhibits gene BAD.",
    }]
    reference = {"Pathway-A": {"TP53", "MDM2"}, "Pathway-B": {"AKT1", "BAD"}}
    _, pcer = evaluate_pcer(records, reference, predicted_column="predicted_answer", target_column="answer", pathway_column="pathway_id", synonyms={})
    assert pcer["pcer"]["hit_at_1"] == 1.0 and pcer["entity_consistency"]["f1"] == 1.0, pcer

    trajectories = [np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)]
    _, pcte = evaluate_pcte(trajectories, trajectories, "cosine", 32)
    assert pcte["mean_pcte"] == 0.0, pcte

    csp = evaluate_pair(parse_steps(records[0]["answer"]), parse_steps(records[0]["predicted_answer"]))
    assert csp["exact_match"] == 1.0 and csp["reactant_match"] == 1.0, csp

    _, cki = evaluate_cki([{
        "case_id": "or-case", "ko_set_size": 1, "wt_survival": 0.9, "ko_survival": 0.8,
        "gold_gate": "redundant", "predicted_gate": "redundant", "wt_distribution": [0.9, 0.1],
        "ko_distribution": [0.8, 0.2],
    }])
    assert cki["gate_accuracy"] == 1.0, cki

    expression = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float)
    _, cells = evaluate_cells(np.zeros_like(expression), expression, expression, 20)
    assert cells["metrics"]["expression_pearson"] == 1.0 and cells["metrics"]["delta_pearson"] == 1.0, cells

    shuffles = shuffled_candidates(["A activates B", "B activates C", "C activates D"], 2, random.Random(7))
    assert len(shuffles) == 2
    shuffle_metrics = rank_summary([{"label": "gold", "score": -0.1}] + [{"label": "shuffled", "score": -1.0} for _ in shuffles])
    assert shuffle_metrics["hit_at_1"] == 1.0, shuffle_metrics

    directional = evaluate_directional([
        {"text": "A activates B", "label": "positive", "score": -0.1},
        {"text": "B activates A", "label": "negative", "negative_type": "direction_reversal", "score": -1.0},
    ])
    assert directional["directionality_accuracy"] == 1.0, directional

    trajectory = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=float)
    _, counterfactual = evaluate_counterfactual(trajectory, trajectory, trajectory, 32)
    assert counterfactual["metrics"]["counterfactual_pcte"] == 0.0, counterfactual

    _, biosafety = evaluate_biosafety([{
        "gold": {"risk_labels": ["R1"], "evidence_ids": ["E1"], "severity": 2},
        "prediction": {"risk_labels": ["R1"], "evidence_ids": ["E1"], "severity": 2},
    }])
    assert biosafety["risk_metrics"]["f1"] == 1.0 and biosafety["mean_severity_absolute_error"] == 0.0, biosafety
    print("All downstream metric smoke tests passed.")


if __name__ == "__main__":
    main()
