from __future__ import annotations

import json
import unittest

from experiments.validate_matrix import MATRIX_PATH, validate_research_plan


class ResearchPlanTests(unittest.TestCase):
    def test_current_matrix_and_post_current_generation_contract(self) -> None:
        matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
        self.assertEqual(validate_research_plan(matrix), [])

    def test_token_resolution_plan_cannot_reuse_layer_checkpoint(self) -> None:
        matrix = json.loads(MATRIX_PATH.read_text(encoding="utf-8"))
        token_item = next(
            item
            for item in matrix["combinations"]
            if item["id"] == "plan012_token_resolution_stepwise"
        )
        token_item["forbids_graph_layer_checkpoint_per_token"] = False
        errors = validate_research_plan(matrix)
        self.assertTrue(any("graph-layer checkpoint reuse" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
