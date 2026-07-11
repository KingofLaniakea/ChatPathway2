from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - local documentation-only environments
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class HamiltonianDynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        from method.dynamics.hamiltonian import LatentHamiltonianDynamics

        self.model_class = LatentHamiltonianDynamics
        torch.manual_seed(7)

    def test_learned_structure_is_exactly_skew(self) -> None:
        model = self.model_class(8, variant="hnn", structure_mode="learned_skew")
        structure = model.structure_matrix()
        self.assertTrue(torch.allclose(structure + structure.T, torch.zeros_like(structure), atol=1e-7))

    def test_orthogonal_poisson_structure_stays_full_rank(self) -> None:
        model = self.model_class(8, variant="hnn", structure_mode="orthogonal_poisson")
        structure = model.structure_matrix()
        identity = torch.eye(8)
        self.assertTrue(torch.allclose(structure + structure.T, torch.zeros_like(structure), atol=1e-5))
        self.assertTrue(torch.allclose(structure @ structure, -identity, atol=1e-5))
        self.assertEqual(int(torch.linalg.matrix_rank(structure)), 8)

    def test_conservative_power_is_zero(self) -> None:
        model = self.model_class(8, variant="hnn").eval()
        state = torch.randn(5, 8)
        power = model.energy_rate_terms(0.0, state)["conservative_power"]
        self.assertLess(float(power.abs().max()), 1e-5)

    def test_forced_damped_constraints(self) -> None:
        model = self.model_class(8, variant="forced_damped_hnn").eval()
        state = torch.randn(5, 8)
        damping = model.damping_diagonal()
        terms = model.energy_rate_terms(0.0, state)
        self.assertTrue(bool(torch.all(damping >= 0)))
        self.assertTrue(bool(torch.all(terms["dissipative_power"] <= 1e-7)))
        self.assertLess(float(terms["force_power"].abs().max()), 1e-7)

    def test_training_gradient_reaches_structural_parameters(self) -> None:
        model = self.model_class(8, variant="forced_damped_hnn").train()
        state = torch.randn(3, 8)
        loss = model(0.25, state).pow(2).mean()
        loss.backward()
        self.assertIsNotNone(model.raw_reflections.grad)
        self.assertIsNotNone(model.raw_damping.grad)
        self.assertIsNotNone(model.force_net[-1].weight.grad)


@unittest.skipIf(torch is None, "PyTorch is required")
class GroupSplitTests(unittest.TestCase):
    def test_source_groups_do_not_cross_split(self) -> None:
        import pandas as pd

        from method.training.common import stable_group_split

        frame = pd.DataFrame(
            {
                "source_json": ["a.json", "a.json", "b.json", "b.json", "c.json"],
                "question": ["a1", "a2", "b1", "b2", "c1"],
            }
        )
        train, validation = stable_group_split(
            frame,
            validation_fraction=0.2,
            seed=7,
            group_column="source_json",
        )
        self.assertTrue(set(train.source_json).isdisjoint(set(validation.source_json)))

    def test_missing_identity_column_is_rejected(self) -> None:
        import pandas as pd

        from method.training.common import stable_group_split

        with self.assertRaises(ValueError):
            stable_group_split(
                pd.DataFrame({"question": ["a", "b"]}),
                validation_fraction=0.2,
                seed=7,
                group_column="source_json",
            )

    def test_explicit_validation_overlap_is_rejected(self) -> None:
        import pandas as pd

        from method.training.common import ensure_disjoint_groups

        train = pd.DataFrame({"source_json": ["a.json", "b.json"]})
        validation = pd.DataFrame({"source_json": ["b.json", "c.json"]})
        with self.assertRaises(ValueError):
            ensure_disjoint_groups(train, validation, group_column="source_json")


@unittest.skipIf(torch is None, "PyTorch is required")
class CollateMaskTests(unittest.TestCase):
    def test_content_token_equal_to_pad_id_stays_attended(self) -> None:
        from method.training.sft import make_collate_fn

        collate = make_collate_fn(pad_id=9)
        result = collate(
            [
                {
                    "input_ids": torch.tensor([1, 9, 2]),
                    "labels": torch.tensor([-100, 9, 2]),
                    "prompt_tokens_dropped": 0,
                    "answer_tokens_dropped": 0,
                },
                {
                    "input_ids": torch.tensor([3]),
                    "labels": torch.tensor([3]),
                    "prompt_tokens_dropped": 0,
                    "answer_tokens_dropped": 0,
                },
            ]
        )
        self.assertEqual(result["attention_mask"].tolist(), [[1, 1, 1], [1, 0, 0]])


if __name__ == "__main__":
    unittest.main()
