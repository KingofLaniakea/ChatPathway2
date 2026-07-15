from __future__ import annotations

import unittest

try:
    import torch  # noqa: F401

    from method.training.hamiltonian_pretrain import stability_decision
except ModuleNotFoundError:  # Minimal local environments may omit PyTorch.
    stability_decision = None  # type: ignore[assignment]


@unittest.skipIf(stability_decision is None, "PyTorch is required")
class HamiltonianPretrainStabilityTests(unittest.TestCase):
    def test_stability_requires_coverage_improvement_and_no_regression(self) -> None:
        passed, reason = stability_decision(
            [
                {"dynamics": 1.0, "coverage": 0.99},
                {"dynamics": 0.8, "coverage": 0.99},
            ],
            minimum_epochs=2,
            minimum_coverage=0.95,
            minimum_relative_improvement=0.01,
            maximum_relative_regression=0.02,
        )
        self.assertTrue(passed)
        self.assertEqual(reason, "finite_covered_improved_and_non_regressing")

    def test_stability_rejects_low_coverage(self) -> None:
        passed, reason = stability_decision(
            [
                {"dynamics": 1.0, "coverage": 0.99},
                {"dynamics": 0.8, "coverage": 0.5},
            ],
            minimum_epochs=2,
            minimum_coverage=0.95,
            minimum_relative_improvement=0.01,
            maximum_relative_regression=0.02,
        )
        self.assertFalse(passed)
        self.assertEqual(reason, "insufficient_trajectory_coverage")


if __name__ == "__main__":
    unittest.main()
