from __future__ import annotations

import unittest

try:
    import torch

    from method.training.staged_objectives import (
        flatten_span_groups,
        gradient_conflict_statistics,
        linear_warmup_scale,
        multiscale_time_points,
        scale_gradient,
    )
except ModuleNotFoundError:  # Minimal local environments may omit PyTorch.
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch is required")
class StagedObjectiveTests(unittest.TestCase):
    def test_multiscale_time_retains_layer_boundaries(self) -> None:
        values = multiscale_time_points(
            [0, 0, 1, 2, 2],
            layer_dt=0.25,
            substep_dt=0.05,
            device="cpu",
        )
        self.assertTrue(torch.allclose(values, torch.tensor([0.0, 0.25, 0.30, 0.55, 0.80, 0.85])))

    def test_flatten_spans_retains_layer_identity_and_reports_truncation(self) -> None:
        groups = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 7]])]
        spans, layers, total = flatten_span_groups(groups, maximum_events=2)
        self.assertEqual(spans, [(1, 2), (3, 4)])
        self.assertEqual(layers, [0, 0])
        self.assertEqual(total, 3)

    def test_gradient_routing_scales_only_the_source_gradient(self) -> None:
        source = torch.tensor(2.0, requires_grad=True)
        dynamics_parameter = torch.tensor(3.0, requires_grad=True)
        loss = scale_gradient(source, 0.25) * dynamics_parameter
        source_grad, dynamics_grad = torch.autograd.grad(loss, (source, dynamics_parameter))
        self.assertAlmostEqual(float(source_grad), 0.75)
        self.assertAlmostEqual(float(dynamics_grad), 2.0)

    def test_linear_warmup_starts_at_zero_and_reaches_one(self) -> None:
        self.assertEqual(linear_warmup_scale(0, 100, 0.1), 0.0)
        self.assertEqual(linear_warmup_scale(5, 100, 0.1), 0.5)
        self.assertEqual(linear_warmup_scale(10, 100, 0.1), 1.0)

    def test_gradient_conflict_reports_opposition(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        left = parameter.sum()
        right = -parameter.sum()
        stats = gradient_conflict_statistics(left, right, [parameter])
        self.assertAlmostEqual(stats["cosine"], -1.0, places=6)
        self.assertAlmostEqual(stats["angle_degrees"], 180.0, places=5)


if __name__ == "__main__":
    unittest.main()
