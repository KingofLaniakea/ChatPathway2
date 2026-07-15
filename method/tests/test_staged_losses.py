from __future__ import annotations

import math
import unittest

try:
    import torch

    from method.training.framework_a import (
        stage1_reference_kl,
        supervised_logit_positions,
    )
    from method.training.latent_ae import latent_geometry_losses
except ModuleNotFoundError:  # Minimal local environments may omit model dependencies.
    torch = None  # type: ignore[assignment]


@unittest.skipIf(torch is None, "PyTorch and training dependencies are required")
class StagedLossTests(unittest.TestCase):
    def test_supervised_positions_apply_causal_shift(self) -> None:
        labels = torch.tensor([[-100, -100, 7, 8], [-100, 9, -100, -100]])
        positions = supervised_logit_positions(labels, maximum=8)
        self.assertEqual(positions.tolist(), [[0, 1], [0, 2], [1, 0]])

    def test_identical_teacher_and_student_have_zero_kl(self) -> None:
        logits = torch.tensor([[[1.0, -1.0], [0.5, 0.25]]])
        positions = torch.tensor([[0, 0], [0, 1]])
        loss = stage1_reference_kl(logits, logits[0, positions[:, 1]], positions)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_geometry_losses_do_not_assign_coordinate_semantics(self) -> None:
        root_two = math.sqrt(2.0)
        latent = torch.tensor(
            [[root_two, 0.0], [-root_two, 0.0], [0.0, root_two], [0.0, -root_two]]
        )
        losses = latent_geometry_losses(latent)
        self.assertAlmostEqual(float(losses["latent_mean"]), 0.0, places=6)
        self.assertAlmostEqual(float(losses["latent_variance"]), 0.0, places=6)
        self.assertAlmostEqual(float(losses["latent_covariance"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
