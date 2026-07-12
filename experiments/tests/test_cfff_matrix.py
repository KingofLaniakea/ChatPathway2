from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.run_cfff_matrix import build_jobs, run_scheduler


class CfffMatrixSchedulerTests(unittest.TestCase):
    def test_job_graph_uses_four_gpu_sft_then_single_gpu_arms(self) -> None:
        root = Path("/assets")
        jobs = build_jobs([11], root, "/python")
        by_key = {job.key: job for job in jobs}

        self.assertEqual(len(jobs), 9)
        self.assertEqual(by_key["11:sft"].resources, 4)
        self.assertEqual(by_key["11:ae"].resources, 1)
        self.assertEqual(by_key["11:ae"].dependencies, ("11:sft",))
        self.assertEqual(
            by_key["11:exp001_hnn_reconae_joint_direct:infer"].dependencies,
            ("11:exp001_hnn_reconae_joint_direct:train",),
        )
        self.assertIn("torch.distributed.run", by_key["11:sft"].command)

    def test_dry_run_does_not_require_runtime_assets(self) -> None:
        jobs = build_jobs([11], Path("/missing"), "/python")
        with tempfile.TemporaryDirectory() as directory:
            result = run_scheduler(
                jobs,
                gpus=["0", "1", "2", "3"],
                profile="cfff",
                log_dir=Path(directory),
                poll_seconds=0.01,
                skip_existing=True,
                dry_run=True,
            )
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
