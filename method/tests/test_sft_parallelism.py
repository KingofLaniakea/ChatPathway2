from __future__ import annotations

import unittest

from method.training.sft import LengthGroupedDistributedSampler


class _Lengths:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def estimated_text_length(self, index: int) -> int:
        return self.values[index]


class SftParallelismTests(unittest.TestCase):
    def test_length_grouped_sampler_balances_ddp_steps_and_covers_data(self) -> None:
        dataset = _Lengths([100, 10, 90, 20, 80, 30, 70, 40, 60, 50])
        samplers = [
            LengthGroupedDistributedSampler(
                dataset,  # type: ignore[arg-type]
                num_replicas=4,
                rank=rank,
                batch_size=1,
                seed=17,
            )
            for rank in range(4)
        ]
        for sampler in samplers:
            sampler.set_epoch(2)
        shards = [list(sampler) for sampler in samplers]

        self.assertTrue(all(len(shard) == 3 for shard in shards))
        self.assertEqual(set().union(*(set(shard) for shard in shards)), set(range(10)))
        for step in range(3):
            step_lengths = [dataset.values[shard[step]] for shard in shards]
            self.assertLessEqual(max(step_lengths) - min(step_lengths), 30)

    def test_length_grouped_sampler_changes_group_order_by_epoch(self) -> None:
        dataset = _Lengths(list(range(32)))
        sampler = LengthGroupedDistributedSampler(
            dataset,  # type: ignore[arg-type]
            num_replicas=4,
            rank=0,
            batch_size=1,
            seed=23,
        )
        sampler.set_epoch(1)
        first = list(sampler)
        sampler.set_epoch(2)
        second = list(sampler)
        self.assertNotEqual(first, second)
        self.assertEqual(set(first), set(second))


if __name__ == "__main__":
    unittest.main()
