from __future__ import annotations

import unittest

from method.training.framework_a_ddp import (
    MatchedGlobalBatchTrainSampler,
    StridedEvaluationSampler,
)


class StridedEvaluationSamplerTests(unittest.TestCase):
    def test_every_row_is_assigned_once_without_padding(self) -> None:
        source = list(range(11))
        partitions = [
            list(StridedEvaluationSampler(source, rank=rank, world_size=4))
            for rank in range(4)
        ]

        self.assertEqual(partitions, [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7]])
        flattened = [value for partition in partitions for value in partition]
        self.assertEqual(sorted(flattened), source)
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_more_ranks_than_rows_yields_empty_tail_partitions(self) -> None:
        source = [0, 1]
        lengths = [
            len(StridedEvaluationSampler(source, rank=rank, world_size=4))
            for rank in range(4)
        ]
        self.assertEqual(lengths, [1, 1, 0, 0])


class MatchedGlobalBatchTrainSamplerTests(unittest.TestCase):
    @staticmethod
    def reconstruct_global_order(samplers: list[MatchedGlobalBatchTrainSampler]) -> list[int]:
        partitions = [list(sampler) for sampler in samplers]
        return [
            partitions[rank][offset]
            for offset in range(len(partitions[0]))
            for rank in range(len(partitions))
        ]

    def test_two_and_four_ranks_receive_identical_global_batches(self) -> None:
        source = list(range(10))
        two_rank = [
            MatchedGlobalBatchTrainSampler(
                source,
                rank=rank,
                world_size=2,
                global_batch_size=12,
                seed=17,
            )
            for rank in range(2)
        ]
        four_rank = [
            MatchedGlobalBatchTrainSampler(
                source,
                rank=rank,
                world_size=4,
                global_batch_size=12,
                seed=17,
            )
            for rank in range(4)
        ]

        two_order = self.reconstruct_global_order(two_rank)
        four_order = self.reconstruct_global_order(four_rank)
        self.assertEqual(two_order, four_order)
        self.assertEqual(len(two_order), 12)
        self.assertEqual(sorted(set(two_order)), source)
        self.assertEqual(two_rank[0].padding_rows, 2)

    def test_epoch_changes_deterministically(self) -> None:
        sampler = MatchedGlobalBatchTrainSampler(
            list(range(20)),
            rank=0,
            world_size=2,
            global_batch_size=4,
            seed=9,
        )
        epoch_zero = list(sampler)
        sampler.set_epoch(1)
        epoch_one = list(sampler)
        sampler.set_epoch(0)
        self.assertNotEqual(epoch_zero, epoch_one)
        self.assertEqual(list(sampler), epoch_zero)


if __name__ == "__main__":
    unittest.main()
