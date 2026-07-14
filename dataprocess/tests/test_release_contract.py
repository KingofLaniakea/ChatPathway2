from __future__ import annotations

import itertools
import unittest

from dataprocess.release_contract import (
    OVERLAP_CONTRACT,
    PARTITIONS,
    PRIMARY_PROMPT_PROFILE,
    normalized_pair,
)


class ReleaseContractTests(unittest.TestCase):
    def test_every_partition_pair_has_a_fixed_overlap_contract(self) -> None:
        expected = set(itertools.combinations(PARTITIONS, 2))
        self.assertEqual(set(OVERLAP_CONTRACT), expected)
        for contract in OVERLAP_CONTRACT.values():
            self.assertEqual(set(contract), {"family", "organism"})

    def test_primary_profile_is_explicit_organism(self) -> None:
        self.assertEqual(
            PRIMARY_PROMPT_PROFILE,
            "explicit_organism_source_native_ids",
        )

    def test_partition_pairs_normalize_deterministically(self) -> None:
        self.assertEqual(normalized_pair("test", "train"), ("train", "test"))
        with self.assertRaises(ValueError):
            normalized_pair("train", "train")


if __name__ == "__main__":
    unittest.main()
