from __future__ import annotations

import unittest

from dataprocess.select_training_coverage import (
    RecordRows,
    family_diversity_order,
    select_optimization_records,
    validation_families,
)


def record(
    record_id: str,
    family: str,
    organism: str,
    total_steps: int,
    source_order: int,
) -> RecordRows:
    row = {
        "record_id": record_id,
        "sample_id": f"{record_id}:prefix=1",
        "pathway_family_id": family,
        "organism": organism,
        "total_step": str(total_steps),
        "prefix_step_count": "1",
    }
    return RecordRows(
        record_id=record_id,
        family=family,
        organism=organism,
        total_steps=total_steps,
        rows=(row,),
        source_order=source_order,
    )


class CoverageSelectionTests(unittest.TestCase):
    def test_family_cap_retains_every_family_and_is_nested(self) -> None:
        records = [
            record(f"a-{index}", "00001", f"org-{index}", 3 + index, index)
            for index in range(6)
        ] + [
            record(f"b-{index}", "00002", f"other-{index}", 5 + index, 10 + index)
            for index in range(3)
        ]
        cap_two = select_optimization_records(records, maximum_per_family=2, seed=7)
        cap_four = select_optimization_records(records, maximum_per_family=4, seed=7)
        self.assertEqual({item.family for item in cap_two}, {"00001", "00002"})
        self.assertLessEqual(max(
            sum(item.family == family for item in cap_two)
            for family in {"00001", "00002"}
        ), 2)
        self.assertTrue(
            {item.record_id for item in cap_two}
            <= {item.record_id for item in cap_four}
        )

    def test_distinct_organisms_are_prioritized_before_repeats(self) -> None:
        records = [
            record("same-1", "00001", "same", 3, 0),
            record("same-2", "00001", "same", 4, 1),
            record("other-1", "00001", "other", 9, 2),
        ]
        ordered = family_diversity_order(records, seed=11)
        self.assertEqual(len({item.organism for item in ordered[:2]}), 2)

    def test_validation_family_selection_preserves_leading_zero_identity(self) -> None:
        families = {"00001", "00002", "00003", "00004"}
        selected = validation_families(families, fraction=0.25, seed=13)
        self.assertTrue(selected)
        self.assertTrue(selected < families)
        self.assertTrue(all(len(item) == 5 for item in selected))


if __name__ == "__main__":
    unittest.main()
