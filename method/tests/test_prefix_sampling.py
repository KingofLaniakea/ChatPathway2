from __future__ import annotations

import unittest

from method.training.prefix_sampling import EpochPrefixView


def rows() -> list[dict[str, object]]:
    return [
        {
            "record_id": record,
            "sample_id": f"{record}:prefix={prefix}",
            "prefix_step_count": prefix,
        }
        for record in ("record-a", "record-b")
        for prefix in (1, 4, 9)
    ]


def horizon_rows() -> list[dict[str, object]]:
    horizons = ((1, "long_target"), (4, "middle_target"), (9, "short_target"))
    return [
        {
            "record_id": "record-a",
            "sample_id": f"record-a:prefix={prefix}",
            "prefix_step_count": prefix,
            "prefix_horizon": horizon,
        }
        for prefix, horizon in reversed(horizons)
    ]


class EpochPrefixViewTests(unittest.TestCase):
    def test_one_per_record_exposes_one_sample_per_epoch(self) -> None:
        view = EpochPrefixView(
            rows(),
            sampling_mode="one_per_record",
            policy="sft_cycle",
            seed=17,
        )
        self.assertEqual(view.eligible_row_count, 6)
        self.assertEqual(view.record_count, 2)
        self.assertEqual(len(view), 2)
        self.assertEqual(len(set(view.selected_row_indices())), 2)

    def test_sft_cycle_covers_long_middle_and_short_for_each_record(self) -> None:
        view = EpochPrefixView(
            rows(),
            sampling_mode="one_per_record",
            policy="sft_cycle",
            seed=23,
        )
        seen: dict[str, list[int]] = {"record-a": [], "record-b": []}
        for epoch in range(1, 5):
            view.set_epoch(epoch)
            for index in range(len(view)):
                row = view.row(index)
                seen[str(row["record_id"])].append(int(row["prefix_step_count"]))
        for values in seen.values():
            self.assertEqual(sorted(values), [1, 4, 9, 9])

    def test_selection_is_deterministic(self) -> None:
        left = EpochPrefixView(
            rows(),
            sampling_mode="one_per_record",
            policy="dynamics_cycle",
            seed=29,
        )
        right = EpochPrefixView(
            rows(),
            sampling_mode="one_per_record",
            policy="dynamics_cycle",
            seed=29,
        )
        for epoch in range(1, 9):
            left.set_epoch(epoch)
            right.set_epoch(epoch)
            self.assertEqual(left.selected_row_indices(), right.selected_row_indices())

    def test_all_rows_mode_preserves_every_input_row(self) -> None:
        source = rows()
        view = EpochPrefixView(
            source,
            sampling_mode="all_rows",
            policy="balanced_cycle",
            seed=31,
        )
        self.assertEqual(len(view), len(source))
        self.assertEqual(
            [view.row(index)["sample_id"] for index in range(len(view))],
            [row["sample_id"] for row in source],
        )

    def test_duplicate_prefix_for_one_record_is_rejected(self) -> None:
        source = rows()
        source.append(dict(source[0]))
        with self.assertRaisesRegex(ValueError, "duplicate prefix_step_count"):
            EpochPrefixView(
                source,
                sampling_mode="one_per_record",
                policy="sft_cycle",
                seed=37,
            )

    def test_explicit_horizon_labels_drive_selection_not_row_position(self) -> None:
        view = EpochPrefixView(
            horizon_rows(),
            sampling_mode="one_per_record",
            policy="balanced_cycle",
            seed=0,
        )
        observed = set()
        for epoch in range(1, 4):
            view.set_epoch(epoch)
            observed.add(str(view.row(0)["prefix_horizon"]))
        self.assertEqual(
            observed,
            {"long_target", "middle_target", "short_target"},
        )

    def test_mixed_or_duplicate_horizon_labels_fail_closed(self) -> None:
        mixed = horizon_rows()
        mixed[0].pop("prefix_horizon")
        with self.assertRaisesRegex(ValueError, "mixes explicit and missing"):
            EpochPrefixView(
                mixed,
                sampling_mode="one_per_record",
                policy="balanced_cycle",
                seed=41,
            )

        duplicate = horizon_rows()
        duplicate[0]["prefix_horizon"] = "middle_target"
        with self.assertRaisesRegex(ValueError, "duplicate prefix_horizon"):
            EpochPrefixView(
                duplicate,
                sampling_mode="one_per_record",
                policy="balanced_cycle",
                seed=43,
            )

    def test_degenerate_single_horizon_is_a_valid_deterministic_fallback(self) -> None:
        view = EpochPrefixView(
            [
                {
                    "record_id": "short-record",
                    "sample_id": "short-record:prefix=1",
                    "prefix_step_count": 1,
                    "prefix_horizon": "degenerate_target",
                }
            ],
            sampling_mode="one_per_record",
            policy="dynamics_cycle",
            seed=47,
        )
        for epoch in range(1, 5):
            view.set_epoch(epoch)
            self.assertEqual(view.row(0)["prefix_horizon"], "degenerate_target")


if __name__ == "__main__":
    unittest.main()
