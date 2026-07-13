from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    from method.training.csv_io import read_training_csv
except ModuleNotFoundError:  # Minimal local test environments may omit pandas.
    read_training_csv = None  # type: ignore[assignment]


@unittest.skipIf(read_training_csv is None, "pandas is required")
class TrainingCsvIoTests(unittest.TestCase):
    def test_identity_columns_keep_leading_zero_and_empty_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train.csv"
            source.write_text(
                "pathway_family_id,record_id,prefix_step_count,phenotype\n"
                "00051,abc123,2,\n",
                encoding="utf-8",
            )
            frame = read_training_csv(source)
        self.assertEqual(frame.loc[0, "pathway_family_id"], "00051")
        self.assertEqual(frame.loc[0, "prefix_step_count"], "2")
        self.assertEqual(frame.loc[0, "phenotype"], "")


if __name__ == "__main__":
    unittest.main()
