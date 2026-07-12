from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from method.inference.csv_io import read_csv_text_rows


class InferenceCsvIoTests(unittest.TestCase):
    def test_preserves_leading_zero_identity_empty_value_and_quoted_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["pathway_family_id", "phenotype", "question"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pathway_family_id": "00051",
                        "phenotype": "",
                        "question": 'Line one, "quoted".\nLine two.',
                    }
                )

            fieldnames, rows = read_csv_text_rows(path)

        self.assertEqual(
            fieldnames,
            ["pathway_family_id", "phenotype", "question"],
        )
        self.assertEqual(rows[0]["pathway_family_id"], "00051")
        self.assertEqual(rows[0]["phenotype"], "")
        self.assertEqual(rows[0]["question"], 'Line one, "quoted".\nLine two.')

    def test_limit_is_applied_without_type_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.csv"
            path.write_text("id\n00001\n00002\n", encoding="utf-8")
            _, rows = read_csv_text_rows(path, limit=1)

        self.assertEqual(rows, [{"id": "00001"}])


if __name__ == "__main__":
    unittest.main()
