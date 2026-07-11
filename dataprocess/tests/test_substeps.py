from __future__ import annotations

import unittest

from dataprocess.substeps import PathwaySubstep, parse_substeps, split_substeps


class SubstepParserTests(unittest.TestCase):
    def test_splits_period_separated_biological_events_in_order(self) -> None:
        text = (
            "Compound glucose is converted to Compound glucose-6-phosphate. "
            "Gene GPI converts Compound glucose-6-phosphate to Compound fructose-6-phosphate. "
            "Pathway glycolysis produces Compound pyruvate."
        )

        self.assertEqual(
            split_substeps(text),
            (
                "Compound glucose is converted to Compound glucose-6-phosphate.",
                "Gene GPI converts Compound glucose-6-phosphate to Compound fructose-6-phosphate.",
                "Pathway glycolysis produces Compound pyruvate.",
            ),
        )

    def test_source_items_are_authoritative_boundaries_with_provenance(self) -> None:
        parsed = parse_substeps(
            "This aggregate is intentionally different.",
            source_items=(
                "Gene A activates Gene B",
                "   ",
                "Gene B inhibits Gene C. Gene C activates Pathway D.",
                "Gene A activates Gene B",
            ),
        )

        self.assertEqual(
            parsed,
            (
                PathwaySubstep(0, "Gene A activates Gene B", 0),
                PathwaySubstep(1, "Gene B inhibits Gene C.", 2),
                PathwaySubstep(2, "Gene C activates Pathway D.", 2),
                PathwaySubstep(3, "Gene A activates Gene B", 3),
            ),
        )

    def test_decimals_dotted_identifiers_and_abbreviations_do_not_split(self) -> None:
        text = (
            "Enzyme EC 1.2.3.4 converts 3.5 mM Compound A, e.g. alpha-D-glucose, "
            "in E. coli. Gene B activates Pathway C."
        )

        self.assertEqual(
            split_substeps(text),
            (
                "Enzyme EC 1.2.3.4 converts 3.5 mM Compound A, e.g. alpha-D-glucose, in E. coli.",
                "Gene B activates Pathway C.",
            ),
        )

    def test_hgvs_and_reference_abbreviations_do_not_split(self) -> None:
        text = (
            "Gene TP53 variant p.R175H is described in Fig. 2 and inhibits Gene MDM2. "
            "Pathway apoptosis is activated."
        )

        self.assertEqual(
            split_substeps(text),
            (
                "Gene TP53 variant p.R175H is described in Fig. 2 and inhibits Gene MDM2.",
                "Pathway apoptosis is activated.",
            ),
        )

    def test_terminal_quotes_and_question_marks_stay_with_substep(self) -> None:
        text = 'Gene A activates "Gene B." Does Gene B inhibit Gene C? Pathway D follows.'
        self.assertEqual(
            split_substeps(text),
            (
                'Gene A activates "Gene B."',
                "Does Gene B inhibit Gene C?",
                "Pathway D follows.",
            ),
        )

    def test_whitespace_is_normalized_and_output_is_deterministic(self) -> None:
        text = "  Gene A\n activates   Gene B.   Gene B inhibits Gene C.  "
        expected = (
            "Gene A activates Gene B.",
            "Gene B inhibits Gene C.",
        )
        self.assertEqual(split_substeps(text), expected)
        self.assertEqual(split_substeps(text), expected)

    def test_blank_source_items_fall_back_to_aggregate_text(self) -> None:
        self.assertEqual(
            split_substeps(
                "Gene A activates Gene B.",
                source_items=("", "   "),
            ),
            ("Gene A activates Gene B.",),
        )

    def test_blank_input_returns_empty_tuple(self) -> None:
        self.assertEqual(parse_substeps("  "), ())

    def test_records_are_json_serializable(self) -> None:
        record = parse_substeps(
            "ignored",
            source_items=("Gene A activates Gene B.",),
        )[0]
        self.assertEqual(
            record.as_dict(),
            {
                "substep": 0,
                "text": "Gene A activates Gene B.",
                "source_item_index": 0,
            },
        )

    def test_rejects_invalid_types(self) -> None:
        with self.assertRaises(TypeError):
            parse_substeps(None)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            parse_substeps("text", source_items="not-a-sequence")
        with self.assertRaises(TypeError):
            parse_substeps("text", source_items=("valid", 3))  # type: ignore[list-item]


if __name__ == "__main__":
    unittest.main()
