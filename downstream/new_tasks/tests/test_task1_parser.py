from __future__ import annotations

import unittest

from downstream.new_tasks.task1_substep_csp import parse_atomic_clause


class Task1SubstepParserTests(unittest.TestCase):
    def assert_event(
        self,
        text: str,
        *,
        source: tuple[str, ...],
        relation: str,
        target: tuple[str, ...],
    ) -> None:
        event = parse_atomic_clause(text, step=0, substep=0)
        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event.source, source)
        self.assertEqual(event.relation, relation)
        self.assertEqual(event.target, target)

    def test_plural_conversion_preserves_complete_compound_names(self) -> None:
        self.assert_event(
            "Compound D-Erythrose 4-phosphate and compound Glycerone phosphate are converted to "
            "compound Sedoheptulose 1,7-bisphosphate in a irreversible way.",
            source=("d-erythrose 4-phosphate", "glycerone phosphate"),
            relation="convert",
            target=("sedoheptulose 1,7-bisphosphate",),
        )

    def test_hyphenated_entity_words_are_not_relation_matches(self) -> None:
        self.assert_event(
            "Gene Stress-activated protein kinase jnk-1 and gene GLH-binding kinase 1 "
            "activates gene Transcription factor fos-1.",
            source=("glh-binding kinase 1", "stress-activated protein kinase jnk-1"),
            relation="activate",
            target=("transcription factor fos-1",),
        )

    def test_expression_template_is_normalized_to_regulation(self) -> None:
        self.assert_event(
            "Gene Forkhead box protein O regulates the expression of gene "
            "Mitogen-activated protein kinase kinase kinase dlk-1.",
            source=("forkhead box protein o",),
            relation="regulate",
            target=("mitogen-activated protein kinase kinase kinase dlk-1",),
        )

    def test_activity_template_excludes_via_mediator_from_target(self) -> None:
        self.assert_event(
            "Gene 1-phosphatidylinositol 4,5-bisphosphate phosphodiesterase beta egl-8 can affect "
            "the activity of gene Protein kinase C-like 2 via compound Calcium cation.",
            source=("1-phosphatidylinositol 4,5-bisphosphate phosphodiesterase beta egl-8",),
            relation="affect_activity",
            target=("protein kinase c-like 2",),
        )

    def test_indirect_link_template_excludes_effect_qualifier(self) -> None:
        self.assert_event(
            "Gene Ras-like GTP-binding protein rhoA has an indirect link with gene "
            "Stress-activated protein kinase JNK and exerts an indirect effect.",
            source=("ras-like gtp-binding protein rhoa",),
            relation="indirect_link",
            target=("stress-activated protein kinase jnk",),
        )


if __name__ == "__main__":
    unittest.main()
