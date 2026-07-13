from __future__ import annotations

import json
import unittest

from method.training.sequence import (
    IncompleteSupervisionError,
    encode_supervised,
    pathway_step_substep_texts,
    trim_prompt_ids,
)


class CharacterTokenizer:
    """Deterministic fast-tokenizer stand-in: one token per character."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(character) for character in text]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        del add_special_tokens
        output: dict[str, object] = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            output["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return output


class SequenceEncodingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()

    def answer(self) -> str:
        return json.dumps(
            {
                "remaining_steps": [
                    {
                        "step": 1,
                        "layer": "layer 1",
                        "substeps": [
                            {"substep": 0, "text": "A activates B."},
                            {"substep": 1, "text": "C inhibits D."},
                        ],
                    },
                    {
                        "step": 2,
                        "layer": "layer 2",
                        "substeps": [{"substep": 0, "text": "B produces E."}],
                    },
                ],
                "predicted_phenotype": None,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    def test_hierarchical_substeps_remain_grouped_by_graph_layer(self) -> None:
        answer = self.answer()
        encoded = encode_supervised(
            self.tokenizer,
            "PROMPT",
            answer,
            max_length=len(answer) + 100,
            answer_budget_fraction=0.5,
        )
        self.assertEqual(pathway_step_substep_texts(answer), (("A activates B.", "C inhibits D."), ("B produces E.",)))
        self.assertEqual([len(group) for group in encoded.step_span_groups], [2, 1])
        self.assertEqual(encoded.semantic_steps_total, 2)
        self.assertEqual(encoded.semantic_steps_retained, 2)
        self.assertEqual(encoded.substeps_retained, 3)
        for group in encoded.step_span_groups:
            for start, end in group:
                self.assertGreater(end, start)
                self.assertGreaterEqual(start, len("PROMPT"))

    def test_truncation_keeps_only_complete_layer_groups(self) -> None:
        answer = self.answer()
        full = encode_supervised(
            self.tokenizer,
            "P",
            answer,
            max_length=len(answer) + 20,
            answer_budget_fraction=0.5,
        )
        second_start = full.step_span_groups[1][0][0] - 1
        truncated = encode_supervised(
            self.tokenizer,
            "P",
            answer,
            max_length=second_start,
            answer_budget_fraction=0.9,
            truncation_policy="measure",
        )
        self.assertEqual(truncated.semantic_steps_total, 2)
        self.assertEqual(truncated.semantic_steps_retained, 1)
        self.assertEqual([len(group) for group in truncated.step_span_groups], [2])
        self.assertGreater(truncated.answer_tokens_dropped, 0)

    def test_prompt_trimming_keeps_instruction_head_and_recent_tail(self) -> None:
        values = list(range(1000))
        trimmed = trim_prompt_ids(values, 120, head_tokens=30)
        self.assertEqual(trimmed[:30], values[:30])
        self.assertEqual(trimmed[30:], values[-90:])

    def test_legacy_text_steps_are_conservatively_split(self) -> None:
        answer = json.dumps(
            {
                "remaining_steps": [
                    {"step": 1, "layer": "layer 1", "text": "A activates B. C inhibits D."}
                ],
                "predicted_phenotype": None,
            }
        )
        self.assertEqual(
            pathway_step_substep_texts(answer),
            (("A activates B.", "C inhibits D."),),
        )

    def test_training_fails_instead_of_truncating_json(self) -> None:
        answer = self.answer()
        with self.assertRaises(IncompleteSupervisionError):
            encode_supervised(
                self.tokenizer,
                "P",
                answer,
                max_length=32,
            )

    def test_v3_event_texts_remain_grouped_by_layer(self) -> None:
        answer = json.dumps(
            {
                "schema_version": "pathway_continuation_v3",
                "remaining_layers": [
                    {
                        "layer_index": 1,
                        "events": [
                            {"text": "A activates B."},
                            {"text": "C inhibits D."},
                        ],
                    },
                    {
                        "layer_index": 2,
                        "events": [{"text": "B produces E."}],
                    },
                ],
            }
        )
        self.assertEqual(
            pathway_step_substep_texts(answer),
            (("A activates B.", "C inhibits D."), ("B produces E.",)),
        )


if __name__ == "__main__":
    unittest.main()
