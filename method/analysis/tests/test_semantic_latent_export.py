from __future__ import annotations

import json
import unittest

import numpy as np

from method.analysis.semantic_latent_export import (
    encode_role,
    pad_trajectories,
    stable_sample_id,
)


class CharacterTokenizer:
    """Dependency-free tokenizer with exact character offsets."""

    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [index + 1 for index, _ in enumerate(text)]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, object]:
        del add_special_tokens
        result: dict[str, object] = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return result


class SemanticLatentExportTests(unittest.TestCase):
    def test_same_layer_substeps_produce_one_point(self) -> None:
        answer = json.dumps(
            {
                "remaining_steps": [
                    {
                        "step": 2,
                        "layer": "layer 2",
                        "substeps": [
                            {"text": "A activates B"},
                            {"text": "C inhibits D"},
                        ],
                    },
                    {
                        "step": 3,
                        "layer": "layer 3",
                        "substeps": [{"text": "B binds E"}],
                    },
                ],
                "predicted_phenotype": None,
            },
            ensure_ascii=False,
        )
        encoded = encode_role(
            CharacterTokenizer(),
            question="Continue this pathway",
            answer=answer,
            sample_index=0,
            role="gold",
            max_length=4096,
            answer_budget_fraction=0.5,
            max_steps=128,
        )
        self.assertEqual(encoded.point_spec.length, 3)  # anchor + two layers
        self.assertEqual(len(encoded.point_spec.layer_span_groups[0]), 2)
        self.assertEqual(len(encoded.point_spec.layer_span_groups[1]), 1)

    def test_max_steps_caps_layers_not_substeps(self) -> None:
        answer = json.dumps(
            {
                "remaining_steps": [
                    {
                        "step": 1,
                        "layer": "layer 1",
                        "substeps": [{"text": "A to B"}, {"text": "C to D"}],
                    },
                    {"step": 2, "layer": "layer 2", "substeps": [{"text": "B to E"}]},
                ]
            }
        )
        encoded = encode_role(
            CharacterTokenizer(),
            question="q",
            answer=answer,
            sample_index=0,
            role="gold",
            max_length=4096,
            answer_budget_fraction=0.5,
            max_steps=1,
        )
        self.assertEqual(encoded.point_spec.length, 2)
        self.assertEqual(len(encoded.point_spec.layer_span_groups[0]), 2)

    def test_padding_preserves_lengths_and_uses_float32(self) -> None:
        padded, lengths = pad_trajectories(
            [np.ones((2, 3), dtype=np.float64), np.full((4, 3), 2, dtype=np.float32)]
        )
        self.assertEqual(padded.shape, (2, 4, 3))
        self.assertEqual(padded.dtype, np.float32)
        np.testing.assert_array_equal(lengths, np.asarray([2, 4]))
        np.testing.assert_array_equal(padded[0, 2:], np.zeros((2, 3)))

    def test_fallback_sample_id_is_stable(self) -> None:
        row = {"question": "q", "source_json": "source.json", "prefix_step_count": "2"}
        self.assertEqual(stable_sample_id(row, 7), stable_sample_id(row, 7))
        self.assertTrue(stable_sample_id(row, 7).startswith("semantic-"))


if __name__ == "__main__":
    unittest.main()
