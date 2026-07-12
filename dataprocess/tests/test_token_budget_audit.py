from __future__ import annotations

import json
import unittest

from dataprocess.audit_token_budget import audit_rows


class CharacterTokenizer:
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
            output["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]
        return output


def answer(text: str) -> str:
    return json.dumps(
        {
            "remaining_steps": [
                {
                    "step": 1,
                    "layer": "layer 1",
                    "substeps": [{"substep": 0, "text": text}],
                }
            ],
            "predicted_phenotype": None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class TokenBudgetAuditTests(unittest.TestCase):
    def test_reports_full_and_zero_layer_retention(self) -> None:
        rows = [
            {"question": "short", "answer": answer("A activates B.")},
            {"question": "short", "answer": answer("X" * 500)},
        ]

        report = audit_rows(
            rows,
            CharacterTokenizer(),
            max_length=220,
            answer_budget_fraction=0.5,
        )

        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["rows_with_semantic_steps"], 2)
        self.assertEqual(report["rows_full_semantic_step_retention"], 1)
        self.assertEqual(report["rows_zero_retained_semantic_steps"], 1)
        self.assertEqual(report["rows_answer_truncated"], 1)
        self.assertEqual(report["semantic_steps_total"], 2)
        self.assertEqual(report["semantic_steps_retained"], 1)
        self.assertEqual(report["semantic_step_retention_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
