from __future__ import annotations

import json
import unittest

from method.inference.json_retry import (
    generation_validity,
    repair_prompt,
    retry_token_budget,
)


class InferenceJsonRetryTests(unittest.TestCase):
    def test_v3_generation_must_be_complete_and_schema_valid(self) -> None:
        payload = {
            "schema_version": "pathway_continuation_v3",
            "remaining_layers": [
                {
                    "layer_index": 1,
                    "events": [
                        {
                            "source": [{"canonical_id": "A", "name": "A"}],
                            "relation": "activation",
                            "target": [{"canonical_id": "B", "name": "B"}],
                            "text": "A activates B.",
                        }
                    ],
                }
            ],
        }
        self.assertEqual(
            generation_validity(json.dumps(payload), expected_first_layer=1)[:2],
            (True, True),
        )
        self.assertEqual(
            generation_validity(json.dumps(payload), expected_first_layer=2)[:2],
            (True, False),
        )
        self.assertEqual(
            generation_validity(f"```json\n{json.dumps(payload)}\n```")[:2],
            (False, False),
        )
        self.assertEqual(generation_validity('{"schema_version":')[:2], (False, False))

    def test_repair_prompt_preserves_original_but_omits_failed_output(self) -> None:
        original = "<|im_start|>user\nORIGINAL<|im_end|>\n<|im_start|>assistant\n"
        prompt = repair_prompt(original, "{bad", "invalid_json", 2)
        self.assertIn("ORIGINAL", prompt)
        self.assertNotIn("{bad", prompt)
        self.assertIn("repair attempt 2 of 3", prompt)
        self.assertIn("exactly one complete JSON object", prompt)

    def test_final_attempt_uses_full_retry_budget(self) -> None:
        values = {
            "max_new_tokens": 1024,
            "retry_max_new_tokens": 8192,
            "max_json_attempts": 3,
        }
        self.assertEqual(retry_token_budget(**values, attempt=2), 2048)
        self.assertEqual(retry_token_budget(**values, attempt=3), 8192)


if __name__ == "__main__":
    unittest.main()
