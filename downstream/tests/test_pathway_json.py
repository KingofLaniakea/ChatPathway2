from __future__ import annotations

import json
import unittest

from downstream.common.pathway_json import parse_pathway_payload


class PathwayJsonV3Tests(unittest.TestCase):
    def payload(self) -> dict[str, object]:
        return {
            "schema_version": "pathway_continuation_v3",
            "remaining_layers": [
                {
                    "layer_index": 2,
                    "events": [
                        {
                            "source": [{"canonical_id": "ko:K1", "name": "A"}],
                            "relation": "activation",
                            "target": [{"canonical_id": "ko:K2", "name": "B"}],
                            "text": "A activates B.",
                        }
                    ],
                },
                {
                    "layer_index": 3,
                    "events": [
                        {
                            "source": [{"canonical_id": "ko:K2", "name": "B"}],
                            "relation": "inhibition",
                            "target": [{"canonical_id": "ko:K3", "name": "C"}],
                            "text": "B inhibits C.",
                        }
                    ],
                },
            ],
        }

    def test_v3_is_strictly_parsed_for_downstream_tasks(self) -> None:
        parsed = parse_pathway_payload(json.dumps(self.payload()))
        self.assertTrue(parsed.json_valid)
        self.assertTrue(parsed.schema_valid)
        self.assertEqual([step.step for step in parsed.steps], [2, 3])
        self.assertEqual(parsed.steps[0].substeps, ("A activates B.",))

    def test_v3_rejects_nonconsecutive_layers(self) -> None:
        payload = self.payload()
        payload["remaining_layers"][1]["layer_index"] = 4  # type: ignore[index]
        parsed = parse_pathway_payload(payload)
        self.assertFalse(parsed.schema_valid)

    def test_v3_rejects_extra_model_output_fields(self) -> None:
        payload = self.payload()
        payload["organism"] = "hsa"
        parsed = parse_pathway_payload(payload)
        self.assertFalse(parsed.schema_valid)


if __name__ == "__main__":
    unittest.main()
