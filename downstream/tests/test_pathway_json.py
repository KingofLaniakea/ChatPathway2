from __future__ import annotations

import json
import unittest

from downstream.common.pathway_json import parse_pathway_payload


class PathwayJsonV4Tests(unittest.TestCase):
    def payload(self) -> dict[str, object]:
        return {
            "schema_version": "pathway_continuation_v4",
            "remaining_layers": [
                {
                    "layer_index": 2,
                    "events": [
                        {
                            "event_type": "relation",
                            "source": [{"canonical_id": "ko:K1", "aliases": [], "name": "A"}],
                            "action": {"kind": "relation", "relation_class": "PPrel", "subtypes": ["activation"], "reversibility": None},
                            "mediators": [],
                            "target": [{"canonical_id": "ko:K2", "aliases": [], "name": "B"}],
                            "text": "A activates B.",
                        }
                    ],
                },
                {
                    "layer_index": 3,
                    "events": [
                        {
                            "event_type": "relation",
                            "source": [{"canonical_id": "ko:K2", "aliases": [], "name": "B"}],
                            "action": {"kind": "relation", "relation_class": "PPrel", "subtypes": ["inhibition"], "reversibility": None},
                            "mediators": [],
                            "target": [{"canonical_id": "ko:K3", "aliases": [], "name": "C"}],
                            "text": "B inhibits C.",
                        }
                    ],
                },
            ],
        }

    def test_v4_is_strictly_parsed_for_downstream_tasks(self) -> None:
        parsed = parse_pathway_payload(json.dumps(self.payload()))
        self.assertTrue(parsed.json_valid)
        self.assertTrue(parsed.schema_valid)
        self.assertEqual([step.step for step in parsed.steps], [2, 3])
        self.assertEqual(parsed.steps[0].substeps, ("A activates B.",))

    def test_v4_rejects_nonconsecutive_layers(self) -> None:
        payload = self.payload()
        payload["remaining_layers"][1]["layer_index"] = 4  # type: ignore[index]
        parsed = parse_pathway_payload(payload)
        self.assertFalse(parsed.schema_valid)

    def test_v4_rejects_extra_model_output_fields(self) -> None:
        payload = self.payload()
        payload["organism"] = "hsa"
        parsed = parse_pathway_payload(payload)
        self.assertFalse(parsed.schema_valid)


if __name__ == "__main__":
    unittest.main()
