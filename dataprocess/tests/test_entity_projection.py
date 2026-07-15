from __future__ import annotations

import copy
import unittest

from dataprocess.entity_projection import project_event, project_record
from dataprocess.prompt_profiles import (
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
)


def _event(source: list[dict[str, str]], target: list[dict[str, str]]) -> dict:
    source = [{**entity, "aliases": entity.get("aliases", [])} for entity in source]
    target = [{**entity, "aliases": entity.get("aliases", [])} for entity in target]
    return {
        "event_type": "relation",
        "source": source,
        "action": {
            "kind": "relation",
            "relation_class": "PPrel",
            "subtypes": ["activation"],
            "reversibility": None,
        },
        "mediators": [],
        "target": target,
        "text": "A activates B.",
    }


class EntityProjectionTests(unittest.TestCase):
    def test_source_native_profiles_preserve_identical_answer(self) -> None:
        native = _event(
            [{"canonical_id": "hsa:207", "name": "AKT1"}],
            [{"canonical_id": "gene:5594", "name": "MAPK1"}],
        )
        original = copy.deepcopy(native)

        explicit = project_event(
            native,
            organism="hsa",
            profile=EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
        )
        implicit = project_event(
            native,
            organism="hsa",
            profile=NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
        )

        self.assertTrue(explicit.eligible)
        self.assertTrue(implicit.eligible)
        self.assertEqual(explicit.projected, implicit.projected)
        self.assertEqual(explicit.projected, native)
        self.assertIsNot(explicit.projected, native)
        self.assertIsNot(explicit.projected["source"], native["source"])
        self.assertEqual(native, original)

    def test_neutral_kegg_entities_are_identity_only_neutralized(self) -> None:
        neutral = _event(
            [
                {"canonical_id": "ko:K00844", "name": "hexokinase"},
                {"canonical_id": "cpd:C00031", "name": "D-glucose"},
                {"canonical_id": "glycan:G00001", "name": "N-glycan"},
            ],
            [
                {"canonical_id": "gl:G00002", "name": "glycan product"},
                {"canonical_id": "rn:R01786", "name": "hexose reaction"},
                {"canonical_id": "ec:2.7.1.1", "name": "hexokinase activity"},
            ],
        )
        result = project_event(
            neutral,
            organism="hsa",
            profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM,
        )
        self.assertTrue(result.eligible)
        self.assertNotEqual(result.projected, neutral)
        projected_entities = result.projected["source"] + result.projected["target"]
        self.assertTrue(
            all(entity["name"] == entity["canonical_id"] for entity in projected_entities)
        )
        self.assertNotIn("hexokinase", result.projected["text"])
        self.assertIn("ko:K00844", result.projected["text"])
        self.assertEqual(dict(result.rejection_reason_counts), {})

    def test_organism_and_source_native_namespaces_are_rejected(self) -> None:
        cases = (
            ("hsa:207", "organism_specific_namespace"),
            ("mmu:11651", "organism_specific_namespace"),
            ("gene:207", "source_native_namespace"),
        )
        for canonical_id, reason in cases:
            with self.subTest(canonical_id=canonical_id):
                organism = canonical_id.split(":", 1)[0]
                if organism == "gene":
                    organism = "hsa"
                result = project_event(
                    _event(
                        [{"canonical_id": canonical_id, "name": "entity"}],
                        [{"canonical_id": "ko:K00001", "name": "product"}],
                    ),
                    organism=organism,
                    profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM,
                )
                self.assertFalse(result.eligible)
                self.assertIsNone(result.projected)
                self.assertEqual(result.rejection_reason_counts[reason], 1)

    def test_removing_an_organism_prefix_does_not_create_neutral_id(self) -> None:
        result = project_event(
            _event(
                [{"canonical_id": "207", "name": "AKT1"}],
                [{"canonical_id": "ko:K00001", "name": "product"}],
            ),
            organism="hsa",
            profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM,
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.rejection_reason_counts["missing_namespace"], 1)

    def test_source_native_names_are_removed_after_neutral_id_eligibility(self) -> None:
        for name in (
            "AKT1 (hsa:207)",
            "hsa AKT1",
            "AKT1 gene:207",
            "AKT1 mmu:11651",
        ):
            with self.subTest(name=name):
                result = project_event(
                    _event(
                        [{"canonical_id": "ko:K04456", "name": name}],
                        [{"canonical_id": "ko:K00001", "name": "product"}],
                    ),
                    organism="hsa",
                    profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM,
                )
                self.assertTrue(result.eligible)
                self.assertNotIn(name, str(result.projected))
                self.assertEqual(result.projected["source"][0]["name"], "ko:K04456")

    def test_pathway_internal_and_unknown_namespaces_are_rejected(self) -> None:
        cases = (
            ("path:hsa04010", "pathway_namespace"),
            ("node:17", "internal_namespace"),
            ("foo:1", "unknown_namespace"),
        )
        for canonical_id, reason in cases:
            with self.subTest(canonical_id=canonical_id):
                result = project_event(
                    _event(
                        [{"canonical_id": canonical_id, "name": "entity"}],
                        [{"canonical_id": "ko:K00001", "name": "product"}],
                    ),
                    organism="hsa",
                    profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM,
                )
                self.assertFalse(result.eligible)
                self.assertEqual(result.rejection_reason_counts[reason], 1)

    def test_one_bad_entity_rejects_a_multi_entity_multi_layer_record(self) -> None:
        good = _event(
            [
                {"canonical_id": "ko:K00001", "name": "enzyme one"},
                {"canonical_id": "cpd:C00001", "name": "water"},
            ],
            [{"canonical_id": "rn:R00001", "name": "reaction one"}],
        )
        bad = _event(
            [{"canonical_id": "ko:K00002", "name": "enzyme two"}],
            [{"canonical_id": "hsa:207", "name": "AKT1"}],
        )
        record = {
            "organism": "hsa",
            "observed_layers": [{"layer_index": 0, "events": [good]}],
            "remaining_layers": [{"layer_index": 1, "events": [bad]}],
        }
        original = copy.deepcopy(record)

        first = project_record(
            record, profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM
        )
        second = project_record(
            record, profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM
        )

        self.assertFalse(first.eligible)
        self.assertIsNone(first.projected)
        self.assertEqual(
            dict(first.rejection_reason_counts),
            {"organism_specific_namespace": 1},
        )
        self.assertEqual(
            dict(first.rejection_reason_counts),
            dict(second.rejection_reason_counts),
        )
        self.assertEqual(record, original)

    def test_malformed_event_in_any_layer_fails_closed(self) -> None:
        record = {
            "organism": "hsa",
            "layers": [
                {
                    "layer_index": 0,
                    "events": [
                        _event(
                            [{"canonical_id": "ko:K00001", "name": "enzyme"}],
                            [{"canonical_id": "cpd:C00001", "name": "water"}],
                        )
                    ],
                },
                {"layer_index": 1, "events": ["not-an-event"]},
            ],
        }
        result = project_record(
            record, profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.rejection_reason_counts["malformed_event"], 1)


if __name__ == "__main__":
    unittest.main()
