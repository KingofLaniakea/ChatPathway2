from __future__ import annotations

import json
import unittest

from dataprocess.prompt_profiles import (
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    PROMPT_PROFILE_METADATA,
    PROMPT_PROFILE_NAMES,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
    render_pathway_question,
)


def observed_payload(canonical_id: str) -> dict[str, object]:
    return {
        "observed_layers": [
            {
                "layer_index": 0,
                "events": [
                    {
                        "source": [{"canonical_id": canonical_id, "name": "A"}],
                        "relation": "activation",
                        "target": [{"canonical_id": "ko:K00002", "name": "B"}],
                        "text": "A activates B.",
                    }
                ],
            }
        ]
    }


class PromptProfileTests(unittest.TestCase):
    def test_profile_names_and_metadata_are_exact(self) -> None:
        self.assertEqual(
            PROMPT_PROFILE_NAMES,
            (
                "explicit_organism_source_native_ids",
                "no_explicit_organism_source_native_ids",
                "species_neutral_ids_no_organism",
            ),
        )
        self.assertEqual(
            PROMPT_PROFILE_METADATA[EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS],
            {
                "organism_conditioning": "explicit",
                "entity_id_space": "source_native",
                "entity_mapping_status": "not_applicable",
            },
        )
        self.assertEqual(
            PROMPT_PROFILE_METADATA[NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS],
            {
                "organism_conditioning": "implicit_in_source_native_ids",
                "entity_id_space": "source_native",
                "entity_mapping_status": "not_applicable",
            },
        )
        self.assertEqual(
            PROMPT_PROFILE_METADATA[SPECIES_NEUTRAL_IDS_NO_ORGANISM],
            {
                "organism_conditioning": "absent_after_neutralization",
                "entity_id_space": "species_neutral_kegg",
                "entity_mapping_status": "complete",
            },
        )

    def test_explicit_profile_renders_organism_and_complete_multiline_shape(self) -> None:
        question = render_pathway_question(
            observed_payload("hsa:1"),
            next_layer_index=3,
            organism="hsa",
            profile=EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
        )

        self.assertIn("Organism (KEGG code): hsa", question)
        self.assertIn("first remaining layer must use layer_index 3", question)
        self.assertIn("Do not use Markdown", question)
        self.assertIn("Do not add extra keys", question)
        self.assertNotIn("Events inside one layer are an unordered set", question)
        for forbidden in (
            "KEGG pathway ID:",
            '"pathway_id"',
            '"pathway_family_id"',
            '"pathway_title"',
            '"pathway_block"',
            '"phenotype"',
        ):
            self.assertNotIn(forbidden, question)

        skeleton = question.split("Required output JSON format:\n", 1)[1].split(
            "\n\nObserved prefix:\n", 1
        )[0]
        shape = json.loads(skeleton)
        self.assertEqual(set(shape), {"schema_version", "remaining_layers"})
        self.assertEqual(shape["schema_version"], "pathway_continuation_v3")
        layer = shape["remaining_layers"][0]
        self.assertEqual(set(layer), {"layer_index", "events"})
        self.assertEqual(layer["layer_index"], 3)
        event = layer["events"][0]
        self.assertEqual(set(event), {"source", "relation", "target", "text"})
        self.assertEqual(set(event["source"][0]), {"canonical_id", "name"})
        self.assertEqual(set(event["target"][0]), {"canonical_id", "name"})

    def test_no_explicit_profile_hides_name_but_preserves_native_ids(self) -> None:
        question = render_pathway_question(
            observed_payload("hsa:1"),
            next_layer_index=1,
            organism="hsa",
            profile=NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
        )

        self.assertNotIn("Organism (KEGG code):", question)
        self.assertIn('"canonical_id":"hsa:1"', question)

    def test_species_neutral_profile_hides_organism_and_accepts_neutral_ids(self) -> None:
        question = render_pathway_question(
            observed_payload("ko:K00001"),
            next_layer_index=1,
            organism="hsa",
            profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM,
        )

        self.assertNotIn("Organism (KEGG code):", question)
        self.assertNotIn("hsa:", question)
        self.assertIn('"canonical_id":"ko:K00001"', question)

    def test_species_neutral_profile_fails_closed_on_organism_prefixed_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "organism-prefixed IDs"):
            render_pathway_question(
                observed_payload("hsa:1"),
                next_layer_index=1,
                organism="hsa",
                profile=SPECIES_NEUTRAL_IDS_NO_ORGANISM,
            )

    def test_model_metadata_and_invalid_arguments_fail_closed(self) -> None:
        payload = observed_payload("ko:K00001")
        payload["observed_layers"][0]["phenotype"] = "forbidden"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "forbidden model metadata"):
            render_pathway_question(
                payload,
                next_layer_index=1,
                organism="hsa",
                profile=EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
            )
        with self.assertRaisesRegex(ValueError, "unknown prompt profile"):
            render_pathway_question(
                observed_payload("ko:K00001"),
                next_layer_index=1,
                organism="hsa",
                profile="unknown",
            )
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            render_pathway_question(
                observed_payload("ko:K00001"),
                next_layer_index=True,
                organism="hsa",
                profile=EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
            )


if __name__ == "__main__":
    unittest.main()
