from __future__ import annotations

import csv
import json
import stat
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dataprocess.audit_dataset_release import (
    V3_CSV_FIELDNAMES,
    file_sha256,
    generate_release_audit,
)
from dataprocess.build_structured_dataset import write_profile_control_csv
from dataprocess.prompt_profiles import (
    EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
    SPECIES_NEUTRAL_IDS_NO_ORGANISM,
)
from dataprocess.release_contract import RELEASE_SCHEMA_VERSION
from dataprocess.source_hashes import write_source_graph_hashes
from dataprocess.structured_schema import csv_row, graph_id_for_source
from dataprocess.structured_views import build_structured_records


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(value) for value in text]


def _node(
    node_id: int,
    organism: str,
    label: str,
    *,
    species_neutral: bool,
) -> dict[str, object]:
    canonical_id = (
        f"ko:K{node_id:05d}" if species_neutral else f"{organism}:{node_id}"
    )
    return {
        "node_id": node_id,
        "entry_id": node_id,
        "node_kind": "entry",
        "entity_type": "gene",
        "canonical_id": canonical_id,
        "display_name": (
            f"neutral enzyme K{node_id:05d}"
            if species_neutral
            else f"{label}{node_id}"
        ),
        "resolved_ids": [canonical_id],
        "raw_name": canonical_id,
        "aliases": [],
        "unresolved_tokens": [],
        "component_entry_ids": [],
        "resolved": True,
    }


def _relation(relation_id: int, source: int, target: int) -> dict[str, object]:
    return {
        "relation_id": relation_id,
        "entry1_id": source,
        "entry2_id": target,
        "relation_type": "PPrel",
        "subtypes": [{"name": "activation", "value": "-->"}],
        "subtype_names": ["activation"],
        "semantic_tags": ["activation"],
        "mediator_entry_id": None,
        "has_missing_interaction": False,
        "renderable": True,
    }


def _horizon(prefix: int, layers: int) -> str:
    if prefix == 1:
        return "long_target"
    if prefix == layers - 1:
        return "short_target"
    return "middle_target"


@unittest.skip("historical v3.1 release auditor; formal v4 audit has independent coverage")
class ReleaseAuditV31Tests(unittest.TestCase):
    def _record(
        self,
        graph_root: Path,
        *,
        organism: str,
        family: str,
        source_name: str,
        label: str,
        species_neutral: bool = False,
    ):
        source = f"{organism}/{source_name}.json"
        graph = {
            "metadata": {
                "organism": organism,
                "pathway_id": f"path:{organism}{family}",
                "title": f"metadata title {label}",
            },
            "nodes": [
                _node(
                    index,
                    organism,
                    label,
                    species_neutral=species_neutral,
                )
                for index in range(1, 6)
            ],
            "relations": [
                _relation(index, index + 1, index + 2) for index in range(4)
            ],
            "reactions": [],
        }
        raw = (
            json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        artifact = graph_root / source
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(raw)
        records = build_structured_records(
            graph,
            graph_id=graph_id_for_source(source, raw),
            source_graph_json=source,
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0].layers), 4)
        return records[0]

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=V3_CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)

    def _fixture(self, *, validation_family: str = "00020") -> tuple[tempfile.TemporaryDirectory, dict]:
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        graph_root = root / "processed_graph"
        definitions = {
            "train": ("aaa", "00010", "train_unique", "tr"),
            "validation": ("aaa", validation_family, "validation_unique", "va"),
            "test": ("bbb", "00030", "test_unique", "te"),
            "test_family_only": ("aaa", "00030", "family_only_unique", "fo"),
            "test_organism_only": ("bbb", "00010", "organism_only_unique", "oo"),
        }
        records = {
            partition: [
                self._record(
                    graph_root,
                    organism=organism,
                    family=family,
                    source_name=f"{source_name}_native",
                    label=label,
                ),
                self._record(
                    graph_root,
                    organism=organism,
                    family=family,
                    source_name=f"{source_name}_neutral",
                    label=f"neutral_{label}",
                    species_neutral=True,
                ),
            ]
            for partition, (organism, family, source_name, label) in definitions.items()
        }

        primary_paths: dict[str, Path] = {}
        primary_rows: dict[str, list[dict[str, object]]] = {}
        record_paths: dict[str, Path] = {}
        for partition, partition_records in records.items():
            if partition == "validation":
                prefix_plan = ((partition_records[0], (1,)), (partition_records[1], (2,)))
            else:
                prefix_plan = tuple((record, (1, 2, 3)) for record in partition_records)
            primary_rows[partition] = [
                csv_row(
                    record,
                    prefix,
                    prompt_profile=EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
                    prefix_horizon=_horizon(prefix, len(record.layers)),
                    split=partition,
                )
                for record, prefixes in prefix_plan
                for prefix in prefixes
            ]
            primary_paths[partition] = root / f"{partition}.csv"
            self._write_csv(primary_paths[partition], primary_rows[partition])
            record_paths[partition] = root / f"{partition}_pathway_records_v3.jsonl"
            record_paths[partition].write_text(
                "".join(
                    json.dumps(record.record_object(), sort_keys=True) + "\n"
                    for record in partition_records
                ),
                encoding="utf-8",
            )

        control_paths: dict[str, dict[str, Path]] = {
            profile: {} for profile in (
                NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
                SPECIES_NEUTRAL_IDS_NO_ORGANISM,
            )
        }
        control_outputs: dict[str, dict[str, dict[str, object]]] = {
            profile: {} for profile in control_paths
        }
        for profile in control_paths:
            for partition in definitions:
                control_path = root / f"{partition}.{profile}.csv"
                control_paths[profile][partition] = control_path
                result = write_profile_control_csv(
                    primary_csv_path=primary_paths[partition],
                    record_path=record_paths[partition],
                    output_path=control_path,
                    prompt_profile=profile,
                    split=partition,
                    tokenizer=CharacterTokenizer(),
                    max_length=8192,
                )
                control_outputs[profile][partition] = {
                    **result,
                    "path": control_path.name,
                }

        inventory_path = root / "source_graph_hashes.jsonl"
        source_hash_metadata = write_source_graph_hashes(
            graph_root,
            [
                record.source_graph_json
                for partition_records in records.values()
                for record in partition_records
            ],
            inventory_path,
            overwrite=False,
        )
        split_manifest = {}
        for partition, rows in primary_rows.items():
            split_manifest[partition] = {
                "csv_sha256": file_sha256(primary_paths[partition]),
                "records_sha256": file_sha256(record_paths[partition]),
                "rows": len(rows),
                "records": 2,
                "sources": 2,
                "graphs": 2,
                "families": 1,
                "organisms": 1,
                "rows_dropped_token_budget": 0,
                "prompt_profile": EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
                "prompt_profile_interface_applied": True,
                "prefix_horizon_interface_applied": True,
                "prefix_horizons": dict(
                    sorted(Counter(str(row["prefix_horizon"]) for row in rows).items())
                ),
                "maximum_views_per_graph": 1,
                "maximum_records_in_one_family": 2,
            }
        manifest = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "dataset_build_id": "dataset:0123456789abcdef01234567",
            "csv_header": V3_CSV_FIELDNAMES,
            "inventory": {"graph_files": 10},
            "outputs": {
                **{
                    f"{partition}_records": path.name
                    for partition, path in record_paths.items()
                },
                "source_graph_hashes": inventory_path.name,
            },
            "source_graph_hashes": source_hash_metadata,
            "max_length": 8192,
            "max_records_per_family": 4,
            "primary_prompt_profile": EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS,
            "train_families": ["00010"],
            "validation_families": [validation_family],
            "strict_test_families": ["00030"],
            "test_organisms": ["bbb"],
            "splits": split_manifest,
            "paired_prompt_profiles": {
                "status": "published",
                "published": True,
                "files": control_outputs,
                "profile_contracts": {
                    NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS: {
                        "published": True,
                        "base_sample_contract": "exact_primary_set",
                        "answer_contract": "exact_primary_answer",
                        "species_claim": "no_explicit_name_only_native_ids_can_leak_species",
                    },
                    SPECIES_NEUTRAL_IDS_NO_ORGANISM: {
                        "published": True,
                        "base_sample_contract": "strict_natural_neutral_subset",
                        "answer_contract": "exact_primary_answer_on_shared_base_samples",
                        "mapping_contract": "no_prefix_stripping_or_synthetic_mapping",
                    },
                },
            },
            "prompt_controls": json.loads(json.dumps(control_outputs)),
        }
        manifest_path = root / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return temporary, {
            "root": root,
            "graph_root": graph_root,
            "records": records,
            "primary_paths": primary_paths,
            "control_paths": control_paths,
            "manifest_path": manifest_path,
        }

    def _audit(self, fixture: dict, output_name: str, *, raise_on_failure: bool = True):
        output = fixture["root"] / output_name
        report = generate_release_audit(
            partition_paths=fixture["primary_paths"],
            graph_root=fixture["graph_root"],
            manifest_path=fixture["manifest_path"],
            tokenizer=CharacterTokenizer(),
            max_length=8192,
            output_path=output,
            overwrite=False,
            raise_on_failure=raise_on_failure,
        )
        return report, output

    def test_five_partition_profiles_pairing_and_read_only_audit(self) -> None:
        temporary, fixture = self._fixture()
        self.addCleanup(temporary.cleanup)
        report, output = self._audit(fixture, "data_audit.json")

        self.assertEqual(report["status"], "passed")
        self.assertEqual(set(report["splits"]), set(fixture["primary_paths"]))
        for split_report in report["splits"].values():
            self.assertEqual(
                split_report["prompt_template_valid_rows"], split_report["rows"]
            )

        with fixture["primary_paths"]["train"].open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            train_question = next(csv.DictReader(handle))["question"]
        self.assertEqual(train_question.splitlines().count("Organism (KEGG code): aaa"), 1)
        self.assertNotIn("Pathway title:", train_question)
        self.assertNotIn("Events inside one layer are an unordered set", train_question)

        overlaps = report["required_summary"]["strict_overlap"]
        self.assertTrue(
            overlaps["train_vs_validation"]["biological_contract"]["organism"]["passed"]
        )
        self.assertEqual(
            overlaps["train_vs_validation"]["biological_contract"]["organism"]["policy"],
            "allowed",
        )
        self.assertTrue(
            overlaps["train_vs_test_organism_only"]["biological_contract"]["family"]["passed"]
        )
        self.assertTrue(
            overlaps["test_vs_test_family_only"]["biological_contract"]["family"]["passed"]
        )
        self.assertTrue(
            overlaps["train_vs_test"]["biological_contract"]["family"]["passed"]
        )

        paired = report["paired_prompt_profiles"]
        self.assertEqual(paired["status"], "passed")
        self.assertTrue(paired["manifest_published"])
        self.assertTrue(paired["canonical_files_match_prompt_controls"])
        self.assertTrue(
            all(value["passed"] for value in paired["profile_contracts"].values())
        )
        p0_p1_checks = [
            value
            for key, value in paired["pair_checks"].items()
            if key.endswith(
                ":"
                + EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS
                + "_vs_"
                + NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS
            )
        ]
        self.assertEqual(len(p0_p1_checks), 5)
        self.assertTrue(all(value["passed"] for value in p0_p1_checks))
        self.assertTrue(
            all(
                value["base_sample_policy"] == "exact_primary_set"
                for value in p0_p1_checks
            )
        )
        p0_p2_checks = [
            value
            for key, value in paired["pair_checks"].items()
            if key.endswith(
                ":"
                + EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS
                + "_vs_"
                + SPECIES_NEUTRAL_IDS_NO_ORGANISM
            )
        ]
        self.assertEqual(len(p0_p2_checks), 5)
        self.assertTrue(all(value["passed"] for value in p0_p2_checks))
        self.assertTrue(
            all(
                value["base_sample_policy"] == "strict_natural_neutral_subset"
                for value in p0_p2_checks
            )
        )
        self.assertTrue(
            all(
                value["right_base_samples"] < value["left_base_samples"]
                for value in p0_p2_checks
            )
        )
        self.assertEqual(
            paired["species_neutral_eligibility_from_primary"]["train"][
                "eligible_base_samples"
            ],
            3,
        )
        self.assertGreater(
            paired["species_neutral_eligibility_from_primary"]["train"][
                "rejection_reason_counts"
            ]["organism_specific_namespace"],
            0,
        )
        self.assertEqual(
            report["splits"]["train"]["organism_distribution"]["aaa"],
            {"rows": 6, "records": 2},
        )

        p2_train_path = fixture["control_paths"][
            SPECIES_NEUTRAL_IDS_NO_ORGANISM
        ]["train"]
        with p2_train_path.open("r", encoding="utf-8", newline="") as handle:
            p2_question = next(csv.DictReader(handle))["question"]
        self.assertNotIn("Organism (KEGG code):", p2_question)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)

    def test_forbidden_family_overlap_fails_the_fixed_contract(self) -> None:
        temporary, fixture = self._fixture(validation_family="00010")
        self.addCleanup(temporary.cleanup)
        report, output = self._audit(
            fixture,
            "failed_overlap_audit.json",
            raise_on_failure=False,
        )
        self.assertEqual(report["status"], "failed")
        self.assertIn(
            "train_vs_validation:family_forbidden_contract_failed",
            report["strict_failures"],
        )
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)

    def test_source_graph_hash_mutation_is_detected_and_failure_is_written(self) -> None:
        temporary, fixture = self._fixture()
        self.addCleanup(temporary.cleanup)
        record = fixture["records"]["train"][0]
        artifact = fixture["graph_root"] / record.source_graph_json
        value = json.loads(artifact.read_text(encoding="utf-8"))
        value["metadata"]["title"] = "mutated after hash inventory"
        artifact.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")

        report, output = self._audit(
            fixture,
            "failed_hash_audit.json",
            raise_on_failure=False,
        )
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["source_graph_hashes"]["status"], "failed")
        self.assertTrue(report["source_graph_hashes"]["errors"])
        self.assertIn(
            "source_graph_hashes:verification_failed",
            report["strict_failures"],
        )
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)

    def test_prompt_controls_compatibility_manifest_must_match_canonical_files(self) -> None:
        temporary, fixture = self._fixture()
        self.addCleanup(temporary.cleanup)
        manifest = json.loads(fixture["manifest_path"].read_text(encoding="utf-8"))
        manifest["prompt_controls"][NO_EXPLICIT_ORGANISM_SOURCE_NATIVE_IDS][
            "train"
        ]["rows"] += 1
        fixture["manifest_path"].write_text(json.dumps(manifest), encoding="utf-8")

        report, output = self._audit(
            fixture,
            "failed_prompt_control_manifest_audit.json",
            raise_on_failure=False,
        )
        self.assertEqual(report["status"], "failed")
        self.assertFalse(
            report["paired_prompt_profiles"][
                "canonical_files_match_prompt_controls"
            ]
        )
        self.assertIn(
            "paired_profiles:prompt_controls_manifest_mismatch",
            report["strict_failures"],
        )
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)


if __name__ == "__main__":
    unittest.main()
