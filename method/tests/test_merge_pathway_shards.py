from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from method.inference.csv_io import read_csv_text_rows
from method.inference.merge_pathway_shards import file_sha256, merge_shards


class MergePathwayShardsTests(unittest.TestCase):
    def _write_csv(self, path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_merge_restores_input_order_and_verifies_complete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.csv"
            source_fields = [
                "sample_id",
                "base_sample_id",
                "question",
                "answer",
                "pathway_family_id",
                "prompt_profile",
            ]
            source_rows = [
                {
                    "sample_id": f"s{index}",
                    "base_sample_id": f"base{index}",
                    "question": f"q{index}",
                    "answer": f"a{index}",
                    "pathway_family_id": "00051",
                    "prompt_profile": "explicit_organism_source_native_ids",
                }
                for index in range(5)
            ]
            self._write_csv(source, source_fields, source_rows)
            output_fields = [
                "dataset_index",
                *source_fields,
                "predicted_answer",
                "generated_token_count",
                "total_generated_token_count",
                "finish_reason",
                "generation_attempts",
                "prediction_json_valid",
                "prediction_schema_valid",
            ]
            shard_outputs: list[Path] = []
            shard_progress: list[Path] = []
            for shard_index in range(2):
                output = root / f"direct.shard{shard_index}.csv"
                progress = root / f"direct.shard{shard_index}.progress.jsonl"
                indices = list(range(shard_index, len(source_rows), 2))
                rows = [
                    {
                        "dataset_index": index,
                        **source_rows[index],
                        "predicted_answer": f"p{index}",
                        "generated_token_count": 10 + index,
                        "total_generated_token_count": 10 + index,
                        "finish_reason": "eos",
                        "generation_attempts": 1,
                        "prediction_json_valid": True,
                        "prediction_schema_valid": True,
                    }
                    for index in reversed(indices)
                ]
                self._write_csv(output, output_fields, rows)
                progress.write_text(
                    "".join(
                        json.dumps(
                            {
                                "sample_index": index,
                                "sample_id": source_rows[index]["sample_id"],
                                "base_sample_id": source_rows[index]["base_sample_id"],
                                "record_id": "",
                                "organism": "",
                                "pathway_family_id": source_rows[index]["pathway_family_id"],
                                "prompt_profile": source_rows[index]["prompt_profile"],
                                "gold_answer": source_rows[index]["answer"],
                                "predicted_answer": f"p{index}",
                                "generated_token_count": 10 + index,
                                "total_generated_token_count": 10 + index,
                                "finish_reason": "eos",
                                "generation_attempts": 1,
                                "prediction_json_valid": True,
                                "prediction_schema_valid": True,
                                "status": "completed",
                            }
                        )
                        + "\n"
                        for index in reversed(indices)
                    ),
                    encoding="utf-8",
                )
                manifest = {
                    "base_model_id": "/model",
                    "trained_lora_path": "/adapter",
                    "test_data_path": str(source),
                    "batch_size": 1,
                    "max_length": 8192,
                    "max_new_tokens": 1024,
                    "max_json_attempts": 3,
                    "retry_max_new_tokens": 8192,
                    "limit": None,
                    "shard_count": 2,
                    "shard_index": shard_index,
                    "seed": 7,
                    "completion_marker": "/complete.json",
                    "git_commit": "abc123",
                    "input_sha256": file_sha256(source),
                    "completion_marker_sha256": "marker-hash",
                    "input_rows": len(source_rows),
                    "evaluated_rows": len(rows),
                    "progress_output_sha256": file_sha256(progress),
                }
                output.with_suffix(".run.json").write_text(
                    json.dumps(manifest),
                    encoding="utf-8",
                )
                shard_outputs.append(output)
                shard_progress.append(progress)

            destination = root / "direct.csv"
            progress_destination = root / "direct.progress.jsonl"
            metadata = merge_shards(
                input_path=source,
                shard_outputs=shard_outputs,
                shard_progress=shard_progress,
                output_path=destination,
                progress_output_path=progress_destination,
            )

            _, merged = read_csv_text_rows(destination)
            self.assertEqual([row["dataset_index"] for row in merged], ["0", "1", "2", "3", "4"])
            self.assertEqual([row["predicted_answer"] for row in merged], ["p0", "p1", "p2", "p3", "p4"])
            progress_rows = [json.loads(line) for line in progress_destination.read_text().splitlines()]
            self.assertEqual([row["sample_index"] for row in progress_rows], list(range(5)))
            self.assertEqual(metadata["input_rows"], 5)
            self.assertEqual(metadata["prediction_schema_valid_count"], 5)

    def test_merge_rejects_incomplete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input.csv"
            source.write_text("sample_id,question,answer\ns0,q0,a0\ns1,q1,a1\n", encoding="utf-8")
            output = root / "shard.csv"
            self._write_csv(
                output,
                [
                    "dataset_index", "sample_id", "question", "answer", "predicted_answer",
                    "generated_token_count", "finish_reason", "prediction_json_valid",
                    "prediction_schema_valid", "total_generated_token_count", "generation_attempts",
                ],
                [{
                    "dataset_index": 0, "sample_id": "s0", "question": "q0", "answer": "a0",
                    "predicted_answer": "p0", "generated_token_count": 1, "finish_reason": "eos",
                    "total_generated_token_count": 1, "generation_attempts": 1,
                    "prediction_json_valid": True, "prediction_schema_valid": True,
                }],
            )
            progress = root / "shard.progress.jsonl"
            progress.write_text('{"sample_index":0}\n', encoding="utf-8")
            output.with_suffix(".run.json").write_text(
                json.dumps({
                    "shard_count": 1,
                    "shard_index": 0,
                    "input_rows": 2,
                    "evaluated_rows": 1,
                    "progress_output_sha256": file_sha256(progress),
                }),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "incomplete shard coverage"):
                merge_shards(
                    input_path=source,
                    shard_outputs=[output],
                    shard_progress=[progress],
                    output_path=root / "merged.csv",
                    progress_output_path=root / "merged.progress.jsonl",
                )


if __name__ == "__main__":
    unittest.main()
