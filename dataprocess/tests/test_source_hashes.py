from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataprocess.source_hashes import (
    verify_source_graph_hashes,
    write_source_graph_hashes,
)


class SourceHashTests(unittest.TestCase):
    def test_inventory_is_sorted_complete_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_root = root / "graphs"
            for relative, payload in (("b/b.json", b"b\n"), ("a/a.json", b"a\n")):
                path = graph_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            inventory = root / "source_graph_hashes.jsonl"
            metadata = write_source_graph_hashes(
                graph_root,
                ["b/b.json", "a/a.json", "a/a.json"],
                inventory,
                overwrite=False,
            )
            self.assertEqual(metadata["records"], 2)
            values = [json.loads(line) for line in inventory.read_text().splitlines()]
            self.assertEqual(
                [value["source_graph_json"] for value in values],
                ["a/a.json", "b/b.json"],
            )
            report = verify_source_graph_hashes(
                graph_root,
                inventory,
                expected_sources={"a/a.json", "b/b.json"},
            )
            self.assertEqual(report["errors"], [])

            (graph_root / "a/a.json").write_text("changed\n", encoding="utf-8")
            changed = verify_source_graph_hashes(graph_root, inventory)
            self.assertTrue(changed["errors"])


if __name__ == "__main__":
    unittest.main()
