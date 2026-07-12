from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.check_runtime_assets import (
    RUNTIME_MANIFEST_PATH,
    check_output_parents,
    load_json,
)
from experiments.audit_matrix_consistency import remap_manifest_rows


class RuntimeAssetOutputParentTests(unittest.TestCase):
    def test_consistency_manifest_paths_follow_runtime_profile(self) -> None:
        rows = {
            "base": {
                "train_requires": ["/root/autodl-tmp/data/train.csv"],
                "train_outputs": ["/root/autodl-tmp/checkpoints/seeds/20260711/base"],
            },
            "dependent": {
                "train_requires": ["base", "/root/autodl-tmp/models/model"],
            },
        }

        remapped = remap_manifest_rows(
            rows,
            manifest_root="/root/autodl-tmp",
            target_root="/cfff/assets",
        )

        self.assertEqual(remapped["dependent"]["train_requires"][0], "base")
        self.assertEqual(
            remapped["dependent"]["train_requires"][1],
            "/cfff/assets/models/model",
        )
        self.assertEqual(
            remapped["base"]["train_outputs"][0],
            "/cfff/assets/checkpoints/seeds/20260711/base",
        )

    def test_create_output_dirs_does_not_precreate_ae_checkpoint(self) -> None:
        manifest = load_json(RUNTIME_MANIFEST_PATH)
        entry = manifest["rows"]["base000_shared_sft_reconae"]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = check_output_parents(
                experiment_id="base000_shared_sft_reconae",
                phase="train",
                entry=entry,
                manifest_root=manifest["asset_root"],
                asset_root=str(root),
                create_output_dirs=True,
            )

            ae_root = root / "checkpoints/seeds/20260711/shared/pathway_reconstruction_ae"
            self.assertTrue(ae_root.is_dir())
            self.assertFalse((ae_root / "checkpoint_best").exists())
            self.assertTrue(all(record["ok"] for record in records))


if __name__ == "__main__":
    unittest.main()
