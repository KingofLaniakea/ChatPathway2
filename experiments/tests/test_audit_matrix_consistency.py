from __future__ import annotations

import unittest

from experiments.audit_matrix_consistency import declared_cli_options


class DeclaredCliOptionsTests(unittest.TestCase):
    def test_ddp_entrypoint_inherits_canonical_framework_parser(self) -> None:
        options = declared_cli_options("method.training.framework_a_ddp")

        for option in (
            "--base-model",
            "--sft-lora",
            "--ae-ckpt",
            "--train",
            "--save-dir",
            "--variant",
            "--gradient-accumulation-steps",
            "--validation-group-column",
        ):
            self.assertIn(option, options)

    def test_missing_module_has_no_declared_options(self) -> None:
        self.assertEqual(declared_cli_options("missing.module"), {})


if __name__ == "__main__":
    unittest.main()
