from __future__ import annotations

import os
import tempfile
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

    def test_parser_audit_is_independent_of_current_working_directory(self) -> None:
        previous = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                os.chdir(temporary)
                options = declared_cli_options("method.training.framework_a_ddp")
        finally:
            os.chdir(previous)
        self.assertIn("--base-model", options)


if __name__ == "__main__":
    unittest.main()
