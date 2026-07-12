"""Compatibility entry point for the maintained batch-capable inference path.

``method.inference.pathway`` already performs batched greedy generation.  The
old duplicate implementation selected nine columns before writing, which
dropped stable sample identity, organism, block, phenotype status, and source
provenance.  Keeping this filename as a thin entry point prevents that data
loss without maintaining a second inference contract.
"""

from method.inference.pathway import parse_args, run_inference


if __name__ == "__main__":
    run_inference(parse_args())
