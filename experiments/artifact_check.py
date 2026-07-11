"""Verify that reused experiment artifacts exist without retraining them."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", dest="paths", action="append", required=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="Accepted for uniform experiment-wrapper passthrough; artifact paths are already seed-scoped.",
    )
    args = parser.parse_args()
    missing = [value for value in args.paths if not Path(value).exists()]
    if missing:
        raise FileNotFoundError("missing reused artifacts:\n" + "\n".join(missing))
    for value in args.paths:
        print(f"ok\t{value}")


if __name__ == "__main__":
    main()
