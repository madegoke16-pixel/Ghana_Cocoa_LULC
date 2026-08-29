#!/usr/bin/env python3
"""Starter entry point for boundary preparation.

Implement dataset-specific filename and attribute mappings after source data have
been selected and documented.
"""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    raise SystemExit(
        "Add boundary source files under data/raw/boundaries/<region>/, then "
        "configure their filenames and regional-name fields before processing."
    )


if __name__ == "__main__":
    main()

