#!/usr/bin/env python3
"""Report whether the expected regional input directories contain data."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ghana_cocoa_lulc.config import load_project_config  # noqa: E402


def main() -> int:
    config = load_project_config()
    regions = config["project"]["regions"]
    input_dirs = {key: value for key, value in config["inputs"].items() if key.endswith("_dir")}
    missing: list[str] = []

    for dataset, relative_dir in input_dirs.items():
        for region in regions:
            folder = ROOT / relative_dir / region
            files = [path for path in folder.glob("*") if path.name != ".gitkeep"]
            status = "OK" if files else "MISSING"
            print(f"{status:7} {dataset.removesuffix('_dir'):12} {region:14} {folder}")
            if not files:
                missing.append(str(folder))

    if missing:
        print(f"\nInput validation incomplete: {len(missing)} dataset/region folders are empty.")
        return 1
    print("\nAll expected input folders contain at least one file.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

