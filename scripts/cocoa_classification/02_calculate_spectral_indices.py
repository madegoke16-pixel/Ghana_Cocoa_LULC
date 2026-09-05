#!/usr/bin/env python3
"""Calculate ten vegetation indices for every Sentinel-2 composite tile."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio

from common import BAND_NAMES, FEATURE_NAMES, FLOAT_NODATA, S2_NODATA, calculate_features, log, paired_name, resolve, tiled_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiled Sentinel-2 spectral-index stacks.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument(
        "--season",
        choices=("djf", "wet"),
        default="djf",
        help="Sentinel-2 composite season; wet reads data/raw/sentinel2/wet_<year>.",
    )
    parser.add_argument("--sentinel-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scale-factor", type=float, default=10000.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scale_factor <= 0:
        raise ValueError("--scale-factor must be positive")
    sentinel_dir = resolve(args.sentinel_dir or Path(f"data/raw/sentinel2/{args.season}_{args.year}"))
    output_dir = resolve(args.output_dir or Path(f"data/interim/cocoa_classification/{args.year}/{args.season}/features"))
    tiles = sorted(sentinel_dir.glob(f"ghana_cocoa_s2_{args.season}_{args.year}_*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No Sentinel-2 tiles found in {sentinel_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, tile in enumerate(tiles, 1):
        destination = output_dir / paired_name(tile, f"s2_{args.season}", "indices")
        if destination.exists() and not args.overwrite:
            log(f"[{index}/{len(tiles)}] Exists; skipping {destination.name}")
            continue
        with rasterio.open(tile) as source:
            if source.count != len(BAND_NAMES):
                raise ValueError(f"Expected {len(BAND_NAMES)} bands in {tile}, found {source.count}")
            profile = tiled_profile(source, len(FEATURE_NAMES), "float32", FLOAT_NODATA)
            temporary = destination.with_name(destination.stem + ".part.tif")
            with rasterio.open(temporary, "w", **profile) as output:
                for band, name in enumerate(FEATURE_NAMES, 1):
                    output.set_band_description(band, name)
                output.update_tags(source_bands=",".join(BAND_NAMES), reflectance_scale=args.scale_factor)
                for _, window in source.block_windows(1):
                    raw = source.read(window=window)
                    invalid = np.any((raw == S2_NODATA) | (raw == 0), axis=0)
                    reflectance = raw.astype("float32") / args.scale_factor
                    features = calculate_features(reflectance)
                    features[:, invalid] = np.nan
                    output.write(np.where(np.isfinite(features), features, FLOAT_NODATA).astype("float32"), window=window)
            temporary.replace(destination)
        log(f"[{index}/{len(tiles)}] Saved {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
