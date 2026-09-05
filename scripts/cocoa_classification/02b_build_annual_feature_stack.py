#!/usr/bin/env python3
"""Combine 2017 DJF and wet-season data into aligned 20-band feature tiles.

DJF feature tiles define the output grid. Intersecting wet-season Sentinel-2
tiles (including retry subdivisions such as ``_s00``) are warped onto that grid
before the ten wet-season indices are calculated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, transform_bounds

from common import (
    ANNUAL_FEATURE_NAMES,
    BAND_NAMES,
    FEATURE_NAMES,
    FLOAT_NODATA,
    S2_NODATA,
    calculate_features,
    log,
    resolve,
    tiled_profile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build aligned DJF + wet annual feature tiles.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--djf-feature-dir", type=Path)
    parser.add_argument("--wet-sentinel-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--scale-factor", type=float, default=10000.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def wet_inventory(paths: List[Path], target_crs: object) -> List[Tuple[Path, Tuple[float, float, float, float]]]:
    inventory = []
    for path in paths:
        with rasterio.open(path) as source:
            if source.count != len(BAND_NAMES) or source.crs is None:
                raise ValueError(f"Expected {len(BAND_NAMES)} georeferenced bands: {path}")
            bounds = source.bounds if source.crs == target_crs else transform_bounds(source.crs, target_crs, *source.bounds, densify_pts=21)
            inventory.append((path, tuple(bounds)))
    return inventory


def main() -> int:
    args = parse_args()
    if args.scale_factor <= 0:
        raise ValueError("--scale-factor must be positive")
    djf_dir = resolve(args.djf_feature_dir or Path(f"data/interim/cocoa_classification/{args.year}/djf/features"))
    wet_dir = resolve(args.wet_sentinel_dir or Path(f"data/raw/sentinel2/wet_{args.year}"))
    output_dir = resolve(args.output_dir or Path(f"data/interim/cocoa_classification/{args.year}/annual/features"))
    djf_tiles = sorted(djf_dir.glob(f"ghana_cocoa_indices_{args.year}_*.tif"))
    wet_tiles = sorted(wet_dir.glob(f"ghana_cocoa_s2_wet_{args.year}*.tif"))
    if not djf_tiles or not wet_tiles:
        raise FileNotFoundError(f"Missing DJF feature or wet Sentinel tiles: {djf_dir}, {wet_dir}")
    with rasterio.open(djf_tiles[0]) as reference:
        target_crs = reference.crs
    inventory = wet_inventory(wet_tiles, target_crs)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Annual alignment: {len(djf_tiles)} DJF grids and {len(wet_tiles)} wet tiles")

    for index, djf_path in enumerate(djf_tiles, 1):
        destination = output_dir / djf_path.name.replace("indices", "annual_indices")
        if destination.exists() and not args.overwrite:
            log(f"[{index}/{len(djf_tiles)}] Exists; skipping {destination.name}")
            continue
        with rasterio.open(djf_path) as djf:
            if djf.count != len(FEATURE_NAMES) or djf.crs is None:
                raise ValueError(f"Unexpected DJF feature stack: {djf_path}")
            candidates = [path for path, bounds in inventory if intersects(tuple(djf.bounds), bounds)]
            if not candidates:
                log(f"[{index}/{len(djf_tiles)}] No wet coverage; writing nodata wet features")
            wet_raw = np.full((len(BAND_NAMES), djf.height, djf.width), S2_NODATA, dtype="uint16")
            for wet_path in candidates:
                with rasterio.open(wet_path) as source, WarpedVRT(
                    source,
                    crs=djf.crs,
                    transform=djf.transform,
                    width=djf.width,
                    height=djf.height,
                    resampling=Resampling.bilinear,
                    nodata=S2_NODATA,
                ) as aligned:
                    values = aligned.read()
                    valid = np.all((values != S2_NODATA) & (values != 0), axis=0)
                    missing = np.any((wet_raw == S2_NODATA) | (wet_raw == 0), axis=0)
                    fill = valid & missing
                    wet_raw[:, fill] = values[:, fill]
            wet_invalid = np.any((wet_raw == S2_NODATA) | (wet_raw == 0), axis=0)
            wet_features = calculate_features(wet_raw.astype("float32") / args.scale_factor)
            wet_features[:, wet_invalid] = np.nan
            profile = tiled_profile(djf, len(ANNUAL_FEATURE_NAMES), "float32", FLOAT_NODATA)
            temporary = destination.with_name(destination.stem + ".part.tif")
            with rasterio.open(temporary, "w", **profile) as output:
                for band, name in enumerate(ANNUAL_FEATURE_NAMES, 1):
                    output.set_band_description(band, name)
                output.update_tags(seasons="DJF,WET", wet_resampling="bilinear", reflectance_scale=args.scale_factor)
                for _, window in djf.block_windows(1):
                    djf_values = djf.read(window=window)
                    row0, row1 = int(window.row_off), int(window.row_off + window.height)
                    col0, col1 = int(window.col_off), int(window.col_off + window.width)
                    wet_values = wet_features[:, row0:row1, col0:col1]
                    combined = np.concatenate((djf_values, wet_values), axis=0)
                    output.write(np.where(np.isfinite(combined), combined, FLOAT_NODATA).astype("float32"), window=window)
            temporary.replace(destination)
        log(f"[{index}/{len(djf_tiles)}] Saved {destination.name} from {len(candidates)} wet tile(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
