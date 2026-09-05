#!/usr/bin/env python3
"""Align Dynamic World class 1 (trees) to every Sentinel-2 tile."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling

from common import log, paired_name, resolve, tiled_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create tiled 10 m Dynamic World tree masks.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--season", choices=("djf", "amj"), default="djf")
    parser.add_argument("--dw-raster", type=Path)
    parser.add_argument("--sentinel-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tree-class", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dw_path = resolve(args.dw_raster or Path(f"data/processed/dynamicworld/ghana_cocoa_dynamicworld_{args.year}_mode_clipped.tif"))
    sentinel_dir = resolve(args.sentinel_dir or Path(f"data/raw/sentinel2/{args.season}_{args.year}"))
    output_dir = resolve(args.output_dir or Path(f"data/interim/cocoa_classification/{args.year}/{args.season}/tree_masks"))
    tiles = sorted(sentinel_dir.glob(f"ghana_cocoa_s2_{args.season}_{args.year}_*.tif"))
    if not dw_path.exists() or not tiles:
        raise FileNotFoundError(f"Missing Dynamic World raster or Sentinel tiles: {dw_path}, {sentinel_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dw_path) as dw:
        if dw.count != 1 or dw.crs is None:
            raise ValueError("Dynamic World input must be a one-band georeferenced raster")
        for index, tile in enumerate(tiles, 1):
            destination = output_dir / paired_name(tile, f"s2_{args.season}", "dw_tree")
            if destination.exists() and not args.overwrite:
                log(f"[{index}/{len(tiles)}] Exists; skipping {destination.name}")
                continue
            with rasterio.open(tile) as reference:
                if reference.crs is None:
                    raise ValueError(f"Sentinel tile has no CRS: {tile}")
                profile = tiled_profile(reference, 1, "uint8", 255)
                temporary = destination.with_name(destination.stem + ".part.tif")
                with WarpedVRT(
                    dw, crs=reference.crs, transform=reference.transform,
                    width=reference.width, height=reference.height,
                    resampling=Resampling.nearest, nodata=dw.nodata,
                ) as aligned, rasterio.open(temporary, "w", **profile) as output:
                    output.set_band_description(1, "dw_tree_mask")
                    for _, window in reference.block_windows(1):
                        values = aligned.read(1, window=window, masked=True)
                        mask = np.where(np.ma.getmaskarray(values), 255, (values.data == args.tree_class).astype("uint8"))
                        output.write(mask, 1, window=window)
                temporary.replace(destination)
            log(f"[{index}/{len(tiles)}] Saved {destination.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

