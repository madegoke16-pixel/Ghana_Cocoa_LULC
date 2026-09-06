#!/usr/bin/env python3
"""Mosaic yearly Dynamic World tiles and clip the mosaic to the project AOI.

Example:
    python scripts/dynamicworld/mosaic_and_clip_dynamicworld_lulc.py --year 2025

Inputs:
    data/raw/dynamicworld/ghana_cocoa_dynamicworld_<year>_mode_*.tif

Output:
    data/processed/dynamicworld/
    ghana_cocoa_dynamicworld_<year>_mode_clipped.tif
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.warp import transform_bounds


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = Path("data/raw/dynamicworld")
DEFAULT_OUTPUT_DIR = Path("data/processed/dynamicworld")
DEFAULT_AOI = Path("assets/study_area_gp.gpkg")
OUTPUT_NODATA = -1


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mosaic and clip yearly Ghana Dynamic World tiles."
    )
    parser.add_argument("--year", type=int, required=True, help="Year to process, e.g. 2025.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--aoi-layer", default=None)
    parser.add_argument(
        "--expected-tiles",
        type=int,
        default=0,
        help="Expected number of tiles; 0 disables this check (default: 0).",
    )
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=256,
        help="Approximate rasterio merge memory limit (default: 256 MB).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--gap-filled",
        action="store_true",
        help="Mosaic two-band label/fill-source tiles created with --temporal-gap-fill.",
    )
    return parser.parse_args()


def validate_tiles(tile_paths: list[Path], expected_bands: int) -> tuple[object, tuple[float, float]]:
    reference_crs = None
    reference_resolution = None
    for tile_path in tile_paths:
        with rasterio.open(tile_path) as source:
            if source.count != expected_bands:
                raise ValueError(f"Expected {expected_bands} band(s) in {tile_path}, found {source.count}")
            if source.crs is None:
                raise ValueError(f"Tile has no CRS: {tile_path}")
            if reference_crs is None:
                reference_crs = source.crs
                reference_resolution = source.res
            elif source.crs != reference_crs or source.res != reference_resolution:
                raise ValueError(f"CRS or resolution mismatch: {tile_path}")
    return reference_crs, reference_resolution


def get_aoi_bounds(aoi_path: Path, layer: str, target_crs: object) -> tuple:
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI not found: {aoi_path}")
    aoi = gpd.read_file(aoi_path, layer=layer)
    if aoi.empty or aoi.crs is None:
        raise ValueError("AOI must contain geometry and have a defined CRS")
    aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty]
    if aoi.empty:
        raise ValueError("AOI contains no valid geometry")
    return transform_bounds(aoi.crs, target_crs, *aoi.total_bounds, densify_pts=21)


def validate_output(output_path: Path, expected_crs: object, expected_bands: int) -> None:
    allowed_values = set(range(9)) | {OUTPUT_NODATA}
    observed_values: set[int] = set()
    with rasterio.open(output_path) as source:
        if source.count != expected_bands or source.width < 1 or source.height < 1:
            raise RuntimeError("Output has invalid dimensions or band count")
        if source.crs != expected_crs:
            raise RuntimeError(f"Output CRS mismatch: {source.crs} != {expected_crs}")
        for _, window in source.block_windows(1):
            block_values = set(map(int, source.read(1, window=window).ravel()))
            observed_values.update(block_values)
            unexpected = block_values - allowed_values
            if unexpected:
                raise RuntimeError(f"Unexpected class values: {sorted(unexpected)}")
            if expected_bands == 2:
                source_values = set(map(int, source.read(2, window=window).ravel()))
                unexpected_sources = source_values - {0, 1, 2, 255, OUTPUT_NODATA}
                if unexpected_sources:
                    raise RuntimeError(f"Unexpected fill-source values: {sorted(unexpected_sources)}")
        log(
            f"Validated {source.width} x {source.height} pixels; "
            f"classes={sorted(observed_values)}; CRS={source.crs}"
        )


def write_gapfill_summary(output_path: Path, year: int) -> Path:
    counts: dict[int, int] = defaultdict(int)
    with rasterio.open(output_path, "r+") as source:
        source.set_band_description(1, "label")
        source.set_band_description(2, "fill_source_0_target_1_previous_2_following")
        for _, window in source.block_windows(1):
            labels = source.read(1, window=window)
            provenance = source.read(2, window=window)
            valid = (labels >= 0) & (labels <= 8)
            for code in (0, 1, 2):
                counts[code] += int(((provenance == code) & valid).sum())
            counts[255] += int((~valid).sum())
    total = sum(counts.values())
    periods = {
        0: str(year),
        1: f"{year - 1}-07-01 to {year - 1}-12-31",
        2: f"{year + 1}-01-01 to {year + 1}-06-30",
        255: "unresolved_nodata",
    }
    destination = output_path.with_name(
        f"ghana_cocoa_dynamicworld_{year}_gapfill_sources_mosaic.csv"
    )
    temporary = destination.with_suffix(".part.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("source_code", "source_period", "pixel_count", "percent")
        )
        writer.writeheader()
        for code in (0, 1, 2, 255):
            writer.writerow(
                {
                    "source_code": code,
                    "source_period": periods[code],
                    "pixel_count": counts[code],
                    "percent": round(100 * counts[code] / total, 6) if total else 0,
                }
            )
    temporary.replace(destination)
    return destination


def main() -> int:
    args = parse_args()
    if args.memory_mb < 1 or args.expected_tiles < 0:
        raise ValueError("Memory must be positive and expected tile count cannot be negative")

    input_dir = resolve_path(args.input_dir)
    output_dir = resolve_path(args.output_dir)
    aoi_path = resolve_path(args.aoi)
    product = "gapfilled" if args.gap_filled else "mode"
    expected_bands = 2 if args.gap_filled else 1
    pattern = f"ghana_cocoa_dynamicworld_{args.year}_{product}_*.tif"
    tile_paths = sorted(input_dir.glob(pattern))
    if not tile_paths:
        raise FileNotFoundError(f"No tiles found matching {input_dir / pattern}")
    if args.expected_tiles and len(tile_paths) != args.expected_tiles:
        raise RuntimeError(f"Expected {args.expected_tiles} tiles but found {len(tile_paths)}")

    output_path = output_dir / f"ghana_cocoa_dynamicworld_{args.year}_{product}_clipped.tif"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite to replace it: {output_path}")

    log(f"Found {len(tile_paths)} tiles for {args.year}; validating metadata ...")
    target_crs, resolution = validate_tiles(tile_paths, expected_bands)
    aoi_bounds = get_aoi_bounds(aoi_path, args.aoi_layer, target_crs)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".part.tif")
    if temporary_path.exists():
        temporary_path.unlink()

    log(f"Mosaicking and clipping at resolution {resolution} in {target_crs} ...")
    sources = [rasterio.open(tile_path) for tile_path in tile_paths]
    try:
        merge(
            sources,
            bounds=aoi_bounds,
            res=resolution,
            nodata=OUTPUT_NODATA,
            dtype="int16",
            indexes=list(range(1, expected_bands + 1)),
            method="first",
            target_aligned_pixels=True,
            mem_limit=args.memory_mb,
            dst_path=temporary_path,
            dst_kwds={
                "driver": "GTiff",
                "compress": "DEFLATE",
                "predictor": 1,
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "BIGTIFF": "IF_SAFER",
            },
        )
    finally:
        for source in sources:
            source.close()

    validate_output(temporary_path, target_crs, expected_bands)
    temporary_path.replace(output_path)
    log(f"Saved: {output_path}")
    if args.gap_filled:
        summary_path = write_gapfill_summary(output_path, args.year)
        log(f"Saved mosaic gap-fill summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
