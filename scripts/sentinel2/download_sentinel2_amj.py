#!/usr/bin/env python3
"""Download a Sentinel-2 wet-season composite into the project folder.

The default wet-season definition is April through June of the selected
calendar year. Change it with ``--start-month`` and ``--end-month`` when
required by the study design.

Example:
    python scripts/sentinel2/download_sentinel2_amj.py \
        --year 2025 --ee-project fluted-gateway-485607-u6

Default output:
    data/raw/sentinel2/wet_2025/ghana_cocoa_s2_wet_2025_r###_c###.tif
"""

from __future__ import annotations

import argparse
from datetime import date
import os
from pathlib import Path

import ee
from shapely.geometry import mapping

from download_sentinel2_djf import (
    BANDS,
    CLEAR_THRESHOLD,
    CLOUD_SCORE_BAND,
    CLOUD_SCORE_COLLECTION,
    DEFAULT_AOI,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TILE_SIZE_KM,
    NODATA,
    S2_COLLECTION,
    TARGET_CRS,
    TARGET_SCALE,
    download_tile,
    is_valid_tile,
    log,
    make_tiles,
    mask_s2,
    read_aoi,
    resolve_path,
)


DEFAULT_START_MONTH = 4
DEFAULT_END_MONTH = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a cloud-masked Sentinel-2 wet-season composite locally."
    )
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument(
        "--ee-project",
        "--project",
        dest="ee_project",
        default=os.getenv("EARTHENGINE_PROJECT"),
        help="Earth Engine-enabled Google Cloud project (or EARTHENGINE_PROJECT).",
    )
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--aoi-layer", default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--start-month",
        type=int,
        default=DEFAULT_START_MONTH,
        help="First included month (default: 4, April).",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=DEFAULT_END_MONTH,
        help="Last included month (default: 6, June).",
    )
    parser.add_argument("--cloud-threshold", type=float, default=CLEAR_THRESHOLD)
    parser.add_argument("--scale", type=float, default=TARGET_SCALE)
    parser.add_argument("--crs", default=TARGET_CRS)
    parser.add_argument("--tile-size-km", type=float, default=DEFAULT_TILE_SIZE_KM)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--ee-deadline-seconds", type=int, default=900)
    parser.add_argument("--download-timeout-seconds", type=int, default=900)
    parser.add_argument("--authenticate", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.year < 2016 or args.year > date.today().year:
        raise ValueError(f"--year must be between 2016 and {date.today().year}")
    if not args.ee_project:
        raise ValueError("Provide --ee-project or set EARTHENGINE_PROJECT in .env")
    if not 1 <= args.start_month <= 12 or not 1 <= args.end_month <= 12:
        raise ValueError("Season months must be between 1 and 12")
    if args.start_month > args.end_month:
        raise ValueError("--start-month cannot be later than --end-month")
    if not 0 <= args.cloud_threshold <= 1:
        raise ValueError("--cloud-threshold must be between 0 and 1")
    positive = (
        args.scale,
        args.tile_size_km,
        args.retries,
        args.ee_deadline_seconds,
        args.download_timeout_seconds,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Scale, tile size, retries, and timeout values must be positive")

    estimated_mb = ((args.tile_size_km * 1000 / args.scale) ** 2 * len(BANDS) * 2) / 1e6
    if estimated_mb > 28:
        raise ValueError(
            f"Requested tiles are approximately {estimated_mb:.1f} MB uncompressed; "
            "reduce --tile-size-km to stay safely below Earth Engine's 32 MB limit"
        )


def season_dates(year: int, start_month: int, end_month: int) -> tuple[str, str]:
    start = f"{year}-{start_month:02d}-01"
    if end_month == 12:
        end = f"{year + 1}-01-01"
    else:
        end = f"{year}-{end_month + 1:02d}-01"
    return start, end


def build_wet_composite(
    year: int,
    start_month: int,
    end_month: int,
    roi: ee.Geometry,
    threshold: float,
) -> tuple[ee.Image, int]:
    start, end = season_dates(year, start_month, end_month)
    s2 = ee.ImageCollection(S2_COLLECTION).filterBounds(roi).filterDate(start, end)
    cloud_score = (
        ee.ImageCollection(CLOUD_SCORE_COLLECTION).filterBounds(roi).filterDate(start, end)
    )
    collection = s2.linkCollection(cloud_score, [CLOUD_SCORE_BAND]).map(
        lambda image: mask_s2(image, threshold)
    )
    image_count = collection.size().getInfo()
    if not image_count:
        raise RuntimeError(f"No Sentinel-2 scenes found from {start} to {end}")
    log(f"Wet-season interval: {start} to {end} (end exclusive); scenes: {image_count}")
    composite = collection.select(BANDS).median().round().toUint16().clip(roi).unmask(NODATA)
    return composite, image_count


def main() -> int:
    args = parse_args()
    validate_args(args)
    aoi_path = resolve_path(args.aoi)
    output_dir = resolve_path(args.output_root) / f"wet_{args.year}"
    aoi = read_aoi(aoi_path, args.aoi_layer)
    tiles = make_tiles(aoi, args.crs, args.tile_size_km)
    if not tiles:
        raise RuntimeError("No tiles intersect the AOI")

    estimated_gb = (
        len(tiles)
        * ((args.tile_size_km * 1000 / args.scale) ** 2)
        * len(BANDS)
        * 2
        / 1e9
    )
    log(
        f"AOI: {aoi_path} ({len(aoi)} feature(s)); grid: {len(tiles)} tiles; "
        f"uncompressed estimate: {estimated_gb:.1f} GB"
    )

    if args.authenticate:
        ee.Authenticate()
    ee.Initialize(project=args.ee_project)
    ee.data.setDeadline(args.ee_deadline_seconds * 1000)
    entire_aoi = ee.Geometry(mapping(aoi.to_crs("EPSG:4326").geometry.union_all()))
    composite, _ = build_wet_composite(
        args.year,
        args.start_month,
        args.end_month,
        entire_aoi,
        args.cloud_threshold,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (row, col, geometry) in enumerate(tiles, start=1):
        destination = output_dir / (
            f"ghana_cocoa_s2_wet_{args.year}_r{row:03d}_c{col:03d}.tif"
        )
        if destination.exists() and not args.overwrite and is_valid_tile(destination):
            log(f"[{index}/{len(tiles)}] Valid tile exists; skipping {destination.name}")
            continue
        log(f"[{index}/{len(tiles)}] Downloading {destination.name}")
        download_tile(
            composite,
            geometry,
            destination,
            args.crs,
            args.scale,
            args.retries,
            args.download_timeout_seconds,
        )

    log(f"Download complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
