#!/usr/bin/env python3
"""Download tiled annual Dynamic World mode classifications for the project AOI.

The script reads ``assets/aoi_bounding_box.gpkg``, creates a grid in a projected
CRS, calculates the pixel-wise annual mode of Dynamic World's ``label`` band in
Google Earth Engine, and downloads one GeoTIFF per intersecting grid cell.

Examples
--------
Download 2017 using an Earth Engine Cloud project::

    python scripts/dynamicworld/download_dynamicworld_lulc.py \
        --year 2017 --ee-project YOUR_GOOGLE_CLOUD_PROJECT

Download all configured project years::

    python scripts/dynamicworld/download_dynamicworld_lulc.py \
        --all-configured-years --ee-project YOUR_GOOGLE_CLOUD_PROJECT

The output naming convention is compatible with the supplied mosaic workflow:
``ghana_cocoa_dynamicworld_<year>_mode_r###_c###.tif``.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import os
from pathlib import Path
import sys
from time import sleep
from typing import Optional
from urllib.request import urlopen

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from dotenv import load_dotenv
from shapely.geometry import box, mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
DEFAULT_AOI = Path("assets/study_area_gp.gpkg")
DEFAULT_OUTPUT = Path("data/raw/dynamicworld")
DEFAULT_GRID_CRS = "EPSG:32630"
COLLECTION_ID = "GOOGLE/DYNAMICWORLD/V1"
NODATA = -1


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download annual Dynamic World mode tiles for the Ghana cocoa AOI."
    )
    years = parser.add_mutually_exclusive_group(required=True)
    years.add_argument("--year", type=int, help="Single calendar year to download.")
    years.add_argument(
        "--all-configured-years",
        action="store_true",
        help="Download baseline_year and comparison_years from config/project.yaml.",
    )
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument(
        "--aoi-layer",
        default=None,
        help="Optional GeoPackage layer name; defaults to its first spatial layer.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ee-project",
        default=os.getenv("EARTHENGINE_PROJECT"),
        help="Google Cloud project enabled for Earth Engine (or EARTHENGINE_PROJECT).",
    )
    parser.add_argument(
        "--authenticate",
        action="store_true",
        help="Run the interactive Earth Engine authentication flow first.",
    )
    parser.add_argument("--scale", type=float, default=10.0, help="Output pixel size in metres.")
    parser.add_argument(
        "--tile-size-km",
        type=float,
        default=40.0,
        help="Projected grid-cell width/height. Keep small enough for EE's download limit.",
    )
    parser.add_argument("--grid-crs", default=DEFAULT_GRID_CRS)
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Attempts per tile for transient Earth Engine or network failures.",
    )
    parser.add_argument(
        "--ee-deadline-seconds",
        type=int,
        default=900,
        help="Deadline for each Earth Engine API request (default: 900 seconds).",
    )
    parser.add_argument(
        "--download-timeout-seconds",
        type=int,
        default=900,
        help="Network timeout while transferring each GeoTIFF (default: 900 seconds).",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--temporal-gap-fill",
        action="store_true",
        help=(
            "Fill missing target-year pixels first from Jul-Dec of the previous "
            "year, then Jan-Jun of the following year; export a source-QA band."
        ),
    )
    return parser.parse_args()


def configured_years() -> list[int]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from ghana_cocoa_lulc.config import load_project_config

    analysis = load_project_config()["analysis"]
    return sorted({int(analysis["baseline_year"]), *map(int, analysis["comparison_years"])})


def validate_year(year: int) -> None:
    current_year = date.today().year
    if year < 2015 or year > current_year:
        raise ValueError(f"Year must be between 2015 and {current_year}: {year}")


def read_aoi(path: Path, layer: Optional[str]) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"AOI does not exist: {path}")
    aoi = gpd.read_file(path, layer=layer)
    if aoi.empty:
        raise ValueError(f"AOI contains no features: {path}")
    if aoi.crs is None:
        raise ValueError(f"AOI has no CRS: {path}")
    aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty].copy()
    aoi.geometry = aoi.geometry.make_valid()
    if aoi.empty:
        raise ValueError(f"AOI contains no valid geometries: {path}")
    return aoi


def make_tiles(aoi: gpd.GeoDataFrame, grid_crs: str, tile_size_km: float) -> list[tuple[int, int, dict]]:
    if tile_size_km <= 0:
        raise ValueError("--tile-size-km must be positive")
    projected = aoi.to_crs(grid_crs)
    footprint = projected.geometry.union_all()
    min_x, min_y, max_x, max_y = footprint.bounds
    step = tile_size_km * 1000.0
    tiles: list[tuple[int, int, dict]] = []
    row = 0
    y = min_y
    while y < max_y:
        col = 0
        x = min_x
        while x < max_x:
            clipped = box(x, y, min(x + step, max_x), min(y + step, max_y)).intersection(footprint)
            if not clipped.is_empty and clipped.area > 0:
                tile = gpd.GeoSeries([clipped], crs=grid_crs).to_crs("EPSG:4326").iloc[0]
                tiles.append((row, col, mapping(tile)))
            x += step
            col += 1
        y += step
        row += 1
    return tiles


def mode_for_period(start: str, end: str, region: ee.Geometry) -> tuple[ee.Image, int]:
    collection = (
        ee.ImageCollection(COLLECTION_ID)
        .filterDate(start, end)
        .filterBounds(region)
        .select("label")
    )
    count = collection.size().getInfo()
    if not count:
        raise RuntimeError(f"No Dynamic World scenes found from {start} to {end}")
    return collection.mode().rename("label"), count


def annual_mode(year: int, region: ee.Geometry, temporal_gap_fill: bool) -> ee.Image:
    start = f"{year}-01-01"
    end = f"{year + 1}-01-01"
    primary, count = mode_for_period(start, end, region)
    log(f"Dynamic World scenes intersecting AOI in {year}: {count}")
    if not temporal_gap_fill:
        return primary.unmask(NODATA).toInt16()

    previous_start, previous_end = f"{year - 1}-07-01", f"{year}-01-01"
    next_start, next_end = f"{year + 1}-01-01", f"{year + 1}-07-01"
    previous, previous_count = mode_for_period(previous_start, previous_end, region)
    following, next_count = mode_for_period(next_start, next_end, region)
    log(
        f"Gap-fill scenes: {previous_start} to {previous_end}={previous_count}; "
        f"{next_start} to {next_end}={next_count}"
    )
    label = primary.unmask(previous).unmask(following).unmask(NODATA).rename("label")
    # Priority is expressed by applying lower-priority masks first.
    source = (
        ee.Image.constant(255)
        .where(following.mask(), 2)
        .where(previous.mask(), 1)
        .where(primary.mask(), 0)
        .rename("fill_source")
    )
    return label.addBands(source).toInt16()


def download_file(url: str, destination: Path, timeout_seconds: int) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=timeout_seconds) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_tile(
    image: ee.Image,
    geometry: dict,
    destination: Path,
    scale: float,
    retries: int,
    download_timeout_seconds: int,
    bands: list[str],
) -> None:
    region = ee.Geometry(geometry)
    parameters = {
        "bands": bands,
        "region": region,
        "scale": scale,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    }
    for attempt in range(1, retries + 1):
        try:
            url = image.clip(region).getDownloadURL(parameters)
            download_file(url, destination, download_timeout_seconds)
            return
        except Exception as error:
            if attempt == retries:
                raise
            delay = min(60, 2**attempt)
            detail = str(error).strip().replace("\n", " ")
            log(
                f"Attempt {attempt}/{retries} failed ({type(error).__name__}: {detail}); "
                f"retrying in {delay} seconds ..."
            )
            sleep(delay)


def valid_tile(path: Path, expected_bands: int) -> bool:
    try:
        with rasterio.open(path) as source:
            return (
                source.count == expected_bands
                and source.width > 0
                and source.height > 0
                and source.crs is not None
            )
    except (OSError, rasterio.errors.RasterioError):
        return False


def write_gapfill_summary(tile_paths: list[Path], year: int, output_dir: Path) -> None:
    counts = {0: 0, 1: 0, 2: 0, 255: 0}
    for path in tile_paths:
        with rasterio.open(path) as source:
            labels = source.read(1)
            provenance = source.read(2)
            valid = (labels >= 0) & (labels <= 8)
            for code in (0, 1, 2):
                counts[code] += int(np.count_nonzero(valid & (provenance == code)))
            counts[255] += int(np.count_nonzero(~valid | (provenance == 255)))
    total = sum(counts.values())
    rows = [
        {"source_code": 0, "source_period": str(year), "pixel_count": counts[0]},
        {"source_code": 1, "source_period": f"{year - 1}-07-01 to {year - 1}-12-31", "pixel_count": counts[1]},
        {"source_code": 2, "source_period": f"{year + 1}-01-01 to {year + 1}-06-30", "pixel_count": counts[2]},
        {"source_code": 255, "source_period": "unresolved_nodata", "pixel_count": counts[255]},
    ]
    frame = pd.DataFrame(rows)
    frame["percent"] = np.where(total > 0, frame["pixel_count"] / total * 100, 0).round(6)
    destination = output_dir / f"ghana_cocoa_dynamicworld_{year}_gapfill_sources.csv"
    temporary = destination.with_suffix(".part.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)
    log(f"Saved gap-fill pixel summary: {destination}")


def main() -> int:
    args = parse_args()
    if (
        args.scale <= 0
        or args.retries < 1
        or args.ee_deadline_seconds < 1
        or args.download_timeout_seconds < 1
    ):
        raise ValueError("Scale, retries, and timeout values must be positive")
    if not args.ee_project:
        raise SystemExit("Provide --ee-project or set EARTHENGINE_PROJECT in .env/environment.")

    years = configured_years() if args.all_configured_years else [args.year]
    for year in years:
        validate_year(year)

    aoi_path = resolve_path(args.aoi)
    output_dir = resolve_path(args.output)
    aoi = read_aoi(aoi_path, args.aoi_layer)
    tiles = make_tiles(aoi, args.grid_crs, args.tile_size_km)
    if not tiles:
        raise RuntimeError("No download tiles intersect the AOI")
    log(f"AOI: {aoi_path} ({len(aoi)} feature(s)); download grid: {len(tiles)} tile(s)")

    if args.authenticate:
        ee.Authenticate()
    ee.Initialize(project=args.ee_project)
    ee.data.setDeadline(args.ee_deadline_seconds * 1000)
    log(
        f"Earth Engine deadline: {args.ee_deadline_seconds}s; "
        f"retries per tile: {args.retries}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    entire_aoi = ee.Geometry(mapping(aoi.to_crs("EPSG:4326").geometry.union_all()))
    for year in years:
        image = annual_mode(year, entire_aoi, args.temporal_gap_fill)
        bands = ["label", "fill_source"] if args.temporal_gap_fill else ["label"]
        product = "gapfilled" if args.temporal_gap_fill else "mode"
        log(f"Downloading annual mode for {year} ...")
        completed_tiles = []
        for index, (row, col, geometry) in enumerate(tiles, start=1):
            destination = output_dir / (
                f"ghana_cocoa_dynamicworld_{year}_{product}_r{row:03d}_c{col:03d}.tif"
            )
            if destination.exists() and not args.overwrite and valid_tile(destination, len(bands)):
                log(f"[{index}/{len(tiles)}] Exists; skipping {destination.name}")
                completed_tiles.append(destination)
                continue
            log(f"[{index}/{len(tiles)}] Downloading {destination.name}")
            download_tile(
                image,
                geometry,
                destination,
                args.scale,
                args.retries,
                args.download_timeout_seconds,
                bands,
            )
            if not valid_tile(destination, len(bands)):
                raise RuntimeError(f"Downloaded tile failed validation: {destination}")
            completed_tiles.append(destination)
        if args.temporal_gap_fill:
            write_gapfill_summary(completed_tiles, year, output_dir)
    log("Download complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
