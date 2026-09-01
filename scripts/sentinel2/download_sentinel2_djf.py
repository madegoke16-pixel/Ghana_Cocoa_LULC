#!/usr/bin/env python3
"""Download a Sentinel-2 DJF composite directly into the project folder.

The local AOI is divided into small projected grid cells so each multiband
GeoTIFF stays below Earth Engine's direct-download request limit. Downloads are
resumable: valid existing tiles are skipped unless ``--overwrite`` is supplied.

Example:
    python scripts/sentinel2/download_sentinel2_djf.py \
        --year 2025 --ee-project fluted-gateway-485607-u6

Default output:
    data/raw/sentinel2/djf_2025/ghana_cocoa_s2_djf_2025_r###_c###.tif
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import os
from pathlib import Path
from time import sleep
from typing import Optional
from urllib.request import urlopen

import ee
import geopandas as gpd
import rasterio
from dotenv import load_dotenv
from shapely.geometry import box, mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_BAND = "cs_cdf"
DEFAULT_AOI = Path("assets/aoi_bounding_box.gpkg")
DEFAULT_OUTPUT_ROOT = Path("data/raw/sentinel2")
TARGET_CRS = "EPSG:32630"
TARGET_SCALE = 10.0
DEFAULT_TILE_SIZE_KM = 10.0
CLEAR_THRESHOLD = 0.60
NODATA = 65535
BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a cloud-masked Sentinel-2 DJF composite to local tiled GeoTIFFs."
    )
    parser.add_argument(
        "--year", type=int, required=True,
        help="Season year; 2025 represents December 2024 through February 2025.",
    )
    parser.add_argument(
        "--ee-project", "--project", dest="ee_project",
        default=os.getenv("EARTHENGINE_PROJECT"),
        help="Earth Engine-enabled Google Cloud project (or EARTHENGINE_PROJECT).",
    )
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--aoi-layer", default=None)
    parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
        help="Parent directory; a djf_<year> folder is created inside it.",
    )
    parser.add_argument("--cloud-threshold", type=float, default=CLEAR_THRESHOLD)
    parser.add_argument("--scale", type=float, default=TARGET_SCALE)
    parser.add_argument("--crs", default=TARGET_CRS)
    parser.add_argument(
        "--tile-size-km", type=float, default=DEFAULT_TILE_SIZE_KM,
        help="Projected tile width/height (default: 10 km for the 32 MB request limit).",
    )
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
    if not 0 <= args.cloud_threshold <= 1:
        raise ValueError("--cloud-threshold must be between 0 and 1")
    positive = (
        args.scale, args.tile_size_km, args.retries,
        args.ee_deadline_seconds, args.download_timeout_seconds,
    )
    if any(value <= 0 for value in positive):
        raise ValueError("Scale, tile size, retries, and timeout values must be positive")

    estimated_mb = ((args.tile_size_km * 1000 / args.scale) ** 2 * len(BANDS) * 2) / 1e6
    if estimated_mb > 28:
        raise ValueError(
            f"Requested tiles are approximately {estimated_mb:.1f} MB uncompressed; "
            "reduce --tile-size-km to stay safely below Earth Engine's 32 MB limit"
        )


def read_aoi(path: Path, layer: Optional[str]) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"AOI does not exist: {path}")
    aoi = gpd.read_file(path, layer=layer)
    if aoi.empty or aoi.crs is None:
        raise ValueError("AOI must contain geometry and have a defined CRS")
    aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty].copy()
    aoi.geometry = aoi.geometry.make_valid()
    if aoi.empty:
        raise ValueError("AOI contains no valid geometry")
    return aoi


def make_tiles(
    aoi: gpd.GeoDataFrame, grid_crs: str, tile_size_km: float
) -> list[tuple[int, int, dict]]:
    projected = aoi.to_crs(grid_crs)
    footprint = projected.geometry.union_all()
    min_x, min_y, max_x, max_y = footprint.bounds
    step = tile_size_km * 1000.0
    tiles = []
    row = 0
    y = min_y
    while y < max_y:
        col = 0
        x = min_x
        while x < max_x:
            cell = box(x, y, min(x + step, max_x), min(y + step, max_y))
            clipped = cell.intersection(footprint)
            if not clipped.is_empty and clipped.area > 0:
                wgs84 = gpd.GeoSeries([clipped], crs=grid_crs).to_crs("EPSG:4326").iloc[0]
                tiles.append((row, col, mapping(wgs84)))
            x += step
            col += 1
        y += step
        row += 1
    return tiles


def mask_s2(image: ee.Image, threshold: float) -> ee.Image:
    edge_mask = image.select("B8A").mask().updateMask(image.select("B9").mask())
    clear_mask = image.select(CLOUD_SCORE_BAND).gte(threshold)
    return image.updateMask(edge_mask).updateMask(clear_mask)


def build_composite(
    year: int, roi: ee.Geometry, threshold: float
) -> tuple[ee.Image, int]:
    start = f"{year - 1}-12-01"
    end = f"{year}-03-01"
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
    log(f"DJF interval: {start} to {end} (end exclusive); scenes: {image_count}")
    composite = (
        collection.select(BANDS).median().round().toUint16()
        .clip(roi).unmask(NODATA)
    )
    return composite, image_count


def is_valid_tile(path: Path) -> bool:
    try:
        with rasterio.open(path) as source:
            return (
                source.count == len(BANDS)
                and source.width > 0
                and source.height > 0
                and source.crs is not None
            )
    except (OSError, rasterio.errors.RasterioError):
        return False


def download_file(url: str, destination: Path, timeout_seconds: int) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(url, timeout=timeout_seconds) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if not is_valid_tile(temporary):
            raise RuntimeError(f"Downloaded file is not a valid {len(BANDS)}-band GeoTIFF")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_tile(
    composite: ee.Image, geometry: dict, destination: Path,
    crs: str, scale: float, retries: int, timeout_seconds: int,
) -> None:
    region = ee.Geometry(geometry)
    parameters = {
        "bands": BANDS, "region": region, "scale": scale,
        "crs": crs, "format": "GEO_TIFF",
    }
    for attempt in range(1, retries + 1):
        try:
            url = composite.clip(region).getDownloadURL(parameters)
            download_file(url, destination, timeout_seconds)
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


def main() -> int:
    args = parse_args()
    validate_args(args)
    aoi_path = resolve_path(args.aoi)
    output_dir = resolve_path(args.output_root) / f"djf_{args.year}"
    aoi = read_aoi(aoi_path, args.aoi_layer)
    tiles = make_tiles(aoi, args.crs, args.tile_size_km)
    if not tiles:
        raise RuntimeError("No tiles intersect the AOI")
    estimated_gb = (
        len(tiles) * ((args.tile_size_km * 1000 / args.scale) ** 2)
        * len(BANDS) * 2 / 1e9
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
    composite, _ = build_composite(args.year, entire_aoi, args.cloud_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, (row, col, geometry) in enumerate(tiles, start=1):
        destination = output_dir / (
            f"ghana_cocoa_s2_djf_{args.year}_r{row:03d}_c{col:03d}.tif"
        )
        if destination.exists() and not args.overwrite and is_valid_tile(destination):
            log(f"[{index}/{len(tiles)}] Valid tile exists; skipping {destination.name}")
            continue
        log(f"[{index}/{len(tiles)}] Downloading {destination.name}")
        download_tile(
            composite, geometry, destination, args.crs, args.scale,
            args.retries, args.download_timeout_seconds,
        )

    log(f"Download complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
