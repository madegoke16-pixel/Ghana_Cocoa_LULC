#!/usr/bin/env python3
"""Create tiled Google Drive exports of a Sentinel-2 DJF composite.

The AOI is read locally from ``assets/aoi_bounding_box.gpkg`` and converted to
an Earth Engine geometry; no AOI asset upload is required. A season year of
2025 means 2024-12-01 through 2025-03-01 (end date exclusive).

The script creates one Earth Engine Drive-export task per spectral band. Large
bands are split into 4096 x 4096-pixel GeoTIFF files automatically.

Example
-------
    python scripts/sentinel2/download_sentinel2_djf.py \
        --year 2025 \
        --ee-project fluted-gateway-485607-u6
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import os
from pathlib import Path
from typing import Optional

import ee
import geopandas as gpd
from dotenv import load_dotenv
from shapely.geometry import mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_SCORE_COLLECTION = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"
CLOUD_SCORE_BAND = "cs_cdf"
DEFAULT_AOI = Path("assets/aoi_bounding_box.gpkg")
DEFAULT_DRIVE_FOLDER = "Ghana_Cocoa_Sentinel2"
TARGET_CRS = "EPSG:32630"
TARGET_SCALE = 10
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
        description="Export a cloud-masked Sentinel-2 DJF median composite for the Ghana cocoa AOI."
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Season year; 2025 represents December 2024 through February 2025.",
    )
    parser.add_argument(
        "--ee-project",
        "--project",
        dest="ee_project",
        default=os.getenv("EARTHENGINE_PROJECT"),
        help="Earth Engine-enabled Google Cloud project (or EARTHENGINE_PROJECT).",
    )
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument(
        "--aoi-layer",
        default=None,
        help="Optional GeoPackage layer; the first spatial layer is used by default.",
    )
    parser.add_argument("--drive-folder", "--folder", default=DEFAULT_DRIVE_FOLDER)
    parser.add_argument("--cloud-threshold", type=float, default=CLEAR_THRESHOLD)
    parser.add_argument("--scale", type=float, default=TARGET_SCALE)
    parser.add_argument("--crs", default=TARGET_CRS)
    parser.add_argument(
        "--file-dimensions",
        type=int,
        default=4096,
        help="Width/height of each Drive GeoTIFF shard; must be a multiple of 256.",
    )
    parser.add_argument(
        "--authenticate",
        action="store_true",
        help="Run interactive Earth Engine authentication before initialization.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and collection, but do not start export tasks.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.year < 2016 or args.year > date.today().year:
        raise ValueError(f"--year must be between 2016 and {date.today().year}")
    if not args.ee_project:
        raise ValueError("Provide --ee-project or set EARTHENGINE_PROJECT in .env")
    if not 0 <= args.cloud_threshold <= 1:
        raise ValueError("--cloud-threshold must be between 0 and 1")
    if args.scale <= 0:
        raise ValueError("--scale must be positive")
    if args.file_dimensions <= 0 or args.file_dimensions % 256:
        raise ValueError("--file-dimensions must be a positive multiple of 256")


def read_local_aoi(path: Path, layer: Optional[str]) -> tuple[ee.Geometry, int]:
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
    geometry = aoi.to_crs("EPSG:4326").geometry.union_all()
    return ee.Geometry(mapping(geometry)), len(aoi)


def mask_s2(image: ee.Image, threshold: float) -> ee.Image:
    """Mask invalid image edges and pixels below the Cloud Score+ threshold."""
    edge_mask = image.select("B8A").mask().updateMask(image.select("B9").mask())
    clear_mask = image.select(CLOUD_SCORE_BAND).gte(threshold)
    return image.updateMask(edge_mask).updateMask(clear_mask)


def build_collection(year: int, roi: ee.Geometry, threshold: float) -> ee.ImageCollection:
    start = f"{year - 1}-12-01"
    end = f"{year}-03-01"
    s2 = ee.ImageCollection(S2_COLLECTION).filterBounds(roi).filterDate(start, end)
    cloud_score = (
        ee.ImageCollection(CLOUD_SCORE_COLLECTION).filterBounds(roi).filterDate(start, end)
    )
    collection = s2.linkCollection(cloud_score, [CLOUD_SCORE_BAND]).map(
        lambda image: mask_s2(image, threshold)
    )
    log(f"DJF interval: {start} to {end} (end exclusive)")
    return collection


def create_composite(collection: ee.ImageCollection, roi: ee.Geometry) -> ee.Image:
    """Create a reflectance-preserving uint16 median composite."""
    return (
        collection.select(BANDS)
        .median()
        .round()
        .toUint16()
        .clip(roi)
        .unmask(NODATA)
    )


def create_export_tasks(
    composite: ee.Image,
    roi: ee.Geometry,
    year: int,
    folder: str,
    crs: str,
    scale: float,
    file_dimensions: int,
    dry_run: bool,
) -> list[ee.batch.Task]:
    tasks = []
    for band in BANDS:
        name = f"S2_Ghana_Cocoa_DJF_{year}_{band}_10m"
        log(f"Preparing Drive export: {name}")
        task = ee.batch.Export.image.toDrive(
            image=composite.select(band),
            description=name,
            folder=folder,
            fileNamePrefix=name,
            region=roi,
            crs=crs,
            scale=scale,
            maxPixels=1e13,
            shardSize=256,
            fileDimensions=file_dimensions,
            skipEmptyTiles=True,
            fileFormat="GeoTIFF",
            formatOptions={"cloudOptimized": True, "noData": NODATA},
        )
        if not dry_run:
            task.start()
            log(f"Started task {task.id}")
        tasks.append(task)
    return tasks


def main() -> int:
    args = parse_args()
    validate_args(args)
    aoi_path = resolve_path(args.aoi)

    if args.authenticate:
        ee.Authenticate()
    ee.Initialize(project=args.ee_project)
    roi, feature_count = read_local_aoi(aoi_path, args.aoi_layer)
    log(f"AOI: {aoi_path} ({feature_count} feature(s))")

    collection = build_collection(args.year, roi, args.cloud_threshold)
    image_count = collection.size().getInfo()
    log(f"Sentinel-2 scenes: {image_count}")
    if image_count == 0:
        raise RuntimeError("No Sentinel-2 scenes found for the selected DJF interval")

    composite = create_composite(collection, roi)
    tasks = create_export_tasks(
        composite,
        roi,
        args.year,
        args.drive_folder,
        args.crs,
        args.scale,
        args.file_dimensions,
        args.dry_run,
    )
    if args.dry_run:
        log(f"Dry run complete; {len(tasks)} export tasks validated but not started.")
    else:
        log(f"Started {len(tasks)} exports. Monitor them with: earthengine task list")
        log(f"Completed files will appear in Google Drive folder: {args.drive_folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
