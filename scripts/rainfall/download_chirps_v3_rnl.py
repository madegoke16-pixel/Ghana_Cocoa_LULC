#!/usr/bin/env python3
"""Download, clip, and summarize CHIRPS v3 Final Daily RNL rainfall.

Defaults cover 2000-01-01 through 2025-12-31. Global source rasters are removed
after successful clipping unless ``--keep-raw`` is supplied. Existing valid
clipped rasters are reused, so interrupted runs can be resumed safely.

Example:
    python scripts/rainfall/download_chirps_v3_rnl.py \
        --start-date 2000-01-01 --end-date 2025-12-31
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from time import sleep
from typing import Iterator, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AOI = Path("assets/aoi_bounding_box.gpkg")
DEFAULT_RAW_DIR = Path("data/raw/rainfall/chirps_v3_rnl/global")
DEFAULT_PROCESSED_DIR = Path("data/processed/rainfall/chirps_v3_rnl/daily")
DEFAULT_TABLE_DIR = Path("outputs/tables/rainfall")
BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/daily/final/rnl"
NODATA = -9999.0


def log(message: str) -> None:
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def iso_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; use YYYY-MM-DD") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download CHIRPS v3 Final Daily RNL rainfall for the Ghana cocoa AOI."
    )
    parser.add_argument("--start-date", type=iso_date, default=date(2000, 1, 1))
    parser.add_argument("--end-date", type=iso_date, default=date(2025, 12, 31))
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--aoi-layer", default=None)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--request-delay", type=float, default=0.05)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_date > args.end_date:
        raise ValueError("--start-date cannot be later than --end-date")
    if args.start_date < date(1981, 1, 1):
        raise ValueError("CHIRPS v3 begins in 1981")
    if args.end_date > date.today():
        raise ValueError("--end-date cannot be in the future")
    if args.retries < 1 or args.timeout_seconds < 1 or args.request_delay < 0:
        raise ValueError("Retries/timeout must be positive and delay cannot be negative")


def read_aoi(path: Path, layer: Optional[str]) -> tuple[list[dict], tuple[float, ...]]:
    if not path.exists():
        raise FileNotFoundError(f"AOI does not exist: {path}")
    aoi = gpd.read_file(path, layer=layer)
    if aoi.empty or aoi.crs is None:
        raise ValueError("AOI must contain geometry and have a defined CRS")
    aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty].copy()
    aoi.geometry = aoi.geometry.make_valid()
    if aoi.empty:
        raise ValueError("AOI contains no valid geometry")
    geometry = aoi.to_crs("EPSG:4326").geometry.union_all()
    if geometry.is_empty:
        raise ValueError("AOI union is empty")
    return [geometry.__geo_interface__], tuple(geometry.bounds)


def dates_between(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def make_session(retries: int) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = "Ghana-Cocoa-LULC-CHIRPS-Downloader/1.0"
    return session


def source_filename(day: date) -> str:
    return f"chirps-v3.0.rnl.{day:%Y.%m.%d}.tif"


def source_url(day: date) -> str:
    return f"{BASE_URL}/{day.year}/{source_filename(day)}"


def is_valid_raster(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with rasterio.open(path) as source:
            return source.count == 1 and source.width > 0 and source.height > 0
    except (OSError, rasterio.errors.RasterioError):
        return False


def download_global(
    session: requests.Session, url: str, destination: Path, timeout_seconds: int
) -> bool:
    if is_valid_raster(destination):
        return True
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with session.get(url, stream=True, timeout=(30, timeout_seconds)) as response:
            if response.status_code == 404:
                log(f"Not available (404): {url}")
                return False
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
        if not is_valid_raster(temporary):
            raise RuntimeError("Downloaded content is not a valid GeoTIFF")
        temporary.replace(destination)
        return True
    finally:
        if temporary.exists():
            temporary.unlink()


def clip_raster(source_path: Path, destination: Path, geometries: list[dict]) -> None:
    temporary = destination.with_name(destination.stem + ".part.tif")
    try:
        with rasterio.open(source_path) as source:
            shapes = geometries
            if source.crs is None:
                raise ValueError(f"Source raster has no CRS: {source_path}")
            if str(source.crs) != "EPSG:4326":
                shapes_gdf = gpd.GeoDataFrame.from_features(
                    [{"type": "Feature", "properties": {}, "geometry": geometries[0]}],
                    crs="EPSG:4326",
                ).to_crs(source.crs)
                shapes = [geometry.__geo_interface__ for geometry in shapes_gdf.geometry]
            clipped, transform = mask(source, shapes, crop=True, nodata=NODATA)
            metadata = source.meta.copy()
            metadata.update(
                driver="GTiff",
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=transform,
                nodata=NODATA,
                compress="DEFLATE",
                predictor=3,
                tiled=True,
            )
            with rasterio.open(temporary, "w", **metadata) as output:
                output.write(clipped)
        if not is_valid_raster(temporary):
            raise RuntimeError("Clipped output failed validation")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def rainfall_statistics(path: Path) -> dict[str, Optional[float]]:
    """Return area-weighted mean plus pixel minimum/maximum in millimetres."""
    with rasterio.open(path) as source:
        rainfall = source.read(1, masked=True).astype("float64")
        rainfall = np.ma.masked_where(rainfall < 0, rainfall)
        if rainfall.count() == 0:
            return {"rainfall_mean_mm": None, "rainfall_min_mm": None,
                    "rainfall_max_mm": None, "valid_pixels": 0}

        if source.crs and source.crs.is_geographic:
            rows = np.arange(source.height)
            latitudes = source.transform.f + (rows + 0.5) * source.transform.e
            row_weights = np.cos(np.deg2rad(latitudes))[:, None]
            weights = np.broadcast_to(row_weights, rainfall.shape)
            valid = ~np.ma.getmaskarray(rainfall)
            mean_value = float(np.average(rainfall.data[valid], weights=weights[valid]))
        else:
            mean_value = float(rainfall.mean())

        return {
            "rainfall_mean_mm": mean_value,
            "rainfall_min_mm": float(rainfall.min()),
            "rainfall_max_mm": float(rainfall.max()),
            "valid_pixels": int(rainfall.count()),
        }


def write_tables(results: list[dict], table_dir: Path, label: str) -> None:
    if not results:
        return
    daily = pd.DataFrame(results).sort_values("date")
    daily["date"] = pd.to_datetime(daily["date"])
    daily_path = table_dir / f"chirps_v3_rnl_daily_{label}.csv"
    daily.to_csv(daily_path, index=False, date_format="%Y-%m-%d")

    available = daily.dropna(subset=["rainfall_mean_mm"])
    monthly = (
        available.groupby(["year", "month"], as_index=False)
        .agg(
            rainfall_total_mm=("rainfall_mean_mm", "sum"),
            rainfall_mean_daily_mm=("rainfall_mean_mm", "mean"),
            valid_days=("rainfall_mean_mm", "count"),
        )
    )
    monthly.to_csv(table_dir / f"chirps_v3_rnl_monthly_{label}.csv", index=False)

    annual = (
        available.groupby("year", as_index=False)
        .agg(
            rainfall_total_mm=("rainfall_mean_mm", "sum"),
            rainfall_mean_daily_mm=("rainfall_mean_mm", "mean"),
            valid_days=("rainfall_mean_mm", "count"),
        )
    )
    annual.to_csv(table_dir / f"chirps_v3_rnl_annual_{label}.csv", index=False)


def main() -> int:
    args = parse_args()
    validate_args(args)
    aoi_path = resolve_path(args.aoi)
    raw_dir = resolve_path(args.raw_dir)
    processed_dir = resolve_path(args.processed_dir)
    table_dir = resolve_path(args.table_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    geometries, bounds = read_aoi(aoi_path, args.aoi_layer)
    log(f"AOI: {aoi_path}; EPSG:4326 bounds={bounds}")

    days = list(dates_between(args.start_date, args.end_date))
    label = f"{args.start_date:%Y%m%d}_{args.end_date:%Y%m%d}"
    session = make_session(args.retries)
    results = []
    missing_dates = []
    try:
        for index, day in enumerate(days, start=1):
            filename = source_filename(day)
            raw_path = raw_dir / filename
            year_dir = processed_dir / str(day.year)
            year_dir.mkdir(parents=True, exist_ok=True)
            clipped_path = year_dir / filename.replace(".tif", "_aoi.tif")
            log(f"[{index}/{len(days)}] {day:%Y-%m-%d}")

            if args.overwrite or not is_valid_raster(clipped_path):
                try:
                    if not download_global(
                        session, source_url(day), raw_path, args.timeout_seconds
                    ):
                        missing_dates.append({"date": day.isoformat(), "url": source_url(day)})
                        continue
                    clip_raster(raw_path, clipped_path, geometries)
                    log(f"Saved: {clipped_path}")
                except Exception as error:
                    log(f"ERROR {day:%Y-%m-%d}: {type(error).__name__}: {error}")
                    missing_dates.append({"date": day.isoformat(), "url": source_url(day)})
                    continue
                finally:
                    if not args.keep_raw and raw_path.exists() and is_valid_raster(clipped_path):
                        raw_path.unlink()
            else:
                log("Valid clipped raster exists; reusing it.")

            stats = rainfall_statistics(clipped_path)
            results.append({
                "date": day.isoformat(), "year": day.year,
                "month": day.month, "day": day.day, **stats,
            })
            write_tables(results, table_dir, label)
            if args.request_delay:
                sleep(args.request_delay)
    finally:
        session.close()

    write_tables(results, table_dir, label)
    pd.DataFrame(missing_dates, columns=["date", "url"]).to_csv(
        table_dir / f"chirps_v3_rnl_missing_{label}.csv", index=False
    )
    log(f"Complete: {len(results)} days processed; {len(missing_dates)} unavailable/failed")
    log(f"Daily rasters: {processed_dir}")
    log(f"Summary tables: {table_dir}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
