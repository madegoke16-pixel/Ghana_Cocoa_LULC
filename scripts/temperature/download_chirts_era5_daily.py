#!/usr/bin/env python3
"""Download AOI subsets of daily CHIRTS-ERA5 Tmax and Tmin.

CHIRTS-daily ends in 2016. CHIRTS-ERA5 is the Climate Hazards Center extension
used here to provide a consistent 2000-2025 temperature record. The source
GeoTIFF server supports byte-range access, so this script reads only the Ghana
AOI window instead of downloading each ~26 MB global raster.

Example:
    python scripts/temperature/download_chirts_era5_daily.py \
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AOI = Path("assets/aoi_bounding_box.gpkg")
DEFAULT_OUTPUT_DIR = Path("data/processed/temperature/chirts_era5/daily")
DEFAULT_TABLE_DIR = Path("outputs/tables/temperature")
BASE_URL = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5"
NODATA = -9999.0
VARIABLES = ("Tmax", "Tmin")


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
        description="Subset daily CHIRTS-ERA5 Tmax/Tmin for the Ghana cocoa AOI."
    )
    parser.add_argument("--start-date", type=iso_date, default=date(2000, 1, 1))
    parser.add_argument("--end-date", type=iso_date, default=date(2025, 12, 31))
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--aoi-layer", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--request-delay", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_date > args.end_date:
        raise ValueError("--start-date cannot be later than --end-date")
    if args.start_date < date(1959, 1, 1):
        raise ValueError("The CHIRTS-ERA5 daily archive begins in 1959")
    if args.end_date > date.today():
        raise ValueError("--end-date cannot be in the future")
    if args.retries < 1 or args.timeout_seconds < 1:
        raise ValueError("Retries and timeout must be positive")
    if args.retry_delay < 0 or args.request_delay < 0:
        raise ValueError("Delay values cannot be negative")


def read_aoi(path: Path, layer: Optional[str]) -> list[dict]:
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
    return [geometry.__geo_interface__]


def dates_between(start: date, end: date) -> Iterator[date]:
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def source_url(variable: str, day: date) -> str:
    directory = variable.lower()
    filename = f"CHIRTS-ERA5.daily_{variable}.{day:%Y.%m.%d}.tif"
    return f"{BASE_URL}/{directory}/tifs/daily/{day.year}/{filename}"


def output_path(output_dir: Path, day: date) -> Path:
    return output_dir / str(day.year) / f"chirts_era5_tmax_tmin_{day:%Y.%m.%d}_aoi.tif"


def is_valid_output(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with rasterio.open(path) as source:
            return (
                source.count == 2
                and source.width > 0
                and source.height > 0
                and source.crs is not None
                and source.descriptions == ("Tmax_C", "Tmin_C")
            )
    except (OSError, rasterio.errors.RasterioError):
        return False


def read_remote_subset(url: str, geometries: list[dict]) -> tuple[np.ndarray, dict]:
    vsi_url = f"/vsicurl/{url}"
    with rasterio.open(vsi_url) as source:
        if source.crs is None:
            raise ValueError(f"Remote raster has no CRS: {url}")
        shapes = geometries
        if str(source.crs) != "EPSG:4326":
            frame = gpd.GeoDataFrame.from_features(
                [{"type": "Feature", "properties": {}, "geometry": geometries[0]}],
                crs="EPSG:4326",
            ).to_crs(source.crs)
            shapes = [geometry.__geo_interface__ for geometry in frame.geometry]
        subset, transform = mask(source, shapes, crop=True, nodata=NODATA)
        subset = subset.astype("float32", copy=False)
        subset[~np.isfinite(subset) | (subset <= -9990)] = NODATA
        metadata = source.meta.copy()
        metadata.update(
            driver="GTiff", height=subset.shape[1], width=subset.shape[2],
            transform=transform, count=1, dtype="float32", nodata=NODATA,
        )
        return subset[0], metadata


def create_daily_raster(day: date, destination: Path, geometries: list[dict]) -> None:
    arrays = []
    metadata = None
    for variable in VARIABLES:
        array, current_metadata = read_remote_subset(source_url(variable, day), geometries)
        if metadata is None:
            metadata = current_metadata
        elif (
            array.shape != arrays[0].shape
            or current_metadata["transform"] != metadata["transform"]
            or current_metadata["crs"] != metadata["crs"]
        ):
            raise RuntimeError(f"Tmax/Tmin grids do not align for {day}")
        arrays.append(array)

    temporary = destination.with_name(destination.stem + ".part.tif")
    metadata.update(
        count=2, compress="DEFLATE", predictor=3, tiled=True,
        blockxsize=256, blockysize=256,
    )
    try:
        with rasterio.open(temporary, "w", **metadata) as output:
            output.write(np.stack(arrays))
            output.set_band_description(1, "Tmax_C")
            output.set_band_description(2, "Tmin_C")
            output.update_tags(
                source="Climate Hazards Center CHIRTS-ERA5 daily",
                date=day.isoformat(), units="degrees Celsius",
            )
        if not is_valid_output(temporary):
            raise RuntimeError("Daily output failed validation")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def area_weighted_mean(values: np.ma.MaskedArray, source: rasterio.DatasetReader) -> float:
    rows = np.arange(source.height)
    latitudes = source.transform.f + (rows + 0.5) * source.transform.e
    weights = np.broadcast_to(np.cos(np.deg2rad(latitudes))[:, None], values.shape)
    valid = ~np.ma.getmaskarray(values)
    return float(np.average(values.data[valid], weights=weights[valid]))


def daily_statistics(path: Path) -> dict[str, float]:
    with rasterio.open(path) as source:
        tmax = source.read(1, masked=True).astype("float64")
        tmin = source.read(2, masked=True).astype("float64")
        joint_mask = np.ma.getmaskarray(tmax) | np.ma.getmaskarray(tmin)
        tmax.mask = joint_mask
        tmin.mask = joint_mask
        if tmax.count() == 0:
            raise RuntimeError(f"No valid temperature pixels: {path}")
        tmean = (tmax + tmin) / 2
        dtr = tmax - tmin
        return {
            "tmax_mean_c": area_weighted_mean(tmax, source),
            "tmax_max_c": float(tmax.max()),
            "tmin_mean_c": area_weighted_mean(tmin, source),
            "tmin_min_c": float(tmin.min()),
            "tmean_mean_c": area_weighted_mean(tmean, source),
            "dtr_mean_c": area_weighted_mean(dtr, source),
            "valid_pixels": int(tmax.count()),
        }


def write_tables(results: list[dict], table_dir: Path, label: str) -> None:
    if not results:
        return
    daily = pd.DataFrame(results).sort_values("date")
    daily["date"] = pd.to_datetime(daily["date"])
    daily.to_csv(
        table_dir / f"chirts_era5_daily_{label}.csv", index=False, date_format="%Y-%m-%d"
    )
    aggregations = {
        "tmax_mean_c": ("tmax_mean_c", "mean"),
        "tmax_extreme_c": ("tmax_max_c", "max"),
        "tmin_mean_c": ("tmin_mean_c", "mean"),
        "tmin_extreme_c": ("tmin_min_c", "min"),
        "tmean_mean_c": ("tmean_mean_c", "mean"),
        "dtr_mean_c": ("dtr_mean_c", "mean"),
        "valid_days": ("date", "count"),
    }
    monthly = daily.groupby(["year", "month"], as_index=False).agg(**aggregations)
    monthly.to_csv(table_dir / f"chirts_era5_monthly_{label}.csv", index=False)
    annual = daily.groupby("year", as_index=False).agg(**aggregations)
    annual.to_csv(table_dir / f"chirts_era5_annual_{label}.csv", index=False)


def main() -> int:
    args = parse_args()
    validate_args(args)
    aoi_path = resolve_path(args.aoi)
    output_dir = resolve_path(args.output_dir)
    table_dir = resolve_path(args.table_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    geometries = read_aoi(aoi_path, args.aoi_layer)
    days = list(dates_between(args.start_date, args.end_date))
    label = f"{args.start_date:%Y%m%d}_{args.end_date:%Y%m%d}"
    log(f"AOI: {aoi_path}; days: {len(days)}; variables: Tmax, Tmin")

    results = []
    failures = []
    env_options = {
        "GDAL_HTTP_MAX_RETRY": str(args.retries),
        "GDAL_HTTP_RETRY_DELAY": str(max(1, int(args.retry_delay))),
        "GDAL_HTTP_TIMEOUT": str(args.timeout_seconds),
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    }
    with rasterio.Env(**env_options):
        for index, day in enumerate(days, start=1):
            destination = output_path(output_dir, day)
            destination.parent.mkdir(parents=True, exist_ok=True)
            log(f"[{index}/{len(days)}] {day:%Y-%m-%d}")
            if args.overwrite or not is_valid_output(destination):
                for attempt in range(1, args.retries + 1):
                    try:
                        create_daily_raster(day, destination, geometries)
                        log(f"Saved: {destination}")
                        break
                    except Exception as error:
                        if attempt == args.retries:
                            log(f"FAILED: {type(error).__name__}: {error}")
                            failures.append({"date": day.isoformat(), "error": str(error)})
                            break
                        delay = min(60, args.retry_delay * (2 ** (attempt - 1)))
                        log(f"Attempt {attempt}/{args.retries} failed; retrying in {delay:g}s")
                        sleep(delay)
            else:
                log("Valid daily raster exists; reusing it.")

            if is_valid_output(destination):
                results.append({
                    "date": day.isoformat(), "year": day.year,
                    "month": day.month, "day": day.day,
                    **daily_statistics(destination),
                })
                write_tables(results, table_dir, label)
            if args.request_delay:
                sleep(args.request_delay)

    write_tables(results, table_dir, label)
    pd.DataFrame(failures, columns=["date", "error"]).to_csv(
        table_dir / f"chirts_era5_failures_{label}.csv", index=False
    )
    log(f"Complete: {len(results)} days processed; {len(failures)} failed")
    log(f"Daily rasters: {output_dir}")
    log(f"Summary tables: {table_dir}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
