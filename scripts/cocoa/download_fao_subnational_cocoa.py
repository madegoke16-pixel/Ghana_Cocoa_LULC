#!/usr/bin/env python3
"""Download FAO Admin-1 cocoa statistics for the Ghana cocoa study regions.

FAO's source contains administrative-region totals, not gridded observations.
Consequently, values cannot be clipped to the AOI boundary. The AOI is validated
and recorded as the study footprint; region names may optionally be derived by
intersecting it with an Admin-1 boundary file.

Example:
    python scripts/cocoa/download_fao_subnational_cocoa.py \
        --start-year 2000 --end-year 2024

Optional spatial region selection:
    python scripts/cocoa/download_fao_subnational_cocoa.py \
        --region-boundaries assets/ghana_admin1.gpkg \
        --region-name-field region_name
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AOI = Path("assets/aoi_bounding_box.gpkg")
DEFAULT_RAW_DIR = Path("data/raw/cocoa/fao_subnational_admin1")
DEFAULT_OUTPUT_DIR = Path("data/processed/cocoa/fao_subnational_admin1")
DEFAULT_REGIONS = ("Ashanti", "Western", "Western North")
BASE_URL = "https://api.data.apps.fao.org/api/v2/bigquery"
# Current resource linked from the official FAO catalog (checked 2026-09-05).
DEFAULT_SQL_URL = (
    "https://data.apps.fao.org/catalog/dataset/"
    "397cba06-8b6f-41bb-bef1-026af97efeaa/resource/"
    "a8da849f-4fad-4ef3-b49d-0fd13d015cb6/download/"
    "sub-national-level-1-query.sql"
)
METRICS = {
    "production_tonnes": "Production (Metric Tonnes)",
    "area_ha": "Harvested Area (Hectares)",
}


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download FAO Admin-1 cocoa production and harvested area for the Ghana cocoa AOI."
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--aoi", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--aoi-layer", default=None)
    parser.add_argument("--regions", nargs="+", default=list(DEFAULT_REGIONS))
    parser.add_argument("--region-boundaries", type=Path)
    parser.add_argument("--region-boundaries-layer", default=None)
    parser.add_argument("--region-name-field")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sql-url", default=DEFAULT_SQL_URL)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write partial results and a failures CSV instead of stopping.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start_year > args.end_year:
        raise ValueError("--start-year cannot be later than --end-year")
    if args.start_year < 1900 or args.end_year > datetime.now().year:
        raise ValueError("Year range is outside the supported validation range")
    if args.retries < 1 or args.timeout_seconds < 1 or args.request_delay < 0:
        raise ValueError("Retries/timeout must be positive and delay cannot be negative")
    if args.region_boundaries and not args.region_name_field:
        raise ValueError("--region-name-field is required with --region-boundaries")


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
    return aoi.to_crs("EPSG:4326")


def regions_intersecting_aoi(
    aoi: gpd.GeoDataFrame, boundary_path: Path, layer: Optional[str], name_field: str
) -> List[str]:
    if not boundary_path.exists():
        raise FileNotFoundError(f"Region boundary file does not exist: {boundary_path}")
    boundaries = gpd.read_file(boundary_path, layer=layer)
    if boundaries.empty or boundaries.crs is None:
        raise ValueError("Region boundaries must contain geometry and have a CRS")
    if name_field not in boundaries.columns:
        raise ValueError(
            f"Region name field {name_field!r} not found; available fields: "
            + ", ".join(map(str, boundaries.columns))
        )
    study_geometry = aoi.to_crs(boundaries.crs).geometry.union_all()
    selected = boundaries[
        boundaries.geometry.notna()
        & ~boundaries.geometry.is_empty
        & boundaries.geometry.intersects(study_geometry)
    ]
    regions = selected[name_field].dropna().astype(str).str.strip().drop_duplicates().tolist()
    if not regions:
        raise RuntimeError("No Admin-1 region boundaries intersect the AOI")
    return regions


def make_session(retries: int) -> requests.Session:
    retry = Retry(
        total=retries,
        connect=retries,
        read=retries,
        status=retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers["User-Agent"] = "Ghana-Cocoa-LULC-FAO-Downloader/1.0"
    return session


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().replace("&", "and")
    return re.sub(r"\s+", " ", re.sub(r"[_\-]+", " ", text)).strip()


def find_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    normalized = {normalize_text(column): str(column) for column in frame.columns}
    for candidate in candidates:
        if normalize_text(candidate) in normalized:
            return normalized[normalize_text(candidate)]
    for column in frame.columns:
        normalized_column = normalize_text(column)
        if any(normalize_text(candidate) in normalized_column for candidate in candidates):
            return str(column)
    return None


def response_to_frame(content: bytes, content_type: str) -> pd.DataFrame:
    text = content.decode("utf-8-sig")
    if "json" not in content_type.lower():
        try:
            frame = pd.read_csv(StringIO(text))
            if len(frame.columns) > 1:
                return frame
        except (pd.errors.ParserError, UnicodeDecodeError):
            pass
    payload = json.loads(text)
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        for key in ("data", "results", "result", "rows", "features"):
            rows = payload.get(key)
            if not isinstance(rows, list):
                continue
            if rows and isinstance(rows[0], dict) and "properties" in rows[0]:
                return pd.DataFrame([row.get("properties", {}) for row in rows])
            return pd.DataFrame(rows)
    raise RuntimeError("FAO response was neither a recognizable CSV nor JSON table")


def download_year_metric(
    session: requests.Session,
    year: int,
    metric: str,
    sql_url: str,
    raw_path: Path,
    timeout_seconds: int,
    overwrite: bool,
) -> pd.DataFrame:
    if raw_path.exists() and not overwrite:
        return pd.read_csv(raw_path)
    params = {
        "crop": "All Crops",
        "metric": metric,
        "year": year,
        "sql_url": sql_url,
        "download": "true",
    }
    response = session.get(BASE_URL, params=params, timeout=(30, timeout_seconds))
    response.raise_for_status()
    frame = response_to_frame(response.content, response.headers.get("Content-Type", ""))
    temporary = raw_path.with_suffix(".part.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(raw_path)
    return frame


def identify_columns(frame: pd.DataFrame) -> Dict[str, Optional[str]]:
    return {
        "country": find_column(frame, ("country", "country name", "admin0", "adm0_name")),
        "region": find_column(
            frame, ("region", "region name", "admin1", "admin1_name", "adm1_name")
        ),
        "crop": find_column(
            frame,
            (
                "crop",
                "crop name",
                "proper_name",
                "common_name",
                "cpc_product_name",
                "commodity",
                "item",
                "item_name",
            ),
        ),
        "year": find_column(frame, ("year", "reference year")),
        "value": find_column(
            frame, ("value", "production", "area harvested", "metric value")
        ),
    }


def clean_metric(
    frame: pd.DataFrame,
    value_name: str,
    regions: Sequence[str],
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    columns = identify_columns(frame)
    missing = [name for name in ("region", "crop", "value") if columns[name] is None]
    if missing:
        raise RuntimeError(
            "Could not identify FAO columns " + ", ".join(missing) + "; received: "
            + ", ".join(map(str, frame.columns))
        )
    result = frame.copy()
    if columns["country"]:
        # Exact matching avoids accidentally treating "Afghanistan" as Ghana.
        result = result[result[columns["country"]].map(normalize_text) == "ghana"].copy()
    crop_columns = [
        column
        for column in (
            columns["crop"],
            find_column(result, ("proper_name",)),
            find_column(result, ("common_name",)),
            find_column(result, ("cpc_product_name",)),
        )
        if column is not None
    ]
    crop_columns = list(dict.fromkeys(crop_columns))
    cocoa_mask = pd.Series(False, index=result.index)
    for crop_column in crop_columns:
        cocoa_mask |= result[crop_column].astype(str).str.contains("cocoa", case=False, na=False)
    result = result[cocoa_mask].copy()
    region_map = {normalize_text(region): region for region in regions}
    result["region"] = result[columns["region"]].map(normalize_text).map(region_map)
    result = result[result["region"].notna()].copy()
    source_year = columns["year"] if columns["year"] else "download_year"
    result["year"] = pd.to_numeric(result[source_year], errors="coerce")
    result[value_name] = pd.to_numeric(result[columns["value"]], errors="coerce")
    result = result[result["year"].between(start_year, end_year)].copy()
    result["year"] = result["year"].astype(int)
    return (
        result.groupby(["year", "region"], as_index=False)[value_name]
        .sum(min_count=1)
        .sort_values(["year", "region"])
    )


def coverage_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, group in panel.groupby("region", sort=False):
        observed = group[group["production_tonnes"].notna()]
        rows.append(
            {
                "region": region,
                "first_production_year": observed["year"].min() if not observed.empty else np.nan,
                "last_production_year": observed["year"].max() if not observed.empty else np.nan,
                "production_records": int(group["production_tonnes"].count()),
                "area_records": int(group["area_ha"].count()),
                "yield_records": int(group["yield_t_ha"].count()),
            }
        )
    return pd.DataFrame(rows)


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(".part.csv")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    aoi_path = resolve_path(args.aoi)
    raw_dir = resolve_path(args.raw_dir)
    output_dir = resolve_path(args.output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    aoi = read_aoi(aoi_path, args.aoi_layer)

    regions = list(dict.fromkeys(region.strip() for region in args.regions if region.strip()))
    selection_method = "Configured study-region names"
    if args.region_boundaries:
        regions = regions_intersecting_aoi(
            aoi,
            resolve_path(args.region_boundaries),
            args.region_boundaries_layer,
            args.region_name_field,
        )
        selection_method = "Spatial intersection with supplied Admin-1 boundaries"
    if not regions:
        raise ValueError("At least one target region is required")

    bounds = tuple(round(value, 6) for value in aoi.total_bounds)
    log(f"AOI validated: {aoi_path} | bounds EPSG:4326={bounds}")
    log(f"Regions: {', '.join(regions)}")
    log("FAO values are whole Admin-1 totals; they are not clipped or apportioned to the AOI.")

    session = make_session(args.retries)
    collected: Dict[str, List[pd.DataFrame]] = {name: [] for name in METRICS}
    failures: List[Dict[str, object]] = []
    for value_name, metric in METRICS.items():
        # Preserve the original production cache names from earlier runs. The
        # corrected area metric gets a distinct name, so cached empty responses
        # produced with the old invalid label are never reused.
        metric_slug = (
            "production-tonnes"
            if value_name == "production_tonnes"
            else "harvested-area-hectares"
        )
        for year in range(args.start_year, args.end_year + 1):
            raw_path = raw_dir / f"fao_admin1_{metric_slug}_{year}.csv"
            try:
                log(f"Downloading {metric}: {year}")
                frame = download_year_metric(
                    session,
                    year,
                    metric,
                    args.sql_url,
                    raw_path,
                    args.timeout_seconds,
                    args.overwrite,
                )
                frame["download_year"] = year
                collected[value_name].append(frame)
                log(f"Received {len(frame):,} rows")
            except Exception as error:
                failures.append({"year": year, "metric": metric, "error": str(error)})
                log(f"ERROR {year} {metric}: {error}")
                if not args.continue_on_error:
                    raise
            time.sleep(args.request_delay)

        # Do not spend time downloading harvested area when the source has no
        # Ghana cocoa production coverage. This also prevents empty panels from
        # looking like valid study results.
        if value_name == "production_tonnes" and collected[value_name]:
            production_probe = clean_metric(
                pd.concat(collected[value_name], ignore_index=True),
                value_name,
                regions,
                args.start_year,
                args.end_year,
            )
            if production_probe.empty:
                raise RuntimeError(
                    "The current FAO Admin-1 resource returned no Ghana cocoa records "
                    f"for {args.start_year}-{args.end_year}. The download succeeded, but "
                    "this source presently has no usable coverage for Ashanti, Western, "
                    "or Western North. Do not interpret this as zero production. Consider "
                    "COCOBOD regional purchase data or another Ghana administrative source."
                )

    if any(not frames for frames in collected.values()):
        raise RuntimeError("No usable responses were downloaded for one or more required metrics")
    production_raw = pd.concat(collected["production_tonnes"], ignore_index=True)
    area_raw = pd.concat(collected["area_ha"], ignore_index=True)
    production = clean_metric(
        production_raw, "production_tonnes", regions, args.start_year, args.end_year
    )
    area = clean_metric(area_raw, "area_ha", regions, args.start_year, args.end_year)
    observations = production.merge(area, on=["year", "region"], how="outer")

    full_grid = pd.MultiIndex.from_product(
        [range(args.start_year, args.end_year + 1), regions], names=["year", "region"]
    ).to_frame(index=False)
    panel = full_grid.merge(observations, on=["year", "region"], how="left")
    panel["yield_t_ha"] = np.where(
        panel["area_ha"] > 0, panel["production_tonnes"] / panel["area_ha"], np.nan
    )
    panel["yield_kg_ha"] = panel["yield_t_ha"] * 1000
    panel["data_status"] = np.where(
        panel[["production_tonnes", "area_ha"]].notna().any(axis=1), "Observed", "Missing"
    )
    panel["administrative_note"] = ""
    historical = (panel["region"] == "Western North") & (panel["year"] < 2019)
    panel.loc[historical, "data_status"] = "Not separate region"
    panel.loc[historical, "administrative_note"] = (
        "Western North was not a separate administrative region before 2019; "
        "historical observations may be included under the former Western Region."
    )
    panel["spatial_scope_note"] = "Whole FAO Admin-1 total; not clipped to the AOI"
    panel[["production_tonnes", "area_ha", "yield_kg_ha"]] = panel[
        ["production_tonnes", "area_ha", "yield_kg_ha"]
    ].round(2)
    panel["yield_t_ha"] = panel["yield_t_ha"].round(4)

    label = f"{args.start_year}_{args.end_year}"
    panel_path = output_dir / f"ghana_cocoa_fao_admin1_panel_{label}.csv"
    model_path = output_dir / f"ghana_cocoa_fao_admin1_model_data_{label}.csv"
    coverage_path = output_dir / f"ghana_cocoa_fao_admin1_coverage_{label}.csv"
    write_csv_atomic(panel, panel_path)
    write_csv_atomic(
        panel[["year", "region", "production_tonnes", "area_ha", "yield_t_ha", "yield_kg_ha"]],
        model_path,
    )
    write_csv_atomic(coverage_summary(panel), coverage_path)
    for region in regions:
        safe_name = re.sub(r"[^a-z0-9]+", "_", normalize_text(region)).strip("_")
        write_csv_atomic(panel[panel["region"] == region], output_dir / f"{safe_name}_{label}.csv")
    if failures:
        write_csv_atomic(pd.DataFrame(failures), output_dir / f"download_failures_{label}.csv")

    metadata = {
        "source": "FAO Datalab Subnational Agricultural Production (Admin Level 1)",
        "catalog_url": "https://data.fao.org/catalog/dataset/sub-national-agricultural-production-admin-level-1",
        "api_url": BASE_URL,
        "sql_url": args.sql_url,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "year_range": [args.start_year, args.end_year],
        "regions": regions,
        "region_selection_method": selection_method,
        "aoi_path": str(aoi_path),
        "aoi_bounds_epsg4326": bounds,
        "spatial_scope_warning": "Statistics are whole Admin-1 totals and were not clipped to the AOI.",
        "failed_requests": len(failures),
    }
    metadata_path = output_dir / f"ghana_cocoa_fao_admin1_metadata_{label}.json"
    temporary_metadata = metadata_path.with_suffix(".part.json")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary_metadata.replace(metadata_path)

    log(f"Complete: {panel_path}")
    log(f"Observed panel rows: {(panel['data_status'] == 'Observed').sum():,}/{len(panel):,}")
    if failures:
        log(f"Completed with {len(failures)} failed request(s); see the failures CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
