#!/usr/bin/env python3
"""Calculate complete pixel-level LULC transitions between two aligned maps.

Outputs include transition/change rasters, long and matrix transition tables,
class gains/losses/persistence/net change, cocoa conversion tables, annualized
rates, and equivalent summaries for every study region.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Geod
from rasterio.features import geometry_mask

from common import CLASSES, GROUPS, NODATA, TRANSITION_NODATA, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate aligned LULC transitions and regional summaries.")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--start-raster", type=Path)
    parser.add_argument("--end-raster", type=Path)
    parser.add_argument("--start-area-csv", type=Path)
    parser.add_argument("--end-area-csv", type=Path)
    parser.add_argument("--regions", type=Path, default=Path("assets/study_area_gp.gpkg"))
    parser.add_argument("--regions-layer", default=None)
    parser.add_argument("--region-field", default="adm1_name")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transitions/2017_2025"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def assert_aligned(start: rasterio.io.DatasetReader, end: rasterio.io.DatasetReader) -> None:
    checks = {
        "CRS": start.crs == end.crs,
        "transform": start.transform.almost_equals(end.transform),
        "width": start.width == end.width,
        "height": start.height == end.height,
        "band count": start.count == end.count == 1,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Input rasters are not aligned; failed: " + ", ".join(failed))


def row_pixel_areas_m2(source: rasterio.io.DatasetReader) -> np.ndarray:
    if source.crs.is_projected:
        area = abs(source.transform.a * source.transform.e - source.transform.b * source.transform.d)
        return np.full(source.height, area, dtype="float64")
    geod = Geod(ellps="WGS84")
    areas = np.empty(source.height, dtype="float64")
    x0, x1 = source.transform.c, source.transform.c + source.transform.a
    for row in range(source.height):
        y0 = source.transform.f + row * source.transform.e
        y1 = source.transform.f + (row + 1) * source.transform.e
        area, _ = geod.polygon_area_perimeter([x0, x1, x1, x0], [y0, y0, y1, y1])
        areas[row] = abs(area)
    return areas


def add_counts(
    count_target: np.ndarray,
    area_target: np.ndarray,
    codes: np.ndarray,
    valid: np.ndarray,
    row_areas: np.ndarray,
) -> None:
    if not valid.any():
        return
    selected = codes[valid]
    weights = np.broadcast_to(row_areas[:, None], codes.shape)[valid]
    count_target += np.bincount(selected, minlength=89)[:89]
    area_target += np.bincount(selected, weights=weights, minlength=89)[:89]


def transition_frame(counts: np.ndarray, areas_m2: np.ndarray, region: str) -> pd.DataFrame:
    rows = []
    total_area = float(areas_m2.sum())
    for source_code, (source_name, _) in CLASSES.items():
        source_total = sum(areas_m2[source_code * 10 + target] for target in CLASSES)
        for target_code, (target_name, _) in CLASSES.items():
            code = source_code * 10 + target_code
            area = float(areas_m2[code])
            rows.append(
                {
                    "region": region,
                    "from_code": source_code,
                    "from_class": source_name,
                    "to_code": target_code,
                    "to_class": target_name,
                    "transition_code": code,
                    "pixel_count": int(counts[code]),
                    "area_ha": area / 10_000,
                    "area_sq_km": area / 1_000_000,
                    "percent_of_valid_area": 100 * area / total_area if total_area else 0,
                    "percent_of_from_class": 100 * area / source_total if source_total else 0,
                    "is_persistence": source_code == target_code,
                }
            )
    return pd.DataFrame(rows)


def class_dynamics(transitions: pd.DataFrame, years: int, region: str) -> pd.DataFrame:
    rows = []
    for code, (name, _) in CLASSES.items():
        initial = transitions.loc[transitions.from_code == code, "area_ha"].sum()
        final = transitions.loc[transitions.to_code == code, "area_ha"].sum()
        persistence = transitions.loc[
            (transitions.from_code == code) & (transitions.to_code == code), "area_ha"
        ].sum()
        loss, gain = initial - persistence, final - persistence
        net = final - initial
        swap = 2 * min(gain, loss)
        rows.append(
            {
                "region": region, "class_code": code, "class_name": name,
                "initial_area_ha": initial, "final_area_ha": final,
                "persistence_ha": persistence, "gross_gain_ha": gain,
                "gross_loss_ha": loss, "net_change_ha": net,
                "absolute_net_change_ha": abs(net), "swap_change_ha": swap,
                "total_change_ha": gain + loss,
                "percent_change_from_initial": 100 * net / initial if initial else np.nan,
                "annual_linear_change_ha": net / years,
                "annual_compound_rate_percent":
                    100 * ((final / initial) ** (1 / years) - 1) if initial > 0 and final > 0 else np.nan,
                "persistence_percent": 100 * persistence / initial if initial else np.nan,
                "gain_loss_ratio": gain / loss if loss > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def write_matrix(transitions: pd.DataFrame, value: str, path: Path) -> None:
    matrix = transitions.pivot(index="from_class", columns="to_class", values=value)
    order = [value[0] for value in CLASSES.values()]
    matrix.reindex(index=order, columns=order).to_csv(path)


def grouped_transitions(transitions: pd.DataFrame, region_column: bool = False) -> pd.DataFrame:
    code_to_group = {code: group for group, codes in GROUPS.items() for code in codes}
    grouped = transitions.copy()
    grouped["from_group"] = grouped.from_code.map(code_to_group)
    grouped["to_group"] = grouped.to_code.map(code_to_group)
    keys = (["region"] if region_column else []) + ["from_group", "to_group"]
    result = grouped.groupby(keys, as_index=False).agg(
        pixel_count=("pixel_count", "sum"),
        area_ha=("area_ha", "sum"),
        area_sq_km=("area_sq_km", "sum"),
    )
    result["is_persistence"] = result.from_group == result.to_group
    denominator = result.groupby((["region"] if region_column else []) + ["from_group"])["area_sq_km"].transform("sum")
    result["percent_of_from_group"] = np.where(denominator > 0, 100 * result.area_sq_km / denominator, 0)
    return result


def area_consistency(start_csv: Path, end_csv: Path, dynamics: pd.DataFrame) -> pd.DataFrame:
    """Compare standalone-map class areas with the mutually valid transition footprint."""
    start = pd.read_csv(start_csv).rename(columns={"area_hectares": "standalone_start_area_ha"})
    end = pd.read_csv(end_csv).rename(columns={"area_hectares": "standalone_end_area_ha"})
    columns = ["class_code", "class_name"]
    audit = start[columns + ["standalone_start_area_ha"]].merge(
        end[columns + ["standalone_end_area_ha"]], on=columns, how="outer"
    ).merge(
        dynamics[["class_code", "initial_area_ha", "final_area_ha"]], on="class_code", how="outer"
    )
    audit = audit.rename(columns={
        "initial_area_ha": "comparable_start_area_ha",
        "final_area_ha": "comparable_end_area_ha",
    })
    audit["start_excluded_due_to_end_nodata_ha"] = audit.standalone_start_area_ha - audit.comparable_start_area_ha
    audit["end_excluded_due_to_start_nodata_ha"] = audit.standalone_end_area_ha - audit.comparable_end_area_ha
    return audit


def main() -> int:
    args = parse_args()
    if args.end_year <= args.start_year:
        raise ValueError("--end-year must be later than --start-year")
    start_path = resolve(args.start_raster or Path(f"outputs/lulc/{args.start_year}/ghana_cocoa_lulc_{args.start_year}_dw_xgboost.tif"))
    end_path = resolve(args.end_raster or Path(f"outputs/lulc/{args.end_year}/ghana_cocoa_lulc_{args.end_year}_dw_xgboost.tif"))
    start_area_csv = resolve(args.start_area_csv or Path(f"outputs/lulc/{args.start_year}/ghana_cocoa_lulc_{args.start_year}_class_areas.csv"))
    end_area_csv = resolve(args.end_area_csv or Path(f"outputs/lulc/{args.end_year}/ghana_cocoa_lulc_{args.end_year}_class_areas.csv"))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transition_path = output_dir / f"lulc_transition_codes_{args.start_year}_{args.end_year}.tif"
    change_path = output_dir / f"lulc_change_types_{args.start_year}_{args.end_year}.tif"
    if not start_path.exists() or not end_path.exists():
        raise FileNotFoundError(f"Missing input raster: {start_path} or {end_path}")
    if (transition_path.exists() or change_path.exists()) and not args.overwrite:
        raise FileExistsError("Transition outputs exist; use --overwrite")

    regions = gpd.read_file(resolve(args.regions), layer=args.regions_layer)
    if regions.empty or regions.crs is None or args.region_field not in regions.columns:
        raise ValueError("Regions must have geometry, CRS, and the configured region-name field")
    regions = regions[regions.geometry.notna() & ~regions.geometry.is_empty].copy()
    regions.geometry = regions.geometry.make_valid()

    overall_counts = np.zeros(89, dtype="int64")
    overall_areas = np.zeros(89, dtype="float64")
    region_counts: Dict[str, np.ndarray] = {}
    region_areas: Dict[str, np.ndarray] = {}

    with rasterio.open(start_path) as start, rasterio.open(end_path) as end:
        assert_aligned(start, end)
        regions = regions.to_crs(start.crs)
        region_shapes = {
            str(row[args.region_field]): row.geometry.__geo_interface__
            for _, row in regions.iterrows()
        }
        for name in region_shapes:
            region_counts[name] = np.zeros(89, dtype="int64")
            region_areas[name] = np.zeros(89, dtype="float64")
        row_areas_all = row_pixel_areas_m2(start)
        transition_profile = start.profile.copy()
        transition_profile.update(dtype="uint16", nodata=TRANSITION_NODATA, compress="DEFLATE", predictor=1, tiled=True, BIGTIFF="IF_SAFER")
        change_profile = start.profile.copy()
        change_profile.update(dtype="uint8", nodata=NODATA, compress="DEFLATE", predictor=1, tiled=True, BIGTIFF="IF_SAFER")
        transition_tmp = transition_path.with_name(transition_path.stem + ".part.tif")
        change_tmp = change_path.with_name(change_path.stem + ".part.tif")
        with rasterio.open(transition_tmp, "w", **transition_profile) as transition_out, rasterio.open(change_tmp, "w", **change_profile) as change_out:
            transition_out.set_band_description(1, "transition_code_from_times_10_plus_to")
            change_out.set_band_description(1, "0_persistence_1_cocoa_gain_2_cocoa_loss_3_other_change")
            for _, window in start.block_windows(1):
                before = start.read(1, window=window)
                after = end.read(1, window=window)
                valid = np.isin(before, list(CLASSES)) & np.isin(after, list(CLASSES))
                codes = before.astype("int16") * 10 + after.astype("int16")
                transition_block = np.full(before.shape, TRANSITION_NODATA, dtype="uint16")
                transition_block[valid] = codes[valid].astype("uint16")
                change = np.full(before.shape, NODATA, dtype="uint8")
                change[valid] = np.where(before[valid] == after[valid], 0, 3)
                change[valid & (before != 8) & (after == 8)] = 1
                change[valid & (before == 8) & (after != 8)] = 2
                transition_out.write(transition_block, 1, window=window)
                change_out.write(change, 1, window=window)
                row0 = int(window.row_off)
                row_areas = row_areas_all[row0:row0 + int(window.height)]
                add_counts(overall_counts, overall_areas, codes, valid, row_areas)
                for name, shape in region_shapes.items():
                    inside = geometry_mask([shape], out_shape=before.shape, transform=start.window_transform(window), invert=True)
                    add_counts(region_counts[name], region_areas[name], codes, valid & inside, row_areas)
        transition_tmp.replace(transition_path)
        change_tmp.replace(change_path)

    years = args.end_year - args.start_year
    overall = transition_frame(overall_counts, overall_areas, "All study regions")
    regional = pd.concat(
        [transition_frame(region_counts[name], region_areas[name], name) for name in region_counts],
        ignore_index=True,
    )
    overall.to_csv(output_dir / "transition_long_overall.csv", index=False)
    regional.to_csv(output_dir / "transition_long_by_region.csv", index=False)
    write_matrix(overall, "area_sq_km", output_dir / "transition_matrix_area_sq_km.csv")
    write_matrix(overall, "pixel_count", output_dir / "transition_matrix_pixels.csv")
    write_matrix(overall, "percent_of_from_class", output_dir / "transition_matrix_row_percent.csv")
    dynamics_overall = class_dynamics(overall, years, "All study regions")
    dynamics_regional = pd.concat(
        [class_dynamics(regional[regional.region == name], years, name) for name in region_counts],
        ignore_index=True,
    )
    dynamics_overall.to_csv(output_dir / "class_dynamics_overall.csv", index=False)
    dynamics_regional.to_csv(output_dir / "class_dynamics_by_region.csv", index=False)
    grouped = grouped_transitions(overall)
    grouped_regional = grouped_transitions(regional, region_column=True)
    grouped.to_csv(output_dir / "grouped_transition_overall.csv", index=False)
    grouped_regional.to_csv(output_dir / "grouped_transition_by_region.csv", index=False)
    group_order = list(GROUPS)
    grouped.pivot(index="from_group", columns="to_group", values="area_sq_km").reindex(
        index=group_order, columns=group_order
    ).to_csv(output_dir / "grouped_transition_matrix_area_sq_km.csv")
    if start_area_csv.exists() and end_area_csv.exists():
        area_consistency(start_area_csv, end_area_csv, dynamics_overall).to_csv(
            output_dir / "input_area_consistency.csv", index=False
        )
    cocoa = overall[((overall.from_code == 8) | (overall.to_code == 8)) & (overall.from_code != overall.to_code)].copy()
    cocoa["direction"] = np.where(cocoa.to_code == 8, "cocoa_gain", "cocoa_loss")
    cocoa.to_csv(output_dir / "cocoa_transitions_overall.csv", index=False)
    cocoa_regional = regional[((regional.from_code == 8) | (regional.to_code == 8)) & (regional.from_code != regional.to_code)].copy()
    cocoa_regional["direction"] = np.where(cocoa_regional.to_code == 8, "cocoa_gain", "cocoa_loss")
    cocoa_regional.to_csv(output_dir / "cocoa_transitions_by_region.csv", index=False)

    valid_area = overall.area_ha.sum()
    persistence = overall.loc[overall.is_persistence, "area_ha"].sum()
    metadata = {
        "start_year": args.start_year, "end_year": args.end_year, "interval_years": years,
        "start_raster": str(start_path), "end_raster": str(end_path),
        "start_area_csv": str(start_area_csv), "end_area_csv": str(end_area_csv),
        "valid_comparison_area_ha": valid_area,
        "persistence_area_ha": persistence,
        "changed_area_ha": valid_area - persistence,
        "overall_change_percent": 100 * (valid_area - persistence) / valid_area if valid_area else 0,
        "transition_code_formula": "from_class * 10 + to_class",
        "change_type_codes": {"0": "persistence", "1": "cocoa_gain", "2": "cocoa_loss", "3": "other_change", "255": "nodata"},
    }
    (output_dir / "transition_summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved transition analysis to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
