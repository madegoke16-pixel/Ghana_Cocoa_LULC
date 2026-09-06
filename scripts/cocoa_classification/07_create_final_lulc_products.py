#!/usr/bin/env python3
"""Create final 2017 LULC GeoTIFF, PNG map, and class-area CSV.

Dynamic World classes are retained, except class 1 is named ``other_trees`` and
is split with the annual XGBoost result. A pixel becomes class 9
(``cocoa_plantation``) only when Dynamic World labels it as tree and XGBoost
labels it as cocoa. Dynamic World snow/ice (8) is merged into bare (7), and
cocoa uses the resulting available final code 8.

Prerequisites:
    python scripts/cocoa_classification/05_predict_tree_classes.py \
        --year 2017 --season annual --model xgboost
    python scripts/cocoa_classification/06_mosaic_tree_classification.py \
        --year 2017 --season annual --model xgboost
"""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import geopandas as gpd
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ghana_cocoa_matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from pyproj import Geod
from rasterio.features import geometry_mask, geometry_window
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling
from rasterio.windows import Window

from common import log, resolve


NODATA = 255
DW_TREE = 1
MODEL_COCOA = 2
COCOA_CLASS = 8
CLASSES = {
    0: ("water", "#419BDF"),
    1: ("other_trees", "#397D49"),
    2: ("grass", "#88B053"),
    3: ("flooded_vegetation", "#7A87C6"),
    4: ("crops", "#E49635"),
    5: ("shrub_and_scrub", "#DFC35A"),
    6: ("built", "#C4281B"),
    7: ("bare_and_snow_ice", "#A59B8F"),
    8: ("cocoa_plantation", "#6B3E26"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine yearly Dynamic World and XGBoost cocoa classes into final LULC products."
    )
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--aoi", type=Path, default=Path("assets/study_area_gp.gpkg"))
    parser.add_argument("--aoi-layer", default=None)
    parser.add_argument("--dynamic-world", type=Path)
    parser.add_argument("--cocoa-classification", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lulc/2017"))
    parser.add_argument("--png-max-size", type=int, default=3000)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_aoi(path: Path, layer: str, target_crs: object) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"AOI not found: {path}")
    aoi = gpd.read_file(path, layer=layer)
    if aoi.empty or aoi.crs is None:
        raise ValueError("AOI must contain geometry and have a defined CRS")
    aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty].copy()
    aoi.geometry = aoi.geometry.make_valid()
    if aoi.empty:
        raise ValueError("AOI contains no valid geometry")
    aoi = aoi.to_crs(target_crs)
    return [geometry.__geo_interface__ for geometry in aoi.geometry]


def pixel_area_m2_by_row(transform: object, crs: object, row: int) -> float:
    if crs.is_projected:
        return abs(transform.a * transform.e - transform.b * transform.d)
    geod = Geod(ellps="WGS84")
    x0 = transform.c
    x1 = transform.c + transform.a
    y0 = transform.f + row * transform.e
    y1 = transform.f + (row + 1) * transform.e
    area, _ = geod.polygon_area_perimeter(
        [x0, x1, x1, x0], [y0, y0, y1, y1]
    )
    return abs(area)


def create_lulc(
    dw_path: Path, model_path: Path, aoi_shapes: List[dict], output_path: Path
) -> Tuple[Dict[int, int], Dict[int, float], int, int]:
    counts: Dict[int, int] = defaultdict(int)
    areas_m2: Dict[int, float] = defaultdict(float)
    unresolved_tree_pixels = 0
    rejected_model_cocoa = 0
    with rasterio.open(dw_path) as dw, rasterio.open(model_path) as model:
        if dw.count != 1 or dw.crs is None or model.count != 1 or model.crs is None:
            raise ValueError("Dynamic World and cocoa classification must be one-band georeferenced rasters")
        crop_window = geometry_window(dw, aoi_shapes, pad_x=0, pad_y=0).round_offsets().round_lengths()
        crop_window = crop_window.intersection(Window(0, 0, dw.width, dw.height))
        output_transform = dw.window_transform(crop_window)
        profile = dw.profile.copy()
        profile.update(
            driver="GTiff", width=int(crop_window.width), height=int(crop_window.height),
            transform=output_transform, count=1, dtype="uint8", nodata=NODATA,
            compress="DEFLATE", predictor=1, tiled=True, blockxsize=512,
            blockysize=512, BIGTIFF="IF_SAFER",
        )
        temporary = output_path.with_name(output_path.stem + ".part.tif")
        with WarpedVRT(
            model,
            crs=dw.crs,
            transform=dw.transform,
            width=dw.width,
            height=dw.height,
            resampling=Resampling.nearest,
            nodata=model.nodata,
        ) as aligned_model, rasterio.open(temporary, "w", **profile) as output:
            output.set_band_description(1, "lulc_1_other_trees_7_bare_snow_8_cocoa")
            output.write_colormap(
                1,
                {
                    **{
                        code: tuple(int(color[index:index + 2], 16) for index in (1, 3, 5)) + (255,)
                        for code, (_, color) in CLASSES.items()
                    },
                    NODATA: (0, 0, 0, 0),
                },
            )
            output.update_tags(
                year=str(dw_path.stem),
                cocoa_rule="Dynamic World tree AND XGBoost class 2",
                class_1="other_trees",
                class_7="bare_and_snow_ice",
                class_8="cocoa_plantation",
            )
            for _, out_window in output.block_windows(1):
                source_window = Window(
                    out_window.col_off + crop_window.col_off,
                    out_window.row_off + crop_window.row_off,
                    out_window.width,
                    out_window.height,
                )
                dw_values = dw.read(1, window=source_window)
                model_values = aligned_model.read(1, window=source_window)
                inside = geometry_mask(
                    aoi_shapes, out_shape=(int(out_window.height), int(out_window.width)),
                    transform=dw.window_transform(source_window), invert=True,
                )
                valid_dw = inside & (dw_values >= 0) & (dw_values <= 8)
                final = np.full(dw_values.shape, NODATA, dtype="uint8")
                final[valid_dw] = dw_values[valid_dw].astype("uint8")
                final[valid_dw & (dw_values == 8)] = 7
                cocoa = valid_dw & (dw_values == DW_TREE) & (model_values == MODEL_COCOA)
                final[cocoa] = COCOA_CLASS
                unresolved_tree_pixels += int(
                    np.count_nonzero(valid_dw & (dw_values == DW_TREE) & (model_values == model.nodata))
                )
                rejected_model_cocoa += int(
                    np.count_nonzero(valid_dw & (dw_values != DW_TREE) & (model_values == MODEL_COCOA))
                )
                output.write(final, 1, window=out_window)
                for local_row in range(final.shape[0]):
                    row_values = final[local_row]
                    row_area = pixel_area_m2_by_row(
                        output_transform, dw.crs, int(out_window.row_off) + local_row
                    )
                    values, value_counts = np.unique(row_values[row_values != NODATA], return_counts=True)
                    for value, count in zip(values, value_counts):
                        counts[int(value)] += int(count)
                        areas_m2[int(value)] += int(count) * row_area
        temporary.replace(output_path)
    return counts, areas_m2, unresolved_tree_pixels, rejected_model_cocoa


def write_area_csv(path: Path, counts: Dict[int, int], areas_m2: Dict[int, float]) -> None:
    total_area = sum(areas_m2.values())
    temporary = path.with_suffix(".part.csv")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("class_code", "class_name", "pixel_count", "area_hectares", "area_sq_km", "percent_of_valid_aoi"),
        )
        writer.writeheader()
        for code, (name, _) in CLASSES.items():
            area = areas_m2.get(code, 0.0)
            writer.writerow(
                {
                    "class_code": code, "class_name": name,
                    "pixel_count": counts.get(code, 0),
                    "area_hectares": round(area / 10_000, 2),
                    "area_sq_km": round(area / 1_000_000, 4),
                    "percent_of_valid_aoi": round(100 * area / total_area, 4) if total_area else 0,
                }
            )
    temporary.replace(path)


def write_png(raster_path: Path, png_path: Path, max_size: int, dpi: int, year: int) -> None:
    with rasterio.open(raster_path) as source:
        scale = max(source.width / max_size, source.height / max_size, 1)
        width, height = max(1, round(source.width / scale)), max(1, round(source.height / scale))
        values = source.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        extent = (source.bounds.left, source.bounds.right, source.bounds.bottom, source.bounds.top)
    display = np.ma.masked_where(values == NODATA, values)
    colors = [CLASSES[code][1] for code in range(9)]
    cmap = ListedColormap(colors)
    cmap.set_bad("white", alpha=0)
    norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)
    figure, axis = plt.subplots(figsize=(11, 10))
    axis.imshow(display, cmap=cmap, norm=norm, extent=extent, origin="upper", interpolation="nearest")
    axis.set_title(f"Ghana cocoa-region LULC {year}\nDynamic World with XGBoost tree-class subdivision")
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    axis.legend(
        handles=[Patch(facecolor=color, label=f"{code}: {name.replace('_', ' ')}") for code, (name, color) in CLASSES.items()],
        loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
    )
    figure.tight_layout()
    figure.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.png_max_size < 100 or args.dpi < 50:
        raise ValueError("--png-max-size must be >=100 and --dpi must be >=50")
    dw_path = resolve(args.dynamic_world or Path(f"data/processed/dynamicworld/ghana_cocoa_dynamicworld_{args.year}_mode_clipped.tif"))
    model_path = resolve(
        args.cocoa_classification
        or Path(f"data/processed/cocoa_classification/mosaics/ghana_cocoa_tree_class_xgboost_annual_{args.year}.tif")
    )
    if not dw_path.exists():
        raise FileNotFoundError(f"Dynamic World raster not found: {dw_path}")
    if not model_path.exists():
        raise FileNotFoundError(
            f"XGBoost classification mosaic not found: {model_path}\nRun scripts 05 and 06 for --model xgboost first."
        )
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tif_path = output_dir / f"ghana_cocoa_lulc_{args.year}_dw_xgboost.tif"
    png_path = output_dir / f"ghana_cocoa_lulc_{args.year}_dw_xgboost.png"
    csv_path = output_dir / f"ghana_cocoa_lulc_{args.year}_class_areas.csv"
    existing = [path for path in (tif_path, png_path, csv_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs exist; use --overwrite: " + ", ".join(map(str, existing)))
    with rasterio.open(dw_path) as dw:
        aoi_shapes = read_aoi(resolve(args.aoi), args.aoi_layer, dw.crs)
    log("Combining Dynamic World and XGBoost cocoa classification")
    counts, areas, unresolved, rejected = create_lulc(dw_path, model_path, aoi_shapes, tif_path)
    write_area_csv(csv_path, counts, areas)
    write_png(tif_path, png_path, args.png_max_size, args.dpi, args.year)
    log(f"Saved GeoTIFF: {tif_path}")
    log(f"Saved PNG: {png_path}")
    log(f"Saved area table: {csv_path}")
    log(f"Quality checks: unresolved DW-tree pixels={unresolved:,}; model cocoa outside DW trees ignored={rejected:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
