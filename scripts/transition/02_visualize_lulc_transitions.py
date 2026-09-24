#!/usr/bin/env python3
"""Create transition heatmaps, Sankeys, class-dynamics charts, and change maps."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ghana_cocoa_matplotlib"))

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Patch, Rectangle
from rasterio.warp import Resampling

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from common import CLASSES, GROUPS, NODATA, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize 2017-2025 Ghana cocoa LULC transitions.")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--input-dir", type=Path, default=Path("outputs/transitions/2017_2025"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/transitions/2017_2025/figures"))
    parser.add_argument("--regions", type=Path, default=Path("assets/study_area_gp.gpkg"))
    parser.add_argument("--min-sankey-area-sq-km", type=float, default=5.0)
    parser.add_argument("--max-map-size", type=int, default=3000)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def save_heatmap(frame: pd.DataFrame, output: Path, title: str, dpi: int) -> None:
    order = [item[0] for item in CLASSES.values()]
    matrix = frame.pivot(index="from_class", columns="to_class", values="area_sq_km").reindex(index=order, columns=order)
    figure, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap="YlOrRd", linewidths=0.4, square=True, ax=axis, cbar_kws={"label": "Area (km²)"})
    axis.set_title(title)
    axis.set_xlabel("Class in end year")
    axis.set_ylabel("Class in start year")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def sankey_plot(frame: pd.DataFrame, output: Path, title: str, minimum: float, include_persistence: bool, dpi: int) -> None:
    flows = frame[frame.area_sq_km >= minimum].copy()
    if not include_persistence:
        flows = flows[flows.from_code != flows.to_code]
    source_totals = flows.groupby("from_code").area_sq_km.sum().reindex(CLASSES, fill_value=0)
    target_totals = flows.groupby("to_code").area_sq_km.sum().reindex(CLASSES, fill_value=0)
    total = max(source_totals.sum(), target_totals.sum(), 1)
    gap = 0.012
    usable = 0.9 - gap * (len(CLASSES) - 1)

    def positions(totals: pd.Series) -> Dict[int, List[float]]:
        result, top = {}, 0.95
        for code in CLASSES:
            height = usable * totals[code] / total
            result[code] = [top - height, top]
            top -= height + gap
        return result

    left, right = positions(source_totals), positions(target_totals)
    left_cursor = {code: left[code][0] for code in CLASSES}
    right_cursor = {code: right[code][0] for code in CLASSES}
    figure, axis = plt.subplots(figsize=(16, 11))
    for row in flows.sort_values("area_sq_km", ascending=False).itertuples():
        height = usable * row.area_sq_km / total
        y0a, y0b = left_cursor[row.from_code], left_cursor[row.from_code] + height
        y1a, y1b = right_cursor[row.to_code], right_cursor[row.to_code] + height
        left_cursor[row.from_code] = y0b
        right_cursor[row.to_code] = y1b
        vertices = [(0.12, y0a), (0.42, y0a), (0.58, y1a), (0.88, y1a),
                    (0.88, y1b), (0.58, y1b), (0.42, y0b), (0.12, y0b), (0.12, y0a)]
        codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
                 MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4, MplPath.CLOSEPOLY]
        axis.add_patch(PathPatch(MplPath(vertices, codes), facecolor=CLASSES[row.from_code][1], alpha=0.38, edgecolor="none"))
    for x, locations, year in ((0.08, left, title.split()[0]), (0.88, right, title.split()[-1])):
        for code, (name, color) in CLASSES.items():
            bottom, top = locations[code]
            if top <= bottom:
                continue
            axis.add_patch(Rectangle((x, bottom), 0.04, top - bottom, facecolor=color, edgecolor="black", linewidth=0.5))
            align, text_x = ("right", x - 0.01) if x < 0.5 else ("left", x + 0.05)
            axis.text(text_x, (bottom + top) / 2, name.replace("_", " "), va="center", ha=align, fontsize=9)
        axis.text(x + 0.02, 0.985, year, ha="center", fontweight="bold")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    axis.set_title(title + (" — all flows" if include_persistence else " — changes only"), fontsize=15)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def dynamics_plot(frame: pd.DataFrame, output: Path, title: str, dpi: int) -> None:
    labels = frame.class_name.str.replace("_", " ").tolist()
    x = np.arange(len(frame))
    figure, axis = plt.subplots(figsize=(14, 7))
    axis.bar(x - 0.2, frame.gross_gain_ha / 100, width=0.4, label="Gross gain", color="#2ca25f")
    axis.bar(x + 0.2, -frame.gross_loss_ha / 100, width=0.4, label="Gross loss", color="#de2d26")
    axis.scatter(x, frame.net_change_ha / 100, label="Net change", color="black", zorder=3)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=35, ha="right")
    axis.set_ylabel("Area (km²); losses shown negative")
    axis.set_title(title)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def change_map(path: Path, regions_path: Path, output: Path, max_size: int, dpi: int, title: str) -> None:
    with rasterio.open(path) as source:
        scale = max(source.width / max_size, source.height / max_size, 1)
        width, height = round(source.width / scale), round(source.height / scale)
        values = source.read(1, out_shape=(height, width), resampling=Resampling.nearest)
        transform = source.transform * Affine.scale(source.width / width, source.height / height)
        extent = (source.bounds.left, source.bounds.right, source.bounds.bottom, source.bounds.top)
        crs = source.crs
    labels = {0: ("Persistence", "#d9d9d9"), 1: ("Cocoa gain", "#6B3E26"), 2: ("Cocoa loss", "#fdae6b"), 3: ("Other transition", "#756bb1")}
    rgba = np.zeros((*values.shape, 4), dtype="uint8")
    for code, (_, color) in labels.items():
        rgb = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
        rgba[values == code] = (*rgb, 255)
    regions = gpd.read_file(regions_path).to_crs(crs)
    figure, axis = plt.subplots(figsize=(11, 10))
    axis.imshow(rgba, extent=extent, origin="upper", interpolation="nearest")
    regions.boundary.plot(ax=axis, color="black", linewidth=0.7)
    axis.legend(handles=[Patch(facecolor=color, label=name) for name, color in labels.values()], loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    axis.set_title(title)
    axis.set_xlabel("Longitude")
    axis.set_ylabel("Latitude")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def regional_net_plot(frame: pd.DataFrame, output: Path, title: str, dpi: int) -> None:
    order = [item[0] for item in CLASSES.values()]
    pivot = frame.pivot(index="region", columns="class_name", values="net_change_ha").reindex(columns=order) / 100
    pivot.plot(kind="bar", figsize=(15, 8), color=[value[1] for value in CLASSES.values()])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Net change (km²)")
    plt.title(title)
    plt.xticks(rotation=0)
    plt.legend(title="Class", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close()


def grouped_heatmap(frame: pd.DataFrame, output: Path, title: str, dpi: int) -> None:
    order = list(GROUPS)
    matrix = frame.pivot(index="from_group", columns="to_group", values="area_sq_km").reindex(index=order, columns=order)
    figure, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlGnBu", square=True, ax=axis, cbar_kws={"label": "Area (km²)"})
    axis.set_title(title)
    axis.set_xlabel("Group in end year")
    axis.set_ylabel("Group in start year")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_dir, output_dir = resolve(args.input_dir), resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overall = pd.read_csv(input_dir / "transition_long_overall.csv")
    regional = pd.read_csv(input_dir / "transition_long_by_region.csv")
    dynamics = pd.read_csv(input_dir / "class_dynamics_overall.csv")
    dynamics_regional = pd.read_csv(input_dir / "class_dynamics_by_region.csv")
    grouped = pd.read_csv(input_dir / "grouped_transition_overall.csv")
    title = f"{args.start_year} to {args.end_year} LULC transitions"
    save_heatmap(overall, output_dir / "transition_heatmap_area_sq_km.png", title, args.dpi)
    save_heatmap(overall[overall.from_code != overall.to_code], output_dir / "transition_heatmap_changes_only.png", title + " (changes only)", args.dpi)
    sankey_plot(overall, output_dir / "transition_sankey_all_flows.png", f"{args.start_year} LULC flows {args.end_year}", args.min_sankey_area_sq_km, True, args.dpi)
    sankey_plot(overall, output_dir / "transition_sankey_changes_only.png", f"{args.start_year} LULC flows {args.end_year}", args.min_sankey_area_sq_km, False, args.dpi)
    dynamics_plot(dynamics, output_dir / "class_gains_losses_net.png", title, args.dpi)
    regional_net_plot(dynamics_regional, output_dir / "regional_net_change.png", title, args.dpi)
    grouped_heatmap(grouped, output_dir / "grouped_transition_heatmap.png", title + " (four functional groups)", args.dpi)
    change_map(
        input_dir / f"lulc_change_types_{args.start_year}_{args.end_year}.tif",
        resolve(args.regions), output_dir / "change_type_map.png", args.max_map_size, args.dpi, title,
    )
    for region in regional.region.unique():
        safe = region.lower().replace(" ", "_")
        save_heatmap(regional[regional.region == region], output_dir / f"transition_heatmap_{safe}.png", f"{region}: {title}", args.dpi)
    print(f"Saved transition figures to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
