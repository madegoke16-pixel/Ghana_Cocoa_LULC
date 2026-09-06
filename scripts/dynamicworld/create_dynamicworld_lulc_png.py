#!/usr/bin/env python3
"""Create a PNG of the original yearly Dynamic World classes inside the study area.

The PNG retains all original Dynamic World class codes 0-8 and overlays the
boundaries of Ashanti, Western, and Western North from ``study_area_gp.gpkg``.
The source raster is read at display resolution to avoid loading the full 10 m
raster into memory.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import List

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import geometry_mask, geometry_window
from rasterio.warp import Resampling

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ghana_cocoa_matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AOI = Path("assets/study_area_gp.gpkg")
DEFAULT_OUTPUT_DIR = Path("outputs/maps/dynamicworld")
NODATA_DISPLAY = 255
CLASSES = {
    0: ("Water", "#419BDF"),
    1: ("Trees", "#397D49"),
    2: ("Grass", "#88B053"),
    3: ("Flooded vegetation", "#7A87C6"),
    4: ("Crops", "#E49635"),
    5: ("Shrub and scrub", "#DFC35A"),
    6: ("Built", "#C4281B"),
    7: ("Bare", "#A59B8F"),
    8: ("Snow and ice", "#B39FE1"),
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map original yearly Dynamic World classes within the Ghana study regions."
    )
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--dynamic-world", type=Path)
    parser.add_argument("--study-area", type=Path, default=DEFAULT_AOI)
    parser.add_argument("--study-area-layer", default=None)
    parser.add_argument("--region-name-field", default="adm1_name")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-size", type=int, default=3500)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_size < 500 or args.dpi < 50:
        raise ValueError("--max-size must be at least 500 and --dpi at least 50")
    raster_path = resolve(
        args.dynamic_world
        or Path(
            f"data/processed/dynamicworld/"
            f"ghana_cocoa_dynamicworld_{args.year}_gapfilled_clipped.tif"
        )
    )
    study_path = resolve(args.study_area)
    output_path = resolve(
        args.output
        or DEFAULT_OUTPUT_DIR / f"ghana_dynamicworld_original_classes_{args.year}.png"
    )
    if not raster_path.exists():
        raise FileNotFoundError(f"Dynamic World raster not found: {raster_path}")
    if not study_path.exists():
        raise FileNotFoundError(f"Study-area file not found: {study_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {output_path}")

    study = gpd.read_file(study_path, layer=args.study_area_layer)
    if study.empty or study.crs is None:
        raise ValueError("Study area must contain geometry and have a defined CRS")
    study = study[study.geometry.notna() & ~study.geometry.is_empty].copy()
    study.geometry = study.geometry.make_valid()
    if study.empty:
        raise ValueError("Study area contains no valid geometry")

    with rasterio.open(raster_path) as source:
        if source.count < 1 or source.crs is None:
            raise ValueError("Dynamic World input must contain a georeferenced label band")
        study = study.to_crs(source.crs)
        shapes: List[dict] = [geometry.__geo_interface__ for geometry in study.geometry]
        window = geometry_window(source, shapes).round_offsets().round_lengths()
        scale = max(window.width / args.max_size, window.height / args.max_size, 1)
        width = max(1, round(window.width / scale))
        height = max(1, round(window.height / scale))
        values = source.read(
            1, window=window, out_shape=(height, width), resampling=Resampling.nearest
        )
        source_transform = source.window_transform(window)
        display_transform = source_transform * Affine.scale(
            window.width / width, window.height / height
        )
        inside = geometry_mask(
            shapes, out_shape=(height, width), transform=display_transform, invert=True
        )
        valid = inside & (values >= 0) & (values <= 8)
        display = np.full(values.shape, NODATA_DISPLAY, dtype="uint8")
        display[valid] = values[valid].astype("uint8")
        left, top = display_transform * (0, 0)
        right, bottom = display_transform * (width, height)
        extent = (left, right, bottom, top)

    masked = np.ma.masked_where(display == NODATA_DISPLAY, display)
    colors = [CLASSES[code][1] for code in range(9)]
    cmap = ListedColormap(colors)
    cmap.set_bad("white", alpha=0)
    norm = BoundaryNorm(np.arange(-0.5, 9.5, 1), cmap.N)
    figure, axis = plt.subplots(figsize=(12, 10))
    axis.imshow(
        masked, cmap=cmap, norm=norm, extent=extent,
        origin="upper", interpolation="nearest",
    )
    study.boundary.plot(ax=axis, color="black", linewidth=0.8, zorder=3)
    if args.region_name_field in study.columns:
        label_points = study.geometry.representative_point()
        for name, point in zip(study[args.region_name_field], label_points):
            axis.annotate(
                str(name), (point.x, point.y), ha="center", va="center",
                fontsize=8, fontweight="bold", color="black",
                bbox={"facecolor": "white", "alpha": 0.65, "edgecolor": "none", "pad": 1.5},
            )
    axis.set_title(f"Dynamic World LULC {args.year}\nOriginal classes within the Ghana study regions")
    axis.set_xlabel("Longitude" if study.crs.is_geographic else f"Easting ({study.crs.to_string()})")
    axis.set_ylabel("Latitude" if study.crs.is_geographic else f"Northing ({study.crs.to_string()})")
    axis.legend(
        handles=[Patch(facecolor=color, label=f"{code}: {name}") for code, (name, color) in CLASSES.items()],
        loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False,
    )
    axis.set_aspect("equal")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
