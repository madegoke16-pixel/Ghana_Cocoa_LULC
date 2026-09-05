#!/usr/bin/env python3
"""Build cocoa-positive and pseudo-natural samples inside the DW tree mask.

Natural-tree samples are randomly drawn from Dynamic World tree pixels outside
a buffer around known cocoa points. They are pseudo-labels and must not replace
independently interpreted natural-tree reference data for final validation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from common import FEATURE_NAMES, FLOAT_NODATA, log, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cocoa and pseudo-natural tree training samples.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--season", choices=("djf", "amj"), default="djf")
    parser.add_argument("--cocoa-points", type=Path, default=Path("assets/Cocoa_500_samples_2017.kml"))
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--negative-ratio", type=float, default=2.0)
    parser.add_argument("--exclusion-buffer-m", type=float, default=1000.0)
    parser.add_argument("--spatial-block-km", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def valid_vector(values: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all(values != FLOAT_NODATA))


def main() -> int:
    args = parse_args()
    if args.negative_ratio <= 0 or args.exclusion_buffer_m < 0 or args.spatial_block_km <= 0:
        raise ValueError("Ratios/block size must be positive and the exclusion buffer cannot be negative")
    feature_dir = resolve(args.feature_dir or Path(f"data/interim/cocoa_classification/{args.year}/{args.season}/features"))
    mask_dir = resolve(args.mask_dir or Path(f"data/interim/cocoa_classification/{args.year}/{args.season}/tree_masks"))
    output = resolve(args.output or Path(f"data/interim/cocoa_classification/{args.year}/{args.season}/training_samples.gpkg"))
    feature_tiles = sorted(feature_dir.glob(f"ghana_cocoa_indices_{args.year}_*.tif"))
    if not feature_tiles:
        raise FileNotFoundError(f"No feature tiles found in {feature_dir}")
    points = gpd.read_file(resolve(args.cocoa_points))
    points = points[points.geometry.notna() & ~points.geometry.is_empty].copy()
    if points.empty or points.crs is None or not points.geom_type.eq("Point").all():
        raise ValueError("Cocoa reference file must contain georeferenced Point features")
    points["source_id"] = np.arange(len(points))

    with rasterio.open(feature_tiles[0]) as reference:
        target_crs = reference.crs
    cocoa = points.to_crs(target_crs)
    cocoa_xy = np.column_stack((cocoa.geometry.x, cocoa.geometry.y))
    positives: Dict[int, dict] = {}

    for feature_path in feature_tiles:
        mask_path = mask_dir / feature_path.name.replace("indices", "dw_tree")
        if not mask_path.exists():
            raise FileNotFoundError(f"Paired tree mask missing: {mask_path}")
        with rasterio.open(feature_path) as features, rasterio.open(mask_path) as mask:
            minx, miny, maxx, maxy = features.bounds
            inside = cocoa.geometry.x.between(minx, maxx, inclusive="left") & cocoa.geometry.y.between(miny, maxy, inclusive="left")
            for row in cocoa[inside].itertuples():
                pixel_row, pixel_col = features.index(row.geometry.x, row.geometry.y)
                if not (0 <= pixel_row < features.height and 0 <= pixel_col < features.width):
                    continue
                if mask.read(1, window=((pixel_row, pixel_row + 1), (pixel_col, pixel_col + 1)))[0, 0] != 1:
                    continue
                values = features.read(window=((pixel_row, pixel_row + 1), (pixel_col, pixel_col + 1)))[:, 0, 0]
                if valid_vector(values):
                    positives.setdefault(
                        row.source_id,
                        {"source_id": row.source_id, "label": 1, "class_name": "cocoa", "x": row.geometry.x, "y": row.geometry.y,
                         **dict(zip(FEATURE_NAMES, values.astype(float)))},
                    )
    if len(positives) < 20:
        raise RuntimeError(f"Only {len(positives)} cocoa points fall on valid Dynamic World tree pixels; inspect alignment and labels")
    log(f"Retained {len(positives)}/{len(cocoa)} cocoa points inside the valid DW tree mask")

    requested_negatives = int(np.ceil(len(positives) * args.negative_ratio))
    rng = np.random.default_rng(args.seed)
    candidates: List[dict] = []
    per_tile = max(20, int(np.ceil(requested_negatives * 8 / len(feature_tiles))))
    buffer_squared = args.exclusion_buffer_m ** 2
    for feature_path in feature_tiles:
        mask_path = mask_dir / feature_path.name.replace("indices", "dw_tree")
        with rasterio.open(feature_path) as features, rasterio.open(mask_path) as mask:
            tree_rows, tree_cols = np.where(mask.read(1) == 1)
            if not len(tree_rows):
                continue
            chosen = rng.choice(len(tree_rows), size=min(per_tile, len(tree_rows)), replace=False)
            rows, cols = tree_rows[chosen], tree_cols[chosen]
            xs, ys = rasterio.transform.xy(features.transform, rows, cols, offset="center")
            values = features.read()[:, rows, cols]
            for col_index, (x, y) in enumerate(zip(xs, ys)):
                if np.min((cocoa_xy[:, 0] - x) ** 2 + (cocoa_xy[:, 1] - y) ** 2) < buffer_squared:
                    continue
                vector = values[:, col_index]
                if valid_vector(vector):
                    candidates.append(
                        {"source_id": None, "label": 0, "class_name": "pseudo_natural", "x": x, "y": y,
                         **dict(zip(FEATURE_NAMES, vector.astype(float)))}
                    )
    if len(candidates) < requested_negatives:
        raise RuntimeError(f"Found only {len(candidates)} eligible pseudo-natural pixels; requested {requested_negatives}")
    selected = rng.choice(len(candidates), requested_negatives, replace=False)
    records = list(positives.values()) + [candidates[index] for index in selected]
    samples = gpd.GeoDataFrame(
        records,
        geometry=gpd.points_from_xy([record["x"] for record in records], [record["y"] for record in records]),
        crs=target_crs,
    ).drop(columns=["x", "y"])
    block_m = args.spatial_block_km * 1000.0
    samples["spatial_group"] = (
        np.floor(samples.geometry.x / block_m).astype(int).astype(str) + "_" +
        np.floor(samples.geometry.y / block_m).astype(int).astype(str)
    )
    samples["label_source"] = np.where(
        samples["label"] == 1, "field/reference cocoa point", f"DW tree pixel >={args.exclusion_buffer_m:g} m from cocoa points"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    samples.to_file(output, layer="training_samples", driver="GPKG")
    samples.drop(columns="geometry").to_csv(output.with_suffix(".csv"), index=False)
    log(f"Saved {len(samples)} samples ({len(positives)} cocoa, {requested_negatives} pseudo-natural): {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

