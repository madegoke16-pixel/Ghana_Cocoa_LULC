#!/usr/bin/env python3
"""Predict natural tree versus cocoa only within Dynamic World tree pixels."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import rasterio

from common import FEATURE_NAMES, FLOAT_NODATA, log, resolve, tiled_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply a trained model to tiled tree-mask features.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--season", choices=("djf", "amj"), default="djf")
    parser.add_argument("--model", choices=("random_forest", "xgboost", "mlp"), default="random_forest")
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--feature-dir", type=Path)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.threshold < 1:
        raise ValueError("--threshold must be between 0 and 1")
    model_path = resolve(args.model_file or Path(f"models/cocoa_classification/{args.year}/{args.season}/{args.model}.joblib"))
    feature_dir = resolve(args.feature_dir or Path(f"data/interim/cocoa_classification/{args.year}/{args.season}/features"))
    mask_dir = resolve(args.mask_dir or Path(f"data/interim/cocoa_classification/{args.year}/{args.season}/tree_masks"))
    output_dir = resolve(args.output_dir or Path(f"data/processed/cocoa_classification/{args.year}/{args.season}/{args.model}"))
    artifact = joblib.load(model_path)
    model, names = artifact["pipeline"], tuple(artifact["feature_names"])
    if names != FEATURE_NAMES:
        raise ValueError(f"Model feature order {names} does not match raster order {FEATURE_NAMES}")
    tiles = sorted(feature_dir.glob(f"ghana_cocoa_indices_{args.year}_*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No feature tiles found in {feature_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, feature_path in enumerate(tiles, 1):
        mask_path = mask_dir / feature_path.name.replace("indices", "dw_tree")
        class_path = output_dir / feature_path.name.replace("indices", f"tree_class_{args.model}")
        probability_path = output_dir / feature_path.name.replace("indices", f"cocoa_probability_{args.model}")
        if class_path.exists() and probability_path.exists() and not args.overwrite:
            log(f"[{index}/{len(tiles)}] Outputs exist; skipping {feature_path.name}")
            continue
        with rasterio.open(feature_path) as features, rasterio.open(mask_path) as mask:
            class_profile = tiled_profile(features, 1, "uint8", 255)
            probability_profile = tiled_profile(features, 1, "float32", FLOAT_NODATA)
            class_tmp = class_path.with_name(class_path.stem + ".part.tif")
            probability_tmp = probability_path.with_name(probability_path.stem + ".part.tif")
            with rasterio.open(class_tmp, "w", **class_profile) as classes, rasterio.open(probability_tmp, "w", **probability_profile) as probabilities:
                classes.set_band_description(1, "0_non_tree_1_natural_tree_2_cocoa_255_nodata")
                probabilities.set_band_description(1, "cocoa_probability_within_dw_tree")
                for _, window in features.block_windows(1):
                    values = features.read(window=window)
                    tree = mask.read(1, window=window)
                    valid = (tree == 1) & np.all(np.isfinite(values) & (values != FLOAT_NODATA), axis=0)
                    output_class = np.where(tree == 255, 255, 0).astype("uint8")
                    output_probability = np.full(tree.shape, FLOAT_NODATA, dtype="float32")
                    if valid.any():
                        matrix = values[:, valid].T
                        cocoa_probability = model.predict_proba(matrix)[:, 1]
                        output_probability[valid] = cocoa_probability
                        output_class[valid] = np.where(cocoa_probability >= args.threshold, 2, 1)
                    classes.write(output_class, 1, window=window)
                    probabilities.write(output_probability, 1, window=window)
            class_tmp.replace(class_path)
            probability_tmp.replace(probability_path)
        log(f"[{index}/{len(tiles)}] Predicted {class_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

