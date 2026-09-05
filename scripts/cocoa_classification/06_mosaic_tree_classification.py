#!/usr/bin/env python3
"""Mosaic tiled cocoa/tree predictions into final classified rasters."""

from __future__ import annotations

import argparse
from pathlib import Path

import rasterio
from rasterio.merge import merge

from common import FLOAT_NODATA, log, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic cocoa classification and probability tiles.")
    parser.add_argument("--year", type=int, default=2017)
    parser.add_argument("--season", choices=("djf", "wet", "annual"), default="annual")
    parser.add_argument("--model", choices=("random_forest", "xgboost", "mlp"), default="random_forest")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/cocoa_classification/mosaics"))
    parser.add_argument("--memory-mb", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def mosaic_files(paths: list, destination: Path, nodata: float, memory_mb: int) -> None:
    sources = [rasterio.open(path) for path in paths]
    temporary = destination.with_name(destination.stem + ".part.tif")
    try:
        reference = sources[0]
        merge(
            sources, nodata=nodata, dtype=reference.dtypes[0], method="first",
            target_aligned_pixels=True, mem_limit=memory_mb, dst_path=temporary,
            dst_kwds={"driver": "GTiff", "compress": "DEFLATE", "tiled": True, "BIGTIFF": "IF_SAFER"},
        )
        temporary.replace(destination)
    finally:
        for source in sources:
            source.close()


def main() -> int:
    args = parse_args()
    input_dir = resolve(args.input_dir or Path(f"data/processed/cocoa_classification/{args.year}/{args.season}/{args.model}"))
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    products = {
        "tree_class": (f"*tree_class_{args.model}*.tif", 255),
        "cocoa_probability": (f"*cocoa_probability_{args.model}*.tif", FLOAT_NODATA),
    }
    for product, (pattern, nodata) in products.items():
        paths = sorted(input_dir.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"No tiles match {input_dir / pattern}")
        destination = output_dir / f"ghana_cocoa_{product}_{args.model}_{args.season}_{args.year}.tif"
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists; use --overwrite: {destination}")
        log(f"Mosaicking {len(paths)} {product} tiles")
        mosaic_files(paths, destination, nodata, args.memory_mb)
        log(f"Saved {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
