"""Shared helpers for tiled cocoa-versus-natural-tree classification."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, Tuple

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BAND_NAMES = ("B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12")
FEATURE_NAMES = ("NDVI", "EVI", "NDRE", "NDRE2", "NDMI", "NBR", "GNDVI", "SAVI", "RECI", "IRECI")
S2_NODATA = 65535
FLOAT_NODATA = -9999.0


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def log(message: str) -> None:
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def paired_name(s2_path: Path, source_token: str, target_token: str) -> str:
    if source_token not in s2_path.stem:
        raise ValueError(f"Cannot derive paired filename from {s2_path.name!r}")
    return s2_path.stem.replace(source_token, target_token) + ".tif"


def iter_windows(dataset: rasterio.io.DatasetReader) -> Iterator[Tuple[Tuple[int, int], rasterio.windows.Window]]:
    yield from dataset.block_windows(1)


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype="float32")
    np.divide(numerator, denominator, out=result, where=np.abs(denominator) > 1e-8)
    return result


def calculate_features(reflectance: np.ndarray) -> np.ndarray:
    """Calculate indices from B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12 reflectance."""
    b2, b3, b4, b5, b6, b7, b8, b8a, b11, b12 = reflectance
    features = np.stack(
        [
            safe_ratio(b8 - b4, b8 + b4),
            2.5 * safe_ratio(b8 - b4, b8 + 6.0 * b4 - 7.5 * b2 + 1.0),
            safe_ratio(b8a - b5, b8a + b5),
            safe_ratio(b8a - b6, b8a + b6),
            safe_ratio(b8 - b11, b8 + b11),
            safe_ratio(b8 - b12, b8 + b12),
            safe_ratio(b8 - b3, b8 + b3),
            1.5 * safe_ratio(b8 - b4, b8 + b4 + 0.5),
            safe_ratio(b8a, b5) - 1.0,
            (b7 - b4) * safe_ratio(b6, b5),
        ]
    ).astype("float32")
    features[~np.isfinite(features)] = np.nan
    return features


def tiled_profile(reference: rasterio.io.DatasetReader, count: int, dtype: str, nodata: float) -> Dict[str, object]:
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff", count=count, dtype=dtype, nodata=nodata,
        compress="DEFLATE", predictor=3 if dtype.startswith("float") else 1,
        tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
    )
    return profile


def write_json_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(".part.json")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)

