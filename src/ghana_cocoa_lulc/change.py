"""Core land-cover transition calculations."""

import numpy as np
import pandas as pd


def transition_table(
    baseline: np.ndarray,
    comparison: np.ndarray,
    pixel_area_ha: float,
    nodata_value: int = 0,
) -> pd.DataFrame:
    """Summarize valid pixel transitions and their area in hectares."""
    if baseline.shape != comparison.shape:
        raise ValueError("Baseline and comparison arrays must have equal shapes")
    if pixel_area_ha <= 0:
        raise ValueError("pixel_area_ha must be positive")

    valid = (baseline != nodata_value) & (comparison != nodata_value)
    frame = pd.DataFrame(
        {"from_class": baseline[valid].ravel(), "to_class": comparison[valid].ravel()}
    )
    if frame.empty:
        return pd.DataFrame(columns=["from_class", "to_class", "pixel_count", "area_ha"])

    result = (
        frame.groupby(["from_class", "to_class"], sort=True)
        .size()
        .rename("pixel_count")
        .reset_index()
    )
    result["area_ha"] = result["pixel_count"] * pixel_area_ha
    return result

