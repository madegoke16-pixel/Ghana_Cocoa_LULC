"""Shared definitions for 2017-2025 LULC transition analysis."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NODATA = 255
TRANSITION_NODATA = 65535
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
GROUPS = {
    "water_and_wetland": {0, 3},
    "natural_vegetation": {1, 2, 5},
    "agriculture": {4, 8},
    "developed_and_bare": {6, 7},
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path
