"""Project configuration helpers."""

from pathlib import Path
from typing import Any, Union

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML mapping relative to the project root when needed."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    with candidate.open(encoding="utf-8") as stream:
        content = yaml.safe_load(stream)
    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {candidate}")
    return content


def load_project_config() -> dict[str, Any]:
    """Return the main project configuration."""
    return load_yaml("config/project.yaml")
