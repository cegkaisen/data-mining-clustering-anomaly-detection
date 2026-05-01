"""General utility functions shared by notebooks and modules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def project_root(start: Path | None = None) -> Path:
    """Find the project root by walking upward until ``README.md`` is found.

    Args:
        start: Directory to start from. Defaults to this file's location.

    Returns:
        Path to the project root.
    """
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for path in [current, *current.parents]:
        if (path / "README.md").exists():
            return path
    raise FileNotFoundError("Could not locate project root containing README.md.")


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Load a CSV file from a caller-provided path.

    The path is intentionally supplied by the caller so this function does not
    depend on hardcoded project data locations.
    """
    return pd.read_csv(Path(path), **kwargs)


def ensure_directory(path: str | Path) -> Path:
    """Create an output directory if needed and return it as a ``Path``."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_dataframe(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> Path:
    """Save a dataframe to CSV at a caller-provided path."""
    output_path = Path(path)
    ensure_directory(output_path.parent)
    df.to_csv(output_path, index=False, **kwargs)
    return output_path
