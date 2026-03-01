"""ETL utilities shared across pipeline stages."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("diageno.etl")


def sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest of a file (stream-safe)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(dir_path: Path) -> str:
    """Return combined SHA-256 for all files in a directory tree."""
    h = hashlib.sha256()
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            h.update(sha256_file(p).encode())
    return h.hexdigest()


def load_json(path: Path) -> Any:
    """Load and return JSON from a file."""
    with open(path) as f:
        return json.load(f)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Save a DataFrame as Parquet, creating parent dirs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %d rows → %s", len(df), path)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file into a DataFrame."""
    return pd.read_parquet(path, engine="pyarrow")
