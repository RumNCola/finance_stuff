from __future__ import annotations

import os
import tempfile
from pathlib import Path

import polars as pl

from .models import CANONICAL_SCHEMA


def ticker_path(root: Path, source: str, interval: str, ticker: str) -> Path:
    safe = ticker.replace("/", "%2F").replace("\\", "%5C")
    return root / f"source={source}" / f"interval={interval}" / f"{safe}.parquet"


def merge_history(existing: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    combined = pl.concat([existing, incoming], how="vertical_relaxed")
    return combined.unique(
        ["source", "ticker", "interval", "timestamp"], keep="last", maintain_order=False
    ).sort("timestamp")


def read_history(path: Path) -> pl.DataFrame:
    return pl.read_parquet(path) if path.exists() else pl.DataFrame(schema=CANONICAL_SCHEMA)


def write_history_atomic(
    path: Path, frame: pl.DataFrame, compression: str = "zstd", compression_level: int | None = None
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        kwargs = {"compression": compression}
        if compression_level is not None:
            kwargs["compression_level"] = compression_level
        frame.write_parquet(temp_path, **kwargs)
        os.replace(temp_path, path)
        return path.stat().st_size
    finally:
        temp_path.unlink(missing_ok=True)
