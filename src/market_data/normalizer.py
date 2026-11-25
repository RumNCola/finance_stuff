from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import polars as pl

from .models import CANONICAL_SCHEMA


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=CANONICAL_SCHEMA)


def normalize_yahoo_frame(
    frame: Any, ticker: str, interval: str, source: str = "yahoo"
) -> pl.DataFrame:
    """Convert a yfinance pandas response to the canonical long Polars schema."""
    if frame is None or getattr(frame, "empty", False):
        return _empty()
    columns = frame.columns
    if hasattr(columns, "nlevels") and columns.nlevels > 1:
        # yfinance returns either (field, ticker) or (ticker, field).
        fields = {str(x).lower() for x in columns.get_level_values(0)}
        if "open" not in fields:
            frame = frame.swaplevel(axis=1)
        frame = (
            frame.xs(ticker, axis=1, level=1, drop_level=True)
            if ticker in frame.columns.get_level_values(1)
            else frame.xs(ticker, axis=1, level=0, drop_level=True)
        )
    raw = pl.from_pandas(frame.reset_index())
    rename = {c: c.lower().replace(" ", "_") for c in raw.columns}
    raw = raw.rename(rename)
    if "date" in raw.columns and "timestamp" not in raw.columns:
        raw = raw.rename({"date": "timestamp"})
    if "datetime" in raw.columns and "timestamp" not in raw.columns:
        raw = raw.rename({"datetime": "timestamp"})
    expressions = []
    for name in CANONICAL_SCHEMA:
        if name in {"ticker", "interval", "source", "retrieved_at"}:
            continue
        if name in raw.columns:
            expr = pl.col(name)
            if name == "timestamp":
                expr = expr.cast(pl.Datetime("us", "UTC"), strict=False)
            elif name == "volume":
                expr = expr.cast(pl.Int64, strict=False)
            else:
                expr = expr.cast(pl.Float64, strict=False)
            expressions.append(expr.alias(name))
        else:
            expressions.append(pl.lit(None, dtype=CANONICAL_SCHEMA[name]).alias(name))
    normalized = raw.select(expressions).with_columns(
        pl.lit(ticker).alias("ticker"),
        pl.lit(interval).alias("interval"),
        pl.lit(source).alias("source"),
        pl.lit(datetime.now(timezone.utc))
        .cast(CANONICAL_SCHEMA["retrieved_at"])
        .alias("retrieved_at"),
    )
    return normalized.select(list(CANONICAL_SCHEMA)).sort("timestamp")
