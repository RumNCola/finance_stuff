from __future__ import annotations

import math

import polars as pl

from .models import CANONICAL_SCHEMA, ValidationIssue


def validate_history(frame: pl.DataFrame) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if frame.is_empty():
        return [ValidationIssue("warning", "empty", "Provider returned no observations")]
    missing = [c for c in CANONICAL_SCHEMA if c not in frame.columns]
    if missing:
        issues.append(ValidationIssue("critical", "schema", f"Missing columns: {missing}"))
        return issues
    key = ["source", "ticker", "interval", "timestamp"]
    if frame.select(key).is_duplicated().any():
        issues.append(
            ValidationIssue("critical", "duplicates", "Duplicate canonical keys detected")
        )
    for col in ["open", "high", "low", "close", "adj_close"]:
        if (
            frame[col]
            .drop_nulls()
            .map_elements(lambda x: not math.isfinite(x), return_dtype=pl.Boolean)
            .any()
        ):
            issues.append(ValidationIssue("critical", "non_finite", f"Non-finite values in {col}"))
    if frame.filter(
        (pl.col("high") < pl.max_horizontal("open", "close", "low"))
        | (pl.col("low") > pl.min_horizontal("open", "close", "high"))
    ).height:
        issues.append(
            ValidationIssue("warning", "ohlc", "Some OHLC relationships are inconsistent")
        )
    if frame.filter(pl.col("volume") < 0).height:
        issues.append(ValidationIssue("warning", "volume", "Negative trading volume detected"))
    return issues
