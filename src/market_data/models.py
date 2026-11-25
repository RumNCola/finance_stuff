from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import polars as pl


@dataclass(frozen=True)
class HistoryRequest:
    ticker: str
    start_date: str | None = None
    end_date: str | None = None
    period: str | None = None
    interval: str = "1d"
    include_actions: bool = True
    timeout: int = 15
    repair_data: bool = False


@dataclass
class ProviderResponse:
    data: pl.DataFrame
    requested_ticker: str
    error: str | None = None


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    ticker: str | None = None


@dataclass
class ExtractionResult:
    requested_tickers: list[str]
    successful_tickers: list[str] = field(default_factory=list)
    cached_tickers: list[str] = field(default_factory=list)
    partial_tickers: list[str] = field(default_factory=list)
    failed_tickers: list[str] = field(default_factory=list)
    observations_downloaded: int = 0
    observations_reused: int = 0
    observations_stored: int = 0
    retries: int = 0
    validation_issues: list[ValidationIssue] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    output_paths: list[str] = field(default_factory=list)
    total_bytes_written: int = 0
    execution_time: float = 0.0
    download_time: float = 0.0
    processing_time: float = 0.0
    storage_time: float = 0.0

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.validation_issues if i.severity in {"warning", "info"}]

    def summary(self) -> str:
        return (
            f"Processed {len(self.requested_tickers)} tickers: "
            f"{len(self.successful_tickers)} successful, {len(self.cached_tickers)} cached, "
            f"{len(self.failed_tickers)} failed; {self.observations_stored} observations stored."
        )


CANONICAL_SCHEMA: dict[str, pl.DataType] = {
    "ticker": pl.String,
    "timestamp": pl.Datetime("us", "UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "adj_close": pl.Float64,
    "volume": pl.Int64,
    "dividends": pl.Float64,
    "stock_splits": pl.Float64,
    "capital_gains": pl.Float64,
    "currency": pl.String,
    "exchange": pl.String,
    "interval": pl.String,
    "source": pl.String,
    "retrieved_at": pl.Datetime("us", "UTC"),
}


def utc_now() -> datetime:
    return datetime.now().astimezone().astimezone(__import__("datetime").timezone.utc)
