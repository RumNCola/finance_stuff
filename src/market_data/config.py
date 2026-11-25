from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .exceptions import ConfigurationError


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    tickers: list[str] = Field(min_length=1)
    start_date: str | None = None
    end_date: str | None = None
    period: str | None = None
    interval: str = "1d"
    output_path: Path = Path("./data")
    provider: Literal["yahoo"] = "yahoo"
    workers: int = Field(default=8, ge=1, le=64)
    batch_size: int = Field(default=50, ge=1)
    compression: str = "zstd"
    compression_level: int | None = Field(default=None, ge=1)
    max_retries: int = Field(default=3, ge=1)
    timeout: int = Field(default=15, ge=1)
    include_actions: bool = True
    force_refresh: bool = False
    repair_data: bool = False
    validate_data: bool = Field(default=True, alias="validate")

    @model_validator(mode="after")
    def validate_ranges(self) -> "ExtractionConfig":
        if self.period and (self.start_date or self.end_date):
            raise ConfigurationError("period cannot be combined with start_date or end_date")
        if self.end_date and self.start_date and self.end_date <= self.start_date:
            raise ConfigurationError("end_date must be after start_date")
        cleaned = []
        seen = set()
        for ticker in self.tickers:
            normalized = ticker.strip().upper()
            if normalized and normalized not in seen:
                cleaned.append(normalized)
                seen.add(normalized)
        if not cleaned:
            raise ConfigurationError("tickers must contain at least one non-empty symbol")
        self.tickers = cleaned
        return self
