from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_random_exponential

from .config import ExtractionConfig
from .models import ExtractionResult, HistoryRequest
from .providers import MarketDataProvider, YahooFinanceProvider
from .registry import Registry
from .storage import merge_history, read_history, ticker_path, write_history_atomic
from .validator import validate_history

logger = logging.getLogger(__name__)


def extract_history(
    *,
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    period: str | None = None,
    interval: str = "1d",
    output_path: str | Path = "./data",
    provider: str = "yahoo",
    workers: int = 8,
    batch_size: int = 50,
    compression: str = "zstd",
    compression_level: int | None = None,
    max_retries: int = 3,
    timeout: int = 15,
    include_actions: bool = True,
    force_refresh: bool = False,
    repair_data: bool = False,
    validate: bool = True,
    provider_instance: MarketDataProvider | None = None,
) -> ExtractionResult:
    config = ExtractionConfig(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        period=period,
        interval=interval,
        output_path=Path(output_path),
        provider=provider,
        workers=workers,
        batch_size=batch_size,
        compression=compression,
        compression_level=compression_level,
        max_retries=max_retries,
        timeout=timeout,
        include_actions=include_actions,
        force_refresh=force_refresh,
        repair_data=repair_data,
        validate=validate,
    )
    result = ExtractionResult(config.tickers)
    started = time.perf_counter()
    provider_obj = provider_instance or YahooFinanceProvider()
    root = config.output_path
    registry = Registry(root)

    def download(ticker: str):
        request = HistoryRequest(
            ticker,
            config.start_date,
            config.end_date,
            config.period,
            config.interval,
            config.include_actions,
            config.timeout,
            config.repair_data,
        )
        retrying = Retrying(
            stop=stop_after_attempt(config.max_retries),
            wait=wait_random_exponential(multiplier=0.5, max=8),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )
        return retrying(lambda: provider_obj.download_history(request))

    download_started = time.perf_counter()
    responses = {}
    with ThreadPoolExecutor(max_workers=min(config.workers, len(config.tickers))) as pool:
        futures = {pool.submit(download, ticker): ticker for ticker in config.tickers}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                responses[ticker] = future.result()
            except Exception as exc:
                result.failed_tickers.append(ticker)
                result.errors[ticker] = str(exc)
    result.download_time = time.perf_counter() - download_started
    processing_started = time.perf_counter()
    for ticker, response in responses.items():
        path = ticker_path(root, "yahoo", config.interval, ticker)
        existing = (
            read_history(path)
            if path.exists() and not config.force_refresh
            else pl.DataFrame(schema=response.data.schema)
        )
        if response.data.is_empty() and not existing.is_empty():
            result.cached_tickers.append(ticker)
            result.observations_reused += existing.height
            continue
        issues = validate_history(response.data) if config.validate_data else []
        result.validation_issues.extend(issues)
        if any(i.severity == "critical" for i in issues):
            result.failed_tickers.append(ticker)
            result.errors[ticker] = "critical validation issue"
            continue
        merged = merge_history(existing, response.data)
        result.observations_downloaded += response.data.height
        result.observations_reused += existing.height
        result.successful_tickers.append(ticker)
        storage_started = time.perf_counter()
        size = write_history_atomic(path, merged, config.compression, config.compression_level)
        result.storage_time += time.perf_counter() - storage_started
        result.total_bytes_written += size
        result.observations_stored += merged.height
        result.output_paths.append(str(path))
        registry.upsert(
            source="yahoo",
            ticker=ticker,
            interval=config.interval,
            file_path=str(path),
            frame=merged,
            file_size=size,
        )
    result.processing_time = time.perf_counter() - processing_started
    result.execution_time = time.perf_counter() - started
    return result
