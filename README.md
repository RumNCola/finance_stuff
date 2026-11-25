# Historical Market Data Extraction Engine

Reusable research infrastructure for downloading historical market data from Yahoo Finance, validating it, storing it as Parquet, and querying it with Polars or DuckDB.

## Installation

The project supports Python 3.11 and newer, including Python 3.15. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Quick start

```python
from market_data import extract_history, load_history

result = extract_history(
    tickers=["SQM-B.SN", "AAPL", "MSFT"],
    start_date="2010-01-01",
    end_date="2026-01-01",
    interval="1d",
    output_path="./data",
)
print(result.summary())

prices = load_history(
    output_path="./data",
    tickers=["AAPL"],
    columns=["timestamp", "ticker", "adj_close"],
)
```

The public API returns Polars DataFrames. Yahoo-specific Pandas objects are confined to the provider boundary and are converted immediately.

## Architecture

`extract_history` validates configuration, downloads through the provider interface, normalizes to a canonical Polars schema, validates observations, atomically merges one Parquet file per ticker and interval, and records the committed dataset in SQLite. The provider interface is deliberately small so another data source can be added without changing storage or query code.

Storage is organized as `data/source=yahoo/interval=1d/TICKER.parquet`, with metadata in `data/metadata/registry.sqlite`. Writes use a temporary file followed by replacement; an empty response never replaces valid cached data. Existing observations are merged using `(source, ticker, interval, timestamp)` as the deterministic uniqueness key.

## Querying

```python
from market_data import scan_history, query

lazy = scan_history(output_path="./data", tickers=["AAPL", "MSFT"])
prices = lazy.collect()

sql_result = query(
    "SELECT ticker, timestamp, adj_close FROM market_history ORDER BY ticker, timestamp",
    output_path="./data",
)
```

`scan_history` uses `pl.scan_parquet`, enabling projection and predicate pushdown. `query` registers the Parquet dataset through DuckDB and returns a Polars DataFrame.

## CLI

```bash
market-data download --tickers AAPL,MSFT --start 2010-01-01 --end 2026-01-01
market-data status
market-data validate
market-data inspect
```

## Testing and development

The offline test suite uses temporary directories and mocked provider responses:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

The benchmark scaffold in `benchmarks/benchmark_download.py` is intentionally opt-in because it performs live provider requests. Do not interpret its output as a performance claim without recording the environment and run parameters.

## Data quality and Yahoo Finance limitations

Validation reports empty responses, duplicate keys, non-finite values, OHLC inconsistencies and negative volumes. It does not fill, interpolate or silently delete observations. Weekends, exchange holidays, illiquid sessions, listing dates and delisted securities are not automatically treated as errors.

Yahoo Finance may rate-limit requests, provide restricted intraday history, revise adjusted prices after corporate actions, omit delisted securities, and contain survivorship bias. The end date follows Yahoo's exclusive-end convention. Users are responsible for complying with Yahoo Finance terms, licensing and redistribution restrictions.

This initial implementation has a conservative refresh policy: `force_refresh=True` replaces the local ticker history with the newly retrieved range, while the normal path merges and deduplicates observations. A production deployment should schedule explicit refreshes for revised corporate-action histories and retain audit logs for those replacements.
