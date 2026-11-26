# Historical Market Data Extraction Engine

Reusable research infrastructure for downloading historical market data from Yahoo Finance, validating it, storing it as Parquet, and querying it with Polars or DuckDB.

## Installation

The project supports Python 3.11 and newer, including Python 3.15. From the repository root:

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Run example scripts from the `Sandbox` root with the same interpreter used for installation:

```powershell
.\.venv\Scripts\python.exe examples\international_market.py
.\.venv\Scripts\python.exe examples\chilean_market.py
.\.venv\Scripts\python.exe examples\query_data.py
```

If the virtual environment was created with an interpreter that has since been removed, rebuild it with `py -3.15 -m venv --clear .venv`, then reinstall. The pip upgrade notice is informational and does not indicate a project failure.

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

## Autoregressive study module

The AR study module is designed for learning and model comparison. It calculates simple returns from stored prices, fits ordinary AR(p), mean-adjusted AR(p), and geometric-lag AR(p) models, searches AR(1) through AR(10) by default, reports lag weights, selects the best candidate by AIC or BIC, and compounds predicted returns back into prices.

```python
from market_data import run_ar_study

backtest = run_ar_study(
    companies=["AAPL", "MSFT"],
    horizon=10,
    origin_date="2024-01-02",
    output_path="./data",
    max_p=10,
    selection_criterion="bic",
)

print(backtest.forecasts)  # every model's predicted returns/prices
print(backtest.comparison)  # MAE, RMSE, MAPE and direction accuracy
print(backtest.coefficients)  # intercept, mean and lag weights
print(backtest.selected_models)  # selected p/model per company
```

When `origin_date` is supplied, the model only trains on observations through that date and evaluates the next `horizon` stored observations. When it is omitted, the origin is each ticker's latest stored observation and the output is a genuine forward forecast; comparison metrics are null because future actuals do not exist yet. The module does not claim that lower historical error implies future profitability.

The default selection criterion is BIC: among the tested ordinary AR, mean-adjusted AR, and geometric-lag candidates, the model with the lowest BIC is selected. `best_ar_p` reports the best lag length among ordinary AR models only. For historical studies, RMSE and directional accuracy compare forecasts with held-out prices; they are evaluation metrics, not the in-sample selection rule.

Calling `write_report("ar_study_report.md")` also creates an `ar_study_plots/` directory. It writes one two-panel PNG per ticker: predicted versus actual prices on the left, and the best comparison metric on the right. Each graph contains at most six models, ranked by historical RMSE or, for forward-only studies, by BIC.

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
