from __future__ import annotations

from pathlib import Path

import polars as pl


def _paths(output_path: Path, tickers: list[str] | None, interval: str) -> list[str]:
    base = output_path / "source=yahoo" / f"interval={interval}"
    if not base.exists():
        return []
    if tickers:
        return [
            str(base / f"{t.replace('/', '%2F').replace(chr(92), '%5C')}.parquet")
            for t in tickers
            if (base / f"{t.replace('/', '%2F').replace(chr(92), '%5C')}.parquet").exists()
        ]
    return [str(p) for p in base.glob("*.parquet")]


def scan_history(
    output_path: str | Path = "./data",
    tickers: list[str] | None = None,
    interval: str = "1d",
    start_date: str | None = None,
    end_date: str | None = None,
    columns: list[str] | None = None,
) -> pl.LazyFrame:
    paths = _paths(Path(output_path), tickers, interval)
    if not paths:
        return pl.DataFrame().lazy()
    lf = pl.scan_parquet(paths)
    if tickers:
        lf = lf.filter(pl.col("ticker").is_in([t.upper() for t in tickers]))
    if start_date:
        lf = lf.filter(pl.col("timestamp") >= pl.lit(start_date).str.to_datetime(time_zone="UTC"))
    if end_date:
        lf = lf.filter(pl.col("timestamp") < pl.lit(end_date).str.to_datetime(time_zone="UTC"))
    return lf.select(columns) if columns else lf


def load_history(**kwargs) -> pl.DataFrame:
    return scan_history(**kwargs).collect()


def query(sql: str, output_path: str | Path = "./data") -> pl.DataFrame:
    import duckdb

    connection = duckdb.connect()
    try:
        glob = str(Path(output_path) / "source=*" / "interval=*" / "*.parquet")
        return pl.from_arrow(
            connection.execute(sql.replace("market_history", f"read_parquet('{glob}')")).arrow()
        )
    finally:
        connection.close()
