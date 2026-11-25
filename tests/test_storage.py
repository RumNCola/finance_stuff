import polars as pl

from market_data.models import CANONICAL_SCHEMA
from market_data.storage import write_history_atomic


def test_atomic_parquet_round_trip(tmp_path):
    frame = pl.DataFrame({"ticker": ["AAPL"], "timestamp": ["2024-01-01"]}).with_columns(
        pl.col("timestamp").str.to_datetime(time_zone="UTC")
    )
    for name, dtype in CANONICAL_SCHEMA.items():
        if name not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(name))
    frame = frame.select(list(CANONICAL_SCHEMA))
    path = tmp_path / "a.parquet"
    assert write_history_atomic(path, frame) > 0
    assert pl.read_parquet(path).height == 1
