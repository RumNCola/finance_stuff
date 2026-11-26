from datetime import datetime, timezone

import numpy as np
import polars as pl

import market_data.ar as ar


def _history() -> pl.DataFrame:
    timestamps = [datetime(2020, 1, 1, tzinfo=timezone.utc)]
    for _ in range(80):
        timestamps.append(
            timestamps[-1].replace(day=timestamps[-1].day)
            + __import__("datetime").timedelta(days=1)
        )
    timestamps = [timestamp for timestamp in timestamps if timestamp.weekday() < 5][:60]
    prices = (100 * np.cumprod(1 + np.repeat(0.01, len(timestamps)))).tolist()
    return pl.DataFrame(
        {"ticker": ["TEST"] * len(timestamps), "timestamp": timestamps, "adj_close": prices}
    )


def test_ar_study_returns_coefficients_and_historical_metrics(monkeypatch):
    history = _history()
    monkeypatch.setattr(ar, "load_history", lambda **kwargs: history)
    result = ar.run_ar_study(
        companies=["TEST"],
        horizon=3,
        origin_date="2020-02-10T00:00:00+00:00",
        p_values=(1, 2),
        min_observations=10,
    )
    assert result.errors == {}
    assert result.forecasts.height == 18
    assert set(result.coefficients["p"].to_list()) == {1, 2}
    assert result.comparison["evaluated_steps"].max() == 3
    assert result.selected_models["selected_p"].to_list()[0] in {1, 2}
    assert "geometric_ar_1" in set(result.forecasts["model"].to_list())
    assert result.selected_models["best_ar_p"].to_list()[0] in {1, 2}
