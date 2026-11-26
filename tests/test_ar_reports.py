from datetime import datetime, timezone

import polars as pl

from market_data.ar import ARStudyConfig, ARStudyResult


def test_report_contains_all_rows(tmp_path):
    result = ARStudyResult(
        config=ARStudyConfig(companies=["TEST"], p_values=(1,), min_observations=10),
        origin_dates={"TEST": datetime(2024, 1, 1, tzinfo=timezone.utc)},
        forecasts=pl.DataFrame(
            {
                "ticker": ["TEST", "TEST"],
                "model": ["ar_1", "ar_1"],
                "step": [1, 2],
                "predicted_price": [100.0, 101.0],
                "actual_price": [100.5, 101.5],
            }
        ),
        comparison=pl.DataFrame({"ticker": ["TEST"], "model": ["ar_1"], "rmse": [1.2]}),
        coefficients=pl.DataFrame(
            {"ticker": ["TEST"], "model": ["ar_1"], "lag": [1], "weight": [0.25]}
        ),
        selected_models=pl.DataFrame({"selected_p": [1]}),
    )
    path = result.write_report(tmp_path / "study.md")
    text = path.read_text(encoding="utf-8")
    assert "| 2 | 101.0 |" in text
    assert "| ar_1 | 1.2 |" in text
    assert "| 1 | 0.25 |" in text
