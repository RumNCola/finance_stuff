import pandas as pd

from market_data.normalizer import normalize_yahoo_frame


def test_normalizes_single_ticker_response():
    frame = pd.DataFrame(
        {
            "Open": [1.0],
            "High": [2.0],
            "Low": [0.5],
            "Close": [1.5],
            "Adj Close": [1.5],
            "Volume": [10],
        },
        index=pd.to_datetime(["2024-01-01"]),
    )
    normalized = normalize_yahoo_frame(frame, "AAPL", "1d")
    assert normalized.columns[0] == "ticker"
    assert normalized.height == 1
    assert normalized["close"][0] == 1.5
