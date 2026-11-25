import pytest
from pydantic import ValidationError

from market_data.config import ExtractionConfig


def test_normalizes_and_deduplicates_tickers():
    config = ExtractionConfig(tickers=[" aapl ", "AAPL", "MSFT"])
    assert config.tickers == ["AAPL", "MSFT"]


def test_rejects_period_and_dates():
    with pytest.raises(ValidationError):
        ExtractionConfig(tickers=["AAPL"], period="1y", start_date="2020-01-01")
