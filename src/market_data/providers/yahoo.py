from __future__ import annotations

from ..exceptions import ProviderError
from ..models import HistoryRequest, ProviderResponse
from ..normalizer import normalize_yahoo_frame
from .base import MarketDataProvider


class YahooFinanceProvider(MarketDataProvider):
    source = "yahoo"

    def validate_request(self, request: HistoryRequest) -> None:
        if not request.ticker:
            raise ValueError("ticker cannot be empty")
        if request.period and (request.start_date or request.end_date):
            raise ValueError("period cannot be combined with explicit dates")

    def download_history(self, request: HistoryRequest) -> ProviderResponse:
        self.validate_request(request)
        try:
            import yfinance as yf

            frame = yf.download(
                tickers=request.ticker,
                start=request.start_date,
                end=request.end_date,
                period=request.period,
                interval=request.interval,
                threads=False,
                auto_adjust=False,
                actions=request.include_actions,
                progress=False,
                timeout=request.timeout,
                repair=request.repair_data,
            )
            return ProviderResponse(
                normalize_yahoo_frame(frame, request.ticker, request.interval, self.source),
                request.ticker,
            )
        except Exception as exc:
            raise ProviderError(f"Yahoo download failed for {request.ticker}: {exc}") from exc
