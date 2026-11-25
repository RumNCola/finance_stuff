from abc import ABC, abstractmethod

from ..models import HistoryRequest, ProviderResponse


class MarketDataProvider(ABC):
    @abstractmethod
    def validate_request(self, request: HistoryRequest) -> None:
        """Validate provider-independent and provider-specific request constraints."""

    @abstractmethod
    def download_history(self, request: HistoryRequest) -> ProviderResponse:
        """Download and normalize one ticker into the canonical Polars shape."""
