class MarketDataError(Exception):
    """Base exception for expected package errors."""


class ConfigurationError(MarketDataError, ValueError):
    """Raised when an extraction or query configuration is invalid."""


class ProviderError(MarketDataError):
    """Raised for provider failures that should be reported to the caller."""


class ValidationError(MarketDataError):
    """Raised when data contains a critical validation issue."""
