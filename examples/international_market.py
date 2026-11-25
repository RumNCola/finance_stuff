from market_data import extract_history

result = extract_history(
    tickers=["AAPL", "MSFT", "SPY"], start_date="2010-01-01", end_date="2026-01-01"
)
print(result.summary())
