"""Run a live benchmark explicitly; this script never runs during tests."""

from market_data import extract_history

if __name__ == "__main__":
    for count in (10, 30, 100):
        tickers = ["AAPL"] * count
        result = extract_history(tickers=tickers, period="5y", output_path="benchmark-data")
        print(count, result.summary(), result.execution_time)
