from market_data import load_history, scan_history

print(load_history(tickers=["AAPL"], columns=["timestamp", "ticker", "adj_close"]))
print(scan_history(tickers=["AAPL"]).collect())
