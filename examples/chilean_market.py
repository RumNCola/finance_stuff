from market_data import extract_history

result = extract_history(tickers=["SQM-B.SN", "CHILE.SN", "FALABELLA.SN"], period="10y")
print(result.summary())
