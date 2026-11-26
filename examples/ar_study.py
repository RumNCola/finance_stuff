"""Run AR(p), mean-adjusted AR(p), and geometric-lag models."""

from market_data import run_ar_study

companies = ["COLBUN.SN", "CHILE.SN"]

historical = run_ar_study(
    companies=companies,
    horizon=10,
    origin_date="2024-01-02",
    output_path="./data",
)
print(historical.summary())
historical_report = historical.write_report("ar_study_historical_report.md")
print(f"Full historical report written to {historical_report}")

# Omit origin_date for a forward-looking forecast from each ticker's latest
# stored observation. Actual evaluation metrics remain null until outcomes exist.
forward = run_ar_study(
    companies=companies,
    horizon=10,
    output_path="./data",
)
forward_report = forward.write_report("ar_study_forward_report.md")
print(f"Full forward report written to {forward_report}")
