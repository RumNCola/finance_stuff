"""Educational autoregressive return forecasting and backtesting.

The module intentionally forecasts returns first and compounds them into prices.
It does not calculate investment signals or make claims about profitability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from .query import load_history


@dataclass(frozen=True)
class ARStudyConfig:
    """Configuration for an AR return study."""

    companies: list[str]
    horizon: int = 10
    origin_date: str | None = None
    output_path: str | Path = "./data"
    interval: str = "1d"
    price_column: str = "adj_close"
    p_values: tuple[int, ...] | None = None
    max_p: int = 10
    selection_criterion: Literal["aic", "bic"] = "bic"
    include_mean_adjusted: bool = True
    include_geometric: bool = True
    min_observations: int = 30

    def __post_init__(self) -> None:
        normalized = tuple(
            dict.fromkeys(company.strip().upper() for company in self.companies if company.strip())
        )
        if not normalized:
            raise ValueError("companies must contain at least one non-empty ticker")
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.max_p < 1:
            raise ValueError("max_p must be at least 1")
        candidates = self.p_values or tuple(range(1, self.max_p + 1))
        if not candidates or any(p < 1 for p in candidates):
            raise ValueError("p_values must contain positive integers")
        if self.selection_criterion not in {"aic", "bic"}:
            raise ValueError("selection_criterion must be 'aic' or 'bic'")
        if self.min_observations < max(candidates) + 5:
            raise ValueError("min_observations is too small for the requested p_values")
        object.__setattr__(self, "p_values", tuple(dict.fromkeys(candidates)))
        object.__setattr__(self, "companies", list(normalized))


@dataclass
class ARStudyResult:
    """Forecasts, model coefficients and evaluation metrics from one study."""

    config: ARStudyConfig
    origin_dates: dict[str, datetime]
    forecasts: pl.DataFrame
    comparison: pl.DataFrame
    coefficients: pl.DataFrame
    selected_models: pl.DataFrame
    errors: dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        evaluated = self.comparison.filter(pl.col("rmse").is_not_null()).height
        return (
            f"AR study for {len(self.config.companies)} ticker(s), horizon "
            f"{self.config.horizon}; {evaluated} model/ticker comparison(s) have realized values."
        )

    def to_markdown(self) -> str:
        """Return a complete, human-readable report without table truncation."""
        sections = [
            "# Autoregressive Market Data Study",
            "",
            f"**Summary:** {self.summary()}",
            f"**Companies:** {', '.join(self.config.companies)}  ",
            f"**Origin:** {self.config.origin_date or 'latest stored observation'}  ",
            f"**Horizon:** {self.config.horizon}  ",
            f"**Selection criterion:** {self.config.selection_criterion}",
            "",
            "## Forecasts",
            "",
            _dataframe_to_markdown(self.forecasts),
            "",
            "## Model comparison",
            "",
            _dataframe_to_markdown(self.comparison),
            "",
            "## Coefficients and lag weights",
            "",
            _dataframe_to_markdown(self.coefficients),
            "",
            "## Selected models",
            "",
            _dataframe_to_markdown(self.selected_models),
        ]
        if self.errors:
            sections.extend(["", "## Errors", ""])
            sections.extend(f"- **{ticker}:** {message}" for ticker, message in self.errors.items())
        return "\n".join(sections) + "\n"

    def write_report(self, path: str | Path = "ar_study_report.md") -> Path:
        """Write complete tables and compact comparison plots to Markdown."""
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        plot_paths = self.save_plots(report_path.parent / "ar_study_plots")
        report = self.to_markdown()
        if plot_paths:
            report += "\n## Model comparison graphs\n\n"
            report += "Each graph contains at most six models for readability.\n\n"
            report += "\n".join(
                f"### {path.stem}\n\n"
                f"![{path.stem}]({path.relative_to(report_path.parent).as_posix()})\n"
                for path in plot_paths
            )
        report_path.write_text(report, encoding="utf-8")
        return report_path

    def save_plots(
        self, output_dir: str | Path = "ar_study_plots", max_models_per_graph: int = 6
    ) -> list[Path]:
        """Save one compact forecast/metric comparison graph per ticker."""
        if max_models_per_graph < 1:
            raise ValueError("max_models_per_graph must be at least 1")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "Plotting requires matplotlib; install the project dependencies first."
            ) from exc

        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for ticker in self.config.companies:
            comparison = self.comparison.filter(pl.col("ticker") == ticker)
            has_metrics = comparison.filter(pl.col("rmse").is_not_null()).height > 0
            metric_name = "rmse" if has_metrics else self.config.selection_criterion
            if has_metrics:
                ranked = comparison.sort("rmse").head(max_models_per_graph)
            else:
                ranked = (
                    self.coefficients.filter(pl.col("ticker") == ticker)
                    .group_by("model")
                    .agg(pl.col(metric_name).first())
                    .sort(metric_name)
                    .head(max_models_per_graph)
                )
            models = ranked["model"].to_list()
            if not models:
                continue
            forecast = self.forecasts.filter(
                (pl.col("ticker") == ticker) & pl.col("model").is_in(models)
            ).sort(["model", "step"])
            figure, (price_axis, metric_axis) = plt.subplots(1, 2, figsize=(14, 5))
            for model in models:
                model_frame = forecast.filter(pl.col("model") == model)
                price_axis.plot(
                    model_frame["step"].to_list(),
                    model_frame["predicted_price"].to_list(),
                    marker="o",
                    linewidth=1.5,
                    label=model,
                )
            actual = forecast.filter(pl.col("actual_price").is_not_null())
            if not actual.is_empty():
                actual = actual.unique("step").sort("step")
                price_axis.plot(
                    actual["step"].to_list(),
                    actual["actual_price"].to_list(),
                    color="black",
                    linewidth=2.5,
                    marker="x",
                    label="actual",
                )
            price_axis.set_title(f"{ticker}: forecast prices")
            price_axis.set_xlabel("Forecast step")
            price_axis.set_ylabel("Price")
            price_axis.grid(alpha=0.25)
            price_axis.legend(fontsize="small")
            metric_values = ranked[metric_name].to_list()
            metric_axis.barh(models[::-1], metric_values[::-1])
            metric_axis.set_title(f"{ticker}: {'RMSE' if has_metrics else metric_name.upper()}")
            metric_axis.set_xlabel("Lower is better")
            metric_axis.grid(axis="x", alpha=0.25)
            figure.suptitle("Autoregressive model comparison")
            figure.tight_layout()
            safe_ticker = ticker.replace("/", "%2F").replace("\\", "%5C")
            path = directory / f"{safe_ticker}.png"
            figure.savefig(path, dpi=140, bbox_inches="tight")
            plt.close(figure)
            paths.append(path)
        return paths


def _dataframe_to_markdown(frame: pl.DataFrame) -> str:
    """Render all rows and columns; Polars display limits are not involved."""
    if frame.is_empty():
        return "_No rows._"
    columns = frame.columns
    rows = frame.to_dicts()

    def cell(value: object) -> str:
        if value is None:
            return ""
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def run_ar_study(
    companies: list[str],
    horizon: int = 10,
    origin_date: str | None = None,
    output_path: str | Path = "./data",
    interval: str = "1d",
    price_column: str = "adj_close",
    p_values: tuple[int, ...] | None = None,
    max_p: int = 10,
    selection_criterion: Literal["aic", "bic"] = "bic",
    include_mean_adjusted: bool = True,
    include_geometric: bool = True,
    min_observations: int = 30,
) -> ARStudyResult:
    """Fit AR return models and forecast each requested ticker.

    With ``origin_date`` set, observations strictly after that date are held out
    and used for evaluation. Without it, the origin is the latest stored
    observation and forecasts represent future, not-yet-observed prices.
    """
    config = ARStudyConfig(
        companies=companies,
        horizon=horizon,
        origin_date=origin_date,
        output_path=output_path,
        interval=interval,
        price_column=price_column,
        p_values=tuple(p_values) if p_values is not None else None,
        max_p=max_p,
        selection_criterion=selection_criterion,
        include_mean_adjusted=include_mean_adjusted,
        include_geometric=include_geometric,
        min_observations=min_observations,
    )
    history = load_history(
        output_path=config.output_path,
        tickers=config.companies,
        interval=config.interval,
        columns=["timestamp", "ticker", config.price_column],
    )
    forecasts: list[pl.DataFrame] = []
    comparisons: list[dict] = []
    coefficient_rows: list[dict] = []
    selected_rows: list[dict] = []
    origin_dates: dict[str, datetime] = {}
    errors: dict[str, str] = {}

    for ticker in config.companies:
        try:
            ticker_frame = history.filter(pl.col("ticker") == ticker).sort("timestamp")
            output = _fit_ticker(ticker_frame, ticker, config)
            forecasts.extend(output["forecasts"])
            comparisons.extend(output["comparisons"])
            coefficient_rows.extend(output["coefficients"])
            selected_rows.append(output["selected"])
            origin_dates[ticker] = output["origin"]
        except ValueError as exc:
            errors[ticker] = str(exc)

    return ARStudyResult(
        config=config,
        origin_dates=origin_dates,
        forecasts=_forecast_frame(forecasts),
        comparison=(
            pl.DataFrame(comparisons, schema=_empty_comparison().schema)
            if comparisons
            else _empty_comparison()
        ),
        coefficients=(
            pl.DataFrame(coefficient_rows, schema=_empty_coefficients().schema)
            if coefficient_rows
            else _empty_coefficients()
        ),
        selected_models=(
            pl.DataFrame([*selected_rows], schema=_empty_selected().schema)
            if selected_rows
            else _empty_selected()
        ),
        errors=errors,
    )


def _fit_ticker(frame: pl.DataFrame, ticker: str, config: ARStudyConfig) -> dict:
    if frame.is_empty():
        raise ValueError("no stored history was found")
    clean = frame.select(["timestamp", config.price_column]).drop_nulls().sort("timestamp")
    if clean.height < config.min_observations + 1:
        raise ValueError(
            f"needs at least {config.min_observations + 1} price observations, found {clean.height}"
        )
    timestamps = clean["timestamp"].to_list()
    prices = np.asarray(clean[config.price_column].to_list(), dtype=float)
    returns = prices[1:] / prices[:-1] - 1.0
    return_timestamps = timestamps[1:]
    cutoff = _parse_origin(config.origin_date) if config.origin_date else None
    if cutoff is None:
        train_end = len(returns)
    else:
        train_end = sum(timestamp <= cutoff for timestamp in return_timestamps)
    if train_end < config.min_observations:
        raise ValueError("origin_date leaves too few observations for model fitting")
    train_returns = returns[:train_end]
    last_price = prices[train_end]
    actual_after_origin = {
        timestamp: price
        for timestamp, price in zip(
            timestamps[train_end + 1 :], prices[train_end + 1 :], strict=True
        )
    }
    origin = timestamps[train_end]
    if origin.tzinfo is None:
        origin = origin.replace(tzinfo=timezone.utc)

    models = [("ar", False, p, "ordinary") for p in config.p_values]
    if config.include_mean_adjusted:
        models.extend(("mean_adjusted_ar", True, p, "ordinary") for p in config.p_values)
    if config.include_geometric:
        models.extend(("geometric_ar", False, p, "geometric") for p in config.p_values)
    fits = [
        _fit_candidate(train_returns, p, mean_adjusted, kind)
        for _, mean_adjusted, p, kind in models
    ]
    best_index = min(range(len(fits)), key=lambda index: fits[index][config.selection_criterion])
    best_name, best_mean_adjusted, best_p, best_kind = models[best_index]
    best_fit = fits[best_index]
    ordinary_indices = [index for index, model in enumerate(models) if model[3] == "ordinary"]
    best_ar_index = min(ordinary_indices, key=lambda index: fits[index][config.selection_criterion])
    best_ar_name, _, best_ar_p, _ = models[best_ar_index]
    coefficient_rows = []
    for (name, mean_adjusted, p, kind), fit in zip(models, fits, strict=True):
        coefficient_rows.append(
            {
                "ticker": ticker,
                "model": name,
                "p": p,
                "mean_adjusted": mean_adjusted,
                "kind": kind,
                "shared_beta": fit.get("shared_beta"),
                "mean_return": fit["mean"],
                "intercept": fit["intercept"],
                "lag": 0,
                "weight": None,
                "aic": fit["aic"],
                "bic": fit["bic"],
            }
        )
        for lag, weight in enumerate(fit["coefficients"], start=1):
            coefficient_rows.append(
                {
                    "ticker": ticker,
                    "model": name,
                    "p": p,
                    "mean_adjusted": mean_adjusted,
                    "kind": kind,
                    "shared_beta": fit.get("shared_beta"),
                    "mean_return": fit["mean"],
                    "intercept": fit["intercept"],
                    "lag": lag,
                    "weight": weight,
                    "aic": fit["aic"],
                    "bic": fit["bic"],
                }
            )
    forecast_dates = list(actual_after_origin)[: config.horizon]
    if len(forecast_dates) < config.horizon:
        forecast_dates.extend(
            _next_business_days(
                forecast_dates[-1] if forecast_dates else origin,
                config.horizon - len(forecast_dates),
            )
        )
    forecast_rows = []
    comparisons = []
    model_forecasts: dict[str, list[float]] = {}
    for name, mean_adjusted, p, kind in models:
        model_forecasts[f"{name}_{p}"] = _recursive_forecast(
            train_returns,
            _fit_candidate(train_returns, p, mean_adjusted, kind),
            config.horizon,
        )
    for model_name, model_returns in model_forecasts.items():
        price = last_price
        for step, (timestamp, predicted_return) in enumerate(
            zip(forecast_dates, model_returns, strict=True), start=1
        ):
            price *= 1.0 + predicted_return
            actual = actual_after_origin.get(timestamp)
            forecast_rows.append(
                {
                    "ticker": ticker,
                    "model": model_name,
                    "step": step,
                    "timestamp": timestamp,
                    "predicted_return": predicted_return,
                    "predicted_price": price,
                    "actual_price": actual,
                }
            )
        predictions = [
            row["predicted_price"] for row in forecast_rows if row["model"] == model_name
        ]
        actuals = [actual_after_origin.get(timestamp) for timestamp in forecast_dates]
        comparisons.append(
            _comparison_row(ticker, model_name, predictions, actuals, config.horizon, last_price)
        )
    selected = {
        "ticker": ticker,
        "selected_model": f"{best_name}_{best_p}",
        "selected_p": best_p,
        "selected_kind": best_kind,
        "best_ar_model": f"{best_ar_name}_{best_ar_p}",
        "best_ar_p": best_ar_p,
        "best_ar_criterion_value": fits[best_ar_index][config.selection_criterion],
        "mean_adjusted": best_mean_adjusted,
        "criterion": config.selection_criterion,
        "criterion_value": best_fit[config.selection_criterion],
        "origin": origin,
        "last_price": last_price,
    }
    return {
        "forecasts": [pl.DataFrame(forecast_rows)] if forecast_rows else [],
        "comparisons": comparisons,
        "coefficients": coefficient_rows,
        "selected": selected,
        "origin": origin,
    }


def _fit_candidate(values: np.ndarray, p: int, mean_adjusted: bool, kind: str) -> dict:
    if kind == "geometric":
        return _fit_geometric_model(values, p)
    return _fit_model(values, p, mean_adjusted)


def _fit_model(values: np.ndarray, p: int, mean_adjusted: bool) -> dict:
    if len(values) <= p + 2:
        raise ValueError(f"not enough observations for AR({p})")
    mean = float(values.mean())
    target = values[p:]
    lag_matrix = np.column_stack([values[p - lag - 1 : -lag - 1] for lag in range(p)])
    if mean_adjusted:
        centered = values - mean
        target = centered[p:]
        lag_matrix = np.column_stack([centered[p - lag - 1 : -lag - 1] for lag in range(p)])
        coefficients = np.linalg.lstsq(lag_matrix, target, rcond=None)[0]
        intercept = mean
        residuals = target - lag_matrix @ coefficients
    else:
        design = np.column_stack([np.ones(len(target)), lag_matrix])
        parameters = np.linalg.lstsq(design, target, rcond=None)[0]
        intercept = float(parameters[0])
        coefficients = parameters[1:]
        residuals = target - design @ parameters
    rss = max(float(residuals @ residuals), np.finfo(float).eps)
    n = len(residuals)
    parameter_count = p if mean_adjusted else p + 1
    return {
        "mean": mean,
        "mean_adjusted": mean_adjusted,
        "kind": "ordinary",
        "intercept": intercept,
        "coefficients": [float(value) for value in coefficients],
        "aic": n * np.log(rss / n) + 2 * parameter_count,
        "bic": n * np.log(rss / n) + parameter_count * np.log(n),
    }


def _fit_geometric_model(values: np.ndarray, p: int) -> dict:
    """Fit alpha + beta*r[t-1] + beta**2*r[t-2] + ... + beta**p*r[t-p]."""
    if len(values) <= p + 2:
        raise ValueError(f"not enough observations for geometric AR({p})")
    target = values[p:]
    lag_matrix = np.column_stack([values[p - lag - 1 : -lag - 1] for lag in range(p)])

    def evaluate(beta: float) -> tuple[float, float]:
        weights = beta ** np.arange(1, p + 1)
        weighted_lags = lag_matrix @ weights
        alpha = float(np.mean(target - weighted_lags))
        residuals = target - alpha - weighted_lags
        return float(residuals @ residuals), alpha

    grid = np.linspace(-0.99, 0.99, 397)
    losses = [evaluate(float(beta))[0] for beta in grid]
    best_index = int(np.argmin(losses))
    left = grid[max(0, best_index - 1)]
    right = grid[min(len(grid) - 1, best_index + 1)]
    refined = np.linspace(left, right, 101)
    beta = float(refined[int(np.argmin([evaluate(float(value))[0] for value in refined]))])
    rss, alpha = evaluate(beta)
    n = len(target)
    parameter_count = 2
    return {
        "mean": float(values.mean()),
        "mean_adjusted": False,
        "kind": "geometric",
        "intercept": alpha,
        "shared_beta": beta,
        "coefficients": [float(beta**lag) for lag in range(1, p + 1)],
        "aic": n * np.log(max(rss, np.finfo(float).eps) / n) + 2 * parameter_count,
        "bic": n * np.log(max(rss, np.finfo(float).eps) / n) + parameter_count * np.log(n),
    }


def _recursive_forecast(values: np.ndarray, fit: dict, horizon: int) -> list[float]:
    history = list(values.astype(float))
    mean = fit["mean"]
    coefficients = fit["coefficients"]
    forecasts = []
    for _ in range(horizon):
        lags = history[-len(coefficients) :][::-1]
        if fit["mean_adjusted"] and len(coefficients) > 0:
            predicted = mean + sum(
                weight * (lag - mean) for weight, lag in zip(coefficients, lags, strict=True)
            )
        else:
            predicted = fit["intercept"] + sum(
                weight * lag for weight, lag in zip(coefficients, lags, strict=True)
            )
        forecasts.append(float(predicted))
        history.append(float(predicted))
    return forecasts


def _comparison_row(
    ticker: str,
    model: str,
    predictions: list[float],
    actuals: list[float | None],
    horizon: int,
    base_price: float,
) -> dict:
    pairs = [
        (prediction, actual)
        for prediction, actual in zip(predictions, actuals, strict=True)
        if actual is not None
    ]
    if not pairs:
        return {
            "ticker": ticker,
            "model": model,
            "horizon": horizon,
            "evaluated_steps": 0,
            "mae": None,
            "rmse": None,
            "mape": None,
            "directional_accuracy": None,
        }
    prediction_array = np.asarray([pair[0] for pair in pairs])
    actual_array = np.asarray([pair[1] for pair in pairs])
    errors = prediction_array - actual_array
    non_zero = actual_array != 0
    mape = (
        float(np.mean(np.abs(errors[non_zero] / actual_array[non_zero])) * 100)
        if non_zero.any()
        else None
    )
    previous = base_price
    directions = []
    for prediction, actual in pairs:
        directions.append(np.sign(prediction - previous) == np.sign(actual - previous))
        previous = actual
    return {
        "ticker": ticker,
        "model": model,
        "horizon": horizon,
        "evaluated_steps": len(pairs),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mape": mape,
        "directional_accuracy": float(np.mean(directions)),
    }


def _parse_origin(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc)


def _next_business_days(start: datetime, count: int) -> list[datetime]:
    result = []
    candidate = start
    while len(result) < count:
        candidate += timedelta(days=1)
        if candidate.weekday() < 5:
            result.append(candidate)
    return result


def _forecast_frame(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "model": pl.String,
                "step": pl.Int64,
                "timestamp": pl.Datetime("us", "UTC"),
                "predicted_return": pl.Float64,
                "predicted_price": pl.Float64,
                "actual_price": pl.Float64,
            }
        )
    return pl.concat(frames, how="vertical_relaxed").sort(["ticker", "model", "step"])


def _empty_comparison() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "model": pl.String,
            "horizon": pl.Int64,
            "evaluated_steps": pl.Int64,
            "mae": pl.Float64,
            "rmse": pl.Float64,
            "mape": pl.Float64,
            "directional_accuracy": pl.Float64,
        }
    )


def _empty_coefficients() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "model": pl.String,
            "p": pl.Int64,
            "mean_adjusted": pl.Boolean,
            "kind": pl.String,
            "shared_beta": pl.Float64,
            "mean_return": pl.Float64,
            "intercept": pl.Float64,
            "lag": pl.Int64,
            "weight": pl.Float64,
            "aic": pl.Float64,
            "bic": pl.Float64,
        }
    )


def _empty_selected() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "selected_model": pl.String,
            "selected_p": pl.Int64,
            "selected_kind": pl.String,
            "best_ar_model": pl.String,
            "best_ar_p": pl.Int64,
            "best_ar_criterion_value": pl.Float64,
            "mean_adjusted": pl.Boolean,
            "criterion": pl.String,
            "criterion_value": pl.Float64,
            "origin": pl.Datetime("us", "UTC"),
            "last_price": pl.Float64,
        }
    )
