"""Pick the best forecaster per segment (or blend the strongest ones)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from trendscope.evaluation.backtest import BacktestReport
from trendscope.models.base import ForecastResult
from trendscope.models.registry import make_model

log = logging.getLogger(__name__)

LOWER_IS_BETTER = {"mae", "rmse", "mape", "smape", "mase"}


@dataclass
class Selection:
    weights: dict[str, float]
    metric: str
    best_error: float
    leaderboard: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def label(self) -> str:
        if len(self.weights) == 1:
            return next(iter(self.weights))
        return "ensemble(" + "+".join(sorted(self.weights)) + ")"

    @property
    def confidence(self) -> float:
        """Crude 0-1 confidence derived from backtest percentage error."""
        if not np.isfinite(self.best_error):
            return 0.0
        return float(np.clip(1 - self.best_error / 100.0, 0.0, 1.0))


def choose_models(
    report: BacktestReport,
    metric: str = "smape",
    ensemble: bool = True,
    tolerance: float = 1.25,
    fallback: str = "seasonal_naive",
) -> Selection:
    """Choose a single model or an inverse-error-weighted blend.

    Models whose mean backtest error is within ``tolerance`` x the best error are blended.
    """
    if metric not in LOWER_IS_BETTER:
        raise ValueError(f"Selection metric must be one of {sorted(LOWER_IS_BETTER)}")
    summary = report.summary()
    if summary.empty or metric not in summary:
        return Selection(weights={fallback: 1.0}, metric=metric, best_error=float("nan"))

    board = summary[[metric, "rmse"]].dropna().sort_values([metric, "rmse"])
    if board.empty:
        return Selection(weights={fallback: 1.0}, metric=metric, best_error=float("nan"))
    best_name = board.index[0]
    best_err = float(board.iloc[0][metric])

    if not ensemble or len(board) == 1:
        return Selection({best_name: 1.0}, metric, best_err, board)

    close = board[board[metric] <= best_err * tolerance]
    inv = 1.0 / close[metric].clip(lower=1e-6)
    weights = (inv / inv.sum()).to_dict()
    return Selection({k: float(v) for k, v in weights.items()}, metric, best_err, board)


def blend(results: dict[str, ForecastResult], weights: dict[str, float], label: str) -> ForecastResult:
    """Weighted average of point forecasts and interval bounds."""
    usable = {k: w for k, w in weights.items() if k in results}
    total = sum(usable.values())
    if not usable or total <= 0:
        raise ValueError("No forecasts to blend")
    norm = {k: w / total for k, w in usable.items()}
    first = results[next(iter(norm))]
    yhat = sum(results[k].yhat * w for k, w in norm.items())
    lower = sum(results[k].lower * w for k, w in norm.items())
    upper = sum(results[k].upper * w for k, w in norm.items())
    return ForecastResult(
        model=label,
        yhat=yhat.reindex(first.yhat.index),
        lower=lower.reindex(first.yhat.index),
        upper=upper.reindex(first.yhat.index),
        meta={"weights": norm, "components": {k: results[k].meta for k in norm}},
    )


def final_forecast(
    series: pd.Series,
    selection: Selection,
    frequency: str,
    horizon: int,
    interval_width: float = 0.8,
) -> ForecastResult:
    """Refit the selected model(s) on the full history and produce the forward forecast."""
    results: dict[str, ForecastResult] = {}
    for name in selection.weights:
        try:
            results[name] = make_model(name, frequency, interval_width).fit_predict(series, horizon)
        except Exception as exc:  # noqa: BLE001
            log.warning("Final fit of %s failed: %s", name, exc)
    if not results:
        log.warning("All selected models failed; falling back to seasonal naive")
        return make_model("seasonal_naive", frequency, interval_width).fit_predict(series, horizon)
    if len(results) == 1:
        return next(iter(results.values()))
    return blend(results, selection.weights, selection.label)
