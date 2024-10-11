"""Point and interval accuracy metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-9


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def mae(actual, pred) -> float:
    return float(np.mean(np.abs(_arr(actual) - _arr(pred))))


def rmse(actual, pred) -> float:
    return float(np.sqrt(np.mean((_arr(actual) - _arr(pred)) ** 2)))


def mape(actual, pred) -> float:
    """Mean absolute percentage error (%), ignoring zero actuals."""
    a, p = _arr(actual), _arr(pred)
    mask = np.abs(a) > EPS
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def smape(actual, pred) -> float:
    """Symmetric MAPE (%), bounded in [0, 200]."""
    a, p = _arr(actual), _arr(pred)
    denom = np.abs(a) + np.abs(p)
    mask = denom > EPS
    if not mask.any():
        return 0.0
    return float(np.mean(2 * np.abs(a[mask] - p[mask]) / denom[mask]) * 100)


def mase(actual, pred, train, season: int = 1) -> float:
    """Mean absolute scaled error relative to an in-sample seasonal-naive forecast."""
    tr = _arr(train)
    if len(tr) <= season:
        season = 1
    scale = np.mean(np.abs(tr[season:] - tr[:-season])) if len(tr) > season else np.nan
    if not scale or not np.isfinite(scale):
        return float("nan")
    return mae(actual, pred) / scale


def coverage(actual, lower, upper) -> float:
    """Share of actuals that fall inside the prediction interval."""
    a = _arr(actual)
    return float(np.mean((a >= _arr(lower)) & (a <= _arr(upper))))


def bias(actual, pred) -> float:
    """Signed percentage bias: positive means over-forecasting."""
    a, p = _arr(actual), _arr(pred)
    total = a.sum()
    return float((p.sum() - total) / total * 100) if abs(total) > EPS else float("nan")


def score_all(actual: pd.Series, result, train: pd.Series | None = None, season: int = 1) -> dict:
    """Compute every metric for a ForecastResult aligned to ``actual``."""
    pred = result.yhat.reindex(actual.index)
    scores = {
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "mape": mape(actual, pred),
        "smape": smape(actual, pred),
        "bias": bias(actual, pred),
        "coverage": coverage(actual, result.lower.reindex(actual.index), result.upper.reindex(actual.index)),
    }
    if train is not None:
        scores["mase"] = mase(actual, pred, train, season)
    return scores
