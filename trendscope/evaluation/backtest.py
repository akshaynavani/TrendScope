"""Rolling-origin backtesting across models for a single series."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from trendscope.evaluation.metrics import score_all
from trendscope.features import SEASONAL_PERIOD
from trendscope.models.base import ModelUnavailable
from trendscope.models.registry import make_model

log = logging.getLogger(__name__)


@dataclass
class BacktestReport:
    scores: pd.DataFrame          # one row per (model, fold)
    predictions: pd.DataFrame     # out-of-sample predictions with actuals
    failures: dict[str, str] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Mean metrics per model across folds."""
        if self.scores.empty:
            return pd.DataFrame()
        return self.scores.drop(columns=["fold", "cutoff"]).groupby("model").mean(numeric_only=True)


def rolling_origins(n: int, folds: int, horizon: int, step: int, min_train: int) -> list[int]:
    """Return training cut-off positions, oldest first."""
    cutoffs = []
    for i in range(folds):
        cutoff = n - horizon - (folds - 1 - i) * step
        if cutoff >= min_train:
            cutoffs.append(cutoff)
    return cutoffs


def backtest_series(
    series: pd.Series,
    models: list[str],
    frequency: str,
    folds: int = 3,
    horizon: int = 13,
    step: int = 13,
    min_train: int = 52,
    interval_width: float = 0.8,
) -> BacktestReport:
    season = SEASONAL_PERIOD.get(frequency, 1)
    cutoffs = rolling_origins(len(series), folds, horizon, step, min_train)
    if not cutoffs:
        log.warning("Series too short for backtesting (n=%d)", len(series))

    score_rows, pred_frames, failures = [], [], {}
    for name in models:
        for fold, cutoff in enumerate(cutoffs):
            train = series.iloc[:cutoff]
            test = series.iloc[cutoff : cutoff + horizon]
            try:
                model = make_model(name, frequency, interval_width)
                result = model.fit_predict(train, len(test))
            except ModelUnavailable as exc:
                failures[name] = str(exc)
                break
            except Exception as exc:  # noqa: BLE001 - keep other models running
                log.warning("Backtest %s fold %d failed: %s", name, fold, exc)
                failures[name] = repr(exc)
                continue
            scores = score_all(test, result, train=train, season=season)
            score_rows.append({"model": name, "fold": fold, "cutoff": train.index[-1], **scores})
            frame = result.to_frame()
            frame["actual"] = test.values
            frame["fold"] = fold
            pred_frames.append(frame)

    return BacktestReport(
        scores=pd.DataFrame(score_rows),
        predictions=pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame(),
        failures=failures,
    )
