"""Common forecaster interface so backtesting and selection stay model-agnostic."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ForecastResult:
    model: str
    yhat: pd.Series
    lower: pd.Series
    upper: pd.Series
    meta: dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"ds": self.yhat.index, "yhat": self.yhat.values, "yhat_lower": self.lower.values,
             "yhat_upper": self.upper.values, "model": self.model}
        )


class Forecaster(ABC):
    """Fit on a regularly spaced ``pd.Series`` and forecast ``horizon`` steps ahead."""

    name: str = "base"

    def __init__(self, frequency: str = "W", interval_width: float = 0.8):
        self.frequency = frequency
        self.interval_width = interval_width
        self._train: pd.Series | None = None

    @abstractmethod
    def fit(self, series: pd.Series) -> "Forecaster":
        ...

    @abstractmethod
    def predict(self, horizon: int) -> ForecastResult:
        ...

    def fit_predict(self, series: pd.Series, horizon: int) -> ForecastResult:
        return self.fit(series).predict(horizon)

    def _check_fitted(self) -> pd.Series:
        if self._train is None:
            raise RuntimeError(f"{self.name} must be fit before predict")
        return self._train


class ModelUnavailable(RuntimeError):
    """Raised when an optional dependency (e.g. prophet) is not installed."""
