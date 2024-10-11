"""Seasonal-naive baseline: repeat the last observed season."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from trendscope.features import SEASONAL_PERIOD, future_index
from trendscope.models.base import Forecaster, ForecastResult


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"

    def fit(self, series: pd.Series) -> "SeasonalNaive":
        self._train = series.astype(float)
        return self

    def predict(self, horizon: int) -> ForecastResult:
        train = self._check_fitted()
        m = SEASONAL_PERIOD.get(self.frequency, 1)
        if len(train) < m:
            m = 1
        idx = future_index(train, horizon)
        last_season = train.iloc[-m:].values
        reps = int(np.ceil(horizon / m))
        yhat = np.tile(last_season, reps)[:horizon]

        # Residual scale from in-sample seasonal differences; widen with sqrt(k seasons ahead).
        diffs = (train - train.shift(m)).dropna()
        sigma = float(diffs.std()) if len(diffs) > 1 else float(train.std() or 0.0)
        z = stats.norm.ppf(0.5 + self.interval_width / 2)
        k = np.floor(np.arange(horizon) / m) + 1
        half = z * sigma * np.sqrt(k)
        yhat_s = pd.Series(yhat, index=idx)
        return ForecastResult(
            model=self.name,
            yhat=yhat_s,
            lower=pd.Series(np.maximum(yhat - half, 0), index=idx),
            upper=pd.Series(yhat + half, index=idx),
            meta={"season_length": m},
        )
