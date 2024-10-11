"""ARIMA forecaster built on statsmodels' SARIMAX.

Order is chosen by a small AIC grid search. For long seasonal periods (weekly data,
m=52) a seasonal ARIMA is slow and unstable, so yearly seasonality is modelled with
Fourier terms as exogenous regressors instead (dynamic harmonic regression).
"""

from __future__ import annotations

import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from trendscope.features import PERIODS_PER_YEAR, SEASONAL_PERIOD, future_index
from trendscope.models.base import Forecaster, ForecastResult

log = logging.getLogger(__name__)

MAX_NATIVE_SEASONAL_PERIOD = 24


def fourier_terms(n_start: int, n: int, period: float, k: int) -> np.ndarray:
    t = np.arange(n_start, n_start + n)
    cols = []
    for i in range(1, k + 1):
        cols.append(np.sin(2 * np.pi * i * t / period))
        cols.append(np.cos(2 * np.pi * i * t / period))
    return np.column_stack(cols) if cols else np.empty((n, 0))


class ArimaForecaster(Forecaster):
    name = "arima"

    def __init__(
        self,
        frequency: str = "W",
        interval_width: float = 0.8,
        max_p: int = 2,
        max_q: int = 2,
        d_values: tuple[int, ...] = (0, 1),
        fourier_k: int = 3,
        log_transform: bool = True,
    ):
        super().__init__(frequency, interval_width)
        self.max_p = max_p
        self.max_q = max_q
        self.d_values = d_values
        self.fourier_k = fourier_k
        self.log_transform = log_transform
        self.result = None
        self.order: tuple[int, int, int] | None = None
        self.seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._use_fourier = False
        self._period = SEASONAL_PERIOD.get(frequency, 1)

    # -- helpers -----------------------------------------------------------------
    def _transform(self, y: np.ndarray) -> np.ndarray:
        return np.log1p(np.clip(y, 0, None)) if self.log_transform else y

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        return np.expm1(y) if self.log_transform else y

    def _exog(self, start: int, n: int) -> np.ndarray | None:
        if not self._use_fourier:
            return None
        return fourier_terms(start, n, PERIODS_PER_YEAR[self.frequency], self.fourier_k)

    # -- API ---------------------------------------------------------------------
    def fit(self, series: pd.Series) -> "ArimaForecaster":
        self._train = series.astype(float)
        y = self._transform(self._train.values)
        n = len(y)
        m = self._period

        if m > MAX_NATIVE_SEASONAL_PERIOD:
            self._use_fourier = n >= m  # need at least one full year to estimate harmonics
            seasonal_candidates = [(0, 0, 0, 0)]
        elif m > 1 and n >= 2 * m:
            seasonal_candidates = [(0, 0, 0, 0), (1, 0, 0, m), (0, 1, 1, m)]
        else:
            seasonal_candidates = [(0, 0, 0, 0)]
        exog = self._exog(0, n)

        best = (np.inf, None, None, None)
        grid = itertools.product(range(self.max_p + 1), self.d_values, range(self.max_q + 1))
        for (p, d, q), sorder in itertools.product(list(grid), seasonal_candidates):
            if p == 0 and q == 0 and d == 0 and exog is None:
                continue
            trend = "c" if d == 0 else "n"
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(
                        y,
                        exog=exog,
                        order=(p, d, q),
                        seasonal_order=sorder,
                        trend=trend,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, maxiter=200)
            except (ValueError, np.linalg.LinAlgError) as exc:
                log.debug("SARIMAX%s%s failed: %s", (p, d, q), sorder, exc)
                continue
            if np.isfinite(res.aic) and res.aic < best[0]:
                best = (res.aic, res, (p, d, q), sorder)

        if best[1] is None:
            raise RuntimeError("No ARIMA configuration could be fit")
        _, self.result, self.order, self.seasonal_order = best
        log.debug("Selected ARIMA%s%s aic=%.2f", self.order, self.seasonal_order, best[0])
        return self

    def predict(self, horizon: int) -> ForecastResult:
        train = self._check_fitted()
        if self.result is None:
            raise RuntimeError("fit() did not produce a model")
        exog = self._exog(len(train), horizon)
        fc = self.result.get_forecast(steps=horizon, exog=exog)
        alpha = 1 - self.interval_width
        conf = np.asarray(fc.conf_int(alpha=alpha))
        idx = future_index(train, horizon)
        mean = self._inverse(np.asarray(fc.predicted_mean))
        lower = self._inverse(conf[:, 0])
        upper = self._inverse(conf[:, 1])
        return ForecastResult(
            model=self.name,
            yhat=pd.Series(np.clip(mean, 0, None), index=idx),
            lower=pd.Series(np.clip(lower, 0, None), index=idx),
            upper=pd.Series(np.clip(upper, 0, None), index=idx),
            meta={
                "order": self.order,
                "seasonal_order": self.seasonal_order,
                "fourier_k": self.fourier_k if self._use_fourier else 0,
                "aic": float(self.result.aic),
            },
        )
