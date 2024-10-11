"""Prophet forecaster. Prophet is an optional dependency (``pip install trendscope[prophet]``)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from trendscope.features import FREQ_ALIASES, future_index
from trendscope.models.base import Forecaster, ForecastResult, ModelUnavailable


def _import_prophet():
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ModelUnavailable("prophet is not installed; pip install 'trendscope[prophet]'") from exc
    # cmdstanpy is chatty at INFO level.
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    return Prophet


class ProphetForecaster(Forecaster):
    name = "prophet"

    def __init__(
        self,
        frequency: str = "W",
        interval_width: float = 0.8,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        country_holidays: str | None = None,
    ):
        super().__init__(frequency, interval_width)
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.country_holidays = country_holidays
        self.model = None

    def fit(self, series: pd.Series) -> "ProphetForecaster":
        Prophet = _import_prophet()
        self._train = series.astype(float)
        df = pd.DataFrame({"ds": self._train.index, "y": self._train.values})
        self.model = Prophet(
            interval_width=self.interval_width,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            yearly_seasonality=len(df) >= _periods_per_year(self.frequency),
            weekly_seasonality=self.frequency == "D",
            daily_seasonality=False,
        )
        if self.frequency == "M":
            # Monthly data cannot support the default Fourier order of 10.
            self.model.yearly_seasonality = False
            self.model.add_seasonality(name="yearly", period=365.25, fourier_order=4)
        if self.country_holidays and self.frequency == "D":
            self.model.add_country_holidays(country_name=self.country_holidays)
        self.model.fit(df)
        return self

    def predict(self, horizon: int) -> ForecastResult:
        train = self._check_fitted()
        future = self.model.make_future_dataframe(
            periods=horizon, freq=FREQ_ALIASES[self.frequency], include_history=False
        )
        fc = self.model.predict(future)
        idx = future_index(train, horizon)
        return ForecastResult(
            model=self.name,
            yhat=pd.Series(np.clip(fc["yhat"].values, 0, None), index=idx),
            lower=pd.Series(np.clip(fc["yhat_lower"].values, 0, None), index=idx),
            upper=pd.Series(np.clip(fc["yhat_upper"].values, 0, None), index=idx),
            meta={"seasonality_mode": self.seasonality_mode,
                  "changepoints": int(len(self.model.changepoints))},
        )


def _periods_per_year(freq: str) -> int:
    return {"D": 365, "W": 52, "M": 12}[freq]
