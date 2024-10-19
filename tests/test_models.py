import numpy as np
import pandas as pd
import pytest

from trendscope.evaluation.backtest import backtest_series, rolling_origins
from trendscope.models.arima import ArimaForecaster, fourier_terms
from trendscope.models.base import ForecastResult
from trendscope.models.naive import SeasonalNaive
from trendscope.models.registry import available_models, make_model, prophet_available
from trendscope.selection import blend, choose_models


def _check_result(res: ForecastResult, horizon: int, train: pd.Series):
    assert len(res.yhat) == horizon
    assert res.yhat.index[0] > train.index[-1]
    assert (res.lower <= res.yhat + 1e-6).all()
    assert (res.upper >= res.yhat - 1e-6).all()
    assert (res.yhat >= 0).all()


def test_seasonal_naive_repeats_last_season(weekly_series):
    res = SeasonalNaive(frequency="W").fit_predict(weekly_series, 10)
    np.testing.assert_allclose(res.yhat.values, weekly_series.iloc[-52:-42].values)
    _check_result(res, 10, weekly_series)


def test_fourier_terms_shape():
    X = fourier_terms(0, 10, 52, 3)
    assert X.shape == (10, 6)


def test_arima_forecast_tracks_level(weekly_series):
    model = ArimaForecaster(frequency="W", max_p=1, max_q=1)
    res = model.fit_predict(weekly_series, 8)
    _check_result(res, 8, weekly_series)
    assert model.order is not None
    # Forecast should be in the same ballpark as recent history.
    assert abs(res.yhat.mean() / weekly_series.iloc[-26:].mean() - 1) < 0.5


@pytest.mark.skipif(not prophet_available(), reason="prophet not installed")
def test_prophet_forecast(weekly_series):
    res = make_model("prophet", "W").fit_predict(weekly_series, 8)
    _check_result(res, 8, weekly_series)


def test_registry_rejects_unknown():
    with pytest.raises(KeyError):
        available_models(["xgboost"])


def test_rolling_origins():
    assert rolling_origins(100, 3, 10, 10, 50) == [70, 80, 90]
    assert rolling_origins(60, 3, 10, 10, 45) == [50]


def test_backtest_and_selection(weekly_series):
    report = backtest_series(weekly_series, ["seasonal_naive", "arima"], "W",
                             folds=2, horizon=8, step=8, min_train=104)
    summary = report.summary()
    assert set(summary.index) == {"seasonal_naive", "arima"}
    assert len(report.predictions) == 2 * 2 * 8

    sel = choose_models(report, metric="smape", ensemble=False)
    assert len(sel.weights) == 1
    assert 0 <= sel.confidence <= 1

    sel_ens = choose_models(report, metric="smape", ensemble=True, tolerance=100)
    assert sum(sel_ens.weights.values()) == pytest.approx(1.0)


def test_blend_weights_point_forecasts():
    idx = pd.date_range("2024-01-01", periods=2, freq="W-MON")
    a = ForecastResult("a", pd.Series([10.0, 10.0], idx), pd.Series([8.0, 8.0], idx), pd.Series([12.0, 12.0], idx))
    b = ForecastResult("b", pd.Series([20.0, 20.0], idx), pd.Series([18.0, 18.0], idx), pd.Series([22.0, 22.0], idx))
    out = blend({"a": a, "b": b}, {"a": 3, "b": 1}, "ens")
    np.testing.assert_allclose(out.yhat.values, [12.5, 12.5])
    assert out.meta["weights"] == {"a": 0.75, "b": 0.25}
