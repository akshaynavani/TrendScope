import numpy as np
import pandas as pd
import pytest

from trendscope.evaluation import metrics
from trendscope.models.base import ForecastResult


def test_perfect_forecast_has_zero_error():
    a = [10, 20, 30]
    assert metrics.mae(a, a) == 0
    assert metrics.rmse(a, a) == 0
    assert metrics.mape(a, a) == 0
    assert metrics.smape(a, a) == 0


def test_mape_ignores_zero_actuals():
    assert metrics.mape([0, 100], [5, 110]) == pytest.approx(10.0)


def test_smape_is_bounded():
    assert metrics.smape([0, 0], [10, 10]) == pytest.approx(200.0)


def test_mase_scales_by_naive_error():
    train = [1, 2, 3, 4, 5]  # naive in-sample MAE = 1
    assert metrics.mase([6, 7], [7, 8], train) == pytest.approx(1.0)


def test_coverage_and_bias():
    assert metrics.coverage([1, 5, 10], [0, 0, 0], [2, 6, 8]) == pytest.approx(2 / 3)
    assert metrics.bias([100, 100], [110, 110]) == pytest.approx(10.0)


def test_score_all_aligns_on_index():
    idx = pd.date_range("2024-01-01", periods=3, freq="W-MON")
    actual = pd.Series([100.0, 100.0, 100.0], index=idx)
    pred = pd.Series([90.0, 100.0, 110.0], index=idx)
    res = ForecastResult("m", pred, pred - 5, pred + 5)
    scores = metrics.score_all(actual, res, train=pd.Series(np.arange(10.0)))
    assert scores["mae"] == pytest.approx(20 / 3)
    assert scores["coverage"] == pytest.approx(1 / 3)
    assert "mase" in scores
