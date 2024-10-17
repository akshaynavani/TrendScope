import numpy as np
import pandas as pd
import pytest

from trendscope.config import Config, DataConfig
from trendscope.db.sqlite_backend import SqliteBackend


@pytest.fixture
def weekly_series() -> pd.Series:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2021-01-04", periods=156, freq="W-MON")
    t = np.arange(len(idx))
    y = 1000 + 3 * t + 150 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 20, len(idx))
    return pd.Series(y, index=idx, name="revenue")


@pytest.fixture
def small_config(tmp_path) -> Config:
    cfg = Config()
    cfg.output_dir = str(tmp_path / "output")
    cfg.data = DataConfig(
        start_date="2022-01-01",
        end_date="2024-06-30",
        regions=["NA", "EMEA"],
        product_lines=["Software"],
        channels=["Direct"],
        seed=7,
    )
    cfg.forecast.models = ["arima", "seasonal_naive"]
    cfg.forecast.horizon = 13
    cfg.backtest.folds = 2
    return cfg


@pytest.fixture
def memory_backend():
    backend = SqliteBackend.from_path(":memory:")
    yield backend
    backend.close()
