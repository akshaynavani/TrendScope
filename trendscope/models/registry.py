"""Name -> forecaster factory lookup."""

from __future__ import annotations

import importlib.util
from typing import Callable

from trendscope.models.arima import ArimaForecaster
from trendscope.models.base import Forecaster
from trendscope.models.naive import SeasonalNaive
from trendscope.models.prophet_model import ProphetForecaster

MODEL_REGISTRY: dict[str, Callable[..., Forecaster]] = {
    "arima": ArimaForecaster,
    "prophet": ProphetForecaster,
    "seasonal_naive": SeasonalNaive,
}


def prophet_available() -> bool:
    return importlib.util.find_spec("prophet") is not None


def available_models(requested: list[str]) -> list[str]:
    """Filter ``requested`` down to models that are known and installed."""
    out = []
    for name in requested:
        if name not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model {name!r}; choose from {sorted(MODEL_REGISTRY)}")
        if name == "prophet" and not prophet_available():
            continue
        out.append(name)
    return out


def make_model(name: str, frequency: str, interval_width: float = 0.8, **kwargs) -> Forecaster:
    try:
        factory = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model {name!r}") from exc
    return factory(frequency=frequency, interval_width=interval_width, **kwargs)
