"""Turn daily segment aggregates into clean, regularly spaced time series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trendscope.ingest import SEGMENT_KEYS

FREQ_ALIASES = {"D": "D", "W": "W-MON", "M": "MS"}
SEASONAL_PERIOD = {"D": 7, "W": 52, "M": 12}
PERIODS_PER_YEAR = {"D": 365, "W": 52, "M": 12}


@dataclass
class SegmentSeries:
    key: tuple[str, str, str]
    series: pd.Series  # indexed by period start, values = revenue

    @property
    def segment_id(self) -> str:
        return " | ".join(self.key)

    @property
    def as_dict(self) -> dict[str, str]:
        return dict(zip(SEGMENT_KEYS, self.key))


def resample_series(daily: pd.Series, freq: str) -> pd.Series:
    """Sum daily revenue into the target frequency, filling missing days with zero."""
    if freq not in FREQ_ALIASES:
        raise ValueError(f"Unsupported frequency {freq!r}")
    daily = daily.sort_index()
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index, fill_value=0.0)
    alias = FREQ_ALIASES[freq]
    out = daily.resample(alias, label="left", closed="left").sum()
    # Drop a trailing partial period so the last point isn't artificially low.
    if freq != "D" and len(out) > 1:
        last_start = out.index[-1]
        expected_end = last_start + pd.tseries.frequencies.to_offset(alias)
        if daily.index.max() < expected_end - pd.Timedelta(days=1):
            out = out.iloc[:-1]
    out.index.freq = pd.tseries.frequencies.to_offset(alias)
    return out


def cap_outliers(series: pd.Series, window: int = 9, z: float = 4.0) -> pd.Series:
    """Winsorise points far from a rolling median (robust to promo spikes)."""
    if len(series) < window:
        return series
    med = series.rolling(window, center=True, min_periods=1).median()
    mad = (series - med).abs().rolling(window, center=True, min_periods=1).median()
    scale = 1.4826 * mad.replace(0, np.nan)
    upper = med + z * scale
    lower = (med - z * scale).clip(lower=0)
    return series.clip(lower=lower.fillna(series), upper=upper.fillna(series))


def build_segment_series(
    daily: pd.DataFrame,
    freq: str = "W",
    value_col: str = "revenue",
    min_history: int = 0,
    cap: bool = True,
) -> list[SegmentSeries]:
    """Split the daily aggregate into one regular series per segment."""
    out: list[SegmentSeries] = []
    for key, grp in daily.groupby(SEGMENT_KEYS, sort=True):
        s = grp.set_index("ds")[value_col].astype(float)
        s = s.groupby(level=0).sum()
        s = resample_series(s, freq)
        if cap:
            s = cap_outliers(s)
        if len(s) < min_history:
            continue
        s.name = value_col
        out.append(SegmentSeries(key=tuple(str(k) for k in key), series=s))
    return out


def train_test_split(series: pd.Series, test_size: int) -> tuple[pd.Series, pd.Series]:
    if test_size <= 0 or test_size >= len(series):
        raise ValueError("test_size must be between 1 and len(series) - 1")
    return series.iloc[:-test_size], series.iloc[-test_size:]


def future_index(series: pd.Series, horizon: int) -> pd.DatetimeIndex:
    freq = series.index.freq or pd.infer_freq(series.index)
    offset = pd.tseries.frequencies.to_offset(freq)
    return pd.date_range(series.index[-1] + offset, periods=horizon, freq=offset)
