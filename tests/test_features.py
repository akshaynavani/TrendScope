import numpy as np
import pandas as pd

from trendscope.features import (
    build_segment_series,
    cap_outliers,
    future_index,
    resample_series,
    train_test_split,
)


def test_resample_weekly_drops_partial_periods():
    idx = pd.date_range("2024-01-03", "2024-02-07", freq="D")  # Wed .. Wed
    s = pd.Series(1.0, index=idx)
    weekly = resample_series(s, "W")
    assert (weekly == 7).all()
    assert weekly.index[0] == pd.Timestamp("2024-01-08")


def test_resample_fills_missing_days_with_zero():
    s = pd.Series([5.0, 5.0], index=pd.to_datetime(["2024-01-01", "2024-01-31"]))
    monthly = resample_series(s, "M")
    assert monthly.iloc[0] == 10.0


def test_cap_outliers_clips_spike():
    s = pd.Series(np.full(30, 100.0) + np.arange(30) % 3)
    s.iloc[15] = 10_000
    capped = cap_outliers(s)
    assert capped.iloc[15] < 1_000
    assert capped.iloc[0] == s.iloc[0]


def test_build_segment_series_groups_and_filters():
    days = pd.date_range("2024-01-01", periods=70, freq="D")
    rows = []
    for region in ["NA", "EMEA"]:
        for d in days:
            rows.append({"region": region, "product_line": "SW", "channel": "Direct",
                         "ds": d, "revenue": 10.0})
    daily = pd.DataFrame(rows)
    segs = build_segment_series(daily, "W", min_history=5)
    assert [s.segment_id for s in segs] == ["EMEA | SW | Direct", "NA | SW | Direct"]
    assert build_segment_series(daily, "W", min_history=50) == []


def test_split_and_future_index(weekly_series):
    train, test = train_test_split(weekly_series, 10)
    assert len(test) == 10 and len(train) == len(weekly_series) - 10
    fut = future_index(weekly_series, 4)
    assert fut[0] == weekly_series.index[-1] + pd.Timedelta(weeks=1)
    assert len(fut) == 4
