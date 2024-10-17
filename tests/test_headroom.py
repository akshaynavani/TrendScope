import pandas as pd
import pytest

from trendscope.config import SizingConfig
from trendscope.models.base import ForecastResult
from trendscope.sizing.headroom import (
    SegmentInputs,
    annualise,
    lookup_tam,
    score_opportunities,
    size_headroom,
)

SEG = {"region": "NA", "product_line": "SW", "channel": "Direct"}


def _market(tam_2024=1_000_000.0, tam_2025=1_100_000.0):
    return pd.DataFrame([
        {**SEG, "year": 2024, "tam": tam_2024},
        {**SEG, "year": 2025, "tam": tam_2025},
    ])


def _inputs(level: float, confidence: float = 0.9) -> SegmentInputs:
    hist_idx = pd.date_range("2023-01-02", periods=104, freq="W-MON")
    fc_idx = pd.date_range(hist_idx[-1] + pd.Timedelta(weeks=1), periods=52, freq="W-MON")
    fc = pd.Series(level, index=fc_idx)
    return SegmentInputs(
        segment=SEG,
        history=pd.Series(level * 0.9, index=hist_idx),
        forecast=ForecastResult("arima", fc, fc * 0.8, fc * 1.2),
        frequency="W",
        confidence=confidence,
    )


def test_annualise_scales_short_horizons():
    assert annualise(pd.Series([10.0] * 26), "W") == pytest.approx(520.0)
    assert annualise(pd.Series([10.0] * 60), "W") == pytest.approx(520.0)


def test_lookup_tam_falls_back_to_nearest_year():
    assert lookup_tam(_market(), SEG, 2030) == 1_100_000.0
    assert pd.isna(lookup_tam(_market(), {**SEG, "region": "APAC"}, 2024))


def test_headroom_is_capturable_minus_forecast():
    cfg = SizingConfig(penetration_ceiling=0.5)
    row = size_headroom(_inputs(level=5_000.0), _market(), cfg)
    # forecast annual = 5000 * 52 = 260k, TAM(2025) = 1.1M -> capturable 550k
    assert row["forecast_annual_revenue"] == pytest.approx(260_000)
    assert row["capturable_revenue"] == pytest.approx(550_000)
    assert row["headroom"] == pytest.approx(290_000)
    assert row["headroom_low"] < row["headroom"] < row["headroom_high"]
    assert row["forecast_growth"] == pytest.approx(1 / 0.9 - 1)


def test_headroom_never_negative():
    cfg = SizingConfig(penetration_ceiling=0.1)
    row = size_headroom(_inputs(level=50_000.0), _market(), cfg)
    assert row["headroom"] == 0


def test_score_ranks_and_tiers():
    cfg = SizingConfig()
    sized = pd.DataFrame([
        {"segment_id": "a", "headroom": 900_000, "forecast_growth": 0.2, "confidence": 0.9},
        {"segment_id": "b", "headroom": 100_000, "forecast_growth": 0.0, "confidence": 0.5},
        {"segment_id": "c", "headroom": 0, "forecast_growth": -0.1, "confidence": 0.8},
    ])
    out = score_opportunities(sized, cfg)
    assert list(out["segment_id"]) == ["a", "b", "c"]
    assert out.loc[0, "opportunity_rank"] == 1
    assert out.loc[0, "tier"] == "Invest"
    assert out.loc[2, "tier"] == "Defend"
    assert out["headroom_share"].sum() == pytest.approx(1.0)
