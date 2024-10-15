"""Business headroom sizing: how much addressable revenue is left per segment.

For each segment::

    capturable      = TAM(year) * penetration_ceiling
    forecast_annual = next-12-months forecast revenue (annualised if horizon < 1 year)
    headroom        = max(0, capturable - forecast_annual)

Headroom is reported with a low/high range from the forecast interval, then combined with
forecast growth and model confidence into a 0-100 opportunity score used for ranking.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from trendscope.config import SizingConfig
from trendscope.features import PERIODS_PER_YEAR
from trendscope.models.base import ForecastResult


@dataclass
class SegmentInputs:
    segment: dict[str, str]          # region / product_line / channel
    history: pd.Series
    forecast: ForecastResult
    frequency: str
    confidence: float = 0.5


def annualise(values: pd.Series, frequency: str) -> float:
    """Sum the first year of ``values``; scale up if fewer periods are available."""
    ppy = PERIODS_PER_YEAR[frequency]
    if len(values) == 0:
        return 0.0
    if len(values) >= ppy:
        return float(values.iloc[:ppy].sum())
    return float(values.mean() * ppy)


def lookup_tam(market: pd.DataFrame, segment: dict[str, str], year: int) -> float:
    """TAM for ``segment`` in ``year``; nearest available year if that one is missing."""
    mask = np.ones(len(market), dtype=bool)
    for key, value in segment.items():
        mask &= market[key].astype(str).values == str(value)
    rows = market.loc[mask]
    if rows.empty:
        return float("nan")
    exact = rows[rows["year"] == year]
    if not exact.empty:
        return float(exact["tam"].iloc[0])
    nearest = rows.iloc[(rows["year"] - year).abs().argsort()].iloc[0]
    return float(nearest["tam"])


def size_headroom(inputs: SegmentInputs, market: pd.DataFrame, cfg: SizingConfig) -> dict:
    fc = inputs.forecast
    ppy = PERIODS_PER_YEAR[inputs.frequency]
    history = inputs.history
    ttm = float(history.iloc[-ppy:].sum()) if len(history) >= ppy else annualise(history, inputs.frequency)

    fc_annual = annualise(fc.yhat, inputs.frequency)
    fc_low = annualise(fc.lower, inputs.frequency)
    fc_high = annualise(fc.upper, inputs.frequency)

    current_year = int(history.index[-1].year)
    forecast_year = int(fc.yhat.index[min(len(fc.yhat), ppy) // 2].year) if len(fc.yhat) else current_year
    tam_now = lookup_tam(market, inputs.segment, current_year)
    tam_fc = lookup_tam(market, inputs.segment, forecast_year)
    capturable = tam_fc * cfg.penetration_ceiling if np.isfinite(tam_fc) else float("nan")

    def _headroom(rev: float) -> float:
        return max(0.0, capturable - rev) if np.isfinite(capturable) else float("nan")

    return {
        **inputs.segment,
        "model": fc.model,
        "confidence": inputs.confidence,
        "ttm_revenue": ttm,
        "forecast_annual_revenue": fc_annual,
        "forecast_growth": (fc_annual / ttm - 1) if ttm > 0 else float("nan"),
        "forecast_year": forecast_year,
        "tam_current": tam_now,
        "tam_forecast": tam_fc,
        "current_share": ttm / tam_now if tam_now and np.isfinite(tam_now) else float("nan"),
        "forecast_share": fc_annual / tam_fc if tam_fc and np.isfinite(tam_fc) else float("nan"),
        "capturable_revenue": capturable,
        "headroom": _headroom(fc_annual),
        # A low forecast leaves more headroom, so the bounds swap.
        "headroom_low": _headroom(fc_high),
        "headroom_high": _headroom(fc_low),
    }


def _minmax(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return pd.Series(0.5, index=s.index)
    return ((s - lo) / (hi - lo)).fillna(0.0)


def score_opportunities(sized: pd.DataFrame, cfg: SizingConfig) -> pd.DataFrame:
    """Add a 0-100 ``opportunity_score``, ``opportunity_rank`` and an action ``tier``."""
    if sized.empty:
        return sized
    out = sized.copy()
    w_total = cfg.headroom_weight + cfg.growth_weight + cfg.confidence_weight
    if w_total <= 0:
        raise ValueError("Sizing weights must sum to a positive number")
    growth = out["forecast_growth"].clip(-0.5, 1.0)
    score = (
        cfg.headroom_weight * _minmax(out["headroom"].fillna(0))
        + cfg.growth_weight * _minmax(growth.fillna(growth.median()))
        + cfg.confidence_weight * out["confidence"].clip(0, 1).fillna(0)
    ) / w_total
    out["opportunity_score"] = (score * 100).round(1)
    out["opportunity_rank"] = out["opportunity_score"].rank(ascending=False, method="min").astype(int)
    out["headroom_share"] = out["headroom"] / out["headroom"].sum() if out["headroom"].sum() > 0 else 0.0

    def _tier(row) -> str:
        if row["headroom"] <= 0:
            return "Defend"           # at or above the realistic penetration ceiling
        if row["opportunity_score"] >= 66:
            return "Invest"
        if row["opportunity_score"] >= 33:
            return "Grow"
        return "Monitor"

    out["tier"] = out.apply(_tier, axis=1)
    return out.sort_values("opportunity_rank").reset_index(drop=True)
