"""Synthetic sales generator so the pipeline can be run end-to-end locally.

Each segment (region x product line x channel) gets its own level, trend, yearly and
weekly seasonality, noise and a handful of promotional spikes. A matching market-size
(TAM) table is produced with varying penetration so headroom differs across segments.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trendscope.config import DataConfig


@dataclass
class SegmentProfile:
    region: str
    product_line: str
    channel: str
    base_daily_revenue: float
    annual_growth: float
    yearly_amplitude: float
    weekly_amplitude: float
    noise: float
    avg_order_value: float
    penetration: float
    tam_growth: float


REGION_SCALE = {"NA": 1.6, "EMEA": 1.2, "APAC": 1.0, "LATAM": 0.5}
PRODUCT_AOV = {"Hardware": 1800.0, "Software": 650.0, "Services": 3200.0}


def build_profiles(cfg: DataConfig, rng: np.random.Generator) -> list[SegmentProfile]:
    profiles = []
    for region, product, channel in itertools.product(cfg.regions, cfg.product_lines, cfg.channels):
        scale = REGION_SCALE.get(region, 1.0) * (1.0 if channel == "Direct" else 0.65)
        profiles.append(
            SegmentProfile(
                region=region,
                product_line=product,
                channel=channel,
                base_daily_revenue=float(rng.uniform(8_000, 30_000) * scale),
                annual_growth=float(rng.normal(0.08, 0.07)),
                yearly_amplitude=float(rng.uniform(0.05, 0.3)),
                weekly_amplitude=float(rng.uniform(0.05, 0.25)),
                noise=float(rng.uniform(0.08, 0.2)),
                avg_order_value=PRODUCT_AOV.get(product, 1000.0) * float(rng.uniform(0.8, 1.2)),
                penetration=float(rng.uniform(0.06, 0.4)),
                tam_growth=float(rng.uniform(0.02, 0.1)),
            )
        )
    return profiles


def _segment_daily_revenue(p: SegmentProfile, dates: pd.DatetimeIndex, rng) -> np.ndarray:
    t_years = (dates - dates[0]).days.values / 365.25
    trend = (1 + p.annual_growth) ** t_years
    doy = dates.dayofyear.values
    # Peak in Q4 (fiscal year-end push), trough mid-year.
    yearly = 1 + p.yearly_amplitude * np.cos(2 * np.pi * (doy - 340) / 365.25)
    weekday = dates.dayofweek.values
    weekly = np.where(weekday >= 5, 1 - 2 * p.weekly_amplitude, 1 + 0.4 * p.weekly_amplitude)
    noise = rng.lognormal(mean=0.0, sigma=p.noise, size=len(dates))

    promo = np.ones(len(dates))
    n_promos = max(1, len(dates) // 120)
    for start in rng.choice(len(dates) - 7, size=n_promos, replace=False):
        promo[start : start + rng.integers(3, 8)] *= rng.uniform(1.3, 1.9)

    return p.base_daily_revenue * trend * yearly * weekly * noise * promo


def generate_sales(cfg: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(raw_sales, market_size)`` DataFrames."""
    rng = np.random.default_rng(cfg.seed)
    dates = pd.date_range(cfg.start_date, cfg.end_date, freq="D")
    profiles = build_profiles(cfg, rng)

    sales_frames = []
    market_rows = []
    for seg_idx, p in enumerate(profiles):
        daily = _segment_daily_revenue(p, dates, rng)
        # Split each day's revenue into a few orders to look like transactional data.
        n_orders = np.maximum(1, np.round(daily / p.avg_order_value).astype(int))
        n_orders = np.minimum(n_orders, 4)  # keep row counts manageable
        repeat_dates = np.repeat(dates.values, n_orders)
        repeat_rev = np.repeat(daily / n_orders, n_orders)
        repeat_rev *= rng.uniform(0.85, 1.15, size=len(repeat_rev))
        units = np.maximum(1, np.round(repeat_rev / p.avg_order_value * 3)).astype(int)
        frame = pd.DataFrame(
            {
                "order_id": [f"S{seg_idx:02d}-{i:07d}" for i in range(len(repeat_dates))],
                "order_date": pd.to_datetime(repeat_dates),
                "region": p.region,
                "product_line": p.product_line,
                "channel": p.channel,
                "units": units,
                "revenue": np.round(repeat_rev, 2),
            }
        )
        sales_frames.append(frame)

        yearly_rev = frame.groupby(frame["order_date"].dt.year)["revenue"].sum()
        first_year = int(yearly_rev.index.min())
        base_tam = yearly_rev.iloc[0] / p.penetration
        last_year = int(yearly_rev.index.max()) + 3  # include forward-looking estimates
        for year in range(first_year, last_year + 1):
            market_rows.append(
                {
                    "region": p.region,
                    "product_line": p.product_line,
                    "channel": p.channel,
                    "year": year,
                    "tam": round(base_tam * (1 + p.tam_growth) ** (year - first_year), 2),
                }
            )

    raw_sales = pd.concat(sales_frames, ignore_index=True)
    market_size = pd.DataFrame(market_rows)
    return raw_sales, market_size
