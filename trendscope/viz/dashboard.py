"""Compose chart builders into dashboard PNGs written to the output directory."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from trendscope.models.base import ForecastResult
from trendscope.viz import plots
from trendscope.viz.theme import apply_theme, save

log = logging.getLogger(__name__)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def overview_dashboard(daily: pd.DataFrame, sized: pd.DataFrame, predictions: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plots.plot_region_trends(axes[0, 0], daily)
    plots.plot_model_mix(axes[0, 1], sized)
    if not predictions.empty:
        plots.plot_residuals(axes[1, 0], predictions)
    else:
        axes[1, 0].set_axis_off()
    plots.plot_share_vs_growth(axes[1, 1], sized)
    fig.suptitle("TrendScope — sales forecast overview", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)
    return path


def accuracy_dashboard(leaderboard: pd.DataFrame, metric: str, path: Path) -> Path:
    n_segments = leaderboard["segment_id"].nunique()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * n_segments + 1.5)))
    plots.plot_accuracy_heatmap(ax, leaderboard, metric)
    fig.tight_layout()
    save(fig, path)
    return path


def headroom_dashboard(sized: pd.DataFrame, path: Path, top_n: int = 12) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1.3, 1]})
    plots.plot_headroom_bars(axes[0], sized, top_n=top_n)
    plots.plot_share_vs_growth(axes[1], sized)
    fig.suptitle("Opportunity sizing", x=0.01, ha="left", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)
    return path


def forecast_grid(
    series: dict[str, pd.Series],
    forecasts: dict[str, ForecastResult],
    segment_ids: list[str],
    path: Path,
    cols: int = 3,
) -> Path:
    ids = [s for s in segment_ids if s in forecasts]
    rows = max(1, math.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3.6 * rows), squeeze=False)
    for ax, seg in zip(axes.flat, ids):
        plots.plot_segment_forecast(ax, series[seg], forecasts[seg], seg)
    for ax in list(axes.flat)[len(ids):]:
        ax.set_axis_off()
    fig.tight_layout()
    save(fig, path)
    return path


def render_all(
    output_dir: Path,
    daily: pd.DataFrame,
    series: dict[str, pd.Series],
    forecasts: dict[str, ForecastResult],
    leaderboard: pd.DataFrame,
    predictions: pd.DataFrame,
    sized: pd.DataFrame,
    metric: str = "smape",
    per_segment: bool = True,
) -> list[Path]:
    """Render every dashboard; returns the list of written files."""
    apply_theme()
    out = Path(output_dir) / "dashboards"
    out.mkdir(parents=True, exist_ok=True)
    written = [
        overview_dashboard(daily, sized, predictions, out / "overview.png"),
        headroom_dashboard(sized, out / "headroom.png"),
    ]
    if not leaderboard.empty:
        written.append(accuracy_dashboard(leaderboard, metric, out / "accuracy.png"))
    top = sized.sort_values("rank")["segment_id"].head(9).tolist()
    written.append(forecast_grid(series, forecasts, top, out / "top_segments_forecast.png"))

    if per_segment:
        seg_dir = out / "segments"
        seg_dir.mkdir(exist_ok=True)
        for seg, fc in forecasts.items():
            fig, ax = plt.subplots(figsize=(10, 4))
            plots.plot_segment_forecast(ax, series[seg], fc, seg)
            path = seg_dir / f"{_slug(seg)}.png"
            save(fig, path)
            written.append(path)
    log.info("Wrote %d dashboard images to %s", len(written), out)
    return written
