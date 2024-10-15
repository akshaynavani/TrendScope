"""Individual Seaborn chart builders. Each draws onto a supplied Axes."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter, PercentFormatter

from trendscope.models.base import ForecastResult
from trendscope.viz.theme import (
    CATEGORICAL,
    INTERVAL_ALPHA,
    SEQUENTIAL_CMAP,
    SURFACE,
    TEXT_SECONDARY,
    TIER_COLORS,
    color_map,
    money,
)


def plot_segment_forecast(
    ax: plt.Axes,
    history: pd.Series,
    forecast: ForecastResult,
    title: str,
    lookback: int | None = 104,
) -> plt.Axes:
    """Actuals, forecast line and prediction interval for one segment."""
    hist = history.iloc[-lookback:] if lookback else history
    ax.plot(hist.index, hist.values, color=CATEGORICAL[0], label="Actual")
    # Connect the forecast to the last actual so there's no visual gap.
    fc_x = [hist.index[-1], *forecast.yhat.index]
    fc_y = [hist.values[-1], *forecast.yhat.values]
    ax.plot(fc_x, fc_y, color=CATEGORICAL[1], linestyle="--", label=f"Forecast ({forecast.model})")
    ax.fill_between(
        forecast.yhat.index, forecast.lower.values, forecast.upper.values,
        color=CATEGORICAL[1], alpha=INTERVAL_ALPHA, linewidth=0, label="Interval",
    )
    ax.axvline(hist.index[-1], color=TEXT_SECONDARY, linewidth=0.8, linestyle=":")
    ax.set_title(title, loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.legend(loc="upper left", fontsize=8)
    return ax


def plot_region_trends(ax: plt.Axes, daily: pd.DataFrame, freq: str = "MS") -> plt.Axes:
    """Total revenue over time, one line per region."""
    df = (
        daily.set_index("ds")
        .groupby("region")["revenue"]
        .resample(freq)
        .sum()
        .reset_index()
    )
    # Drop the trailing partial period.
    df = df[df["ds"] < df["ds"].max()]
    palette = color_map(df["region"].unique().tolist())
    sns.lineplot(data=df, x="ds", y="revenue", hue="region", palette=palette, ax=ax)
    ax.set_title("Revenue by region", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("Monthly revenue")
    ax.yaxis.set_major_formatter(FuncFormatter(money))
    ax.legend(title=None, fontsize=8, ncol=2)
    return ax


def plot_accuracy_heatmap(ax: plt.Axes, leaderboard: pd.DataFrame, metric: str = "smape") -> plt.Axes:
    """Segment x model backtest error heatmap (lower is better)."""
    pivot = leaderboard.pivot_table(index="segment_id", columns="model", values=metric)
    sns.heatmap(
        pivot, ax=ax, cmap=SEQUENTIAL_CMAP, annot=True, fmt=".1f", annot_kws={"fontsize": 7},
        linewidths=2, linecolor=SURFACE, cbar_kws={"label": f"{metric.upper()} (%)"},
    )
    ax.set_title(f"Backtest {metric.upper()} by segment and model", loc="left")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=7)
    return ax


def plot_headroom_bars(ax: plt.Axes, sized: pd.DataFrame, top_n: int = 12) -> plt.Axes:
    """Stacked bars: forecast revenue + remaining headroom up to the capturable ceiling."""
    df = sized.nlargest(top_n, "headroom").iloc[::-1]
    labels = df["segment_id"]
    ax.barh(labels, df["forecast_annual_revenue"], color=CATEGORICAL[0],
            edgecolor=SURFACE, linewidth=2, label="Forecast revenue (next 12m)")
    ax.barh(labels, df["headroom"], left=df["forecast_annual_revenue"], color=CATEGORICAL[2],
            edgecolor=SURFACE, linewidth=2, label="Headroom to ceiling")
    err_low = (df["headroom"] - df["headroom_low"]).clip(lower=0)
    err_high = (df["headroom_high"] - df["headroom"]).clip(lower=0)
    ax.errorbar(
        df["forecast_annual_revenue"] + df["headroom"], labels,
        xerr=[err_low, err_high], fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=3,
    )
    for y, (_, row) in enumerate(df.iterrows()):
        ax.text(row["forecast_annual_revenue"] + row["headroom"], y, f"  {money(row['headroom'])}",
                va="center", fontsize=7, color=TEXT_SECONDARY)
    ax.set_title(f"Top {len(df)} segments by headroom", loc="left")
    ax.set_xlabel("Annual revenue")
    ax.xaxis.set_major_formatter(FuncFormatter(money))
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_share_vs_growth(ax: plt.Axes, sized: pd.DataFrame) -> plt.Axes:
    """Opportunity map: market share vs forecast growth, bubble = headroom, colour = tier."""
    df = sized.copy()
    sizes = 40 + 400 * (df["headroom"].fillna(0) / max(df["headroom"].max(), 1))
    for tier, grp in df.groupby("tier"):
        ax.scatter(
            grp["current_share"], grp["forecast_growth"], s=sizes.loc[grp.index],
            color=TIER_COLORS.get(tier, CATEGORICAL[6]), edgecolor=SURFACE, linewidth=2,
            alpha=0.9, label=tier,
        )
    for _, row in df.nsmallest(3, "opportunity_rank").iterrows():
        ax.annotate(row["segment_id"], (row["current_share"], row["forecast_growth"]),
                    xytext=(6, 6), textcoords="offset points", fontsize=7, color=TEXT_SECONDARY)
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_title("Opportunity map", loc="left")
    ax.set_xlabel("Current market share")
    ax.set_ylabel("Forecast growth (next 12m)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(title="Tier", fontsize=8, markerscale=0.5)
    return ax


def plot_model_mix(ax: plt.Axes, sized: pd.DataFrame) -> plt.Axes:
    """How often each model (or ensemble) won segment selection."""
    counts = sized["model"].value_counts().sort_values()
    ax.barh(counts.index, counts.values, color=CATEGORICAL[0], edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(counts.values):
        ax.text(v, y, f" {v}", va="center", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title("Selected model per segment", loc="left")
    ax.set_xlabel("Segments")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.tick_params(axis="y", labelsize=7)
    return ax


def plot_residuals(ax: plt.Axes, predictions: pd.DataFrame) -> plt.Axes:
    """Distribution of backtest percentage errors per model."""
    df = predictions.copy()
    df = df[df["actual"] > 0]
    df["pct_error"] = (df["yhat"] - df["actual"]) / df["actual"]
    df["pct_error"] = df["pct_error"].clip(-1, 1)
    palette = color_map(df["model"].unique().tolist())
    sns.boxplot(data=df, x="pct_error", y="model", hue="model", palette=palette,
                ax=ax, fliersize=2, linewidth=1, legend=False)
    ax.axvline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax.set_title("Backtest error distribution", loc="left")
    ax.set_xlabel("Forecast error (% of actual)")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    return ax


__all__ = [
    "plot_segment_forecast",
    "plot_region_trends",
    "plot_accuracy_heatmap",
    "plot_headroom_bars",
    "plot_share_vs_growth",
    "plot_model_mix",
    "plot_residuals",
]
