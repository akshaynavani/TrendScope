"""Shared Seaborn/matplotlib styling so every dashboard reads as one system."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: dashboards are written to files (also on Databricks)

import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Validated categorical order (colour-blind-safe on adjacent pairs). Assigned in fixed
# order and never cycled; past eight series, fold into "Other" or facet.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL_CMAP = "Blues"           # magnitude: one hue, light -> dark
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
INTERVAL_ALPHA = 0.18

# Tier colours are reserved for status and always shipped with a text label.
TIER_COLORS = {"Invest": "#1baf7a", "Grow": "#2a78d6", "Monitor": "#9a9893", "Defend": "#eda100"}


def apply_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        rc={
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT_PRIMARY,
            "axes.titleweight": "semibold",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
            "font.family": "sans-serif",
        },
    )
    sns.set_palette(CATEGORICAL)


def color_map(names: list[str]) -> dict[str, str]:
    """Stable entity -> colour mapping (by sorted name, so filtering never repaints)."""
    ordered = sorted(dict.fromkeys(names))
    if len(ordered) > len(CATEGORICAL):
        raise ValueError("More than eight categories: fold extras into 'Other' or facet")
    return {name: CATEGORICAL[i] for i, name in enumerate(ordered)}


def money(x: float, _pos=None) -> str:
    """Axis formatter: 1.2M / 340K."""
    ax = abs(x)
    if ax >= 1e9:
        return f"{x / 1e9:.1f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.1f}M"
    if ax >= 1e3:
        return f"{x / 1e3:.0f}K"
    return f"{x:.0f}"


def save(fig: plt.Figure, path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
