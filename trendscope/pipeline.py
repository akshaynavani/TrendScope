"""End-to-end orchestration: ingest -> forecast -> backtest/select -> size -> persist -> plot."""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trendscope.config import Config
from trendscope.db.backend import SqlBackend, get_backend
from trendscope.evaluation.backtest import backtest_series
from trendscope.features import SegmentSeries, build_segment_series
from trendscope.ingest import (
    SEGMENT_KEYS,
    fetch_market_size,
    fetch_segment_daily,
    fetch_segment_summary,
    seed_synthetic,
)
from trendscope.models.base import ForecastResult
from trendscope.models.registry import available_models
from trendscope.persist import export_csv, persist_results
from trendscope.selection import choose_models, final_forecast
from trendscope.sizing.headroom import SegmentInputs, score_opportunities, size_headroom

log = logging.getLogger(__name__)


@dataclass
class SegmentOutcome:
    segment: SegmentSeries
    forecast: ForecastResult
    selection_label: str
    confidence: float
    scores: pd.DataFrame
    predictions: pd.DataFrame
    failures: dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineResult:
    run_id: str
    daily: pd.DataFrame
    outcomes: list[SegmentOutcome]
    headroom: pd.DataFrame
    forecasts: pd.DataFrame
    scores: pd.DataFrame
    predictions: pd.DataFrame
    artifacts: list[Path] = field(default_factory=list)

    @property
    def leaderboard(self) -> pd.DataFrame:
        if self.scores.empty:
            return self.scores
        cols = ["segment_id", "model"]
        return self.scores.groupby(cols, as_index=False).mean(numeric_only=True)


def forecast_segment(seg: SegmentSeries, models: list[str], cfg: Config) -> SegmentOutcome:
    """Backtest every model on one segment, select, and refit for the forward forecast."""
    fc_cfg, bt_cfg = cfg.forecast, cfg.backtest
    report = backtest_series(
        seg.series,
        models=models,
        frequency=fc_cfg.frequency,
        folds=bt_cfg.folds,
        horizon=bt_cfg.horizon,
        step=bt_cfg.step,
        min_train=fc_cfg.min_history,
        interval_width=fc_cfg.interval_width,
    )
    selection = choose_models(report, metric=fc_cfg.selection_metric, ensemble=fc_cfg.ensemble)
    forecast = final_forecast(
        seg.series, selection, fc_cfg.frequency, fc_cfg.horizon, fc_cfg.interval_width
    )

    scores = report.scores.copy()
    preds = report.predictions.copy()
    for frame in (scores, preds):
        if not frame.empty:
            for k, v in seg.as_dict.items():
                frame[k] = v
            frame["segment_id"] = seg.segment_id
    log.info("%-40s -> %s (%s=%.2f)", seg.segment_id, selection.label,
             selection.metric, selection.best_error)
    return SegmentOutcome(seg, forecast, selection.label, selection.confidence,
                          scores, preds, report.failures)


def _forecast_segment_job(args):
    return forecast_segment(*args)


def run_forecasts(segments: list[SegmentSeries], models: list[str], cfg: Config, workers: int = 1):
    if workers <= 1:
        return [forecast_segment(s, models, cfg) for s in segments]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_forecast_segment_job, [(s, models, cfg) for s in segments]))


def build_headroom(outcomes: list[SegmentOutcome], market: pd.DataFrame,
                   summary: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rows = []
    for o in outcomes:
        inputs = SegmentInputs(
            segment=o.segment.as_dict,
            history=o.segment.series,
            forecast=o.forecast,
            frequency=cfg.forecast.frequency,
            confidence=o.confidence,
        )
        row = size_headroom(inputs, market, cfg.sizing)
        row["model"] = o.selection_label
        row["segment_id"] = o.segment.segment_id
        rows.append(row)
    sized = pd.DataFrame(rows)
    if not summary.empty and not sized.empty:
        hist = summary[SEGMENT_KEYS + ["yoy_growth"]].rename(columns={"yoy_growth": "historical_yoy_growth"})
        sized = sized.merge(hist, on=SEGMENT_KEYS, how="left")
    return score_opportunities(sized, cfg.sizing)


def run_pipeline(
    cfg: Config,
    backend: SqlBackend | None = None,
    seed: bool = False,
    persist: bool = True,
    plots: bool = True,
    workers: int = 1,
    segment_filter: str | None = None,
) -> PipelineResult:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    owns_backend = backend is None
    backend = backend or get_backend(cfg)
    try:
        if seed:
            log.info("Seeding synthetic data")
            seed_synthetic(backend, cfg)

        daily = fetch_segment_daily(backend, cfg.data.start_date, cfg.data.end_date)
        if daily.empty:
            raise RuntimeError("No sales data found; run with --seed or load data first")
        segments = build_segment_series(daily, cfg.forecast.frequency, min_history=cfg.forecast.min_history)
        if segment_filter:
            segments = [s for s in segments if segment_filter.lower() in s.segment_id.lower()]
        if not segments:
            raise RuntimeError("No segments with enough history to forecast")
        models = available_models(cfg.forecast.models)
        log.info("Forecasting %d segments with models %s", len(segments), models)

        outcomes = run_forecasts(segments, models, cfg, workers)

        as_of = max(o.segment.series.index[-1] for o in outcomes)
        market = fetch_market_size(backend)
        summary = fetch_segment_summary(backend, daily["ds"].max())
        headroom = build_headroom(outcomes, market, summary, cfg)

        forecasts = pd.concat(
            [o.forecast.to_frame().assign(**o.segment.as_dict, segment_id=o.segment.segment_id)
             for o in outcomes],
            ignore_index=True,
        )
        scores = pd.concat([o.scores for o in outcomes], ignore_index=True)
        predictions = pd.concat([o.predictions for o in outcomes], ignore_index=True)

        result = PipelineResult(run_id, daily, outcomes, headroom, forecasts, scores, predictions)

        if persist:
            run_meta = {
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "frequency": cfg.forecast.frequency,
                "horizon": cfg.forecast.horizon,
                "n_segments": len(outcomes),
                "as_of": pd.Timestamp(as_of),
            }
            persist_results(backend, run_id, run_meta, forecasts, scores, headroom)
            result.artifacts += export_csv(
                cfg.output_path, run_id, forecasts=forecasts, backtest_scores=scores, headroom=headroom
            )

        if plots:
            from trendscope.viz.dashboard import render_all

            result.artifacts += render_all(
                cfg.output_path / run_id,
                daily=daily,
                series={o.segment.segment_id: o.segment.series for o in outcomes},
                forecasts={o.segment.segment_id: o.forecast for o in outcomes},
                leaderboard=result.leaderboard,
                predictions=predictions,
                sized=headroom,
                metric=cfg.forecast.selection_metric,
            )
        return result
    finally:
        if owns_backend:
            backend.close()
