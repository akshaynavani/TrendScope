"""Command-line entry point: ``trendscope <command>``."""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from trendscope.config import load_config
from trendscope.db.backend import get_backend
from trendscope.ingest import load_csv, seed_synthetic


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="Path to YAML config (default: config/default.yaml)")
    p.add_argument("--backend", choices=["sqlite", "databricks"], help="Override backend")
    p.add_argument("-v", "--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trendscope", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_seed = sub.add_parser("seed", help="Generate synthetic sales + market data into the backend")
    _common(p_seed)

    p_load = sub.add_parser("load", help="Load real sales/market data from CSV")
    _common(p_load)
    p_load.add_argument("--sales", required=True, help="CSV with order_id,order_date,region,...")
    p_load.add_argument("--market", help="CSV with region,product_line,channel,year,tam")

    p_run = sub.add_parser("run", help="Forecast, backtest, size headroom and render dashboards")
    _common(p_run)
    p_run.add_argument("--seed", action="store_true", help="Seed synthetic data first")
    p_run.add_argument("--frequency", choices=["D", "W", "M"])
    p_run.add_argument("--horizon", type=int)
    p_run.add_argument("--models", help="Comma-separated, e.g. arima,prophet")
    p_run.add_argument("--segment", help="Only forecast segments whose id contains this text")
    p_run.add_argument("--workers", type=int, default=1, help="Parallel worker processes")
    p_run.add_argument("--no-persist", action="store_true", help="Skip writing results to SQL/CSV")
    p_run.add_argument("--no-plots", action="store_true", help="Skip rendering dashboards")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    if args.backend:
        cfg.backend = args.backend

    if args.command == "seed":
        with get_backend(cfg) as backend:
            seed_synthetic(backend, cfg)
        return 0

    if args.command == "load":
        with get_backend(cfg) as backend:
            load_csv(backend, args.sales, args.market)
        return 0

    if args.command == "run":
        from trendscope.pipeline import run_pipeline

        if args.frequency:
            cfg.forecast.frequency = args.frequency
        if args.horizon:
            cfg.forecast.horizon = args.horizon
        if args.models:
            cfg.forecast.models = [m.strip() for m in args.models.split(",") if m.strip()]

        result = run_pipeline(
            cfg,
            seed=args.seed,
            persist=not args.no_persist,
            plots=not args.no_plots,
            workers=args.workers,
            segment_filter=args.segment,
        )
        cols = ["opportunity_rank", "segment_id", "tier", "model", "forecast_annual_revenue",
                "headroom", "forecast_growth", "opportunity_score"]
        with pd.option_context("display.width", 160, "display.float_format", "{:,.2f}".format):
            print(result.headroom[cols].head(10).to_string(index=False))
        print(f"\nRun {result.run_id}: {len(result.outcomes)} segments, "
              f"{len(result.artifacts)} artifacts written to {cfg.output_path / result.run_id}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
