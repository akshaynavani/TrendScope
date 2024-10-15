"""Write pipeline results back to SQL and to CSV for downstream BI tools."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from trendscope.db.backend import SqlBackend, load_sql

log = logging.getLogger(__name__)

TABLE_COLUMNS = {
    "segment_forecasts": ["run_id", "region", "product_line", "channel", "ds", "yhat",
                          "yhat_lower", "yhat_upper", "model"],
    "backtest_scores": ["run_id", "region", "product_line", "channel", "model", "fold", "cutoff",
                        "mae", "rmse", "mape", "smape", "bias", "coverage", "mase"],
    "segment_headroom": ["run_id", "region", "product_line", "channel", "model", "confidence",
                         "ttm_revenue", "forecast_annual_revenue", "forecast_growth", "forecast_year",
                         "tam_current", "tam_forecast", "current_share", "forecast_share",
                         "capturable_revenue", "headroom", "headroom_low", "headroom_high",
                         "opportunity_score", "rank", "headroom_share", "tier"],
    "forecast_runs": ["run_id", "created_at", "frequency", "horizon", "n_segments", "as_of"],
}


def init_results_schema(backend: SqlBackend) -> None:
    backend.execute_script(load_sql("results_schema"))


def _conform(df: pd.DataFrame, table: str, run_id: str) -> pd.DataFrame:
    out = df.copy()
    out["run_id"] = run_id
    cols = TABLE_COLUMNS[table]
    for col in cols:
        if col not in out.columns:
            out[col] = None
    return out[cols]


def persist_results(
    backend: SqlBackend,
    run_id: str,
    run_meta: dict,
    forecasts: pd.DataFrame,
    scores: pd.DataFrame,
    headroom: pd.DataFrame,
) -> None:
    init_results_schema(backend)
    frames = {
        "segment_forecasts": forecasts,
        "backtest_scores": scores,
        "segment_headroom": headroom,
        "forecast_runs": pd.DataFrame([run_meta]),
    }
    for table, df in frames.items():
        if df is None or df.empty:
            continue
        backend.write_frame(_conform(df, table, run_id), backend.table(table), mode="append")
        log.info("Persisted %d rows to %s", len(df), table)


def export_csv(output_dir: Path, run_id: str, **frames: pd.DataFrame) -> list[Path]:
    out = Path(output_dir) / "exports" / run_id
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in frames.items():
        if df is None or df.empty:
            continue
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        written.append(path)
    return written
