"""SQL ingestion: create tables, load raw data and pull segment-level series."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from trendscope.config import Config
from trendscope.data.synthetic import generate_sales
from trendscope.db.backend import SqlBackend, load_sql

log = logging.getLogger(__name__)

SEGMENT_KEYS = ["region", "product_line", "channel"]
RAW_SALES_COLUMNS = ["order_id", "order_date", "region", "product_line", "channel", "units", "revenue"]


def init_schema(backend: SqlBackend) -> None:
    backend.execute_script(load_sql("schema"))


def load_raw_sales(backend: SqlBackend, df: pd.DataFrame, replace: bool = True) -> int:
    missing = set(RAW_SALES_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"raw sales is missing columns: {sorted(missing)}")
    df = df[RAW_SALES_COLUMNS].copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.dropna(subset=["order_date", "revenue"])
    backend.write_frame(df, backend.table("raw_sales"), mode="replace" if replace else "append")
    log.info("Loaded %d raw sales rows", len(df))
    return len(df)


def load_market_size(backend: SqlBackend, df: pd.DataFrame, replace: bool = True) -> int:
    backend.write_frame(df, backend.table("market_size"), mode="replace" if replace else "append")
    log.info("Loaded %d market size rows", len(df))
    return len(df)


def load_csv(backend: SqlBackend, sales_csv: str | Path, market_csv: str | Path | None = None) -> None:
    """Load real data exported as CSV into the raw tables."""
    init_schema(backend)
    load_raw_sales(backend, pd.read_csv(sales_csv, parse_dates=["order_date"]))
    if market_csv:
        load_market_size(backend, pd.read_csv(market_csv))


def seed_synthetic(backend: SqlBackend, cfg: Config) -> None:
    init_schema(backend)
    sales, market = generate_sales(cfg.data)
    load_raw_sales(backend, sales)
    load_market_size(backend, market)


def fetch_segment_daily(backend: SqlBackend, start_date: str, end_date: str) -> pd.DataFrame:
    df = backend.run_file("segment_daily_sales", {"start_date": start_date, "end_date": end_date})
    df["ds"] = pd.to_datetime(df["ds"])
    for col in ("revenue", "units", "orders"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_market_size(backend: SqlBackend) -> pd.DataFrame:
    df = backend.run_file("market_size")
    df["year"] = df["year"].astype(int)
    df["tam"] = pd.to_numeric(df["tam"])
    return df


def fetch_segment_summary(backend: SqlBackend, as_of: str | pd.Timestamp) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of)
    params = {
        "as_of": as_of_ts.strftime("%Y-%m-%d"),
        "ttm_start": (as_of_ts - pd.DateOffset(years=1)).strftime("%Y-%m-%d"),
        "prior_start": (as_of_ts - pd.DateOffset(years=2)).strftime("%Y-%m-%d"),
    }
    df = backend.run_file("segment_summary", params)
    for col in ("ttm_revenue", "prior_revenue", "yoy_growth"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def segment_id(row) -> str:
    return " | ".join(str(row[k]) for k in SEGMENT_KEYS)
