import pandas as pd

from trendscope.data.synthetic import generate_sales
from trendscope.db.backend import split_statements
from trendscope.ingest import (
    fetch_market_size,
    fetch_segment_daily,
    fetch_segment_summary,
    seed_synthetic,
)


def test_split_statements_drops_comments():
    sql = "-- header\nCREATE TABLE a (x INT);\n-- only a comment\n;SELECT 1"
    assert split_statements(sql) == ["CREATE TABLE a (x INT)", "SELECT 1"]


def test_synthetic_generator_shapes(small_config):
    sales, market = generate_sales(small_config.data)
    assert set(sales["region"]) == {"NA", "EMEA"}
    assert sales["order_id"].is_unique
    assert (sales["revenue"] > 0).all()
    # Market table covers history plus forward-looking years.
    assert market["year"].max() == 2027
    assert (market["tam"] > 0).all()


def test_seed_and_query_roundtrip(memory_backend, small_config):
    seed_synthetic(memory_backend, small_config)
    daily = fetch_segment_daily(memory_backend, "2022-01-01", "2024-06-30")
    assert not daily.empty
    assert {"region", "product_line", "channel", "ds", "revenue", "orders"} <= set(daily.columns)
    assert pd.api.types.is_datetime64_any_dtype(daily["ds"])
    # One row per segment-day.
    assert not daily.duplicated(["region", "product_line", "channel", "ds"]).any()

    market = fetch_market_size(memory_backend)
    assert len(market) == 2 * (2027 - 2022 + 1)

    summary = fetch_segment_summary(memory_backend, "2024-06-30")
    assert len(summary) == 2
    assert summary["ttm_revenue"].gt(0).all()
    assert summary["yoy_growth"].notna().all()


def test_seed_replaces_existing_rows(memory_backend, small_config):
    seed_synthetic(memory_backend, small_config)
    first = memory_backend.query("SELECT COUNT(*) AS n FROM raw_sales")["n"].iloc[0]
    seed_synthetic(memory_backend, small_config)
    second = memory_backend.query("SELECT COUNT(*) AS n FROM raw_sales")["n"].iloc[0]
    assert first == second
