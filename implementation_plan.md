# TrendScope — Implementation Plan

**Sales Forecasting & Opportunity Sizing**

Tech: Python, SQL, Prophet, ARIMA (statsmodels), Databricks, Seaborn.

## Goal

A batch analytics pipeline that:

1. Ingests transactional sales data with SQL (Databricks SQL warehouse in production,
   SQLite locally so the project runs without a cluster).
2. Aggregates it into per-segment time series (segment = region x product line x channel).
3. Forecasts each segment with Prophet and ARIMA, backtests both, and picks the best
   model per segment (or a weighted ensemble).
4. Sizes "business headroom" per segment: total addressable market (TAM) minus current
   and forecasted captured revenue, adjusted by a penetration ceiling and growth momentum.
5. Renders Seaborn dashboards (trend + forecast bands, accuracy heatmap, headroom bars)
   and writes results back to SQL tables.

## Key design decisions

- **Pluggable SQL backend**: a thin `SqlBackend` abstraction with a SQLAlchemy/SQLite
  implementation and a `databricks-sql-connector` implementation. SQL lives in `.sql`
  files under `sql/` and is shared by both (ANSI-ish subset).
- **Synthetic data generator** for local runs: trend + yearly/weekly seasonality +
  noise + promotional shocks per segment, plus a market-size reference table.
- **Model interface**: every forecaster implements `fit(series)` / `predict(horizon)`
  returning point forecast plus lower/upper intervals, so selection/backtesting is
  model-agnostic. Prophet is an optional import so ARIMA-only runs still work.
- **Model selection** via rolling-origin backtest on MAPE/sMAPE; ties broken by RMSE.
- **Headroom** = `max(0, TAM * penetration_ceiling - forecast_annual_revenue)`, plus a
  prioritisation score combining headroom, forecast growth and forecast confidence.
- Config via YAML + environment variables (`.env.example`); no secrets in repo.

## Phases

### Phase 1 — Scaffolding, schema and ingestion
- `pyproject.toml`, `requirements.txt`, `.env.example`, `config/default.yaml`
- `trendscope/config.py` — typed config loader (YAML + env overrides)
- `sql/schema.sql`, `sql/segment_daily_sales.sql`, `sql/market_size.sql`
- `trendscope/db/backend.py` (abstract), `db/sqlite_backend.py`, `db/databricks_backend.py`
- `trendscope/data/synthetic.py` — synthetic sales + market size generator
- `trendscope/ingest.py` — run schema, load raw data, pull segment series via SQL

### Phase 2 — Forecasting models
- `trendscope/features.py` — resampling, gap filling, outlier capping, train/test split
- `trendscope/models/base.py` — `Forecaster` protocol + `ForecastResult`
- `trendscope/models/arima.py` — SARIMAX with small AIC grid search
- `trendscope/models/prophet_model.py` — Prophet wrapper with holidays/regressors
- `trendscope/models/naive.py` — seasonal-naive baseline
- `trendscope/models/registry.py`

### Phase 3 — Backtesting, model selection and headroom sizing
- `trendscope/evaluation/metrics.py` — MAPE, sMAPE, RMSE, MASE, interval coverage
- `trendscope/evaluation/backtest.py` — rolling-origin cross validation
- `trendscope/selection.py` — best-model / inverse-error ensemble per segment
- `trendscope/sizing/headroom.py` — headroom + opportunity prioritisation score

### Phase 4 — Visualization dashboards and result persistence
- `trendscope/viz/theme.py`, `viz/plots.py`, `viz/dashboard.py` (Seaborn multi-panel PNGs)
- `sql/results_schema.sql`, `trendscope/persist.py` — write forecasts/headroom back

### Phase 5 — Orchestration, Databricks job, tests
- `trendscope/pipeline.py` — end-to-end orchestrator; `trendscope/cli.py` (argparse)
- `databricks/trendscope_job.py` notebook-style entry + `databricks/job.json`
- `tests/` — unit tests for metrics, features, headroom, SQL ingest, pipeline smoke test
- `.gitignore`, `README.md` with architecture diagram
