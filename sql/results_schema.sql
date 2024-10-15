-- Output tables written by the pipeline. Every row is tagged with the run that produced it.

CREATE TABLE IF NOT EXISTS {prefix}forecast_runs (
    run_id        VARCHAR(64),
    created_at    VARCHAR(32),
    frequency     VARCHAR(4),
    horizon       INTEGER,
    n_segments    INTEGER,
    as_of         DATE
);

CREATE TABLE IF NOT EXISTS {prefix}segment_forecasts (
    run_id        VARCHAR(64),
    region        VARCHAR(32),
    product_line  VARCHAR(64),
    channel       VARCHAR(32),
    ds            DATE,
    yhat          DOUBLE,
    yhat_lower    DOUBLE,
    yhat_upper    DOUBLE,
    model         VARCHAR(128)
);

CREATE TABLE IF NOT EXISTS {prefix}backtest_scores (
    run_id        VARCHAR(64),
    region        VARCHAR(32),
    product_line  VARCHAR(64),
    channel       VARCHAR(32),
    model         VARCHAR(64),
    fold          INTEGER,
    cutoff        DATE,
    mae           DOUBLE,
    rmse          DOUBLE,
    mape          DOUBLE,
    smape         DOUBLE,
    bias          DOUBLE,
    coverage      DOUBLE,
    mase          DOUBLE
);

CREATE TABLE IF NOT EXISTS {prefix}segment_headroom (
    run_id                  VARCHAR(64),
    region                  VARCHAR(32),
    product_line            VARCHAR(64),
    channel                 VARCHAR(32),
    model                   VARCHAR(128),
    confidence              DOUBLE,
    ttm_revenue             DOUBLE,
    forecast_annual_revenue DOUBLE,
    forecast_growth         DOUBLE,
    forecast_year           INTEGER,
    tam_current             DOUBLE,
    tam_forecast            DOUBLE,
    current_share           DOUBLE,
    forecast_share          DOUBLE,
    capturable_revenue      DOUBLE,
    headroom                DOUBLE,
    headroom_low            DOUBLE,
    headroom_high           DOUBLE,
    opportunity_score       DOUBLE,
    rank                    INTEGER,
    headroom_share          DOUBLE,
    tier                    VARCHAR(16)
);

-- Convenience view: latest run's headroom ranking.
CREATE VIEW IF NOT EXISTS {prefix}latest_headroom AS
SELECT h.*
FROM {prefix}segment_headroom h
WHERE h.run_id = (SELECT MAX(run_id) FROM {prefix}forecast_runs);
