-- Raw input tables. `{prefix}` is replaced with "" (SQLite) or "catalog.schema." (Databricks).

CREATE TABLE IF NOT EXISTS {prefix}raw_sales (
    order_id      VARCHAR(64),
    order_date    DATE,
    region        VARCHAR(32),
    product_line  VARCHAR(64),
    channel       VARCHAR(32),
    units         INTEGER,
    revenue       DOUBLE
);

-- Total addressable market per segment and calendar year (from market research).
CREATE TABLE IF NOT EXISTS {prefix}market_size (
    region        VARCHAR(32),
    product_line  VARCHAR(64),
    channel       VARCHAR(32),
    year          INTEGER,
    tam           DOUBLE
);
