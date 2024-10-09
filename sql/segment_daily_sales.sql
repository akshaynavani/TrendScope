-- Daily revenue per segment. Resampling to W/M happens in pandas so the query stays
-- portable between SQLite and Databricks SQL.
SELECT
    region,
    product_line,
    channel,
    order_date               AS ds,
    SUM(revenue)             AS revenue,
    SUM(units)               AS units,
    COUNT(DISTINCT order_id) AS orders
FROM {prefix}raw_sales
WHERE order_date >= :start_date
  AND order_date <= :end_date
GROUP BY region, product_line, channel, order_date
ORDER BY region, product_line, channel, order_date
