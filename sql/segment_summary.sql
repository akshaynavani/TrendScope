-- Trailing-twelve-month revenue and YoY growth per segment, used by the headroom model.
WITH ttm AS (
    SELECT region, product_line, channel, SUM(revenue) AS ttm_revenue
    FROM {prefix}raw_sales
    WHERE order_date > :ttm_start AND order_date <= :as_of
    GROUP BY region, product_line, channel
),
prior AS (
    SELECT region, product_line, channel, SUM(revenue) AS prior_revenue
    FROM {prefix}raw_sales
    WHERE order_date > :prior_start AND order_date <= :ttm_start
    GROUP BY region, product_line, channel
)
SELECT
    t.region,
    t.product_line,
    t.channel,
    t.ttm_revenue,
    p.prior_revenue,
    CASE WHEN p.prior_revenue > 0
         THEN (t.ttm_revenue - p.prior_revenue) / p.prior_revenue
         ELSE NULL END AS yoy_growth
FROM ttm t
LEFT JOIN prior p
  ON t.region = p.region
 AND t.product_line = p.product_line
 AND t.channel = p.channel
