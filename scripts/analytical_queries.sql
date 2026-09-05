-- ============================================================
-- AI Data Analyst Agent — Phase 1 Ground-Truth Queries
--
-- GLOBAL ASSUMPTION (applies to every revenue figure below):
--   revenue = SUM(order_items.price)
--   restricted to orders.order_status = 'delivered'
--   (excludes canceled, unavailable, and all in-flight statuses)
--   Freight is NOT included in "revenue" — it's a pass-through
--   shipping cost, not merchandise revenue.
-- ============================================================


-- ------------------------------------------------------------
-- Q1. Total revenue by month
-- ------------------------------------------------------------
SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
    SUM(oi.price) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY 1
ORDER BY 1;


-- ------------------------------------------------------------
-- Q2. Which products generated the highest revenue
-- ------------------------------------------------------------
SELECT
    oi.product_id,
    ct.product_category_name_english,
    SUM(oi.price) AS revenue,
    COUNT(DISTINCT oi.order_id) AS orders_count
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY oi.product_id, ct.product_category_name_english
ORDER BY revenue DESC
LIMIT 20;


-- ------------------------------------------------------------
-- Q3. Which region (customer_state) grew fastest
-- ASSUMPTION: "region" = customer_state. Growth = year-over-year
-- revenue change, 2017 -> 2018, restricted to Jan-Aug for both
-- years (the only months where both years have complete data;
-- the dataset ends Aug/Oct 2018). States with < 20 orders in
-- 2017 are excluded to avoid divide-by-near-zero noise.
-- ------------------------------------------------------------
WITH by_state_year AS (
    SELECT
        c.customer_state,
        EXTRACT(YEAR FROM o.order_purchase_timestamp) AS yr,
        SUM(oi.price) AS revenue,
        COUNT(DISTINCT o.order_id) AS order_count
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
      AND EXTRACT(MONTH FROM o.order_purchase_timestamp) BETWEEN 1 AND 8
      AND EXTRACT(YEAR FROM o.order_purchase_timestamp) IN (2017, 2018)
    GROUP BY 1, 2
)
SELECT
    a.customer_state,
    a.revenue AS revenue_2017,
    b.revenue AS revenue_2018,
    ROUND(((b.revenue - a.revenue) / a.revenue) * 100, 1) AS pct_growth
FROM by_state_year a
JOIN by_state_year b ON b.customer_state = a.customer_state AND b.yr = 2018
WHERE a.yr = 2017 AND a.order_count >= 20
ORDER BY pct_growth DESC
LIMIT 10;


-- ------------------------------------------------------------
-- Q4. Top 10 customers by revenue
-- ASSUMPTION: grouped by customer_unique_id, since the same
-- real person can have multiple customer_id rows across orders.
-- ------------------------------------------------------------
SELECT
    c.customer_unique_id,
    SUM(oi.price) AS revenue,
    COUNT(DISTINCT o.order_id) AS order_count
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
GROUP BY c.customer_unique_id
ORDER BY revenue DESC
LIMIT 10;


-- ------------------------------------------------------------
-- Q5. Which products experienced declining sales
-- ASSUMPTION: compares each product's revenue in H2 2017 vs
-- H1 2018 (the two most complete consecutive half-year windows
-- in the dataset). Only products with revenue > 0 in both
-- windows are eligible, so a decline reflects an existing
-- product slowing down, not a product simply going out of stock.
-- ------------------------------------------------------------
WITH product_half AS (
    SELECT
        oi.product_id,
        CASE
            WHEN o.order_purchase_timestamp BETWEEN '2017-07-01' AND '2017-12-31' THEN 'h2_2017'
            WHEN o.order_purchase_timestamp BETWEEN '2018-01-01' AND '2018-06-30' THEN 'h1_2018'
        END AS half,
        oi.price
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
      AND o.order_purchase_timestamp BETWEEN '2017-07-01' AND '2018-06-30'
),
pivoted AS (
    SELECT
        product_id,
        SUM(price) FILTER (WHERE half = 'h2_2017') AS revenue_h2_2017,
        SUM(price) FILTER (WHERE half = 'h1_2018') AS revenue_h1_2018
    FROM product_half
    GROUP BY product_id
)
SELECT
    product_id,
    revenue_h2_2017,
    revenue_h1_2018,
    ROUND(((revenue_h1_2018 - revenue_h2_2017) / revenue_h2_2017) * 100, 1) AS pct_change
FROM pivoted
WHERE revenue_h2_2017 > 0 AND revenue_h1_2018 > 0
ORDER BY pct_change ASC
LIMIT 20;


-- ------------------------------------------------------------
-- Q6. Average order value
-- ASSUMPTION: order value = SUM(price) per order (freight
-- excluded, consistent with the revenue definition above).
-- ------------------------------------------------------------
WITH order_totals AS (
    SELECT o.order_id, SUM(oi.price) AS order_value
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id
)
SELECT ROUND(AVG(order_value), 2) AS avg_order_value
FROM order_totals;


-- ------------------------------------------------------------
-- Q7. Which month had the highest revenue
-- ------------------------------------------------------------
SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
    SUM(oi.price) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_status = 'delivered'
GROUP BY 1
ORDER BY revenue DESC
LIMIT 1;


-- ------------------------------------------------------------
-- Q8. Which category has the highest average order value
-- (replaces the original profit-margin question — Olist has
-- no cost data, so margin isn't computable)
-- ASSUMPTION: "order value" per category = price per line item
-- attributed to that category (not the full multi-category
-- order total). Categories with < 20 line items excluded.
-- ------------------------------------------------------------
SELECT
    ct.product_category_name_english,
    ROUND(AVG(oi.price), 2) AS avg_item_value,
    COUNT(*) AS line_item_count
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
GROUP BY ct.product_category_name_english
HAVING COUNT(*) >= 20
ORDER BY avg_item_value DESC
LIMIT 10;


-- ------------------------------------------------------------
-- Q9. Are there unusual sales drops
-- ASSUMPTION: flags calendar days where delivered-order revenue
-- falls more than 2 standard deviations below the trailing
-- 30-day mean. Requires at least 30 prior days of history, so
-- the first month of data is excluded from flagging.
-- ------------------------------------------------------------
WITH daily_revenue AS (
    SELECT
        o.order_purchase_timestamp::date AS day,
        SUM(oi.price) AS revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
),
with_rolling_stats AS (
    SELECT
        day,
        revenue,
        AVG(revenue) OVER (ORDER BY day ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS rolling_avg,
        STDDEV(revenue) OVER (ORDER BY day ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS rolling_stddev,
        COUNT(*) OVER (ORDER BY day ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING) AS window_size
    FROM daily_revenue
)
SELECT day, revenue, ROUND(rolling_avg, 2) AS rolling_avg_30d
FROM with_rolling_stats
WHERE window_size = 30
  AND revenue < rolling_avg - 2 * rolling_stddev
ORDER BY day;


-- ------------------------------------------------------------
-- Q10. Why did revenue decline in a selected period
-- This is a diagnostic question, not a single-query answer.
-- It is the target task the eventual LangGraph agent should
-- reproduce: pick a drop identified by Q9 (or a low month from
-- Q1), then break it down by region, category, and order status
-- to build an evidence-backed explanation. Ground truth for this
-- one should be written as a short narrative once Q1/Q3/Q9 have
-- been run against your data and an actual period is picked —
-- it can't be pre-filled generically.
-- ------------------------------------------------------------
-- Supporting breakdown query (parameterize :period_start / :period_end
-- once a specific drop is chosen from Q1 or Q9's output):
--
-- SELECT ct.product_category_name_english, SUM(oi.price) AS revenue
-- FROM orders o
-- JOIN order_items oi ON oi.order_id = o.order_id
-- JOIN products p ON p.product_id = oi.product_id
-- LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
-- WHERE o.order_status = 'delivered'
--   AND o.order_purchase_timestamp BETWEEN :period_start AND :period_end
-- GROUP BY 1
-- ORDER BY revenue DESC;