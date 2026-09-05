-- Q10 Diagnostic Step 1: rule out a delivery-status artifact.
-- Compares TOTAL orders placed per month (any status) against
-- DELIVERED orders/revenue per month. If total orders placed in
-- June 2018 is roughly flat vs May but delivered revenue drops,
-- that points to a delivery/status lag, not a real demand decline.

SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
    COUNT(DISTINCT o.order_id) AS total_orders_placed,
    COUNT(DISTINCT o.order_id) FILTER (WHERE o.order_status = 'delivered') AS delivered_orders,
    COALESCE(SUM(oi.price) FILTER (WHERE o.order_status = 'delivered'), 0) AS delivered_revenue
FROM orders o
LEFT JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_purchase_timestamp BETWEEN '2018-03-01' AND '2018-08-31'
GROUP BY 1
ORDER BY 1;


-- Q10 Diagnostic Step 2: if Step 1 shows demand IS genuinely down
-- (total_orders_placed also drops in June), break down June 2018
-- revenue by category vs May 2018 to find which categories drove it.

SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
    ct.product_category_name_english,
    SUM(oi.price) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
LEFT JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
  AND o.order_purchase_timestamp BETWEEN '2018-05-01' AND '2018-06-30'
GROUP BY 1, 2
ORDER BY 2, 1;


-- Q10 Diagnostic Step 3: same breakdown by region (customer_state),
-- to check if the dip is broad-based or concentrated in one region.

SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
    c.customer_state,
    SUM(oi.price) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_status = 'delivered'
  AND o.order_purchase_timestamp BETWEEN '2018-05-01' AND '2018-06-30'
GROUP BY 1, 2
ORDER BY 2, 1;


-- Q10 Diagnostic Step 4: test the Mother's Day seasonality hypothesis.
-- If May 2018 was inflated by pre-Mother's-Day gifting, watches_gifts
-- (and possibly perfumery/health_beauty) should show a spike in May
-- relative to both April and June, i.e. a single-month bump-then-drop
-- shape, not a sustained plateau followed by a real cliff.

SELECT
    DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
    ct.product_category_name_english,
    SUM(oi.price) AS revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
JOIN products p ON p.product_id = oi.product_id
JOIN category_translation ct ON ct.product_category_name = p.product_category_name
WHERE o.order_status = 'delivered'
  AND o.order_purchase_timestamp BETWEEN '2018-02-01' AND '2018-07-31'
  AND ct.product_category_name_english IN ('watches_gifts', 'perfumery', 'health_beauty', 'garden_tools', 'sports_leisure')
GROUP BY 1, 2
ORDER BY 2, 1;