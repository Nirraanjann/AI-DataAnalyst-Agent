"""
app/tools/analyst_queries.py

One function per Phase 1 business question (Q1-Q9; Q10 is a narrative
finding, not a queryable function -- see evaluation_questions.json).

SQL bodies are lifted directly from sql/analytical_queries.sql /
scripts/generate_ground_truth.py's QUERIES dict and are NOT rewritten
here. The only intentional deviation from the original .sql text is
swapping hardcoded LIMIT literals for bound :limit parameters, on
functions whose signature takes a `limit` argument -- the query logic
itself (joins, filters, grouping, ordering) is untouched.

Each function returns plain Python types (list[dict], float, dict),
never raw SQLAlchemy Row objects, so this module is a clean boundary
for whatever calls it next (tests today, an LLM tool call in Phase 3+).
"""

from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.engine import Engine


def _to_jsonable(value):
    """Decimal -> float, date/datetime -> ISO date string (YYYY-MM-DD),
    matching the string format already baked into
    evaluation_questions.json. Everything else passes through."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()[:10]
    return value


def _rows_to_dicts(result) -> list[dict]:
    columns = result.keys()
    rows = result.fetchall()
    return [{k: _to_jsonable(v) for k, v in zip(columns, row)} for row in rows]


def _rename_keys(rows: list[dict], mapping: dict) -> list[dict]:
    return [
        {mapping.get(k, k): v for k, v in row.items()}
        for row in rows
    ]


# ------------------------------------------------------------
# Q1. Total revenue by month
# ------------------------------------------------------------
_Q1_SQL = """
    SELECT
        DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
        SUM(oi.price) AS revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    ORDER BY 1;
"""


def get_monthly_revenue(engine: Engine) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(_Q1_SQL))
        return _rows_to_dicts(result)


# ------------------------------------------------------------
# Q2. Which products generated the highest revenue
# (LIMIT literal -> :limit; query logic unchanged)
# ------------------------------------------------------------
_Q2_SQL = """
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
    LIMIT :limit;
"""


def get_top_products(engine: Engine, limit: int = 20) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(_Q2_SQL), {"limit": limit})
        rows = _rows_to_dicts(result)
    return _rename_keys(rows, {"product_category_name_english": "category"})


# ------------------------------------------------------------
# Q3. Which region grew fastest
# (LIMIT 10 stays hardcoded -- this function takes no limit arg,
# matching the handoff's specified signature)
# ------------------------------------------------------------
_Q3_SQL = """
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
"""


def get_fastest_growing_regions(engine: Engine) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(_Q3_SQL))
        return _rows_to_dicts(result)


# ------------------------------------------------------------
# Q4. Top customers by revenue (LIMIT literal -> :limit)
# ------------------------------------------------------------
_Q4_SQL = """
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
    LIMIT :limit;
"""


def get_top_customers(engine: Engine, limit: int = 10) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(_Q4_SQL), {"limit": limit})
        return _rows_to_dicts(result)


# ------------------------------------------------------------
# Q5. Declining products (LIMIT literal -> :limit)
# ------------------------------------------------------------
_Q5_SQL = """
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
    LIMIT :limit;
"""


def get_declining_products(engine: Engine, limit: int = 20) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(_Q5_SQL), {"limit": limit})
        return _rows_to_dicts(result)


# ------------------------------------------------------------
# Q6. Average order value (scalar)
# ------------------------------------------------------------
_Q6_SQL = """
    WITH order_totals AS (
        SELECT o.order_id, SUM(oi.price) AS order_value
        FROM orders o
        JOIN order_items oi ON oi.order_id = o.order_id
        WHERE o.order_status = 'delivered'
        GROUP BY o.order_id
    )
    SELECT ROUND(AVG(order_value), 2) AS avg_order_value
    FROM order_totals;
"""


def get_average_order_value(engine: Engine) -> float:
    with engine.connect() as conn:
        value = conn.execute(text(_Q6_SQL)).scalar_one()
    return float(value)


# ------------------------------------------------------------
# Q7. Peak revenue month (single row)
# ------------------------------------------------------------
_Q7_SQL = """
    SELECT
        DATE_TRUNC('month', o.order_purchase_timestamp)::date AS month,
        SUM(oi.price) AS revenue
    FROM orders o
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY 1
    ORDER BY revenue DESC
    LIMIT 1;
"""


def get_peak_revenue_month(engine: Engine) -> dict:
    with engine.connect() as conn:
        result = conn.execute(text(_Q7_SQL))
        rows = _rows_to_dicts(result)
    return rows[0]


# ------------------------------------------------------------
# Q8. Avg order value by category (LIMIT literal -> :limit)
# ------------------------------------------------------------
_Q8_SQL = """
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
    LIMIT :limit;
"""


def get_avg_order_value_by_category(engine: Engine, limit: int = 10) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(_Q8_SQL), {"limit": limit})
        rows = _rows_to_dicts(result)
    return _rename_keys(rows, {"product_category_name_english": "category"})


# ------------------------------------------------------------
# Q9. Revenue anomalies (no LIMIT in the original query)
# ------------------------------------------------------------
_Q9_SQL = """
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
"""


def get_revenue_anomalies(engine: Engine) -> list[dict]:
    with engine.connect() as conn:
        result = conn.execute(text(_Q9_SQL))
        return _rows_to_dicts(result)