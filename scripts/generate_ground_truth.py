"""
scripts/generate_ground_truth.py

Runs the remaining analytical queries directly against Postgres and
prints each result set as JSON -- no terminal pager, no truncation,
no manual copy-paste. This is the reusable pattern for populating
evaluation_questions.json as the question set grows toward 50-100
entries in later phases.

Note: Q10 is intentionally excluded -- it's a narrative/diagnostic
question, not a query with a single ground-truth result.

Usage:
    python scripts/generate_ground_truth.py            # runs all
    python scripts/generate_ground_truth.py q03 q09    # runs a subset
"""

import json
import os
import sys
from decimal import Decimal

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

REQUIRED_ENV_VARS = ["DB_USER", "DB_PASSWORD", "DB_NAME"]
missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
if missing:
    print(
        f"ERROR: missing required environment variable(s): {', '.join(missing)}\n"
        f"Checked for a .env file starting from: {os.getcwd()}\n"
        "Make sure .env is in the project root and defines DB_USER, "
        "DB_PASSWORD, DB_NAME (and optionally DB_HOST, DB_PORT).",
        file=sys.stderr,
    )
    sys.exit(1)

DB_URL = (
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
    f"/{os.getenv('DB_NAME')}"
)

QUERIES = {
    "q03": """
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
        ORDER BY pct_growth DESC;
    """,
    "q04": """
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
    """,
    "q05": """
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
    """,
    "q08": """
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
    """,
    "q09": """
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
    """,
}


def to_jsonable(value):
    if isinstance(value, Decimal):
        return float(value)
    return str(value) if hasattr(value, "isoformat") else value


def run_query(engine, sql):
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        columns = result.keys()
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
    return [{k: to_jsonable(v) for k, v in row.items()} for row in rows]


def main():
    requested = [q.lower() for q in sys.argv[1:]] or list(QUERIES.keys())
    engine = create_engine(DB_URL)

    output = {}
    for qid in requested:
        if qid not in QUERIES:
            print(f"Skipping unknown id: {qid}", file=sys.stderr)
            continue
        print(f"Running {qid} ...", file=sys.stderr)
        output[qid] = run_query(engine, QUERIES[qid])
        print(f"  -> {len(output[qid])} row(s)", file=sys.stderr)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()