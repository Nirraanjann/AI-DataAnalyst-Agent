"""
app/agent/tool_definitions.py

Anthropic-shaped tool-use schemas for the Phase 2 analyst functions,
plus a name -> function registry so the router can dispatch a
tool_use block straight into a Phase 2 call. llm_client.py converts
these to Gemini's functionDeclaration format internally.

`engine` is never part of a tool's input_schema -- it's an internal
dependency the router injects, not something the LLM should ever see
or guess at.

Phase 4 addition: get_revenue_by_category and get_revenue_by_region
are not part of the original Q1-Q9 set. They exist to support chained
investigation questions (e.g. "why did revenue decline in a given
period") where the agent needs to break a specific date range down by
category or region after spotting a trend via get_monthly_revenue.

Phase 5 addition (hybrid architecture): get_schema and run_sql_query
are the fallback path for questions that don't map to any of the 11
functions above. Their descriptions explicitly tell the model to try
the other tools first -- these two exist for genuine gaps, not as a
shortcut around the deterministic tools the evaluation set depends on.
"""

from app.tools import analyst_queries as q
from app.tools import sql_query_tool as sql_tool

TOOLS = [
    {
        "name": "get_monthly_revenue",
        "description": (
            "Total revenue by calendar month, across all history. "
            "Revenue = SUM(order_items.price) for delivered orders "
            "only (freight excluded). Use for questions about revenue "
            "trends over time, month-over-month comparisons, or "
            "'how did revenue change' type questions."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_top_products",
        "description": (
            "Products ranked by total revenue (delivered orders only), "
            "highest first. Use for 'which products/items generated the "
            "most revenue' type questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many top products to return. Defaults to 20.",
                }
            },
        },
    },
    {
        "name": "get_fastest_growing_regions",
        "description": (
            "Customer states ranked by revenue growth % from Jan-Aug "
            "2017 to Jan-Aug 2018 (delivered orders, states with at "
            "least 20 orders in 2017 only). Returns top 10. Use for "
            "'which region/state grew fastest' type questions."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_top_customers",
        "description": (
            "Customers ranked by total revenue (delivered orders only), "
            "highest first, identified by customer_unique_id. Use for "
            "'who are our best/top customers' type questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many top customers to return. Defaults to 10.",
                }
            },
        },
    },
    {
        "name": "get_declining_products",
        "description": (
            "Products ranked by revenue decline % between H2 2017 "
            "(Jul-Dec) and H1 2018 (Jan-Jun), delivered orders only, "
            "restricted to products with sales in both halves. Most "
            "declined first. Use for 'which products are losing sales/ "
            "declining' type questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many declining products to return. Defaults to 20.",
                }
            },
        },
    },
    {
        "name": "get_average_order_value",
        "description": (
            "A single number: the average order value (sum of item "
            "prices per order, delivered orders only) across the whole "
            "dataset. Use for 'what is the average order value' type "
            "questions."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_peak_revenue_month",
        "description": (
            "The single calendar month with the highest total revenue "
            "(delivered orders only). Use for 'which month had the "
            "highest/peak revenue' type questions."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_avg_order_value_by_category",
        "description": (
            "Product categories ranked by average line-item price "
            "(delivered orders only, categories with at least 20 line "
            "items), highest first. This is a substitute for profit "
            "margin -- Olist has no cost data. Use for 'which category "
            "has the highest margin/value' type questions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "How many categories to return. Defaults to 10.",
                }
            },
        },
    },
    {
        "name": "get_revenue_anomalies",
        "description": (
            "Days where daily revenue (delivered orders only) fell more "
            "than 2 standard deviations below its trailing 30-day "
            "average. Use for 'were there any unusual revenue drops/ "
            "anomalies' type questions."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_revenue_by_category",
        "description": (
            "Revenue broken down by product category (delivered orders "
            "only) for a specific date range you provide. Use this to "
            "investigate WHERE a revenue change came from within a "
            "period -- e.g. after spotting a monthly revenue dip via "
            "get_monthly_revenue, use this on that same period to see "
            "which categories drove it. Not useful without a concrete "
            "date range already in mind."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start of the date range, ISO format (YYYY-MM-DD), inclusive.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End of the date range, ISO format (YYYY-MM-DD), inclusive.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_revenue_by_region",
        "description": (
            "Revenue broken down by customer state/region (delivered "
            "orders only) for a specific date range you provide. Use "
            "this to investigate WHERE a revenue change came from "
            "within a period -- e.g. after spotting a monthly revenue "
            "dip via get_monthly_revenue, use this on that same period "
            "to see which regions drove it. Not useful without a "
            "concrete date range already in mind."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start of the date range, ISO format (YYYY-MM-DD), inclusive.",
                },
                "end_date": {
                    "type": "string",
                    "description": "End of the date range, ISO format (YYYY-MM-DD), inclusive.",
                },
            },
            "required": ["start_date", "end_date"],
        },
    },
    {
        "name": "get_schema",
        "description": (
            "Returns the database's table and column names. Call this "
            "ONLY if you've already decided none of the other tools "
            "above can answer the question, and you're about to use "
            "run_sql_query -- you need this first to know what columns "
            "exist. Do not call this for questions the other tools "
            "already cover."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_sql_query",
        "description": (
            "Runs a single, read-only SQL SELECT query you write "
            "yourself. LAST RESORT: only use this if the question "
            "genuinely cannot be answered by any of the other tools "
            "above (for example, a direct numeric comparison across two "
            "periods that needs one query to compute both totals and a "
            "difference). Call get_schema first if you don't already "
            "know the exact table/column names. Only SELECT (or WITH "
            "... SELECT) statements are allowed -- DROP, DELETE, "
            "UPDATE, INSERT, ALTER, TRUNCATE, CREATE, and similar are "
            "rejected before execution. Results are capped at 500 rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "A single read-only SQL SELECT statement.",
                }
            },
            "required": ["sql_query"],
        },
    },
]

REGISTRY = {
    "get_monthly_revenue": q.get_monthly_revenue,
    "get_top_products": q.get_top_products,
    "get_fastest_growing_regions": q.get_fastest_growing_regions,
    "get_top_customers": q.get_top_customers,
    "get_declining_products": q.get_declining_products,
    "get_average_order_value": q.get_average_order_value,
    "get_peak_revenue_month": q.get_peak_revenue_month,
    "get_avg_order_value_by_category": q.get_avg_order_value_by_category,
    "get_revenue_anomalies": q.get_revenue_anomalies,
    "get_revenue_by_category": q.get_revenue_by_category,
    "get_revenue_by_region": q.get_revenue_by_region,
    "get_schema": sql_tool.get_schema,
    "run_sql_query": sql_tool.run_sql_query,
}