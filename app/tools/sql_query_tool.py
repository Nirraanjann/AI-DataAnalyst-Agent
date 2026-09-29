"""
app/tools/sql_query_tool.py

Phase 5, hybrid architecture: the fallback path for questions that
don't map to any of the 11 existing Phase 2/4 functions. Two tools are
registered from this file (see tool_definitions.py):

  - get_schema     -- returns table/column names so the LLM can write
                       correct SQL without us hand-maintaining a schema
                       description that drifts from the real database.
  - run_sql_query   -- validates (sql_guardrails) and executes the SQL
                       the LLM wrote, against the read-only engine.

This module does NOT call the LLM itself -- there's no generation step
here. The LLM writes `sql_query` directly as the tool-call argument,
same pattern as every other tool in this project (Approach A: the
model picks a tool and supplies its arguments in one call). Router
preference -- try the 11 fixed functions first, reach for these two
only when nothing else fits -- is a routing/prompt concern that lives
in nodes.py / router.py, not here.
"""

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.tools.sql_guardrails import (
    SQLGuardrailError,
    enforce_row_limit,
    validate_select_only,
)

_schema_cache: str | None = None


def get_schema(engine: Engine) -> dict:
    """
    Introspects the live schema via SQLAlchemy's inspector and returns a
    plain-text table/column description. Cached at module level after
    the first call so it only hits the database once per process, not
    once per question -- the schema doesn't change at runtime.

    Returned as {"schema": "..."} (a dict, not a bare string) to match
    the shape every other tool in this project returns, so it flows
    into graph state the same way.
    """
    global _schema_cache
    if _schema_cache is None:
        inspector = inspect(engine)
        lines = []
        for table_name in sorted(inspector.get_table_names()):
            columns = inspector.get_columns(table_name)
            col_desc = ", ".join(f"{c['name']} ({c['type']})" for c in columns)
            lines.append(f"- {table_name}: {col_desc}")
        _schema_cache = "\n".join(lines)

    return {"schema": _schema_cache}


def run_sql_query(engine: Engine, sql_query: str) -> list[dict] | dict:
    """
    Validates and executes LLM-generated SQL.

    Returns a list of row dicts on success, or {"error": "..."} on a
    guardrail rejection or DB error. Deliberately returns structured
    data rather than raising, so a bad query becomes something the
    agent can see and react to (e.g. retry with corrected SQL) instead
    of crashing the graph.
    """
    try:
        validated = validate_select_only(sql_query)
    except SQLGuardrailError as exc:
        return {"error": f"Query rejected: {exc}"}

    limited = enforce_row_limit(validated)

    try:
        with engine.connect() as conn:
            result = conn.execute(text(limited))
            rows = [dict(row._mapping) for row in result]
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: any
        # DB-level failure (bad column name, syntax error the LLM
        # produced, etc.) should come back as data the agent can react
        # to, not an unhandled exception that crashes the graph.
        return {"error": f"Query failed: {exc}"}

    return rows