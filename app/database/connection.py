"""
app/database/connection.py

Single reusable SQLAlchemy engine factory, extracted from the connection
string logic already used in scripts/load_data.py and
scripts/generate_ground_truth.py. Both old scripts and any new code
(Phase 2 tools, tests, and eventually Phase 3+) should go through this
file rather than building their own connection string.

Phase 5 addition: get_readonly_engine(), a separate engine used only by
app/tools/sql_query_tool.py for LLM-generated SQL. It connects as
DB_READONLY_USER (a Postgres role with SELECT-only grants -- see
scripts/create_readonly_role.sql) and sets a statement_timeout on every
connection, so a runaway or pathological generated query can't hang the
process. This is defense-in-depth *underneath* the allowlist in
sql_guardrails.py, not a replacement for it.
"""

import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()

REQUIRED_ENV_VARS = ["DB_USER", "DB_PASSWORD", "DB_NAME"]

# Phase 5: how long (ms) any single query on the read-only engine may
# run before Postgres kills it. Applies only to get_readonly_engine(),
# not the main engine used by the trusted Phase 2/4 functions.
_QUERY_TIMEOUT_MS = int(os.getenv("SQL_QUERY_TIMEOUT_MS", "5000"))


def _validate_env() -> None:
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


def get_database_url() -> str:
    _validate_env()
    return (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
        f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
        f"/{os.getenv('DB_NAME')}"
    )


_engine: Engine | None = None
_readonly_engine: Engine | None = None


def get_engine() -> Engine:
    """
    Returns a single shared SQLAlchemy engine for the whole process.
    Created lazily on first call, so importing this module never
    triggers a DB connection or .env validation on its own.
    """
    global _engine
    if _engine is None:
        _engine = create_engine(get_database_url())
    return _engine


def get_readonly_engine() -> Engine:
    """
    Phase 5: a separate engine for LLM-generated SQL (SQLQueryTool).

    Connects as DB_READONLY_USER/DB_READONLY_PASSWORD if set. Falls back
    to the main DB_USER/DB_PASSWORD if the read-only role hasn't been
    created yet, so this doesn't hard-fail before
    scripts/create_readonly_role.sql has been run -- but that fallback
    means no real DB-level restriction until the role exists and the
    env vars point at it, so treat it as a setup TODO, not a permanent
    state.
    """
    global _readonly_engine
    if _readonly_engine is None:
        _validate_env()
        user = os.getenv("DB_READONLY_USER", os.getenv("DB_USER"))
        password = os.getenv("DB_READONLY_PASSWORD", os.getenv("DB_PASSWORD"))
        url = (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '5432')}"
            f"/{os.getenv('DB_NAME')}"
        )
        _readonly_engine = create_engine(
            url,
            connect_args={"options": f"-c statement_timeout={_QUERY_TIMEOUT_MS}"},
        )
    return _readonly_engine