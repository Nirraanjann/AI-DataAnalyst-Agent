"""
app/database/connection.py

Single reusable SQLAlchemy engine factory, extracted from the connection
string logic already used in scripts/load_data.py and
scripts/generate_ground_truth.py. Both old scripts and any new code
(Phase 2 tools, tests, and eventually Phase 3+) should go through this
file rather than building their own connection string.
"""

import os
import sys

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()

REQUIRED_ENV_VARS = ["DB_USER", "DB_PASSWORD", "DB_NAME"]


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