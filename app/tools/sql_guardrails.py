"""
app/tools/sql_guardrails.py

Phase 5 safety layer for LLM-generated SQL (app/tools/sql_query_tool.py).
Every query the LLM writes passes through here before it ever reaches
the database -- this is the allowlist, sitting on top of (not instead
of) the read-only DB role used by connection.get_readonly_engine() and
that engine's statement_timeout.

Validation is deliberately conservative: reject anything ambiguous
rather than try to be clever about what's "probably fine". A false
rejection just costs a retry; a false acceptance is a real risk.
"""

import re

# Phase 5: hard cap on rows returned by any LLM-generated query,
# independent of what the query itself asks for.
MAX_ROWS = 500

_FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE",
    "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL", "COPY", "VACUUM",
    "REINDEX", "MERGE", "ATTACH", "DETACH", "REPLACE", "SET", "RESET",
]

_COMMENT_RE = re.compile(r"--[^\n]*|/\*.*?\*/", re.DOTALL)
_KEYWORD_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
_LEADING_KEYWORD_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)


class SQLGuardrailError(ValueError):
    """Raised when generated SQL fails allowlist validation."""


def validate_select_only(sql: str) -> str:
    """
    Returns the query (comments stripped, trailing ';' stripped)
    unchanged in substance if it passes. Raises SQLGuardrailError with a
    specific, model-legible reason if it doesn't -- callers should feed
    that message back to the LLM as a tool error so it can retry with a
    corrected query, rather than just failing the whole turn.
    """
    if not sql or not sql.strip():
        raise SQLGuardrailError("Empty query.")

    # Strip comments first so a forbidden keyword or a second statement
    # can't be hidden inside one.
    stripped = _COMMENT_RE.sub(" ", sql).strip()

    # Allow exactly one optional trailing semicolon. Anything else
    # containing ';' is a second statement -- reject.
    body = stripped[:-1] if stripped.endswith(";") else stripped
    if ";" in body:
        raise SQLGuardrailError(
            "Only a single SQL statement is allowed (found a ';' before the end)."
        )

    if not _LEADING_KEYWORD_RE.match(body):
        raise SQLGuardrailError("Only SELECT statements (or WITH ... SELECT) are allowed.")

    match = _KEYWORD_RE.search(body)
    if match:
        raise SQLGuardrailError(
            f"Disallowed keyword '{match.group(1).upper()}' found. Only read-only "
            "SELECT queries are permitted."
        )

    return body


def enforce_row_limit(sql: str, max_rows: int = MAX_ROWS) -> str:
    """
    Wraps the already-validated query in an outer SELECT with a hard
    LIMIT, regardless of whether the query already has its own LIMIT.
    A generated LIMIT could be missing, wrong, or larger than we want to
    hand back to the agent/answer node.
    """
    return f"SELECT * FROM ({sql}) AS _guarded_subquery LIMIT {max_rows}"