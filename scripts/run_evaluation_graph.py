"""
scripts/run_evaluation_graph.py

Graph-version counterpart to run_evaluation.py. Runs every
tool-callable question (q01-q09) from evaluation_questions.json
through the Phase 4 LangGraph agent instead of the Phase 3 pipeline.
Same two checks, same EXPECTED_TOOL map, same approx_equal tolerance
-- only the result-extraction shape differs, since app.agent.graph.ask
returns a GraphState dict with a `tool_calls` list, not an
AnalystResult dataclass with single `tool_name`/`tool_result` fields.

Additionally checks NEEDS CLARIFICATION and DECLINED, since these are
new possible outcomes in Phase 4 that didn't exist in Phase 3 -- a
question that passed cleanly in Phase 3 but now gets flagged as
ambiguous or declined is a regression worth seeing explicitly, not
silently lumped in with "numeric mismatch".

q10 is excluded for the same reason as run_evaluation.py -- narrative/
multi-tool, and (per this session's findings) the current tool set
can't produce its full ground-truth answer anyway.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.database.connection import get_engine
from app.agent.graph import ask

# Same expected-tool map as run_evaluation.py -- kept in sync manually,
# not imported, so this file has no import-time dependency on the
# Phase 3 script.
EXPECTED_TOOL = {
    "q01": "get_monthly_revenue",
    "q02": "get_top_products",
    "q03": "get_fastest_growing_regions",
    "q04": "get_top_customers",
    "q05": "get_declining_products",
    "q06": "get_average_order_value",
    "q07": "get_peak_revenue_month",
    "q08": "get_avg_order_value_by_category",
    "q09": "get_revenue_anomalies",
}


def approx_equal(actual, expected, rel_tol=1e-3) -> bool:
    """Recursively compares floats/ints/strings/dicts/lists, tolerating
    small float differences (rounding, Decimal->float conversion)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return actual == expected
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(actual, expected, rel_tol=rel_tol, abs_tol=1e-6)
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(approx_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        return all(approx_equal(a, e) for a, e in zip(actual, expected))
    return actual == expected


def main():
    eval_path = Path(__file__).resolve().parent.parent / "evaluation_questions.json"
    questions = json.loads(eval_path.read_text())

    engine = get_engine()

    tool_correct = 0
    numeric_correct = 0
    clean_pass = 0  # tool_ok AND numeric_ok AND not clarification/declined
    total = 0

    for entry in questions:
        qid = entry["id"]
        if qid not in EXPECTED_TOOL:
            continue  # skip q10

        total += 1
        result = ask(entry["question"], engine)

        # Phase 4 introduces two outcomes Phase 3 never had: the graph
        # can ask for clarification, or decline, before ever calling a
        # tool. Surface these explicitly rather than let them silently
        # read as "wrong tool" or "numeric mismatch".
        needs_clarification = result["needs_clarification"]
        declined = result["declined"]

        got_tool = result["tool_calls"][0]["tool_name"] if result["tool_calls"] else None
        got_result = result["tool_calls"][0]["result"] if result["tool_calls"] else None

        tool_ok = got_tool == EXPECTED_TOOL[qid]
        if tool_ok:
            tool_correct += 1

        numeric_ok = False
        if not needs_clarification and not declined:
            numeric_ok = approx_equal(got_result, entry["expected_value"])
            if numeric_ok:
                numeric_correct += 1

        clean = tool_ok and numeric_ok and not needs_clarification and not declined
        if clean:
            clean_pass += 1

        status = "PASS" if clean else "FAIL"
        print(f"[{status}] {qid}: {entry['question']}")

        if needs_clarification:
            print(f"    REGRESSION: flagged as needing clarification (Phase 3 answered this directly)")
            print(f"    clarifying question asked: {result['final_answer']}")
        elif declined:
            print(f"    REGRESSION: declined (Phase 3 answered this directly)")
            print(f"    decline reason: {result['decline_reason']}")
        else:
            print(f"    expected tool: {EXPECTED_TOOL[qid]:35s} | got: {got_tool}")
            print(f"    tool_selection: {'OK' if tool_ok else 'WRONG'} | numeric: {'OK' if numeric_ok else 'MISMATCH'}")
            if not numeric_ok:
                print(f"    expected: {entry['expected_value']}")
                print(f"    actual:   {got_result}")
        print()

    print("=" * 60)
    print(f"Tool selection accuracy:  {tool_correct}/{total}")
    print(f"Numeric correctness:      {numeric_correct}/{total}")
    print(f"Clean pass (no clarification/decline): {clean_pass}/{total}")


if __name__ == "__main__":
    main()