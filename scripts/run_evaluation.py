"""
scripts/run_evaluation.py

Runs every tool-callable question (q01-q09) from evaluation_questions.json
through the Phase 3 pipeline. Checks two things independently:
    1. tool_selection  -- did the LLM pick the correct Phase 2 function?
    2. numeric_correctness -- does the actual tool_result match ground
       truth (within float tolerance), regardless of tool selection?

q10 is excluded -- it's a narrative/diagnostic answer, not a single
tool call, and isn't part of Phase 3's scope.
"""

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.database.connection import get_engine
from app.services.analyst_service import ask

# Ground truth for which tool SHOULD answer each question.
# Not stored in evaluation_questions.json itself, so maintained here.
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
    total = 0

    for entry in questions:
        qid = entry["id"]
        if qid not in EXPECTED_TOOL:
            continue  # skip q10

        total += 1
        result = ask(entry["question"], engine)

        tool_ok = result.tool_name == EXPECTED_TOOL[qid]
        if tool_ok:
            tool_correct += 1

        numeric_ok = False
        if result.error is None:
            numeric_ok = approx_equal(result.tool_result, entry["expected_value"])
            if numeric_ok:
                numeric_correct += 1

        status = "PASS" if (tool_ok and numeric_ok) else "FAIL"
        print(f"[{status}] {qid}: {entry['question']}")
        print(f"    expected tool: {EXPECTED_TOOL[qid]:35s} | got: {result.tool_name}")
        print(f"    tool_selection: {'OK' if tool_ok else 'WRONG'} | numeric: {'OK' if numeric_ok else 'MISMATCH'}")
        if result.error:
            print(f"    error: {result.error}")
        if not numeric_ok and result.error is None:
            print(f"    expected: {entry['expected_value']}")
            print(f"    actual:   {result.tool_result}")
        print()

    print("=" * 60)
    print(f"Tool selection accuracy:  {tool_correct}/{total}")
    print(f"Numeric correctness:      {numeric_correct}/{total}")


if __name__ == "__main__":
    main()