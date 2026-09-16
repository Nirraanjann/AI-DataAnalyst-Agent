import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.database.connection import get_engine
from app.services.analyst_service import ask

questions = [
    # -- Reworded versions of the original 9 --
    ("How much revenue came in each month?", "get_monthly_revenue"),
    ("What's the typical order size in dollars?", "get_average_order_value"),
    ("Best month for sales?", "get_peak_revenue_month"),
    ("Which state's sales are growing the quickest?", "get_fastest_growing_regions"),
    ("Show me the 3 highest-spending customers.", "get_top_customers"),

    # -- Ambiguous phrasing --
    ("What's our best category?", None),  # ambiguous: revenue? margin? volume? -- no single correct tool
    ("How are we doing this quarter?", None),  # too vague for any single tool

    # -- Genuinely out of scope --
    ("What's our customer refund rate?", None),  # no refund data / no tool for this
    ("How many customers do we have in total?", None),  # no tool counts total customers
    ("What's the weather like in Sao Paulo?", None),  # totally unrelated
]

engine = get_engine()

for q, expected in questions:
    result = ask(q, engine)
    print("=" * 60)
    print("Q:", q)
    print("Expected:", expected if expected else "(no confident tool / decline)")
    print("Got tool:", result.tool_name, "| Args:", result.tool_input)
    print("Answer:", result.answer)
    print("Error:", result.error)