import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.database.connection import get_engine
from app.services.analyst_service import ask

questions = [
    "What was the average order value?",
    "Which products generated the highest revenue?",
    "Which month had the highest revenue?",
    "Who are the top 5 customers by revenue?",
    "Were there any unusual drops in revenue?",
]

engine = get_engine()

for q in questions:
    result = ask(q, engine)
    print("=" * 60)
    print("Q:", q)
    print("Tool:", result.tool_name, "| Args:", result.tool_input)
    print("Answer:", result.answer)
    print("Error:", result.error)