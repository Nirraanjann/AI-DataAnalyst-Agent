import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.database.connection import get_engine
from app.agent.graph import ask

engine = get_engine()

result = ask("What is the average order value, and separately, which products generated the highest revenue?", engine)

print("IS COMPOUND:", result["is_compound"])
print("STEP COUNT:", result["step_count"])
print()
print("TOOL CALLS:")
for c in result["tool_calls"]:
    print(f"  {c['tool_name']}({c['tool_input']})")
print()
print("FINAL ANSWER:")
print(result["final_answer"])