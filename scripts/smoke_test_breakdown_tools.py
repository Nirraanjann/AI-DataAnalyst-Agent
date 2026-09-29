from app.database.connection import get_engine
from app.agent.graph import ask

engine = get_engine()

question = "Why did revenue decline from May 2018 to June 2018?"
result = ask(question, engine)

print("Final answer:")
print(result["final_answer"])
print()
print(f"Step count: {result['step_count']}")
print(f"Tool calls made:")
for call in result["tool_calls"]:
    print(f"  - {call}")
    