"""
app/agent/graph.py

Wires the Phase 4 LangGraph agent:

    START -> planning -+-> (ambiguous) -> answer -> END
                        +-> tool_step -+-> (declined) -> answer -> END
                                       +-> decide -+-> (enough evidence) -> answer -> END
                                                    +-> tool_step (loop)

This is the entry point the rest of the app (tests, eventual FastAPI
endpoint in Phase 8) should call -- analogous to Phase 3's
analyst_service.ask(), but returns the full GraphState rather than an
AnalystResult, since there's now more to inspect (plan, tool_calls
list, whether clarification was needed).
"""

from langgraph.graph import END, StateGraph
from sqlalchemy.engine import Engine

from app.agent.nodes import answer_node, make_decide_node, make_tool_step_node, planning_node
from app.agent.state import GraphState, initial_state


def build_graph(engine: Engine):
    tool_step_node = make_tool_step_node(engine)
    decide_node = make_decide_node()

    graph = StateGraph(GraphState)
    graph.add_node("planning", planning_node)
    graph.add_node("tool_step", tool_step_node)
    graph.add_node("decide", decide_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("planning")

    graph.add_conditional_edges(
        "planning",
        lambda s: "answer" if s["needs_clarification"] else "tool_step",
        {"answer": "answer", "tool_step": "tool_step"},
    )

    graph.add_conditional_edges(
        "tool_step",
        lambda s: "answer" if s["declined"] else "decide",
        {"answer": "answer", "decide": "decide"},
    )

    graph.add_conditional_edges(
        "decide",
        lambda s: "answer" if s["has_enough_evidence"] else "tool_step",
        {"answer": "answer", "tool_step": "tool_step"},
    )

    graph.add_edge("answer", END)

    return graph.compile()


def ask(question: str, engine: Engine, max_steps: int = 4) -> GraphState:
    """Run the agent end to end. Returns the final GraphState -- see
    app/agent/state.py for fields (final_answer, tool_calls,
    needs_clarification, declined, etc.)."""
    compiled = build_graph(engine)
    state = initial_state(question, max_steps=max_steps)
    return compiled.invoke(state)