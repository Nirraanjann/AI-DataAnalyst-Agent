"""
app/agent/router.py

Sends a natural-language business question to the LLM along with the
9 Phase 2 tool schemas. The LLM picks (at most) one tool and args;
this module extracts that choice without executing anything --
execution happens in analyst_service.py.
"""

from dataclasses import dataclass

from app.agent.tool_definitions import TOOLS
from app.services.llm_client import create_message

SYSTEM_PROMPT = """You are a routing layer for an e-commerce analytics \
system built on the Olist Brazilian e-commerce dataset.

Your only job is to pick the single tool that answers the user's \
question and supply its arguments. Do not answer from your own \
knowledge -- always call a tool.

Non-negotiable definition: revenue = SUM(order_items.price) for \
orders where order_status = 'delivered' only. Freight is excluded. \
If a question implies a different definition of revenue, still use \
this definition.

If no tool genuinely answers the question, do not call the closest \
one anyway -- respond with plain text explaining that no tool covers \
this question."""


@dataclass
class RoutingResult:
    tool_name: str | None
    tool_input: dict
    tool_use_id: str | None
    raw_text: str | None  # populated when the LLM declined to call a tool


def route_question(question: str) -> RoutingResult:
    response = create_message(
        messages=[{"role": "user", "content": question}],
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        max_tokens=512,
    )

    for block in response.content:
        if block.type == "tool_use":
            return RoutingResult(
                tool_name=block.name,
                tool_input=block.input,
                tool_use_id=block.id,
                raw_text=None,
            )

    text_parts = [b.text for b in response.content if b.type == "text"]
    return RoutingResult(
        tool_name=None,
        tool_input={},
        tool_use_id=None,
        raw_text=" ".join(text_parts) or "No tool selected and no explanation given.",
    )