"""
app/services/analyst_service.py

End-to-end Phase 3 pipeline:
    question -> router picks tool+args -> call Phase 2 function ->
    LLM phrases the result as a natural-language answer.

This is the one function the rest of the app (tests, eventual API
endpoint) should call.
"""

from dataclasses import dataclass, field
from typing import Any

from sqlalchemy.engine import Engine

from app.agent.router import route_question
from app.agent.tool_definitions import REGISTRY
from app.services.llm_client import create_message

ANSWER_SYSTEM_PROMPT = """You turn a tool's raw query result into a \
short, direct natural-language answer to the user's original \
business question. State the number(s) plainly. Do not add caveats \
that were not in the data. Do not mention SQL, tools, or the \
pipeline -- answer as if you looked this up yourself."""


@dataclass
class AnalystResult:
    question: str
    tool_name: str | None
    tool_input: dict = field(default_factory=dict)
    tool_result: Any = None
    answer: str = ""
    error: str | None = None


def ask(question: str, engine: Engine) -> AnalystResult:
    routing = route_question(question)

    if routing.tool_name is None:
        return AnalystResult(question=question, tool_name=None, answer=routing.raw_text)

    func = REGISTRY.get(routing.tool_name)
    if func is None:
        return AnalystResult(
            question=question,
            tool_name=routing.tool_name,
            tool_input=routing.tool_input,
            error=f"LLM selected unknown tool '{routing.tool_name}'.",
        )

    try:
        result = func(engine, **routing.tool_input)
    except Exception as exc:  # noqa: BLE001 -- surface any tool failure to caller
        return AnalystResult(
            question=question,
            tool_name=routing.tool_name,
            tool_input=routing.tool_input,
            error=f"Tool execution failed: {exc}",
        )

    answer = _phrase_answer(question, routing.tool_name, result)

    return AnalystResult(
        question=question,
        tool_name=routing.tool_name,
        tool_input=routing.tool_input,
        tool_result=result,
        answer=answer,
    )


def _phrase_answer(question: str, tool_name: str, result: Any) -> str:
    prompt = (
        f"Original question: {question}\n"
        f"Tool called: {tool_name}\n"
        f"Tool result (JSON): {result}"
    )
    response = create_message(
        messages=[{"role": "user", "content": prompt}],
        system=ANSWER_SYSTEM_PROMPT,
        max_tokens=512,
    )
    text_parts = [b.text for b in response.content if b.type == "text"]
    return " ".join(text_parts).strip()