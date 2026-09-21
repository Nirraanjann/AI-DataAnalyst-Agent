"""
app/agent/nodes.py

LangGraph node functions for the Phase 4 agent. None of these rewrite
router.py, tool_definitions.py, or analyst_queries.py -- they call
those as-is, per the Phase 4 handoff's DO NOT CHANGE list.

Multi-step tool calling (needed for q10-style "why did revenue
decline" questions) is achieved by feeding route_question() a richer
prompt string on iterations after the first, not by changing
route_question()'s signature or behavior.
"""

from typing import Callable

from sqlalchemy.engine import Engine

from app.agent.router import route_question
from app.agent.state import GraphState, ToolCallRecord
from app.agent.tool_definitions import REGISTRY, TOOLS
from app.services.llm_client import create_message

# ---------------------------------------------------------------------------
# Planning / disambiguation node
# (Phase 3 KNOWN PROBLEM #1: ambiguous questions were silently resolved to
# one interpretation. This node makes that choice explicit, or asks.)
# ---------------------------------------------------------------------------

def _tools_summary() -> str:
    return "\n".join(f"- {t['name']}: {t['description']}" for t in TOOLS)


# Built once at import time from the real tool list, so the planning node's
# ambiguity judgment is grounded in what tools actually exist -- not in
# abstract reasoning about what a business question could theoretically
# mean. This was the fix for a real regression: "which region grew
# fastest?" was being flagged as ambiguous ("by revenue, order volume, or
# customer acquisition?") even though only ONE tool
# (get_fastest_growing_regions) exists to answer it -- the model had no
# way to know that without seeing the actual tool inventory.
PLANNING_SYSTEM_PROMPT = f"""You are the planning/disambiguation step for an \
e-commerce analytics agent (Olist dataset). Below is the exact, complete \
list of tools this system can call -- there are no others:

{_tools_summary()}

You do not answer the question and you do not call tools yourself. Your \
only job is to decide whether the question is genuinely ambiguous, GIVEN \
THIS EXACT TOOL LIST -- not based on what a business question could \
theoretically mean in the abstract.

A question is AMBIGUOUS only if it could reasonably match two or more \
tools FROM THE LIST ABOVE that would give different, non-overlapping \
answers -- for example, "best category" could match get_top_products \
(highest revenue) or get_avg_order_value_by_category (highest average \
value), which are two real, different tools with two real, different \
answers.

A question is NOT ambiguous in these cases -- treat it as CLEAR:
- The question matches exactly one tool from the list above, even if the \
wording (e.g. "grew fastest", "declining", "best") could in the abstract \
mean something else -- if no tool in the list above measures that other \
thing, it isn't a real ambiguity, just a hypothetical one. Use the one \
matching tool's own built-in methodology (time window, definition, etc.) \
without asking.
- The question doesn't specify an optional detail that has a sensible \
default (how many results to return, which identifier to show items by).
- The question is casually or informally phrased but still matches \
exactly one tool.

When in doubt, prefer CLEAR over AMBIGUOUS. Only flag AMBIGUOUS when two \
or more tools IN THE LIST ABOVE would both genuinely apply and give \
different answers -- never based on interpretations with no matching tool.

Separately, a question can also be COMPOUND: it asks about two or more \
completely distinct business topics in one message (e.g. "What is the \
average order value, AND SEPARATELY, which products generated the most \
revenue?" -- two unrelated questions joined into one message). This is \
different from AMBIGUOUS -- a compound question isn't unclear, it just \
has multiple parts that will need to be answered one at a time. Only use \
COMPOUND when the topics are genuinely unrelated business questions, not \
when a single tool's own output naturally covers several related facets \
of one topic.

Respond with exactly one line in one of these three formats, nothing else:
CLEAR: <one-sentence restatement of what you will answer>
COMPOUND: <one-sentence noting the distinct topics found>
AMBIGUOUS: <the clarifying question to ask the user>"""


def planning_node(state: GraphState) -> GraphState:
    response = create_message(
        messages=[{"role": "user", "content": state["question"]}],
        system=PLANNING_SYSTEM_PROMPT,
        max_tokens=200,
    )
    text = " ".join(
        b.text for b in response.content if b.type == "text" and b.text
    ).strip()

    if text.upper().startswith("AMBIGUOUS:"):
        state["needs_clarification"] = True
        state["clarification_question"] = text.split(":", 1)[1].strip()
        state["is_compound"] = False
    elif text.upper().startswith("COMPOUND:"):
        state["needs_clarification"] = False
        state["is_compound"] = True
        state["plan"] = text.split(":", 1)[1].strip() if ":" in text else text
    else:
        state["needs_clarification"] = False
        state["is_compound"] = False
        state["plan"] = text.split(":", 1)[1].strip() if ":" in text else text

    return state


# ---------------------------------------------------------------------------
# Tool-step node: routes (via router.route_question, unmodified) and
# executes exactly one tool call per invocation. The graph loops back into
# this node when more evidence is needed.
# ---------------------------------------------------------------------------


def _build_routing_prompt(state: GraphState) -> str:
    """First call on an ordinary (non-compound) question: pass the
    question through completely unchanged -- exactly the original,
    working behavior, with no framing text anywhere near it, so nothing
    can bias a tool's own parameters (e.g. limit).

    The multi-topic framing below is ONLY added when the planning node
    already identified the question as genuinely compound
    (state["is_compound"]). Applying it unconditionally to every
    question was tried and caused a real regression (q08 got limit=1
    injected on an ordinary single-topic question) -- gating it behind
    an explicit planning-node decision, rather than prose disclaimers,
    is the fix."""
    framing = (
        "You are one step in a multi-step process that can call more "
        "than one tool across several turns, because the ORIGINAL "
        "question below was identified as having more than one "
        "distinct business topic. Call the tool for just ONE of those "
        "topics now -- you will be asked again for the remaining "
        "topic(s), so do not decline just because a single tool can't "
        "cover every topic by itself. This is about how many topics "
        "are in the question, not about how many rows any tool "
        "returns -- if a tool has its own result-count parameter (e.g. "
        "\"top N\"), always use that tool's normal default."
    )

    if not state["tool_calls"]:
        if state.get("is_compound"):
            return f"{framing}\n\nQuestion: {state['question']}"
        return state["question"]

    evidence_lines = [
        f"{i}. Called `{c['tool_name']}` with args {c['tool_input']} -> result: {c['result']}"
        for i, c in enumerate(state["tool_calls"], start=1)
    ]
    evidence_block = "\n".join(evidence_lines)

    prefix = f"{framing}\n\n" if state.get("is_compound") else ""
    return (
        f"{prefix}"
        f"Original question: {state['question']}\n\n"
        f"Tool calls made so far:\n{evidence_block}\n\n"
        "Based on what's already known, what ADDITIONAL tool call (if "
        "any) is needed to fully answer every part of the original "
        "question? If every part is already covered, do not call a "
        "tool -- respond with plain text saying the existing evidence "
        "is sufficient."
    )


def make_tool_step_node(engine: Engine) -> Callable[[GraphState], GraphState]:
    def tool_step_node(state: GraphState) -> GraphState:
        prompt_question = _build_routing_prompt(state)
        routing = route_question(prompt_question)
        state["step_count"] += 1

        if routing.tool_name is None:
            if state["step_count"] == 1:
                # No tool at all on the very first attempt -- genuinely
                # out of scope, matches Phase 3's decline behavior.
                state["declined"] = True
                state["decline_reason"] = routing.raw_text
            else:
                # "No further tool needed" on a later step is a normal
                # loop-exit signal, not a decline.
                state["has_enough_evidence"] = True
            return state

        func = REGISTRY.get(routing.tool_name)
        if func is None:
            state["declined"] = True
            state["decline_reason"] = f"LLM selected unknown tool '{routing.tool_name}'."
            return state

        try:
            result = func(engine, **routing.tool_input)
        except Exception as exc:  # noqa: BLE001 -- surface any tool failure to caller
            state["declined"] = True
            state["decline_reason"] = f"Tool execution failed: {exc}"
            return state

        state["tool_calls"].append(
            ToolCallRecord(
                tool_name=routing.tool_name,
                tool_input=routing.tool_input,
                result=result,
            )
        )
        return state

    return tool_step_node


# ---------------------------------------------------------------------------
# Decide node: after a successful tool call, is there enough evidence to
# answer, or should the graph loop back for another tool call?
# Termination safety: max_steps is enforced here, independent of what the
# LLM says, so the loop cannot run indefinitely.
# ---------------------------------------------------------------------------

DECIDE_SYSTEM_PROMPT = """You are deciding whether an analytics agent has \
enough evidence to fully answer the user's ORIGINAL question, given the \
tool results gathered so far.

First check: does the original question have more than one distinct part \
(e.g. "X, and separately Y")? If so, has a tool result already addressed \
EACH part, or only some of them?

Respond with exactly one word: YES if every distinct part of the original \
question is now covered by a tool result, or NO if at least one part \
still has no tool result addressing it."""


def make_decide_node() -> Callable[[GraphState], GraphState]:
    def decide_node(state: GraphState) -> GraphState:
        if state["declined"] or state["has_enough_evidence"]:
            return state

        if state["step_count"] >= state["max_steps"]:
            # Hard cap -- answer with whatever evidence exists rather than
            # loop further, regardless of what the LLM would otherwise say.
            state["has_enough_evidence"] = True
            return state

        evidence_lines = [
            f"- {c['tool_name']}({c['tool_input']}) -> {c['result']}"
            for c in state["tool_calls"]
        ]
        prompt = (
            f"Original question: {state['question']}\n\n"
            f"Tool results so far:\n" + "\n".join(evidence_lines)
        )
        response = create_message(
            messages=[{"role": "user", "content": prompt}],
            system=DECIDE_SYSTEM_PROMPT,
            max_tokens=10,
        )
        text = " ".join(
            b.text for b in response.content if b.type == "text" and b.text
        ).strip().upper()

        state["has_enough_evidence"] = text.startswith("YES")
        return state

    return decide_node


# ---------------------------------------------------------------------------
# Answer node: describes tool results only. Explicitly forbidden from doing
# its own arithmetic (Phase 3 KNOWN PROBLEM #2) -- if a derived number would
# require combining tool outputs, it says so instead of computing it.
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """You turn one or more tool results into a short, \
direct natural-language answer to the user's original business question.

State the numbers plainly, exactly as given in the tool results. Do NOT \
perform any arithmetic, aggregation, or derived calculation that is not \
already present in a tool result verbatim -- for example, do not sum, \
average, or otherwise combine numbers from one or more results. If fully \
answering the question would require such a combined figure that no tool \
result already contains, report the individual numbers you do have and \
explicitly say the combined figure was not computed, rather than \
computing it yourself.

Do not mention SQL, tools, or the pipeline -- answer as if you looked this \
up yourself."""


def answer_node(state: GraphState) -> GraphState:
    if state["needs_clarification"]:
        state["final_answer"] = state["clarification_question"]
        return state

    if state["declined"]:
        state["final_answer"] = (
            state["decline_reason"] or "I don't have a tool that answers this."
        )
        return state

    evidence_lines = [
        f"- {c['tool_name']}({c['tool_input']}) -> {c['result']}"
        for c in state["tool_calls"]
    ]
    prompt = (
        f"Original question: {state['question']}\n\n"
        f"Tool results:\n" + "\n".join(evidence_lines)
    )
    response = create_message(
        messages=[{"role": "user", "content": prompt}],
        system=ANSWER_SYSTEM_PROMPT,
        max_tokens=768,
    )
    text_parts = [b.text for b in response.content if b.type == "text" and b.text]
    state["final_answer"] = " ".join(text_parts).strip()
    return state