"""
app/agent/nodes.py

LangGraph node functions for the Phase 4 agent. None of these rewrite
router.py's route_question() signature/behavior, analyst_queries.py's
existing Q1-Q9 functions, or tool_definitions.py's TOOLS/REGISTRY
structure for the original 11 entries -- Phase 5 only ADDED two new
entries (get_schema, run_sql_query) to that file, per the Phase 4
handoff's DO NOT CHANGE list.

Multi-step tool calling (needed for q10-style "why did revenue
decline" questions) is achieved by feeding route_question() a richer
prompt string on iterations after the first, not by changing
route_question()'s signature or behavior.

Update (Phase 4): added comparison-awareness to the routing prompt and
the decide node. A period-over-period comparison question (e.g. "why
did revenue decline from May to June?") was being answered with a
SINGLE tool call using a combined date range spanning both periods,
which can only show a total, not a comparison. Both the routing prompt
(so a second, per-period call gets requested) and the decide node's
YES/NO check (so a merged-range result isn't mistaken for "enough
evidence") needed the fix -- fixing only one would still cut the loop
short.

Update (Phase 5): two changes to support the new SQL fallback tools.
(1) tool_step_node now routes get_schema/run_sql_query calls to the
read-only DB engine (connection.get_readonly_engine()) instead of the
main engine, so the DB-level restriction from
scripts/create_readonly_role.sql actually applies -- every other tool
still uses the main engine, unchanged. (2) the comparison-question
logic (both the routing prompt's comparison_note and the decide node's
Check 2) was generalized: it used to assume a comparison could only be
satisfied by two separate per-period calls to the same breakdown tool.
Now a single run_sql_query result is also accepted as sufficient, as
long as that result itself contains a separate, identifiable figure
for each period being compared (not just a combined total) --
otherwise the decide node would wrongly force a second, redundant call
even when one well-written SQL query already answered the comparison
directly.

Update (Phase 5, retest fix): raw tool results are now sanitized right
after a successful call, converting any Decimal values (psycopg2's
return type for Postgres NUMERIC/DECIMAL columns -- returned by both
run_sql_query and the original Phase 2 functions) to plain floats.
Without this, decide_node and answer_node's prompts showed literal
text like "Decimal('111512.94')" in the final answer, since Python's
default str()/repr of a Decimal includes the class name. Sanitizing
once here, right after the tool call, means every downstream node
(decide, answer) automatically sees clean numbers with no other
changes needed.

Update (Phase 5, deterministic get_schema enforcement): two separate
retests saw the LLM attempt run_sql_query with guessed (wrong, from
the original Kaggle CSV filenames) table names before ever calling
get_schema, despite the router prompt explicitly saying to call
get_schema first. It self-corrected via the resulting DB error both
times, but at the cost of a wasted step each time. tool_step_node now
tracks state["schema_fetched"] and rejects a premature run_sql_query
call outright (no DB round-trip, no psycopg2 error text) with a clear
instructive message instead, rather than relying on the LLM to follow
the prompt's instruction on its own.
"""

from decimal import Decimal
from typing import Callable

from sqlalchemy.engine import Engine

from app.agent.router import route_question
from app.agent.state import GraphState, ToolCallRecord
from app.agent.tool_definitions import REGISTRY, TOOLS
from app.database.connection import get_readonly_engine
from app.services.llm_client import create_message

# Phase 5: these two tools run LLM-generated/schema-introspection code
# against the database and should use the restricted read-only engine,
# never the main one -- every other tool in REGISTRY keeps using the
# main engine passed into make_tool_step_node.
SQL_FALLBACK_TOOL_NAMES = {"get_schema", "run_sql_query"}


def _sanitize_result(value):
    """Recursively converts Decimal values to float so downstream
    prompts (decide_node, answer_node) show plain numbers instead of
    Python's Decimal('...') repr. Applied once, right after a tool call
    succeeds, so every consumer of state["tool_calls"] sees already-
    clean data -- no changes needed in decide_node or answer_node
    themselves."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _sanitize_result(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_result(v) for v in value]
    return value


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
    is the fix.

    The comparison_note below addresses a separate, later-discovered
    gap: a period-over-period comparison question (e.g. "why did
    revenue decline from May to June?") was being answered with ONE
    tool call using a combined date range spanning both periods --
    which can only show a total, not a comparison. This note is always
    included on loop iterations (not gated behind is_compound) since a
    comparison isn't a compound-topic question, it's a single-topic
    question needing evidence that separates the two periods.

    Phase 5 update: the note now also offers run_sql_query as an
    alternative to a second per-period call, since a single SQL query
    can compute both periods' totals and the difference directly --
    closing the comparison-question gap without needing answer_node to
    aggregate anything itself."""
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
    comparison_note = (
        "If the original question compares two specific time periods "
        "(e.g. month-over-month, before/after a date), check whether any "
        "tool call above already gives you a separate, identifiable figure "
        "for EACH period, or only a single combined figure spanning both "
        "periods together. A combined range cannot show the difference "
        "between the periods -- it only shows their total. If that's the "
        "case here, you have two options: (a) call the same breakdown "
        "tool again, restricted to just ONE of the specific periods, so "
        "each period can be retrieved separately and then compared, or "
        "(b) if no existing tool naturally does this, use run_sql_query "
        "to write a single query that computes both periods' totals and "
        "the difference between them directly.\n\n"
    )
    return (
        f"{prefix}"
        f"{comparison_note}"
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

        # Phase 5: deterministic enforcement -- two separate retests saw
        # the LLM attempt run_sql_query with guessed (wrong) table names
        # before ever calling get_schema, despite the prompt telling it
        # to call get_schema first. Reject here, with no DB round-trip,
        # rather than relying on the LLM to remember the instruction.
        if routing.tool_name == "run_sql_query" and not state.get("schema_fetched"):
            state["tool_calls"].append(
                ToolCallRecord(
                    tool_name=routing.tool_name,
                    tool_input=routing.tool_input,
                    result={
                        "error": (
                            "You must call get_schema first to see the actual "
                            "table and column names before writing SQL. Call "
                            "get_schema now, then retry run_sql_query."
                        )
                    },
                )
            )
            return state

        # Phase 5: get_schema/run_sql_query execute against the
        # restricted read-only role, never the main engine. Every other
        # tool keeps using the main engine, unchanged from Phase 2/4.
        tool_engine = (
            get_readonly_engine() if routing.tool_name in SQL_FALLBACK_TOOL_NAMES else engine
        )

        try:
            result = func(tool_engine, **routing.tool_input)
        except Exception as exc:  # noqa: BLE001 -- surface any tool failure to caller
            state["declined"] = True
            state["decline_reason"] = f"Tool execution failed: {exc}"
            return state

        # Phase 5 retest fix: strip Decimal (psycopg2's NUMERIC type)
        # down to float before this ever reaches a prompt -- otherwise
        # decide_node/answer_node see literal "Decimal('...')" text.
        result = _sanitize_result(result)

        if routing.tool_name == "get_schema":
            state["schema_fetched"] = True

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

Check 1: does the original question have more than one distinct part \
(e.g. "X, and separately Y")? If so, has a tool result already addressed \
EACH part, or only some of them?

Check 2: does the original question ask to COMPARE two specific time \
periods (e.g. month-over-month, before/after a date)? A result is \
sufficient only if it contains a separate, identifiable figure for EACH \
period being compared -- this can come from two separate tool calls each \
scoped to one period, OR from a single run_sql_query result that itself \
returns each period's figure as its own distinct value (e.g. separate \
rows or columns per period, or a total plus a difference). A tool result \
computed over ONE combined date range spanning both periods, with no \
per-period breakdown anywhere in it, is NOT sufficient -- it only shows \
their total, not the difference.

Respond with exactly one word: YES only if every distinct part of the \
original question is covered, AND any implied period comparison has \
per-period figures available (not just one merged-range total). Respond \
NO otherwise."""


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