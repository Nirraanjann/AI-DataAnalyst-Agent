"""
app/agent/state.py

Shared graph state for the Phase 4 LangGraph agent.

Design notes (per PHASE4_HANDOFF.md):
- `tool_calls` is a growing list, not a single slot, so the graph can
  support multi-tool-call questions like q10 ("why did revenue
  decline"), where the agent may need to call get_monthly_revenue,
  then a category/region breakdown tool, etc.
- `needs_clarification` / `clarification_question` exist so the
  Planning node has somewhere to put its output instead of silently
  picking one interpretation (Phase 3 KNOWN PROBLEM #1).
- `step_count` + `max_steps` enforce the termination-safety check
  (NEXT TASK #10) so the tool-call loop cannot run indefinitely.
- No raw arithmetic field is exposed here on purpose: derived
  calculations belong in a dedicated node/tool output, not floated
  as a free-form value the phrasing step can silently produce
  (Phase 3 KNOWN PROBLEM #2).
"""

from typing import Any, Optional, TypedDict


class ToolCallRecord(TypedDict):
    """One executed tool call and its result, in order."""
    tool_name: str
    tool_input: dict[str, Any]
    result: Any


class GraphState(TypedDict, total=False):
    # Input
    question: str

    # Planning / disambiguation node output
    plan: Optional[str]                       # short plan text, if any
    needs_clarification: bool
    clarification_question: Optional[str]     # set only if needs_clarification
    is_compound: bool                         # true only if the question has multiple
                                               # distinct topics (e.g. "X, and separately Y") --
                                               # gates the multi-topic routing framing so
                                               # ordinary single-topic questions never see it

    # Tool selection + execution loop
    tool_calls: list[ToolCallRecord]          # accumulates across loop iterations
    step_count: int
    max_steps: int                            # e.g. 4; hard cap on loop iterations

    # Loop control
    has_enough_evidence: bool                 # set by the "decide" node

    # Output
    final_answer: Optional[str]
    declined: bool                            # true if out-of-scope, no tool fit
    decline_reason: Optional[str]


def initial_state(question: str, max_steps: int = 4) -> GraphState:
    """Factory for a fresh GraphState at the start of a run."""
    return GraphState(
        question=question,
        plan=None,
        needs_clarification=False,
        clarification_question=None,
        is_compound=False,
        tool_calls=[],
        step_count=0,
        max_steps=max_steps,
        has_enough_evidence=False,
        final_answer=None,
        declined=False,
        decline_reason=None,
    )