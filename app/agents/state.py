"""
State schema for LangGraph workflow.
Defines the structure of data flowing through the multi-agent system.
"""
from typing import TypedDict, Optional, List, Dict, Any, FrozenSet
import uuid


class TaskEnvelope(TypedDict):
    """
    Structured contract between agents.
    Every specialist must populate and return this in task_result.
    Required for agent coordination and the reflect loop.
    """
    task_id: str          # Unique ID for this task execution
    goal: str             # What the agent was asked to accomplish
    inputs: dict          # Inputs received (user_input, intent, constraints)
    constraints: dict     # Budget, format, tone, max_results, etc.
    result: dict          # Raw result content from the agent
    confidence: float     # 0.0–1.0 how confident the agent is in the result
    evidence: list        # Sources/tools used as evidence
    status: str           # "success" | "partial" | "failed"
    next_actions: list    # What could be done next (for multi-step planning)
    agent: str            # Which agent produced this


def make_envelope(
    agent: str,
    goal: str,
    inputs: dict,
    result_content: str,
    status: str,
    confidence: float,
    tools_used: list = None,
    constraints: dict = None,
    next_actions: list = None,
) -> TaskEnvelope:
    """Helper to build a TaskEnvelope with sensible defaults."""
    return TaskEnvelope(
        task_id=uuid.uuid4().hex[:12],
        goal=goal,
        inputs=inputs,
        constraints=constraints or {},
        result={"content": result_content},
        confidence=confidence,
        evidence=tools_used or [],
        status=status,
        next_actions=next_actions or [],
        agent=agent,
    )


class AgentState(TypedDict):
    """State passed between agents in the workflow."""

    # Input
    user_input: str
    conversation_history: Optional[List[Dict[str, str]]]
    user_id: Optional[str]
    session_id: Optional[str]
    output_mode: Optional[str]
    # Whose long-term memory to *read*. Defaults to the caller. A guest
    # reads the owner's memory while still writing under its own id —
    # conflating the two would let recruiter chatter pollute the owner.
    memory_owner_id: Optional[str]
    # Visibility filter for memory retrieval. None = the caller's own
    # memory, every visibility. A guest carries [PUBLIC], which is what
    # lets the recruiter view read the owner's public records.
    memory_visibilities: Optional[list]

    # Set by a caller that has already stored this utterance, so the memory
    # node retrieves but does not write. The streaming workflow sets it when it
    # escalates a turn after its own init has run; without it the same sentence
    # lands in the conversation thread twice.
    skip_user_ingest: Optional[bool]

    # Initialization a caller already performed and is handing over, so it is
    # not paid for twice. The streaming path retrieves memory and runs the
    # planner, then escalates a turn to the tool workflow — which used to run
    # both again from scratch. `parallel_init_node` adopts these instead.
    # Keys: memory_context, memory_prompt, query_category, selected_agent,
    # detected_intent, planner_confidence, execution_plan.
    prefetched: Optional[Dict[str, Any]]

    # Set when the model provider rate-limited this turn. Read by `reflect_node`
    # so a closed rate-limit window is never answered by re-running the whole
    # specialist — the single most expensive response available to it.
    rate_limited: Optional[bool]

    # Authorization: capabilities of the authenticated caller, propagated from
    # the verified JWT. Specialist agents filter their tool registries against
    # this, so a restricted caller cannot reach privileged tools through
    # conversation. None means "unrestricted" (internal/CLI invocation).
    scopes: Optional[FrozenSet[str]]

    # Memory context
    memory_context: Optional[Dict[str, Any]]
    memory_prompt: Optional[str]

    # Planner outputs — now includes confidence gating
    detected_intent: Optional[str]
    selected_agent: Optional[str]
    planner_confidence: Optional[float]       # 0.0–1.0
    needs_clarification: Optional[bool]       # True → ask user before routing
    clarification_question: Optional[str]     # What to ask the user
    clarification_reason: Optional[str]       # Why it was asked, or suppressed

    # Source-aware routing (app.memory.sources). Decided deterministically
    # before the planner's opinion is consulted, because the planner scores
    # intent from query text alone and cannot tell "my current CPI" (a stored
    # fact) from "the current date" (the clock).
    query_category: Optional[str]             # QueryCategory value
    memory_sources: Optional[List[str]]       # MemorySource values, in precedence order
    profile_intent: Optional[str]             # Legacy profile label, drives tool choice
    followup_subject: Optional[str]           # Entity a follow-up refers back to

    # The graph edge this turn takes, settled by a *node* so the decision and
    # everything decided alongside it survives. A LangGraph conditional-edge
    # function returns a route and its state writes are discarded, which is why
    # this is recorded rather than recomputed at the edge.
    route: Optional[str]

    # The tools this category may not answer without — "at least one of these".
    # Written at the routing edge, rendered into the specialist's directive, and
    # checked after the reasoning loop by `app.agents.grounding`. An empty list
    # means the category needs no particular lookup.
    required_tools: Optional[List[str]]

    # Which academic capability a SCHEDULE_TEMPORAL turn wants: timetable,
    # attendance, exams or plans. One category covers four tables, and the
    # required tools differ per table.
    schedule_intent: Optional[str]

    # The provenance of the answer just given — what produced it, so "how did
    # you know?" reads a record instead of asking a model to reconstruct its
    # own reasoning, which it cannot do and will invent. It lives in
    # `app.memory.provenance` keyed by conversation rather than on this state:
    # it must outlive the turn that wrote it in order to answer a question
    # asked on the *next* one, and state does not survive that boundary.

    # The observed verdict on what this turn's tools produced — ANSWERABLE,
    # NO_DATA, TOOL_ERROR, PARTIALLY_ANSWERABLE. Derived from tool results
    # rather than from the model's account of them, because "the lookup failed"
    # and "there is nothing on file" are opposite claims that arrive looking
    # identical. See app.memory.answerability.
    answerability: Optional[str]

    # Whether this turn's answer actually rests on a lookup — satisfied,
    # no_data, failed, skipped, or not_required. Written by the reasoning loop
    # (see app.agents.grounding) and read by `reflect_node`, which retries a
    # `skipped` turn: the tool was there and the model did not call it, which
    # is the most recoverable failure the system has.
    #
    # Declared here because LangGraph only propagates keys the state schema
    # names. An undeclared key written by a node is silently dropped before the
    # next one runs, so leaving this out does not fail loudly — it just makes
    # the retry never fire.
    grounding: Optional[str]

    # Task agent outputs — now uses TaskEnvelope
    task_result: Optional[TaskEnvelope]
    agent_reasoning: Optional[str]

    # Actions this turn prepared and did not perform — the PENDING_CONFIRMATION
    # results the gateway returned in place of a consequential tool call.
    # Written by the reasoning loop itself rather than by each agent, so a turn
    # that held something cannot reach the response layer looking like a turn
    # that completed it. Empty on every ordinary turn.
    pending_actions: Optional[List[Any]]

    # Multi-step plan execution (Fix 1: True Agentic Planning)
    execution_plan: Optional[List[Dict[str, Any]]]  # Ordered steps: [{step, agent, goal}, ...]
    current_step_index: Optional[int]               # Index of the step currently executing
    step_results: Optional[Dict[str, Any]]          # Completed step results: {"1": summary, ...}
    inter_step_context: Optional[str]               # Formatted prior-step results for next agent

    # Reflect loop control
    iteration_count: Optional[int]            # How many specialist attempts so far
    reflect_outcome: Optional[str]            # "retry" | "next_step" | "done"

    # Reflect learning (Fix 2: Reflect Loop Learns from Failures)
    reflect_failure_context: Optional[str]    # Why the previous attempt failed (injected on retry)

    # Response agent output
    display_text: Optional[str]
    speech_text: Optional[str]

    # Metadata
    current_agent: Optional[str]
    execution_path: Optional[List[str]]
    request_id: Optional[str]          # Unique ID for end-to-end tracing
    error: Optional[str]
