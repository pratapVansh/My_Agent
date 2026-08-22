"""
Voice turn router — decides between the streaming conversational path and the
full tool-calling LangGraph workflow.

This runs *before* anything else on every spoken turn, so its cost is pure
added latency the speaker experiences as the assistant being slow to answer.
It is therefore heuristic-first: an unambiguous transcript is classified in
microseconds, and the LLM is consulted only for genuinely ambiguous input,
under a tight timeout, defaulting to the streaming path.

The previous version always made an LLM call, and made it against
``llama3-8b-8192`` — a model Groq has decommissioned. Every voice turn paid for
a request that could only fail, then fell back to "conversational" anyway.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Dict, List, Optional

from app.agents import query_intent
from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)

ROUTE_CONVERSATIONAL = "conversational"
ROUTE_TOOL = "tool_required"

# Small, fast, currently-supported Groq model. Classification does not need the
# large model, and using it here would just add latency before the reply.
#
# `llama-3.1-8b-instant` used to be this and was decommissioned by Groq — every
# call 404'd, and because the failure path defaults to `conversational`, an
# ambiguous *spoken* turn silently lost access to the tools. That is the worst
# available failure for this component: it does not degrade the answer, it
# removes the assistant's ability to look anything up, and it does so quietly.
# Measured before landing: 3/3 correct classifications, mean 574 ms, inside the
# 1.2 s budget below.
_ROUTER_MODEL = "openai/gpt-oss-20b"

# Generous on purpose, and not a latency cost — generation stops at the closing
# brace either way. `gpt-oss` reasons before answering and bills that reasoning
# from this same budget, so a tight cap (this was 16) is spent entirely on
# reasoning and returns empty content, which Groq then rejects as invalid JSON.
_ROUTER_MAX_TOKENS = 512

# Hard ceiling on the router. Past this the streaming path is simply better
# than making the user wait longer for a routing decision.
_ROUTER_TIMEOUT_SECONDS = 1.2

# Phrases that can only mean "go do something with a tool".
_TOOL_PATTERNS = re.compile(
    r"\b("
    r"send (an? )?(e-?mail|message)|draft (an? )?e-?mail|e-?mail (him|her|them|me)"
    r"|find( me)? (a )?job|search (for )?jobs?|job (search|opening|listing|posting)"
    r"|apply (for|to)"
    r"|my (attendance|timetable|time table|schedule|classes|marks|grades)"
    r"|(what|which) class(es)?|next class|class today|attendance"
    r"|remember (that|this)|forget (that|about)|update my (profile|details)"
    r"|(add|save) (this )?to my profile"
    r"|look ?up|search (the )?(web|internet)"
    r")\b",
    re.IGNORECASE,
)

# Phrases that are plainly chat and never need a tool.
_CHAT_PATTERNS = re.compile(
    r"^\s*("
    r"h(i|ey|ello)|yo|good (morning|afternoon|evening)|thanks?|thank you|ok(ay)?|cool"
    r"|yes|yeah|yep|no|nope|nah|sure|got it|never ?mind|stop|wait"
    r"|how are you|what'?s up|who are you|what can you do|tell me about yourself"
    r"|repeat( that)?|say (that )?again|sorry"
    r")\b",
    re.IGNORECASE,
)

_ROUTER_SYSTEM_PROMPT = (
    "You route a voice assistant's turns. Answer with JSON only.\n"
    'Reply {"route":"tool_required"} when the user wants an action performed or '
    "private/live data fetched: sending email, searching jobs, reading their "
    "timetable, attendance, marks, or profile, or searching the web.\n"
    'Reply {"route":"conversational"} for chat, opinions, explanations, general '
    "knowledge, and follow-up questions about what was just said."
)


def classify_heuristically(
    transcript: str, history: Optional[List[Dict[str, str]]] = None
) -> Optional[str]:
    """
    Route by category, then by keyword, or return None when ambiguous.

    `history` is passed through to the classifier so this asks the same
    question the rest of the system asks. Without it, a turn was classified
    here with no context and again downstream with context, and the two
    disagreed: "what is on my resume" is DOCUMENT_RESUME cold and was
    CONVERSATION_FOLLOWUP warm, so the router sent it to the tools while the
    streaming path answered it from a prompt. Same utterance, same turn,
    opposite handling — decided by which component happened to be asking.
    """
    text = (transcript or "").strip()
    if not text:
        return ROUTE_CONVERSATIONAL

    # ── The deterministic answer, asked of the one classifier ────────────────
    # Ahead of every keyword, and not itself a keyword rule: this asks
    # `query_intent` — the same module the graph and the streaming path route
    # with — whether this category can be answered without tools at all.
    #
    # A keyword list cannot carry that guarantee. Each new phrasing of "am I a
    # good fit" is a new pattern, the list is never finished, and every gap in
    # it is a spoken question answered by a model inventing a qualification. So
    # the wording stops being the mechanism: the category is.
    #
    # `escalation_reason` is asked without a grounding verdict on purpose. This
    # runs before any retrieval, so the conditional rule — stream a personal
    # question only when its sources actually arrived — is not answerable here
    # and would have to be guessed. Only the unconditional rule is decided at
    # this point; the conditional one is decided by `run_streaming_workflow`
    # once it knows, which is the same source of truth read at the moment it
    # has an answer.
    #
    # This is a fast path, not the enforcement. `run_streaming_workflow`
    # re-derives the same verdict and escalates on its own, so a turn that
    # reaches it by any other route — the `/agents/stream` socket, an LLM
    # fallback that guessed wrong, this function timing out — still cannot be
    # answered without the tools its category requires.
    #
    # The transcript is passed as well as its category, which brings the third
    # rule in: a turn asking for a consequential capability — a send, a
    # deletion — takes the tool path even when the classifier put it in some
    # other category, because a tool-free path cannot perform one and will say
    # it did. A reply to a pending action can land here too ("send it",
    # "cancel"), and routing it to the tools is harmless: `decide_route` hands
    # it to the confirmation gateway on arrival, before any specialist runs.
    if query_intent.escalation_reason(
        query_intent.classify(
            text, has_context=bool(history), history=history,
        ).category,
        text=text,
    ):
        return ROUTE_TOOL

    if _TOOL_PATTERNS.search(text):
        return ROUTE_TOOL
    if _CHAT_PATTERNS.match(text):
        return ROUTE_CONVERSATIONAL
    # Very short utterances with no tool keyword are acknowledgements or
    # follow-ups; a tool call is never the right response to three words.
    if len(text.split()) <= 3:
        return ROUTE_CONVERSATIONAL
    return None


async def determine_route(
    transcript: str, history: Optional[List[Dict[str, str]]] = None
) -> str:
    """Classify a spoken turn as ``conversational`` or ``tool_required``."""
    fast = classify_heuristically(transcript, history)
    if fast is not None:
        logger.debug("Route (heuristic): %s", fast)
        return fast

    messages: List[Dict[str, str]] = [{"role": "system", "content": _ROUTER_SYSTEM_PROMPT}]
    for msg in (history or [])[-2:]:
        if isinstance(msg, dict) and msg.get("role") in ("user", "assistant") and msg.get("content"):
            messages.append({"role": msg["role"], "content": str(msg["content"])[:300]})
    messages.append({"role": "user", "content": transcript})

    try:
        response = await asyncio.wait_for(
            groq_service.chat_completion(
                messages=messages,
                model=_ROUTER_MODEL,
                temperature=0.0,
                max_tokens=_ROUTER_MAX_TOKENS,
                response_format={"type": "json_object"},
            ),
            timeout=_ROUTER_TIMEOUT_SECONDS,
        )
        route = json.loads(response.get("content") or "{}").get("route", "")
        route = str(route).lower().strip()
        if route in (ROUTE_CONVERSATIONAL, ROUTE_TOOL):
            logger.debug("Route (llm): %s", route)
            return route
        return ROUTE_CONVERSATIONAL
    except asyncio.TimeoutError:
        logger.info("Router timed out after %.1fs; streaming instead", _ROUTER_TIMEOUT_SECONDS)
        return ROUTE_CONVERSATIONAL
    except Exception as exc:
        logger.warning("Router failed, defaulting to conversational: %s", exc)
        return ROUTE_CONVERSATIONAL
