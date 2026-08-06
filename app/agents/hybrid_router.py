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

from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)

ROUTE_CONVERSATIONAL = "conversational"
ROUTE_TOOL = "tool_required"

# Small, fast, currently-supported Groq model. Classification does not need a
# 70B model, and a large one here would just add latency before the reply.
_ROUTER_MODEL = "llama-3.1-8b-instant"

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


def classify_heuristically(transcript: str) -> Optional[str]:
    """Route by keyword, or return None when the transcript is ambiguous."""
    text = (transcript or "").strip()
    if not text:
        return ROUTE_CONVERSATIONAL

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
    fast = classify_heuristically(transcript)
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
                max_tokens=16,
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
