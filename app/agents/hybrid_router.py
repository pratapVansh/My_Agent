"""
Hybrid Agent Router — Phase 5.5
Performs ultra-fast intent classification to decide between
streaming conversation and full agent tool execution.
"""
import json
import logging
from typing import List, Dict

from app.services.groq_service import groq_service

logger = logging.getLogger(__name__)

async def determine_route(transcript: str, history: List[Dict[str, str]] = None) -> str:
    """
    Classify the transcript into 'conversational' or 'tool_required'.
    Returns the string route.
    """
    system_prompt = """You are a fast intent router for a voice assistant.
Your job is to determine if the user's input requires executing a tool/agent (e.g., searching jobs, sending emails, updating profile, fetching academic info) OR if it is a normal conversational chat.

Rules:
1. If the user asks to perform an action, search for something, or retrieve data, route to "tool_required".
2. If the user is just chatting, asking general knowledge questions, or responding to small talk, route to "conversational".

Output ONLY a JSON object exactly like this:
{"route": "conversational"} OR {"route": "tool_required"}
"""

    messages = [{"role": "system", "content": system_prompt}]
    if history:
        # Include just the last 2 messages for context
        for msg in history[-2:]:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
            
    messages.append({"role": "user", "content": transcript})
    
    try:
        response = await groq_service.chat_completion(
            messages=messages,
            model="llama3-8b-8192",  # Fast model for routing
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        content = response.get("content", "{}")
        data = json.loads(content)
        route = data.get("route", "conversational").lower().strip()
        if route not in ["conversational", "tool_required"]:
            return "conversational"
        return route
    except Exception as e:
        logger.warning("Router failed, defaulting to conversational: %s", e)
        return "conversational"
