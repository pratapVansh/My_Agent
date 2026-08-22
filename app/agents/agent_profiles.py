"""
Single source of truth for each specialist agent's identity.

Two execution paths address the same agents: the tool-calling LangGraph workflow
and the low-latency streaming path used for voice. Each previously carried its
own hand-written description of every agent, so the two drifted and any change
to an agent's remit had to be made twice.

Capability text lives here once. The tool-calling agents append their own
tool-usage protocol on top; the streaming path uses the base text alone, since
it deliberately runs without tools.
"""
from __future__ import annotations

from typing import Dict


AGENT_PROFILES: Dict[str, Dict[str, str]] = {
    "job": {
        "display_name": "Job Agent",
        "summary": "Job search, applications, and career guidance",
        "capabilities": (
            "You handle job search, applications, and career guidance: finding "
            "relevant openings, application and interview preparation, and "
            "advice grounded in the user's own résumé and skills."
        ),
    },
    "email": {
        "display_name": "Email Agent",
        "summary": "Email composition, management, and organization",
        "capabilities": (
            "You handle email: drafting, replies and follow-ups, saved drafts "
            "and templates. Sending always requires the user's explicit "
            "approval and is never done silently."
        ),
    },
    "academic": {
        "display_name": "Academic Agent",
        "summary": "Timetable, attendance, exams, and study planning",
        "capabilities": (
            "You handle the user's academic life: their timetable, attendance, "
            "exams, and study planning. Schedule facts come from their stored "
            "timetable, never from memory of a previous conversation."
        ),
    },
    "profile": {
        "display_name": "Profile Agent",
        "summary": "User profile, resume, skills, projects, and general assistance",
        "capabilities": (
            "You handle the user's own profile — résumé, skills, projects, "
            "education, saved preferences — and anything that does not belong "
            "to another specialist."
        ),
    },
}


def get_capabilities(agent_name: str, default: str = "") -> str:
    """Return the shared capability description for an agent."""
    profile = AGENT_PROFILES.get(agent_name)
    return profile["capabilities"] if profile else default


def get_summary(agent_name: str, default: str = "") -> str:
    """Return the one-line summary used for agent registries and docs."""
    profile = AGENT_PROFILES.get(agent_name)
    return profile["summary"] if profile else default
