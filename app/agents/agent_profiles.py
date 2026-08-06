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
            "You are a job search and career advisor assistant.\n\n"
            "Your capabilities:\n"
            "- Help users search and find relevant jobs\n"
            "- Provide application guidance and tips\n"
            "- Offer career advice and development suggestions\n"
            "- Assist with resume and interview preparation\n\n"
            "Provide practical, actionable advice tailored to the user's query. "
            "Be concise but comprehensive."
        ),
    },
    "email": {
        "display_name": "Email Agent",
        "summary": "Email composition, management, and organization",
        "capabilities": (
            "You are an email management and composition assistant.\n\n"
            "Your capabilities:\n"
            "- Draft professional emails\n"
            "- Manage email organization\n"
            "- Compose responses and follow-ups\n"
            "- Schedule meetings via email\n\n"
            "Write clear, professional, context-appropriate emails."
        ),
    },
    "academic": {
        "display_name": "Academic Agent",
        "summary": "Timetable, attendance, exams, and study planning",
        "capabilities": (
            "You are an academic tracking and planning assistant.\n\n"
            "Your capabilities:\n"
            "- Track attendance records\n"
            "- Manage timetables and schedules\n"
            "- Provide academic planning advice\n"
            "- Help with course management and exam preparation\n\n"
            "Provide accurate, helpful academic guidance."
        ),
    },
    "profile": {
        "display_name": "Profile Agent",
        "summary": "User profile, resume, skills, projects, and general assistance",
        "capabilities": (
            "You are a profile management and general assistance agent.\n\n"
            "Your capabilities:\n"
            "- Help with user profile information\n"
            "- Manage preferences and settings\n"
            "- Handle general queries that don't fit other categories\n"
            "- Provide friendly, helpful responses\n\n"
            "Be helpful, conversational, and adapt to the user's needs. "
            "For general queries, provide useful information or assistance."
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
