"""
Email drafting tool powered by RAG on ChromaDB.
Generates personalized email drafts and never sends them.
"""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List
import logging

from app.memory.memory_manager import memory_manager
from app.services.groq_service import groq_service
from app.services.langsmith_service import traceable


logger = logging.getLogger(__name__)


class EmailDraftTool:
    """Create personalized email drafts using RAG context."""

    @traceable(name="tool_email_draft", run_type="tool", tags=["tool", "email"])
    async def draft_email(
        self,
        user_id: str,
        query: str,
        tone: str = "professional",
        recipient_name: str = "",
    ) -> Dict[str, Any]:
        rag_context = await self._retrieve_rag_context(user_id=user_id, query=query)
        if not rag_context:
            rag_context = "No relevant memory context found."

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an email drafting assistant. Generate only JSON with keys: "
                    "subject, greeting, body, closing, signature. Keep content concise and personalized."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Draft request: {query}\n"
                    f"Tone: {tone}\n"
                    f"Recipient Name: {recipient_name or 'Not provided'}\n"
                    f"RAG Context:\n{rag_context}"
                ),
            },
        ]

        parsed = {
            "subject": "",
            "greeting": "",
            "body": "",
            "closing": "",
            "signature": "",
        }
        raw_output = ""

        try:
            response = await groq_service.chat_completion(
                messages=messages,
                temperature=0.5,
                max_tokens=700,
            )
            raw_output = response.get("content", "")
            parsed = self._safe_parse_json(raw_output, fallback=parsed)
        except Exception as e:
            logger.error(
                "Email draft generation failed for user=%s: %s", user_id, e, exc_info=True
            )

        return {
            "tool": "email_draft",
            "success": True,
            "user_id": user_id,
            "not_sent": True,
            "query": query,
            "tone": tone,
            "recipient_name": recipient_name,
            "rag_context": rag_context,
            "draft": parsed,
            "raw_model_output": raw_output,
        }

    async def _retrieve_rag_context(self, user_id: str, query: str) -> str:
        """Retrieve and format memory context as plain text for prompting."""
        try:
            logger.debug("Email RAG retrieval started for user_id=%s", user_id)

            # Profile facts are fetched alongside vector context: they hold the
            # user's name and explicit preferences (e.g. preferred_tone), which
            # are exactly what a personalised draft needs and were previously
            # hardcoded out of this prompt.
            long_term, profile_facts = await asyncio.gather(
                memory_manager.search_long_term(user_id=user_id, query=query, limit=5),
                memory_manager.get_profile_facts(user_id=user_id),
                return_exceptions=True,
            )

            if isinstance(long_term, BaseException):
                logger.warning("Email RAG vector lookup failed for user=%s: %s", user_id, long_term)
                long_term = {}
            if isinstance(profile_facts, BaseException):
                logger.warning("Email RAG profile lookup failed for user=%s: %s", user_id, profile_facts)
                profile_facts = []

            context_text = memory_manager.format_context_for_prompt(
                {
                    "chat_history": [],
                    "preferences": [],
                    "profile_facts": profile_facts,
                    "long_term": long_term,
                }
            ).strip()

            if not context_text:
                logger.debug("Email RAG retrieval completed with empty context for user_id=%s", user_id)
                return ""

            logger.debug(
                "Email RAG retrieval completed for user_id=%s with %s chars",
                user_id,
                len(context_text),
            )
            return context_text
        except Exception as e:
            logger.exception("Email RAG retrieval failed for user_id=%s: %s", user_id, str(e))
            return ""

    def _safe_parse_json(self, text: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
        import json

        if not text:
            return fallback

        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.split("```json", 1)[1].rsplit("```", 1)[0].strip()
        elif cleaned.startswith("```"):
            cleaned = cleaned.split("```", 1)[1].rsplit("```", 1)[0].strip()

        try:
            obj = json.loads(cleaned)
            if not isinstance(obj, dict):
                return fallback
            return {
                "subject": str(obj.get("subject", "")),
                "greeting": str(obj.get("greeting", "")),
                "body": str(obj.get("body", "")),
                "closing": str(obj.get("closing", "")),
                "signature": str(obj.get("signature", "")),
            }
        except Exception:
            return fallback


email_draft_tool = EmailDraftTool()
