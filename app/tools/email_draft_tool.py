"""
Email drafting tool powered by RAG on ChromaDB.
Generates personalized email drafts and never sends them.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.memory.memory_manager import memory_manager
from app.services.groq_service import groq_service
from app.services.langsmith_service import traceable


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
        except Exception:
            pass

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

    async def _retrieve_rag_context(self, user_id: str, query: str) -> Dict[str, List[Dict[str, Any]]]:
        long_term = memory_manager.long_term.search_all(
            user_id=user_id,
            query=query,
            limit=5,
        )
        return long_term

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
