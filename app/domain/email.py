"""
Email drafts and reusable templates.

Extracted verbatim from `ShortTermMemory`; behaviour is unchanged.
"""
from typing import Any, Dict, List, Optional
import logging
import uuid

from sqlalchemy import and_, desc, select

from app.db.session import async_session_maker
from app.domain.models import EmailDraft, EmailTemplate

logger = logging.getLogger(__name__)


class EmailRepository:
    """Draft and template persistence. Drafts are never sent from here."""

    def __init__(self):
        self.async_session_maker = async_session_maker

    # ── Drafts ──────────────────────────────────────────────────────────

    async def save_draft(
        self,
        user_id: str,
        subject: str,
        body: str,
        recipient_name: Optional[str] = None,
        tone: str = "professional",
        greeting: Optional[str] = None,
        closing: Optional[str] = None,
        signature: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Persist an email draft. Returns draft ID."""
        async with self.async_session_maker() as session:
            draft = EmailDraft(
                user_id=user_id,
                subject=subject,
                body=body,
                recipient_name=recipient_name,
                tone=tone,
                greeting=greeting,
                closing=closing,
                signature=signature,
                status="draft",
                context=context or {},
            )
            session.add(draft)
            await session.commit()
            return str(draft.id)

    async def get_drafts(
        self, user_id: str, limit: int = 10, status: str = "draft"
    ) -> List[Dict[str, Any]]:
        """Retrieve saved email drafts, newest first."""
        async with self.async_session_maker() as session:
            query = (
                select(EmailDraft)
                .where(and_(EmailDraft.user_id == user_id, EmailDraft.status == status))
                .order_by(desc(EmailDraft.created_at))
                .limit(limit)
            )
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "subject": r.subject,
                    "recipient_name": r.recipient_name,
                    "tone": r.tone,
                    "greeting": r.greeting,
                    "body": r.body,
                    "closing": r.closing,
                    "signature": r.signature,
                    "status": r.status,
                    "context": r.context or {},
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def mark_draft_sent(
        self,
        draft_id: str,
        user_id: str,
        sent_to: str,
    ) -> bool:
        """Mark a draft as sent and record the recipient address."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(EmailDraft).where(
                    and_(
                        EmailDraft.id == uuid.UUID(draft_id),
                        EmailDraft.user_id == user_id,
                    )
                )
            )
            draft = result.scalar_one_or_none()
            if not draft:
                return False
            draft.status = "sent"
            draft.context = {**(draft.context or {}), "sent_to": sent_to}
            await session.commit()
            return True

    # ── Templates ───────────────────────────────────────────────────────

    async def save_template(
        self,
        user_id: str,
        name: str,
        subject_template: str,
        body_template: str,
        tone: str = "professional",
        placeholders: Optional[List[str]] = None,
    ) -> str:
        """Save a reusable email template. Returns template ID."""
        async with self.async_session_maker() as session:
            tmpl = EmailTemplate(
                user_id=user_id,
                name=name,
                subject_template=subject_template,
                body_template=body_template,
                tone=tone,
                placeholders=placeholders or [],
            )
            session.add(tmpl)
            await session.commit()
            return str(tmpl.id)

    async def get_templates(
        self, user_id: str, name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve email templates for a user."""
        async with self.async_session_maker() as session:
            query = select(EmailTemplate).where(EmailTemplate.user_id == user_id)
            if name:
                query = query.where(EmailTemplate.name == name)
            query = query.order_by(desc(EmailTemplate.created_at))
            result = await session.execute(query)
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "name": r.name,
                    "tone": r.tone,
                    "subject_template": r.subject_template,
                    "body_template": r.body_template,
                    "placeholders": r.placeholders or [],
                    "created_at": r.created_at.isoformat(),
                }
                for r in rows
            ]


# Singleton instance
email_repository = EmailRepository()
