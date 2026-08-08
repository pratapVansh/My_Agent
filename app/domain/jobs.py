"""
Saved job listings.

Extracted verbatim from `ShortTermMemory`; behaviour is unchanged.
"""
from typing import Any, Dict, List, Optional
import logging

from sqlalchemy import and_, desc, select
from sqlalchemy.exc import IntegrityError

from app.db.session import async_session_maker
from app.domain.models import JobBookmark

logger = logging.getLogger(__name__)


class JobsRepository:
    """Job bookmark persistence."""

    def __init__(self):
        self.async_session_maker = async_session_maker

    async def save_bookmark(
        self,
        user_id: str,
        title: str,
        url: str,
        company: Optional[str] = None,
        snippet: Optional[str] = None,
        rank_score: Optional[float] = None,
        search_query: Optional[str] = None,
        skills_matched: Optional[List[str]] = None,
    ) -> str:
        """
        Save a job bookmark. Returns "already_saved" if the URL is a duplicate.

        The pre-check is only a fast path. Two concurrent calls for the same URL
        can both pass it, so the authoritative guard is the unique constraint on
        (user_id, url) and the IntegrityError it raises — that converts a race
        into the same "already_saved" outcome instead of a duplicate row.
        """
        if await self.is_bookmarked(user_id, url):
            return "already_saved"

        async with self.async_session_maker() as session:
            bookmark = JobBookmark(
                user_id=user_id,
                title=title,
                url=url,
                company=company,
                snippet=snippet,
                rank_score=rank_score,
                search_query=search_query,
                skills_matched=skills_matched or [],
            )
            session.add(bookmark)
            try:
                await session.commit()
            except IntegrityError:
                await session.rollback()
                logger.debug(
                    "Concurrent bookmark insert for user=%s url=%s resolved as duplicate",
                    user_id, url,
                )
                return "already_saved"
            return str(bookmark.id)

    async def is_bookmarked(self, user_id: str, url: str) -> bool:
        """Return True if this URL is already bookmarked by the user."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(JobBookmark).where(
                    and_(JobBookmark.user_id == user_id, JobBookmark.url == url)
                )
            )
            return result.scalars().first() is not None

    async def get_bookmarks(
        self, user_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve saved job bookmarks for a user, newest first."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(JobBookmark)
                .where(JobBookmark.user_id == user_id)
                .order_by(desc(JobBookmark.created_at))
                .limit(limit)
            )
            rows = result.scalars().all()
            return [
                {
                    "id": str(r.id),
                    "title": r.title,
                    "url": r.url,
                    "company": r.company,
                    "snippet": r.snippet,
                    "rank_score": r.rank_score,
                    "search_query": r.search_query,
                    "skills_matched": r.skills_matched or [],
                    "saved_at": r.created_at.isoformat(),
                }
                for r in rows
            ]

    async def get_bookmarked_urls(self, user_id: str) -> set:
        """Return a set of all bookmarked URLs for fast deduplication."""
        async with self.async_session_maker() as session:
            result = await session.execute(
                select(JobBookmark.url).where(JobBookmark.user_id == user_id)
            )
            return {row[0] for row in result.all()}


# Singleton instance
jobs_repository = JobsRepository()
