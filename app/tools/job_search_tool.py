"""
Job search tool powered by Tavily.
Filters and ranks job results, then personalizes ranking using user memory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config import settings
from app.memory.memory_manager import memory_manager
from app.services.langsmith_service import traceable


class JobSearchTool:
    """Search and rank jobs using Tavily + memory personalization."""

    @traceable(name="tool_job_search", run_type="tool", tags=["tool", "job"])
    async def search_jobs(
        self,
        user_id: str,
        query: str,
        location: Optional[str] = None,
        max_results: int = 10,
        min_score: float = 0.2,
    ) -> Dict[str, Any]:
        """Search jobs, apply filtering, and rank by relevance and personalization."""
        search_query = self._build_search_query(query=query, location=location)
        personalization_text = await self._build_personalization_profile(user_id=user_id, query=query)

        raw_results = await self._search_with_tavily(search_query=search_query, max_results=max_results * 2)
        filtered = self._filter_results(raw_results, min_score=min_score)
        ranked = self._rank_results(filtered, personalization_text=personalization_text, query=query)

        return {
            "tool": "job_search",
            "success": True,
            "query": search_query,
            "user_id": user_id,
            "total_candidates": len(raw_results),
            "total_filtered": len(filtered),
            "results": ranked[:max_results],
            "personalization_profile": personalization_text,
        }

    async def _search_with_tavily(self, search_query: str, max_results: int) -> List[Dict[str, Any]]:
        if not settings.tavily_api_key:
            return []

        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=settings.tavily_api_key)
            response = await client.search(
                query=search_query,
                max_results=max_results,
                search_depth="advanced",
                include_raw_content=False,
            )
            return response.get("results", [])
        except Exception:
            return []

    async def _build_personalization_profile(self, user_id: str, query: str) -> str:
        context = await memory_manager.retrieve_context(
            user_id=user_id,
            session_id=f"job_search_{user_id}",
            query=query,
        )
        return memory_manager.format_context_for_prompt(context)

    def _build_search_query(self, query: str, location: Optional[str]) -> str:
        base = f"{query} jobs"
        if location:
            base = f"{base} in {location}"
        return f"{base} site:linkedin.com OR site:indeed.com OR site:wellfound.com"

    def _filter_results(self, results: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for item in results:
            score = float(item.get("score", 0.0) or 0.0)
            url = str(item.get("url", ""))
            title = str(item.get("title", "")).lower()
            if score < min_score:
                continue
            if not url.startswith("http"):
                continue
            if "job" not in title and "career" not in title and "hiring" not in title:
                continue
            filtered.append(item)
        return filtered

    def _rank_results(
        self,
        results: List[Dict[str, Any]],
        personalization_text: str,
        query: str,
    ) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query)
        profile_tokens = self._tokenize(personalization_text)

        scored: List[Dict[str, Any]] = []
        for item in results:
            title = str(item.get("title", ""))
            content = str(item.get("content", ""))
            base_score = float(item.get("score", 0.0) or 0.0)

            text_tokens = self._tokenize(f"{title} {content}")
            query_overlap = self._overlap_ratio(text_tokens, query_tokens)
            profile_overlap = self._overlap_ratio(text_tokens, profile_tokens)

            final_score = (base_score * 0.6) + (query_overlap * 0.25) + (profile_overlap * 0.15)
            scored.append(
                {
                    "title": title,
                    "url": item.get("url"),
                    "snippet": content,
                    "source_score": base_score,
                    "query_overlap": round(query_overlap, 4),
                    "profile_overlap": round(profile_overlap, 4),
                    "rank_score": round(final_score, 4),
                }
            )

        scored.sort(key=lambda x: x["rank_score"], reverse=True)
        return scored

    def _tokenize(self, text: str) -> set[str]:
        if not text:
            return set()
        cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
        return {tok for tok in cleaned.split() if len(tok) > 2}

    def _overlap_ratio(self, text_tokens: set[str], ref_tokens: set[str]) -> float:
        if not text_tokens or not ref_tokens:
            return 0.0
        overlap = text_tokens.intersection(ref_tokens)
        return len(overlap) / max(len(ref_tokens), 1)


job_search_tool = JobSearchTool()
