"""
Job search tool powered by Tavily.
Phase 2 upgrades:
- Pulls user skills from Qdrant and injects them into search query
- Filters out already-bookmarked/seen URLs (deduplication)
- Scores each result against user's actual skill set
- Returns skills_matched per job result
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from app.config import settings
from app.memory.memory_manager import memory_manager
from app.domain.jobs import jobs_repository
from app.memory.short_term_memory import short_term_memory
from app.services.langsmith_service import traceable

logger = logging.getLogger(__name__)


class JobSearchTool:
    """Search and rank jobs using Tavily + user skills + deduplication."""

    @traceable(name="tool_job_search", run_type="tool", tags=["tool", "job"])
    async def search_jobs(
        self,
        user_id: str,
        query: str,
        location: Optional[str] = None,
        max_results: int = 10,
        min_score: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Search jobs, filter seen URLs, rank by user skills + relevance.

        Returns:
            Dict with results, skills used, and seen_count (deduplicated)
        """
        # Fetch user skills from Qdrant
        user_skills = await self._fetch_user_skills(user_id)

        # Build enriched search query using top skills
        search_query = self._build_search_query(query=query, location=location, skills=user_skills)

        # Fetch already-seen URLs for deduplication
        seen_urls = await jobs_repository.get_bookmarked_urls(user_id)

        # Search Tavily
        raw_results = await self._search_with_tavily(search_query=search_query, max_results=max_results * 2)

        # Filter: min score + must look like a job + not already seen
        filtered = self._filter_results(raw_results, min_score=min_score, seen_urls=seen_urls)
        seen_count = len(raw_results) - len(filtered)

        # Fix 8: semantic skill matching for each job result
        # One batched Cohere embed call + parallel Qdrant queries (no N×embedding calls)
        semantic_matches: List[List[str]] = []
        if user_skills and filtered:
            job_texts = [
                f"{r.get('title', '')[:200]} {r.get('content', '')[:300]}"
                for r in filtered
            ]
            semantic_matches = await self._batch_semantic_skill_match(job_texts, user_id)

        # Rank: base score + query overlap + semantic skill match
        ranked = self._rank_results(
            filtered, query=query, user_skills=user_skills, semantic_matches=semantic_matches
        )

        return {
            "tool": "job_search",
            "success": True,
            "query": search_query,
            "user_id": user_id,
            "user_skills": user_skills[:10],  # top 10 for prompt injection
            "total_candidates": len(raw_results),
            "total_filtered": len(filtered),
            "seen_skipped": seen_count,
            "results": ranked[:max_results],
        }

    async def _fetch_user_skills(self, user_id: str) -> List[str]:
        """
        Pull user's skills from Qdrant skills_chunks.
        Returns a flat list of skill tokens (e.g. ['Python', 'React', 'Node.js']).
        """
        try:
            skills_data = await memory_manager.retrieve_skills(
                user_id=user_id,
                query="technical skills programming languages tools frameworks",
                limit=5,
            )
            if not skills_data:
                return []

            skill_tokens: List[str] = []
            for item in skills_data:
                content = item.get("content", "")
                # Split on common delimiters used in resume skill lists
                for token in content.replace(",", " ").replace("|", " ").replace("/", " ").split():
                    token = token.strip(" -•()")
                    if len(token) > 1:
                        skill_tokens.append(token)

            # Deduplicate preserving order
            seen: Set[str] = set()
            unique: List[str] = []
            for t in skill_tokens:
                lower = t.lower()
                if lower not in seen:
                    seen.add(lower)
                    unique.append(t)

            return unique[:30]  # cap at 30 skills

        except Exception as e:
            logger.warning(
                "Could not fetch skills for user=%s; ranking without skill match: %s",
                user_id, e,
            )
            return []

    def _build_search_query(
        self, query: str, location: Optional[str], skills: List[str]
    ) -> str:
        base = f"{query} jobs"
        if location:
            base = f"{base} in {location}"
        # Inject top 5 skills into search query for better Tavily results
        if skills:
            top_skills = " ".join(skills[:5])
            base = f"{base} {top_skills}"
        return f"{base} site:linkedin.com OR site:indeed.com OR site:wellfound.com"

    async def _search_with_tavily(self, search_query: str, max_results: int) -> List[Dict[str, Any]]:
        if not settings.tavily_api_key:
            logger.warning("TAVILY_API_KEY is not configured — job search returns no results")
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
        except Exception as e:
            # Degrade gracefully for the user, but make the cause visible:
            # an expired key otherwise looks identical to "no jobs found".
            logger.error("Tavily search failed: %s", e, exc_info=True)
            return []

    def _filter_results(
        self,
        results: List[Dict[str, Any]],
        min_score: float,
        seen_urls: Set[str],
    ) -> List[Dict[str, Any]]:
        filtered: List[Dict[str, Any]] = []
        for item in results:
            score = float(item.get("score", 0.0) or 0.0)
            url = str(item.get("url", ""))
            title = str(item.get("title", "")).lower()

            if score < min_score:
                continue
            if not url.startswith("http"):
                continue
            # Skip already bookmarked/seen jobs
            if url in seen_urls:
                continue

            filtered.append(item)
        return filtered

    def _extract_company(self, title: str, url: str, content: str) -> str:
        """
        Extract actual company name from job title or content.
        Handles common patterns like:
        - "Backend Intern at Google - Indeed"
        - "Software Engineer | Meta - LinkedIn"
        - "Intern Developer - Stripe | Wellfound"
        """
        import re
        # Known job board domains to strip
        boards = ["indeed", "linkedin", "wellfound", "glassdoor", "naukri",
                  "monster", "ziprecruiter", "dice", "simplyhired", "hired"]

        # Try "at CompanyName" pattern first (most reliable)
        at_match = re.search(r"\bat\s+([A-Z][A-Za-z0-9&.,'\- ]{1,40}?)(?:\s*[-|]|\s*$)", title)
        if at_match:
            candidate = at_match.group(1).strip(" -|")
            if candidate.lower() not in boards and len(candidate) > 1:
                return candidate

        # Try "Title | Company" or "Title - Company" pattern
        parts = re.split(r"\s*[|\-–]\s*", title)
        for part in reversed(parts):  # rightmost part often has company or board
            part = part.strip()
            if not part:
                continue
            if part.lower() in boards:
                continue
            # Skip if part contains only job-title words (e.g. "Backend Intern")
            title_words = {"intern", "engineer", "developer", "software", "backend",
                           "frontend", "fullstack", "data", "analyst", "scientist",
                           "manager", "associate", "junior", "senior", "lead", "role"}
            part_words = set(part.lower().split())
            if part_words and part_words.issubset(title_words):
                continue
            if len(part) > 1:
                return part

        # Try extracting domain from URL as last resort
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        if domain_match:
            domain = domain_match.group(1).split(".")[0].capitalize()
            if domain.lower() not in boards:
                return domain

        return "Unknown Company"

    async def _batch_semantic_skill_match(
        self,
        job_texts: List[str],
        user_id: str,
    ) -> List[List[str]]:
        """
        Fix 8: Semantic skill matching via Cohere embeddings + Qdrant.

        For each job text, find which of the user's stored skills are
        semantically similar — even when surface strings differ:
          'python 3.11'  → matches 'Python'
          'py'           → matches 'Python'
          'ML'           → matches 'Machine Learning'
          'ReactJS'      → matches 'React'

        Strategy:
          1. Batch-embed all job texts in ONE Cohere API call.
          2. Run one Qdrant similarity query per job in parallel.
          3. Return matched skill tokens per job.

        Falls back to empty lists gracefully on any error so downstream
        ranking degrades to substring-only (never crashes).
        """
        try:
            from app.services.cohere_service import cohere_service
            from app.services.qdrant_service import qdrant_service

            if not job_texts:
                return []

            # Single batch Cohere call — far cheaper than N individual calls
            truncated = [t[:500] for t in job_texts]
            embeddings = await cohere_service.embed_batch(
                texts=truncated,
                input_type="search_query",
            )

            if len(embeddings) != len(job_texts):
                return [[] for _ in job_texts]

            # Parallel Qdrant queries — one per job text
            async def query_one(embedding: List[float]) -> List[str]:
                try:
                    results = await qdrant_service.query_points(
                        collection_name="skills_chunks",
                        query_vector=embedding,
                        limit=8,
                        score_threshold=0.62,  # empirically tuned threshold
                        filter_conditions={"user_id": user_id, "type": "skills"},
                    )
                    if not results:
                        return []
                    matched: List[str] = []
                    seen: Set[str] = set()
                    for result in results:
                        skill_text = result.payload.get("text", "")
                        for token in skill_text.replace(",", " ").replace("|", " ").split():
                            token = token.strip(" -•()")
                            if len(token) > 1 and token.lower() not in seen:
                                seen.add(token.lower())
                                matched.append(token)
                    return matched[:8]
                except Exception:
                    return []

            match_lists = await asyncio.gather(*[query_one(emb) for emb in embeddings])
            return list(match_lists)

        except Exception as e:
            logger.warning(
                "Semantic skill match failed for user=%s; falling back to substring matching: %s",
                user_id, e,
            )
            return [[] for _ in job_texts]

    def _rank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        user_skills: List[str],
        semantic_matches: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Rank filtered job results by composite score.

        Fix 8: Merges semantic skill matches (from Cohere/Qdrant) with
        substring skill matches as a complementary fallback. Final
        skills_matched is the union, deduplicated and capped at 8.
        """
        query_tokens = self._tokenize(query)

        scored: List[Dict[str, Any]] = []
        for idx, item in enumerate(results):
            title = str(item.get("title", ""))
            content = str(item.get("content", ""))
            url = str(item.get("url", ""))
            base_score = float(item.get("score", 0.0) or 0.0)

            text_tokens = self._tokenize(f"{title} {content}")
            query_overlap = self._overlap_ratio(text_tokens, query_tokens)

            # Semantic matches from Cohere (primary)
            sem_matched: List[str] = (
                semantic_matches[idx] if semantic_matches and idx < len(semantic_matches) else []
            )

            # Substring matches as fallback complement
            sub_matched = [
                s for s in user_skills
                if s.lower() in f"{title} {content}".lower()
            ]

            # Union of both match strategies, deduplicated
            seen_lower: Set[str] = set()
            all_matched: List[str] = []
            for s in sem_matched + sub_matched:
                if s.lower() not in seen_lower:
                    seen_lower.add(s.lower())
                    all_matched.append(s)

            skill_match_ratio = len(all_matched) / max(len(user_skills), 1) if user_skills else 0.0

            final_score = (base_score * 0.5) + (query_overlap * 0.25) + (skill_match_ratio * 0.25)

            # Clean job title
            import re
            clean_title = re.split(
                r"\s*[-|]\s*(?:Indeed|LinkedIn|Wellfound|Glassdoor|Naukri|Monster)\b",
                title, flags=re.IGNORECASE
            )[0].strip()

            scored.append({
                "title": clean_title or title,
                "company": self._extract_company(title, url, content),
                "url": url,
                "snippet": content[:300],
                "source_score": base_score,
                "query_overlap": round(query_overlap, 4),
                "skill_match_ratio": round(skill_match_ratio, 4),
                "skills_matched": all_matched[:8],
                "rank_score": round(final_score, 4),
            })

        scored.sort(key=lambda x: x["rank_score"], reverse=True)
        return scored

    def _tokenize(self, text: str) -> set:
        if not text:
            return set()
        cleaned = "".join(ch if ch.isalnum() else " " for ch in text.lower())
        return {tok for tok in cleaned.split() if len(tok) > 2}

    def _overlap_ratio(self, text_tokens: set, ref_tokens: set) -> float:
        if not text_tokens or not ref_tokens:
            return 0.0
        return len(text_tokens & ref_tokens) / max(len(ref_tokens), 1)


job_search_tool = JobSearchTool()
