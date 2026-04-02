"""
Qdrant-based Long-term Memory Implementation.
Stores persistent information with Cohere embeddings and text chunking.
"""
from qdrant_client.models import PointStruct
from typing import List, Dict, Any, Optional
import asyncio
import uuid
import logging
import re
from app.services.qdrant_service import qdrant_service
from app.services.cohere_service import cohere_service
from app.services.chunking_service import chunking_service
from app.services.debug_logger import log_step

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LongTermMemoryQdrant:
    """
    Long-term memory using Qdrant + Cohere + Chunking.
    Maintains API compatibility with ChromaDB implementation.
    """

    def __init__(self):
        """Initialize services and collection names."""
        self.qdrant = qdrant_service
        self.cohere = cohere_service
        self.chunker = chunking_service

        # Collection names
        self.collections = {
            "resume": "resume_chunks",
            "skills": "skills_chunks",
            "projects": "projects_chunks"
        }

    def _classify_chunk_types(self, chunk_text: str) -> List[str]:
        """Classify resume chunk into one or more specialized collections."""
        text = chunk_text.lower()

        skill_markers = ["skills", "technology", "technologies", "programming", "tools", "framework"]
        project_markers = ["project", "projects", "built", "developed", "internship", "experience"]
        chunk_types: List[str] = []

        if any(marker in text for marker in skill_markers):
            chunk_types.append("skills")
        if any(marker in text for marker in project_markers):
            chunk_types.append("projects")
        return chunk_types

    def _normalize_resume_text(self, text: str) -> str:
        """Normalize OCR/PDF text artifacts for more reliable parsing."""
        cleaned = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _detect_name(self, lines: List[str]) -> Optional[str]:
        """Detect a likely name from top lines without hallucinating values."""
        for line in lines[:5]:
            # Strip leading/trailing punctuation
            line = line.strip(" -|:")
            if not line:
                continue
            # Strip phone numbers, emails, roll numbers appended to the name line
            # e.g. "Vansh Pratap Singh+91-6392306428" → "Vansh Pratap Singh"
            line = re.split(r"[+\|]|\s*\d{7,}", line)[0].strip(" -|:")
            if not line or len(line.split()) > 6:
                continue
            lower = line.lower()
            if any(x in lower for x in ["@", "http", "linkedin", "github", "phone", "email"]):
                continue
            if ":" in line:
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,80}", line):
                return line
        return None

    def _infer_type_from_text(self, text: str) -> str:
        """Infer semantic type from text when explicit section headers are missing."""
        lower = text.lower()
        if any(k in lower for k in ["skills", "technologies", "tools", "framework", "languages"]):
            return "skills"
        if any(k in lower for k in ["project", "built", "developed", "implemented", "designed"]):
            return "projects"
        if any(k in lower for k in ["experience", "worked", "intern", "employment", "role", "company"]):
            return "experience"
        if any(k in lower for k in ["education", "bachelor", "master", "university", "college", "cgpa", "gpa"]):
            return "education"
        if any(k in lower for k in ["achievement", "award", "winner", "certification", "certified"]):
            return "other"
        return "other"

    def _importance_for_type(self, section_type: str) -> str:
        if section_type in {"name", "skills", "projects", "experience", "education"}:
            return "high"
        return "medium"

    def _extract_semantic_resume_chunks(self, resume_text: str) -> List[Dict[str, Any]]:
        """
        Extract semantic resume chunks with robust fallback.
        Guarantees non-empty output for non-empty input and at least 3 chunks when possible.
        """
        cleaned = self._normalize_resume_text(resume_text)
        if not cleaned:
            return []

        lines = [ln.strip() for ln in cleaned.split("\n")]
        non_empty_lines = [ln for ln in lines if ln]
        chunks: List[Dict[str, Any]] = []

        # 1) Name detection
        detected_name = self._detect_name(non_empty_lines)
        if detected_name:
            chunks.append(
                {
                    "type": "name",
                    "content": detected_name,
                    "tags": ["identity"],
                    "importance": "high",
                }
            )

        # 2) Section extraction by common resume headings
        heading_map = {
            "skills": [
                "skills", "technical skills", "technologies", "tools", "tech stack",
                "programming languages", "programming languages & tools", "languages & tools",
                "technical expertise", "core competencies", "key skills", "skill set",
                "technologies used", "frameworks", "software skills", "it skills",
            ],
            "projects": [
                "projects", "project experience", "key projects", "personal projects",
                "academic projects", "notable projects", "selected projects",
            ],
            "experience": [
                "experience", "work experience", "employment", "internship", "professional experience",
                "work history", "career history", "job experience", "internships",
                "professional background", "industry experience",
            ],
            "education": [
                "education", "academic background", "academics", "qualification",
                "educational background", "academic qualifications", "degrees",
                "educational qualifications", "scholastic details",
            ],
            "achievements": [
                "achievements", "awards", "certifications", "honors", "accomplishments",
                "certificates", "recognitions", "honors & awards",
            ],
        }

        def heading_to_type(value: str) -> Optional[str]:
            v = value.strip().lower().strip(":").strip()
            # Exact match first
            for t, candidates in heading_map.items():
                if v in candidates:
                    return t
            # Partial match for headings like "SKILLS & TOOLS" or "TECHNICAL SKILLS (Python, Java)"
            for t, candidates in heading_map.items():
                for candidate in candidates:
                    if candidate in v or v in candidate:
                        return t
            return None

        current_type: Optional[str] = None
        section_buffers: Dict[str, List[str]] = {
            "skills": [],
            "projects": [],
            "experience": [],
            "education": [],
            "achievements": [],
            "other": [],
        }

        for line in non_empty_lines:
            maybe_type = heading_to_type(line)
            if maybe_type:
                current_type = maybe_type
                continue

            if current_type is None:
                section_buffers["other"].append(line)
            else:
                section_buffers[current_type].append(line)

        # 3) Build section chunks
        if section_buffers["skills"]:
            raw = " ".join(section_buffers["skills"])
            skill_tokens = [s.strip(" -•") for s in re.split(r"[,|/]", raw) if s.strip()]
            if skill_tokens:
                chunks.append(
                    {
                        "type": "skills",
                        "content": ", ".join(skill_tokens),
                        "tags": ["skills", "technologies"],
                        "importance": "high",
                    }
                )

        if section_buffers["projects"]:
            project_lines = section_buffers["projects"]
            current_project: List[str] = []
            project_entries: List[str] = []
            for line in project_lines:
                new_project_signal = bool(re.match(r"^[\-*•]\s+", line)) or "project" in line.lower()
                if new_project_signal and current_project:
                    project_entries.append(" ".join(current_project).strip())
                    current_project = [line.lstrip("-*• ").strip()]
                else:
                    current_project.append(line.lstrip("-*• ").strip())
            if current_project:
                project_entries.append(" ".join(current_project).strip())

            for entry in project_entries:
                if not entry:
                    continue
                chunks.append(
                    {
                        "type": "projects",
                        "content": entry,
                        "tags": ["project"],
                        "importance": "high",
                    }
                )

        if section_buffers["experience"]:
            exp_text = " ".join(section_buffers["experience"]).strip()
            if exp_text:
                chunks.append(
                    {
                        "type": "experience",
                        "content": exp_text,
                        "tags": ["experience", "work"],
                        "importance": "high",
                    }
                )

        if section_buffers["education"]:
            edu_text = " ".join(section_buffers["education"]).strip()
            if edu_text:
                chunks.append(
                    {
                        "type": "education",
                        "content": edu_text,
                        "tags": ["education", "academics"],
                        "importance": "high",
                    }
                )

        # Achievements map to other because collection schema does not include achievements.
        if section_buffers["achievements"]:
            ach_text = " ".join(section_buffers["achievements"]).strip()
            if ach_text:
                chunks.append(
                    {
                        "type": "other",
                        "content": ach_text,
                        "tags": ["achievements"],
                        "importance": "medium",
                    }
                )

        if section_buffers["other"]:
            other_text = " ".join(section_buffers["other"]).strip()
            if other_text:
                chunks.append(
                    {
                        "type": "other",
                        "content": other_text,
                        "tags": ["general"],
                        "importance": "medium",
                    }
                )

        # 4) Guaranteed fallback: if extraction is sparse, build paragraph blocks.
        # Do not infer new skills/projects from unlabeled text.
        meaningful_chunks = [c for c in chunks if c.get("content", "").strip()]
        if len(meaningful_chunks) < 3:
            paragraphs = [p.strip() for p in re.split(r"\n\n+", cleaned) if p.strip()]
            for para in paragraphs:
                if len(" ".join(para.split()).split()) < 8:
                    continue
                meaningful_chunks.append(
                    {
                        "type": "other",
                        "content": " ".join(para.split()),
                        "tags": ["fallback"],
                        "importance": "medium",
                    }
                )
                if len(meaningful_chunks) >= 5:
                    break

        # 5) Final fallback to token windows (100-150 words) as other.
        if len(meaningful_chunks) < 3:
            words = cleaned.split()
            window = 120
            step = 110
            for i in range(0, len(words), step):
                block = words[i:i + window]
                if not block:
                    break
                meaningful_chunks.append(
                    {
                        "type": "other",
                        "content": " ".join(block),
                        "tags": ["fallback"],
                        "importance": "medium",
                    }
                )
                if len(meaningful_chunks) >= 5:
                    break

        # De-duplicate very similar chunk content while preserving order.
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for chunk in meaningful_chunks:
            content = re.sub(r"\s+", " ", chunk.get("content", "")).strip()
            if not content:
                continue
            key = content.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(
                {
                    "type": chunk.get("type", "other"),
                    "content": content,
                    "tags": chunk.get("tags", ["general"]),
                    "importance": chunk.get("importance", "medium"),
                }
            )

        return deduped

    async def initialize(self):
        """Initialize collections in Qdrant."""
        try:
            for collection_name in self.collections.values():
                await self.qdrant.ensure_collection(collection_name)
            logger.info("Long-term memory (Qdrant) initialized")
        except Exception as e:
            logger.error(f"Failed to initialize long-term memory: {str(e)}")
            raise

    async def store_resume(
        self,
        user_id: str,
        resume_text: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store resume information with chunking and embeddings.

        Args:
            user_id: User identifier
            resume_text: Full resume text
            metadata: Additional metadata

        Returns:
            Document ID (parent ID for all chunks)
        """
        try:
            parent_id = f"resume_{user_id}_{uuid.uuid4().hex[:8]}"

            cleaned_resume = self._normalize_resume_text(resume_text)
            if not cleaned_resume:
                logger.warning("Empty or malformed resume text received; skipping storage")
                return parent_id

            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "user_id": user_id,
                "type": "resume",
                "parent_id": parent_id
            })

            # Semantic extraction with guaranteed fallback for robust resume ingestion.
            semantic_chunks = self._extract_semantic_resume_chunks(cleaned_resume)

            if not semantic_chunks:
                logger.warning("Semantic extraction returned no chunks; using generic tokenizer chunking")
                generic_chunks = self.chunker.chunk_text(text=cleaned_resume, metadata=meta)
                semantic_chunks = [
                    {
                        "type": "other",
                        "content": chunk.text,
                        "tags": ["fallback"],
                        "importance": "medium",
                        "chunk_index": chunk.metadata.get("chunk_index", idx),
                    }
                    for idx, chunk in enumerate(generic_chunks)
                    if chunk.text.strip()
                ]

            if not semantic_chunks:
                logger.warning("No chunks created from resume text after fallback")
                return parent_id

            # Generate embeddings for all chunks
            chunk_texts = [chunk["content"] for chunk in semantic_chunks]
            embeddings = await self.cohere.embed_batch(
                texts=chunk_texts,
                input_type="search_document"
            )

            # Guard: if Cohere returns a partial batch, zip() would silently
            # drop trailing chunks — better to abort and let the caller retry.
            if len(embeddings) != len(chunk_texts):
                raise ValueError(
                    f"Embedding batch size mismatch for resume: "
                    f"expected {len(chunk_texts)}, got {len(embeddings)}. "
                    f"Aborting to prevent partial write."
                )

            # Create Qdrant points
            resume_points = []
            skills_points = []
            projects_points = []
            for idx, (chunk, embedding) in enumerate(zip(semantic_chunks, embeddings)):
                # Generate UUID for Qdrant compatibility
                point_id = str(uuid.uuid4())
                section_type = chunk.get("type", "other")
                resume_payload = {
                    **meta,
                    "type": "resume",
                    "semantic_type": section_type,
                    "importance": chunk.get("importance", "medium"),
                    "tags": chunk.get("tags", []),
                    "text": chunk["content"],
                    "chunk_index": idx,
                    "total_chunks": len(semantic_chunks),
                    "string_id": f"{parent_id}_chunk_{idx}"
                }
                resume_points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=resume_payload
                ))

                # Also route relevant chunks to specialized collections for strict retrieval.
                specialized_types: List[str] = []
                if section_type == "skills":
                    specialized_types.append("skills")
                if section_type == "projects":
                    specialized_types.append("projects")
                if "skills" in specialized_types:
                    skills_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                **meta,
                                "type": "skills",
                                "semantic_type": section_type,
                                "importance": chunk.get("importance", "medium"),
                                "tags": chunk.get("tags", []),
                                "text": chunk["content"],
                                "chunk_index": idx,
                                "total_chunks": len(semantic_chunks),
                                "string_id": f"skills_{parent_id}_chunk_{idx}",
                            },
                        )
                    )
                if "projects" in specialized_types:
                    projects_points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embedding,
                            payload={
                                **meta,
                                "type": "projects",
                                "semantic_type": section_type,
                                "importance": chunk.get("importance", "medium"),
                                "tags": chunk.get("tags", []),
                                "text": chunk["content"],
                                "chunk_index": idx,
                                "total_chunks": len(semantic_chunks),
                                "string_id": f"projects_{parent_id}_chunk_{idx}",
                            },
                        )
                    )

            # Collect stale point IDs BEFORE inserting new data.
            # Pattern: upsert new → delete old (readers always see at least one
            # version, never a zero-data gap).
            #
            # CRITICAL: we also scroll old resume chunks so they don't
            # accumulate across uploads — without this, retrieve_resume()
            # picks an arbitrary chunk group, not necessarily the latest one.
            old_resume_points, old_skills_points, old_projects_points = await asyncio.gather(
                self.qdrant.scroll_collection(
                    collection_name=self.collections["resume"],
                    filter_conditions={"user_id": user_id},
                    limit=200,
                ),
                self.qdrant.scroll_collection(
                    collection_name=self.collections["skills"],
                    filter_conditions={"user_id": user_id},
                ),
                self.qdrant.scroll_collection(
                    collection_name=self.collections["projects"],
                    filter_conditions={"user_id": user_id},
                ),
            )
            old_resume_ids = [p["id"] for p in old_resume_points]
            old_skills_ids = [p["id"] for p in old_skills_points]
            old_projects_ids = [p["id"] for p in old_projects_points]

            # Upsert new data first so readers are never left with nothing.
            await self.qdrant.upsert_points(
                collection_name=self.collections["resume"],
                points=resume_points
            )

            if skills_points:
                await self.qdrant.upsert_points(
                    collection_name=self.collections["skills"],
                    points=skills_points,
                )

            if projects_points:
                await self.qdrant.upsert_points(
                    collection_name=self.collections["projects"],
                    points=projects_points,
                )

            # Delete stale points AFTER the new data is live.
            #
            # CRITICAL FIX: only delete old skills/projects when we actually
            # upserted replacements. If the new resume has no skills section,
            # skills_points is empty — deleting old_skills_ids without a
            # replacement would permanently destroy the user's skills data.
            if old_resume_ids:
                await self.qdrant.delete_points(
                    collection_name=self.collections["resume"],
                    point_ids=old_resume_ids,
                )
            if skills_points and old_skills_ids:
                await self.qdrant.delete_points(
                    collection_name=self.collections["skills"],
                    point_ids=old_skills_ids,
                )
            if projects_points and old_projects_ids:
                await self.qdrant.delete_points(
                    collection_name=self.collections["projects"],
                    point_ids=old_projects_ids,
                )

            logger.info(
                f"Stored resume for user '{user_id}' as {len(semantic_chunks)} chunks; "
                f"skills_chunks={len(skills_points)}, projects_chunks={len(projects_points)}"
            )
            return parent_id

        except Exception as e:
            logger.error(f"Failed to store resume: {str(e)}")
            raise

    async def store_skill(
        self,
        user_id: str,
        skill_name: str,
        skill_level: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store a skill.

        Args:
            user_id: User identifier
            skill_name: Name of the skill
            skill_level: Proficiency level
            metadata: Additional metadata

        Returns:
            Document ID
        """
        try:
            # Generate UUID for Qdrant compatibility
            point_id = str(uuid.uuid4())
            string_id = f"skill_{user_id}_{uuid.uuid4().hex[:8]}"

            # Format skill document
            document = f"{skill_name}: {skill_level}"
            if metadata and "description" in metadata:
                document += f" - {metadata['description']}"

            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "user_id": user_id,
                "skill_name": skill_name,
                "skill_level": skill_level,
                "type": "skills",
                "parent_id": string_id,
                "string_id": string_id
            })

            # Generate embedding (skills are usually short, single chunk)
            embedding = await self.cohere.embed_text(
                text=document,
                input_type="search_document"
            )

            # Create Qdrant point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    **meta,
                    "text": document
                }
            )

            # Upsert to Qdrant
            await self.qdrant.upsert_points(
                collection_name=self.collections["skills"],
                points=[point]
            )

            logger.info(f"Stored skill '{skill_name}' for user '{user_id}'")
            return string_id

        except Exception as e:
            logger.error(f"Failed to store skill: {str(e)}")
            raise

    async def store_project(
        self,
        user_id: str,
        project_name: str,
        description: str,
        technologies: List[str],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Store project information.

        Args:
            user_id: User identifier
            project_name: Project name
            description: Project description
            technologies: List of technologies used
            metadata: Additional metadata

        Returns:
            Document ID (parent ID for chunks)
        """
        try:
            parent_id = f"project_{user_id}_{uuid.uuid4().hex[:8]}"

            # Format project document
            document = (
                f"Project: {project_name}\n"
                f"Description: {description}\n"
                f"Technologies: {', '.join(technologies)}"
            )

            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "user_id": user_id,
                "project_name": project_name,
                "technologies": ",".join(technologies),
                "type": "projects",
                "parent_id": parent_id
            })

            # Chunk the project description (if long)
            chunks = self.chunker.chunk_text(
                text=document,
                metadata=meta
            )

            if not chunks:
                logger.warning("No chunks created from project")
                return parent_id

            # Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.cohere.embed_batch(
                texts=chunk_texts,
                input_type="search_document"
            )

            if len(embeddings) != len(chunk_texts):
                raise ValueError(
                    f"Embedding batch size mismatch for project '{project_name}': "
                    f"expected {len(chunk_texts)}, got {len(embeddings)}. "
                    f"Aborting to prevent partial write."
                )

            # Create Qdrant points
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                # Generate UUID for Qdrant compatibility
                point_id = str(uuid.uuid4())
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        **chunk.metadata,
                        "text": chunk.text,
                        "string_id": f"{parent_id}_chunk_{chunk.metadata['chunk_index']}"
                    }
                ))

            # Upsert to Qdrant
            await self.qdrant.upsert_points(
                collection_name=self.collections["projects"],
                points=points
            )

            logger.info(
                f"Stored project '{project_name}' as {len(chunks)} chunks"
            )
            return parent_id

        except Exception as e:
            logger.error(f"Failed to store project: {str(e)}")
            raise

    async def retrieve_resume(self, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the most recent resume for a user.

        Args:
            user_id: User identifier

        Returns:
            Resume content and metadata, or None
        """
        try:
            logger.debug("retrieve_resume called for user_id=%s", user_id)

            # Get all resume chunks for user
            results = await self.qdrant.scroll_collection(
                collection_name=self.collections["resume"],
                filter_conditions={"user_id": user_id},
                limit=100
            )

            if not results:
                return None

            # Group by parent_id and get most recent
            parent_groups = {}
            for point in results:
                parent_id = point["payload"].get("parent_id")
                if parent_id not in parent_groups:
                    parent_groups[parent_id] = []
                parent_groups[parent_id].append(point)

            # Get the most recent parent_id (could add timestamp comparison)
            latest_parent = list(parent_groups.keys())[-1]
            chunks = parent_groups[latest_parent]

            # Sort chunks by index
            sorted_chunks = sorted(
                chunks,
                key=lambda x: x["payload"].get("chunk_index", 0)
            )

            # Reconstruct text
            full_text = "\n\n".join(
                chunk["payload"]["text"] for chunk in sorted_chunks
            )

            return {
                "content": full_text,
                "metadata": sorted_chunks[0]["payload"]
            }

        except Exception as e:
            logger.error(f"Failed to retrieve resume: {str(e)}")
            return None

    async def retrieve_skills(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> Any:
        """
        Retrieve skills for a user.

        Args:
            user_id: User identifier
            query: Optional semantic search query
            limit: Maximum number of results

        Returns:
            List of skills with metadata
        """
        try:
            log_step("RETRIEVAL FILTER", {"user_id": user_id, "type": "skills"})

            if query:
                # Semantic search
                query_embedding = await self.cohere.embed_text(
                    text=query,
                    input_type="search_query"
                )

                log_step("EMBEDDING DONE", {"input_type": "search_query", "target": "skills"})

                results = await self.qdrant.query_points(
                    collection_name=self.collections["skills"],
                    query_vector=query_embedding,
                    limit=limit,
                    filter_conditions={"user_id": user_id, "type": "skills"}
                )

                if not results:
                    return "NO_DATA"

                return [
                    {
                        "content": result.payload["text"],
                        "metadata": result.payload,
                        "score": result.score
                    }
                    for result in results
                ]
            else:
                # Get all skills from skills_chunks
                results = await self.qdrant.scroll_collection(
                    collection_name=self.collections["skills"],
                    filter_conditions={"user_id": user_id, "type": "skills"},
                    limit=limit
                )

                if not results:
                    return "NO_DATA"

                return [
                    {
                        "content": point["payload"]["text"],
                        "metadata": point["payload"]
                    }
                    for point in results
                ]

        except Exception as e:
            logger.error(f"Failed to retrieve skills: {str(e)}")
            return []

    async def retrieve_projects(
        self,
        user_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> Any:
        """
        Retrieve projects for a user.

        Args:
            user_id: User identifier
            query: Optional semantic search query
            limit: Maximum number of results

        Returns:
            List of projects with metadata
        """
        try:
            log_step("RETRIEVAL FILTER", {"user_id": user_id, "type": "projects"})

            if query:
                # Semantic search
                query_embedding = await self.cohere.embed_text(
                    text=query,
                    input_type="search_query"
                )

                log_step("EMBEDDING DONE", {"input_type": "search_query", "target": "projects"})

                results = await self.qdrant.query_points(
                    collection_name=self.collections["projects"],
                    query_vector=query_embedding,
                    limit=limit * 3,  # Get more chunks, then group
                    score_threshold=0.3,
                    filter_conditions={"user_id": user_id, "type": "projects"}
                )

                if not results:
                    return "NO_DATA"

                # Group chunks by parent_id
                parent_groups = {}
                for result in results:
                    parent_id = result.payload.get("parent_id")
                    if parent_id not in parent_groups:
                        parent_groups[parent_id] = []
                    parent_groups[parent_id].append(result)

                # Reconstruct projects
                projects = []
                for parent_id, chunks in list(parent_groups.items())[:limit]:
                    sorted_chunks = sorted(
                        chunks,
                        key=lambda x: x.payload.get("chunk_index", 0)
                    )

                    full_text = "\n\n".join(
                        chunk.payload["text"] for chunk in sorted_chunks
                    )

                    # Use highest score from chunks
                    max_score = max(chunk.score for chunk in sorted_chunks)

                    projects.append({
                        "content": full_text,
                        "metadata": sorted_chunks[0].payload,
                        "score": max_score
                    })

                return projects

            else:
                # Get all projects from projects_chunks
                results = await self.qdrant.scroll_collection(
                    collection_name=self.collections["projects"],
                    filter_conditions={"user_id": user_id, "type": "projects"},
                    limit=limit * 10  # Get more to account for chunks
                )

                if not results:
                    return "NO_DATA"

                # Group by parent_id
                parent_groups = {}
                for point in results:
                    parent_id = point["payload"].get("parent_id")
                    if parent_id not in parent_groups:
                        parent_groups[parent_id] = []
                    parent_groups[parent_id].append(point)

                # Reconstruct projects
                projects = []
                for parent_id, chunks in list(parent_groups.items())[:limit]:
                    sorted_chunks = sorted(
                        chunks,
                        key=lambda x: x["payload"].get("chunk_index", 0)
                    )

                    full_text = "\n\n".join(
                        chunk["payload"]["text"] for chunk in sorted_chunks
                    )

                    projects.append({
                        "content": full_text,
                        "metadata": sorted_chunks[0]["payload"]
                    })

                return projects

        except Exception as e:
            logger.error(f"Failed to retrieve projects: {str(e)}")
            return []

    async def _fallback_resume_search(
        self,
        user_id: str,
        query: str,
        semantic_type: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback: semantic search on resume_chunks when dedicated collection is empty.
        Filters only by user_id (no semantic_type index required on existing collections).
        """
        try:
            query_embedding = await self.cohere.embed_text(
                text=query,
                input_type="search_query"
            )
            results = await self.qdrant.query_points(
                collection_name=self.collections["resume"],
                query_vector=query_embedding,
                limit=limit,
                score_threshold=0.25,
                filter_conditions={"user_id": user_id}
            )
            if not results:
                return []
            return [
                {
                    "content": r.payload["text"],
                    "metadata": r.payload,
                    "score": r.score,
                    "source": "resume_fallback"
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Fallback resume search failed for {semantic_type}: {str(e)}")
            return []

    async def search_all(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search across all collections for relevant information.
        Falls back to resume_chunks when dedicated collections are empty.

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results per collection

        Returns:
            Dictionary with results from each collection
        """
        try:
            log_step("USER QUERY", {"user_id": user_id, "query": query})

            # Run all three primary lookups in parallel (~200ms gain)
            skills, projects, resume = await asyncio.gather(
                self.retrieve_skills(user_id, query, limit),
                self.retrieve_projects(user_id, query, limit),
                self.retrieve_resume(user_id),
            )

            # Fallback lookups for missing dedicated collections (run in parallel too)
            needs_skills_fallback = skills == "NO_DATA"
            needs_projects_fallback = projects == "NO_DATA"

            if needs_skills_fallback or needs_projects_fallback:
                fallback_tasks = []
                if needs_skills_fallback:
                    fallback_tasks.append(
                        self._fallback_resume_search(user_id, query, "skills", limit)
                    )
                if needs_projects_fallback:
                    fallback_tasks.append(
                        self._fallback_resume_search(user_id, query, "projects", limit)
                    )
                fallback_results = await asyncio.gather(*fallback_tasks)

                idx = 0
                if needs_skills_fallback:
                    skills_fallback = fallback_results[idx]; idx += 1
                    skills_result = skills_fallback
                    skills_status = "FALLBACK" if skills_fallback else "NO_DATA"
                else:
                    skills_result = skills
                    skills_status = "OK"

                if needs_projects_fallback:
                    projects_fallback = fallback_results[idx]
                    projects_result = projects_fallback
                    projects_status = "FALLBACK" if projects_fallback else "NO_DATA"
                else:
                    projects_result = projects
                    projects_status = "OK"
            else:
                skills_result = skills
                skills_status = "OK"
                projects_result = projects
                projects_status = "OK"

            return {
                "resume": resume or {},
                "skills": skills_result,
                "projects": projects_result,
                "skills_status": skills_status,
                "projects_status": projects_status,
            }

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"resume": {}, "skills": [], "projects": []}


# Singleton instance
long_term_memory_qdrant = LongTermMemoryQdrant()
