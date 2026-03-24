"""
Qdrant-based Long-term Memory Implementation.
Stores persistent information with Cohere embeddings and text chunking.
"""
from qdrant_client.models import PointStruct
from typing import List, Dict, Any, Optional
import uuid
import logging
from app.services.qdrant_service import qdrant_service
from app.services.cohere_service import cohere_service
from app.services.chunking_service import chunking_service

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

            # Prepare metadata
            meta = metadata or {}
            meta.update({
                "user_id": user_id,
                "type": "resume",
                "parent_id": parent_id
            })

            # Chunk the resume text
            chunks = self.chunker.chunk_text(
                text=resume_text,
                metadata=meta
            )

            if not chunks:
                logger.warning("No chunks created from resume text")
                return parent_id

            # Generate embeddings for all chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.cohere.embed_batch(
                texts=chunk_texts,
                input_type="search_document"
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
                collection_name=self.collections["resume"],
                points=points
            )

            logger.info(
                f"Stored resume for user '{user_id}' as {len(chunks)} chunks"
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
                "type": "skill",
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
                "type": "project",
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
    ) -> List[Dict[str, Any]]:
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
            if query:
                # Semantic search
                query_embedding = await self.cohere.embed_text(
                    text=query,
                    input_type="search_query"
                )

                results = await self.qdrant.search(
                    collection_name=self.collections["skills"],
                    query_vector=query_embedding,
                    limit=limit,
                    filter_conditions={"user_id": user_id}
                )

                return [
                    {
                        "content": result.payload["text"],
                        "metadata": result.payload,
                        "score": result.score
                    }
                    for result in results
                ]
            else:
                # Get all skills (no filtering)
                results = await self.qdrant.scroll_collection(
                    collection_name=self.collections["skills"],
                    filter_conditions={"user_id": user_id},
                    limit=limit
                )

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
    ) -> List[Dict[str, Any]]:
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
            if query:
                # Semantic search
                query_embedding = await self.cohere.embed_text(
                    text=query,
                    input_type="search_query"
                )

                results = await self.qdrant.search(
                    collection_name=self.collections["projects"],
                    query_vector=query_embedding,
                    limit=limit * 3,  # Get more chunks, then group
                    filter_conditions={"user_id": user_id}
                )

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
                # Get all projects
                results = await self.qdrant.scroll_collection(
                    collection_name=self.collections["projects"],
                    filter_conditions={"user_id": user_id},
                    limit=limit * 10  # Get more to account for chunks
                )

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

    async def search_all(
        self,
        user_id: str,
        query: str,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        Search across all collections for relevant information.

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results per collection

        Returns:
            Dictionary with results from each collection
        """
        try:
            return {
                "resume": await self.retrieve_resume(user_id) or {},
                "skills": await self.retrieve_skills(user_id, query, limit),
                "projects": await self.retrieve_projects(user_id, query, limit)
            }

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return {"resume": {}, "skills": [], "projects": []}


# Singleton instance
long_term_memory_qdrant = LongTermMemoryQdrant()
