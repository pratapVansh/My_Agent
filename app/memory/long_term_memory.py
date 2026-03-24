"""
ChromaDB Long-term Memory Implementation.
Stores persistent information: resume, skills, projects, experiences.
"""
import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Any, Optional
import uuid
from app.config import settings


class LongTermMemory:
    """
    Long-term memory using ChromaDB.
    Stores user profile data that persists across sessions.
    """

    def __init__(self):
        """Initialize ChromaDB client and collections."""
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Collections for different types of long-term data
        self.resume_collection = self.client.get_or_create_collection(
            name="resume",
            metadata={"description": "User resume and professional summary"}
        )

        self.skills_collection = self.client.get_or_create_collection(
            name="skills",
            metadata={"description": "User skills and competencies"}
        )

        self.projects_collection = self.client.get_or_create_collection(
            name="projects",
            metadata={"description": "User projects and work experience"}
        )

    def store_resume(self, user_id: str, resume_text: str, metadata: Optional[Dict] = None) -> str:
        """
        Store resume information.

        Args:
            user_id: User identifier
            resume_text: Full resume text
            metadata: Additional metadata (e.g., update_date, version)

        Returns:
            Document ID
        """
        doc_id = f"resume_{user_id}_{uuid.uuid4().hex[:8]}"
        meta = metadata or {}
        meta["user_id"] = user_id
        meta["type"] = "resume"

        self.resume_collection.add(
            documents=[resume_text],
            metadatas=[meta],
            ids=[doc_id]
        )

        return doc_id

    def store_skill(self, user_id: str, skill_name: str, skill_level: str, metadata: Optional[Dict] = None) -> str:
        """
        Store a skill.

        Args:
            user_id: User identifier
            skill_name: Name of the skill
            skill_level: Proficiency level (beginner, intermediate, expert)
            metadata: Additional metadata

        Returns:
            Document ID
        """
        doc_id = f"skill_{user_id}_{uuid.uuid4().hex[:8]}"
        meta = metadata or {}
        meta.update({
            "user_id": user_id,
            "skill_name": skill_name,
            "skill_level": skill_level,
            "type": "skill"
        })

        document = f"{skill_name}: {skill_level}"
        if metadata and "description" in metadata:
            document += f" - {metadata['description']}"

        self.skills_collection.add(
            documents=[document],
            metadatas=[meta],
            ids=[doc_id]
        )

        return doc_id

    def store_project(
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
            Document ID
        """
        doc_id = f"project_{user_id}_{uuid.uuid4().hex[:8]}"
        meta = metadata or {}
        meta.update({
            "user_id": user_id,
            "project_name": project_name,
            "technologies": ",".join(technologies),
            "type": "project"
        })

        document = f"Project: {project_name}\nDescription: {description}\nTechnologies: {', '.join(technologies)}"

        self.projects_collection.add(
            documents=[document],
            metadatas=[meta],
            ids=[doc_id]
        )

        return doc_id

    def retrieve_resume(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent resume for a user."""
        results = self.resume_collection.get(
            where={"user_id": user_id}
        )

        if results and results["documents"]:
            return {
                "content": results["documents"][0],
                "metadata": results["metadatas"][0]
            }
        return None

    def retrieve_skills(self, user_id: str, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve skills for a user.

        Args:
            user_id: User identifier
            query: Optional semantic search query
            limit: Maximum number of results

        Returns:
            List of skills with metadata
        """
        if query:
            results = self.skills_collection.query(
                query_texts=[query],
                where={"user_id": user_id},
                n_results=limit
            )
            if results and results["documents"]:
                return [
                    {
                        "content": doc,
                        "metadata": meta
                    }
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0])
                ]
        else:
            results = self.skills_collection.get(
                where={"user_id": user_id},
                limit=limit
            )
            if results and results["documents"]:
                return [
                    {
                        "content": doc,
                        "metadata": meta
                    }
                    for doc, meta in zip(results["documents"], results["metadatas"])
                ]

        return []

    def retrieve_projects(self, user_id: str, query: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve projects for a user.

        Args:
            user_id: User identifier
            query: Optional semantic search query
            limit: Maximum number of results

        Returns:
            List of projects with metadata
        """
        if query:
            results = self.projects_collection.query(
                query_texts=[query],
                where={"user_id": user_id},
                n_results=limit
            )
            if results and results["documents"]:
                return [
                    {
                        "content": doc,
                        "metadata": meta
                    }
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0])
                ]
        else:
            results = self.projects_collection.get(
                where={"user_id": user_id},
                limit=limit
            )
            if results and results["documents"]:
                return [
                    {
                        "content": doc,
                        "metadata": meta
                    }
                    for doc, meta in zip(results["documents"], results["metadatas"])
                ]

        return []

    def search_all(self, user_id: str, query: str, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all collections for relevant information.

        Args:
            user_id: User identifier
            query: Search query
            limit: Maximum results per collection

        Returns:
            Dictionary with results from each collection
        """
        return {
            "resume": self.retrieve_resume(user_id) or {},
            "skills": self.retrieve_skills(user_id, query, limit),
            "projects": self.retrieve_projects(user_id, query, limit)
        }


# Singleton instance
long_term_memory = LongTermMemory()
