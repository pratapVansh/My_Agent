"""
Qdrant vector database service wrapper.
Provides async operations for vector storage and retrieval.
"""
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
    PayloadSchemaType,
    CreateFieldIndex
)
from typing import List, Dict, Any, Optional
import uuid
import logging
from app.config import settings
from app.services.debug_logger import log_step

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QdrantService:
    """
    Service wrapper for Qdrant Vector Database.
    Implements collection management, CRUD operations, and search.
    """

    def __init__(self):
        """Initialize Qdrant client with cloud connection."""
        self.client = AsyncQdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=settings.qdrant_timeout
        )
        self.vector_size = settings.cohere_embedding_dimension

    async def ensure_collection(
        self,
        collection_name: str,
        vector_size: Optional[int] = None
    ) -> bool:
        """
        Create collection if it doesn't exist, and ensure user_id index.

        Args:
            collection_name: Name of the collection
            vector_size: Size of vectors (default from settings)

        Returns:
            True if collection exists or was created

        Raises:
            Exception: If collection creation fails
        """
        try:
            # Check if collection exists
            collections = await self.client.get_collections()
            existing_names = [col.name for col in collections.collections]

            if collection_name not in existing_names:
                # Create collection with cosine distance and HNSW index
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size or self.vector_size,
                        distance=Distance.COSINE
                    )
                )

                logger.info(
                    f"Created collection '{collection_name}' with "
                    f"{vector_size or self.vector_size} dimensions"
                )

            # Create payload indexes required for filtering.
            for field_name in ["user_id", "type", "semantic_type"]:
                try:
                    await self.client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=PayloadSchemaType.KEYWORD
                    )
                    logger.info(
                        f"Created payload index for '{field_name}' in '{collection_name}'"
                    )
                except Exception as e:
                    # Index might already exist
                    if "already exists" not in str(e).lower():
                        logger.warning(
                            f"Could not create {field_name} index: {str(e)}"
                        )

            return True

        except Exception as e:
            logger.error(f"Failed to ensure collection '{collection_name}': {str(e)}")
            raise

    async def upsert_points(
        self,
        collection_name: str,
        points: List[PointStruct]
    ) -> bool:
        """
        Upsert points to a collection.

        Args:
            collection_name: Name of the collection
            points: List of PointStruct objects

        Returns:
            True if upsert successful

        Raises:
            Exception: If upsert fails
        """
        try:
            if not points:
                logger.warning("No points to upsert")
                return True

            await self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            logger.info(
                f"Upserted {len(points)} points to '{collection_name}'"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to upsert points to '{collection_name}': {str(e)}"
            )
            raise

    async def query_points(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[ScoredPoint]:
        """
        Query points in a collection using vector similarity.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score (optional)
            filter_conditions: Filter by metadata (e.g., {"user_id": "123"})

        Returns:
            List of ScoredPoint objects with id, score, and payload

        Raises:
            Exception: If search fails
        """
        try:
            # Build filter if conditions provided
            query_filter = None
            if filter_conditions:
                conditions = [
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
                query_filter = Filter(must=conditions)

            results = await self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True
            )

            log_step(
                "QDRANT RESULT",
                {
                    "collection": collection_name,
                    "limit": limit,
                    "filters": filter_conditions or {},
                    "count": len(results.points),
                },
            )
            logger.info(
                f"Query in '{collection_name}': found {len(results.points)} results"
            )
            return results.points

        except Exception as e:
            logger.error(f"Query failed in '{collection_name}': {str(e)}")
            raise

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[ScoredPoint]:
        """
        Backward-compatible wrapper. Prefer query_points().
        """
        return await self.query_points(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions,
        )

    async def get_by_id(
        self,
        collection_name: str,
        point_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific point by ID.

        Args:
            collection_name: Name of the collection
            point_id: Point ID to retrieve

        Returns:
            Point data or None if not found

        Raises:
            Exception: If retrieval fails
        """
        try:
            points = await self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id]
            )

            if points:
                return {
                    "id": str(points[0].id),
                    "payload": points[0].payload
                }

            return None

        except Exception as e:
            logger.error(
                f"Failed to get point '{point_id}' from '{collection_name}': {str(e)}"
            )
            raise

    async def delete_points(
        self,
        collection_name: str,
        point_ids: List[str]
    ) -> bool:
        """
        Delete points from a collection.

        Args:
            collection_name: Name of the collection
            point_ids: List of point IDs to delete

        Returns:
            True if deletion successful

        Raises:
            Exception: If deletion fails
        """
        try:
            if not point_ids:
                return True

            await self.client.delete(
                collection_name=collection_name,
                points_selector=point_ids
            )

            logger.info(
                f"Deleted {len(point_ids)} points from '{collection_name}'"
            )
            return True

        except Exception as e:
            logger.error(
                f"Failed to delete points from '{collection_name}': {str(e)}"
            )
            raise

    async def delete_by_filter(
        self,
        collection_name: str,
        filter_conditions: Dict[str, Any]
    ) -> bool:
        """
        Delete all points matching filter conditions (e.g. all points for a user).

        Args:
            collection_name: Name of the collection
            filter_conditions: e.g. {"user_id": "vansh"}

        Returns:
            True if deletion successful
        """
        try:
            conditions = [
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in filter_conditions.items()
            ]
            await self.client.delete(
                collection_name=collection_name,
                points_selector=Filter(must=conditions)
            )
            logger.info(f"Deleted points from '{collection_name}' matching {filter_conditions}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete by filter from '{collection_name}': {str(e)}")
            raise

    async def scroll_collection(
        self,
        collection_name: str,
        filter_conditions: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Scroll through collection points (for migration/export).

        Args:
            collection_name: Name of the collection
            filter_conditions: Filter by metadata
            limit: Batch size for scrolling

        Returns:
            List of points with id and payload

        Raises:
            Exception: If scroll fails
        """
        try:
            # Build filter
            scroll_filter = None
            if filter_conditions:
                conditions = [
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                    for key, value in filter_conditions.items()
                ]
                scroll_filter = Filter(must=conditions)

            points, _ = await self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )

            logger.info(f"Scrolled {len(points)} points from '{collection_name}'")

            return [
                {
                    "id": str(point.id),
                    "payload": point.payload
                }
                for point in points
            ]

        except Exception as e:
            logger.error(f"Scroll failed in '{collection_name}': {str(e)}")
            raise

    async def health_check(self) -> bool:
        """
        Verify Qdrant connectivity.

        Returns:
            True if connection is healthy, False otherwise
        """
        try:
            # Try to get collections list
            collections = await self.client.get_collections()
            logger.info(
                f"Qdrant health check passed. "
                f"Found {len(collections.collections)} collections"
            )
            return True

        except Exception as e:
            logger.error(f"Qdrant health check failed: {str(e)}")
            return False


# Global service instance (singleton pattern)
qdrant_service = QdrantService()
