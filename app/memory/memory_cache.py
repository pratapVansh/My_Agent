"""
Simple in-memory cache for memory retrieval optimization.
Reduces latency by caching recent memory lookups.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib


class MemoryCache:
    """
    Fast in-memory cache for memory retrieval results.
    Uses LRU-style eviction with time-based expiry.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        """
        Initialize memory cache.

        Args:
            ttl_seconds: Time-to-live for cache entries (default 5 minutes)
            max_size: Maximum cache size (default 100 entries)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}

    def _make_key(self, user_id: str, query: Optional[str] = None) -> str:
        """Generate cache key from user_id and query."""
        if query:
            query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
            return f"{user_id}:{query_hash}"
        return f"{user_id}:all"

    def get(self, user_id: str, query: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached memory context.

        Args:
            user_id: User identifier
            query: Optional query string

        Returns:
            Cached context or None if not found/expired
        """
        key = self._make_key(user_id, query)
        entry = self._cache.get(key)

        if not entry:
            return None

        # Check if expired
        age = (datetime.now() - entry["timestamp"]).total_seconds()
        if age > self.ttl_seconds:
            del self._cache[key]
            return None

        # Update access time (LRU)
        entry["last_access"] = datetime.now()
        return entry["data"]

    def set(self, user_id: str, context: Dict[str, Any], query: Optional[str] = None):
        """
        Store context in cache.

        Args:
            user_id: User identifier
            context: Memory context to cache
            query: Optional query string
        """
        # Evict old entries if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        key = self._make_key(user_id, query)
        self._cache[key] = {
            "data": context,
            "timestamp": datetime.now(),
            "last_access": datetime.now()
        }

    def _evict_oldest(self):
        """Evict least recently accessed entry."""
        if not self._cache:
            return

        oldest_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k]["last_access"]
        )
        del self._cache[oldest_key]

    def invalidate(self, user_id: str):
        """
        Invalidate all cache entries for a user.

        Args:
            user_id: User identifier
        """
        keys_to_delete = [k for k in self._cache.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_delete:
            del self._cache[key]

    def clear(self):
        """Clear entire cache."""
        self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# Global cache instance
memory_cache = MemoryCache(ttl_seconds=300, max_size=100)
