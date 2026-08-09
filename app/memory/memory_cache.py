"""
Short-lived cache for assembled memory context.

Retrieval touches Postgres, Cohere and Qdrant, so caching it is worth real
latency — but a cache in front of a memory system is a correctness problem
before it is a performance one, and this one had three defects:

**It was never invalidated.** Nothing outside this module called `invalidate`.
A résumé upload, a name correction, a deleted memory — none of them cleared the
cache, so retrieval kept serving the superseded version for the full TTL. On the
deletion path that is not staleness, it is the deletion silently failing for
five minutes.

**It handed out its own storage.** `get()` returned the cached dict itself, and
the caller then assigned fresh chat history into it. Every read mutated the
cache, and two concurrent requests for the same user mutated the same object
while the other was reading it.

**It keyed on the user alone.** `RetrievalScope` exists precisely because
"whose memory" and "which visibilities" are different questions — a guest reads
the *owner's* public records. A key that names only the user cannot distinguish
those two contexts, so the first path to pass a scope through would have served
owner-private context to a guest out of the cache. That path does not exist
today; the key is scoped now so it never can.

Expiry is measured on a monotonic clock. Wall-clock time moves backwards across
NTP corrections and DST, and an entry written just before a backward jump would
otherwise never expire.
"""
from typing import Any, Dict, Iterable, Optional
import copy
import hashlib
import threading
import time


class MemoryCache:
    """
    Fast in-memory cache for memory retrieval results.

    LRU eviction with time-based expiry. Entries are copied on the way in and
    on the way out, so no caller can reach into cache storage.
    """

    def __init__(self, ttl_seconds: int = 300, max_size: int = 100):
        """
        Args:
            ttl_seconds: Time-to-live for cache entries (default 5 minutes)
            max_size: Maximum cache size (default 100 entries)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        # Retrieval runs concurrently for voice and text on the same user, and
        # eviction walks the whole dict — a lock is cheaper than reasoning about
        # which of these operations happen to be atomic under the GIL.
        self._lock = threading.Lock()

    def _make_key(
        self,
        user_id: str,
        query: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> str:
        """
        Cache key for one user, query and retrieval scope.

        `scope` distinguishes contexts that read *different data for the same
        user id* — the owner reading their own memory at every visibility, and a
        guest reading the owner's public records. Omitting it would let those
        two share an entry.

        The user id is hashed into the key rather than concatenated raw: an id
        containing the separator could otherwise be crafted to collide with
        another user's key.
        """
        digest = hashlib.sha256(
            f"{user_id}\x00{scope or ''}\x00{query or ''}".encode("utf-8")
        ).hexdigest()[:24]
        return f"{user_id}:{digest}"

    def get(
        self,
        user_id: str,
        query: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Cached memory context, or None when absent or expired.

        Returns a deep copy. The caller owns what it gets back and can mutate
        it freely — `retrieve_context` overwrites chat history and profile facts
        on every hit, and doing that to the cache's own dict was corrupting the
        entry for every subsequent reader.
        """
        key = self._make_key(user_id, query, scope)

        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None

            if time.monotonic() - entry["stored_at"] > self.ttl_seconds:
                del self._cache[key]
                return None

            entry["last_access"] = time.monotonic()
            data = entry["data"]

        return copy.deepcopy(data)

    def set(
        self,
        user_id: str,
        context: Dict[str, Any],
        query: Optional[str] = None,
        scope: Optional[str] = None,
    ) -> None:
        """Store context, copying it so later mutation by the caller cannot leak in."""
        snapshot = copy.deepcopy(context)
        key = self._make_key(user_id, query, scope)
        now = time.monotonic()

        with self._lock:
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_oldest_locked()
            self._cache[key] = {
                "data": snapshot,
                "stored_at": now,
                "last_access": now,
            }

    def _evict_oldest_locked(self) -> None:
        """Evict least recently accessed entry. Caller holds the lock."""
        if not self._cache:
            return
        oldest_key = min(
            self._cache.keys(), key=lambda k: self._cache[k]["last_access"]
        )
        del self._cache[oldest_key]

    def invalidate(self, user_id: str) -> int:
        """
        Drop every entry for a user. Returns how many were dropped.

        Called from every write path. A cache that outlives a write is a cache
        that answers questions with data the user has already corrected or
        deleted, and the count is returned so an erasure can report what it
        actually cleared instead of assuming.
        """
        prefix = f"{user_id}:"
        with self._lock:
            doomed = [k for k in self._cache if k.startswith(prefix)]
            for key in doomed:
                del self._cache[key]
        return len(doomed)

    def invalidate_many(self, user_ids: Iterable[str]) -> int:
        """Drop entries for several users — used by bulk maintenance sweeps."""
        return sum(self.invalidate(user_id) for user_id in user_ids)

    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._cache)


# Global cache instance
memory_cache = MemoryCache(ttl_seconds=300, max_size=100)
