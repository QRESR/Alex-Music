"""In-memory TTL cache for search results and stream URLs."""

import time
from typing import Any, Optional
from dataclasses import dataclass, field


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    """Simple in-memory cache with per-key TTL."""

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        if time.time() > entry.expires_at:
            del self._store[key]
            return None
        return entry.value

    def set(self, key: str, value: Any, ttl: int):
        self._store[key] = CacheEntry(
            value=value,
            expires_at=time.time() + ttl,
        )

    def delete(self, key: str):
        self._store.pop(key, None)

    def cleanup(self):
        """Remove all expired entries."""
        now = time.time()
        expired = [k for k, v in self._store.items() if now > v.expires_at]
        for k in expired:
            del self._store[k]

    def __len__(self):
        return len(self._store)


# Global cache instances
search_cache = TTLCache()
stream_cache = TTLCache()
metadata_cache = TTLCache()
