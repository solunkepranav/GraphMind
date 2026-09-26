import time
import threading
from collections import OrderedDict

class LRUCache:
    """Thread-safe LRU Cache with optional TTL expiration."""

    def __init__(self, maxsize: int = 500, ttl_seconds: float = 3600):
        self.maxsize = maxsize
        self.ttl_seconds = ttl_seconds
        self._cache = OrderedDict()
        self._timestamps = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key not in self._cache:
                return None
            
            # Check TTL
            inserted_at = self._timestamps.get(key, 0)
            if time.time() - inserted_at > self.ttl_seconds:
                del self._cache[key]
                del self._timestamps[key]
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key: str, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            self._timestamps[key] = time.time()

            # Evict least recently used if over maxsize
            if len(self._cache) > self.maxsize:
                oldest_key, _ = self._cache.popitem(last=False)
                self._timestamps.pop(oldest_key, None)

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()

    def __len__(self):
        with self._lock:
            return len(self._cache)
