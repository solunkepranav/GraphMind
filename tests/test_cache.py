import time
import pytest
from src.cache import LRUCache

def test_lru_cache_basic_operations():
    cache = LRUCache(maxsize=3)
    cache.set("k1", "v1")
    cache.set("k2", "v2")

    assert cache.get("k1") == "v1"
    assert cache.get("k2") == "v2"
    assert cache.get("k3") is None

def test_lru_cache_eviction():
    cache = LRUCache(maxsize=2)
    cache.set("a", 1)
    cache.set("b", 2)
    # Access "a" so "b" becomes least recently used
    _ = cache.get("a")
    cache.set("c", 3)

    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None  # "b" was evicted

def test_lru_cache_ttl_expiration():
    cache = LRUCache(maxsize=5, ttl_seconds=0.05)
    cache.set("temp", "val")
    assert cache.get("temp") == "val"
    time.sleep(0.06)
    assert cache.get("temp") is None
