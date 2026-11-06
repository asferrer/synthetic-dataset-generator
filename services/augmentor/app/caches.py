"""
Lightweight in-process LRU caches for expensive per-background artifacts.

Batches reuse a small pool of background images, so depth maps (HTTP to the
depth service), scene analyses (HTTP to SAM3) and decoded object images are
prime cache candidates. Thread-safe for use from concurrent asyncio workers.
"""

from collections import OrderedDict
from threading import Lock
from typing import Any, Optional


class LRUCache:
    """Minimal thread-safe LRU keyed by hashable values (e.g. file paths)."""

    def __init__(self, maxsize: int = 32):
        self.maxsize = maxsize
        self._data: "OrderedDict[Any, Any]" = OrderedDict()
        self._lock = Lock()

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)
