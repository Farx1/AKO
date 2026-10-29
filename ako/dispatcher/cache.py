"""Persistent cache for kernel selection results.

Layout (schema v2)::

    {
      "version": 2,
      "entries": {
        "<gpu_key>": {
          "<op_name>": {
            "<shape_key>": {
              "kernel": "rmsnorm_triton",
              "latency_p50_ms": 0.27,
              "baseline_ms": 0.42,
              "speedup": 1.56,
              "max_error": 0.00031,
              "created_at": "2025-01-01T00:00:00+00:00"
            }
          }
        }
      }
    }

Schema v1 (``gpu_key -> op -> shape_key -> kernel_name`` as a bare string) is
still readable and is upgraded in memory on load.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CACHE_SCHEMA_VERSION = 2
DEFAULT_CACHE_PATH = Path.home() / ".ako" / "kernel_cache.json"

logger = logging.getLogger(__name__)


def default_cache_path() -> Path:
    """Cache location, overridable with the ``AKO_CACHE_PATH`` env var."""
    override = os.environ.get("AKO_CACHE_PATH")
    return Path(override).expanduser() if override else DEFAULT_CACHE_PATH


@dataclass(frozen=True)
class CacheStats:
    """Runtime counters for cache reuse (paper metric: cache hit rate)."""

    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "lookups": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }


class KernelCache:
    """JSON-backed cache of ``(gpu, op, shape+dtype) -> kernel`` decisions."""

    def __init__(self, path: Path | None = None, *, autosave: bool = True) -> None:
        self.path = Path(path) if path is not None else default_cache_path()
        self.autosave = autosave
        self._entries: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
        self.load()

    # -- persistence ---------------------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            self._entries = {}
            return
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt cache must never break dispatch — start fresh, but keep the
            # old file aside so the next save does not destroy it.
            logger.warning("Unreadable kernel cache %s (%s); starting empty", self.path, exc)
            if isinstance(exc, json.JSONDecodeError):
                with contextlib.suppress(OSError):
                    os.replace(self.path, self.path.with_name(self.path.name + ".corrupt"))
            self._entries = {}
            return
        self._entries = self._migrate(raw)

    @staticmethod
    def _migrate(raw: dict[str, Any]) -> dict[str, Any]:
        if raw.get("version") == CACHE_SCHEMA_VERSION:
            return raw.get("entries", {})
        # v1: the document itself is the entries map, values are kernel names.
        return {k: v for k, v in raw.items() if k != "version" and isinstance(v, dict)}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": CACHE_SCHEMA_VERSION, "entries": self._entries}
        # Write then rename: a crash mid-write never leaves a truncated cache.
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        os.replace(tmp, self.path)

    # -- lookups -------------------------------------------------------
    def get_record(self, gpu_key: str, op_name: str, shape_key: str) -> dict[str, Any] | None:
        """Return the full record for a key, or ``None`` on a miss."""
        entry = self._entries.get(gpu_key, {}).get(op_name, {}).get(shape_key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        return {"kernel": entry} if isinstance(entry, str) else dict(entry)

    def get(self, gpu_key: str, op_name: str, shape_key: str) -> str | None:
        """Return the cached kernel name for a key, or ``None`` on a miss."""
        record = self.get_record(gpu_key, op_name, shape_key)
        return record.get("kernel") if record else None

    def set(
        self,
        gpu_key: str,
        op_name: str,
        shape_key: str,
        kernel_name: str,
        **metrics: Any,
    ) -> None:
        """Store a selection decision plus the metrics that justified it."""
        record: dict[str, Any] = {
            "kernel": kernel_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        record.update({k: v for k, v in metrics.items() if v is not None})
        self._entries.setdefault(gpu_key, {}).setdefault(op_name, {})[shape_key] = record
        if self.autosave:
            self.save()

    def clear(self) -> None:
        """Drop every entry (and the file on disk when autosave is on)."""
        self._entries = {}
        self._hits = 0
        self._misses = 0
        if self.autosave:
            self.save()

    # -- introspection -------------------------------------------------
    def stats(self) -> CacheStats:
        return CacheStats(hits=self._hits, misses=self._misses)

    def to_dict(self) -> dict[str, Any]:
        return {"version": CACHE_SCHEMA_VERSION, "entries": dict(self._entries)}

    def __len__(self) -> int:
        return sum(len(shapes) for ops in self._entries.values() for shapes in ops.values())
