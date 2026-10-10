"""Tests for the persistent kernel cache."""

from __future__ import annotations

import json

from ako.dispatcher.cache import CACHE_SCHEMA_VERSION, KernelCache


def test_records_carry_metrics(tmp_path) -> None:
    cache = KernelCache(path=tmp_path / "cache.json")
    cache.set("L40S_0", "rmsnorm", "fp16_8x2048", "rmsnorm_triton", speedup=1.56, max_error=3e-4)

    record = cache.get_record("L40S_0", "rmsnorm", "fp16_8x2048")
    assert record is not None
    assert record["kernel"] == "rmsnorm_triton"
    assert record["speedup"] == 1.56
    assert "created_at" in record


def test_persists_and_reloads(tmp_path) -> None:
    path = tmp_path / "cache.json"
    KernelCache(path=path).set("GPU_A", "rope", "shape", "rope_triton")

    payload = json.loads(path.read_text())
    assert payload["version"] == CACHE_SCHEMA_VERSION

    assert KernelCache(path=path).get("GPU_A", "rope", "shape") == "rope_triton"


def test_reads_v1_schema(tmp_path) -> None:
    """Caches written before schema v2 stored a bare kernel name."""
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"GPU_A": {"rmsnorm": {"fp16_8x2048": "rmsnorm_triton"}}}))

    cache = KernelCache(path=path)
    assert cache.get("GPU_A", "rmsnorm", "fp16_8x2048") == "rmsnorm_triton"
    assert cache.get_record("GPU_A", "rmsnorm", "fp16_8x2048") == {"kernel": "rmsnorm_triton"}


def test_corrupt_file_does_not_break_dispatch(tmp_path) -> None:
    path = tmp_path / "cache.json"
    path.write_text("{not json")
    cache = KernelCache(path=path)
    assert cache.get("GPU_A", "rmsnorm", "shape") is None
    assert len(cache) == 0


def test_stats_track_hit_rate(tmp_path) -> None:
    cache = KernelCache(path=tmp_path / "cache.json")
    cache.set("GPU_A", "rmsnorm", "shape", "rmsnorm_ref")

    cache.get("GPU_A", "rmsnorm", "shape")
    cache.get("GPU_A", "rmsnorm", "other-shape")
    stats = cache.stats()

    assert (stats.hits, stats.misses) == (1, 1)
    assert stats.hit_rate == 0.5
    assert stats.to_dict()["lookups"] == 2


def test_clear(tmp_path) -> None:
    cache = KernelCache(path=tmp_path / "cache.json")
    cache.set("GPU_A", "rmsnorm", "shape", "rmsnorm_ref")
    assert len(cache) == 1
    cache.clear()
    assert len(cache) == 0
    assert KernelCache(path=tmp_path / "cache.json").get("GPU_A", "rmsnorm", "shape") is None
