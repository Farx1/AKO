"""Tests for the kernel registry and the adaptive selector."""

from __future__ import annotations

import time

import pytest
import torch

from ako.dispatcher.cache import KernelCache
from ako.dispatcher.hardware import get_device_info
from ako.dispatcher.registry import KernelEntry, KernelRegistry
from ako.dispatcher.selector import KernelSelector, shape_key
from ako.ops._registry import register_all
from ako.ops.rmsnorm_ref import rmsnorm_ref


def _selector(tmp_path, registry: KernelRegistry, **kwargs) -> KernelSelector:
    return KernelSelector(
        registry=registry,
        cache=KernelCache(path=tmp_path / "cache.json"),
        warmup=1,
        repeats=3,
        **kwargs,
    )


def _registry_with(*entries: KernelEntry) -> KernelRegistry:
    registry = KernelRegistry()
    for entry in entries:
        registry.register(entry)
    return registry


def _baseline(fn=rmsnorm_ref, priority: int = 10) -> KernelEntry:
    return KernelEntry("rmsnorm", "rmsnorm_ref", fn, backend="pytorch", priority=priority)


def test_kernel_registry() -> None:
    registry = _registry_with(_baseline())
    assert registry.default("rmsnorm") is not None
    assert registry.get("rmsnorm", "rmsnorm_ref") is not None
    assert registry.list_ops() == ["rmsnorm"]


def test_registry_replaces_same_named_kernel() -> None:
    registry = _registry_with(_baseline(), _baseline())
    assert len(registry.list_kernels("rmsnorm")) == 1


def test_registry_orders_by_priority() -> None:
    registry = _registry_with(
        KernelEntry("rmsnorm", "low", rmsnorm_ref, backend="triton", priority=1),
        _baseline(priority=99),
    )
    assert registry.default("rmsnorm").kernel_name == "rmsnorm_ref"


def test_shape_key_separates_dtype_and_variant() -> None:
    x = torch.randn(2, 4, dtype=torch.float32)
    assert shape_key((x,)) != shape_key((x.half(),))
    assert shape_key((x,), "neox") != shape_key((x,), "interleaved")


def test_selector_fallback_cpu(tmp_path) -> None:
    selector = _selector(tmp_path, _registry_with(_baseline()), auto_tune=False)
    x = torch.randn(2, 16, 32)
    weight = torch.ones(32)
    out = selector.select("rmsnorm", x, weight)(x, weight)
    assert out.shape == x.shape


def test_selector_unknown_op(tmp_path) -> None:
    selector = _selector(tmp_path, KernelRegistry())
    with pytest.raises(ValueError, match="No kernels registered"):
        selector.select("nope", torch.randn(2))


def test_selector_prefers_faster_valid_kernel(tmp_path) -> None:
    def slow(x, weight, eps=1e-6):
        time.sleep(0.002)
        return rmsnorm_ref(x, weight, eps)

    registry = _registry_with(
        _baseline(fn=slow),
        KernelEntry("rmsnorm", "fast", rmsnorm_ref, backend="triton", priority=5),
    )
    selector = _selector(tmp_path, registry, allow_cpu_tuning=True)

    x, weight = torch.randn(2, 8, 32), torch.ones(32)
    result = selector.select_entry("rmsnorm", x, weight)

    assert result.entry.kernel_name == "fast"
    assert result.candidates_evaluated == 2
    assert result.speedup > 1.0


def test_selector_rejects_numerically_wrong_kernel(tmp_path) -> None:
    def wrong(x, weight, eps=1e-6):
        return torch.zeros_like(x)

    registry = _registry_with(
        _baseline(),
        KernelEntry("rmsnorm", "wrong", wrong, backend="triton", priority=5),
    )
    selector = _selector(tmp_path, registry, allow_cpu_tuning=True)

    result = selector.select_entry("rmsnorm", torch.randn(2, 8, 32), torch.ones(32))

    assert result.entry.kernel_name == "rmsnorm_ref"
    assert any("wrong" in reason for reason in result.rejected)


def test_selector_survives_broken_kernel(tmp_path) -> None:
    def broken(x, weight, eps=1e-6):
        raise RuntimeError("kernel launch failed")

    registry = _registry_with(
        _baseline(),
        KernelEntry("rmsnorm", "broken", broken, backend="triton", priority=5),
    )
    selector = _selector(tmp_path, registry, allow_cpu_tuning=True)

    result = selector.select_entry("rmsnorm", torch.randn(2, 8, 32), torch.ones(32))

    assert result.entry.kernel_name == "rmsnorm_ref"
    assert any("kernel launch failed" in reason for reason in result.rejected)


def test_selection_is_cached_and_reused(tmp_path) -> None:
    registry = _registry_with(
        _baseline(),
        KernelEntry("rmsnorm", "alt", rmsnorm_ref, backend="triton", priority=5),
    )
    selector = _selector(tmp_path, registry, allow_cpu_tuning=True)
    x, weight = torch.randn(2, 8, 32), torch.ones(32)

    first = selector.select_entry("rmsnorm", x, weight)
    second = selector.select_entry("rmsnorm", x, weight)

    assert first.source == "benchmark"
    assert second.source == "cache"
    assert second.from_cache
    assert second.entry.kernel_name == first.entry.kernel_name
    assert selector.cache.stats().hits == 1


def test_select_memoizes_hot_path(tmp_path) -> None:
    """Repeated dispatch must not re-touch the persistent cache."""
    selector = _selector(tmp_path, _registry_with(_baseline()), auto_tune=False)
    x, weight = torch.randn(2, 8, 32), torch.ones(32)

    for _ in range(5):
        selector.select("rmsnorm", x, weight)
    assert selector.cache.stats().total == 1

    selector.clear_memo()
    selector.select("rmsnorm", x, weight)
    assert selector.cache.stats().total == 2


def test_memo_distinguishes_shape_dtype_and_variant(tmp_path) -> None:
    selector = _selector(tmp_path, _registry_with(_baseline()), auto_tune=False)
    weight = torch.ones(32)

    selector.select("rmsnorm", torch.randn(2, 8, 32), weight)
    selector.select("rmsnorm", torch.randn(4, 8, 32), weight)
    selector.select("rmsnorm", torch.randn(2, 8, 32).half(), weight)
    selector.select("rmsnorm", torch.randn(2, 8, 32), weight, extra_key="variant")

    assert selector.cache.stats().total == 4


def test_forced_kernel_bypasses_cache(tmp_path) -> None:
    registry = _registry_with(
        _baseline(),
        KernelEntry("rmsnorm", "alt", rmsnorm_ref, backend="triton", priority=5),
    )
    selector = _selector(tmp_path, registry, allow_cpu_tuning=True)
    x, weight = torch.randn(2, 8, 32), torch.ones(32)

    result = selector.select_entry("rmsnorm", x, weight, forced="alt")
    assert result.entry.kernel_name == "alt"
    assert result.source == "forced"
    assert len(selector.cache) == 0


def test_selection_result_to_dict(tmp_path) -> None:
    selector = _selector(tmp_path, _registry_with(_baseline()), auto_tune=False)
    payload = selector.select_entry("rmsnorm", torch.randn(2, 4), torch.ones(4)).to_dict()
    assert payload["selected_kernel"] == "rmsnorm_ref"
    assert payload["operation"] == "rmsnorm"
    assert payload["source"] == "default"


def test_register_all_covers_every_op() -> None:
    registry = register_all(KernelRegistry())
    assert registry.list_ops() == ["rmsnorm", "rmsnorm_residual", "rope", "swiglu"]
    for op in registry.list_ops():
        default = registry.default(op)
        assert default is not None and default.backend == "pytorch"


def test_kernel_cache_roundtrip(tmp_path) -> None:
    cache = KernelCache(path=tmp_path / "cache.json")
    cache.set("GPU_A", "rmsnorm", "fp16_8x2048", "triton_v1")
    assert cache.get("GPU_A", "rmsnorm", "fp16_8x2048") == "triton_v1"


def test_device_info_cpu() -> None:
    info = get_device_info(torch.device("cpu"))
    assert info.device_type == "cpu"
    assert info.compute_backend == "cpu"
