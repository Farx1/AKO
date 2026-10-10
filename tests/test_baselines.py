"""Tests for the stronger PyTorch baselines and the timing methodology.

Two problems motivated these:

1. A speedup measured only against the unfused `rmsnorm_ref` overstates the win,
   because PyTorch ships its own fused RMSNorm and `torch.compile` fuses the same
   graph. Both must be candidates before a number is quotable.
2. Synchronizing around a single call costs more than a small kernel takes, which
   produced impossible readings — latency falling as the shape grew.
"""

from __future__ import annotations

import pytest
import torch

from ako.benchmark.bench_op import MAX_INNER_ITERS, bench_op
from ako.benchmark.report import BenchmarkReport
from ako.dispatcher.hardware import get_device_info
from ako.dispatcher.registry import KernelRegistry
from ako.ops._registry import register_all
from ako.ops.rmsnorm_ref import rmsnorm_ref
from ako.ops.rmsnorm_torch import (
    is_native_available,
    rmsnorm_compiled,
    rmsnorm_torch_native,
    reset_compiled_cache,
)
from ako.validation.numerical import error_budget, max_abs_error

pytestmark = pytest.mark.skipif(
    not is_native_available(), reason="torch.nn.functional.rms_norm requires PyTorch >= 2.4"
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_native_baseline_matches_reference(dtype: torch.dtype) -> None:
    x = torch.randn(2, 32, 128, dtype=dtype)
    weight = torch.randn(128, dtype=dtype)

    ref = rmsnorm_ref(x, weight)
    out = rmsnorm_torch_native(x, weight)

    assert out.dtype == dtype
    assert max_abs_error(ref, out) < error_budget(ref)


def test_native_baseline_is_registered_as_a_candidate() -> None:
    registry = register_all(KernelRegistry())
    names = {k.kernel_name for k in registry.list_kernels("rmsnorm")}
    assert "rmsnorm_torch_native" in names

    # The numerical reference is picked by backend == "pytorch": it must stay the
    # readable reference, not a fused kernel that could hide its own bug.
    pytorch_backed = [k for k in registry.list_kernels("rmsnorm") if k.backend == "pytorch"]
    assert [k.kernel_name for k in pytorch_backed] == ["rmsnorm_ref"]

    # And the safe default when autotuning is off is unchanged.
    assert registry.default("rmsnorm").kernel_name == "rmsnorm_ref"


def test_compiled_baselines_cover_every_op(monkeypatch) -> None:
    """Every op needs a fused competitor, not just RMSNorm.

    RoPE in particular has no fused primitive in PyTorch, so `torch.compile` is
    its only honest baseline — without it the RoPE speedup is measured purely
    against the intermediates the reference materializes.
    """
    monkeypatch.setenv("AKO_ENABLE_COMPILE", "1")
    registry = register_all(KernelRegistry(), force=True)

    for op_name, compiled in (
        ("rmsnorm", "rmsnorm_compiled"),
        ("rmsnorm_residual", "rmsnorm_residual_compiled"),
        ("rope", "rope_compiled"),
    ):
        names = {k.kernel_name for k in registry.list_kernels(op_name)}
        assert compiled in names, f"{op_name} has no torch.compile baseline"
        assert registry.get(op_name, compiled).backend == "torch_compile"


def test_compiled_baselines_are_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("AKO_ENABLE_COMPILE", raising=False)
    registry = register_all(KernelRegistry(), force=True)
    for op_name in ("rmsnorm", "rmsnorm_residual", "rope"):
        backends = {k.backend for k in registry.list_kernels(op_name)}
        assert "torch_compile" not in backends


@pytest.mark.slow
def test_compiled_baseline_matches_reference() -> None:
    reset_compiled_cache()
    x = torch.randn(2, 16, 64)
    weight = torch.randn(64)
    try:
        out = rmsnorm_compiled(x, weight)
    except Exception as exc:  # Inductor needs a working toolchain
        pytest.skip(f"torch.compile unavailable here: {exc}")
    assert max_abs_error(rmsnorm_ref(x, weight), out) < error_budget(out)


def test_bench_batches_fast_calls_together() -> None:
    """A microsecond-scale op must not be timed one call at a time."""
    x = torch.randn(8, 8)
    weight = torch.ones(8)
    result = bench_op(rmsnorm_ref, (x, weight), warmup=2, repeats=3)

    assert result.inner_iters > 1
    assert result.inner_iters <= MAX_INNER_ITERS


def test_bench_honours_explicit_inner() -> None:
    x = torch.randn(4, 64)
    result = bench_op(rmsnorm_ref, (x, torch.ones(64)), warmup=1, repeats=2, inner=3)
    assert result.inner_iters == 3


def test_bench_reports_per_call_latency() -> None:
    """Batching must not inflate the reported number: it is per single call."""
    x = torch.randn(256, 512)
    weight = torch.ones(512)
    single = bench_op(rmsnorm_ref, (x, weight), warmup=2, repeats=5, inner=1)
    batched = bench_op(rmsnorm_ref, (x, weight), warmup=2, repeats=5, inner=8)

    assert batched.latency_p50_ms == pytest.approx(single.latency_p50_ms, rel=0.85)


def test_cpu_results_have_no_device_timing() -> None:
    result = bench_op(rmsnorm_ref, (torch.randn(4, 16), torch.ones(16)), warmup=1, repeats=2)
    assert result.gpu_p50_ms is None
    assert result.launch_overhead_ms is None
    assert "gpu_p50_ms" not in result.to_dict()


def test_comparison_reports_speedup_against_the_best_baseline() -> None:
    report = BenchmarkReport(operation="rmsnorm", device=get_device_info("cpu"))
    report.add_comparison(
        shape=[8, 2048, 4096],
        baseline_ms=1.20,
        candidate_ms=0.30,
        max_error=1e-4,
        selected_kernel="rmsnorm_triton",
        best_baseline_ms=0.45,
        best_baseline_kernel="rmsnorm_torch_native",
    )
    row = report.comparisons[0]

    # 4x against the naive reference, but only 1.5x against what PyTorch already offers.
    assert row["speedup"] == 4.0
    assert row["speedup_vs_best_baseline"] == 1.5
    assert row["best_baseline_kernel"] == "rmsnorm_torch_native"
    assert row["baseline_kernel"] == "rmsnorm_ref"
