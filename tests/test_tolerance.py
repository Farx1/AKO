"""Tests for dtype-aware numerical tolerances.

Context: on an RTX 3070 the bf16 RoPE kernel showed ~1.5e-2 absolute error
against the reference, above the flat 1e-2 threshold used at the time. That is
not a kernel defect — it is the bf16 representation floor. These tests pin the
reasoning down so the tolerance is never re-tuned by hand again.
"""

from __future__ import annotations

import pytest
import torch

from ako.dispatcher.cache import KernelCache
from ako.dispatcher.registry import KernelEntry, KernelRegistry
from ako.dispatcher.selector import KernelSelector
from ako.ops.rmsnorm_ref import rmsnorm_ref
from ako.validation.numerical import error_budget, error_in_ulps, max_abs_error, ulp


def test_ulp_matches_binade_spacing() -> None:
    # bf16 keeps 8 mantissa bits: in [4, 8) the spacing is 2^2 * 2^-7 = 0.03125.
    assert ulp(torch.bfloat16, 5.19) == pytest.approx(0.03125)
    assert ulp(torch.bfloat16, 1.0) == pytest.approx(0.0078125)
    # fp16 keeps 11: in [4, 8) the spacing is 2^2 * 2^-10.
    assert ulp(torch.float16, 5.19) == pytest.approx(0.00390625)
    assert ulp(torch.float32, 5.19) == pytest.approx(2**2 * 2**-23)


def test_ulp_handles_zero() -> None:
    assert ulp(torch.bfloat16, 0.0) > 0


def test_bf16_rounding_alone_exceeds_a_flat_1e_2_threshold() -> None:
    """Casting an exact fp32 result to bf16 already costs more than 1e-2."""
    exact = torch.randn(4096, dtype=torch.float32) * 5.0
    rounded = exact.to(torch.bfloat16)

    error = max_abs_error(exact, rounded.float())

    assert error > 1e-2  # the old threshold, unreachable before any kernel runs
    assert error <= ulp(torch.bfloat16, exact.abs().max().item()) / 2
    assert error < error_budget(rounded)


def test_budget_scales_with_dtype_and_magnitude() -> None:
    small = torch.randn(128, dtype=torch.bfloat16) * 0.01
    large = torch.randn(128, dtype=torch.bfloat16) * 100.0
    assert error_budget(large) > error_budget(small)

    values = torch.randn(128) * 5.0
    assert error_budget(values.to(torch.float32)) < error_budget(values.to(torch.float16))
    assert error_budget(values.to(torch.float16)) < error_budget(values.to(torch.bfloat16))


def test_budget_accepts_tuple_outputs() -> None:
    a = torch.randn(64, dtype=torch.bfloat16)
    b = torch.randn(64, dtype=torch.bfloat16) * 50.0
    assert error_budget((a, b)) == error_budget(b)


def test_error_in_ulps_is_dtype_independent() -> None:
    """One ulp of disagreement reads as ~1 whatever the dtype."""
    for dtype in (torch.float16, torch.bfloat16):
        reference = (torch.randn(256) * 5.0).to(dtype)
        spacing = ulp(dtype, reference.float().abs().max().item())
        shifted = reference.float() + spacing
        assert error_in_ulps(reference, shifted.to(dtype)) == pytest.approx(1.0, rel=0.3)


def _selector_with(candidate, tmp_path, **kwargs) -> KernelSelector:
    registry = KernelRegistry()
    registry.register(
        KernelEntry("rmsnorm", "rmsnorm_ref", rmsnorm_ref, backend="pytorch", priority=10)
    )
    registry.register(KernelEntry("rmsnorm", "candidate", candidate, backend="triton", priority=5))
    return KernelSelector(
        registry=registry,
        cache=KernelCache(path=tmp_path / "cache.json"),
        warmup=1,
        repeats=3,
        allow_cpu_tuning=True,
        **kwargs,
    )


def _one_ulp_off(x, weight, eps=1e-6):
    """A correct kernel that happens to round to the neighbouring bf16 value."""
    out = rmsnorm_ref(x, weight, eps).float()
    return (out + ulp(torch.bfloat16, out.abs().max().item())).to(x.dtype)


def test_flat_threshold_would_reject_a_correct_bf16_kernel(tmp_path) -> None:
    """The regression this replaces: a 1e-2 absolute threshold drops the kernel."""
    x = torch.randn(64, 512, dtype=torch.bfloat16) * 5.0
    weight = torch.ones(512, dtype=torch.bfloat16)

    strict = _selector_with(_one_ulp_off, tmp_path, error_threshold=1e-2)
    result = strict.select_entry("rmsnorm", x, weight)

    assert result.entry.kernel_name == "rmsnorm_ref"
    assert any("candidate" in reason for reason in result.rejected)


def test_dtype_aware_budget_keeps_a_correct_bf16_kernel(tmp_path) -> None:
    x = torch.randn(64, 512, dtype=torch.bfloat16) * 5.0
    weight = torch.ones(512, dtype=torch.bfloat16)

    selector = _selector_with(_one_ulp_off, tmp_path)  # error_threshold=None
    result = selector.select_entry("rmsnorm", x, weight)

    assert result.rejected == ()
    assert result.candidates_evaluated == 2


def test_budget_still_rejects_a_wrong_kernel(tmp_path) -> None:
    def wrong(x, weight, eps=1e-6):
        return torch.zeros_like(x)

    x = torch.randn(64, 512, dtype=torch.bfloat16) * 5.0
    selector = _selector_with(wrong, tmp_path)
    result = selector.select_entry("rmsnorm", x, torch.ones(512, dtype=torch.bfloat16))

    assert result.entry.kernel_name == "rmsnorm_ref"
    assert any("ulps" in reason for reason in result.rejected)
