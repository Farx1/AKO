"""Tests for the fused residual-add + RMSNorm operation."""

from __future__ import annotations

import pytest
import torch

from ako.ops.rmsnorm import rmsnorm, rmsnorm_residual
from ako.ops.rmsnorm_ref import rmsnorm_ref, rmsnorm_residual_ref
from ako.ops.rmsnorm_triton import (
    MAX_HIDDEN_DIM,
    is_triton_available,
    rmsnorm_residual_triton,
)
from ako.validation.numerical import assert_close, error_budget, max_abs_error_any


@pytest.mark.parametrize("shape", [(2, 16, 64), (1, 4, 1024)])
def test_residual_matches_unfused_composition(shape: tuple[int, ...]) -> None:
    x = torch.randn(*shape)
    residual = torch.randn(*shape)
    weight = torch.randn(shape[-1])

    normed, new_residual = rmsnorm_residual_ref(x, residual, weight)

    assert_close(new_residual, x + residual, atol=1e-6, rtol=1e-6)
    assert_close(normed, rmsnorm_ref(x + residual, weight), atol=1e-6, rtol=1e-6)


def test_residual_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        rmsnorm_residual_ref(torch.randn(2, 8), torch.randn(2, 4), torch.ones(8))


def test_residual_dispatch_cpu() -> None:
    x = torch.randn(2, 8, 128)
    residual = torch.randn(2, 8, 128)
    weight = torch.ones(128)

    normed, new_residual = rmsnorm_residual(x, residual, weight)
    ref_normed, ref_residual = rmsnorm_residual_ref(x, residual, weight)

    assert_close(normed, ref_normed, atol=1e-6, rtol=1e-6)
    assert_close(new_residual, ref_residual, atol=1e-6, rtol=1e-6)


def test_forced_unknown_kernel_raises() -> None:
    x = torch.randn(2, 8, 32)
    with pytest.raises(ValueError, match="Unknown kernel"):
        rmsnorm(x, torch.ones(32), kernel="does_not_exist")


@pytest.mark.gpu
@pytest.mark.skipif(not is_triton_available(), reason="CUDA + Triton required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_residual_triton_matches_ref(dtype: torch.dtype) -> None:
    x = torch.randn(2, 512, 4096, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(4096, dtype=dtype, device="cuda")

    ref = rmsnorm_residual_ref(x, residual, weight)
    out = rmsnorm_residual_triton(x, residual, weight)
    assert max_abs_error_any(ref, out) < error_budget(ref)


@pytest.mark.gpu
@pytest.mark.skipif(not is_triton_available(), reason="CUDA + Triton required")
def test_triton_rejects_oversized_hidden_dim() -> None:
    x = torch.randn(1, MAX_HIDDEN_DIM + 2, device="cuda")
    with pytest.raises(ValueError, match="hidden_dim"):
        rmsnorm_residual_triton(x, x, torch.ones(MAX_HIDDEN_DIM + 2, device="cuda"))
