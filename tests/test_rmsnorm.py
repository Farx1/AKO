"""Tests for RMSNorm reference and Triton kernels."""

from __future__ import annotations

import pytest
import torch

from ako.ops.rmsnorm import rmsnorm
from ako.ops.rmsnorm_ref import rmsnorm_ref
from ako.ops.rmsnorm_triton import is_triton_available, rmsnorm_triton
from ako.validation.numerical import assert_close, error_budget, max_abs_error


@pytest.mark.parametrize("shape", [(2, 128, 512), (1, 32, 1024), (4, 64, 256)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rmsnorm_ref_shapes(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    x = torch.randn(*shape, dtype=dtype)
    weight = torch.ones(shape[-1], dtype=dtype)
    out = rmsnorm_ref(x, weight)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_rmsnorm_ref_known_value() -> None:
    x = torch.tensor([[3.0, 4.0]])
    weight = torch.ones(2)
    out = rmsnorm_ref(x, weight)
    expected = x / torch.sqrt(torch.tensor(12.5))
    assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.gpu
@pytest.mark.skipif(not is_triton_available(), reason="CUDA + Triton required")
@pytest.mark.parametrize("shape", [(2, 128, 512), (1, 2048, 4096)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_triton_matches_ref(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    weight = torch.ones(shape[-1], dtype=dtype, device="cuda")
    ref = rmsnorm_ref(x, weight)
    out = rmsnorm_triton(x, weight)
    assert max_abs_error(ref, out) < error_budget(ref)


@pytest.mark.gpu
@pytest.mark.skipif(not is_triton_available(), reason="CUDA + Triton required")
def test_rmsnorm_dispatch() -> None:
    x = torch.randn(2, 512, 1024, dtype=torch.float16, device="cuda")
    weight = torch.ones(1024, dtype=torch.float16, device="cuda")
    out_ref = rmsnorm(x, weight, kernel="rmsnorm_ref")
    out_auto = rmsnorm(x, weight)
    assert_close(out_ref, out_auto, atol=1e-2, rtol=1e-2)
