"""Tests for the SwiGLU reference, dispatch wrapper and Triton kernel."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ako.ops.swiglu import swiglu
from ako.ops.swiglu_ref import swiglu_ref
from ako.ops.swiglu_triton import is_triton_available, swiglu_triton
from ako.validation.numerical import assert_close, error_budget, max_abs_error


def test_swiglu_ref_matches_manual_formula() -> None:
    gate = torch.randn(4, 64)
    up = torch.randn(4, 64)
    assert_close(swiglu_ref(gate, up), F.silu(gate) * up, atol=1e-6, rtol=1e-6)


def test_swiglu_zero_up_is_zero() -> None:
    gate = torch.randn(3, 16)
    out = swiglu_ref(gate, torch.zeros(3, 16))
    assert torch.count_nonzero(out) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_swiglu_ref_preserves_dtype(dtype: torch.dtype) -> None:
    gate = torch.randn(2, 32, dtype=dtype)
    up = torch.randn(2, 32, dtype=dtype)
    out = swiglu_ref(gate, up)
    assert out.dtype == dtype
    assert out.shape == gate.shape
    assert torch.isfinite(out).all()


def test_swiglu_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        swiglu_ref(torch.randn(2, 8), torch.randn(2, 16))


def test_swiglu_dispatch_cpu_matches_reference() -> None:
    # Non power of two hidden dim: the same ragged case the RMSNorm regression
    # covers, here it exercises the masked tail of the Triton block.
    gate = torch.randn(2, 8448)
    up = torch.randn(2, 8448)
    assert_close(swiglu(gate, up), swiglu_ref(gate, up), atol=1e-6, rtol=1e-6)
    assert_close(swiglu(gate, up, kernel="swiglu_ref"), swiglu_ref(gate, up), atol=1e-6, rtol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not is_triton_available(), reason="CUDA + Triton required")
@pytest.mark.parametrize("shape", [(8, 2048, 5632), (2, 8448)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_swiglu_triton_matches_ref(shape: tuple[int, ...], dtype: torch.dtype) -> None:
    gate = torch.randn(*shape, dtype=dtype, device="cuda")
    up = torch.randn(*shape, dtype=dtype, device="cuda")
    ref = swiglu_ref(gate, up)
    assert max_abs_error(ref, swiglu_triton(gate, up)) < error_budget(ref)
