"""Tests for RoPE reference and Triton kernels."""

from __future__ import annotations

import math

import pytest
import torch

from ako.ops.rope import rope, rope_qk
from ako.ops.rope_ref import build_rope_tables, rope_qk_ref, rope_ref
from ako.ops.rope_triton import is_triton_available, rope_triton
from ako.validation.numerical import assert_close, error_budget, max_abs_error


def test_rope_tables_shape_and_values() -> None:
    cos, sin = build_rope_tables(4, 8)
    assert cos.shape == sin.shape == (4, 4)
    # Position 0 is the identity rotation.
    assert torch.allclose(cos[0], torch.ones(4))
    assert torch.allclose(sin[0], torch.zeros(4))
    # First frequency of position 1 is theta = 1 rad.
    assert cos[1, 0].item() == pytest.approx(math.cos(1.0), abs=1e-6)


def test_rope_tables_reject_odd_head_dim() -> None:
    with pytest.raises(ValueError):
        build_rope_tables(4, 7)


@pytest.mark.parametrize("style", ["neox", "interleaved"])
def test_rope_ref_matches_manual_rotation(style: str) -> None:
    batch, seq_len, heads, head_dim = 2, 6, 3, 8
    half = head_dim // 2
    x = torch.randn(batch, seq_len, heads, head_dim)
    cos, sin = build_rope_tables(seq_len, head_dim)
    out = rope_ref(x, cos, sin, style)

    if style == "neox":
        lo, hi = x[..., :half], x[..., half:]
        expected_lo = lo * cos[None, :, None, :] - hi * sin[None, :, None, :]
        assert_close(out[..., :half], expected_lo, atol=1e-6, rtol=1e-6)
    else:
        lo, hi = x[..., 0::2], x[..., 1::2]
        expected_hi = lo * sin[None, :, None, :] + hi * cos[None, :, None, :]
        assert_close(out[..., 1::2], expected_hi, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("style", ["neox", "interleaved"])
def test_rope_preserves_pair_norm(style: str) -> None:
    """A rotation is orthogonal, so it cannot change the norm of a head vector."""
    x = torch.randn(1, 12, 2, 16, dtype=torch.float64)
    cos, sin = build_rope_tables(12, 16, dtype=torch.float64)
    out = rope_ref(x, cos, sin, style)
    assert_close(out.norm(dim=-1), x.norm(dim=-1), atol=1e-10, rtol=1e-10)


def test_rope_position_zero_is_identity() -> None:
    x = torch.randn(1, 3, 1, 8)
    cos, sin = build_rope_tables(3, 8)
    out = rope_ref(x, cos, sin)
    assert_close(out[:, 0], x[:, 0], atol=1e-6, rtol=1e-6)


def test_rope_dot_product_depends_only_on_relative_position() -> None:
    """The defining RoPE property: <R_m q, R_n k> is a function of (m - n)."""
    head_dim = 16
    seq_len = 32
    q = torch.randn(1, 1, 1, head_dim, dtype=torch.float64).expand(1, seq_len, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim, dtype=torch.float64).expand(1, seq_len, 1, head_dim)
    cos, sin = build_rope_tables(seq_len, head_dim, dtype=torch.float64)
    q_rot, k_rot = rope_qk_ref(q.contiguous(), k.contiguous(), cos, sin)

    # Same relative offset (+4) at two different absolute positions.
    dot_a = (q_rot[0, 10] * k_rot[0, 6]).sum()
    dot_b = (q_rot[0, 25] * k_rot[0, 21]).sum()
    assert dot_a.item() == pytest.approx(dot_b.item(), abs=1e-8)


def test_rope_offset_shifts_positions() -> None:
    cos_full, sin_full = build_rope_tables(8, 8)
    cos_off, sin_off = build_rope_tables(4, 8, offset=4)
    assert_close(cos_off, cos_full[4:], atol=1e-6, rtol=1e-6)
    assert_close(sin_off, sin_full[4:], atol=1e-6, rtol=1e-6)


def test_rope_rejects_bad_shapes() -> None:
    x = torch.randn(2, 4, 8)  # missing the head axis
    cos, sin = build_rope_tables(4, 8)
    with pytest.raises(ValueError):
        rope_ref(x, cos, sin)

    x4 = torch.randn(2, 4, 1, 8)
    with pytest.raises(ValueError):
        rope_ref(x4, cos[:2], sin[:2])
    with pytest.raises(ValueError):
        rope_ref(x4, cos, sin, "unknown-style")


def test_rope_dispatch_cpu_matches_reference() -> None:
    x = torch.randn(1, 8, 4, 32)
    cos, sin = build_rope_tables(8, 32)
    assert_close(rope(x, cos, sin), rope_ref(x, cos, sin), atol=1e-6, rtol=1e-6)

    q_out, k_out = rope_qk(x, x, cos, sin)
    assert_close(q_out, k_out, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_rope_ref_dtype_roundtrip(dtype: torch.dtype) -> None:
    x = torch.randn(1, 8, 2, 16, dtype=dtype)
    cos, sin = build_rope_tables(8, 16, dtype=torch.float32)
    out = rope_ref(x, cos, sin)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


@pytest.mark.gpu
@pytest.mark.skipif(not is_triton_available(), reason="CUDA + Triton required")
@pytest.mark.parametrize("style", ["neox", "interleaved"])
@pytest.mark.parametrize("shape", [(2, 128, 8, 64), (1, 2048, 32, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rope_triton_matches_ref(style: str, shape: tuple[int, ...], dtype: torch.dtype) -> None:
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    cos, sin = build_rope_tables(shape[1], shape[-1], device="cuda", dtype=torch.float32)
    ref = rope_ref(x, cos, sin, style)
    out = rope_triton(x, cos, sin, style)
    # Budget scaled by the dtype and the magnitude actually produced: in bf16 the
    # gap between representable values near 5.0 is 0.031, so the reference's own
    # rounding already costs ~1.5e-2 and a flat 1e-2 could never hold.
    assert max_abs_error(ref, out) < error_budget(ref)


@pytest.mark.gpu
@pytest.mark.skipif(not is_triton_available(), reason="CUDA + Triton required")
def test_rope_dispatch_gpu() -> None:
    x = torch.randn(2, 512, 8, 64, dtype=torch.float16, device="cuda")
    cos, sin = build_rope_tables(512, 64, device="cuda", dtype=torch.float32)
    forced = rope(x, cos, sin, kernel="rope_ref")
    auto = rope(x, cos, sin)
    assert_close(forced, auto, atol=1e-2, rtol=1e-2)
