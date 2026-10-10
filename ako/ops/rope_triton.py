"""Triton GPU implementation of rotary position embeddings (RoPE).

One program handles one ``(batch, position, head)`` row of ``head_dim`` values:
the rotation is applied entirely in registers, so ``Q``/``K`` are read once and
written once (see paper §8, fused RoPE).
"""

from __future__ import annotations

import torch

from ako.ops.rope_ref import ROPE_STYLES

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """True when Triton is importable *and* a CUDA device is present."""
    return TRITON_AVAILABLE and torch.cuda.is_available()


if TRITON_AVAILABLE:

    @triton.jit
    def _rope_kernel_impl(
        x_ptr,
        cos_ptr,
        sin_ptr,
        y_ptr,
        n_rows,
        seq_len,
        n_heads,
        head_dim,
        half_dim,
        stride_x_row,
        stride_y_row,
        stride_table_row,
        INTERLEAVED: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return

        # Rows are laid out as ((batch * seq_len) + pos) * n_heads + head.
        pos = (row_idx // n_heads) % seq_len

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < half_dim

        table_ptr = pos * stride_table_row + offsets
        cos = tl.load(cos_ptr + table_ptr, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + table_ptr, mask=mask, other=0.0).to(tl.float32)

        x_row_ptr = x_ptr + row_idx * stride_x_row
        y_row_ptr = y_ptr + row_idx * stride_y_row

        if INTERLEAVED:
            lo_off = 2 * offsets
            hi_off = 2 * offsets + 1
        else:
            lo_off = offsets
            hi_off = half_dim + offsets

        x_lo = tl.load(x_row_ptr + lo_off, mask=mask, other=0.0).to(tl.float32)
        x_hi = tl.load(x_row_ptr + hi_off, mask=mask, other=0.0).to(tl.float32)

        y_lo = x_lo * cos - x_hi * sin
        y_hi = x_lo * sin + x_hi * cos

        tl.store(y_row_ptr + lo_off, y_lo, mask=mask)
        tl.store(y_row_ptr + hi_off, y_hi, mask=mask)

    # A rotation covers head_dim // 2 pairs, so the block is sized from the head
    # dimension and only the scheduling knobs are autotuned. ``_rope_kernel_impl``
    # stays reachable for CPU validation with TRITON_INTERPRET=1.
    _rope_kernel = triton.autotune(
        configs=[
            triton.Config({}, num_warps=nw, num_stages=ns) for nw in (2, 4, 8) for ns in (1, 2)
        ],
        key=["head_dim", "INTERLEAVED"],
    )(
        triton.heuristics({"BLOCK_SIZE": lambda args: triton.next_power_of_2(args["head_dim"] // 2)})(
            _rope_kernel_impl
        )
    )


def rope_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    style: str = "neox",
) -> torch.Tensor:
    """RoPE via Triton kernel (CUDA only).

    Args:
        x: Input ``(batch, seq_len, n_heads, head_dim)`` on a CUDA device.
        cos: Cosine table ``(seq_len, head_dim // 2)``.
        sin: Sine table ``(seq_len, head_dim // 2)``.
        style: ``"neox"`` or ``"interleaved"``.

    Returns:
        Rotated tensor, same shape/dtype as ``x``.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")
    if not x.is_cuda:
        raise ValueError("rope_triton requires a CUDA tensor")
    if style not in ROPE_STYLES:
        raise ValueError(f"Unknown RoPE style '{style}', expected one of {ROPE_STYLES}")
    if x.ndim != 4:
        raise ValueError(
            f"Expected x of shape (batch, seq_len, n_heads, head_dim), got {tuple(x.shape)}"
        )

    batch, seq_len, n_heads, head_dim = x.shape
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half_dim = head_dim // 2
    if cos.shape != (seq_len, half_dim) or sin.shape != cos.shape:
        raise ValueError(
            f"Expected cos/sin of shape {(seq_len, half_dim)}, "
            f"got {tuple(cos.shape)} / {tuple(sin.shape)}"
        )

    x_2d = x.contiguous().view(-1, head_dim)
    cos_c = cos.contiguous()
    sin_c = sin.contiguous()
    y = torch.empty_like(x_2d)
    n_rows = x_2d.shape[0]

    _rope_kernel[(n_rows,)](
        x_2d,
        cos_c,
        sin_c,
        y,
        n_rows,
        seq_len,
        n_heads,
        head_dim,
        half_dim,
        x_2d.stride(0),
        y.stride(0),
        cos_c.stride(0),
        INTERLEAVED=(style == "interleaved"),
    )
    return y.view(batch, seq_len, n_heads, head_dim)


def rope_qk_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    style: str = "neox",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the Triton RoPE kernel to a query/key pair."""
    return rope_triton(q, cos, sin, style), rope_triton(k, cos, sin, style)
