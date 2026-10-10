"""Triton GPU implementations of RMSNorm (plain and residual-fused).

Both kernels process one row per program and keep the whole row in registers:
the input is read once, the statistics are computed in fp32, and the result is
written once (paper §7.2, fused normalization).

``BLOCK_SIZE`` is derived from ``hidden_dim`` with a heuristic instead of being
autotuned, so a row can never be silently truncated by a too-small block; the
autotuner then only explores the scheduling knobs (``num_warps``).
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on CPU-only machines
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False

#: Largest row a single-block kernel can hold; beyond this the caller must
#: fall back to the PyTorch reference (the dispatcher does so automatically).
MAX_HIDDEN_DIM = 65536

_NUM_WARPS = (2, 4, 8, 16)


def is_triton_available() -> bool:
    """True when Triton is importable *and* a CUDA device is present."""
    return TRITON_AVAILABLE and torch.cuda.is_available()


if TRITON_AVAILABLE:

    def _autotuned(kernel):
        """Wrap a kernel: ``BLOCK_SIZE`` from the row width, ``num_warps`` explored.

        The undecorated ``@triton.jit`` functions stay reachable under their
        ``_impl`` names so the kernel bodies can be executed on a CPU-only
        machine with ``TRITON_INTERPRET=1`` (the autotuner needs a GPU driver,
        the interpreter does not).
        """
        with_block_size = triton.heuristics(
            {"BLOCK_SIZE": lambda args: triton.next_power_of_2(args["hidden_dim"])}
        )(kernel)
        return triton.autotune(
            configs=[triton.Config({}, num_warps=nw) for nw in _NUM_WARPS],
            key=["hidden_dim"],
        )(with_block_size)

    @triton.jit
    def _rmsnorm_kernel_impl(
        x_ptr,
        weight_ptr,
        y_ptr,
        n_rows,
        hidden_dim,
        eps,
        stride_x_row,
        stride_y_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim

        x = tl.load(x_ptr + row_idx * stride_x_row + offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        variance = tl.sum(x * x, axis=0) / hidden_dim
        y = x * tl.rsqrt(variance + eps) * weight

        tl.store(y_ptr + row_idx * stride_y_row + offsets, y, mask=mask)

    @triton.jit
    def _rmsnorm_residual_kernel_impl(
        x_ptr,
        residual_ptr,
        weight_ptr,
        y_ptr,
        residual_out_ptr,
        n_rows,
        hidden_dim,
        eps,
        stride_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        row_off = row_idx * stride_row + offsets

        x = tl.load(x_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        # The residual stream is rounded to the storage dtype before being
        # normalized, so the fused kernel matches the reference bit-for-bit in
        # its accumulation order; it is written once and kept in registers.
        new_residual = (x + residual).to(residual_out_ptr.dtype.element_ty).to(tl.float32)
        tl.store(residual_out_ptr + row_off, new_residual, mask=mask)

        variance = tl.sum(new_residual * new_residual, axis=0) / hidden_dim
        y = new_residual * tl.rsqrt(variance + eps) * weight

        tl.store(y_ptr + row_off, y, mask=mask)

    _rmsnorm_kernel = _autotuned(_rmsnorm_kernel_impl)
    _rmsnorm_residual_kernel = _autotuned(_rmsnorm_residual_kernel_impl)


def _check_inputs(x: torch.Tensor, weight: torch.Tensor, fn_name: str) -> int:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")
    if not x.is_cuda:
        raise ValueError(f"{fn_name} requires a CUDA tensor")

    hidden_dim = x.shape[-1]
    if hidden_dim > MAX_HIDDEN_DIM:
        raise ValueError(
            f"{fn_name} supports hidden_dim <= {MAX_HIDDEN_DIM}, got {hidden_dim}"
        )
    if weight.shape != (hidden_dim,):
        raise ValueError(f"Expected weight of shape ({hidden_dim},), got {tuple(weight.shape)}")
    return hidden_dim


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm via Triton kernel (CUDA only)."""
    hidden_dim = _check_inputs(x, weight, "rmsnorm_triton")

    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, hidden_dim)
    y = torch.empty_like(x_2d)

    _rmsnorm_kernel[(x_2d.shape[0],)](
        x_2d,
        weight.contiguous(),
        y,
        x_2d.shape[0],
        hidden_dim,
        eps,
        x_2d.stride(0),
        y.stride(0),
    )
    return y.view(orig_shape)


def rmsnorm_residual_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMSNorm via Triton kernel (CUDA only).

    Returns:
        Tuple ``(normed, new_residual)`` where ``new_residual = x + residual``.
    """
    hidden_dim = _check_inputs(x, weight, "rmsnorm_residual_triton")
    if x.shape != residual.shape:
        raise ValueError(f"Shape mismatch: x {tuple(x.shape)} vs residual {tuple(residual.shape)}")

    orig_shape = x.shape
    x_2d = x.contiguous().view(-1, hidden_dim)
    residual_2d = residual.contiguous().view(-1, hidden_dim)
    y = torch.empty_like(x_2d)
    residual_out = torch.empty_like(x_2d)

    _rmsnorm_residual_kernel[(x_2d.shape[0],)](
        x_2d,
        residual_2d,
        weight.contiguous(),
        y,
        residual_out,
        x_2d.shape[0],
        hidden_dim,
        eps,
        x_2d.stride(0),
    )
    return y.view(orig_shape), residual_out.view(orig_shape)
