"""Triton GPU implementation of the fused SwiGLU activation.

One program handles one row of ``hidden_dim`` values: ``gate`` and ``up`` are
each read once, ``silu(gate) * up`` is evaluated in registers (fp32), and the
result is written once. Eager PyTorch instead materializes the ``silu``
intermediate, so it pays an extra write and an extra read per element.

Decorators follow :mod:`ako.ops.rmsnorm_triton`: ``BLOCK_SIZE`` comes from a
heuristic on the row width (a row can never be silently truncated) and the
autotuner only explores ``num_warps``.
"""

from __future__ import annotations

import torch

from ako.ops.rmsnorm_triton import MAX_HIDDEN_DIM, TRITON_AVAILABLE, is_triton_available

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    # Same decorator sandwich as the RMSNorm kernels; nothing here differs, so it
    # is imported rather than re-spelled.
    from ako.ops.rmsnorm_triton import _autotuned

    @triton.jit
    def _swiglu_kernel_impl(
        gate_ptr,
        up_ptr,
        y_ptr,
        n_rows,
        hidden_dim,
        stride_row,
        BLOCK_SIZE: tl.constexpr,
    ):
        row_idx = tl.program_id(0)
        if row_idx >= n_rows:
            return

        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < hidden_dim
        row_off = row_idx * stride_row + offsets

        gate = tl.load(gate_ptr + row_off, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(up_ptr + row_off, mask=mask, other=0.0).to(tl.float32)

        y = gate * tl.sigmoid(gate) * up

        tl.store(y_ptr + row_off, y, mask=mask)

    _swiglu_kernel = _autotuned(_swiglu_kernel_impl)


def swiglu_triton(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fused ``silu(gate) * up`` via Triton kernel (CUDA only)."""
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")
    if not gate.is_cuda:
        raise ValueError("swiglu_triton requires a CUDA tensor")
    if gate.shape != up.shape:
        raise ValueError(f"Shape mismatch: gate {tuple(gate.shape)} vs up {tuple(up.shape)}")

    hidden_dim = gate.shape[-1]
    if hidden_dim > MAX_HIDDEN_DIM:
        raise ValueError(f"swiglu_triton supports hidden_dim <= {MAX_HIDDEN_DIM}, got {hidden_dim}")

    orig_shape = gate.shape
    gate_2d = gate.contiguous().view(-1, hidden_dim)
    up_2d = up.contiguous().view(-1, hidden_dim)
    y = torch.empty_like(gate_2d)

    _swiglu_kernel[(gate_2d.shape[0],)](
        gate_2d,
        up_2d,
        y,
        gate_2d.shape[0],
        hidden_dim,
        gate_2d.stride(0),
    )
    return y.view(orig_shape)


__all__ = ["MAX_HIDDEN_DIM", "is_triton_available", "swiglu_triton"]
