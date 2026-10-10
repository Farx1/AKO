"""PyTorch reference implementation of the SwiGLU activation.

The MLP of a LLaMA-style block computes ``down(silu(gate(x)) * up(x))``. The
activation itself — ``silu(gate) * up`` — has no fused primitive in PyTorch: in
eager mode ``silu`` materializes a full intermediate that the multiply then
reads back, so both inputs and one temporary cross memory. That temporary is
what the fused kernel removes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def swiglu_ref(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Element-wise ``silu(gate) * up``.

    Args:
        gate: Gate projection output ``(..., hidden_dim)``.
        up: Up projection output, same shape and dtype as ``gate``.

    Returns:
        Tensor with the same shape and dtype as ``gate``.
    """
    if gate.shape != up.shape:
        raise ValueError(f"Shape mismatch: gate {tuple(gate.shape)} vs up {tuple(up.shape)}")

    # Same upcast rule as rmsnorm_ref: the activation is evaluated in fp32 so the
    # reference is never the precision bottleneck of a half-precision comparison.
    compute_dtype = torch.float32 if gate.dtype in (torch.float16, torch.bfloat16) else gate.dtype
    return (F.silu(gate.to(compute_dtype)) * up.to(compute_dtype)).to(gate.dtype)
