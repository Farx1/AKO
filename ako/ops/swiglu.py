"""Unified SwiGLU API with adaptive kernel dispatch."""

from __future__ import annotations

import torch

from ako.dispatcher.selector import get_selector
from ako.ops._registry import register_all


def swiglu(gate: torch.Tensor, up: torch.Tensor, *, kernel: str | None = None) -> torch.Tensor:
    """SwiGLU activation ``silu(gate) * up`` with automatic kernel selection.

    Args:
        gate: Gate projection output ``(..., hidden_dim)``.
        up: Up projection output, same shape and dtype as ``gate``.
        kernel: Force a specific kernel name (e.g. ``"swiglu_ref"``).

    Returns:
        Tensor with the same shape and dtype as ``gate``.
    """
    register_all()
    fn = get_selector().select("swiglu", gate, up, forced=kernel)
    return fn(gate, up)


__all__ = ["swiglu"]
