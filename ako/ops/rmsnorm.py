"""Unified RMSNorm API with adaptive kernel dispatch."""

from __future__ import annotations

import torch

from ako.dispatcher.selector import get_selector
from ako.ops._registry import register_all


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    kernel: str | None = None,
) -> torch.Tensor:
    """RMSNorm with automatic kernel selection.

    Args:
        x: Input tensor ``(..., hidden_dim)``.
        weight: Scale weights ``(hidden_dim,)``.
        eps: Epsilon for stability.
        kernel: Force a specific kernel name (e.g. ``"rmsnorm_ref"``).

    Returns:
        Normalized tensor.
    """
    register_all()
    fn = get_selector().select("rmsnorm", x, weight, forced=kernel)
    return fn(x, weight, eps)


def rmsnorm_residual(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    kernel: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMSNorm with automatic kernel selection.

    Args:
        x: Sub-layer output ``(..., hidden_dim)``.
        residual: Incoming residual stream, same shape as ``x``.
        weight: Scale weights ``(hidden_dim,)``.
        eps: Epsilon for stability.
        kernel: Force a specific kernel name.

    Returns:
        Tuple ``(normed, new_residual)`` where ``new_residual = x + residual``.
    """
    register_all()
    fn = get_selector().select("rmsnorm_residual", x, residual, weight, forced=kernel)
    return fn(x, residual, weight, eps)
