"""Unified RoPE API with adaptive kernel dispatch."""

from __future__ import annotations

import torch

from ako.dispatcher.selector import get_selector
from ako.ops._registry import register_all
from ako.ops.rope_ref import build_rope_tables


def rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    style: str = "neox",
    *,
    kernel: str | None = None,
) -> torch.Tensor:
    """Rotary position embedding with automatic kernel selection.

    Args:
        x: Input ``(batch, seq_len, n_heads, head_dim)``.
        cos: Cosine table ``(seq_len, head_dim // 2)`` from :func:`build_rope_tables`.
        sin: Sine table ``(seq_len, head_dim // 2)``.
        style: ``"neox"`` (default) or ``"interleaved"``.
        kernel: Force a specific kernel name (e.g. ``"rope_ref"``).

    Returns:
        Rotated tensor, same shape and dtype as ``x``.
    """
    register_all()
    fn = get_selector().select(
        "rope",
        x,
        cos,
        sin,
        forced=kernel,
        extra_key=style,
        call_kwargs={"style": style},
    )
    return fn(x, cos, sin, style)


def rope_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    style: str = "neox",
    *,
    kernel: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to a query/key pair with automatic kernel selection."""
    return (
        rope(q, cos, sin, style, kernel=kernel),
        rope(k, cos, sin, style, kernel=kernel),
    )


__all__ = ["build_rope_tables", "rope", "rope_qk"]
