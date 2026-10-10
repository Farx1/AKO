"""Stronger PyTorch baseline for RoPE.

Unlike RMSNorm, PyTorch ships no fused rotary embedding, so the honest
competitor is ``torch.compile`` over the reference: Inductor fuses the same
`cat`/`stack` graph that makes `rope_ref` slow, which is exactly what the Triton
kernel does by hand.

A speedup over `rope_ref` alone mostly measures the intermediates the reference
materializes. The number worth quoting is the one over this baseline.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from ako.ops.rmsnorm_torch import is_compile_enabled
from ako.ops.rope_ref import rope_ref

_compiled_fn: Callable[..., Any] | None = None

__all__ = ["is_compile_enabled", "reset_compiled_cache", "rope_compiled"]


def rope_compiled(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, style: str = "neox"
) -> torch.Tensor:
    """RoPE through ``torch.compile`` (Inductor fuses the reference graph).

    Compiled lazily with ``dynamic=True`` so a benchmark sweep does not trigger a
    recompilation per shape. ``style`` is a Python string, so each value compiles
    its own specialization — which is what we want to measure anyway.
    """
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(rope_ref, dynamic=True)
    return _compiled_fn(x, cos, sin, style)


def reset_compiled_cache() -> None:
    """Drop the compiled callable (tests, or after changing compile options)."""
    global _compiled_fn
    _compiled_fn = None
