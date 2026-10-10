"""Stronger PyTorch baseline for SwiGLU.

PyTorch ships no fused ``silu(gate) * up`` primitive, so — as for RoPE — the
honest competitor is ``torch.compile`` over the reference: Inductor fuses the
activation and the multiply into a single pass, which is exactly what the Triton
kernel does by hand. A speedup over ``swiglu_ref`` alone mostly measures the
intermediate the reference materializes; the number worth quoting is the one
over this baseline.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from ako.ops.rmsnorm_torch import is_compile_enabled
from ako.ops.swiglu_ref import swiglu_ref

_compiled_fn: Callable[..., Any] | None = None

__all__ = ["is_compile_enabled", "reset_compiled_cache", "swiglu_compiled"]


def swiglu_compiled(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """SwiGLU through ``torch.compile`` (Inductor fuses the reference graph).

    Compiled lazily with ``dynamic=True`` so a benchmark sweep does not trigger a
    recompilation per shape.
    """
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(swiglu_ref, dynamic=True)
    return _compiled_fn(gate, up)


def reset_compiled_cache() -> None:
    """Drop the compiled callable (tests, or after changing compile options)."""
    global _compiled_fn
    _compiled_fn = None
