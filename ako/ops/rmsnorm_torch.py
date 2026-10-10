"""Stronger PyTorch baselines for RMSNorm.

``rmsnorm_ref`` is a readable, unfused reference: it upcasts to fp32 and
materializes every intermediate, so a fused Triton kernel beats it by a wide
margin almost by construction. A speedup measured against it alone says little.

Two harder baselines are therefore registered as dispatch candidates:

- :func:`rmsnorm_torch_native` — ``torch.nn.functional.rms_norm``, fused inside
  PyTorch since 2.4. This is what a user would realistically already be running.
- :func:`rmsnorm_compiled` — ``torch.compile`` over the reference, which fuses
  the same graph through Inductor. Opt-in (``AKO_ENABLE_COMPILE=1``) because the
  first call pays a multi-second compilation.

Beating the naive reference is expected; beating these two is the actual claim.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import torch
import torch.nn.functional as F

from ako.ops.rmsnorm_ref import rmsnorm_ref, rmsnorm_residual_ref

_compiled_fn: Callable[..., Any] | None = None
_compiled_residual_fn: Callable[..., Any] | None = None


def is_native_available() -> bool:
    """True when this PyTorch exposes the fused ``rms_norm`` primitive (>= 2.4)."""
    return hasattr(F, "rms_norm")


def is_compile_enabled() -> bool:
    """True when ``torch.compile`` baselines are explicitly enabled."""
    return os.environ.get("AKO_ENABLE_COMPILE", "").strip().lower() in {"1", "true", "yes", "on"}


def rmsnorm_torch_native(
    x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """RMSNorm through PyTorch's own fused kernel."""
    if not is_native_available():
        raise RuntimeError("torch.nn.functional.rms_norm requires PyTorch >= 2.4")
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def rmsnorm_compiled(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm through ``torch.compile`` (Inductor fuses the reference graph).

    Compiled lazily with ``dynamic=True`` so a benchmark sweep does not trigger a
    recompilation per shape. The first call is slow by design — that compilation
    cost is one of the metrics the paper asks for.
    """
    global _compiled_fn
    if _compiled_fn is None:
        _compiled_fn = torch.compile(rmsnorm_ref, dynamic=True)
    return _compiled_fn(x, weight, eps)


def rmsnorm_residual_compiled(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMSNorm through ``torch.compile``.

    Inductor fuses the add into the normalization, which is precisely what the
    Triton kernel does — so this, not the unfused reference, is the bar.
    """
    global _compiled_residual_fn
    if _compiled_residual_fn is None:
        _compiled_residual_fn = torch.compile(rmsnorm_residual_ref, dynamic=True)
    return _compiled_residual_fn(x, residual, weight, eps)


def reset_compiled_cache() -> None:
    """Drop the compiled callables (tests, or after changing compile options)."""
    global _compiled_fn, _compiled_residual_fn
    _compiled_fn = None
    _compiled_residual_fn = None
