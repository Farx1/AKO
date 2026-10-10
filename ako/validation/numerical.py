"""Numerical comparison utilities for kernel validation.

A flat absolute tolerance does not survive a change of dtype. In bf16 the gap
between two representable values around magnitude 5 is ``2^-5 = 0.031``, so two
*correct* implementations that round to neighbouring values differ by more than
1e-2 — below the representation floor, an absolute threshold rejects correct
kernels. Budgets here are therefore derived from the dtype's precision and the
magnitude of the reference output.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

#: Relative spacing to allow between two correct implementations, per dtype.
#: These are the tolerances ``torch.testing.assert_close`` uses by default.
DTYPE_RTOL: dict[torch.dtype, float] = {
    torch.float64: 1e-7,
    torch.float32: 1.3e-6,
    torch.float16: 1e-3,
    torch.bfloat16: 1.6e-2,
}

#: Floor added to every budget, so near-zero outputs are not compared to zero.
DEFAULT_ATOL = 1e-5


def max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def max_abs_error_any(a: Any, b: Any) -> float:
    """Max absolute error between two kernel outputs.

    Accepts tensors or (possibly nested) sequences of tensors, so fused kernels
    returning several tensors — e.g. ``(normed, residual)`` — can be validated
    with the same call as single-output kernels.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return max_abs_error(a, b)
    if isinstance(a, Sequence) and isinstance(b, Sequence):
        if len(a) != len(b):
            raise ValueError(f"Output arity mismatch: {len(a)} vs {len(b)}")
        if not a:
            return 0.0
        return max(max_abs_error_any(x, y) for x, y in zip(a, b))
    raise TypeError(f"Cannot compare outputs of types {type(a)!r} and {type(b)!r}")


def max_rel_error(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    denom = a.float().abs().clamp_min(eps)
    return ((a.float() - b.float()).abs() / denom).max().item()


def assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> None:
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def _max_magnitude(reference: Any) -> tuple[float, torch.dtype]:
    if isinstance(reference, torch.Tensor):
        return reference.float().abs().max().item(), reference.dtype
    if isinstance(reference, Sequence) and reference:
        magnitudes, dtypes = zip(*(_max_magnitude(item) for item in reference))
        return max(magnitudes), dtypes[0]
    raise TypeError(f"Cannot measure magnitude of {type(reference)!r}")


def ulp(dtype: torch.dtype, magnitude: float) -> float:
    """Gap between two representable ``dtype`` values near ``magnitude``.

    This is the smallest difference two implementations can possibly have once
    their results are stored: an error below one ulp is a rounding difference,
    not a disagreement.
    """
    magnitude = abs(magnitude)
    info = torch.finfo(dtype)
    if magnitude < info.tiny:
        return info.tiny
    return math.ldexp(1.0, math.floor(math.log2(magnitude))) * info.eps


def error_budget(
    reference: Any,
    *,
    rtol: float | None = None,
    atol: float = DEFAULT_ATOL,
    tol_scale: float = 1.0,
) -> float:
    """Largest absolute error two correct implementations may show.

    Scaled by the dtype's relative precision and the magnitude actually present
    in ``reference``, so the same call is meaningful in fp32, fp16 and bf16.

    Args:
        reference: Baseline output — a tensor or a sequence of tensors.
        rtol: Relative tolerance; defaults to the dtype's entry in ``DTYPE_RTOL``.
        atol: Absolute floor for near-zero outputs.
        tol_scale: Budget multiplier for variants whose accumulation is
            deliberately lower precision (M5 uses ``sqrt(K)/8`` for ``ACC_FP16``).

    Returns:
        The absolute error budget.
    """
    magnitude, dtype = _max_magnitude(reference)
    if rtol is None:
        rtol = DTYPE_RTOL.get(dtype, 1e-3)
    return tol_scale * (atol + rtol * magnitude)


def error_in_ulps(reference: Any, candidate: Any) -> float:
    """Max absolute error expressed in units of the output dtype's spacing."""
    magnitude, dtype = _max_magnitude(reference)
    return max_abs_error_any(reference, candidate) / ulp(dtype, magnitude)
