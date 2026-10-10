"""Adaptive Kernel Optimizer (AKO) — GPU kernel autotuning for LLM inference."""

__version__ = "0.2.0"

from ako.ops.rmsnorm import rmsnorm, rmsnorm_residual
from ako.ops.rope import build_rope_tables, rope, rope_qk
from ako.ops.swiglu import swiglu

__all__ = [
    "__version__",
    "build_rope_tables",
    "rmsnorm",
    "rmsnorm_residual",
    "rope",
    "rope_qk",
    "swiglu",
]
