"""The fixed reference measurement (invariant 4).

The same ``torch.matmul`` is re-timed at the start of every cell. Monotone
thermal drift is caught by re-measuring the first candidate at the end of a
sweep; a compositor wake or a browser tab is not, because it contaminates a
contiguous block of cells and leaves first-vs-last drift at 0. A canary that
never changes shape is the only thing that makes that block visible.
"""

from __future__ import annotations

from functools import lru_cache

import torch

from ako.benchmark.bench_op import BenchResult, bench_op

#: Fixed by definition: a canary that moves measures nothing.
CANARY_N = 4096
CANARY_DTYPE = torch.float16


@lru_cache(maxsize=4)
def _operands(device: str, n: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocated once: 100 MB per re-timing would fragment the allocator."""
    generator = torch.Generator(device=device).manual_seed(0)
    return tuple(
        torch.randn(n, n, generator=generator, dtype=dtype, device=device) for _ in range(2)
    )


def canary_result(device: str | None = None, *, n: int = CANARY_N, repeats: int = 20) -> BenchResult:
    """Time the canary. ``n`` is overridable only so the path can be smoke-run on a
    CPU; every recorded canary uses :data:`CANARY_N`.

    The L2 flush is off here: at 4096 cubed the working set is 100 MB against a
    4 MB L2, so flushing would add its own cost and noise to a reference whose
    only job is to be identical every time.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = CANARY_DTYPE if torch.device(device).type == "cuda" else torch.float32
    a, b = _operands(str(device), n, dtype)
    return bench_op(
        torch.matmul, (a, b), warmup=5, repeats=repeats,
        kernel_name="canary_matmul", flush_l2=False,
    )


def canary_p50(device: str | None = None) -> float:
    """Median canary latency in ms, the number written to the sweep table."""
    return canary_result(device).latency_p50_ms
