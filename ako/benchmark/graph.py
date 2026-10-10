"""CUDA-graph capture for decode-sized rows.

ARCHITECTURE.md section 1.6: at ``M <= 64`` the entire visible fusion gain is
two kernel launches, and it disappears under CUDA graphs. A row that cannot be
produced both ways cannot support that claim.

Capture runs on a side stream after a short warmup, with the caller's tensors
held live so every replay hits the same addresses. Candidates that allocate
unpredictably (``torch.compile``, split-K workspaces) fail capture; the caller
records ``graph_captured=0`` and times the uncaptured path instead of crashing
the sweep.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class CapturedKernel:
    """A captured callable, or a failed capture that must not be timed as a graph."""

    replay: Callable[[], None] | None
    captured: bool
    reason: str | None = None


def try_capture(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    warmup: int = 3,
) -> CapturedKernel:
    """Capture ``fn(*args, **kwargs)`` if the device and the kernel allow it.

    Returns an uncaptured result on CPU, when CUDA graphs are unavailable, or
    when capture raises (typically an allocation that the graph pool cannot
    pin). The tensors in ``args`` must outlive the returned object.
    """
    kwargs = kwargs or {}
    x = args[0] if args else None
    if not isinstance(x, torch.Tensor) or not x.is_cuda:
        return CapturedKernel(None, False, "not_cuda")
    if not hasattr(torch.cuda, "CUDAGraph"):
        return CapturedKernel(None, False, "cuda_graph_unavailable")

    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn(*args, **kwargs)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn(*args, **kwargs)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        return CapturedKernel(None, False, f"{type(exc).__name__}: {exc}")

    return CapturedKernel(graph.replay, True)
