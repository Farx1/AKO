"""Lightweight PyTorch profiler wrapper (optional deep profiling)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import torch


@contextmanager
def profile_cuda(
    activities: list[Any] | None = None,
) -> Generator[torch.profiler.profile, None, None]:
    """Context manager wrapping ``torch.profiler.profile`` for CUDA ops."""
    if activities is None:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        yield prof
