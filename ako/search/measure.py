"""Measure a candidate only after the correctness gate has passed.

ROADMAP.md M5. Timing a failing verdict is a bug: :meth:`TrialDB.record`
rejects it. Decode rows are produced both with and without CUDA-graph capture
(:mod:`ako.benchmark.graph`) so the launch-overhead claim can be tested.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from ako.benchmark.bench_op import BenchResult, bench_op, compare_paired
from ako.benchmark.env import sample_env
from ako.search.correctness import CheckResult, KernelFn, check
from ako.search.spec import Problem


def single(
    fn: KernelFn,
    args: tuple[Any, ...],
    *,
    kernel_name: str = "candidate",
    warmup: int = 5,
    repeats: int = 20,
    call_kwargs: dict[str, Any] | None = None,
    graph: bool = False,
) -> BenchResult:
    """One timed pass. Callers must have a passing :func:`check` in hand."""
    return bench_op(
        fn,
        args,
        warmup=warmup,
        repeats=repeats,
        kernel_name=kernel_name,
        call_kwargs=call_kwargs,
        graph=graph,
    )


def gate_then_measure(
    fn: KernelFn,
    prob: Problem,
    args: tuple[Any, ...],
    *,
    device: torch.device | str = "cpu",
    tol_scale: float = 1.0,
    stochastic: bool = False,
    kernel_name: str = "candidate",
    graph: bool = False,
) -> tuple[CheckResult, BenchResult | None]:
    """Gate first. A failing verdict returns ``None`` instead of a timing."""
    result = check(
        fn, prob, device=device, tol_scale=tol_scale, stochastic=stochastic
    )
    if result.verdict != "ok":
        return result, None
    return result, single(fn, args, kernel_name=kernel_name, graph=graph)


def env_snapshot() -> dict[str, Any]:
    """Clock/temp/power at measurement time. Missing sensors stay ``None``."""
    return sample_env()


def paired(
    incumbent: Callable[..., torch.Tensor],
    candidate: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    **kwargs: Any,
) -> dict[str, Any]:
    """Interleaved A/B. ``speedup_median`` > 1 means the candidate is faster."""
    return compare_paired(incumbent, candidate, args, **kwargs)
