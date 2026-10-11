"""Single-operation GPU/CPU benchmarking."""

from __future__ import annotations

import random
import statistics
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Sequence

import torch


DTYPE_ALIASES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def parse_dtypes(raw: Sequence[str]) -> list[torch.dtype]:
    """Map CLI dtype names (``fp16``/``bf16``/``fp32``) to torch dtypes."""
    unknown = [d for d in raw if d not in DTYPE_ALIASES]
    if unknown:
        raise ValueError(
            f"Unknown dtype(s) {unknown}; expected one of {sorted(DTYPE_ALIASES)}"
        )
    return [DTYPE_ALIASES[d] for d in raw]


@dataclass(frozen=True)
class BenchResult:
    """Latency statistics for one kernel invocation.

    ``latency_*`` is amortized wall-clock per call — what a caller actually pays,
    launch overhead included. ``gpu_p50_ms`` is device time measured with CUDA
    events. Their difference is the per-call launch overhead, which is what makes
    a fast kernel lose at small shapes.
    """

    kernel_name: str
    shape: tuple[int, ...]
    dtype: str
    device: str
    latency_p50_ms: float
    latency_p95_ms: float
    latency_mean_ms: float
    latency_std_ms: float
    samples_ms: list[float]
    gpu_p50_ms: float | None = None
    inner_iters: int = 1
    peak_memory_mb: float | None = None
    #: p50 of each independent run (see ``runs`` in :func:`bench_op`).
    run_p50s: tuple[float, ...] = ()
    #: ``flushed``, ``off``, or ``n/a`` when the device has no L2 to flush. A row
    #: measured hot is not comparable to one measured cold, so the mode is data.
    l2_state: str = "n/a"
    #: Per-call flush cost already subtracted from the latencies above.
    l2_flush_ms: float = 0.0
    #: 1 when the timed inner loop replayed a CUDA graph, 0 otherwise.
    graph_captured: int = 0

    @property
    def launch_overhead_ms(self) -> float | None:
        """Wall-clock cost that is not device time, or None without device timing.

        Not clamped: a negative value means the two clocks disagree, which is a
        measurement problem to see, not a zero overhead.
        """
        if self.gpu_p50_ms is None:
            return None
        return self.latency_p50_ms - self.gpu_p50_ms

    @property
    def _spread_samples(self) -> list[float]:
        """Values the spread statistics describe: run p50s, or raw samples if one run."""
        return list(self.run_p50s) if len(self.run_p50s) > 1 else list(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self._spread_samples or [0.0])

    @property
    def iqr_ms(self) -> float:
        vals = self._spread_samples
        return _percentile(vals, 0.75) - _percentile(vals, 0.25)

    @property
    def cv(self) -> float:
        """Coefficient of variation (stdev / mean); 0.0 when a single value."""
        vals = self._spread_samples
        mean = statistics.mean(vals) if vals else 0.0
        if len(vals) < 2 or mean <= 0:
            return 0.0
        return statistics.stdev(vals) / mean

    @property
    def unstable(self) -> bool:
        """True when the run-to-run spread is too wide to compare against."""
        return self.cv > UNSTABLE_CV

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "kernel_name": self.kernel_name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "device": self.device,
            "latency_p50_ms": round(self.latency_p50_ms, 4),
            "latency_p95_ms": round(self.latency_p95_ms, 4),
            "latency_mean_ms": round(self.latency_mean_ms, 4),
            "latency_std_ms": round(self.latency_std_ms, 4),
            "inner_iters": self.inner_iters,
            "runs": max(len(self.run_p50s), 1),
            "median_ms": round(self.median_ms, 4),
            "p25_ms": round(_percentile(self._spread_samples, 0.25), 4),
            "p75_ms": round(_percentile(self._spread_samples, 0.75), 4),
            "iqr_ms": round(self.iqr_ms, 4),
            "cv": round(self.cv, 4),
            "unstable": self.unstable,
            "l2_state": self.l2_state,
            "graph_captured": self.graph_captured,
        }
        if self.l2_flush_ms:
            payload["l2_flush_ms"] = round(self.l2_flush_ms, 4)
        if self.gpu_p50_ms is not None:
            payload["gpu_p50_ms"] = round(self.gpu_p50_ms, 4)
            payload["launch_overhead_ms"] = round(self.latency_p50_ms - self.gpu_p50_ms, 4)
        if self.peak_memory_mb is not None:
            payload["peak_memory_mb"] = round(self.peak_memory_mb, 2)
        return payload


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]


#: Work per timed batch. Synchronizing around a single call costs ~0.1-0.3 ms of
#: instrumentation, which swamps a kernel that runs in tens of microseconds — and
#: produces impossible readings, like latency falling as the shape grows.
TARGET_BATCH_MS = 1.0
MAX_INNER_ITERS = 200

#: Above this coefficient of variation the machine was too noisy for the number to
#: be worth comparing; reported rather than hidden.
UNSTABLE_CV = 0.10

#: Enough writes to evict any consumer-card L2 (4 MB on a 3070).
L2_FLUSH_BYTES = 32 * 1024 * 1024


@lru_cache(maxsize=None)
def _flush_buffer(device: torch.device) -> torch.Tensor:
    """One 32 MiB scratch buffer per device, kept alive for the whole process."""
    return torch.empty(L2_FLUSH_BYTES, dtype=torch.int8, device=device)


@lru_cache(maxsize=None)
def _flush_cost_ms(device: torch.device) -> tuple[float, float]:
    """Wall and device cost of one flush, calibrated once so it can be subtracted.

    Measured through :func:`bench_op` itself (with the flush off, so this does not
    recurse) rather than through a second timing path.
    """
    buffer = _flush_buffer(device)
    result = bench_op(
        lambda b: b.zero_(), (buffer,),
        warmup=5, repeats=20, inner=10, flush_l2=False, kernel_name="l2_flush",
    )
    return result.latency_p50_ms, result.gpu_p50_ms or 0.0


def _calibrate_inner(
    fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], is_cuda: bool
) -> int:
    """Pick how many calls to time together so the batch lasts ~``TARGET_BATCH_MS``."""
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args, **kwargs)
    if is_cuda:
        torch.cuda.synchronize()
    single_ms = (time.perf_counter() - t0) * 1000
    if single_ms <= 0:
        return MAX_INNER_ITERS
    return max(1, min(MAX_INNER_ITERS, int(TARGET_BATCH_MS / single_ms)))


def bench_op(
    fn: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    *,
    warmup: int = 10,
    repeats: int = 50,
    kernel_name: str = "unknown",
    call_kwargs: dict[str, Any] | None = None,
    inner: int | None = None,
    runs: int = 1,
    flush_l2: bool = True,
    graph: bool = False,
) -> BenchResult:
    """Benchmark ``fn(*args, **call_kwargs)`` and return latency percentiles in ms.

    Calls are timed in batches rather than one at a time, so the measurement is
    not dominated by its own synchronization. On CUDA, device time is also
    recorded with events; the gap to wall-clock is the launch overhead.

    Args:
        fn: Callable to measure.
        args: Positional arguments; ``args[0]`` must be a tensor.
        warmup: Untimed calls before measuring.
        repeats: Number of timed batches.
        kernel_name: Label carried into the report.
        call_kwargs: Keyword arguments forwarded to ``fn``.
        inner: Calls per timed batch; calibrated automatically when ``None``.
        runs: Independent repetitions of the whole measurement. ``1`` (the default)
            keeps the historical single-shot behaviour; with more, the spread
            statistics (median/IQR/cv) describe run-to-run variability instead of
            batch-to-batch noise.
        flush_l2: Zero a 32 MiB buffer before **every** call, so each call starts
            on a cold L2. Not once per timed batch: batches are sized to
            ``TARGET_BATCH_MS``, so at 1024 cubed fp16 (~86 us) ``inner`` is ~11 and
            calls 2..11 would still run hot — on the very shape the flush exists
            for. The flush cost is calibrated once and subtracted. No-op off CUDA.
        graph: When True on CUDA, capture ``fn`` after warmup and replay the
            graph in the timed loop. Capture failure (allocation, compile modes)
            falls back to the uncaptured path and records ``graph_captured=0``.
            The L2 flush stays *outside* the graph: it is measurement hygiene,
            not part of the decode step being compared.

    Returns:
        Latency statistics, per single call.
    """
    x = args[0]
    assert isinstance(x, torch.Tensor)
    kwargs = call_kwargs or {}
    is_cuda = x.is_cuda

    flush = _flush_buffer(x.device) if flush_l2 and is_cuda else None
    l2_state = "flushed" if flush is not None else ("off" if is_cuda else "n/a")
    # ponytail: one calibrated cost subtracted from both wall and device time.
    # Ceiling: it assumes the flush does not overlap the kernel. Upgrade path if
    # that ever matters, time fn with and without the flush per shape instead.
    flush_wall_ms, flush_gpu_ms = _flush_cost_ms(x.device) if flush is not None else (0.0, 0.0)

    for _ in range(warmup):
        fn(*args, **kwargs)
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(x.device)

    timed: Callable[..., Any] = fn
    graph_captured = 0
    if graph and is_cuda:
        from ako.benchmark.graph import try_capture

        captured = try_capture(fn, args, kwargs)
        if captured.captured and captured.replay is not None:
            replay = captured.replay

            def timed(*_a: Any, **_k: Any) -> None:
                replay()

            graph_captured = 1

    if inner is None:
        inner = _calibrate_inner(timed, args, kwargs, is_cuda)

    samples: list[float] = []
    gpu_samples: list[float] = []
    run_p50s: list[float] = []

    for _ in range(max(1, runs)):
        run_samples: list[float] = []
        for _ in range(repeats):
            if is_cuda:
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                start.record()
                for _ in range(inner):
                    if flush is not None:
                        flush.zero_()
                    timed(*args, **kwargs)
                end.record()
                torch.cuda.synchronize()
                wall_ms = (time.perf_counter() - t0) * 1000
                gpu_samples.append(max(start.elapsed_time(end) / inner - flush_gpu_ms, 0.0))
            else:
                t0 = time.perf_counter()
                for _ in range(inner):
                    timed(*args, **kwargs)
                wall_ms = (time.perf_counter() - t0) * 1000
            run_samples.append(max(wall_ms / inner - flush_wall_ms, 0.0))
        run_p50s.append(_percentile(run_samples, 0.50))
        samples.extend(run_samples)

    peak_mb = None
    if is_cuda:
        peak_mb = torch.cuda.max_memory_allocated(x.device) / (1024**2)

    return BenchResult(
        kernel_name=kernel_name,
        shape=tuple(x.shape),
        dtype=str(x.dtype).replace("torch.", ""),
        device=str(x.device),
        latency_p50_ms=_percentile(samples, 0.50),
        latency_p95_ms=_percentile(samples, 0.95),
        latency_mean_ms=statistics.mean(samples),
        latency_std_ms=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        samples_ms=samples,
        gpu_p50_ms=_percentile(gpu_samples, 0.50) if gpu_samples else None,
        inner_iters=inner,
        peak_memory_mb=peak_mb,
        run_p50s=tuple(run_p50s),
        l2_state=l2_state,
        l2_flush_ms=flush_wall_ms,
        graph_captured=graph_captured,
    )


def bootstrap_ci(
    values: Sequence[float],
    *,
    seed: int = 0,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the median of ``values``.

    Stdlib only and seeded, so the interval is reproducible.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    pool = list(values)
    medians = [statistics.median(rng.choices(pool, k=len(pool))) for _ in range(resamples)]
    tail = (1.0 - confidence) / 2
    return (_percentile(medians, tail), _percentile(medians, 1.0 - tail))


def compare_paired(
    fn_a: Callable[..., torch.Tensor],
    fn_b: Callable[..., torch.Tensor],
    args: tuple[Any, ...],
    *,
    pairs: int = 11,
    seed: int = 0,
    warmup: int = 5,
    repeats: int = 20,
    call_kwargs: dict[str, Any] | None = None,
    inner: int | None = None,
    flush_l2: bool = True,
    graph: bool = False,
) -> dict[str, Any]:
    """Compare two kernels by measuring them in alternation (A, B, A, B, ...).

    Interleaving means both kernels see the same thermal and clock state, so the
    per-pair ratio cancels drift that blocked A-then-B measurement would bake in.

    Returns:
        ``speedup_median`` (time of ``fn_a`` over time of ``fn_b``, so > 1 means
        ``fn_b`` is faster), its bootstrap 95% interval, whether that interval
        excludes 1.0 (significant), and whether either side was unstable.
    """
    ratios: list[float] = []
    a_ms: list[float] = []
    b_ms: list[float] = []
    unstable = False
    captured = True
    for index in range(pairs):
        common = dict(
            warmup=warmup,
            repeats=repeats,
            call_kwargs=call_kwargs,
            inner=inner,
            flush_l2=flush_l2,
            graph=graph,
        )
        # Alternate who runs first. A fixed A-then-B order made a kernel look
        # faster than itself (median 0.79, CI still containing 1) because the
        # second call saw a warmer card.
        if index % 2 == 0:
            res_a = bench_op(fn_a, args, kernel_name="a", **common)
            res_b = bench_op(fn_b, args, kernel_name="b", **common)
        else:
            res_b = bench_op(fn_b, args, kernel_name="b", **common)
            res_a = bench_op(fn_a, args, kernel_name="a", **common)
        unstable = unstable or res_a.unstable or res_b.unstable
        captured = captured and bool(res_a.graph_captured) and bool(res_b.graph_captured)
        a_ms.append(res_a.latency_p50_ms)
        b_ms.append(res_b.latency_p50_ms)
        if res_b.latency_p50_ms > 0:
            ratios.append(res_a.latency_p50_ms / res_b.latency_p50_ms)
    low, high = bootstrap_ci(ratios, seed=seed)
    return {
        "pairs": len(ratios),
        "a_median_ms": round(statistics.median(a_ms), 4) if a_ms else 0.0,
        "b_median_ms": round(statistics.median(b_ms), 4) if b_ms else 0.0,
        "speedup_median": round(statistics.median(ratios), 4) if ratios else 0.0,
        "speedup_ci95": [round(low, 4), round(high, 4)],
        "significant": bool(ratios) and (low > 1.0 or high < 1.0),
        "unstable": unstable,
        "graph_captured": captured if graph else False,
    }


def bench_shapes(
    fn: Callable[..., torch.Tensor],
    shapes: Sequence[tuple[int, ...]],
    *,
    dtype: torch.dtype = torch.float16,
    device: str = "cuda",
    weight_fn: Callable[[int], torch.Tensor] | None = None,
    warmup: int = 10,
    repeats: int = 30,
    kernel_name: str = "unknown",
) -> list[BenchResult]:
    """Benchmark ``fn`` across multiple input shapes."""
    results: list[BenchResult] = []
    for shape in shapes:
        hidden = shape[-1]
        x = torch.randn(*shape, dtype=dtype, device=device)
        weight = weight_fn(hidden) if weight_fn else torch.ones(hidden, dtype=dtype, device=device)
        result = bench_op(fn, (x, weight), warmup=warmup, repeats=repeats, kernel_name=kernel_name)
        results.append(result)
    return results
