"""Adaptive kernel selection with benchmark-driven caching.

Implements Algorithm 1 of the AKO paper:

1. look up ``(op, hardware, shape, dtype)`` in the persistent cache;
2. on a miss, validate every candidate numerically against the baseline;
3. benchmark the survivors and keep the fastest median latency;
4. cache the decision together with the metrics that justified it.

The baseline kernel is always part of the candidate set, so the selected kernel
can never be slower than the baseline (paper §13, non-regression).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch

from ako.benchmark.bench_op import bench_op
from ako.dispatcher.cache import KernelCache
from ako.dispatcher.hardware import get_device_info
from ako.dispatcher.registry import KernelEntry, KernelFn, KernelRegistry, get_registry
from ako.validation.numerical import error_budget, error_in_ulps, max_abs_error_any

#: Backends that count as *baselines* rather than candidates: everything a user
#: could already run without AKO. The speedup worth publishing is the gain over
#: the fastest of these, not over the unfused reference — beating an unfused
#: reference mostly measures the intermediates it materializes.
BASELINE_BACKENDS = frozenset({"pytorch", "torch_native", "torch_compile"})


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def shape_key(tensors: tuple[torch.Tensor, ...], extra: str | None = None) -> str:
    """Build the cache key for a call: shapes + dtype (+ an optional variant tag)."""
    if not tensors:
        raise ValueError("At least one tensor is required to build a shape key")
    parts = [str(tuple(tensors[0].shape)), str(tensors[0].dtype).replace("torch.", "")]
    parts.extend(str(tuple(t.shape)) for t in tensors[1:])
    if extra:
        parts.append(extra)
    return "_".join(parts)


@dataclass(frozen=True)
class SelectionResult:
    """Outcome of one autotuning pass, mirrored into the cache."""

    op_name: str
    entry: KernelEntry
    shape_key: str
    #: How the kernel was chosen: ``cache``, ``benchmark``, ``default``
    #: (autotuning off or nothing to compare against) or ``forced``.
    source: str
    latency_p50_ms: float | None = None
    baseline_ms: float | None = None
    max_error: float | None = None
    candidates_evaluated: int = 0
    rejected: tuple[str, ...] = ()
    best_baseline_ms: float | None = None
    best_baseline_kernel: str | None = None

    @property
    def from_cache(self) -> bool:
        return self.source == "cache"

    @property
    def speedup(self) -> float | None:
        """Gain over the unfused reference. Kept for continuity; not quotable."""
        if self.baseline_ms and self.latency_p50_ms:
            return self.baseline_ms / self.latency_p50_ms
        return None

    @property
    def speedup_vs_best_baseline(self) -> float | None:
        """Gain over the fastest baseline available — the number to publish."""
        if self.best_baseline_ms and self.latency_p50_ms:
            return self.best_baseline_ms / self.latency_p50_ms
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.op_name,
            "selected_kernel": self.entry.kernel_name,
            "backend": self.entry.backend,
            "shape_key": self.shape_key,
            "source": self.source,
            "from_cache": self.from_cache,
            "latency_p50_ms": self.latency_p50_ms,
            "baseline_ms": self.baseline_ms,
            "speedup": round(self.speedup, 3) if self.speedup else None,
            "best_baseline_ms": self.best_baseline_ms,
            "best_baseline_kernel": self.best_baseline_kernel,
            "speedup_vs_best_baseline": (
                round(self.speedup_vs_best_baseline, 3) if self.speedup_vs_best_baseline else None
            ),
            "max_error": self.max_error,
            "candidates_evaluated": self.candidates_evaluated,
            "rejected": list(self.rejected),
        }


class KernelSelector:
    """Select the best kernel for an operation given device, shape, and dtype."""

    def __init__(
        self,
        registry: KernelRegistry | None = None,
        cache: KernelCache | None = None,
        *,
        auto_tune: bool | None = None,
        error_threshold: float | None = None,
        min_speedup: float = 1.0,
        warmup: int = 5,
        repeats: int = 20,
        allow_cpu_tuning: bool = False,
    ) -> None:
        self.registry = get_registry() if registry is None else registry
        # NB: an empty KernelCache is falsy (it defines __len__), so this must
        # be an explicit None check or a fresh cache would shadow the caller's.
        self.cache = KernelCache() if cache is None else cache
        self.auto_tune = _env_flag("AKO_AUTOTUNE", True) if auto_tune is None else auto_tune
        self.allow_cpu_tuning = allow_cpu_tuning
        # None means "derive the budget from the dtype and the output magnitude".
        # A fixed absolute threshold cannot work across dtypes: in bf16 the gap
        # between representable values around 5.0 is already 0.031, so a 1e-2
        # threshold would reject correct kernels and silently keep the baseline.
        self.error_threshold = error_threshold
        self.min_speedup = min_speedup
        self.warmup = warmup
        self.repeats = repeats
        self._memo: dict[tuple[Any, ...], KernelFn] = {}

    # -- public API ----------------------------------------------------
    def select(
        self,
        op_name: str,
        x: torch.Tensor,
        *args: torch.Tensor,
        forced: str | None = None,
        extra_key: str | None = None,
        call_kwargs: dict[str, Any] | None = None,
    ) -> KernelFn:
        """Return the kernel function to call for this operation.

        This is the dispatch hot path, so a repeated call resolves through an
        in-process memo built from cheap hashables — no string key, no device
        probe, no JSON cache lookup. The paper's non-regression bound only
        holds if the dispatch overhead ``δ`` stays small next to the kernel.
        """
        memo_key = (
            op_name,
            forced,
            extra_key,
            x.dtype,
            x.shape,
            x.device.type,
            x.device.index,
            tuple(a.shape for a in args),
        )
        fn = self._memo.get(memo_key)
        if fn is not None:
            return fn

        fn = self.select_entry(
            op_name,
            x,
            *args,
            forced=forced,
            extra_key=extra_key,
            call_kwargs=call_kwargs,
        ).entry.fn
        self._memo[memo_key] = fn
        return fn

    def clear_memo(self) -> None:
        """Drop the in-process fast path (after registering new kernels)."""
        self._memo.clear()

    def select_entry(
        self,
        op_name: str,
        x: torch.Tensor,
        *args: torch.Tensor,
        forced: str | None = None,
        extra_key: str | None = None,
        call_kwargs: dict[str, Any] | None = None,
    ) -> SelectionResult:
        """Same as :meth:`select`, but returns the full decision record."""
        kernels = self.registry.list_kernels(op_name)
        if not kernels:
            raise ValueError(f"No kernels registered for op '{op_name}'")

        key = shape_key((x, *args), extra_key)

        if forced:
            entry = self.registry.get(op_name, forced)
            if entry is None:
                available = ", ".join(k.kernel_name for k in kernels)
                raise ValueError(
                    f"Unknown kernel '{forced}' for op '{op_name}' (available: {available})"
                )
            return SelectionResult(op_name, entry, key, source="forced")

        device_info = get_device_info(x.device)
        record = self.cache.get_record(device_info.cache_key, op_name, key)
        if record:
            entry = self.registry.get(op_name, record["kernel"])
            if entry is not None:
                return SelectionResult(
                    op_name,
                    entry,
                    key,
                    source="cache",
                    latency_p50_ms=record.get("latency_p50_ms"),
                    baseline_ms=record.get("baseline_ms"),
                    max_error=record.get("max_error"),
                    best_baseline_ms=record.get("best_baseline_ms"),
                    best_baseline_kernel=record.get("best_baseline_kernel"),
                )

        # No autotuning: fall back to the highest-priority registered kernel.
        tunable_device = x.device.type == "cuda" or self.allow_cpu_tuning
        if not self.auto_tune or not tunable_device or len(kernels) == 1:
            default = self.registry.default(op_name)
            assert default is not None
            return SelectionResult(op_name, default, key, source="default")

        return self._benchmark_and_select(
            op_name,
            kernels,
            (x, *args),
            key=key,
            call_kwargs=call_kwargs or {},
            gpu_key=device_info.cache_key,
        )

    # -- autotuning ----------------------------------------------------
    def _benchmark_and_select(
        self,
        op_name: str,
        kernels: list[KernelEntry],
        tensors: tuple[torch.Tensor, ...],
        *,
        key: str,
        call_kwargs: dict[str, Any],
        gpu_key: str,
    ) -> SelectionResult:
        baseline = next((k for k in kernels if k.backend == "pytorch"), kernels[-1])
        ref_out = baseline.fn(*tensors, **call_kwargs)

        base_result = bench_op(
            baseline.fn,
            tensors,
            warmup=self.warmup,
            repeats=self.repeats,
            kernel_name=baseline.kernel_name,
            call_kwargs=call_kwargs,
        )
        baseline_ms = base_result.latency_p50_ms

        budget = error_budget(ref_out) if self.error_threshold is None else self.error_threshold

        best_entry = baseline
        best_latency = baseline_ms
        best_error = 0.0
        evaluated = 1
        rejected: list[str] = []
        # Every baseline that survives, so the published speedup is the gain over
        # the fastest thing the user could already run — not over the reference.
        baseline_latencies: dict[str, float] = {baseline.kernel_name: baseline_ms}

        for entry in kernels:
            if entry.kernel_name == baseline.kernel_name:
                continue
            try:
                candidate_out = entry.fn(*tensors, **call_kwargs)
                error = max_abs_error_any(ref_out, candidate_out)
                if error > budget:
                    rejected.append(
                        f"{entry.kernel_name}: max_error={error:.3e} > budget={budget:.3e} "
                        f"({error_in_ulps(ref_out, candidate_out):.1f} ulps)"
                    )
                    continue
                result = bench_op(
                    entry.fn,
                    tensors,
                    warmup=self.warmup,
                    repeats=self.repeats,
                    kernel_name=entry.kernel_name,
                    call_kwargs=call_kwargs,
                )
            except Exception as exc:  # a broken candidate must never break dispatch
                rejected.append(f"{entry.kernel_name}: {type(exc).__name__}: {exc}")
                continue

            evaluated += 1
            latency = result.latency_p50_ms
            if entry.backend in BASELINE_BACKENDS:
                baseline_latencies[entry.kernel_name] = latency
            # Require a real win over the *baseline* before leaving the safe path.
            if latency < best_latency and baseline_ms >= latency * self.min_speedup:
                best_entry, best_latency, best_error = entry, latency, error

        best_baseline_kernel = min(baseline_latencies, key=baseline_latencies.__getitem__)
        best_baseline_ms = baseline_latencies[best_baseline_kernel]

        self.cache.set(
            gpu_key,
            op_name,
            key,
            best_entry.kernel_name,
            latency_p50_ms=round(best_latency, 4),
            baseline_ms=round(baseline_ms, 4),
            speedup=round(baseline_ms / best_latency, 3) if best_latency > 0 else None,
            best_baseline_ms=round(best_baseline_ms, 4),
            best_baseline_kernel=best_baseline_kernel,
            speedup_vs_best_baseline=(
                round(best_baseline_ms / best_latency, 3) if best_latency > 0 else None
            ),
            max_error=best_error,
            error_budget=budget,
            backend=best_entry.backend,
        )

        return SelectionResult(
            op_name,
            best_entry,
            key,
            source="benchmark",
            latency_p50_ms=best_latency,
            baseline_ms=baseline_ms,
            max_error=best_error,
            candidates_evaluated=evaluated,
            rejected=tuple(rejected),
            best_baseline_ms=best_baseline_ms,
            best_baseline_kernel=best_baseline_kernel,
        )


_global_selector: KernelSelector | None = None


def get_selector() -> KernelSelector:
    """Process-wide selector shared by every :mod:`ako.ops` entry point."""
    global _global_selector
    if _global_selector is None:
        _global_selector = KernelSelector()
    return _global_selector


def reset_selector() -> None:
    """Drop the process-wide selector (used by tests and CLI overrides)."""
    global _global_selector
    _global_selector = None
