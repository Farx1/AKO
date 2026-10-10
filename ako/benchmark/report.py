"""Benchmark report aggregation and export."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ako.benchmark.bench_op import BenchResult
from ako.dispatcher.hardware import DeviceInfo


@dataclass
class BenchmarkReport:
    """Structured benchmark output for one operation."""

    operation: str
    device: DeviceInfo
    results: list[dict[str, Any]] = field(default_factory=list)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_result(self, result: BenchResult, **extra: Any) -> None:
        row = result.to_dict()
        row.update(extra)
        self.results.append(row)

    def add_comparison(
        self,
        *,
        shape: list[int],
        baseline_ms: float,
        candidate_ms: float,
        max_error: float,
        selected_kernel: str,
        dtype: str | None = None,
        baseline_kernel: str = "rmsnorm_ref",
        best_baseline_ms: float | None = None,
        best_baseline_kernel: str | None = None,
        **extra: Any,
    ) -> None:
        """Record a candidate against the reference, and against the best baseline.

        ``speedup`` is measured against the naive reference and kept for
        continuity. ``speedup_vs_best_baseline`` is the defensible number: the
        gain over the fastest non-candidate path available on this machine —
        typically PyTorch's own fused kernel rather than the unfused reference.
        """
        speedup = baseline_ms / candidate_ms if candidate_ms > 0 else 0.0
        row = {
            "shape": shape,
            "baseline_ms": round(baseline_ms, 4),
            "baseline_kernel": baseline_kernel,
            "candidate_ms": round(candidate_ms, 4),
            "speedup": round(speedup, 3),
            "max_error": max_error,
            "selected_kernel": selected_kernel,
        }
        if dtype is not None:
            row["dtype"] = dtype
        if best_baseline_ms is not None:
            row["best_baseline_ms"] = round(best_baseline_ms, 4)
            row["best_baseline_kernel"] = best_baseline_kernel
            row["speedup_vs_best_baseline"] = (
                round(best_baseline_ms / candidate_ms, 3) if candidate_ms > 0 else 0.0
            )
        row.update(extra)
        self.comparisons.append(row)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "gpu": self.device.device_name,
            "device": self.device.to_dict(),
            "created_at": self.created_at,
            "results": self.results,
            "comparisons": self.comparisons,
        }


def save_report(report: BenchmarkReport, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report.to_dict(), indent=2))
    return out
