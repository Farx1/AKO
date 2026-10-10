"""Regression test helpers for kernel outputs across versions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_baseline(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def compare_reports(current: dict[str, Any], baseline: dict[str, Any], *, rtol: float = 0.1) -> list[str]:
    """Return list of regression warnings comparing latency reports."""
    warnings: list[str] = []
    base_map = {(tuple(c["shape"]), c["selected_kernel"]): c for c in baseline.get("comparisons", [])}
    for row in current.get("comparisons", []):
        key = (tuple(row["shape"]), row["selected_kernel"])
        if key not in base_map:
            continue
        prev = base_map[key]["candidate_ms"]
        curr = row["candidate_ms"]
        if prev > 0 and (curr - prev) / prev > rtol:
            warnings.append(f"Regression on {key}: {prev:.3f}ms -> {curr:.3f}ms")
    return warnings
