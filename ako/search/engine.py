"""Budgeted search: enumerate, filter, gate, then time. Never the reverse.

ROADMAP.md M6. The first go/no-go is a pinned A/B (``scripts/sweep.py --ab``),
not a budgeted sweep: sampling an empty DB can miss both tiles and produce a
null that is fully explained by enumeration order.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from ako.benchmark.bench_op import BenchResult
from ako.search.correctness import CheckResult
from ako.search.db import TrialDB
from ako.search.rank import order
from ako.search.space import enumerate_candidates
from ako.search.spec import Candidate, DeviceCaps, Problem

CheckFn = Callable[[Candidate], CheckResult]
TimeFn = Callable[[Candidate], BenchResult]
CompareFn = Callable[[Candidate, Candidate], dict[str, Any]]


@dataclass
class SearchOutcome:
    sweep_id: str
    incumbent: Candidate | None
    evaluated: int
    skipped_blacklist: int
    infeasible: int
    rejected: int
    status: str
    elapsed_s: float
    log: tuple[str, ...] = ()


def search(
    db: TrialDB,
    prob: Problem,
    caps: DeviceCaps,
    *,
    budget_s: float,
    prior: str = "none",
    check_candidate: CheckFn,
    time_candidate: TimeFn,
    compare_to_incumbent: CompareFn | None = None,
    blocks: Any = None,
) -> SearchOutcome:
    """Run the loop. A failing gate is recorded and never timed.

    ``check_candidate`` / ``time_candidate`` are injected so the CPU tests can
    stub the GPU. Production callers pass closures over :func:`ako.search.builders`
    and :func:`ako.search.measure.gate_then_measure`.
    """
    sweep_id = db.begin_sweep(prob, caps, budget_s=budget_s)
    banned = db.blacklist(caps, prob.dtype, caps.toolchain_key)
    incumbent: Candidate | None = None
    evaluated = skipped = infeasible = rejected = 0
    notes: list[str] = []
    t0 = time.monotonic()
    status = "ok"

    pending = [
        (cand, reason)
        for cand, reason in enumerate_candidates(prob, caps, blocks=blocks)
    ]
    pending = order(pending, db, prob, caps, prior=prior)

    for cand, reason in pending:
        if time.monotonic() - t0 >= budget_s:
            status = "budget_exhausted"
            notes.append("budget_exhausted")
            break
        if cand.hash() in banned:
            skipped += 1
            continue
        if reason is not None:
            db.record(sweep_id, prob, cand, caps, verdict="infeasible", reason=reason)
            infeasible += 1
            continue
        gate = check_candidate(cand)
        if gate.verdict != "ok":
            db.record(sweep_id, prob, cand, caps, verdict=gate.verdict, **gate.to_record())
            rejected += 1
            continue
        timed = time_candidate(cand)
        evaluated += 1
        cols: dict[str, Any] = {
            **gate.to_record(),
            "wall_p50_ms": timed.latency_p50_ms,
            "wall_p95_ms": timed.latency_p95_ms,
            "gpu_p50_ms": timed.gpu_p50_ms,
            "cv": timed.cv,
            "iqr_ms": timed.iqr_ms,
            "inner_iters": timed.inner_iters,
            "l2_state": timed.l2_state,
            "graph_captured": timed.graph_captured,
            "peak_mem_mb": timed.peak_memory_mb,
        }
        if incumbent is None:
            incumbent = cand
            cols["incumbent_hash"] = cand.hash()
            cols["incumbent_p50_ms"] = timed.latency_p50_ms
            cols["speedup_vs_incumbent"] = 1.0
            cols["significant"] = 1
        elif compare_to_incumbent is not None:
            cmp = compare_to_incumbent(incumbent, cand)
            cols["incumbent_hash"] = incumbent.hash()
            cols["incumbent_p50_ms"] = cmp.get("a_median_ms")
            cols["speedup_vs_incumbent"] = cmp.get("speedup_median")
            cols["ci_lo"] = (cmp.get("speedup_ci95") or [None, None])[0]
            cols["ci_hi"] = (cmp.get("speedup_ci95") or [None, None])[1]
            cols["significant"] = int(bool(cmp.get("significant")))
            if cmp.get("significant") and (cmp.get("speedup_median") or 0) > 1.0:
                incumbent = cand
        db.record(sweep_id, prob, cand, caps, verdict="ok", **cols)

    elapsed = time.monotonic() - t0
    db.end_sweep(sweep_id, drift_pct=0.0, status=status)
    return SearchOutcome(
        sweep_id=sweep_id,
        incumbent=incumbent,
        evaluated=evaluated,
        skipped_blacklist=skipped,
        infeasible=infeasible,
        rejected=rejected,
        status=status,
        elapsed_s=elapsed,
        log=tuple(notes),
    )
