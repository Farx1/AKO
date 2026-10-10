"""CPU tests for the search loop (ROADMAP.md M6).

A stub registry and a fake timer: ``search()`` never benchmarks a candidate that
failed the gate, always records infeasible rows, respects ``budget_s``, and
leaves the incumbent unchanged when the paired comparison is not significant.
"""

from __future__ import annotations

import time

from ako.benchmark.bench_op import BenchResult
from ako.search.blocks import Block
from ako.search.correctness import CheckResult
from ako.search.db import TrialDB
from ako.search.engine import search
from ako.search.rank import order
from ako.search.spec import Candidate, DeviceCaps, Problem
from ako.search.space import enumerate_candidates


CAPS = DeviceCaps(
    name="stub",
    cc=(8, 6),
    sm_count=40,
    smem_per_block=101376,
    regs_per_sm=65536,
    l2_bytes=4 * 1024**2,
    total_mem_gb=8.0,
    torch_version="test",
    triton_version=None,
)


def _prob() -> Problem:
    return Problem(m=32, n=32, k=32, dtype="fp32", act="relu", bias=True)


def _result(ms: float, name: str = "k") -> BenchResult:
    return BenchResult(
        kernel_name=name,
        shape=(32, 32),
        dtype="float32",
        device="cpu",
        latency_p50_ms=ms,
        latency_p95_ms=ms,
        latency_mean_ms=ms,
        latency_std_ms=0.0,
        samples_ms=[ms],
    )


def _ok(_cand: Candidate) -> CheckResult:
    return CheckResult(verdict="ok", max_abs_err=0.0, budget=1.0)


def _blocks() -> list[Block]:
    def always(prob, caps):
        return True

    def no_params(prob, caps):
        yield {}

    def ok_feasible(params, prob, caps):
        return None

    def bad_feasible(params, prob, caps):
        return "smem"

    good = Block("stub_ok", frozenset({"matmul", "bias", "act"}), always, no_params, ok_feasible)
    bad = Block("stub_bad", frozenset({"matmul", "bias", "act"}), always, no_params, bad_feasible)
    return [good, bad]


def test_infeasible_rows_are_recorded_and_never_timed(tmp_path) -> None:
    db = TrialDB(tmp_path / "t.sqlite")
    timed: list[str] = []

    def time_candidate(cand: Candidate) -> BenchResult:
        timed.append(cand.hash())
        return _result(1.0)

    out = search(
        db,
        _prob(),
        CAPS,
        budget_s=30,
        check_candidate=_ok,
        time_candidate=time_candidate,
        blocks=_blocks(),
    )
    assert out.infeasible >= 1
    n_ok = db.conn.execute("SELECT COUNT(*) FROM trial WHERE verdict='ok'").fetchone()[0]
    n_inf = db.conn.execute("SELECT COUNT(*) FROM trial WHERE verdict='infeasible'").fetchone()[0]
    assert n_inf >= 1
    assert n_ok == len(timed)
    assert all(
        r["verdict"] != "ok" or r["wall_p50_ms"] is not None
        for r in db.conn.execute("SELECT verdict, wall_p50_ms FROM trial")
    )


def test_failing_gate_is_never_benchmarked(tmp_path) -> None:
    db = TrialDB(tmp_path / "t.sqlite")
    timed: list[str] = []

    def check_candidate(cand: Candidate) -> CheckResult:
        if cand.blocks[0] == "stub_ok":
            return CheckResult(verdict="incorrect", max_abs_err=9.0, budget=1e-3, reason="wrong")
        return CheckResult(verdict="ok", max_abs_err=0.0, budget=1.0)

    def time_candidate(cand: Candidate) -> BenchResult:
        timed.append(cand.hash())
        return _result(1.0)

    search(
        db,
        _prob(),
        CAPS,
        budget_s=30,
        check_candidate=check_candidate,
        time_candidate=time_candidate,
        blocks=_blocks(),
    )
    assert timed == []
    n_bad = db.conn.execute("SELECT COUNT(*) FROM trial WHERE verdict='incorrect'").fetchone()[0]
    assert n_bad >= 1
    assert db.conn.execute("SELECT COUNT(*) FROM trial WHERE wall_p50_ms IS NOT NULL").fetchone()[0] == 0


def test_budget_stops_the_loop(tmp_path) -> None:
    db = TrialDB(tmp_path / "t.sqlite")

    def slow_time(cand: Candidate) -> BenchResult:
        time.sleep(0.05)
        return _result(1.0)

    out = search(
        db,
        _prob(),
        CAPS,
        budget_s=0.01,
        check_candidate=_ok,
        time_candidate=slow_time,
        blocks=_blocks(),
    )
    assert out.status == "budget_exhausted"


def test_incumbent_unchanged_when_comparison_is_not_significant(tmp_path) -> None:
    db = TrialDB(tmp_path / "t.sqlite")
    seen: list[str] = []

    def time_candidate(cand: Candidate) -> BenchResult:
        seen.append(cand.hash())
        return _result(1.0 if len(seen) == 1 else 0.5)

    def compare(incumbent: Candidate, cand: Candidate) -> dict:
        return {
            "a_median_ms": 1.0,
            "speedup_median": 2.0,
            "speedup_ci95": [0.9, 2.1],
            "significant": False,
        }

    only_ok = [b for b in _blocks() if b.name == "stub_ok"]
    # Duplicate the good block under a second name so there are two legal candidates.
    twin = Block(
        "stub_ok2",
        frozenset({"matmul", "bias", "act"}),
        lambda p, c: True,
        lambda p, c: iter([{}]),
        lambda params, p, c: None,
    )
    out = search(
        db,
        _prob(),
        CAPS,
        budget_s=30,
        check_candidate=_ok,
        time_candidate=time_candidate,
        compare_to_incumbent=compare,
        blocks=[only_ok[0], twin],
    )
    assert out.evaluated == 2
    assert out.incumbent is not None
    assert out.incumbent.hash() == seen[0]


def test_blacklist_skips_known_bad_hashes(tmp_path) -> None:
    db = TrialDB(tmp_path / "t.sqlite")
    prob = _prob()
    sweep = db.begin_sweep(prob, CAPS, budget_s=1)
    # Record a legal candidate as incorrect so the next search skips it.
    cand = next(c for c, r in enumerate_candidates(prob, CAPS, blocks=_blocks()) if r is None)
    db.record(sweep, prob, cand, CAPS, verdict="incorrect", max_abs_err=1.0, budget=0.1)
    db.end_sweep(sweep, drift_pct=0.0, status="ok")

    timed: list[str] = []
    out = search(
        db,
        prob,
        CAPS,
        budget_s=30,
        check_candidate=_ok,
        time_candidate=lambda c: timed.append(c.hash()) or _result(1.0),
        blocks=_blocks(),
    )
    assert cand.hash() not in timed
    assert out.skipped_blacklist >= 1


def test_order_none_preserves_enumeration() -> None:
    pending = [
        (Candidate("mm", ("a",), ((),)), None),
        (Candidate("mm", ("b",), ((),)), "smem"),
    ]
    assert order(pending, TrialDB(":memory:"), _prob(), CAPS, prior="none") == pending
