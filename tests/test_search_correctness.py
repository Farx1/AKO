"""Correctness gate (ROADMAP.md M5).

An incorrect kernel must be impossible to treat as a valid trial. These tests
pin the ladder, the overflow rule, and the near-miss control that keeps
``ACC_FP16``'s widened tolerance honest.
"""

from __future__ import annotations

import math

import torch

from ako.search.builders import drop_last_k_block, torch_eager
from ako.search.correctness import (
    TARGET_ABS_C,
    CheckResult,
    check,
    check_case,
    input_scale,
    magnitude_ladder,
    make_inputs,
    oracle,
)
from ako.search.db import TrialDB, _OPTIONAL
from ako.search.measure import gate_then_measure
from ako.search.spec import Candidate, DeviceCaps, Problem

CAPS = DeviceCaps(
    name="cpu",
    cc=(0, 0),
    sm_count=0,
    smem_per_block=0,
    regs_per_sm=0,
    l2_bytes=0,
    total_mem_gb=0.0,
    torch_version="test",
    triton_version=None,
    driver=None,
)


def _prob(**kwargs: object) -> Problem:
    defaults: dict[str, object] = dict(m=8, n=8, k=32, dtype="fp32", act="relu", bias=True)
    defaults.update(kwargs)
    return Problem(**defaults)  # type: ignore[arg-type]


def _cand() -> Candidate:
    return Candidate("mm+bias+act", ("torch_eager",), ((),))


def test_magnitude_ladder_is_3_seeds_times_3_targets_plus_zeros() -> None:
    cases = magnitude_ladder(_prob())
    assert len(cases) == 3 * 3 + 1
    assert cases[-1]["zeros"] is True
    targets = {c["target_abs_c"] for c in cases if not c["zeros"]}
    assert targets == set(TARGET_ABS_C)


def test_input_scale_shrinks_with_k_so_4096_does_not_overflow_fp16() -> None:
    small = input_scale(_prob(k=64, dtype="fp16"), 1e3)
    large = input_scale(_prob(k=4096, dtype="fp16"), 1e3)
    assert large < small
    a, b, bias = make_inputs(_prob(m=4, n=4, k=4096, dtype="fp16"), scale=large, seed=0)
    ref = oracle(_prob(m=4, n=4, k=4096, dtype="fp16"), a, b, bias)
    assert torch.isfinite(ref).all()


def test_correct_eager_kernel_is_accepted() -> None:
    prob = _prob()
    result = check(torch_eager(prob), prob, device="cpu")
    assert result.verdict == "ok"
    assert result.max_abs_err is not None
    assert result.budget is not None
    assert result.max_abs_err <= result.budget + 1e-12
    assert any(c.zeros for c in result.magnitudes)


def test_wrong_kernel_is_rejected() -> None:
    prob = _prob()

    def wrong(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(a.shape[0], b.shape[1], dtype=a.dtype, device=a.device)

    result = check(wrong, prob, device="cpu")
    assert result.verdict == "incorrect"
    assert result.max_abs_err is not None and result.max_abs_err > 0


def test_nan_kernel_yields_nan_not_an_exception() -> None:
    prob = _prob()

    def nans(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return torch.full((a.shape[0], b.shape[1]), float("nan"), dtype=a.dtype)

    result = check(nans, prob, device="cpu")
    assert result.verdict == "nan"


def test_oracle_overflow_is_not_a_blacklist_verdict(tmp_path) -> None:
    """A ladder step whose fp32 oracle is non-finite is about the ladder."""
    prob = _prob(m=2, n=2, k=2, dtype="fp16")
    a, b, bias = make_inputs(prob, scale=1.0, seed=0)
    huge = 1.0e30
    case = check_case(
        torch_eager(prob),
        prob,
        a=a.float() * huge,
        b=b.float() * huge,
        bias=None if bias is None else bias.float() * huge,
        target_abs_c=1e30,
        scale=huge,
        zeros=False,
        seed=0,
        tol_scale=1.0,
    )
    assert case.verdict == "oracle_overflow"

    db = TrialDB(tmp_path / "t.sqlite")
    sweep = db.begin_sweep(prob, CAPS, 1.0)
    db.record(sweep, prob, _cand(), CAPS, verdict="oracle_overflow", reason="ladder")
    db.end_sweep(sweep, drift_pct=0.0, status="ok")
    assert _cand().hash() not in db.blacklist(CAPS, "fp16", CAPS.toolchain_key)
    db.close()


def test_near_miss_dropped_k_block_is_rejected() -> None:
    """tol_scale = sqrt(K)/8 must still catch a kernel that drops the last K."""
    k = 64
    prob = _prob(m=16, n=16, k=k, dtype="fp32", act="none")
    tol = math.sqrt(k) / 8.0
    result = check(drop_last_k_block(prob, block_k=8), prob, device="cpu", tol_scale=tol)
    assert result.verdict == "incorrect", (
        f"near-miss passed at tol_scale={tol}: the ACC_FP16 budget is too loose"
    )


def test_stochastic_gate_takes_the_pessimistic_run() -> None:
    prob = _prob(act="none")
    calls = {"n": 0}

    def flaky(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        calls["n"] += 1
        out = torch_eager(prob)(a, b, bias)
        if calls["n"] == 4:
            return out + 999.0
        return out

    result = check(flaky, prob, device="cpu", stochastic=True, gate_runs=2)
    assert result.gate_runs == 2
    assert result.stochastic is True
    assert any(c.verdict == "incorrect" for c in result.magnitudes)
    assert result.verdict == "incorrect"


def test_gate_then_measure_does_not_time_a_failure() -> None:
    prob = _prob()
    a, b, bias = make_inputs(prob, scale=0.1, seed=0)

    def wrong(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(x.shape[0], y.shape[1], dtype=x.dtype)

    gate, bench = gate_then_measure(wrong, prob, (a, b, bias), device="cpu")
    assert gate.verdict == "incorrect"
    assert bench is None


def test_gate_then_measure_times_a_pass() -> None:
    prob = _prob()
    a, b, bias = make_inputs(prob, scale=0.1, seed=0)
    gate, bench = gate_then_measure(torch_eager(prob), prob, (a, b, bias), device="cpu")
    assert gate.verdict == "ok"
    assert bench is not None
    assert bench.latency_p50_ms >= 0


def test_check_result_record_keys_match_trial_schema() -> None:
    payload = CheckResult(verdict="ok", max_abs_err=1e-6, budget=1e-3).to_record()
    assert set(payload) <= set(_OPTIONAL)
