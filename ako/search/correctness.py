"""Correctness gate: an incorrect kernel is never timed.

ROADMAP.md M5, ARCHITECTURE.md invariant 1. The gate runs a magnitude ladder
on the *output* (|C| ≈ {1e-2, 1, 1e3} plus all-zeros), compares against an
fp32 oracle, and records one result per ladder step so a single overflowing
step cannot collapse into a kernel failure.

A non-finite oracle is ``oracle_overflow``: a fact about the ladder, never
about the candidate, and ignored by :meth:`TrialDB.blacklist`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn.functional as F

from ako.search.spec import DType, Problem
from ako.validation.numerical import error_budget, error_in_ulps, max_abs_error

KernelFn = Callable[..., torch.Tensor]

SEEDS = (0, 1, 2)
#: Output-side targets. The input scale is derived from K so a fixed multiplier
#: cannot overflow fp16 at 4096³ or vanish into subnormals at the other end.
TARGET_ABS_C = (1e-2, 1.0, 1e3)
DTYPE_MAP: dict[DType, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}
STOCHASTIC_GATE_RUNS = 5


def apply_act(x: torch.Tensor, act: str) -> torch.Tensor:
    if act == "none":
        return x
    if act == "relu":
        return torch.relu(x)
    if act == "gelu_tanh":
        return F.gelu(x, approximate="tanh")
    if act == "silu":
        return F.silu(x)
    raise ValueError(f"unknown act {act!r}")


def oracle(
    prob: Problem,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """fp32 reference for ``matmul [+ bias] [+ act]``. Never the thing under test."""
    out = a.float() @ b.float()
    if prob.bias:
        if bias is None:
            raise ValueError("problem asks for bias but none was given")
        out = out + bias.float()
    return apply_act(out, prob.act)


def input_scale(prob: Problem, target_abs_c: float) -> float:
    """Scale for ``N(0, s)`` inputs so |C| tracks ``target_abs_c``.

    For zero-mean Gaussian factors, ``rms(C) ≈ s² √K``. Solving for ``s``
    gives ``s = (target / √K)^{1/2}``. Using a fixed input multiplier instead
    is what overflows fp16 at 4096³ (ROADMAP.md M5).
    """
    k = max(int(prob.k), 1)
    target = max(float(target_abs_c), 0.0)
    return math.sqrt(target / math.sqrt(k))


def make_inputs(
    prob: Problem,
    *,
    scale: float,
    seed: int,
    device: torch.device | str = "cpu",
    zeros: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    dtype = DTYPE_MAP[prob.dtype]
    device = torch.device(device)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    if zeros or scale == 0.0:
        a = torch.zeros(prob.m, prob.k, dtype=dtype)
        b = torch.zeros(prob.k, prob.n, dtype=dtype)
        bias = torch.zeros(prob.n, dtype=dtype) if prob.bias else None
    else:
        a = torch.randn(prob.m, prob.k, generator=gen, dtype=torch.float32) * scale
        b = torch.randn(prob.k, prob.n, generator=gen, dtype=torch.float32) * scale
        a, b = a.to(dtype), b.to(dtype)
        bias = None
        if prob.bias:
            bias = (
                torch.randn(prob.n, generator=gen, dtype=torch.float32) * scale
            ).to(dtype)
    a, b = a.to(device), b.to(device)
    if bias is not None:
        bias = bias.to(device)
    return a, b, bias


@dataclass
class MagnitudeCase:
    target_abs_c: float | None
    scale: float
    zeros: bool
    seed: int
    verdict: str
    max_abs_err: float | None = None
    err_ulps: float | None = None
    budget: float | None = None
    reason: str | None = None
    ref_abs_c: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_abs_c": self.target_abs_c,
            "scale": self.scale,
            "zeros": self.zeros,
            "seed": self.seed,
            "verdict": self.verdict,
            "max_abs_err": self.max_abs_err,
            "err_ulps": self.err_ulps,
            "budget": self.budget,
            "reason": self.reason,
            "ref_abs_c": self.ref_abs_c,
        }


@dataclass
class CheckResult:
    verdict: str
    max_abs_err: float | None = None
    err_ulps: float | None = None
    budget: float | None = None
    tol_scale: float = 1.0
    seeds: tuple[int, ...] = SEEDS
    gate_runs: int = 1
    stochastic: bool = False
    magnitudes: list[MagnitudeCase] = field(default_factory=list)
    reason: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Columns :meth:`TrialDB.record` accepts for a gate result."""
        payload: dict[str, Any] = {
            "tol_scale": self.tol_scale,
            "seeds": json.dumps(list(self.seeds)),
            "gate_runs": self.gate_runs,
            "stochastic": int(self.stochastic),
            "magnitudes_json": json.dumps([c.to_dict() for c in self.magnitudes]),
        }
        if self.max_abs_err is not None:
            payload["max_abs_err"] = self.max_abs_err
        if self.err_ulps is not None:
            payload["err_ulps"] = self.err_ulps
        if self.budget is not None:
            payload["budget"] = self.budget
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


def _finite(t: torch.Tensor) -> bool:
    return bool(torch.isfinite(t).all().item())


def check_case(
    fn: KernelFn,
    prob: Problem,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    target_abs_c: float | None,
    scale: float,
    zeros: bool,
    seed: int,
    tol_scale: float,
) -> MagnitudeCase:
    ref = oracle(prob, a, b, bias)
    ref_abs = float(ref.float().abs().max().item()) if ref.numel() else 0.0
    if not _finite(ref):
        return MagnitudeCase(
            target_abs_c,
            scale,
            zeros,
            seed,
            verdict="oracle_overflow",
            reason="fp32 oracle left the finite range",
            ref_abs_c=ref_abs,
        )
    try:
        out = fn(a, b, bias)
    except Exception as exc:
        name = type(exc).__name__
        if "out of memory" in str(exc).lower() or name in {"OutOfMemoryError", "cudaErrorMemoryAllocation"}:
            return MagnitudeCase(
                target_abs_c, scale, zeros, seed, "oom", reason=f"{name}: {exc}",
                ref_abs_c=ref_abs,
            )
        return MagnitudeCase(
            target_abs_c, scale, zeros, seed, "compile_error", reason=f"{name}: {exc}",
            ref_abs_c=ref_abs,
        )
    if not isinstance(out, torch.Tensor):
        return MagnitudeCase(
            target_abs_c, scale, zeros, seed, "incorrect",
            reason=f"kernel returned {type(out)!r}", ref_abs_c=ref_abs,
        )
    out_f = out.float()
    if not _finite(out_f):
        return MagnitudeCase(
            target_abs_c, scale, zeros, seed, "nan",
            reason="candidate produced NaN/Inf", ref_abs_c=ref_abs,
        )
    stored = ref.to(out.dtype)
    err = max_abs_error(stored, out)
    budget = error_budget(stored, tol_scale=tol_scale)
    ulps = error_in_ulps(stored, out)
    verdict = "ok" if err <= budget else "incorrect"
    return MagnitudeCase(
        target_abs_c,
        scale,
        zeros,
        seed,
        verdict,
        max_abs_err=err,
        err_ulps=ulps,
        budget=budget,
        reason=None if verdict == "ok" else f"max_error={err:.3e} > budget={budget:.3e}",
        ref_abs_c=ref_abs,
    )


def magnitude_ladder(prob: Problem) -> list[dict[str, Any]]:
    """The cases the gate runs: 3 seeds × 3 magnitudes, then all-zeros."""
    cases: list[dict[str, Any]] = []
    for seed in SEEDS:
        for target in TARGET_ABS_C:
            cases.append(
                {
                    "seed": seed,
                    "target_abs_c": target,
                    "scale": input_scale(prob, target),
                    "zeros": False,
                }
            )
    cases.append({"seed": 0, "target_abs_c": 0.0, "scale": 0.0, "zeros": True})
    return cases


_VERDICT_RANK = {
    "ok": 0,
    "oracle_overflow": 1,
    "infeasible": 2,
    "incorrect": 3,
    "nan": 4,
    "compile_error": 5,
    "oom": 6,
}


def _collapse(cases: list[MagnitudeCase]) -> str:
    """Pessimistic verdict across ladder steps. ``oracle_overflow`` never wins."""
    actionable = [c.verdict for c in cases if c.verdict != "oracle_overflow"]
    if not actionable:
        return "oracle_overflow"
    return max(actionable, key=lambda v: _VERDICT_RANK.get(v, 0))


def check(
    fn: KernelFn,
    prob: Problem,
    *,
    device: torch.device | str = "cpu",
    tol_scale: float = 1.0,
    stochastic: bool = False,
    gate_runs: int | None = None,
) -> CheckResult:
    """Run the magnitude ladder. Never times the candidate."""
    runs = STOCHASTIC_GATE_RUNS if stochastic else 1
    if gate_runs is not None:
        runs = gate_runs
    all_cases: list[MagnitudeCase] = []
    for _ in range(runs):
        for spec in magnitude_ladder(prob):
            a, b, bias = make_inputs(
                prob,
                scale=spec["scale"],
                seed=spec["seed"],
                device=device,
                zeros=spec["zeros"],
            )
            all_cases.append(
                check_case(
                    fn,
                    prob,
                    a=a,
                    b=b,
                    bias=bias,
                    target_abs_c=spec["target_abs_c"],
                    scale=spec["scale"],
                    zeros=spec["zeros"],
                    seed=spec["seed"],
                    tol_scale=tol_scale,
                )
            )
    verdict = _collapse(all_cases)
    scored = [c for c in all_cases if c.max_abs_err is not None]
    worst = max(scored, key=lambda c: c.max_abs_err or 0.0) if scored else None
    reason = next((c.reason for c in all_cases if c.verdict == verdict and c.reason), None)
    return CheckResult(
        verdict=verdict,
        max_abs_err=worst.max_abs_err if worst else None,
        err_ulps=worst.err_ulps if worst else None,
        budget=worst.budget if worst else None,
        tol_scale=tol_scale,
        seeds=SEEDS,
        gate_runs=runs,
        stochastic=stochastic,
        magnitudes=all_cases,
        reason=reason,
    )
