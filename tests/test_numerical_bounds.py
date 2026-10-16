"""Numerical checks of the bounds derived in docs/THEORY.md.

Every bound is evaluated against an fp64 reference on CPU. The implementations
under test are the repository references (``ako.ops.*_ref``) in fp16, bf16, fp32.
"""

from __future__ import annotations

import math
import time

import pytest
import torch
import torch.nn.functional as F
from hypothesis import given, settings
from hypothesis import strategies as st

from ako.dispatcher.cache import KernelCache
from ako.dispatcher.registry import KernelEntry, KernelRegistry
from ako.dispatcher.selector import KernelSelector
from ako.ops._registry import register_all
from ako.ops.rmsnorm_ref import rmsnorm_ref
from ako.ops.rope_ref import build_rope_tables, rope_ref
from ako.validation.numerical import error_budget, max_abs_error_any

U32 = 2.0**-24
UNIT_ROUNDOFF = {torch.float16: 2.0**-11, torch.bfloat16: 2.0**-8, torch.float32: U32}
# Seeded and derandomized: a failure reproduces. Widths are non-powers of two on
# purpose, since tails and masking live there.
SETTINGS = settings(max_examples=25, deadline=None, derandomize=True)
widths = st.integers(min_value=3, max_value=5000).filter(lambda n: n & (n - 1) != 0)
seeds = st.integers(0, 2**16)


def gamma(k: int, u: float = U32) -> float:
    return k * u / (1 - k * u)


def _randn(*shape: int, seed: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


# -- 1. summation --------------------------------------------------------------


def _pairwise_sum(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Balanced-tree sum in ``x.dtype``; returns (sum, tree depth)."""
    depth = 0
    while x.numel() > 1:
        if x.numel() % 2:
            x = torch.cat([x, x.new_zeros(1)])
        x = x[0::2] + x[1::2]
        depth += 1
    return x[0], depth


@SETTINGS
@given(n=widths, seed=seeds)
def test_summation_bounds(n: int, seed: int) -> None:
    x32 = _randn(n, seed=seed).float()
    exact = x32.double().sum().item()  # fp32 values are exact in fp64
    abs_sum = x32.double().abs().sum().item()

    err_any_order = abs(x32.sum().item() - exact)
    assert err_any_order <= gamma(n - 1) * abs_sum

    tree, depth = _pairwise_sum(x32)
    assert depth == math.ceil(math.log2(n))
    assert abs(tree.item() - exact) <= gamma(depth) * abs_sum

    # Higham and Mary, Theorem 2.4 with lambda = 6: probabilistic, seeded here.
    lam = 6.0
    gamma_tilde = math.exp(lam * math.sqrt(n) * U32 + n * U32**2 / (1 - U32)) - 1
    assert err_any_order <= gamma_tilde * abs_sum


# -- 2. RMSNorm ----------------------------------------------------------------


def _rmsnorm64(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x, w = x.double(), w.double()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@SETTINGS
@given(
    n=widths,
    dtype=st.sampled_from([torch.float16, torch.bfloat16, torch.float32]),
    seed=seeds,
)
def test_rmsnorm_ref_within_derived_bound(n: int, dtype: torch.dtype, seed: int) -> None:
    x = (_randn(4, n, seed=seed) * 3).to(dtype)
    w = (1 + 0.1 * _randn(n, seed=seed + 1)).to(dtype)
    exact = _rmsnorm64(x, w)
    u = UNIT_ROUNDOFF[dtype]
    delta32 = (1 - gamma(n + 2)) ** -0.5 * (1 + U32) ** 4 - 1
    # Underflow: fp16 outputs below the normal range are absolute, not relative.
    eta = torch.finfo(dtype).tiny * torch.finfo(dtype).eps / 2
    bound = (u + (1 + u) * delta32) * exact.abs() + eta
    err = (rmsnorm_ref(x, w).double() - exact).abs()
    assert bool((err <= bound).all()), f"max excess {(err - bound).max().item():.3e}"


# -- 3. RoPE -------------------------------------------------------------------


def _pair_norms(y: torch.Tensor) -> torch.Tensor:
    half = y.shape[-1] // 2
    return torch.sqrt(y[..., :half] ** 2 + y[..., half:] ** 2)


def test_rope_is_orthogonal_in_fp64() -> None:
    x = _randn(2, 33, 3, 10, seed=0)
    cos, sin = build_rope_tables(33, 10, dtype=torch.float64)
    torch.testing.assert_close(
        _pair_norms(rope_ref(x, cos, sin)), _pair_norms(x), rtol=1e-13, atol=0
    )


@pytest.mark.parametrize("offset", [0, 2**10, 2**16])
def test_rope_table_error_bounded_and_grows_with_position(offset: int) -> None:
    seq, head_dim, base = 37, 12, 10000.0
    x = _randn(1, seq, 2, head_dim, seed=offset)
    c32, s32 = build_rope_tables(seq, head_dim, offset=offset, base=base)
    c64, s64 = build_rope_tables(seq, head_dim, offset=offset, base=base, dtype=torch.float64)
    delta = _pair_norms(rope_ref(x, c32.double(), s32.double()) - rope_ref(x, c64, s64))

    omega = base ** (-2.0 * torch.arange(head_dim // 2, dtype=torch.float64) / head_dim)
    theta = torch.outer(torch.arange(offset, offset + seq, dtype=torch.float64), omega)
    c = 2 * math.log(base) + 4
    bound = (c * U32 * theta + 3 * U32)[None, :, None, :] * _pair_norms(x)
    assert bool((delta <= bound).all())
    if offset == 2**16:  # far from the origin the angle error dominates table rounding
        assert (delta / _pair_norms(x)).max().item() > 100 * U32


# -- 4. SwiGLU -----------------------------------------------------------------


@SETTINGS
@given(n=widths, dtype=st.sampled_from([torch.float16, torch.bfloat16]), seed=seeds)
def test_swiglu_first_order_propagation(n: int, dtype: torch.dtype, seed: int) -> None:
    g, v = _randn(n, seed=seed) * 4, _randn(n, seed=seed + 1)
    # Keep inputs in the normal range: subnormal rounding is absolute, not relative.
    g, v = g.sign() * g.abs().clamp_min(1e-3), v.sign() * v.abs().clamp_min(1e-3)
    y = F.silu(g) * v
    y_hat = F.silu(g.to(dtype).double()) * v.to(dtype).double()
    kappa = (1 + g * (1 - torch.sigmoid(g))).abs()
    # First-order bound, with 5 % for the O(u^2) remainder.
    bound = 1.05 * (kappa + 1) * UNIT_ROUNDOFF[dtype] * y.abs()
    assert bool(((y_hat - y).abs() <= bound).all())


# -- candidates vs native torch in the same dtype ------------------------------

NATIVE = {
    "rmsnorm": lambda x, w: F.rms_norm(x, (x.shape[-1],), w, 1e-6),
    "swiglu": lambda g, v: F.silu(g) * v,
}
EXACT = {"rmsnorm": _rmsnorm64, "swiglu": lambda g, v: F.silu(g.double()) * v.double()}


@pytest.mark.parametrize("op", sorted(NATIVE))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_candidates_within_twice_native_error(op: str, dtype: torch.dtype) -> None:
    if op == "rmsnorm" and not hasattr(F, "rms_norm"):
        pytest.skip("F.rms_norm needs PyTorch >= 2.4")
    second = _randn(1000, seed=2) if op == "rmsnorm" else _randn(8, 1000, seed=2)
    args = ((_randn(8, 1000, seed=1) * 3).to(dtype), (1 + 0.1 * second).to(dtype))
    exact = EXACT[op](*args)
    native_err = (NATIVE[op](*args).double() - exact).abs().max().item()
    candidates = [
        e
        for e in register_all(KernelRegistry()).list_kernels(op)
        if e.backend not in ("triton", "torch_compile")  # GPU / compile: GPU suite
    ]
    assert candidates
    for entry in candidates:
        err = (entry.fn(*args).double() - exact).abs().max().item()
        assert err <= 2 * native_err, f"{entry.kernel_name}: {err:.3e} > 2 x {native_err:.3e}"


# -- dispatcher: whatever is selected passed the gate --------------------------


@settings(max_examples=15, deadline=None, derandomize=True)
@given(errors=st.lists(st.sampled_from([0.0, 0.5, 2.0, 1e3]), min_size=1, max_size=3))
def test_selected_kernel_always_passes_gate(tmp_path_factory, errors: list[float]) -> None:
    x, w = torch.randn(4, 300), torch.ones(300)
    ref = rmsnorm_ref(x, w)
    budget = error_budget(ref)

    def slow_baseline(x, weight, eps=1e-6):
        time.sleep(0.001)
        return rmsnorm_ref(x, weight, eps)

    registry = KernelRegistry()
    registry.register(KernelEntry("rmsnorm", "ref", slow_baseline, backend="pytorch", priority=10))
    for i, k in enumerate(errors):
        # Fake CPU backend: faster than the baseline and off by k budgets, so a
        # wrong one would win on speed alone.
        def fake(x, weight, eps=1e-6, _shift=k * budget):
            return ref + _shift

        registry.register(KernelEntry("rmsnorm", f"fake{i}", fake, backend="custom", priority=5))
    selector = KernelSelector(
        registry=registry,
        cache=KernelCache(path=tmp_path_factory.mktemp("cache") / "cache.json"),
        warmup=1,
        repeats=3,
        allow_cpu_tuning=True,
    )
    chosen = selector.select_entry("rmsnorm", x, w)
    assert max_abs_error_any(ref, chosen.entry.fn(x, w)) <= budget
    if min(errors) <= 0.5:  # a correct fast candidate exists: the gate must not hide it
        assert chosen.entry.kernel_name != "ref"
