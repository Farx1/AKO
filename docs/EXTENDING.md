# Extending AKO with your own kernel

AKO's dispatch layer is a registry plus a selector. Anything callable that
matches an operation's calling convention can be registered as a candidate; the
selector then proves it numerically against the PyTorch reference and, if it
survives, times it against the baselines before it is allowed to run.

This document is the contract. Everything below is quoted from the code, and the
end-to-end example runs on CPU.

## 1. The three objects you touch

### `KernelEntry` — what you register

`ako/dispatcher/registry.py`:

```python
@dataclass(frozen=True)
class KernelEntry:
    op_name: str
    kernel_name: str
    fn: KernelFn          # KernelFn = Callable[..., torch.Tensor]
    backend: str
    priority: int = 0
```

- `op_name` must be an operation AKO already dispatches: `"rmsnorm"`,
  `"rmsnorm_residual"`, `"rope"`, `"swiglu"`. Registering a new `op_name` is legal but
  nothing calls it — you need a public wrapper of your own (see §6).
- `kernel_name` is the identity used everywhere: forced dispatch
  (`ako.rmsnorm(..., kernel="my_kernel")`), the cache record, benchmark reports.
  Registering the same `kernel_name` twice **replaces** the previous entry.
- `backend` is a free string. Two values are load-bearing:
  `"pytorch"` marks the numerical reference (do not use it), and the members of
  `BASELINE_BACKENDS = {"pytorch", "torch_native", "torch_compile"}`
  (`ako/dispatcher/selector.py:32`) count toward the published
  `speedup_vs_best_baseline`. A third-party kernel should use anything else —
  `"triton"`, `"cuda"`, `"custom"`, your library's name.
- `priority` only decides the **fallback**: the highest-priority entry is what
  `registry.default(op)` returns, which is what dispatch uses when autotuning is
  off, on CPU, or when a single kernel is registered. Built-in priorities are
  pytorch `10`, torch_native/torch_compile `7`, triton `5`
  (`ako/ops/_registry.py:16`). Register below `10` unless you want your kernel to
  become the unmeasured default.

### `KernelRegistry` — where it goes

```python
class KernelRegistry:
    def register(self, entry: KernelEntry) -> None
    def list_ops(self) -> list[str]
    def list_kernels(self, op_name: str) -> list[KernelEntry]      # priority desc
    def get(self, op_name: str, kernel_name: str) -> KernelEntry | None
    def default(self, op_name: str) -> KernelEntry | None
```

`ako.dispatcher.registry.get_registry()` returns the process-wide instance that
`ako.rmsnorm` / `ako.rope` dispatch through.

### `KernelSelector` — what decides

`ako/dispatcher/selector.py`:

```python
KernelSelector(
    registry: KernelRegistry | None = None,
    cache: KernelCache | None = None,
    *,
    auto_tune: bool | None = None,        # defaults to env AKO_AUTOTUNE, default on
    error_threshold: float | None = None, # None = dtype-derived budget
    min_speedup: float = 1.0,
    warmup: int = 5,
    repeats: int = 20,
    allow_cpu_tuning: bool = False,
)

selector.select(op_name, x, *args, forced=None, extra_key=None, call_kwargs=None) -> KernelFn
selector.select_entry(op_name, x, *args, forced=None, extra_key=None, call_kwargs=None) -> SelectionResult
selector.clear_memo() -> None
```

`ako.dispatcher.selector.get_selector()` returns the process-wide selector.

## 2. The calling convention per operation

Your `fn` is invoked **positionally** by the public wrapper, so it must accept
exactly the arity below. Return types must match too, because the validator
compares outputs element by element.

| `op_name` | call | returns |
|---|---|---|
| `rmsnorm` | `fn(x, weight, eps)` — `x (..., hidden_dim)`, `weight (hidden_dim,)`, `eps: float` | `Tensor`, same shape/dtype as `x` |
| `rmsnorm_residual` | `fn(x, residual, weight, eps)` | `(normed, new_residual)` where `new_residual = x + residual` |
| `rope` | `fn(x, cos, sin, style)` — `x (batch, seq_len, n_heads, head_dim)`, `cos`/`sin` `(seq_len, head_dim // 2)`, `style: str` in `{"neox", "interleaved"}` | `Tensor`, same shape/dtype as `x` |
| `swiglu` | `fn(gate, up)` — both `(..., hidden_dim)`, same shape and dtype | `Tensor`, same shape/dtype as `gate` |

The references are the specification: `ako/ops/rmsnorm_ref.py::rmsnorm_ref`,
`rmsnorm_residual_ref`, `ako/ops/rope_ref.py::rope_ref`,
`ako/ops/swiglu_ref.py::swiglu_ref`.

Note that `rope` passes `style` twice — as `extra_key` (so the cache
distinguishes the two styles) and inside `call_kwargs` (so the benchmark and the
validator call your kernel the same way dispatch will). Any per-call argument
that changes behaviour must go through `extra_key`, or the selector's memo and
the cache key will conflate two different calls (`ako/ops/rope.py:33`).

## 3. Complete example (runs on CPU)

Save as `extend_example.py` and run it. It registers one correct kernel and one
deliberately wrong one, forces a call, then asks the selector to choose.

```python
"""Register a third-party RMSNorm kernel and let AKO measure it."""

import torch

import ako
from ako.dispatcher.registry import KernelEntry, get_registry
from ako.dispatcher.selector import KernelSelector, get_selector
from ako.ops._registry import register_all


def rmsnorm_mine(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Correct kernel: same signature and contract as ``rmsnorm_ref``."""
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_c = x.to(compute_dtype)
    inv_rms = torch.rsqrt(x_c.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_c * inv_rms * weight.to(compute_dtype)).to(x.dtype)


def rmsnorm_wrong(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Plausible but wrong: mean absolute value instead of root-mean-square."""
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_c = x.to(compute_dtype)
    inv_rms = torch.rsqrt(x_c.abs().mean(dim=-1, keepdim=True) + eps)
    return (x_c * inv_rms * weight.to(compute_dtype)).to(x.dtype)


register_all()  # built-in reference + baselines; idempotent
registry = get_registry()
for name, fn in (("rmsnorm_mine", rmsnorm_mine), ("rmsnorm_wrong", rmsnorm_wrong)):
    registry.register(
        KernelEntry(
            op_name="rmsnorm",
            kernel_name=name,
            fn=fn,
            backend="custom",
            priority=1,
        )
    )
get_selector().clear_memo()  # the dispatch memo may hold a pre-registration decision

torch.manual_seed(0)
x = torch.randn(4096, 2048)
weight = torch.randn(2048)

# Forced dispatch: skips selection entirely, useful to unit-test your kernel.
forced = ako.rmsnorm(x, weight, kernel="rmsnorm_mine")
assert torch.allclose(forced, ako.rmsnorm(x, weight, kernel="rmsnorm_ref"), atol=1e-5)

# Measured selection. allow_cpu_tuning=True because autotuning is CUDA-only by
# default; on a GPU box, plain ako.rmsnorm(x, weight) does this for you.
selector = KernelSelector(allow_cpu_tuning=True, warmup=3, repeats=10)
result = selector.select_entry("rmsnorm", x, weight)
print("selected:", result.entry.kernel_name, "source:", result.source)
print("rejected:", result.rejected)
print(result.to_dict())
```

Actual output on a CPU-only machine (fp32, 4096x2048):

```
selected: rmsnorm_mine source: benchmark
rejected: ('rmsnorm_wrong: max_error=1.796e+00 > budget=2.836e-05 (1883079.0 ulps)',)
```

Three things happened, and they are the whole contract: the wrong kernel was
rejected before it was ever timed; the correct one was timed against the
reference and the `torch_native` baseline and had to win to be selected; and the
decision was written to the cache so the next process skips the benchmark.

The rejection line is deterministic under the seed. The *selected* line is not:
on CPU these three correct kernels are all within measurement noise of each
other, so re-running can select `rmsnorm_ref` or `rmsnorm_torch_native` instead.
That is the honest reading of a 1.00x speedup — see "Selection times a single
run" below, and raise `min_speedup` if you want a margin before a switch is
allowed.

Re-running prints `source: cache`. Delete the record — or point
`AKO_CACHE_PATH` at a scratch file — to force a fresh measurement:

```bash
AKO_CACHE_PATH=/tmp/ako_cache.json python extend_example.py
```

`SelectionResult` (`ako/dispatcher/selector.py:53`) carries `op_name`, `entry`,
`shape_key`, `source` (`"cache" | "benchmark" | "default" | "forced"`),
`latency_p50_ms`, `baseline_ms`, `max_error`, `candidates_evaluated`,
`rejected`, `best_baseline_ms`, `best_baseline_kernel`, plus the properties
`from_cache`, `speedup`, `speedup_vs_best_baseline` and `to_dict()`.

## 4. What the harness guarantees

- **Rejection outside the error budget.** Every candidate is run once and
  compared to the reference with `max_abs_error_any(ref_out, candidate_out)`
  (`ako/validation/numerical.py:35`, which also handles multi-output kernels such
  as `(normed, residual)`). The budget is
  `error_budget(ref) = atol + rtol * max|ref|` with `atol = 1e-5` and `rtol` taken
  from the output dtype (fp32 `1.3e-6`, fp16 `1e-3`, bf16 `1.6e-2`) — the
  tolerances `torch.testing.assert_close` uses. Over budget means the kernel is
  never benchmarked and never dispatched; the reason string, including the error
  in ulps, lands in `SelectionResult.rejected`. Set
  `KernelSelector(error_threshold=...)` to replace the derived budget with a
  fixed absolute one.
- **A candidate that raises cannot break dispatch.** Any exception from your
  kernel is caught, recorded in `rejected` as `"<name>: <ExcType>: <msg>"`, and
  selection continues (`selector.py:303`).
- **Paired measurement.** Baseline and candidates are timed in the same process,
  on the same tensors, through the same `bench_op` primitive: warmup, then an
  inner-loop calibration so one timed batch lasts about `TARGET_BATCH_MS = 1.0`
  (capped at `MAX_INNER_ITERS = 200`), then `repeats` batches; the reported
  latency is the p50 of the per-call samples. On CUDA, device time is also
  recorded with CUDA events, and the gap to wall-clock is
  `launch_overhead_ms` (`ako/benchmark/bench_op.py`).
- **Non-regression.** The reference is always in the candidate set and starts as
  the incumbent. A candidate replaces it only if
  `latency < best_latency and baseline_ms >= latency * min_speedup`
  (`selector.py:312`), so the selected kernel is never slower than the baseline
  it was measured against. Raise `min_speedup` above `1.0` to demand a margin.
- **Cache keyed by GPU + op + shape + dtype.** The key is
  `DeviceInfo.cache_key` (`"<device_name with _>_<index>"`) then `op_name` then
  `shape_key(tensors, extra)` — `"(4096, 2048)_float32_(2048,)"` above: the first
  tensor's shape, its dtype, then the remaining tensors' shapes, plus the
  `extra_key` variant tag. Records are JSON at `~/.ako/kernel_cache.json`
  (override with `AKO_CACHE_PATH`) and carry the metrics that justified the
  decision: latency, baseline, speedups, `max_error`, `error_budget`, backend.
  A corrupt cache file is discarded rather than raised.

## 5. What it does not guarantee

- **No backward pass.** Every kernel here is inference-only; nothing is checked
  against autograd, and no candidate is exercised under `.backward()`.
- **No input validation on your behalf.** The selector calls your `fn` with
  whatever the caller passed. Contiguity, dtype support, alignment, maximum
  hidden dimension — guard them inside your kernel and raise; a raise is handled
  (§4), a silent wrong answer is only caught if it exceeds the error budget on
  the *one* input the selector happened to validate with.
- **Validation is one shape, one draw.** Correctness is established on the exact
  tensors of the call that triggered autotuning, not over a distribution. Your
  own test suite is still your test suite.
- **No cross-shape generalization.** A decision is valid for one
  `(gpu, op, shape, dtype, variant)` key. A new shape re-benchmarks; a decision
  cached on one machine says nothing about another.
- **Nothing is re-validated over time.** The cache has no expiry and no
  invalidation on driver, PyTorch, or kernel-source changes. If you change your
  kernel's numerics, clear the cache yourself (`KernelCache.clear()`, or delete
  the file).
- **No autotuning off CUDA.** On CPU, `select_entry` short-circuits to
  `registry.default(op)` with `source="default"` unless you pass
  `allow_cpu_tuning=True` — so the priority you registered with is what runs.
  The same short-circuit applies when `AKO_AUTOTUNE` is off or a single kernel
  is registered.
- **Selection times a single run.** `bench_op` defaults to `runs=1`, so the
  selector compares one measurement pass per kernel: no repetition of the whole
  pass, no confidence interval, no outlier rejection. Two selections can
  disagree on kernels that are within noise of each other. Repetition is
  available but opt-in, because a higher default would multiply the cost of
  every first dispatch: pass `runs=N` for the median, IQR and `unstable` flag,
  and use `compare_paired` when you need a bootstrap interval on a ratio. Those
  are the tools the published benchmarks use; the selector deliberately does
  not pay for them.
- **`ako.models` and `ako.experiments` are not library surface.** They are
  research scaffolding for the paper's measurements.

## 6. Adding a whole new operation

If your kernel implements an op AKO does not dispatch yet, the registry accepts
it, but nothing calls it. The files an existing op touches
(`ako/ops/_registry.py:31`) are: `ako/ops/<op>_ref.py` (the PyTorch reference —
this is what your kernel will be validated against, so it is not optional),
optionally `<op>_torch.py` and `<op>_triton.py`, `ako/ops/<op>.py` (the public
wrapper: `register_all()` then `get_selector().select(...)` then `fn(...)`),
the registration lists in `_registry.py`, and the `__all__` exports in
`ako/ops/__init__.py` and `ako/__init__.py`.

## 7. Testing a Triton kernel without a GPU

Triton's interpreter runs a `@triton.jit` body on CPU tensors when
`TRITON_INTERPRET=1` is set **before** Triton is imported. That is why the checks
live in a standalone script executed as a subprocess
(`tests/_triton_interpreter_checks.py`, run by `tests/test_triton_interpreter.py`).

The pattern this imposes on your kernel: keep the bare `@triton.jit` function
importable under its own name, and apply `triton.heuristics` / `triton.autotune`
to a *wrapper*, so the raw body can be launched with its meta-parameters passed
explicitly. AKO's kernels keep the undecorated body as `_<name>_kernel_impl`
(`ako/ops/rmsnorm_triton.py:40`).

Then launch it directly and compare against the reference:

```python
import os
os.environ["TRITON_INTERPRET"] = "1"   # before importing torch/triton

import torch
import triton
from ako.ops.rmsnorm_ref import rmsnorm_ref
from ako.ops.rmsnorm_triton import _rmsnorm_kernel_impl
from ako.validation.numerical import max_abs_error

n_rows, hidden_dim = 3, 100            # not a power of two: exercises the masked tail
x = torch.randn(n_rows, hidden_dim)
weight = torch.randn(hidden_dim)
y = torch.empty_like(x)
_rmsnorm_kernel_impl[(n_rows,)](
    x, weight, y, n_rows, hidden_dim, 1e-6, x.stride(0), y.stride(0),
    BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
)
print(max_abs_error(y, rmsnorm_ref(x, weight)))   # < 1e-5
```

This validates kernel *logic* — indexing, masking, pair offsets, dtype rounding.
It says nothing about performance, and it bypasses the autotuner, which needs a
real driver to time configs. Interpreted execution is slow: keep the shapes tiny.

To hook your kernel into the existing suite, add an entry to the `checks` dict of
`tests/_triton_interpreter_checks.py` and the matching name to the expected-keys
list in `tests/test_triton_interpreter.py`. GPU-only tests carry both
`@pytest.mark.gpu` and
`@pytest.mark.skipif(not is_triton_available(), reason="CUDA + Triton required")`;
CI runs `pytest -q -m "not gpu"`.
