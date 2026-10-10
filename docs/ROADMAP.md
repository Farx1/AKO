# AKO roadmap

Companion to `docs/ARCHITECTURE.md`. Every milestone follows the same order:
plan → minimal implementation → tests → correctness → benchmark → analysis →
decision. If a benchmark invalidates a hypothesis, the plan changes; the
architecture is not defended.

Legend:

- **[GPU]** requires the RTX 3070.
- **[CPU]** developed and tested on a machine without CUDA. Triton kernel logic
  is exercised through `TRITON_INTERPRET=1` using the existing subprocess harness
  `tests/_triton_interpreter_checks.py`.

Hour estimates are working hours, not calendar time.

---

## M1 — Can this machine resolve the effect at all? [GPU] — 4 h

**Status.** Done. Overnight 480 min / 640 points finished
(`reports/rtx3070/noise_floor.json`). Published verdict: **`stop`**.
Drift first→last was only −1.58% (would have passed), but **cv p95 = 0.113**
exceeds the 0.10 hard stop (p50 spread 42% of median; late points settle near
cv 0.01 under `sw_power_cap` at ~84 °C). No ranking is publishable on this
card. Downstream GPU go/no-go (M6 A/B) is not a ranking claim; the project
continues as CPU composer + schema + feasibility map.

**Objective.** Establish the measurement noise floor of the 3070 before writing
any search code, because the ranking *is* the product and a card that drifts more
than the effects being ranked cannot produce one.

**Files.**
- `ako/dispatcher/hardware.py` (modify): `DeviceInfo` gains `sm_count`,
  `smem_per_block`, `l2_bytes`, `regs_per_sm`; fix `driver_version`, which today
  stores `torch.cuda.get_device_capability()`.
- `ako/benchmark/bench_op.py` (modify): `flush_l2: bool = True`, a 32 MiB int8
  buffer `.zero_()`ed **inside the inner loop, before every call** — not once per
  timed batch. `_calibrate_inner` sizes a batch to `TARGET_BATCH_MS = 1 ms`, so
  at 1024³ fp16 (~86 µs) `inner ≈ 11` and a per-batch flush leaves calls 2..11
  running on hot L2, on the very shape the flush exists for. The flush cost is
  calibrated once and subtracted; `l2_state` records the mode.
- `ako/benchmark/env.py` (new, ~40 lines): `sample_env()` parsing
  `nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw`; `preflight()`
  asserting `--query-compute-apps` is empty and `utilization.gpu < 3`.
  `preflight()` must run **before `torch.cuda.init()`**, or filter its own PID —
  otherwise the sweep process trips its own compute-apps assertion once CUDA is
  initialized. State the ordering in the docstring.
- `ako/benchmark/canary.py` (new, ~30 lines): `canary_p50()`, one fixed
  `torch.matmul` 4096³ timing, called at the start of every cell (invariant 4).
- `scripts/noise_floor.py` (new, ~60 lines).

**Measurable success criterion.**
`torch.matmul` at 4096³ fp16, 10 s preheat then 20 measurements over 15 min:
**p50 drift between the first and last measurement < 5%**, and per-measurement
CV < 0.05. Output written to `reports/rtx3070/noise_floor.json` with
`sm_clock_mhz`, `temp_c`, `power_w` per point.

**Test.** `tests/test_env.py` [CPU]: `sample_env()` returns `None` fields instead
of raising when `nvidia-smi` is absent; `flush_l2=True` on CPU is a no-op and
`bench_op` results are unchanged. Plus [GPU]: a flushed and an unflushed 1024³
measurement differ by a **nonzero, reported** amount — if they do not, the flush
is not working and every wave-quantization number downstream is measured on hot
L2.

**Decision.** Drift ≥ 5% after preheat → the drift gate and paired comparison are
not hygiene but the only viable measurement mode, and every subsequent milestone
reports paired ratios only, never absolute rankings. **This binds M8**: on a
220 W consumer card with no lockable clocks — the branch this is most likely to
take — the M8 prior cannot be fitted on `log(wall_p50_ms)`, since that pools
absolute latencies across sweeps and thermal states. Its target is
`log(speedup_vs_incumbent)` against the sweep's frozen incumbent, which is
invariant to clock state by construction. Drift ≥ 15% → stop; no ranking is
publishable on this card. Renting a stabler one is **not** the fallback (one
3070, decision of 2026-08-13): the project reduces to the CPU-side composer, the
schema and the feasibility map, and says so.

---

## M2 — The trial database [CPU] — 4 h

**Objective.** Ship the artifact the whole project points at, before anything
writes to it.

**Files.** `ako/search/db.py` (new, ~150 lines: schema from
`ARCHITECTURE.md` §3.5, six methods), `ako/search/spec.py` (new, ~90 lines:
`Problem`, `Candidate`, `DeviceCaps`, `caps()` wrapping `get_device_info`).

**Measurable success criterion.** Round-trip of 1000 synthetic rows in < 1 s;
`blacklist()` returns exactly the `cand_hash` set with
`verdict IN ('incorrect','nan','compile_error')` for the matching
`(gpu, dtype, toolchain_key)` and nothing else — the §3.5 contract verbatim, not
`verdict != 'ok'`; `neighbours()` on a seeded 20-row fixture returns the shapes
in the correct log-space order.

**Test.** `tests/test_search_db.py`: the three assertions above, plus a variant
recorded `incorrect` at shape A being blacklisted at shape B, plus a `drifted`
sweep being excluded by the join. Plus the **negative** assertion: a row with
`verdict='oom'`, `'infeasible'` or `'oracle_overflow'` at shape A must NOT appear
in `blacklist()` for shape B — those verdicts are shape- or oracle-dependent
(an OOM at 4096³ says nothing about 1024³; `waves`/`wave_eff` infeasibility is a
function of `(M,N,K)`), and encoding `verdict != 'ok'` here would permanently
delete candidates that are valid at other shapes, shrinking the search space M7
is supposed to measure. Plus: a row written under a different `toolchain_key` is
excluded from `blacklist()` and from `fit_features()`.

**Decision.** None; this is infrastructure. If it takes more than 6 h, the schema
is too wide — cut columns, not tests.

---

## M3 — Candidate space and free filtering [CPU] — 5 h

**Objective.** Enumerate the four compositions and the variant grid, and discard
the impossible ones with arithmetic instead of a compiler.

**Files.** `ako/search/constraints.py` (new, ~70 lines: `smem_bytes`, `waves`,
`wave_eff`, `global_mem_bytes`, `feasible`), `ako/search/space.py` (new, ~70
lines: `partitions`, `enumerate_candidates`), `ako/search/blocks.py` (new, ~140
lines: the seven blocks including `torch_addmm_act`, torch ones only at this
stage), `ako/dispatcher/registry.py` (modify: `TemplateEntry`,
`register_template`).

`global_mem_bytes` sums inputs + the fp32 oracle + the split-K workspace + the
32 MiB flush buffer and compares against `total_mem_gb` minus 600 MB of display
headroom, so an over-large candidate is a free `infeasible` row rather than a
GPU-side OOM that fragments the allocator for the candidates after it.

**Measurable success criterion.**
- `partitions(("matmul","bias","act"))` yields exactly 4 compositions.
- `smem_bytes` for `BM=BN=128, BK=64, num_stages=4`, fp16 = **131072 B**, which
  exceeds 101376 and is rejected on cc 8.6; the same config is accepted against a
  synthetic cc 8.0 `DeviceCaps` with 164 KB.
- Over the exact 720-point grid `BLOCK_M{16,32,64,128,256} × BLOCK_N{32,64,128,
  256} × BLOCK_K{32,64,128} × num_warps{2,4,8} × num_stages{2,3,4,5}` in fp16,
  the shared-memory predicate keeps **441**.

**Test.** `tests/test_search_space.py`: the three assertions above. The 441 count
is asserted against the grid *defined in the test file*, not against the full
candidate space — adding `GROUP_M`/`SPLIT_K` axes multiplies the count without
changing the predicate.

**Known defect to avoid.** The register-pressure predicate must not be
`BLOCK_M * BLOCK_N // (num_warps * 32) > 64`. At `128×128` with 4 warps that is
128 fp32 accumulator registers per thread, well under the 255 architectural
limit, and the filter would silently delete the tile family most likely to win at
large shapes. Use 255, and assert in the test that `128×128/4 warps` survives.

**Decision.** If the feasible count after all predicates is under ~50 per
problem, the filters are too aggressive — loosen before measuring, because a
filter bug is indistinguishable from "no candidate can win" in the results.

---

## M4 — One Triton template, constexpr epilogues [CPU via TRITON_INTERPRET=1] — 8 h

**Objective.** Get every composition and every tiling variant out of one kernel,
with no code generation.

**Files.** `ako/search/kernels/mm_fused.py` (new, ~200 lines: `apply_epi`,
`_mm_fused`, `epi_kernel`, `launch`), `ako/search/blocks.py` (modify: register
`triton_mm` and `triton_mm_fused`), `tests/_triton_interpreter_checks.py`
(modify: add the four new checks).

**Pre-check, before any of the 8 h are spent.** Assert that the epilogue kernel
actually executes under `TRITON_INTERPRET=1`. `tl.math.tanh` does **not** exist
in the installed Triton 3.7.1 (`hasattr(tl.math,'tanh')` is False; tl.math
exposes only abs/ceil/cos/div_rn/erf/exp/exp2/fdiv/floor/fma/log/log2/rsqrt/sin/
sqrt/sqrt_rn/umulhi), and the `triton.language.extra.libdevice.tanh` it moved to
fails in the interpreter with `InterpreterError: cannot convert None of type
NoneType to tensor`. Since `act=gelu_tanh` is the `Problem` default and the act
of the whole M7 grid, a libdevice epilogue makes this entire [CPU] milestone
unrunnable. `apply_epi` therefore uses `tanh(x) = 2*sigmoid(2x) - 1`
(ARCHITECTURE.md §3.4) — `tl.sigmoid`, `tl.maximum`, `tl.dot` and `tl.atomic_add`
are all verified working in the interpreter. If the pre-check fails, the
milestone is re-planned, not started.

**Measurable success criterion.** Under `TRITON_INTERPRET=1`, over 4 shapes ×
{fp16, bf16} × `SPLIT_K ∈ {1,4}` × all 4 compositions = 64 cases, the fused
output matches `F.gelu(torch.addmm(bias, a, b), approximate="tanh")` computed in
fp32 and downcast, within `error_budget(reference)` from
`ako/validation/numerical.py`:

- **48 cases are legal** (all `SPLIT_K=1`, plus `SPLIT_K=4` on the two
  compositions whose GEMM carries no epilogue), of which **24 fp16 checks pass
  numerically**. The 24 legal bf16 cases are **NOT TESTED** on CPU: the Triton
  3.7.1 interpreter stores bf16 as uint16 and `create_dot` hands the raw bits to
  `np.matmul`, so the fused output comes back at ~1e11 against a ~-0.15
  reference. They are gated by `feasible()` here and first validated on the GPU
  at M5 — a masking or dtype bug on the bf16 path therefore surfaces during that
  sweep, not before it.
- **16 checks are recorded infeasible with reason `splitk_with_epilogue`** —
  `SPLIT_K=4` × {`mm+bias+act`, `mm+bias|act`} × 4 shapes × 2 dtypes. This is an
  **asserted outcome of the test**, not an exception to it: the test fails if
  `feasible()` lets any of the 16 through, and fails if it rejects any of the 48.
  Stating the gate as "zero failures out of 64" would make it unsatisfiable,
  since the M4 decision below declares those same 16 illegal.

The 4 shapes must include at least one where `M % BLOCK_M`, `N % BLOCK_N` and
`K % BLOCK_K` are **all nonzero** (e.g. 65 × 97 × 129). Every dimension in the
M7 grid is a multiple of 8, so if M4 does not exercise the masked /
partially-vectorized path, no masking bug ever surfaces anywhere in the project.

**Test.** `tests/test_triton_interpreter.py` (existing harness, extended). The
interpreter is 100–1000× slower than the GPU, so every dimension here stays
≤ 129 (small, but not necessarily aligned — see the unaligned shape above).

**Decision.** If `SPLIT_K > 1` with a non-empty fused epilogue produces wrong
results — it will, because `atomic_add` makes partial tiles visible and a
non-linear epilogue applied per-partial is incorrect — that is not a bug to fix
but a **legality rule to name and record**: `feasible()` returns
`"splitk_with_epilogue"`, the rejection lands in the DB, and the coupling
(at M ≤ 64 the winning root cannot carry the epilogue) is itself a result. Split-K
also breaks a linear bias inside the GEMM — it would be added `SPLIT_K` times —
so the rule covers `mm+bias|act` as well as `mm+bias+act`. Legal split-K is still
**run-to-run nondeterministic** (`atomic_add` accumulation order), which invariant
1 handles with the repeated gate specified in M5; the block declares
`nondeterministic=True`.

---

## M5 — Correctness gate and measurement wrapper [GPU] — 8 h

**Status.** CPU-complete: numerical gate, CUDA-graph capture path
(`ako/benchmark/graph.py`, `bench_op(..., graph=True)`), and `Block.build`
wired for every block except `torch.compile`. GPU execution of the graph path
and of the bf16 half of the fused template waits until the M1 overnight trace
releases the card.

**Objective.** Make an incorrect kernel impossible to benchmark.

**Files.** `ako/search/correctness.py` (new, ~90 lines: `check()`, `CheckResult`,
`magnitude_ladder()`), `ako/search/measure.py` (new, ~80 lines: `single()`,
`preflight()`, `env_snapshot()` over the existing `bench_op`/`compare_paired`),
`ako/benchmark/graph.py` (new, ~50 lines, 3 h of the 8: CUDA-graph capture on a
side stream with pinned static input/output tensors, replayed inside `bench_op`
under a `graph=True` flag; capture skipped and `graph_captured=0` recorded for
candidates that allocate — `torch.compile` modes, split-K workspaces).

Graph capture is not optional polish: invariant 8 requires decode rows printed
with and without it, and ARCHITECTURE §1.6 rests the whole decode analysis on
"the entire visible gain is 2 kernel launches and it disappears under CUDA
graphs". Today `bench_op` calls `fn(*args)` in a loop with no static-input path,
so that row cannot be produced at all — and the decode row is where the split-K
claim lives.

**Measurable success criterion — the magnitude ladder is output-side.** The gate
runs 3 seeds × 3 magnitudes + an all-zeros case. The magnitudes are stated as
targets on the **output**, `|C| ≈ {1e-2, 1, 1e3}`, and the input scale is derived
per problem as `s ∝ sqrt(target / K)` — the safe input range depends on K, so a
fixed input multiplier is not portable across the grid. Concretely at 4096³ fp16,
inputs at `×1e3` would give `|C| ~ 1e6·sqrt(4096) = 6.4e7` against an fp16 max of
65504: reference and candidate are both `inf`, `max_abs_error` is `inf - inf =
nan`, and **every** candidate is classified `verdict='nan'` — which invariant 1
then makes permanent, blacklisting every correct candidate before a single
benchmark runs. The mirror failure at `×1e-3` (`|C| ~ 6.4e-5`, just above the
fp16 min normal 6.1e-5) makes the outputs subnormal, so `error_budget`'s
`atol=DEFAULT_ATOL=1e-5` floor (`ako/validation/numerical.py:94,113`) swamps the
signal and the check passes vacuously for any kernel.

Two rules follow:

1. **Non-finite reference ⇒ `verdict='oracle_overflow'`**, skipped and not
   failed, and ignored by `blacklist()`. A ladder step whose fp32 oracle leaves
   the output dtype's finite range is a fact about the ladder, never about the
   candidate.
2. **One result per magnitude case**, written to `magnitudes_json`, not one
   collapsed verdict — otherwise a single bad ladder step is indistinguishable
   from a broken kernel.

**Nondeterministic candidates.** Any candidate whose params include `SPLIT_K > 1`
(or whose block declares `nondeterministic=True`) runs the gate **N=5 times and
passes only if the worst run passes**; `max_abs_err` is the max over runs and the
row carries `stochastic=1`, `gate_runs=5`.

**Time bound.** Derived, not asserted: the gate executes 10 cases (3 seeds × 3
magnitudes + zeros) per candidate, ×5 for `SPLIT_K>1`. At 4096³ a candidate is
~6.9 ms (ARCHITECTURE §1.6), so the candidate executions alone are ≥ 69 ms before
the oracle or the 16.8M-element comparison — the earlier "< 50 ms per candidate"
was arithmetically impossible. Budget **≤ 250 ms per deterministic candidate at
4096³** with the oracle cached per `(problem, seed, magnitude)`, and report the
gate's share of sweep wall clock.

**Test.** `tests/test_search_correctness.py` [partly GPU]: wrong kernel rejected;
correct kernel accepted; a NaN-producing kernel yields `verdict='nan'` rather
than an exception; `tol_scale` appears in the recorded row; a ladder step whose
oracle overflows yields `oracle_overflow` and does **not** appear in
`blacklist()`; two gate runs of the same split-K candidate on the same seed may
differ and the recorded verdict is the pessimistic one.

Plus a **near-miss negative control**: a kernel that drops the final K-block (or
applies the activation before the bias), run at `tol_scale = sqrt(K)/8`, asserted
**rejected**. The only negative control today is a grossly wrong kernel (bias
omitted), while `ACC_FP16` is accepted at a ~8× widened tolerance at K=4096 —
so every headline `ACC_FP16` win currently rests on a tolerance with no measured
false-negative rate.

**Decision.** If the fp32 oracle at 4096³ costs more than the benchmark it gates,
compute it once per `(problem, seed, magnitude)` and cache it in memory for the
sweep. If the near-miss control **passes** at `tol_scale = sqrt(K)/8`, the
tolerance is too loose and must be tightened before M7 — this is a decision that
blocks M7, not a note. If `ACC_FP16` fails the gate at every K, drop it from the
grid and record why — it is the single largest theoretical lever on this card, so
its elimination is itself worth publishing.

**Test (graph).** A captured and an uncaptured run of the same kernel agree
numerically.

---

## M6 — The search loop, and the first go/no-go [GPU] — 6 h

**Status.** CPU-complete: `ako/search/engine.py`, `ako/search/rank.py`,
`scripts/sweep.py`, and the stubbed engine tests. The GPU go/no-go
(`--ab 128x128,64x128 --shapes 1024,4096`) is **not** launched while
`noise_floor.py` holds the card. The wave model must use `DeviceCaps.sm_count`
(40 on this laptop), never a hardcoded 46.

**Skinny-K SiLU, measured and not publishable.** `scripts/winnable_region.py`
is the §1.6 cell (M=N=4096, K=64, bias+SiLU) against addmm-then-SiLU and
`torch.compile(mode="max-autotune")`, with gelu_tanh vs
`torch._addmm_activation` as the vendor control and 1024³ SiLU as the square
control. The template skips the inner-loop mask when the tile divides the
shape (`EVEN_K` / `EVEN_MN`). The run in
`reports/rtx3070/winnable_region.json` was taken at **24% GPU utilization**
(`publishable: false`). Every absolute series is `unstable` (cv > 0.10), so
milliseconds are not quotable. Paired ratios, the only mode M1 still allows:

- SiLU K=64: fused / addmm+SiLU **1.94×**, 95% CI [1.93, 2.34], excludes 1.
  Against compile **2.49×**, CI [2.30, 2.63]. Inductor logged
  `Not enough SMs to use max_autotune_gemm mode` on this 40-SM card, so that
  baseline is the compile path the laptop actually runs.
- GELU K=64 vs `_addmm_activation`: **1.40×**, CI [1.06, 1.63]. On this shape
  that API issued two CUDA launches, so this is not a win over a fused cuBLASLt
  epilogue.
- Square SiLU: **0.84×** vs addmm+SiLU, CI crosses 1. Not a win.

Do not put these numbers in the README until a run with utilization under 3%.

**CUDA graphs, same card, same caveat.** `scripts/graph_survival.py` replays
both sides inside a captured graph (`reports/rtx3070/graph_survival.json`,
utilization still 24%, `publishable: false`, every series `unstable`). Paired
ratios:

- Skinny K=64: the lead **survives** the graph, **1.71×** vs addmm+SiLU, CI
  [1.63, 1.91]. Launch overhead was not the win. That matches the traffic
  hypothesis in §1.6.
- Decode M=64, N=K=4096: uncaptured fused is **slower** (0.77×, CI [0.60, 0.86]).
  Under a graph it flips to **1.34×**, CI [1.15, 1.64]. §1.6 expected a launch
  win that graphs would erase. The measured order is the other way: eager
  dispatch loses, the graphed epilogue wins. Do not treat that flip as a
  ranking until the card is idle.

**Split-K was the wrong lever.** Atomic split-K on the decode GEMM
(M=64, N=K=4096, no epilogue, graphed) gets slower as the split grows: 1.03× at
split 1, **0.51×** at split 8 (`reports/rtx3070/splitk_decode.json`). Replacing
the atomic with a partial-tile store plus one reduction
(`_splitk_reduce_impl`) makes a fused epilogue legal — it runs on the sum, once —
but does not beat cuBLAS on that GEMM either. The cells that do, still under a
desktop load (utilization 36%, `publishable: false`, absolute series unstable),
reproduced with 21 pairs (`reports/rtx3070/reproduce_leads.json`):

- SiLU K=64, graphed: **1.74×** vs addmm+SiLU, CI [1.48, 1.95].
- GEMM M=16, N=K=4096, graphed, no split: **1.90×** vs `a @ b`, CI [1.66, 2.05].

Short rows are where this card's cuBLAS launch leaves SMs idle. More splits do
not fill them usefully. The tile that matches M does.

**The short-row region, one sweep.** `scripts/short_m_region.py` varies only M
at N=K=4096 fp16, tile height equal to M for 16/32/64 and 64 above that
(`reports/rtx3070/short_m_region.json`, utilization 38%, `publishable: false`).
Graphed paired ratios, Triton over `a @ b`:

| M | ratio | 95% CI | excludes 1 |
|---|---|---|---|
| 1 | 1.77× | [1.58, 2.97] | yes |
| 8 | 1.71× | [1.50, 2.61] | yes |
| 16 | 1.62× | [1.17, 2.27] | yes |
| 32 | 1.95× | [1.40, 2.99] | yes |
| 64 | 0.90× | crosses 1 | no |
| 128 | 0.94× | crosses 1 | no |
| 256 | 1.00× | [0.92, 1.14] | no |

The same M=16 tile **without** a graph is 0.93×, CI crosses 1. The device work
is ahead; eager launch overhead spends it. A bias (`addmm`) at M=16 stays ahead
(1.97×) but the interval reaches 1.09, so that cell is not the claim. Rule this
card can use, and only inside a CUDA graph: the matched tile for M≤32, cuBLAS
from M≥64. Absolute milliseconds stay unquotable.

**Objective.** Produce the first table showing a rank inversion, on three cells,
in minutes rather than hours.

**Files.** `ako/search/engine.py` (new, ~180 lines), `ako/search/rank.py` (new,
~60 lines, `prior ∈ {"none","nn"}` only), `scripts/sweep.py` (new, ~90 lines;
`--ab BM×BN,BM×BN` pins an explicit two-config comparison instead of a budgeted
search, which is what the M6 go/no-go needs).

**Measurable success criterion — a pinned A/B, not a budgeted sweep.**
`scripts/sweep.py --ab 128x128,64x128 --shapes 1024,4096 --dtype fp16` measures
exactly two configs (128×128 and 64×128, **same `num_warps`, `num_stages`,
`BLOCK_K`**) at both shapes, paired and interleaved, and completes in < 10 min.
Ties flagged, sweep `status` recorded.

The self-test claim is restricted to the **1024² direction**: 64 tiles / 46 SMs =
2 waves at 70% utilization for 128×128, against 128 tiles at 93% for 64×128, so
the wave model predicts 64×128 ahead there. It does **not** predict the flip back
at 4096² — 1024/46 = 23 waves at 97.0% versus 2048/46 = 45 waves at 98.9% still
marginally favours 64×128 — so the 4096² ordering is recorded as a measurement of
a different mechanism (data reuse, L2 traffic), and is a hypothesis, not a gate.

A budgeted sweep cannot serve as the go/no-go here: at ~5.3 s per candidate
(see the cost model), `--budget 120` measures ~22 of 441 configs in enumeration
order on an empty DB (`prior='none'`), which need not contain both tiles at both
shapes. A null result would then be fully explained by sampling rather than by a
broken timing path.

**Test.** `tests/test_search_engine.py` [CPU]: with a stub block registry and a
fake timer, `search()` never benchmarks a candidate that failed the gate, always
records infeasible rows, respects `budget_s`, and leaves the incumbent unchanged
when `compare_paired` reports `significant=False`.

**Decision — this is the project's first real fork.**
- 64×128 ahead of 128×128 at 1024², significantly → the harness is validated and
  M7 runs.
- **128×128 ahead at 1024² → the harness is broken**, not the thesis. That single
  direction is what the wave model predicts from first principles; its inversion
  means the L2 flush, the preheat or the timing path is wrong. Do not run the
  full grid until it reproduces.
- Whatever happens at 4096² is data. It does not gate M7 in either direction.

---

## M7 — The full grid, with Inductor as a competitor [GPU] — 12 h (mostly unattended)

**Objective.** The dense rank-inversion map, with `torch.compile` inside the same
table.

**Files.** `ako/search/blocks.py` (modify: register `torch_compile` with the
three modes and `torch_addmm_act`), `scripts/warm_compile.py` (new, ~50 lines:
the unmeasured warm pass), `scripts/report_ranking.py` (new, ~120 lines).

**Grid — 14 cells, fp16, `act=gelu_tanh`.** The count is derived from this
explicit list of `(M, N, K)` triples, not from summing the rows: the 4096³ cell
is deliberately shared between the compute-bound and wave-quantization rows.

```
decode              (1,4096,4096) (8,4096,4096) (16,4096,4096)
                    (32,4096,4096) (64,4096,4096)
compute-bound       (512,4096,4096) (2048,4096,4096) (4096,4096,4096)
small-K             (4096,4096,32) (4096,4096,64) (4096,4096,128) (4096,4096,256)
wave quantization   (1024,1024,4096) + (4096,4096,4096) [shared with compute-bound]
unaligned           (1001,5121,3071)
                                                        => 14 unique cells
```

| row | what it can show |
|---|---|
| decode | split-K wins where cuBLAS is weakest; measured **with and without** CUDA-graph capture via `ako/benchmark/graph.py` (M5) |
| compute-bound | parity with cuBLAS, reported as parity |
| small-K | traffic 168.82 → 34.60 MB at K=64; the one honest large fusion win |
| wave quantization | the M6 self-test, at full budget |
| unaligned | `M, N, K` all indivisible by 8 — the masked/partially-vectorized Triton path and the point where cuBLAS drops off tensor cores. Replaces the earlier `1000×5120×3072`, every dimension of which is a multiple of 8, so it exercised nothing the rest of the grid does not. A cuBLAS generic-kernel fallback here is a much stronger result. |

**Warm compile pass, before the sweep and outside its budget.**
`scripts/warm_compile.py` builds every `cost_class="autotune"` candidate
(`torch_compile(max-autotune)`) and the frozen incumbent `torch.compile(default)`
for all 14 cells, with `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` pinned.
Its wall clock is reported as search cost (invariant 7) but is not charged to
`budget_s`. Without it, criterion 4 below has no execution path: Inductor's
`max-autotune` runs its own exhaustive Triton GEMM autotune per new shape —
minutes — so one such candidate would consume an entire `budget_s=180` cell, and
the incumbent's compile cost would be charged to every cell unconditionally.

**Measurable success criteria.**
1. All 4 compositions pass the gate on **every** cell, or the failures are
   recorded with reasons.
2. The top-1 `(composition, params)` changes on **≥3 cells**, reported separately
   for composition and for variant.
3. K=64 shows **≥3× versus eager**; 4096³ shows 0.9–1.1× versus
   `torch._addmm_activation` (the fused cuBLASLt path — `torch.addmm` alone
   reaches only the `BIAS` epilogue and is the weaker baseline).
4. The fraction of cells where AKO beats `torch.compile(max-autotune)` by >10% is
   published, **including if it is zero**.
5. Every row carries `peak_mem_mb`; `fast_p` (correct AND > p× eager) is reported
   alongside latency.
6. Every cell carries a canary p50 (invariant 4). Cells deviating >5% from the
   sweep median canary are marked contaminated and excluded from the report —
   "mostly unattended" on the machine's display GPU means mid-sweep contamination
   is expected, and it leaves first-vs-last drift at 0%.

**Test.** `tests/test_report_ranking.py` [CPU]: on a fixture `.sqlite`, the
report marks a sub-IQR delta as a tie, keeps only `status='ok'` sweeps (so
`drifted` **and** `aborted`/`NULL` sweeps are both excluded), drops contaminated
cells, excludes rows from a foreign `toolchain_key`, and prints the 4%
wall-clock figure next to the 57% traffic figure for the 4096³ row.

**Decision.** See "hypotheses to invalidate" — this milestone is where the
pre-registered kill condition fires.

---

## M8 — Transfer: does the DB predict an unseen shape? [GPU + CPU] — 6 h

**Objective.** The actual contribution: a ranking predicted for a shape never
measured.

**Files.** `ako/search/rank.py` (modify: `prior="borda3"`, then `prior="model"` —
a `numpy.linalg.lstsq` fit of **`log(speedup_vs_incumbent)`** on ~10 features:
log m/n/k, fractional wave count, `smem_bytes/smem_per_block`,
`k/(BLOCK_K*SPLIT_K)`, arithmetic intensity, one-hot composition, `ACC_FP16`),
`scripts/report_transfer.py` (new, ~80 lines).

The target is the paired ratio against the sweep's frozen incumbent, **not**
`log(wall_p50_ms)`. M1's decision forbids pooling absolute latencies across
sweeps and thermal states on a card with no lockable clocks; a paired ratio is
invariant to clock state by construction. `fit_features` states this in its
contract (ARCHITECTURE §3.5) and excludes rows whose `incumbent_hash` differs.

**Measurable success criterion.** Leave-one-cell-out over the 14 cells, **plus**
leave-one-*family*-out (5 folds: decode, compute-bound, small-K, wave, unaligned)
reported alongside it — the 14 cells are 5 tightly clustered families, so a
held-out decode cell's nearest measured neighbour is always another decode cell
and leave-one-cell-out is not independent. With `budget ∈ {1,2,5,10,20,60}`
candidates measured, the prior selects a candidate whose latency is **within 5%
of that cell's exhaustive best at budget ≤ 5**. Reported as a regret-versus-
budget curve per cell, budget in wall-clock seconds as well as candidate count.

**Pre-registered decision rule, fixed now and not after seeing the number.**
"Below `prior='nn'` on a majority of cells" is a coin flip at 9/14 with no
significance rule, and it decides the project's actual contribution. Instead:
report the **per-cell paired regret difference** (`model` − `nn`) with a
bootstrap CI over cells, and require the CI on the mean difference to **exclude
0** in the favourable direction, on both the cell folds and the family folds.
If 14 cells cannot support a CI that separates the two priors, say so and
**expand the grid before M8 runs**, not after.

**Test.** `tests/test_rank.py` [CPU]: on a synthetic DB where latency is a known
function of the features, `prior="model"` recovers the argmin in ≤2 candidates;
on a DB with one measured shape, `prior="borda3"` degrades exactly to
`prior="nn"`.

**Decision.** No cost model is written before `prior="nn"` is measured and shown
insufficient. Budget for model complexity is fixed at "reads `fit_features`, fits
in under a second, no new dependency" until lstsq is measured insufficient. No
GBDT, no RL, no LLM.

---

## M9 — The dtype axis, then extension, only if M7 and M8 hold [GPU] — 10 h+

M9 survives the single-card constraint, with a renamed objective: its first item
stopped being a nice-to-have and is now the **second half of the ARCHITECTURE
§1.5 claim** — the rank inversion across dtype, on the one card.

Not planned in detail beyond that. In priority order, each gated on the previous:
bf16 as a second dtype pass over the M7 grid (a new column value, no code —
but the bf16 path is **NOT TESTED** on CPU at M4, see that milestone, so its
first real execution is the M5 GPU gate and this hour budget assumes that gate
already passed); a
`reduce_row_v1` template reusing `apply_epi` and the same DB, which is what adds
RMSNorm and softmax; `torch.library.custom_op` export of the winner (~10 lines)
so `torch.compile` can call it inside a real model.

**Removed 2026-08-13** — "one rented A100 or L4 hour to measure the fraction of
shapes whose optimal composition differs from the 3070's". The project has one
RTX 3070 and no second card will be rented or borrowed, so the inter-hardware
rank-inversion measurement is **NOT TESTED** and out of scope. It is recorded
here rather than deleted so a reader sees it was planned, then ruled out by
hardware, not by a result. See ARCHITECTURE.md §1.5.

Explicitly never in this plan: multi-GPU scheduling, MLIR, CUDA C backends, an
agentic layer.

---

## Hypotheses to invalidate

The most important section. Each row is a benchmark result that would make us
abandon a direction rather than defend it.

| # | Hypothesis | What would invalidate it | What we do then |
|---|---|---|---|
| M1 | The 3070 is stable enough to rank candidates | p50 drift ≥ 15% over a 15-minute sweep after preheat, or CV > 0.10 on a repeated identical measurement | Stop. No ranking is publishable here. Renting a card with lockable clocks is out of scope (one 3070, 2026-08-13), so the only branch left is reducing the project to the CPU-side composer, the schema and the feasibility map. |
| M3 | Free arithmetic filtering meaningfully cuts the space | The shared-memory predicate removes < 10% of the grid on cc 8.6 | Drop `feasible()` to a smoke check; the "39% never compiles" claim is dead and must not appear in any report. |
| M4 | One constexpr template covers the whole variant space | The compile time per constexpr tuple exceeds ~3 s, making a 400-candidate sweep cost hours | Cut the grid to a hand-picked 40 configs and report that search width — do not pretend to 441. |
| M5 | `ACC_FP16` is a usable variant | It fails the gate at every K ≥ 1024 even at `tol_scale = sqrt(K)/8` | Remove it from the grid and publish the elimination. The 40.6 → 81.3 TFLOP/s lever is then unavailable on this card and every "% of peak" claim uses 40.6. |
| M6 | The harness can measure a known effect | In the pinned A/B, 128×128 beats 64×128 at 1024² (the direction the wave model predicts). The 4096² ordering is not part of this test. | The harness is broken. Fix the L2 flush / preheat / timing path. Do not run M7. |
| M7 | The **composition** ranking flips across shapes | Top-1 `composition` is constant across all 14 cells and only `params` moves | **Pre-registered pivot, declared before the first sweep:** the composer collapses to a hardcoded enum of 4, and all remaining effort goes to the variant ranker and to transfer. This is data, not defeat — but only because it is written down now. |
| M7 | Fusion pays somewhere on this card | K=64 shows < 2× versus eager | The fusion hypothesis has no supporting measurement on a 3070. Report it as such and move the headline entirely onto variant ranking. Do not headline the 4096³ 57% traffic reduction, which buys 4%. |
| M7 | There is a region where AKO beats the shipped compiler | `torch.compile(max-autotune)` is within measurement noise of the best found candidate on **all** cells | Claims 1 and 2 evaporate. The project is then a search-cost paper, not a kernel paper: the only survivor is the transfer curve. Decide this from the M7 data, not from hope. |
| M7 | Tiling optima are roughly composition-independent | On one cell, a full per-composition sweep gives a different top-1 tile than the top-8 re-measure shortcut | The shortcut is invalid and the grid cost triples. This check is a **scheduled task inside M7**, not a note in a risks section, because the headline number depends on it. |
| M8 | A measured ranking transfers to an unseen shape | The bootstrap CI on the mean per-cell regret difference versus `prior="nn"` includes 0, on cell folds or on family folds | Publish the negative result: nearest-measured-shape is the state of the art, and here is the map of where it fails. Do not escalate to a GBDT chasing a number. |
| M8 | The search budget is affordable | Reaching within 5% of the best requires > 20 candidates on a majority of cells | The transfer claim is dead at useful budgets. Report the regret curve anyway — it is the number Ansor and Inductor do not publish. |
| M9 | The **dtype** ranking flips on one card (the half of the §1.5 claim that replaces the inter-GPU one) | Top-1 `(composition, params)` is identical between fp16 and bf16 on every cell of the M7 grid, with paired CIs excluding a flip | The dtype axis is dead. Restate §1.5 as shape-only and say so in the report — do not keep the dtype wording and quietly drop the measurement. |

Deliberately **not** a row: *"the ranking transfers to another GPU"*. It needs a
second card, so on this project it has no falsification path at all — an
untestable row in a table of kill conditions is worse than none. It is recorded
as **NOT TESTED** in ARCHITECTURE.md §1.5 and as a removed step in M9. No other
row lost its testability under the single-card constraint: M1–M8 are all measured
on one 3070 by construction.

---

## Cost model of the work itself

Measured assumptions to re-check at M6, currently estimates. **The per-candidate
cost is derived from `compare_paired`'s actual call count, not from a flat
number**, because invariant 3 mandates a paired interleaved comparison for every
candidate and that dominates at the compute-bound cells.

With `compare_paired`'s defaults (`ako/benchmark/bench_op.py:272-283` —
`pairs=11`, `warmup=5`, `repeats=20`) and a 6.9 ms candidate at 4096³,
`_calibrate_inner` returns `inner = max(1, int(1.0/6.9)) = 1`, so one `bench_op`
is ~25 calls ≈ 172 ms, one pair ≈ 345 ms, one comparison ≈ 3.8 s.

| grid row | candidate | paired compare | per candidate (cold) |
|---|---|---|---|
| 4096³ compute-bound | ~6.9 ms | ≈ 3.8 s | ≈ **5.3 s** (JIT + gate + paired) |
| small-K (K=64) | ~107 µs | `inner ≈ 9`, ≈ 0.5 s | ≈ 2.1 s |
| decode (M ≤ 64) | ~75 µs | `inner ≈ 13`, ≈ 0.5 s | ≈ 2.1 s |

- Full exhaustive 441-config grid at 4096³ ≈ 39 min **per cell**; the full grid is
  therefore ≈ **11 h cold**, not 6 h — which is why M7 is budgeted at 12 h.
- `budget_s = 180` at a compute-bound cell buys ~34 candidates, **7.7% of the
  grid**, not "3 min per cell" of meaningful coverage. Either drop `pairs` and
  `repeats` for the screening pass, or run the two-stage protocol: a cheap
  **unpaired** screen over the ordered candidate list, then a **paired**
  re-measure of the top-k only. Pick one and state it in `engine.py`; the
  reported search cost must be the one actually paid.
- Compile cost is **not** in these numbers: `cost_class="autotune"` candidates
  (`torch.compile(max-autotune)`) and the frozen incumbent are built in the
  unmeasured warm pass with `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR`
  pinned, and every build is bounded by the block's `max_build_s`. Warm re-runs
  of the Triton candidates are ~0.5 s each.

8 GB caps three fp16 square matrices at N ≈ 34000; the display compositor takes
200–600 MB and is a pre-flight gate, not a surprise. Declared here rather than
discovered as an OOM.
