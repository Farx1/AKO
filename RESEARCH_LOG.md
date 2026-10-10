# Research log

Single GPU: RTX 3070 Laptop, 40 SMs. Paired ratios only. Absolute milliseconds
are not quotable while a series is `unstable` or the card is above 3%
utilization. An entry is a lead until a later run agrees with it. Agreement
means the 95% interval sits on the same side of 1 (both exclude 1 in the same
direction, or both contain 1).

## Already measured, not yet this log's reproduction

| Lead | Run | Ratio | 95% CI | Relation to 1 |
|---|---|---|---|---|
| SiLU, M=N=4096, K=64, graphed vs addmm+SiLU | `graph_survival.json` | 1.71× | [1.63, 1.91] | above |
| same | `reproduce_leads.json` | 1.74× | [1.48, 1.95] | above |
| GEMM M=16, N=K=4096, graphed vs `a @ b` | `reproduce_leads.json` | 1.90× | [1.66, 2.05] | above |
| same, 11 pairs | `short_m_region.json` | 1.62× | [1.17, 2.27] | above |
| GEMM M=32, same, one sweep | `short_m_region.json` | 1.95× | [1.40, 2.99] | above |
| GEMM M=64, same, one sweep | `short_m_region.json` | 0.90× | crosses 1 | contains 1 |
| GEMM M=16, no graph | `short_m_region.json` | 0.93× | crosses 1 | contains 1 |
| Atomic split-K, M=64, split 8 | `splitk_decode.json` | 0.51× | below 1 | below |

Split-K, atomic or partial-tile, does not beat cuBLAS on the M=64 GEMM.
That negative is repeated inside `splitk_reduce.json` and is not reopened here.

## Pre-registered before the next run

Script: `scripts/reproduce_short_m.py`. 21 pairs, graphed unless noted.
Tile height equals M for 16, 32 and 64. N=K=4096, fp16, no bias, no split.
Vendor is `a @ b`. Speedup is vendor/fused; above 1 means Triton is faster.

- **H1.** Graphed M=16 and graphed M=32 both have a CI entirely above 1.
- **H2.** Graphed M=64 does not (CI contains 1, or lies below 1).
- **H3.** Ungraphed M=16 does not have a CI entirely above 1.

Pass: H1, H2 and H3 all hold. That reproduces the region rule (matched tile
inside a CUDA graph for M≤32, cuBLAS from M≥64). Fail any one: the rule is
not reproducible, and the next experiment is chosen from the miss.

## Run 2 — `reports/rtx3070/reproduce_short_m.json`

Utilization 28% at preflight, no CUDA compute process. `publishable: false`.
Every absolute series `unstable`. 21 pairs. Seed 1.

| Cell | Ratio | 95% CI | Relation to 1 | Hypothesis |
|---|---|---|---|---|
| M=16 graphed | 1.50× | [1.35, 1.72] | above | H1 holds |
| M=32 graphed | 1.13× | [1.03, 1.38] | above | H1 holds, lower bound is 1.03 |
| M=64 graphed | 0.97× | [0.85, 1.07] | contains 1 | H2 holds |
| M=16 no graph | 1.98× | [1.23, 2.51] | above | **H3 fails** |

The region rule does not pass. H1 and H2 agree with the first sweep: graphed
M=16 and M=32 stay above 1, graphed M=64 contains 1. H3 does not: the first
sweep's ungraphed M=16 contained 1 (0.93×), this run lies entirely above 1.
"Only inside a CUDA graph" is not reproducible.

M=16 graphed is the cell that has agreed three times (1.90×, 1.62×, 1.50×,
all CIs above 1). M=32's lead shrank from 1.95× to 1.13×. Direction agrees,
size does not. Do not quote M≤32 as a stable magnitude.

## Pre-registered after the H3 miss

Script: `scripts/reproduce_short_m_eager.py`. Same tile, 21 pairs, no graph.
The question is whether the ungraphed win is itself reproducible.

- **H4.** Ungraphed M=16 has a CI entirely above 1.
- **H5.** Ungraphed M=64 does not.
- **H6.** Ungraphed M=32 is recorded, not predicted. It becomes part of the
  claim only if its CI is entirely above 1 and H4 and H5 hold.

Pass of H4 and H5: the short-row lead does not need a CUDA graph, and the
M=64 boundary still holds. Fail H4: the ungraphed win was one noisy run, and
the reproducible result stays the graphed M=16 cell only.

## Run 3 — `reports/rtx3070/reproduce_short_m_eager.json`

Utilization 23%, no CUDA compute process. 21 pairs, no graph. Seed 2.

| Cell | Ratio | 95% CI | Relation to 1 | Hypothesis |
|---|---|---|---|---|
| M=16 | 0.95× | [0.73, 1.09] | contains 1 | **H4 fails** |
| M=32 | 0.78× | [0.60, 0.85] | below | H6: not a win |
| M=64 | 0.68× | [0.49, 0.88] | below | H5 holds |

Ungraphed M=16 has now been measured three times: 0.93× (contains 1),
1.98× (above), 0.95× (contains 1). The above-1 run does not repeat.
Without a graph the short-row tile is not faster, and at M=32 and M=64 it
is slower on this run.

## Result

Reproduced, and only this:

**Graphed GEMM, M=16, N=K=4096, fp16, tile 16×128×64, vs `a @ b`.**
Three runs, every 95% interval entirely above 1:

| Run | Pairs | Ratio | 95% CI |
|---|---|---|---|
| `reproduce_leads.json` | 21 | 1.90× | [1.66, 2.05] |
| `short_m_region.json` | 11 | 1.62× | [1.17, 2.27] |
| `reproduce_short_m.json` | 21 | 1.50× | [1.35, 1.72] |

Not part of the result:

- M≤32 as a block. M=32 graphed stayed above 1 twice, but 1.95× then 1.13×.
- "Requires a CUDA graph" as a proved mechanism. The ungraphed cell does not
  reproduce a win, which is consistent with graphs mattering, and one of the
  three ungraphed runs did win. Two failures out of three is enough to drop
  the ungraphed claim, not enough to explain the outlier.
- Absolute milliseconds. Every series above was `unstable`, and utilization
  was 23–38%. `publishable` stays false.

## Next

An idle-card run (utilization under 3%) of the graphed M=16 cell, same tile,
21 pairs. If that interval is also entirely above 1, the lead can leave this
log. Until then it stays a paired result taken on a busy desktop GPU.
The card is still near 26% with no CUDA process, so that run cannot start.

## Pre-registered: is the harness inventing the lead?

Script: `scripts/harness_self_test.py`. Same shape as the lead (M=16, N=K=4096,
fp16), 31 pairs, CUDA graph on both sides. Utilization is expected to stay
above 3%. This does not make the lead publishable. It only asks whether a
ratio above 1 can appear when the two sides are the same kernel.

- **H7.** Graphed `a @ b` versus itself: the CI contains 1, and the median is
  between 0.9 and 1.1. If this fails, every speedup in this log is suspect
  and the next work is to fix the timer, not to claim a result.
- **H8.** Graphed Triton tile 16×128×64 versus `a @ b`: the CI lies entirely
  above 1.2. Above 1 but below 1.2 counts as a miss: the earlier M=32 cell
  showed that "barely above 1" does not survive the next run.

Pass both: the lead survived a stricter bar and a same-kernel control, and
the following experiment is a stack of four of these GEMMs (a tiny MLP) to
see whether the ratio survives composition.

## Run 4 — `reports/rtx3070/harness_self_test.json`

Utilization 30%. 31 pairs. Order was still always A then B.

| Cell | Ratio | 95% CI | Relation to 1 |
|---|---|---|---|
| matmul vs itself | 0.79× | [0.61, 1.16] | contains 1 |
| Triton vs matmul | 2.53× | [2.01, 3.62] | above |

**H7 fails.** The interval contains 1, but the median is 0.79, outside
[0.9, 1.1]. A kernel measured second looks faster. **H8 is not counted.**
The 2.53× was produced by the same ordered timer. Do not quote it.

## Pre-registered after the order bug

`compare_paired` now alternates which side runs first. Same script, same
thresholds, rerun.

- **H9.** Graphed matmul versus itself: CI contains 1 and median is in
  [0.9, 1.1].
- **H10.** Graphed Triton versus matmul: CI lies entirely above 1.2.

Fail H9: the timer is still not sane, and no speedup from this machine is
quoted. Pass H9 and fail H10: the lead does not clear 1.2 once order is
removed. Pass both: only then does the M=16 cell stay a candidate, still
not a finished result while utilization is above 3%.

## Run 5 — alternating order, `reports/rtx3070/harness_self_test.json`

A first alternating attempt returned all-zero latencies and was discarded
(two measurements had overlapped on the card). This rerun was alone.
Utilization 39%.

| Cell | Ratio | 95% CI | Verdict |
|---|---|---|---|
| matmul vs itself | 1.04× | [0.79, 1.29] | H9 holds: contains 1, median in [0.9, 1.1] |
| Triton vs matmul | 3.44× | [2.14, 4.88] | H10 holds on this run only |

The same-kernel median moved from 0.79 (fixed order) to 1.04 (alternating).
The control interval is still about ±25%. Across runs the M=16 lower bound
has been 1.17, 1.35, 1.66 and now 2.14. The direction stays above 1. The
size does not. This is not an irrefutable magnitude, and utilization 39%
keeps `publishable` false. It is not a technological change: one graphed
GEMM of 16 rows on one busy laptop.

## Pre-registered: does it survive four layers?

Script: `scripts/mlp4_short_m.py`. Four GEMMs, M=16, each K=N=4096, fp16,
no bias, no activation, both sides CUDA-graphed, alternating pairs.

- **H11.** The Triton stack versus the matmul stack has a CI entirely above
  1.2.

If H11 fails, the single-GEMM lead does not compose, and there is no
end-to-end claim. If it holds, the next question is still an idle-card
repeat. Neither outcome finishes the project.

## Run 6 — four GEMMs, void

`reports/rtx3070/mlp4_short_m.json` first write: ratio 1.61×, CI [1.41, 1.79],
and **stack error NaN**. Unscaled fp16 weights overflow by the third layer
(values grow like K^(n/2)). A timed NaN is not a result. H11 is void.
The script now scales each weight by K^(-1/2) and refuses to time a non-finite
output. That rerun is H12, same speed bar: CI entirely above 1.2, and the
absolute error against the matmul stack must be finite and ≤ 1.

## Run 7 — scaled stack, one run

`reports/rtx3070/mlp4_short_m.json`. Utilization 38%. Error versus the matmul
stack: 0.0039 (finite, under the cap of 1). Ratio **1.64×**, CI [1.53, 1.84],
so the speed bar of H12 holds **once**.

Still not the objective:

- One run is a lead. The single GEMM's magnitude moved from 1.5× to 3.4×
  across repeats; this 1.64× has not been repeated.
- The card was at 38%. The same-kernel control on that kind of load had a
  CI about ±25% wide. This interval is tighter, and it has not been checked
  against a fresh same-kernel control in the same run.
- Four synthetic GEMMs on one laptop are not a change in how people run models.

## Next

Repeat the scaled four-GEMM stack, same script, and a same-kernel stack
control in the same process. Count it only if the control median stays in
[0.9, 1.1] and the Triton interval stays entirely above 1.2. An idle card
(utilization under 3%) is still required before any number leaves this log.
The objective, irrefutable evidence and a result that changes practice, is
not met.
