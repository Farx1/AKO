# AKO research: choosing precision between composed blocks

## Assessment and prior art

AKO already has a numerical gate, measured kernel selection, a trial database,
composition enumeration, resource filtering and a parameterized GEMM/epilogue.
These make it a useful base for systems research. It does not yet adapt arbitrary
programs to arbitrary hardware; several search blocks still have `build=None`.

The general idea of composable kernels is established:

| Primary source | Existing contribution | Consequence for AKO |
|---|---|---|
| [Mirage, OSDI 2025](https://arxiv.org/abs/2405.05751) | Multi-level superoptimization and probabilistic equivalence checking | Composition alone is not novel |
| [TileLang](https://arxiv.org/abs/2504.17577) | Composable tiled programming and scheduling | Prefer building a decision layer over existing languages |
| [ThunderKittens](https://arxiv.org/abs/2410.20399) | Hardware-oriented tile primitives | A collection of blocks is not sufficient differentiation |
| [Sound Mixed-Precision Optimization with Rewriting](https://arxiv.org/abs/1707.02118) | Joint rewriting and precision tuning with sound analysis | Joint precision/optimization is not new |
| [Accuracy-Aware Mixed-Precision GPU Auto-Tuning, TPDS 2026](https://ac.liacs.nl/blog/accuracy-tuning-tpds) | Tunes precision and kernel parameters together | Must compare against accuracy-aware tuning, not only naive baselines |

A narrower opportunity is **budgeted selection of compositions and intermediate
representations under explicit numerical contracts**. This targeted review does not
establish scientific priority. Broad claims in earlier planning documents about what
no other system does need rechecking against this literature.

## A reproduced counterexample in AKO

Sharing the epilogue function does not guarantee equal fused and unfused results.
The unfused GEMM normally materializes its accumulator at input precision before
the next kernel reloads it. With exactly representable fp16 inputs,

$$A=[2048,1],\quad B=[1,1]^T,\quad b=-2048,$$

the fp32 accumulator is exactly 2049. fp16 spacing at 2048 is 2; ties-to-even
rounding gives $Q_{16}(2049)=2048$. Therefore

$$Q_{16}(\operatorname{ReLU}(AB+b))=1,$$
$$Q_{16}(\operatorname{ReLU}(Q_{16}(AB)+b))=0.$$

The intermediate matmul is correctly rounded, yet its rounding error dominates the
small final result. Local validation of each block does not establish equivalence
to the fused reference. Both programs are legitimate under different semantics;
this study explicitly chooses fp32 dot/epilogue followed by a final fp16 cast as its
oracle. It does not declare PyTorch's staged semantics universally incorrect.

Scaling A and the bias by $2^e$, for $e\in\{-4,-2,0,2,4\}$, preserves this witness
with outputs $2^e$ versus zero, without overflow or subnormals. Zero-padding K and
duplicating rows/columns gives the same result on ragged shapes.

This is known floating-point cancellation applied to a concrete AKO assumption,
not a claim to have discovered a new numerical phenomenon.

## Implementation

The optional `FP32_BOUNDARIES` parameter is an integer bitmask: bit i promotes the
materialized output of segment i to fp32. It cannot address the final output, whose
dtype remains unchanged. Zero preserves the existing execution path; the default
search grid is unchanged. Candidate hashing already includes parameter values.
The memory filter conservatively adds a complete fp32 tensor for every marked cut.
This extends its estimate; it does not prove the existing allocation model exact.

Split-K's first workspace was already fp32. Marking it again is redundant and
conservatively overbudgeted. Split-K performance is not studied here; a future
enumerator should canonicalize equivalent effective precision schedules.

There are three abstract choices at every inter-operation gap: fuse, materialize
at input precision, or materialize at fp32. Thus an n-operation chain has at most
$3^{n-1}$ such plans before legality filtering and duplicate elimination. For the
three-operation chain, four partitions become nine partition/precision plans.

## Reproduction and results

```bash
TRITON_INTERPRET=1 PYTHONPATH=. python scripts/research_precision_boundaries.py
PYTHONPATH=. pytest tests/test_precision_boundaries.py tests/test_candidates.py tests/test_trial_db.py -q
```

The [complete report](../reports/research/precision_boundaries.json) contains every
case and source SHA256s. These are **actual Triton bodies executed through its CPU
interpreter**, not compiled GPU measurements. The interpreter does not reproduce
all GPU lowering, reduction-order, atomic or performance behavior. bf16 is excluded
because of the documented Triton 3.7.1 interpreter dot limitation.

- 90 constructed executions: two shapes × five scales × nine plans.
- 30 fail the chosen output budget, as expected for these adversarial cases.
- Among the 40 legacy mask-zero cases, 20 fail. Promoting the first boundary
  repairs those 20; promoting only a later boundary cannot recover lost information.
- 81 ordinary seeded cases: three seeds × three activations × nine plans. All pass
  the existing magnitude-scaled output budget.

| Composition, scale 1 | Default result | First boundary promoted | Fused oracle |
|---|---:|---:|---:|
| `mm+bias+act` | 1 | no boundary | 1 |
| `mm+bias|act` | 1 | 1 | 1 |
| `mm|bias+act` | 0 | 1 | 1 |
| `mm|bias|act` | 0 | 1 | 1 |

The constructed failure fraction is not a real-workload failure-rate estimate.
Passing 81 random cases does not prove correctness on all inputs. Intermediate
exponentials can emit an interpreter overflow warning; the saved outputs/errors
are finite. No GPU speedup is claimed.

## Conditional composition bound

Suppose ideal blocks $f_i$ are $L_i$-Lipschitz on a domain containing both ideal and
perturbed trajectories, and implementation/rounding errors are uniformly bounded
by $\delta_i$ on that domain. Triangle inequality yields
$E_i\le L_iE_{i-1}+\delta_i$. By induction, from $E_0=0$,

$$E_n\le\sum_{i=1}^{n}\delta_i\prod_{j=i+1}^{n}L_j.$$

This standard forward-error bound motivates allocating precision across boundaries.
Empirical maxima on random samples cannot be substituted for uniform guarantees.
Dot-accumulation error must also be counted. Bias and ReLU have Lipschitz constants
1, but an absolute error of 1 can still be a 100% relative error after cancellation.

## Research priorities and decision gates

1. **Boundary precision conditioned on numerical sensitivity.** Determine whether
   only a few boundaries need fp32 under a specified input domain. Compare against
   uniform low precision, all-fp32 intermediates, exhaustive joint search and an
   accuracy-aware tuner. Include checking overhead and tuning cost.
2. **Budgeted transfer to unseen shapes.** Use validated trials to shortlist at most
   five candidates, targeting less than 5% latency regret. Nearest-shape transfer and
   equal-budget random search are mandatory baselines. This is an objective, not a result.
3. **Context-aware composition.** Explain the existing decode regression by comparing
   isolated calls, decoder loops and CUDA graph replay. Include dispatch, conversions
   and launches when deciding which blocks are actually fast together.

First run the precision schedules on the existing GPU, validating adversarial and
ordinary inputs before timing survivors against cuBLASLt epilogues and Inductor.
Record paired repeated timings, numerical error, peak memory and compilation cost.
Evaluate unseen shapes and held-out input distributions. A useful accuracy/speed or
tuning-cost trade-off would justify continuing; a numerical repair alone is not a
publishable performance advance. Cross-device adaptation remains an untested hypothesis.
