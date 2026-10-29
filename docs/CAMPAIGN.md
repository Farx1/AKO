# Running the measurement campaign (RTX 3070)

Operating manual for one evening in front of the GPU. Follow the sections in order.
Every command is copy-pasteable from the repository root.

All commands assume the GPU environment is active:

```bash
cd /path/to/AKO                  # wherever the repo lives
source activate_gpu.sh           # micromamba env "ako"; installs once via scripts/setup_micromamba_ako.sh
```

After `source activate_gpu.sh`, `python` is the GPU interpreter. If you use a CUDA-enabled
`.venv` instead, replace `python` with `.venv/bin/python` everywhere below.

---

## 1. Prerequisites — one command each

| # | Check | Command | Expected |
|---|-------|---------|----------|
| 1 | Driver + GPU visible | `nvidia-smi` | A table with `NVIDIA GeForce RTX 3070`, a driver version, and no `command not found` |
| 2 | PyTorch sees the GPU | `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` | `True NVIDIA GeForce RTX 3070 ...` |
| 3 | Triton imports and is usable | `python -c "import triton; from ako.ops.rmsnorm_triton import is_triton_available; print(triton.__version__, is_triton_available())"` | A version (e.g. `3.x.y`) and `True` |

`is_triton_available()` is the exact gate the dispatcher and every GPU test use: it is `True`
only when Triton imports **and** `torch.cuda.is_available()`.

One command that covers 2 and 3 at once, with a readable table:

```bash
python scripts/check_environment.py
```

If check 3 prints `False` while check 2 prints `True`, the Triton kernels will silently not be
measured: the sweeps fall back to PyTorch baselines and every speedup column reads `n/a`. Fix it
before spending the evening.

Optional but cheap correctness gate before an hour of benchmarking (~1–2 min):

```bash
pytest -q -m gpu --tb=line
```

All GPU tests must pass. A failure here means the kernels are wrong, not slow — measuring them is
pointless until it is green.

---

## 2. Preparing the machine

Measurements on a laptop GPU are only as trustworthy as the clocks behind them. Three things,
in order.

### 2.1 Close everything else using the GPU

```bash
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv
```

The list must be empty (or contain only your own shell). Browsers with hardware acceleration,
Discord, Steam, video calls, another Python session holding a CUDA context — all of them steal SMs
and memory bandwidth mid-sweep. The campaign compares kernels *against each other* inside one run;
a background load that comes and goes does not add noise evenly, it moves the winner.

Plug in AC power and set the Windows/WSL power profile to performance. On battery the driver caps
the clocks and the whole sweep measures the power policy, not the kernels.

### 2.2 Warm the card up

An idle 3070 laptop sits at low clocks, boosts hard on the first kernel, then throttles as it heats.
Benchmarked cold, the first shapes in a sweep run on a fast GPU and the last ones on a slow GPU —
which is exactly the comparison the report claims to make. Warm the card until temperature and
clock plateau, then measure.

The smoke run in section 3 is the warm-up. Run it, then watch until the numbers stop moving
(typically 2–4 minutes of load):

```bash
nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,power.draw --format=csv -l 2
```

Stop it with Ctrl-C once the temperature stops climbing and `clocks.current.sm` stays flat.

### 2.3 Lock the clocks — OPTIONAL, requires administrator rights

This is optional. **The campaign is valid without it**; locking clocks only lowers the variance
between repeats, which makes small differences (below ~10%) readable instead of ambiguous.

Read the maximum first:

```bash
nvidia-smi --query-gpu=name,clocks.max.sm,clocks.max.mem --format=csv
```

Then lock the SM clock to a value the card can sustain without throttling — roughly 70–80% of
`clocks.max.sm`. For a 3070 laptop reporting ~1620 MHz, 1200 MHz is a good sustained target:

```bash
sudo nvidia-smi -pm 1                      # persistence mode (Linux only)
sudo nvidia-smi --lock-gpu-clocks=1200,1200
```

Restore afterwards:

```bash
sudo nvidia-smi --reset-gpu-clocks
sudo nvidia-smi -pm 0                      # only if you enabled it above
```

**On WSL this usually fails.** The driver lives on the Windows side, so `-pm` and
`--lock-gpu-clocks` typically answer `Not Supported` from a Linux shell. Run the same
`nvidia-smi --lock-gpu-clocks=1200,1200` from an *elevated* Windows PowerShell, or skip it
entirely — again, the campaign stays valid, with higher variance. The health gate in section 3
tells you whether the variance is acceptable.

---

## 3. Smoke test — run this FIRST

```bash
python scripts/run_campaign.py --quick --output reports/smoke
```

**Duration:** the script announces `Estimated duration: ~2 min 0 s`. On CPU it finishes in about
8 seconds; on a cold GPU expect **1 to 3 minutes**, dominated by CUDA context creation and Triton
compiling each kernel for the first time. That compilation is cached, so the real campaign does not
pay it again.

It runs the seven steps on tiny shapes. Healthy output looks like this:

```
AKO campaign — mode quick, device cuda, dtypes fp16 bf16, output reports/smoke
Steps: environment, health, rmsnorm, rope, swiglu, serving, profile
Estimated duration: ~2 min 0 s

[1/7] environment (~5s)
  done in 0.4s
[2/7] health (~10s)
  bench drift 2.1% (tolerance 10%) — OK
  done in 3.2s
[3/7] rmsnorm (~25s)
  done in 18.4s
...
[7/7] profile (~20s)
  done in 12.1s
        Campaign summary — the six numbers
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ Metric                                  ┃ Value ┃
...
Campaign written to reports/smoke/campaign.json
```

What must be true before going further:

- The banner says **`device cuda`** and **`dtypes fp16 bf16`**. If it says `device cpu` and prints
  `No CUDA device: CPU dry-run`, stop — go back to section 1.
- Step 2 prints `bench drift X% (tolerance 10%) — OK`. This measures the same kernel twice and
  compares. On CUDA a drift above the tolerance **aborts the campaign** (exit code 2) — that is the
  gate protecting you from an evening of unusable numbers. If it aborts: close other workloads,
  let the card settle longer, then re-run. `--health-tolerance 0.15` raises the bar if you knowingly
  accept the noise.
- All seven steps say `done in ...`, none say `failed`.
- The summary table shows numbers, not `n/a` everywhere. On a working GPU the RMSNorm/RoPE/SwiGLU
  rows carry a value; `n/a` on those three rows means no Triton candidate was measured (check 3
  of section 1 is the cause).

`reports/smoke/` is throwaway — delete it whenever.

---

## 4. The full campaign

```bash
python scripts/run_campaign.py --output reports/rtx3070
```

**Duration:** the script announces `Estimated duration: ~34 min 5 s`. Budget **35 to 75 minutes**
in practice: the full run sets `AKO_ENABLE_COMPILE=1` (the honest competitor for AKO is
`torch.compile`, not the unfused reference), and `torch.compile` recompiles for each new shape and
dtype — 24 shape/dtype combinations for RMSNorm alone. Per-step progress and elapsed time are
printed as it goes.

Notes:

- `reports/rtx3070/` already contains the JSON from the previous GPU run. They will be
  **overwritten** — that is intended, and git still has the old ones.
- **Do not add `--resume` on the first run.** `--resume` skips any step whose output file already
  exists, so with the old files in place it would skip rmsnorm, rope and profile entirely. It exists
  to restart after a failure (section 6).
- Every artefact is flushed to disk after each step, so a crash at step 6 keeps steps 1–5.
- A failing step does not stop the run (except `health`, which aborts by design); it is recorded in
  `campaign.json` and the campaign continues.

Leave the machine alone while it runs. No browsing, no builds, no video.

---

## 5. What comes out

All files land in `reports/rtx3070/`.

| File | Contents | Use in the README |
|------|----------|-------------------|
| `campaign.json` | Per-step status/duration/errors, plus `summary` with the six headline numbers (RMSNorm decode & prefill, RoPE, SwiGLU, end-to-end decode, adaptive vs best fixed policy), all measured against the *best baseline* | The headline claims. Every "Nx faster" sentence quotes this file |
| `environment.json` | Python/torch/triton versions, CUDA runtime, device capability, memory, and `nvidia-smi` clocks/power-limit/temperature | The "Environment" / reproducibility block — proves the numbers came from a known machine state |
| `health.json` | The same kernel measured twice, drift, tolerance, verdict | "Known measurement limits" — the evidence that the bench itself was stable that evening |
| `baseline_rmsnorm.json` | `results[]`: per shape/dtype/kernel latency p50/p95/mean/std, GPU time, inner iterations. `comparisons[]`: reference vs best baseline vs Triton, `speedup_vs_best_baseline`, `max_error` | The RMSNorm results table, and two of the three figures (`baseline-gap.png`, `shape-sensitivity.png`) |
| `rope_benchmark.json` | Same structure for RoPE across heads/head-dims/styles | The RoPE table and `fixed-strategy-penalty.png` |
| `swiglu_benchmark.json` | Same structure for SwiGLU over `(batch, seq_len, intermediate_dim)` | The SwiGLU row. New op: no figure and no `docs/RESULTS.md` support yet |
| `serving_simulation.json` | Per-policy (eager / compile / triton / adaptive) total, prefill, decode, TTFT p50/p95 on a replayed request trace, plus `crossover_batch` | The "adaptive beats every fixed strategy" claim and the crossover point |
| `model_profile.json` | Prefill and decode latency for torch / ako / torch_compile decoder stacks, launch overhead, `speedup_vs_best_baseline`, top operators | The end-to-end decoder number — the only figure that is not a microbenchmark |
| `logs/<step>.log` | Full stdout+stderr of each sub-script | Debugging only; not published |

The number worth quoting is always `speedup_vs_best_baseline`, never `speedup` (which is measured
against the unfused reference and flatters the result). A value below `1.00x` means the baseline
wins on that shape class — publish it anyway, that is the point of the negative-results section.

---

## 6. If a step fails

The run continues and the failure is recorded. At the end you get:

```
Failed steps: rope (see reports/rtx3070/logs)
```

Read the cause first:

```bash
cat reports/rtx3070/logs/rope.log
```

**Re-run only the failed steps** — `--resume` skips every step whose output file already exists,
which after a partial run is exactly the successful ones:

```bash
python scripts/run_campaign.py --output reports/rtx3070 --resume
```

**Or re-run one step directly**, with the exact arguments the campaign uses (the
`AKO_ENABLE_COMPILE=1` prefix is required: without it the `torch.compile` baseline is missing and
`speedup_vs_best_baseline` becomes incomparable with the other steps):

```bash
# rmsnorm
AKO_ENABLE_COMPILE=1 python scripts/benchmark_rmsnorm.py --device cuda \
  --output-dir reports/rtx3070 --repeats 30 --dtypes fp16 bf16 \
  --batches 1 8 --seq-lens 1 512 2048 --hidden-dims 2048 4096

# rope
AKO_ENABLE_COMPILE=1 python scripts/benchmark_rope.py --device cuda \
  --output-dir reports/rtx3070 --repeats 30 --dtypes fp16 bf16 \
  --batches 1 8 --seq-lens 1 512 2048 --num-heads 32 --head-dims 128

# swiglu
AKO_ENABLE_COMPILE=1 python scripts/benchmark_swiglu.py --device cuda \
  --output-dir reports/rtx3070 --repeats 30 --dtypes fp16 bf16 \
  --batches 1 8 --seq-lens 1 512 2048 --intermediate-dims 5632

# serving
AKO_ENABLE_COMPILE=1 python scripts/simulate_serving.py --device cuda \
  --output-dir reports/rtx3070 --dtype fp16 \
  --preset small --requests 64 --max-batch 8 \
  --median-prompt 256 --median-decode 64 --max-prompt 1024

# profile
AKO_ENABLE_COMPILE=1 python scripts/profile_model.py --device cuda \
  --output-dir reports/rtx3070 --dtype fp16 --repeats 30 \
  --batch-size 1 --seq-len 512 --hidden-dim 2048 \
  --n-layers 4 --n-heads 32 --n-kv-heads 4
```

Then refresh `campaign.json` and its summary by re-running the campaign with `--resume` (it will
find every output present, run nothing, and just rewrite the summary).

Two failures with a known cause:

- **`health` aborts** (exit code 2, "the GPU is not stabilized"). Nothing after it is trustworthy.
  Close other workloads, warm the card (section 2.2), re-run. `--health-tolerance 0.15` if you
  accept the noise.
- **A step fails with CUDA out of memory.** Re-run that one step directly with a smaller grid —
  drop `2048` from `--seq-lens`, or `4096` from `--hidden-dims`. Note the change: the report grid no
  longer matches the campaign default.

---

## 7. Regenerating the figures

```bash
python scripts/plot_results.py --report-dir reports/rtx3070 --output-dir docs/images
```

Takes a few seconds and rewrites three PNGs:
`docs/images/fixed-strategy-penalty.png`, `docs/images/baseline-gap.png`,
`docs/images/shape-sensitivity.png`.

It reads only `baseline_rmsnorm.json` and `rope_benchmark.json`; if both are missing it exits with
`No reports in ...`. There is no SwiGLU figure yet.

Optionally regenerate the results tables:

```bash
python scripts/summarize_reports.py --report-dir reports/rtx3070 --output docs/RESULTS.md
```

Be aware it also expects three artefacts the campaign does **not** produce — `environment.txt`,
`pytest_gpu.txt` and `selected_config_{fp16,bf16}.json` — and prints `*(missing ...)*` rows in their
place. To fill them in:

```bash
python scripts/check_environment.py | tee reports/rtx3070/environment.txt
pytest -q -m gpu --tb=line | tee reports/rtx3070/pytest_gpu.txt

AKO_ENABLE_COMPILE=1 python scripts/optimize_model.py --device cuda --dtype fp16 --fresh \
  --batch-size 4 --seq-len 1024 --hidden-dim 2048 --num-heads 16 --head-dim 128 \
  --output-dir reports/rtx3070
mv -f reports/rtx3070/selected_config.json reports/rtx3070/selected_config_fp16.json

AKO_ENABLE_COMPILE=1 python scripts/optimize_model.py --device cuda --dtype bf16 --fresh \
  --batch-size 4 --seq-len 1024 --hidden-dim 2048 --num-heads 16 --head-dim 128 \
  --output-dir reports/rtx3070
mv -f reports/rtx3070/selected_config.json reports/rtx3070/selected_config_bf16.json
```

(That is what `scripts/run_gpu_validation.sh` does; it predates the campaign script and covers a
different, older grid. Use the campaign, not that script.)

---

## 8. What to send back

Paste the final summary table from the terminal, and send these exact paths:

```
reports/rtx3070/campaign.json
reports/rtx3070/environment.json
reports/rtx3070/health.json
reports/rtx3070/baseline_rmsnorm.json
reports/rtx3070/rope_benchmark.json
reports/rtx3070/swiglu_benchmark.json
reports/rtx3070/serving_simulation.json
reports/rtx3070/model_profile.json
```

Plus, **only if a step failed**, the matching log:

```
reports/rtx3070/logs/<step>.log
```

One command that bundles all of it, logs included:

```bash
tar czf ako-campaign-$(date +%F).tar.gz reports/rtx3070
```

If only one file can travel, send `reports/rtx3070/campaign.json`: it carries the six headline
numbers, the per-step status and the error message of anything that failed.

---

## 9. How a number becomes a claim

What each timing is, and when a speedup counts.

- **Warmup**: `warmup` untimed calls per kernel and shape (10 by default in
  `bench_op`) before any sample, so compilation, autotuning and allocator
  growth stay out of the samples.
- **Timing**: calls are timed in batches of ~1 ms (`TARGET_BATCH_MS`), with
  wall clock around the batch and CUDA events inside it. One sample is the
  batch time divided by the number of calls. The L2 is flushed before every
  call and the calibrated flush cost is subtracted (`l2_state` in the JSON).
- **Samples**: every row of a sweep report now carries `samples_ms`, the raw
  per-call samples, next to `median_ms`, `p25_ms`, `p75_ms` and `iqr_ms`.
- **Interval**: `python scripts/speedup_ci.py <report.json>` recomputes each
  speedup as `median(best baseline) / median(candidate)` with a seeded
  percentile bootstrap (2000 resamples, the two sides resampled
  independently). A speedup is called significant only when its 95 % interval
  excludes 1. Paired experiments (`compare_paired`) use the median of per-pair
  ratios instead, measured in alternation.
- **Clocks**: `scripts/run_campaign.py` and `scripts/noise_floor.py` sample SM
  clock, temperature, power and throttle reasons through `nvidia-smi`
  (`ako/benchmark/env.py`). Read them before trusting a sweep.
- **Seeds**: inputs come from `torch.randn` under the default generator unless
  a script seeds its own (`torch.Generator().manual_seed(k)`); the bootstrap is
  seeded (`--seed`, default 0).

Run-to-run variance: one sweep is one process, so run it in several fresh
processes (`--output reports/rtx3070/samples/runN`) and pass all the reports to
`speedup_ci.py`; it pools their samples and re-picks the best baseline on the
pooled medians. `reports/rtx3070/samples/` holds such a campaign (5 runs). The
older reports directly under `reports/rtx3070/` predate `samples_ms`, so
`speedup_ci.py` prints `n/a` for them rather than an invented interval.
