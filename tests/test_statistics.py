"""Tests for the benchmark statistics: repetitions, CV guard, paired bootstrap."""

from __future__ import annotations

import random
import statistics

import pytest
import torch

from ako.benchmark.bench_op import (
    UNSTABLE_CV,
    BenchResult,
    bench_op,
    bootstrap_ci,
    bootstrap_ratio_ci,
    compare_paired,
)
from ako.ops.rmsnorm_ref import rmsnorm_ref


TRUE_RATIO = 1.4


def _fake_measure(rng: random.Random, mean_ms: float) -> float:
    """Simulated timing: known mean, 10% noise, never negative."""
    return mean_ms * (1.0 + rng.uniform(-0.1, 0.1))


def _result(run_p50s: tuple[float, ...]) -> BenchResult:
    return BenchResult(
        kernel_name="fake",
        shape=(1,),
        dtype="float32",
        device="cpu",
        latency_p50_ms=statistics.median(run_p50s),
        latency_p95_ms=max(run_p50s),
        latency_mean_ms=statistics.mean(run_p50s),
        latency_std_ms=statistics.stdev(run_p50s),
        samples_ms=list(run_p50s),
        run_p50s=run_p50s,
    )


def test_bootstrap_ci_contains_true_ratio() -> None:
    rng = random.Random(1234)
    ratios = [_fake_measure(rng, TRUE_RATIO) / _fake_measure(rng, 1.0) for _ in range(30)]
    low, high = bootstrap_ci(ratios, seed=0)
    assert low < TRUE_RATIO < high
    assert low > 1.0  # a real 40% difference is detected as significant


def test_bootstrap_ci_is_deterministic_and_ordered() -> None:
    values = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 0.8]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    low, high = bootstrap_ci(values, seed=7)
    assert low <= statistics.median(values) <= high
    assert bootstrap_ci([]) == (0.0, 0.0)


def test_cv_and_iqr_match_stdlib() -> None:
    runs = (1.00, 1.02, 0.99, 1.01, 0.98)
    result = _result(runs)
    assert result.cv == statistics.stdev(runs) / statistics.mean(runs)
    assert result.median_ms == statistics.median(runs)
    assert result.iqr_ms == pytest.approx(1.01 - 0.99)  # index-truncated p75 - p25
    assert not result.unstable


def test_unstable_flag_surfaces_in_json() -> None:
    noisy = _result((1.0, 2.0, 3.0, 0.5, 4.0))
    assert noisy.cv > UNSTABLE_CV
    payload = noisy.to_dict()
    assert payload["unstable"] is True
    assert payload["cv"] == round(noisy.cv, 4)
    assert payload["runs"] == 5


def test_bench_op_runs_are_independent_and_default_is_unchanged() -> None:
    x = torch.randn(2, 16, 32)
    weight = torch.ones(32)
    single = bench_op(rmsnorm_ref, (x, weight), warmup=1, repeats=4)
    assert len(single.run_p50s) == 1
    assert len(single.samples_ms) == 4  # historical behaviour preserved
    assert single.to_dict()["runs"] == 1

    repeated = bench_op(rmsnorm_ref, (x, weight), warmup=1, repeats=4, runs=3)
    assert len(repeated.run_p50s) == 3
    assert len(repeated.samples_ms) == 12
    assert repeated.to_dict()["runs"] == 3
    assert repeated.iqr_ms >= 0


def test_compare_paired_reports_bracketed_interval() -> None:
    x = torch.randn(2, 16, 32)
    weight = torch.ones(32)
    out = compare_paired(rmsnorm_ref, rmsnorm_ref, (x, weight), pairs=3, warmup=1, repeats=3)
    low, high = out["speedup_ci95"]
    assert out["pairs"] == 3
    assert low <= out["speedup_median"] <= high
    assert isinstance(out["significant"], bool)


def test_bootstrap_ratio_ci_separates_known_ratio_from_parity() -> None:
    rng = random.Random(1)
    slow = [_fake_measure(rng, 1.4) for _ in range(40)]
    fast = [_fake_measure(rng, 1.0) for _ in range(40)]
    ratio, low, high = bootstrap_ratio_ci(slow, fast)
    assert 1.0 < low <= ratio <= high
    assert low <= TRUE_RATIO <= high * 1.05

    _, low, high = bootstrap_ratio_ci(fast, list(reversed(fast)))
    assert low <= 1.0 <= high  # same samples: never significant


def test_speedup_ci_script_uses_raw_samples_and_never_invents_one() -> None:
    from scripts.speedup_ci import speedup_rows

    rng = random.Random(2)
    row = {"shape": [1, 8], "dtype": "float16"}
    report = {
        "results": [
            {
                **row,
                "kernel_name": "base",
                "samples_ms": [_fake_measure(rng, 2.0) for _ in range(30)],
            },
            {
                **row,
                "kernel_name": "cand",
                "samples_ms": [_fake_measure(rng, 1.0) for _ in range(30)],
            },
        ],
        "comparisons": [{**row, "selected_kernel": "cand", "best_baseline_kernel": "base"}],
    }
    (out,) = speedup_rows(report)
    assert out["significant"] is True and out["ci95"][0] > 1.0

    # A second run where another baseline is faster: pooled, the pick follows the data.
    rerun = {"results": [{**row, "kernel_name": "fast_base", "samples_ms": [0.5] * 30}]}
    (out,) = speedup_rows(report, rerun)
    assert out["baseline"] == "fast_base" and out["ci95"][1] < 1.0

    for r in report["results"]:
        del r["samples_ms"]  # an old report: no interval, not a made-up one
    (out,) = speedup_rows(report)
    assert out["ci95"] is None and out["significant"] is None
