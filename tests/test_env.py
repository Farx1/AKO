"""M1 checks: environment sampling, the L2 flush, and the noise-floor verdict."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from ako.benchmark import env as env_module
from ako.benchmark.bench_op import bench_op
from ako.ops.rmsnorm_ref import rmsnorm_ref

_SPEC = importlib.util.spec_from_file_location(
    "noise_floor", Path(__file__).resolve().parent.parent / "scripts" / "noise_floor.py"
)
noise_floor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(noise_floor)


@pytest.fixture
def without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(env_module.shutil, "which", lambda _: None)
    env_module._throttle_field.cache_clear()
    yield
    env_module._throttle_field.cache_clear()


def test_sample_env_without_nvidia_smi(without_nvidia_smi) -> None:
    """Missing binary yields None fields, never an exception: a machine without
    nvidia-smi must still be able to record a measurement."""
    sample = env_module.sample_env()
    assert set(sample) == {"sm_clock_mhz", "temp_c", "power_w", "throttle_reasons"}
    assert all(value is None for value in sample.values())


def test_preflight_without_nvidia_smi(without_nvidia_smi) -> None:
    assert env_module.preflight() == {"checked": False, "compute_apps": [],
                                      "utilization_gpu": None}


def test_throttle_bits_decode() -> None:
    assert env_module._throttle_names("0x0000000000000044") == [
        "sw_power_cap", "hw_thermal_slowdown"
    ]
    assert env_module._throttle_names("0x0") == []
    assert env_module._throttle_names(None) is None


def test_flush_l2_is_a_noop_on_cpu() -> None:
    """There is no L2 to flush off CUDA, so the flag must not change anything."""
    x = torch.randn(4, 64, 128)
    weight = torch.ones(128)
    common = dict(args=(x, weight), warmup=2, repeats=5, kernel_name="ref")
    flushed = bench_op(rmsnorm_ref, flush_l2=True, **common)
    plain = bench_op(rmsnorm_ref, flush_l2=False, **common)
    for result in (flushed, plain):
        assert result.l2_state == "n/a"
        assert result.l2_flush_ms == 0.0
        assert len(result.samples_ms) == 5
        assert result.latency_p50_ms > 0
    assert flushed.to_dict()["l2_state"] == "n/a"


def _points(p50s, cv=0.01):
    return [{"p50_ms": value, "cv": cv} for value in p50s]


def test_verdict_ladder() -> None:
    """The gate itself, on simulated measurements — the one piece of M1 that must
    not be discovered to be wrong at 4 a.m. on real data."""
    flat = noise_floor.verdict(_points([6.90, 6.91, 6.89, 6.92]))
    assert flat["verdict"] == "pass"
    assert flat["points"] == 4
    assert flat["drift_first_last"] == pytest.approx((6.92 - 6.90) / 6.90, abs=1e-5)

    # 8% first-to-last drift: below the stop threshold, above the pass threshold.
    drifting = noise_floor.verdict(_points([6.90, 7.00, 7.10, 7.45]))
    assert drifting["verdict"] == "paired_only"
    assert 0.07 < drifting["drift_first_last"] < 0.09

    assert noise_floor.verdict(_points([6.90, 7.20, 7.60, 8.30]))["verdict"] == "stop"
    # Direction is irrelevant: clocks boosting is as unrankable as clocks sagging.
    assert noise_floor.verdict(_points([8.30, 7.60, 7.20, 6.90]))["verdict"] == "stop"

    # Drift can be zero and the card still unusable, so cv gates independently.
    assert noise_floor.verdict(_points([6.90] * 4, cv=0.07))["verdict"] == "paired_only"
    assert noise_floor.verdict(_points([6.90] * 4, cv=0.20))["verdict"] == "stop"

    # At M1's 20 points the cv gate is the max, exactly as the milestone states.
    twenty = _points([6.90] * 20)
    twenty[7]["cv"] = 0.40
    assert noise_floor.verdict(twenty)["verdict"] == "stop"
    # Over an overnight trace one contaminated point must not decide the project.
    overnight = _points([6.90] * 100)
    overnight[7]["cv"] = 0.40
    assert noise_floor.verdict(overnight)["verdict"] == "pass"
    assert noise_floor.verdict(overnight)["cv_max"] == 0.4

    assert noise_floor.verdict(_points([6.90]))["verdict"] == "insufficient"
    assert noise_floor.verdict([])["verdict"] == "insufficient"


def test_verdict_reports_the_resolvable_effect() -> None:
    """The number that answers M1's question: what a candidate must beat to be seen."""
    spread = noise_floor.verdict(_points([6.90, 7.00, 6.95, 6.93]))
    assert spread["p50_spread_pct"] == pytest.approx(100 * 0.10 / 6.94, abs=0.05)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_l2_flush_changes_the_number() -> None:
    """A flushed and an unflushed 1024-cubed measurement must differ. If they do
    not, the flush is dead and every wave-quantization number is measured hot."""
    probe = noise_floor.flush_probe("cuda")
    assert probe["l2_state"] == ["off", "flushed"]
    assert probe["flush_cost_ms"] > 0
    assert probe["flush_effective"], probe
