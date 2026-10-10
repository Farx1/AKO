"""Tests for benchmark utilities."""

from __future__ import annotations

import pytest
import torch

from ako.benchmark.bench_op import bench_op, parse_dtypes
from ako.benchmark.report import BenchmarkReport
from ako.dispatcher.hardware import get_device_info
from ako.ops.rmsnorm_ref import rmsnorm_ref
from ako.ops.rope_ref import build_rope_tables, rope_ref


def test_bench_op_cpu() -> None:
    x = torch.randn(4, 64, 128)
    weight = torch.ones(128)
    result = bench_op(rmsnorm_ref, (x, weight), warmup=2, repeats=5, kernel_name="ref")
    assert result.latency_p50_ms >= 0
    assert result.kernel_name == "ref"
    assert len(result.samples_ms) == 5
    assert result.latency_p95_ms >= result.latency_p50_ms


def test_bench_op_passes_call_kwargs() -> None:
    x = torch.randn(1, 8, 2, 16)
    cos, sin = build_rope_tables(8, 16)
    result = bench_op(
        rope_ref,
        (x, cos, sin),
        warmup=1,
        repeats=3,
        kernel_name="rope_ref",
        call_kwargs={"style": "interleaved"},
    )
    assert result.dtype == "float32"


def test_parse_dtypes() -> None:
    assert parse_dtypes(["fp16", "bf16", "fp32"]) == [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    with pytest.raises(ValueError, match="Unknown dtype"):
        parse_dtypes(["int8"])


def test_benchmark_report() -> None:
    report = BenchmarkReport(operation="rmsnorm", device=get_device_info("cpu"))
    report.add_comparison(
        shape=[8, 2048, 4096],
        baseline_ms=0.42,
        candidate_ms=0.27,
        max_error=0.00031,
        selected_kernel="rmsnorm_triton",
    )
    data = report.to_dict()
    assert data["comparisons"][0]["speedup"] > 1.5
