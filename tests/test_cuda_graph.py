"""CUDA-graph capture (ROADMAP.md M5).

A captured and an uncaptured run of the same kernel must agree numerically.
Capture is skipped off CUDA and when the kernel allocates unpredictably.
"""

from __future__ import annotations

import pytest
import torch

from ako.benchmark.bench_op import bench_op
from ako.benchmark.graph import try_capture
from ako.validation.numerical import assert_close


def test_try_capture_is_a_no_op_on_cpu() -> None:
    x = torch.randn(8, 8)
    captured = try_capture(lambda t: t @ t, (x,))
    assert captured.captured is False
    assert captured.replay is None
    assert captured.reason == "not_cuda"


def test_bench_op_graph_flag_on_cpu_records_uncaptured() -> None:
    x = torch.randn(16, 16)
    result = bench_op(lambda t: t @ t, (x,), warmup=1, repeats=2, graph=True)
    assert result.graph_captured == 0
    assert result.latency_p50_ms >= 0


@pytest.mark.gpu
def test_captured_and_uncaptured_matmul_agree() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    a = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    b = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    static_out = torch.empty(64, 64, device="cuda", dtype=torch.float16)

    def mm() -> torch.Tensor:
        static_out.copy_(a @ b)
        return static_out

    eager = a @ b
    mm()
    captured = try_capture(mm, (static_out,))
    assert captured.captured, captured.reason
    static_out.zero_()
    captured.replay()
    assert_close(static_out, eager, atol=1e-3, rtol=1e-3)

    plain = bench_op(lambda x, y: x @ y, (a, b), warmup=2, repeats=5, graph=False, flush_l2=False)
    graphed = bench_op(lambda x, y: x @ y, (a, b), warmup=2, repeats=5, graph=True, flush_l2=False)
    assert graphed.graph_captured == 1
    assert plain.graph_captured == 0
