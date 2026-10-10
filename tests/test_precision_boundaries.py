"""Boundary precision tests; interpreter cases execute actual Triton bodies."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ako.search.constraints import global_mem_bytes
from ako.search.spec import Problem


def test_promoted_boundary_storage_is_budgeted():
    problem = Problem(m=3, n=5, k=17)
    base = global_mem_bytes({}, problem)
    assert global_mem_bytes({"FP32_BOUNDARIES": 1}, problem) - base == 3 * 5 * 4
    assert global_mem_bytes({"FP32_BOUNDARIES": 3}, problem) - base == 2 * 3 * 5 * 4


@pytest.mark.parametrize("mask", [-1, "fp32", True, 0.5])
def test_invalid_precision_storage_mask(mask):
    with pytest.raises(ValueError, match="FP32_BOUNDARIES"):
        global_mem_bytes({"FP32_BOUNDARIES": mask}, Problem())


def test_exact_cancellation_witness_and_local_rounding():
    a = torch.tensor([[2048.0, 1.0]], dtype=torch.float16)
    b = torch.ones((2, 1), dtype=torch.float16)
    bias = torch.tensor([-2048.0], dtype=torch.float16)
    accumulator = a.float() @ b.float()
    assert accumulator.item() == 2049.0
    assert accumulator.half().item() == 2048.0
    assert torch.relu(accumulator + bias.float()).half().item() == 1.0
    assert torch.relu(accumulator.half().float() + bias.float()).half().item() == 0.0


@pytest.mark.skipif(importlib.util.find_spec("triton") is None, reason="requires Triton CPU interpreter")
def test_actual_triton_boundary_schedules():
    script = r'''
import json
import torch
from ako.search.kernels.mm_fused import launch
from ako.search.space import partitions, composition_name
params = dict(BLOCK_M=16, BLOCK_N=32, BLOCK_K=32, num_warps=4, num_stages=2)
a = torch.tensor([[2048., 1.]], dtype=torch.float16)
b = torch.ones((2, 1), dtype=torch.float16)
bias = torch.tensor([-2048.], dtype=torch.float16)
rows = []
for segments in partitions(("matmul", "bias", "act")):
    for mask in range(1 << (len(segments)-1)):
        value = launch(a, b, bias, {**params, "FP32_BOUNDARIES":mask}, segments, "relu")
        expected = 0.0 if segments[0] == ("matmul",) and not mask & 1 else 1.0
        assert value.item() == expected, (segments, mask, value)
        assert value.dtype == torch.float16
        rows.append((composition_name(segments), mask, value.item()))
for mask in [-1, True, 0.5, 2]:
    try:
        launch(a, b, bias, {**params, "FP32_BOUNDARIES":mask}, (("matmul",), ("bias", "act")), "relu")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid mask accepted")
print(json.dumps(rows))
'''
    result = subprocess.run([sys.executable, "-c", script],
                            env={**os.environ, "TRITON_INTERPRET": "1"},
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)) == 9
