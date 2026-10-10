"""Execute actual Triton bodies with TRITON_INTERPRET=1 PYTHONPATH=.

Constructed fp16 cancellation cases and seeded ordinary inputs. No GPU timing.
bf16 is excluded because of the documented interpreter dot limitation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
import triton

from ako.search.kernels.mm_fused import launch
from ako.search.space import composition_name, partitions
from ako.validation.numerical import error_budget, max_abs_error

ROOT = Path(__file__).resolve().parents[1]
PARAMS = {"BLOCK_M": 16, "BLOCK_N": 32, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}


def measure(a, b, bias, act, metadata):
    z = a.float() @ b.float() + bias.float()
    if act == "relu":
        ref = torch.relu(z)
    elif act == "silu":
        ref = z * torch.sigmoid(z)
    else:
        ref = 0.5 * z * (2 * torch.sigmoid(2 * 0.7978845608 * (z + 0.044715 * z**3)))
    ref = ref.half()
    rows = []
    for segments in partitions(("matmul", "bias", "act")):
        for mask in range(1 << (len(segments) - 1)):
            actual = launch(a, b, bias, {**PARAMS, "FP32_BOUNDARIES": mask}, segments, act)
            error = max_abs_error(ref, actual)
            rows.append({**metadata, "act": act, "composition": composition_name(segments),
                         "fp32_boundaries": mask, "first_reference": float(ref[0, 0]),
                         "first_actual": float(actual[0, 0]), "max_abs_error": error,
                         "budget": error_budget(ref), "passes_gate": error <= error_budget(ref)})
    return rows


def run():
    if os.environ.get("TRITON_INTERPRET") != "1":
        raise RuntimeError("Set TRITON_INTERPRET=1 before importing Triton")
    torch.set_num_threads(2)
    rows, random_rows = [], []
    for shape in [(1, 1, 2), (3, 5, 17)]:
        m, n, k = shape
        for exponent in [-4, -2, 0, 2, 4]:
            scale = 2.0**exponent
            a = torch.zeros((m, k), dtype=torch.float16)
            a[:, 0], a[:, 1] = 2048 * scale, scale
            b = torch.ones((k, n), dtype=torch.float16)
            bias = torch.full((n,), -2048 * scale, dtype=torch.float16)
            rows.extend(measure(a, b, bias, "relu", {"shape": shape, "scale": scale}))
    for seed in [0, 1, 2]:
        gen = torch.Generator().manual_seed(seed)
        a = torch.randn((3, 17), generator=gen).half()
        b = torch.randn((17, 5), generator=gen).half()
        bias = torch.randn((5,), generator=gen).half()
        for act in ["relu", "gelu_tanh", "silu"]:
            random_rows.extend(measure(a, b, bias, act, {"seed": seed, "shape": [3, 5, 17]}))
    report = {"mode": "triton_cpu_interpreter_not_gpu", "torch": torch.__version__, "triton": triton.__version__,
              "params": PARAMS, "source_sha256": {
                  str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [Path(__file__), ROOT / "ako/search/kernels/mm_fused.py", ROOT / "ako/search/constraints.py"]},
              "cancellation_cases": rows, "random_cases": random_rows}
    path = ROOT / "reports/research/precision_boundaries.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"cancellation_cases": len(rows), "cancellation_failures": sum(not r["passes_gate"] for r in rows),
                      "random_cases": len(random_rows), "random_failures": sum(not r["passes_gate"] for r in random_rows)}))


if __name__ == "__main__":
    run()
