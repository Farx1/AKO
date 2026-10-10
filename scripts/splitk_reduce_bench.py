"""Time partial-tile split-K against cuBLAS, graphed, on the decode shape.

Atomic split-K lost (reports/rtx3070/splitk_decode.json). This is the same grid
with a store of partials and one reduction, which can also carry SiLU once.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ako.benchmark.bench_op import compare_paired
from ako.benchmark.env import preflight
from ako.search.builders import triton_mm_fused
from ako.search.correctness import apply_act
from ako.search.spec import Problem, caps
from ako.validation.numerical import error_budget, max_abs_error

TILES = (
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "ACC_FP16": False, "num_warps": 4, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "ACC_FP16": False, "num_warps": 4, "num_stages": 3},
)


def main() -> None:
    try:
        health = preflight()
        idle = bool(health.get("checked"))
    except RuntimeError as exc:
        health = {"checked": True, "error": str(exc)}
        idle = False
        print(exc)

    device = torch.device("cuda")
    device_caps = caps(device)
    torch.matmul(torch.randn(1024, 1024, device=device), torch.randn(1024, 1024, device=device))
    torch.cuda.synchronize()

    specs = [
        ("gemm", Problem(m=64, n=4096, k=4096, dtype="fp16", act="none", bias=False)),
        ("silu", Problem(m=64, n=4096, k=4096, dtype="fp16", act="silu", bias=True)),
        ("gemm_m16", Problem(m=16, n=4096, k=4096, dtype="fp16", act="none", bias=False)),
    ]
    rows = []
    for name, prob in specs:
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        a = torch.randn(prob.m, prob.k, dtype=torch.float16, generator=g).to(device)
        b = torch.randn(prob.k, prob.n, dtype=torch.float16, generator=g).to(device)
        bias = torch.randn(prob.n, dtype=torch.float16, generator=g).to(device) if prob.bias else torch.empty(1, device=device, dtype=torch.float16)
        if prob.act == "none":
            def vendor(x, w, bias_, _prob=prob):
                del bias_
                return x @ w
            ref = a @ b
        else:
            def vendor(x, w, bias_, _prob=prob):
                return apply_act(torch.addmm(bias_, x, w), _prob.act).to(dtype=x.dtype)
            ref = vendor(a, b, bias)
        budget = error_budget(ref)
        # M=16 is not a multiple of BLOCK_M=64, so use a 16-row tile there.
        tiles = TILES
        if prob.m < 64:
            tiles = (
                {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "ACC_FP16": False, "num_warps": 4, "num_stages": 3},
                {"BLOCK_M": 16, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "ACC_FP16": False, "num_warps": 4, "num_stages": 3},
            )
        for base in tiles:
            for split_k in (1, 2, 4, 8):
                params = {**base, "SPLIT_K": split_k}
                label = {"cell": name, "BLOCK_N": base["BLOCK_N"], "BLOCK_K": base["BLOCK_K"], "SPLIT_K": split_k, "BLOCK_M": base["BLOCK_M"]}
                fused = triton_mm_fused(prob, params)
                err = max_abs_error(ref, fused(a, b, bias))
                if err > budget:
                    print(label, "REJECTED", err)
                    rows.append({**label, "rejected_error": err})
                    continue
                stats = compare_paired(
                    vendor, fused, (a, b, bias), pairs=5, warmup=2, repeats=10, flush_l2=True, graph=True
                )
                print(
                    f"{name} {label} {stats['speedup_median']}x ci={stats['speedup_ci95']} "
                    f"sig={stats['significant']} cap={stats['graph_captured']}"
                )
                rows.append({**label, "error": err, "vs_vendor_graph": stats})
        del a, b, bias, ref
        torch.cuda.empty_cache()

    out = Path("reports/rtx3070/splitk_reduce.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device_caps.name,
        "sm_count": device_caps.sm_count,
        "preflight": health,
        "publishable": idle,
        "note": "speedup is vendor/fused under a CUDA graph. >1 means the partial-tile split-K is faster. Atomic split-K is the previous file.",
        "rows": rows,
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
