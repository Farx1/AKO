"""Reproduce the two cells whose graph CI excluded 1, with more pairs.

Skinny SiLU (traffic fusion) and M=16 GEMM (short row, where cuBLAS is often
weak). Desktop load is expected. The question is whether the lower end of the
paired interval stays above 1.
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
from ako.search.spec import Problem

def main() -> None:
    try:
        health = preflight()
        idle = bool(health.get("checked"))
    except RuntimeError as exc:
        health = {"checked": True, "error": str(exc)}
        idle = False
        print(exc)

    device = torch.device("cuda")
    torch.matmul(torch.randn(1024, 1024, device=device), torch.randn(1024, 1024, device=device))
    torch.cuda.synchronize()

    rows = []

    silu = Problem(m=4096, n=4096, k=64, dtype="fp16", act="silu", bias=True)
    tile = {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 1, "ACC_FP16": False, "num_warps": 4, "num_stages": 3}
    a = torch.randn(4096, 64, device=device, dtype=torch.float16)
    b = torch.randn(64, 4096, device=device, dtype=torch.float16)
    bias = torch.randn(4096, device=device, dtype=torch.float16)

    def vendor_silu(x, w, bias_):
        return apply_act(torch.addmm(bias_, x, w), "silu").to(dtype=x.dtype)

    stats = compare_paired(vendor_silu, triton_mm_fused(silu, tile), (a, b, bias), pairs=21, warmup=3, repeats=12, flush_l2=False, graph=True)
    print("skinny", stats["speedup_median"], stats["speedup_ci95"], stats["significant"], stats["unstable"])
    rows.append({"cell": "skinny_silu_graph", **stats})

    short = Problem(m=16, n=4096, k=4096, dtype="fp16", act="none", bias=False)
    stile = {"BLOCK_M": 16, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 1, "ACC_FP16": False, "num_warps": 4, "num_stages": 3}
    sa = torch.randn(16, 4096, device=device, dtype=torch.float16)
    sb = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    dummy = torch.empty(1, device=device, dtype=torch.float16)

    def vendor_mm(x, w, bias_):
        del bias_
        return x @ w

    stats = compare_paired(vendor_mm, triton_mm_fused(short, stile), (sa, sb, dummy), pairs=21, warmup=3, repeats=12, flush_l2=True, graph=True)
    print("m16", stats["speedup_median"], stats["speedup_ci95"], stats["significant"], stats["unstable"])
    rows.append({"cell": "m16_gemm_graph", **stats})

    path = Path("reports/rtx3070/reproduce_leads.json")
    path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight": health,
        "publishable": idle,
        "note": "Third look at the two leads. speedup is vendor/fused, graphed. publishable stays false while utilization is above 3%.",
        "rows": rows,
    }, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
