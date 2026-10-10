"""Where a short output row beats cuBLAS, and where it stops.

The reproduced lead is one point: M=16, N=K=4096, no split, under a CUDA graph.
This sweep varies only M. The tile height matches M when that is a legal dot
tile (16, 32, 64); taller problems use 64 so the change is the number of row
tiles, not a new search. M below 16 uses a masked 16-row tile.

    python scripts/short_m_region.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ako.benchmark.bench_op import compare_paired
from ako.benchmark.env import preflight
from ako.search.builders import triton_mm_fused
from ako.search.spec import Problem
from ako.validation.numerical import error_budget, max_abs_error

N = K = 4096
_BASE = {"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 1, "ACC_FP16": False, "num_warps": 4, "num_stages": 3}


def _block_m(m: int) -> int:
    if m in (16, 32, 64):
        return m
    return 64 if m > 64 else 16


def _vendor(prob: Problem):
    if prob.bias:
        def fn(a, b, bias):
            return torch.addmm(bias, a, b)
        return "addmm", fn

    def fn(a, b, bias):
        del bias
        return a @ b

    return "matmul", fn


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

    jobs = [("gemm", m, False, True) for m in (1, 8, 16, 32, 64, 128, 256)]
    jobs.append(("gemm_bias", 16, True, True))
    jobs.append(("gemm_uncaptured", 16, False, False))

    rows = []
    for name, m, bias, graph in jobs:
        prob = Problem(m=m, n=N, k=K, dtype="fp16", act="none", bias=bias)
        bm = _block_m(m)
        tile = {**_BASE, "BLOCK_M": bm}
        g = torch.Generator(device="cpu")
        g.manual_seed(0)
        a = torch.randn(m, K, dtype=torch.float16, generator=g).to(device)
        b = torch.randn(K, N, dtype=torch.float16, generator=g).to(device)
        bvec = torch.randn(N, dtype=torch.float16, generator=g).to(device) if bias else torch.empty(1, device=device, dtype=torch.float16)
        vendor_name, vendor = _vendor(prob)
        ref = vendor(a, b, bvec)
        fused = triton_mm_fused(prob, tile)
        err = max_abs_error(ref, fused(a, b, bvec))
        budget = error_budget(ref)
        label = f"{name} M={m} BM={bm} graph={graph}"
        if err > budget:
            print(label, "REJECTED", err, budget)
            rows.append({"cell": name, "m": m, "block_m": bm, "graph": graph, "rejected_error": err, "error_budget": budget})
            continue
        stats = compare_paired(
            vendor, fused, (a, b, bvec), pairs=11, warmup=2, repeats=10, flush_l2=True, graph=graph
        )
        print(
            f"{label} vs {vendor_name}: {stats['speedup_median']}x ci={stats['speedup_ci95']} "
            f"sig={stats['significant']} unstable={stats['unstable']} cap={stats['graph_captured']}"
        )
        rows.append({
            "cell": name,
            "m": m,
            "n": N,
            "k": K,
            "block_m": bm,
            "bias": bias,
            "graph": graph,
            "vendor": vendor_name,
            "error": err,
            "error_budget": budget,
            "vs_vendor": stats,
        })
        del a, b, bvec, ref
        torch.cuda.empty_cache()

    path = Path("reports/rtx3070/short_m_region.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight": health,
        "publishable": idle,
        "note": (
            "speedup is vendor/fused. >1 means the Triton tile is faster. "
            "gemm rows are graphed a@b. gemm_bias is graphed addmm. "
            "gemm_uncaptured is the M=16 lead without a CUDA graph. "
            "Absolute times stay unquotable when unstable or the card is busy."
        ),
        "rows": rows,
    }, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
