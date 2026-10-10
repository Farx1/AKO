"""Second run of the short-row rule, registered in RESEARCH_LOG.md before launch.

H1: graphed M=16 and M=32 both lie entirely above 1.
H2: graphed M=64 does not.
H3: ungraphed M=16 does not lie entirely above 1.

    python scripts/reproduce_short_m.py
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
_BASE = {
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "GROUP_M": 8,
    "SPLIT_K": 1,
    "ACC_FP16": False,
    "num_warps": 4,
    "num_stages": 3,
}


def _matmul(a, b, bias):
    del bias
    return a @ b


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

    jobs = [(16, True), (32, True), (64, True), (16, False)]
    rows = []
    for m, graph in jobs:
        prob = Problem(m=m, n=N, k=K, dtype="fp16", act="none", bias=False)
        tile = {**_BASE, "BLOCK_M": m}
        g = torch.Generator(device="cpu")
        g.manual_seed(1)
        a = torch.randn(m, K, dtype=torch.float16, generator=g).to(device)
        b = torch.randn(K, N, dtype=torch.float16, generator=g).to(device)
        dummy = torch.empty(1, device=device, dtype=torch.float16)
        fused = triton_mm_fused(prob, tile)
        ref = a @ b
        err = max_abs_error(ref, fused(a, b, dummy))
        budget = error_budget(ref)
        label = f"M={m} graph={graph}"
        if err > budget:
            print(label, "REJECTED", err)
            rows.append({"m": m, "graph": graph, "rejected_error": err})
            continue
        stats = compare_paired(
            _matmul, fused, (a, b, dummy), pairs=21, warmup=3, repeats=12, flush_l2=True, graph=graph
        )
        low, high = stats["speedup_ci95"]
        relation = "above" if low > 1 else "below" if high < 1 else "contains"
        print(
            f"{label} {stats['speedup_median']}x ci={stats['speedup_ci95']} "
            f"{relation} unstable={stats['unstable']} cap={stats['graph_captured']}"
        )
        rows.append({
            "m": m,
            "graph": graph,
            "block_m": m,
            "error": err,
            "relation_to_1": relation,
            "vs_matmul": stats,
        })
        del a, b, ref
        torch.cuda.empty_cache()

    path = Path("reports/rtx3070/reproduce_short_m.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight": health,
        "publishable": idle,
        "hypotheses": ["H1", "H2", "H3"],
        "note": "Second run of the short-row rule. See RESEARCH_LOG.md for the pass rule.",
        "rows": rows,
    }, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
