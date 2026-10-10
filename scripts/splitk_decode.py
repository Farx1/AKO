"""Two hypotheses the roadmap names and this card has not measured.

1. Decode GEMM (M=64, N=K=4096, no epilogue). Output tiles do not fill 40 SMs,
   so SPLIT_K should raise occupancy. Compared with ``a @ b``, under a CUDA graph,
   because an uncaptured launch is not how a decode step runs.

2. Stationary desktop load. The card sits near 24% with no CUDA process. M1
   forbids absolute rankings. This script also reports the coefficient of
   variation of the *paired ratio* on the skinny-K SiLU cell already measured.
   A tight ratio under a load that does not move is a different claim from an
   absolute ranking, and it does not rewrite the M1 stop rule.

    python scripts/splitk_decode.py --output reports/rtx3070/splitk_decode.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch

from ako.benchmark.bench_op import bench_op, bootstrap_ci, compare_paired
from ako.benchmark.env import preflight
from ako.search.builders import triton_mm_fused
from ako.search.constraints import feasible
from ako.search.correctness import apply_act
from ako.search.spec import Problem, caps
from ako.validation.numerical import error_budget, max_abs_error

_TILES = (
    {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 3, "ACC_FP16": False},
    {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 3, "ACC_FP16": False},
    {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 8, "num_stages": 3, "ACC_FP16": False},
)


def _matmul(a, b, bias=None):
    del bias
    return a @ b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/rtx3070/splitk_decode.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this measurement needs CUDA")

    try:
        health = preflight()
        idle = bool(health.get("checked"))
    except RuntimeError as exc:
        health = {"checked": True, "error": str(exc)}
        idle = False
        print(exc)

    device = torch.device("cuda")
    device_caps = caps(device)
    torch.matmul(torch.randn(2048, 2048, device=device), torch.randn(2048, 2048, device=device))
    torch.cuda.synchronize()

    prob = Problem(m=64, n=4096, k=4096, dtype="fp16", act="none", bias=False)
    g = torch.Generator(device="cpu")
    g.manual_seed(0)
    a = torch.randn(prob.m, prob.k, dtype=torch.float16, generator=g).to(device)
    b = torch.randn(prob.k, prob.n, dtype=torch.float16, generator=g).to(device)
    ref = a @ b
    budget = error_budget(ref)
    # The template always takes a bias pointer. Epilogue code 0 never loads it.
    dummy = torch.empty(prob.n, device=device, dtype=torch.float16)

    rows = []
    for split_k in (1, 2, 4, 8):
        for base in _TILES:
            params = {**base, "SPLIT_K": split_k}
            reason = feasible(params, prob, device_caps)
            label = {k: params[k] for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "SPLIT_K")}
            if reason:
                print(f"  skip {label} {reason}")
                rows.append({"tile": label, "skipped": reason})
                continue
            fused = triton_mm_fused(prob, params)
            err = max_abs_error(ref, fused(a, b, dummy))
            if err > budget:
                print(f"  {label} REJECTED {err} budget {budget}")
                rows.append({"tile": label, "rejected_error": err, "error_budget": budget})
                continue
            stats = compare_paired(
                _matmul, fused, (a, b, dummy), pairs=5, warmup=2, repeats=12, flush_l2=True, graph=True
            )
            print(
                f"  {label} graph {stats['speedup_median']}x ci={stats['speedup_ci95']} "
                f"sig={stats['significant']} captured={stats['graph_captured']}"
            )
            rows.append({"tile": label, "error": err, "vs_matmul_graph": stats})

    # Ratio stability of the already-winning skinny SiLU cell, graphed.
    silu = Problem(m=4096, n=4096, k=64, dtype="fp16", act="silu", bias=True)
    tile = {
        "BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "SPLIT_K": 1,
        "ACC_FP16": False, "num_warps": 4, "num_stages": 3,
    }
    sa = torch.randn(silu.m, silu.k, device=device, dtype=torch.float16)
    sb = torch.randn(silu.k, silu.n, device=device, dtype=torch.float16)
    bias = torch.randn(silu.n, device=device, dtype=torch.float16)
    fused_silu = triton_mm_fused(silu, tile)

    def vendor_silu(x, w, bias_):
        return apply_act(torch.addmm(bias_, x, w), "silu").to(dtype=x.dtype)

    ratios = []
    for _ in range(15):
        va = bench_op(vendor_silu, (sa, sb, bias), warmup=2, repeats=10, flush_l2=False, graph=True)
        vb = bench_op(fused_silu, (sa, sb, bias), warmup=2, repeats=10, flush_l2=False, graph=True)
        if vb.latency_p50_ms > 0:
            ratios.append(va.latency_p50_ms / vb.latency_p50_ms)
    low, high = bootstrap_ci(ratios, seed=0)
    ratio_cv = statistics.stdev(ratios) / statistics.mean(ratios) if len(ratios) > 1 else 0.0
    stability = {
        "pairs": len(ratios),
        "speedup_median": round(statistics.median(ratios), 4),
        "speedup_ci95": [round(low, 4), round(high, 4)],
        "ratio_cv": round(ratio_cv, 4),
        "significant": bool(ratios) and (low > 1.0 or high < 1.0),
        "graph_captured": bool(va.graph_captured and vb.graph_captured),
    }
    print(
        f"skinny ratio cv={stability['ratio_cv']} median={stability['speedup_median']}x "
        f"ci={stability['speedup_ci95']}"
    )

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device_caps.name,
        "sm_count": device_caps.sm_count,
        "preflight": health,
        "publishable": idle,
        "note": (
            "Decode rows: speedup is matmul/splitk under a CUDA graph, >1 means split-K is faster. "
            "skinny_silu_ratio_cv is the variation of that paired ratio, not of the absolute times. "
            "It does not override the M1 stop on absolute rankings."
        ),
        "decode_splitk": rows,
        "skinny_silu_graph_ratio": stability,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
