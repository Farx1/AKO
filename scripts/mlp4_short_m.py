"""Four short GEMMs in a row. Registered as H11 in RESEARCH_LOG.md.

    python scripts/mlp4_short_m.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ako.benchmark.bench_op import compare_paired
from ako.benchmark.env import preflight
from ako.search.kernels.mm_fused import launch
from ako.validation.numerical import max_abs_error

M, N = 16, 4096
TILE = {
    "BLOCK_M": 16,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "GROUP_M": 8,
    "SPLIT_K": 1,
    "ACC_FP16": False,
    "num_warps": 4,
    "num_stages": 3,
}


def main() -> None:
    try:
        health = preflight()
        idle = bool(health.get("checked"))
    except RuntimeError as exc:
        health = {"checked": True, "error": str(exc)}
        idle = False
        print(exc)

    device = torch.device("cuda")
    g = torch.Generator(device="cpu")
    g.manual_seed(4)
    x = torch.randn(M, N, dtype=torch.float16, generator=g).to(device)
    weights = [
        torch.randn(N, N, dtype=torch.float16, generator=g).to(device) * (N ** -0.5)
        for _ in range(4)
    ]
    dummy = torch.empty(1, device=device, dtype=torch.float16)

    def matmul_stack(inp):
        y = inp
        for w in weights:
            y = y @ w
        return y

    def triton_stack(inp):
        y = inp
        for w in weights:
            y = launch(y, w, dummy, TILE, segments=(("matmul",),), act="none")
        return y

    ref = matmul_stack(x)
    out = triton_stack(x)
    err = max_abs_error(ref, out)
    print("stack error", err)
    if not (err == err) or err > 1.0:
        raise SystemExit(f"stack numerically unusable: {err}")
    stats = compare_paired(
        matmul_stack, triton_stack, (x,), pairs=21, warmup=3, repeats=10, flush_l2=True, graph=True
    )
    low, high = stats["speedup_ci95"]
    relation = "above" if low > 1 else "below" if high < 1 else "contains"
    h11 = low > 1.2
    print(f"stack {stats['speedup_median']}x ci={stats['speedup_ci95']} {relation} H11={h11}")
    if stats["pairs"] == 0:
        raise SystemExit("timer returned zero latencies; this run is discarded")
    path = Path("reports/rtx3070/mlp4_short_m.json")
    path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight": health,
        "publishable": idle,
        "stack_error": err,
        "h11_pass": h11,
        "relation_to_1": relation,
        "vs_matmul_stack": stats,
    }, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
