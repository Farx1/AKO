"""Same-kernel control, then the M=16 lead at a stricter bar.

Registered in RESEARCH_LOG.md before the run.

H7: graphed matmul versus itself contains 1, median in [0.9, 1.1].
H8: graphed Triton versus matmul lies entirely above 1.2.

    python scripts/harness_self_test.py
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

M, N, K = 16, 4096, 4096
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


def _matmul(a, b, bias):
    del bias
    return a @ b


def _relation(stats: dict) -> str:
    low, high = stats["speedup_ci95"]
    if low > 1:
        return "above"
    if high < 1:
        return "below"
    return "contains"


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

    g = torch.Generator(device="cpu")
    g.manual_seed(3)
    a = torch.randn(M, K, dtype=torch.float16, generator=g).to(device)
    b = torch.randn(K, N, dtype=torch.float16, generator=g).to(device)
    dummy = torch.empty(1, device=device, dtype=torch.float16)

    prob = Problem(m=M, n=N, k=K, dtype="fp16", act="none", bias=False)
    fused = triton_mm_fused(prob, TILE)
    ref = a @ b
    err = max_abs_error(ref, fused(a, b, dummy))
    budget = error_budget(ref)
    if err > budget:
        raise SystemExit(f"fused kernel outside budget: {err} > {budget}")

    common = dict(pairs=31, warmup=3, repeats=12, flush_l2=True, graph=True)
    self_stats = compare_paired(_matmul, _matmul, (a, b, dummy), **common)
    lead_stats = compare_paired(_matmul, fused, (a, b, dummy), **common)
    print(
        f"self {self_stats['speedup_median']}x ci={self_stats['speedup_ci95']} {_relation(self_stats)}"
    )
    print(
        f"lead {lead_stats['speedup_median']}x ci={lead_stats['speedup_ci95']} {_relation(lead_stats)}"
    )
    if self_stats["pairs"] == 0 or lead_stats["pairs"] == 0:
        raise SystemExit("timer returned zero latencies; this run is discarded")

    median = self_stats["speedup_median"]
    h7 = _relation(self_stats) == "contains" and 0.9 <= median <= 1.1
    low = lead_stats["speedup_ci95"][0]
    h8 = low > 1.2
    path = Path("reports/rtx3070/harness_self_test.json")
    path.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preflight": health,
        "publishable": idle,
        "error": err,
        "error_budget": budget,
        "h7_pass": h7,
        "h8_pass": h8,
        "self": self_stats,
        "lead": lead_stats,
    }, indent=2))
    print(f"H7={h7} H8={h8}")
    print("wrote", path)


if __name__ == "__main__":
    main()
