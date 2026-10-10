"""Does the skinny-K SiLU win survive CUDA graphs, and does the decode win die?

ARCHITECTURE.md §1.6: at M=N=4096, K=64 the predicted gain is traffic, so a
CUDA graph (which removes launch overhead and nothing else) should keep it.
At M<=64, N=K=4096 the predicted gain is two launches, so the same graph
should erase it. Both cells are paired against addmm-then-SiLU.

    python scripts/graph_survival.py --output reports/rtx3070/graph_survival.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ako.benchmark.bench_op import compare_paired
from ako.benchmark.env import preflight
from ako.search.builders import triton_mm_fused
from ako.search.spec import Problem, caps
from ako.validation.numerical import error_budget, max_abs_error
from scripts.winnable_region import _inputs, _tile_dict, _vendor, tile_shortlist, wave_eff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/rtx3070/graph_survival.json"))
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=20)
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

    cells = [
        ("skinny_k", Problem(m=4096, n=4096, k=64, dtype="fp16", act="silu", bias=True)),
        ("decode", Problem(m=64, n=4096, k=4096, dtype="fp16", act="silu", bias=True)),
    ]
    rows = []
    for name, prob in cells:
        a, b, bias = _inputs(prob, device)
        vendor_name, vendor = _vendor(prob)
        ref = vendor(a, b, bias)
        budget = error_budget(ref)
        flush = prob.k > 256
        tile = None
        fused = None
        best = -1.0
        for candidate in tile_shortlist(prob, device_caps):
            trial = triton_mm_fused(prob, candidate)
            err = max_abs_error(ref, trial(a, b, bias))
            if err > budget:
                print(f"  {name} {_tile_dict(candidate)} REJECTED {err}")
                continue
            stats = compare_paired(
                vendor, trial, (a, b, bias), pairs=3, warmup=2, repeats=8, flush_l2=flush
            )
            print(f"  {name} probe {_tile_dict(candidate)} -> {stats['speedup_median']}x")
            if stats["speedup_median"] > best:
                best = stats["speedup_median"]
                tile = candidate
                fused = trial
        if tile is None or fused is None:
            rows.append({"cell": name, "problem": prob.as_dict(), "skipped": "no tile in budget"})
            continue

        def _pair(graph: bool):
            stats = compare_paired(
                vendor,
                fused,
                (a, b, bias),
                pairs=args.pairs,
                warmup=3,
                repeats=args.repeats,
                flush_l2=flush,
                graph=graph,
            )
            label = "graph" if graph else "uncaptured"
            print(
                f"  {name} {label}: fused {stats['speedup_median']}x "
                f"ci={stats['speedup_ci95']} significant={stats['significant']} "
                f"captured={stats['graph_captured']}"
            )
            return stats

        print(f"{name} tile={_tile_dict(tile)}")
        row = {
            "cell": name,
            "problem": prob.as_dict(),
            "tile": _tile_dict(tile),
            "wave_eff": round(wave_eff(tile, prob, device_caps), 4),
            "vendor": vendor_name,
            "flush_l2": flush,
            "uncaptured": _pair(False),
            "cuda_graph": _pair(True),
        }
        rows.append(row)
        del a, b, bias, ref
        torch.cuda.empty_cache()

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device_caps.name,
        "sm_count": device_caps.sm_count,
        "preflight": health,
        "publishable": idle,
        "note": (
            "speedup_median is addmm_then_act/fused. >1 means fused is faster. "
            "skinny_k should keep its lead under a CUDA graph (the gain is traffic). "
            "decode should lose that lead under a graph (the gain is launch overhead). "
            "Absolute milliseconds are not quotable when unstable or the card is busy."
        ),
        "cells": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
