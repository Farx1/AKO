"""Paired measurement of the one fusion PyTorch does not already ship.

cuBLASLt, through ``torch._addmm_activation``, already fuses matmul + bias +
ReLU/GELU. SiLU is the gate of every SwiGLU block and is not in that epilogue
enum, so the public PyTorch path is ``addmm`` then a second kernel.

This script times a fused Triton matmul+bias+SiLU against that path, against
eager, and against ``torch.compile``, on the shape family ARCHITECTURE.md §1.6
calls the only large fusion win (large M, N, small K) and on a square control
where fusion cannot move the needle. Comparisons are interleaved pairs, because
the overnight noise floor on this 3070 forbids publishing an absolute ranking.

    python scripts/winnable_region.py --output reports/rtx3070/winnable_region.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from ako.benchmark.bench_op import compare_paired
from ako.benchmark.env import preflight
from ako.search.builders import torch_addmm, torch_addmm_act, torch_eager, triton_mm_fused
from ako.search.constraints import feasible, wave_eff
from ako.search.space import GEMM_AXES, gemm_grid
from ako.search.spec import Problem, caps
from ako.validation.numerical import error_budget, max_abs_error

# A slice of the full grid: split-K cannot carry a fused epilogue, and ACC_FP16
# is a tolerance question, not the fusion question this script answers.
_AXES = {
    **GEMM_AXES,
    "GROUP_M": (8,),
    "SPLIT_K": (1,),
    "ACC_FP16": (False,),
    "BLOCK_M": (64, 128),
    "BLOCK_N": (64, 128, 256),
    "BLOCK_K": (32, 64),
    "num_warps": (4, 8),
    "num_stages": (3, 4),
}


def _legal_tiles(prob: Problem, device_caps) -> list[tuple[float, dict]]:
    legal = []
    for params in gemm_grid(_AXES):
        if feasible(params, prob, device_caps) is not None:
            continue
        legal.append((wave_eff(params, prob, device_caps), params))
    legal.sort(key=lambda item: item[0], reverse=True)
    return legal


def best_fused_tile(prob: Problem, device_caps) -> dict:
    """Highest last-wave occupancy among fused, feasible tiles."""
    legal = _legal_tiles(prob, device_caps)
    if not legal:
        raise RuntimeError(f"no feasible fused tile for {prob}")
    return legal[0][1]


def tile_shortlist(prob: Problem, device_caps, k: int = 4) -> list[dict]:
    """A few distinct tiles, best occupancy first. One per M/N/K block size."""
    seen: set[tuple[int, int, int]] = set()
    picked = []
    for _eff, params in _legal_tiles(prob, device_caps):
        key = (int(params["BLOCK_M"]), int(params["BLOCK_N"]), int(params["BLOCK_K"]))
        if key in seen:
            continue
        seen.add(key)
        picked.append(params)
        if len(picked) == k:
            break
    if not picked:
        raise RuntimeError(f"no feasible fused tile for {prob}")
    return picked


def _tile_dict(tile: dict) -> dict:
    return {k: tile[k] for k in ("BLOCK_M", "BLOCK_N", "BLOCK_K", "num_warps", "num_stages")}


def _inputs(prob: Problem, device: torch.device, seed: int = 0):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[prob.dtype]
    a = torch.randn(prob.m, prob.k, dtype=dtype, generator=g).to(device)
    b = torch.randn(prob.k, prob.n, dtype=dtype, generator=g).to(device)
    bias = torch.randn(prob.n, dtype=dtype, generator=g).to(device) if prob.bias else None
    return a, b, bias


def _vendor(prob: Problem):
    """The strongest kernel PyTorch exposes without compile, for this activation."""
    if prob.act in {"relu", "gelu_tanh"} and prob.bias:
        return "torch._addmm_activation", torch_addmm_act(prob)
    # SiLU (and anything else) is addmm plus a separate epilogue.

    def fn(a, b, bias=None):
        out = torch_addmm(prob)(a, b, bias)
        from ako.search.correctness import apply_act

        return apply_act(out, prob.act).to(dtype=a.dtype)

    return "addmm_then_act", fn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/rtx3070/winnable_region.json"))
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="also pair against torch.compile(mode='max-autotune'); slow first call",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("this measurement needs CUDA")

    try:
        health = preflight()
        idle = True
    except RuntimeError as exc:
        health = {"checked": True, "error": str(exc)}
        idle = False
        print(exc)
        print("not publishable: the card is not idle. numbers below are a smoke test only")
    else:
        idle = bool(health.get("checked"))
        if not idle:
            print("not publishable: nvidia-smi did not answer the preflight")

    device = torch.device("cuda")
    device_caps = caps(device)
    # Preheat so the first pair is not the thermal ramp the noise floor measured.
    torch.matmul(torch.randn(2048, 2048, device=device), torch.randn(2048, 2048, device=device))
    torch.cuda.synchronize()

    cells = [
        Problem(m=4096, n=4096, k=64, dtype="fp16", act="silu", bias=True),
        Problem(m=4096, n=4096, k=64, dtype="fp16", act="gelu_tanh", bias=True),
        Problem(m=1024, n=1024, k=1024, dtype="fp16", act="silu", bias=True),
    ]
    rows = []
    for prob in cells:
        a, b, bias = _inputs(prob, device)
        eager = torch_eager(prob)
        vendor_name, vendor = _vendor(prob)
        ref = eager(a, b, bias)
        vendor_out = vendor(a, b, bias)
        budget = error_budget(ref)
        # Occupancy is a prior, not a measurement. Compile a few block sizes and
        # keep the one a short paired probe says is fastest against the vendor.
        tile = None
        fused = None
        fused_out = None
        # The K=64 output is 32 MiB and this card's L2 is 4 MiB, so a flushed
        # call and an unflushed call both miss. The 32 MiB flush is larger than
        # the kernel and was the noise in the first pass. Square cells still flush.
        flush = prob.k > 256
        best_probe = -1.0
        probe_rows = []
        for candidate in tile_shortlist(prob, device_caps):
            trial = triton_mm_fused(prob, candidate)
            trial_out = trial(a, b, bias)
            err = max_abs_error(ref, trial_out)
            if err > budget:
                probe_rows.append({"tile": _tile_dict(candidate), "rejected_error": err})
                print(f"  probe {_tile_dict(candidate)} REJECTED {err}")
                continue
            stats = compare_paired(
                vendor, trial, (a, b, bias), pairs=3, warmup=3, repeats=10, flush_l2=flush
            )
            probe_rows.append({"tile": _tile_dict(candidate), "vs_vendor_probe": stats["speedup_median"]})
            print(f"  probe {_tile_dict(candidate)} -> {stats['speedup_median']}x vs {vendor_name}")
            if stats["speedup_median"] > best_probe:
                best_probe = stats["speedup_median"]
                tile = candidate
                fused = trial
                fused_out = trial_out
        if tile is None or fused is None or fused_out is None:
            rows.append(
                {
                    "problem": prob.as_dict(),
                    "probes": probe_rows,
                    "skipped_timing": "every fused tile outside the dtype budget",
                    "error_budget": budget,
                    "vendor": vendor_name,
                }
            )
            print(prob.act, prob.k, "REJECTED")
            continue
        row: dict = {
            "problem": prob.as_dict(),
            "tile": _tile_dict(tile),
            "probes": probe_rows,
            "wave_eff": round(wave_eff(tile, prob, device_caps), 4),
            "sm_count": device_caps.sm_count,
            "fused_vs_eager_error": max_abs_error(ref, fused_out),
            "vendor_vs_eager_error": max_abs_error(ref, vendor_out),
            "error_budget": budget,
            "fused_within_budget": True,
            "vendor": vendor_name,
            "flush_l2": flush,
        }

        def _pair(name, other):
            stats = compare_paired(
                other,
                fused,
                (a, b, bias),
                pairs=args.pairs,
                warmup=args.warmup,
                repeats=args.repeats,
                flush_l2=flush,
            )
            # speedup_median is other/fused, so > 1 means fused is faster.
            print(
                f"  {name}: fused {stats['speedup_median']}x "
                f"ci={stats['speedup_ci95']} significant={stats['significant']}"
            )
            return stats

        print(f"{prob.act} M={prob.m} K={prob.k} tile={row['tile']}")
        row["vs_eager"] = _pair("eager", eager)
        row["vs_vendor"] = _pair(vendor_name, vendor)
        if args.compile:
            try:
                compiled = torch.compile(eager, mode="max-autotune")
                compiled(a, b, bias)
                torch.cuda.synchronize()
                row["vs_compile"] = _pair("compile", compiled)
            except Exception as exc:
                row["vs_compile"] = {"error": f"{type(exc).__name__}: {exc}"}
                print("  compile skipped:", exc)
        rows.append(row)
        del a, b, bias, ref, fused_out, vendor_out
        torch.cuda.empty_cache()

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device_caps.name,
        "sm_count": device_caps.sm_count,
        "preflight": health,
        "publishable": idle,
        "note": (
            "speedup_median is baseline/fused from interleaved pairs. "
            ">1 means the fused Triton kernel is faster. "
            "gelu_tanh vs torch._addmm_activation is the vendor control; "
            "silu vs addmm_then_act is the path PyTorch does not fuse. "
            "flush_l2 is off when K<=256: the output does not fit in L2 either way."
        ),
        "cells": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
