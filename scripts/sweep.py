"""Pinned A/B sweep — the M6 go/no-go, not a budgeted search.

    python scripts/sweep.py --ab 128x128,64x128 --shapes 1024,4096 --dtype fp16

Measures exactly two tile configs (same warps, stages, BLOCK_K) at each shape,
paired and interleaved. Completes in minutes. Do not use a budgeted search here:
an empty DB can miss both tiles and produce a null explained by sampling rather
than by a broken timing path.

This laptop 3070 reports 40 SMs, not the 46 of a desktop 3070. The wave model
must read ``DeviceCaps.sm_count``, never a hardcoded 46.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from ako.search.builders import triton_mm_fused
from ako.search.constraints import Params, wave_eff, waves
from ako.search.correctness import make_inputs
from ako.search.db import TrialDB
from ako.search.measure import gate_then_measure, paired
from ako.search.spec import Candidate, Problem, caps

console = Console()

DEFAULT_SHARED: Params = {
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "SPLIT_K": 1,
    "num_warps": 4,
    "num_stages": 3,
    "ACC_FP16": False,
}


def parse_ab(raw: str) -> list[tuple[int, int]]:
    """``128x128,64x128`` -> [(128, 128), (64, 128)]."""
    out: list[tuple[int, int]] = []
    for part in raw.split(","):
        m_s, n_s = part.lower().split("x")
        out.append((int(m_s), int(n_s)))
    if len(out) != 2:
        raise ValueError(f"--ab needs exactly two tiles, got {raw!r}")
    return out


def params_for(block_m: int, block_n: int) -> Params:
    return {**DEFAULT_SHARED, "BLOCK_M": block_m, "BLOCK_N": block_n}


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned A/B tile comparison (M6 go/no-go)")
    parser.add_argument("--ab", required=True, help="Two tiles, e.g. 128x128,64x128")
    parser.add_argument("--shapes", default="1024,4096", help="Square M=N=K values")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--db", type=Path, default=Path("reports/rtx3070/trials.sqlite"))
    parser.add_argument("--output", type=Path, default=Path("reports/rtx3070/m6_ab.json"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("M6 go/no-go needs a CUDA device")

    tiles = parse_ab(args.ab)
    shapes = [int(s) for s in args.shapes.split(",")]
    device_caps = caps("cuda")
    device = torch.device("cuda")
    db = TrialDB(args.db)

    rows: list[dict] = []
    for dim in shapes:
        prob = Problem(m=dim, n=dim, k=dim, dtype=args.dtype, act="relu", bias=True)
        sweep_id = db.begin_sweep(prob, device_caps, budget_s=600)
        a, b, bias = make_inputs(prob, scale=1.0, seed=0, device=device)
        tensor_args = (a, b, bias)
        table = Table(title=f"{dim}^3 {args.dtype}  sm_count={device_caps.sm_count}")
        table.add_column("tile")
        table.add_column("waves")
        table.add_column("wave_eff")
        table.add_column("p50 ms")
        table.add_column("vs pair")

        fns: dict[tuple[int, int], object] = {}
        for bm, bn in tiles:
            p = params_for(bm, bn)
            cand = Candidate(
                composition="mm+bias+act",
                blocks=("triton_mm_fused",),
                params=(tuple(sorted(p.items())),),
            )
            fn = triton_mm_fused(prob, p)
            gate, timed = gate_then_measure(
                fn, prob, tensor_args, device=device, kernel_name=f"{bm}x{bn}"
            )
            if timed is None:
                console.print(f"[red]{bm}x{bn} rejected: {gate.verdict} {gate.reason}[/red]")
                db.record(
                    sweep_id, prob, cand, device_caps, verdict=gate.verdict, **gate.to_record()
                )
                continue
            fns[(bm, bn)] = fn
            n_waves = waves(p, prob, device_caps)
            eff = wave_eff(p, prob, device_caps)
            db.record(
                sweep_id,
                prob,
                cand,
                device_caps,
                verdict="ok",
                **gate.to_record(),
                wall_p50_ms=timed.latency_p50_ms,
                wall_p95_ms=timed.latency_p95_ms,
                gpu_p50_ms=timed.gpu_p50_ms,
                cv=timed.cv,
                graph_captured=timed.graph_captured,
            )
            table.add_row(
                f"{bm}x{bn}",
                str(n_waves),
                f"{eff:.3f}",
                f"{timed.latency_p50_ms:.4f}",
                "-",
            )
            rows.append(
                {
                    "shape": dim,
                    "tile": f"{bm}x{bn}",
                    "waves": n_waves,
                    "wave_eff": eff,
                    "sm_count": device_caps.sm_count,
                    "p50_ms": timed.latency_p50_ms,
                    "verdict": gate.verdict,
                }
            )

        if len(fns) == 2:
            (t0, fn0), (t1, fn1) = list(fns.items())
            cmp = paired(fn0, fn1, tensor_args, pairs=7, warmup=3, repeats=15)
            speedup = cmp["speedup_median"]
            table.add_row("paired", "-", "-", "-", f"{speedup:.3f}x ({t1} vs {t0})")
            rows[-1]["speedup_vs_pair"] = speedup
            rows[-1]["significant"] = cmp["significant"]
        console.print(table)
        db.end_sweep(sweep_id, drift_pct=0.0, status="ok")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"sm_count": device_caps.sm_count, "rows": rows}, indent=2))
    console.print(f"[green]Wrote {args.output}[/green]")
    db.close()


if __name__ == "__main__":
    main()
