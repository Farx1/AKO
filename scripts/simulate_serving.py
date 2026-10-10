"""Replay a simulated serving trace under several kernel-selection policies.

Answers the question a shape sweep cannot: on a workload whose shapes move for
reasons outside anyone's control — variable prompt lengths, dynamic batching,
prefill and decode on the same weights — is selecting per shape worth it?

Example:

    python scripts/simulate_serving.py --device cuda --dtype fp16 \\
        --requests 128 --max-batch 8 --output-dir reports/
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from ako.benchmark.bench_op import parse_dtypes
from ako.dispatcher.hardware import get_device_info
from ako.dispatcher.selector import get_selector
from ako.experiments.serving import (
    batch_trace,
    build_policy_stacks,
    crossover_batch,
    generate_trace,
    run_policy,
)
from ako.models.decoder import ModelConfig

console = Console()

POLICIES = ("eager", "compile", "triton", "adaptive")

#: Shape presets. TinyLlama-1.1B is the reference; `small` fits a laptop GPU and
#: keeps a CPU dry-run tractable.
PRESETS = {
    "tinyllama": dict(
        hidden_dim=2048, n_layers=22, n_heads=32, n_kv_heads=4, intermediate_dim=5632
    ),
    "small": dict(hidden_dim=1024, n_layers=4, n_heads=16, n_kv_heads=4, intermediate_dim=2816),
    "tiny": dict(hidden_dim=256, n_layers=2, n_heads=8, n_kv_heads=2, intermediate_dim=512),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--preset", default="small", choices=sorted(PRESETS))
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--median-prompt", type=int, default=256)
    parser.add_argument("--median-decode", type=int, default=64)
    parser.add_argument("--max-prompt", type=int, default=1024)
    parser.add_argument("--spread", type=float, default=1.0, help="Lognormal sigma of the trace")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    args = parser.parse_args()

    device = torch.device(args.device)
    (dtype,) = parse_dtypes([args.dtype])
    if device.type == "cpu" and dtype is not torch.float32:
        console.print("[yellow]CPU run: falling back to fp32.[/yellow]")
        dtype = torch.float32

    config = ModelConfig(max_seq_len=args.max_prompt + 64, **PRESETS[args.preset])
    trace = generate_trace(
        args.requests,
        seed=args.seed,
        median_prompt=args.median_prompt,
        median_decode=args.median_decode,
        max_prompt=args.max_prompt,
        spread=args.spread,
    )
    batches = batch_trace(trace, max_batch=args.max_batch)

    prompts = [r.prompt_len for r in trace]
    console.print(
        f"Trace: {len(trace)} requests in {len(batches)} batches — prompt lengths "
        f"min {min(prompts)}, median {sorted(prompts)[len(prompts) // 2]}, max {max(prompts)}"
    )

    stacks = build_policy_stacks(
        config, policies=args.policies, device=device, dtype=dtype, seed=args.seed
    )

    results = {}
    for policy, model in stacks.items():
        console.print(f"Running [bold]{policy}[/bold]…")
        # A fresh selector per policy: the autotuning cost is part of what the
        # adaptive policy is being charged for.
        get_selector().clear_memo()
        results[policy] = run_policy(
            policy, model, batches, config=config, device=device, dtype=dtype
        )

    table = Table(title="Serving simulation — total time over the trace")
    table.add_column("Policy")
    table.add_column("Total (ms)")
    table.add_column("Prefill (ms)")
    table.add_column("Decode (ms)")
    table.add_column("TTFT p50 / p95 (ms)")
    table.add_column("vs best")

    completed = {p: r for p, r in results.items() if not r.failed and r.total_ms > 0}
    best_total = min((r.total_ms for r in completed.values()), default=0.0)

    # Failed policies go last and show no timings: a policy that did not finish
    # the trace must never look like the fastest one.
    ordered = sorted(results.items(), key=lambda kv: (bool(kv[1].failed), kv[1].total_ms))
    for policy, result in ordered:
        payload = result.to_dict()
        if result.failed:
            table.add_row(f"{policy} (failed)", "—", "—", "—", "—", "—")
            continue
        table.add_row(
            policy,
            f"{result.total_ms:.1f}",
            f"{sum(result.prefill_ms):.1f}",
            f"{sum(result.decode_ms):.1f}",
            f"{payload.get('ttft_p50_ms', 0):.2f} / {payload.get('ttft_p95_ms', 0):.2f}",
            f"{result.total_ms / best_total:.2f}x" if best_total else "—",
        )
    console.print(table)
    for policy, result in results.items():
        if result.failed:
            console.print(f"[red]{policy} did not complete the trace: {result.failed}[/red]")

    crossovers = {}
    if "adaptive" in results:
        for policy, result in results.items():
            if policy == "adaptive" or result.failed:
                continue
            index = crossover_batch(results["adaptive"], result)
            crossovers[policy] = index
            verdict = f"batch {index}" if index is not None else "never over this trace"
            console.print(f"Adaptive overtakes [bold]{policy}[/bold]: {verdict}")

    selector = get_selector()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": get_device_info(device).to_dict(),
        "model": asdict(config),
        "preset": args.preset,
        "trace": {
            "requests": args.requests,
            "batches": len(batches),
            "max_batch": args.max_batch,
            "median_prompt": args.median_prompt,
            "median_decode": args.median_decode,
            "max_prompt": args.max_prompt,
            "spread": args.spread,
            "seed": args.seed,
            "prompt_len_min": min(prompts),
            "prompt_len_max": max(prompts),
        },
        "dtype": str(dtype).replace("torch.", ""),
        "policies": {policy: result.to_dict() for policy, result in results.items()},
        "crossover_batch": crossovers,
        "kernel_cache": selector.cache.stats().to_dict(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "serving_simulation.json"
    path.write_text(json.dumps(report, indent=2))
    console.print(f"\n[green]Simulation written to {path}[/green]")


if __name__ == "__main__":
    main()
