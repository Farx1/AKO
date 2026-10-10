"""End-to-end kernel optimization step.

Runs Algorithm 1 of the paper for every registered operation on the shapes an
inference workload would actually use, then writes the resulting runtime
configuration so a serving process can load it without re-benchmarking.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from ako.benchmark.bench_op import parse_dtypes
from ako.dispatcher.cache import KernelCache
from ako.dispatcher.hardware import get_device_info
from ako.dispatcher.registry import KernelRegistry
from ako.dispatcher.selector import KernelSelector, SelectionResult
from ako.ops._registry import register_all
from ako.ops.rope_ref import build_rope_tables

console = Console()


def _workload(
    args: argparse.Namespace, dtype: torch.dtype, device: torch.device
) -> dict[str, tuple[tuple[torch.Tensor, ...], dict[str, Any]]]:
    """Representative tensors for each operation, as seen in a decoder layer."""
    hidden = torch.randn(args.batch_size, args.seq_len, args.hidden_dim, dtype=dtype, device=device)
    weight = torch.ones(args.hidden_dim, dtype=dtype, device=device)
    heads = torch.randn(
        args.batch_size, args.seq_len, args.num_heads, args.head_dim, dtype=dtype, device=device
    )
    cos, sin = build_rope_tables(args.seq_len, args.head_dim, device=device, dtype=torch.float32)

    return {
        "rmsnorm": ((hidden, weight), {}),
        "rmsnorm_residual": ((hidden, torch.randn_like(hidden), weight), {}),
        "rope": ((heads, cos, sin), {"style": args.rope_style}),
    }


def _print_summary(selections: list[SelectionResult]) -> None:
    table = Table(title="AKO selected kernels")
    table.add_column("Operation")
    table.add_column("Kernel")
    table.add_column("Backend")
    table.add_column("Latency p50 (ms)")
    table.add_column("vs ref")
    table.add_column("vs best baseline")
    table.add_column("Max error")
    table.add_column("Source")

    for sel in selections:
        table.add_row(
            sel.op_name,
            sel.entry.kernel_name,
            sel.entry.backend,
            f"{sel.latency_p50_ms:.4f}" if sel.latency_p50_ms else "-",
            f"{sel.speedup:.2f}x" if sel.speedup else "-",
            f"{sel.speedup_vs_best_baseline:.2f}x ({sel.best_baseline_kernel})"
            if sel.speedup_vs_best_baseline
            else "-",
            f"{sel.max_error:.2e}" if sel.max_error is not None else "-",
            sel.source,
        )
    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize kernels for inference workloads")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--rope-style", default="neox", choices=["neox", "interleaved"])
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument(
        "--allow-cpu-tuning",
        action="store_true",
        help="Benchmark candidates on CPU too (off by default: only the baseline exists there)",
    )
    parser.add_argument("--fresh", action="store_true", help="Ignore and overwrite cached results")
    args = parser.parse_args()

    device = torch.device(args.device)
    (dtype,) = parse_dtypes([args.dtype])
    device_info = get_device_info(device)

    registry = register_all(KernelRegistry())
    cache = KernelCache(path=args.cache_path)
    if args.fresh:
        cache.clear()

    selector = KernelSelector(
        registry=registry,
        cache=cache,
        auto_tune=True,
        allow_cpu_tuning=args.allow_cpu_tuning,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    selections: list[SelectionResult] = []
    for op_name, (tensors, call_kwargs) in _workload(args, dtype, device).items():
        selections.append(
            selector.select_entry(
                op_name,
                *tensors,
                extra_key=call_kwargs.get("style"),
                call_kwargs=call_kwargs,
            )
        )

    _print_summary(selections)

    config = {
        "gpu": device_info.device_name,
        "compute_backend": device_info.compute_backend,
        "device": device_info.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
            "rope_style": args.rope_style,
        },
        "operations": {sel.op_name: sel.to_dict() for sel in selections},
        "cache": {"path": str(cache.path), **cache.stats().to_dict()},
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "selected_config.json"
    config_path.write_text(json.dumps(config, indent=2))

    console.print(f"\n[green]Selected config written to {config_path}[/green]")


if __name__ == "__main__":
    main()
