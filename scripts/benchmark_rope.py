"""RoPE baseline and Triton benchmark script."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from ako.benchmark.bench_op import parse_dtypes
from ako.benchmark.rope_bench import (
    DEFAULT_BATCHES,
    DEFAULT_HEAD_DIMS,
    DEFAULT_NUM_HEADS,
    DEFAULT_SEQ_LENS,
    run_benchmark,
)

console = Console()


def _print_summary(report) -> None:
    table = Table(title="RoPE Benchmark Summary")
    table.add_column("Shape (B, L, H, D)")
    table.add_column("dtype")
    table.add_column("ref (ms)")
    table.add_column("best baseline")
    table.add_column("Triton (ms)")
    table.add_column("vs ref")
    table.add_column("vs best")
    table.add_column("Max error")

    if not report.comparisons:
        table.add_row("(baselines only)", "-", "-", "-", "-", "-", "-", "-")

    for row in report.comparisons:
        best_ms = row.get("best_baseline_ms")
        table.add_row(
            str(tuple(row["shape"])),
            row.get("dtype", "-"),
            str(row["baseline_ms"]),
            f"{row.get('best_baseline_kernel', '-')} ({best_ms})" if best_ms else "-",
            str(row["candidate_ms"]),
            f"{row['speedup']}x",
            f"{row['speedup_vs_best_baseline']}x" if best_ms else "-",
            f"{row['max_error']:.2e}",
        )
    console.print(table)
    console.print(
        "[dim]'vs best' is the quotable number: the gain over the fastest baseline "
        "available here, not over the unfused reference.[/dim]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RoPE kernels")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--num-heads", type=int, nargs="+", default=DEFAULT_NUM_HEADS)
    parser.add_argument("--head-dims", type=int, nargs="+", default=DEFAULT_HEAD_DIMS)
    parser.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"])
    parser.add_argument("--style", default="neox", choices=["neox", "interleaved"])
    args = parser.parse_args()

    report = run_benchmark(
        batches=args.batches,
        seq_lens=args.seq_lens,
        num_heads=args.num_heads,
        head_dims=args.head_dims,
        dtypes=parse_dtypes(args.dtypes),
        device=args.device,
        output_dir=args.output_dir,
        warmup=args.warmup,
        repeats=args.repeats,
        style=args.style,
    )
    _print_summary(report)
    console.print(f"\n[green]Report saved to {args.output_dir / 'rope_benchmark.json'}[/green]")


if __name__ == "__main__":
    main()
