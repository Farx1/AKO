"""SwiGLU baseline and Triton benchmark script.

Same three-way comparison as the RoPE sweep: the unfused reference, the
``torch.compile`` baseline (the honest competitor, since PyTorch ships no fused
``silu(gate) * up``), and the Triton kernel. Shapes are ``(batch, seq_len,
intermediate_dim)`` — the width the MLP actually activates on, not hidden_dim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from ako.benchmark.bench_op import bench_op, parse_dtypes
from ako.benchmark.report import BenchmarkReport, save_report
from ako.dispatcher.hardware import get_device_info
from ako.ops.swiglu_ref import swiglu_ref
from ako.ops.swiglu_torch import is_compile_enabled, swiglu_compiled
from ako.ops.swiglu_triton import is_triton_available, swiglu_triton
from ako.validation.numerical import max_abs_error

console = Console()

DEFAULT_BATCHES = [1, 8]
DEFAULT_SEQ_LENS = [1, 512, 2048]
DEFAULT_INTERMEDIATE_DIMS = [5632, 11008]

CANDIDATES = {"swiglu_triton"}


def run_benchmark(
    *,
    batches: list[int],
    seq_lens: list[int],
    intermediate_dims: list[int],
    dtypes: list[torch.dtype],
    device: str,
    output_dir: Path,
    warmup: int,
    repeats: int,
) -> BenchmarkReport:
    """Benchmark SwiGLU across shapes and write ``swiglu_benchmark.json``."""
    dev = torch.device(device)
    report = BenchmarkReport(operation="swiglu", device=get_device_info(dev))

    kernels: list[tuple[str, object]] = [("swiglu_ref", swiglu_ref)]
    if is_compile_enabled():
        kernels.append(("swiglu_compiled", swiglu_compiled))
    else:
        console.print(
            "[dim]torch.compile baseline disabled — set AKO_ENABLE_COMPILE=1 to include it.[/dim]"
        )
    if is_triton_available() and dev.type == "cuda":
        kernels.append(("swiglu_triton", swiglu_triton))
    elif dev.type != "cuda":
        console.print(
            "[yellow]Triton/CUDA kernels skipped — running PyTorch baselines on CPU.[/yellow]"
        )

    for batch in batches:
        for seq_len in seq_lens:
            for hidden in intermediate_dims:
                for dtype in dtypes:
                    shape = (batch, seq_len, hidden)
                    gate = torch.randn(*shape, dtype=dtype, device=dev)
                    up = torch.randn(*shape, dtype=dtype, device=dev)

                    ref_out = swiglu_ref(gate, up)
                    measured: dict[str, float] = {}
                    for name, fn in kernels:
                        try:
                            result = bench_op(
                                fn, (gate, up), warmup=warmup, repeats=repeats, kernel_name=name
                            )
                        except Exception as exc:
                            console.print(f"[red]Skip {name} {shape}: {exc}[/red]")
                            continue
                        report.add_result(result, dtype=str(dtype).replace("torch.", ""))
                        measured[name] = result.latency_p50_ms

                    baselines = {n: ms for n, ms in measured.items() if n not in CANDIDATES}
                    if not baselines:
                        continue
                    best_baseline = min(baselines, key=baselines.get)

                    for name, fn in kernels:
                        if name not in CANDIDATES or name not in measured:
                            continue
                        try:
                            error = max_abs_error(ref_out, fn(gate, up))
                        except Exception as exc:
                            console.print(f"[red]Skip comparison {name} {shape}: {exc}[/red]")
                            continue
                        report.add_comparison(
                            shape=list(shape),
                            baseline_ms=measured["swiglu_ref"],
                            candidate_ms=measured[name],
                            max_error=error,
                            selected_kernel=name,
                            dtype=str(dtype).replace("torch.", ""),
                            baseline_kernel="swiglu_ref",
                            best_baseline_ms=baselines[best_baseline],
                            best_baseline_kernel=best_baseline,
                        )

                    del gate, up, ref_out
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_report(report, output_dir / "swiglu_benchmark.json")
    return report


def _print_summary(report: BenchmarkReport) -> None:
    table = Table(title="SwiGLU Benchmark Summary")
    for column in ("Shape (B, L, D)", "dtype", "ref (ms)", "best baseline", "Triton (ms)",
                   "vs ref", "vs best", "Max error"):
        table.add_column(column)

    if not report.comparisons:
        table.add_row("(baselines only)", *["-"] * 7)

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
    parser = argparse.ArgumentParser(description="Benchmark SwiGLU kernels")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=DEFAULT_SEQ_LENS)
    parser.add_argument("--intermediate-dims", type=int, nargs="+",
                        default=DEFAULT_INTERMEDIATE_DIMS)
    parser.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"])
    args = parser.parse_args()

    report = run_benchmark(
        batches=args.batches,
        seq_lens=args.seq_lens,
        intermediate_dims=args.intermediate_dims,
        dtypes=parse_dtypes(args.dtypes),
        device=args.device,
        output_dir=args.output_dir,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    _print_summary(report)
    console.print(f"\n[green]Report saved to {args.output_dir / 'swiglu_benchmark.json'}[/green]")


if __name__ == "__main__":
    main()
