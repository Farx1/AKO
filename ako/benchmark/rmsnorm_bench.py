"""RMSNorm benchmark runner (shared by CLI scripts)."""

from __future__ import annotations

from pathlib import Path

import torch
from rich.console import Console

from ako.benchmark.bench_op import bench_op, parse_dtypes
from ako.benchmark.report import BenchmarkReport, save_report
from ako.dispatcher.hardware import get_device_info
from ako.ops.rmsnorm_ref import rmsnorm_ref
from ako.ops.rmsnorm_torch import (
    is_compile_enabled,
    is_native_available,
    rmsnorm_compiled,
    rmsnorm_torch_native,
)
from ako.ops.rmsnorm_triton import is_triton_available, rmsnorm_triton
from ako.validation.numerical import max_abs_error

console = Console()

DEFAULT_BATCHES = [1, 4, 8, 16]
DEFAULT_SEQ_LENS = [512, 2048, 4096]
DEFAULT_HIDDEN_DIMS = [1024, 2048, 4096]

__all__ = [
    "DEFAULT_BATCHES",
    "DEFAULT_HIDDEN_DIMS",
    "DEFAULT_SEQ_LENS",
    "parse_dtypes",
    "run_benchmark",
]


def run_benchmark(
    *,
    batches: list[int],
    seq_lens: list[int],
    hidden_dims: list[int],
    dtypes: list[torch.dtype],
    device: str,
    output_dir: Path,
    warmup: int,
    repeats: int,
) -> BenchmarkReport:
    dev = torch.device(device)
    device_info = get_device_info(dev)
    report = BenchmarkReport(operation="rmsnorm", device=device_info)

    # Baselines are ordered weakest first. `rmsnorm_ref` is the numerical
    # reference and the historical comparison point; the others are what a user
    # would realistically already be running, and are the honest bar to clear.
    kernels: list[tuple[str, object]] = [("rmsnorm_ref", rmsnorm_ref)]
    if is_native_available():
        kernels.append(("rmsnorm_torch_native", rmsnorm_torch_native))
    if is_compile_enabled():
        kernels.append(("rmsnorm_compiled", rmsnorm_compiled))
    else:
        console.print(
            "[dim]torch.compile baseline disabled — set AKO_ENABLE_COMPILE=1 to include it.[/dim]"
        )
    if is_triton_available() and dev.type == "cuda":
        kernels.append(("rmsnorm_triton", rmsnorm_triton))
    elif dev.type != "cuda":
        console.print(
            "[yellow]Triton/CUDA kernels skipped — running PyTorch baselines on CPU.[/yellow]"
        )

    candidates = {"rmsnorm_triton"}

    for batch in batches:
        for seq_len in seq_lens:
            for hidden in hidden_dims:
                for dtype in dtypes:
                    shape = (batch, seq_len, hidden)
                    x = torch.randn(*shape, dtype=dtype, device=dev)
                    weight = torch.ones(hidden, dtype=dtype, device=dev)

                    ref_out = rmsnorm_ref(x, weight)
                    measured: dict[str, float] = {}
                    for name, fn in kernels:
                        try:
                            result = bench_op(
                                fn,
                                (x, weight),
                                warmup=warmup,
                                repeats=repeats,
                                kernel_name=name,
                            )
                        except Exception as exc:
                            console.print(f"[red]Skip {name} {shape}: {exc}[/red]")
                            continue
                        report.add_result(result, dtype=str(dtype).replace("torch.", ""))
                        measured[name] = result.latency_p50_ms

                    baselines = {n: ms for n, ms in measured.items() if n not in candidates}
                    if not baselines:
                        continue
                    best_baseline = min(baselines, key=baselines.get)

                    for name, fn in kernels:
                        if name not in candidates or name not in measured:
                            continue
                        try:
                            error = max_abs_error(ref_out, fn(x, weight))
                        except Exception as exc:
                            console.print(f"[red]Skip comparison {name} {shape}: {exc}[/red]")
                            continue
                        report.add_comparison(
                            shape=list(shape),
                            baseline_ms=measured["rmsnorm_ref"],
                            candidate_ms=measured[name],
                            max_error=error,
                            selected_kernel=name,
                            dtype=str(dtype).replace("torch.", ""),
                            best_baseline_ms=baselines[best_baseline],
                            best_baseline_kernel=best_baseline,
                        )

                    # Release the shape's tensors before the next one: the largest
                    # shapes fill a laptop GPU, and allocator pressure late in a
                    # sweep shows up as latencies that are not comparable.
                    del x, weight, ref_out
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_report(report, output_dir / "baseline_rmsnorm.json")
    return report
