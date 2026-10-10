"""RoPE benchmark runner (shared by CLI scripts).

Compares three paths on the same shapes:

- ``rope_ref``: PyTorch reference (materializes intermediates)
- ``rope_triton``: fused single-pass kernel
- ``rope_qk_*``: the query+key pair, which is what a decoder layer actually runs
"""

from __future__ import annotations

from pathlib import Path

import torch
from rich.console import Console

from ako.benchmark.bench_op import bench_op
from ako.benchmark.report import BenchmarkReport, save_report
from ako.dispatcher.hardware import get_device_info
from ako.ops.rope_ref import build_rope_tables, rope_ref
from ako.ops.rope_torch import is_compile_enabled, rope_compiled
from ako.ops.rope_triton import is_triton_available, rope_triton
from ako.validation.numerical import max_abs_error

console = Console()

DEFAULT_BATCHES = [1, 8]
DEFAULT_SEQ_LENS = [512, 2048]
DEFAULT_NUM_HEADS = [32]
DEFAULT_HEAD_DIMS = [64, 128]


def run_benchmark(
    *,
    batches: list[int],
    seq_lens: list[int],
    num_heads: list[int],
    head_dims: list[int],
    dtypes: list[torch.dtype],
    device: str,
    output_dir: Path,
    warmup: int,
    repeats: int,
    style: str = "neox",
) -> BenchmarkReport:
    """Benchmark RoPE across shapes and write ``rope_benchmark.json``."""
    dev = torch.device(device)
    device_info = get_device_info(dev)
    report = BenchmarkReport(operation="rope", device=device_info)

    # PyTorch has no fused rotary embedding, so the honest baseline is
    # torch.compile over the reference — it fuses the same graph the Triton
    # kernel fuses by hand. Comparing only against `rope_ref` mostly measures
    # the intermediates the reference materializes.
    kernels: list[tuple[str, object]] = [("rope_ref", rope_ref)]
    if is_compile_enabled():
        kernels.append(("rope_compiled", rope_compiled))
    else:
        console.print(
            "[dim]torch.compile baseline disabled — set AKO_ENABLE_COMPILE=1 to include it.[/dim]"
        )
    if is_triton_available() and dev.type == "cuda":
        kernels.append(("rope_triton", rope_triton))
    elif dev.type != "cuda":
        console.print(
            "[yellow]Triton/CUDA kernels skipped — running PyTorch baselines on CPU.[/yellow]"
        )

    candidates = {"rope_triton"}

    for batch in batches:
        for seq_len in seq_lens:
            for heads in num_heads:
                for head_dim in head_dims:
                    for dtype in dtypes:
                        shape = (batch, seq_len, heads, head_dim)
                        x = torch.randn(*shape, dtype=dtype, device=dev)
                        cos, sin = build_rope_tables(
                            seq_len, head_dim, device=dev, dtype=torch.float32
                        )

                        ref_out = rope_ref(x, cos, sin, style)
                        measured: dict[str, float] = {}
                        for name, fn in kernels:
                            try:
                                result = bench_op(
                                    fn,
                                    (x, cos, sin),
                                    warmup=warmup,
                                    repeats=repeats,
                                    kernel_name=name,
                                    call_kwargs={"style": style},
                                )
                            except Exception as exc:
                                console.print(f"[red]Skip {name} {shape}: {exc}[/red]")
                                continue
                            report.add_result(
                                result,
                                style=style,
                                num_heads=heads,
                                dtype=str(dtype).replace("torch.", ""),
                            )
                            measured[name] = result.latency_p50_ms

                        baselines = {n: ms for n, ms in measured.items() if n not in candidates}
                        if not baselines:
                            continue
                        best_baseline = min(baselines, key=baselines.get)

                        for name, fn in kernels:
                            if name not in candidates or name not in measured:
                                continue
                            try:
                                error = max_abs_error(ref_out, fn(x, cos, sin, style))
                            except Exception as exc:
                                console.print(f"[red]Skip comparison {name} {shape}: {exc}[/red]")
                                continue
                            report.add_comparison(
                                shape=list(shape),
                                baseline_ms=measured["rope_ref"],
                                candidate_ms=measured[name],
                                max_error=error,
                                selected_kernel=name,
                                dtype=str(dtype).replace("torch.", ""),
                                baseline_kernel="rope_ref",
                                best_baseline_ms=baselines[best_baseline],
                                best_baseline_kernel=best_baseline,
                            )

                        del x, cos, sin, ref_out
                        if dev.type == "cuda":
                            torch.cuda.empty_cache()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_report(report, output_dir / "rope_benchmark.json")
    return report
