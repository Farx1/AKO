"""Profile a decoder stack with and without the AKO kernels.

Turns a per-kernel speedup into an end-to-end number (paper §14.6): the same
model is run twice — once on the eager PyTorch path, once through the adaptive
dispatcher — on a prefill pass and on a cached decode step.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from ako.benchmark.bench_op import bench_op, parse_dtypes
from ako.benchmark.profiler import profile_cuda
from ako.dispatcher.hardware import get_device_info
from ako.models.decoder import Backend, ModelConfig, build_stack
from ako.models.loader import load_hf_stacks
from ako.ops.rmsnorm_torch import is_compile_enabled
from ako.validation.numerical import max_abs_error

console = Console()


def _bench_stack(
    model, x: torch.Tensor, *, use_cache: bool, warmup: int, repeats: int, name: str
):
    """Benchmark one forward pass, optionally through a KV cache.

    Returns the full result so the caller can report device time next to
    wall-clock: at decode shapes the kernels are tiny, and the gap between the
    two is what decides whether a backend is compute- or launch-bound.
    """
    if not use_cache:
        with torch.inference_mode():
            return bench_op(model, (x,), warmup=warmup, repeats=repeats, kernel_name=name)

    cache = model.new_kv_cache()
    with torch.inference_mode():
        model(x[:, :-1], kv_cache=cache)
        last = x[:, -1:]

        def step(token: torch.Tensor) -> torch.Tensor:
            # Re-seed the cache length each call so the measured work is constant.
            cache.entries = [
                None if e is None else (e[0][:, : x.shape[1] - 1], e[1][:, : x.shape[1] - 1])
                for e in cache.entries
            ]
            return model(token, kv_cache=cache)

        return bench_op(step, (last,), warmup=warmup, repeats=repeats, kernel_name=name)


def _top_ops(model, x: torch.Tensor, limit: int = 10) -> list[dict[str, Any]]:
    """Top operators by self time, as reported by the PyTorch profiler."""
    with torch.inference_mode(), profile_cuda() as prof:
        model(x)
    key = "self_cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    rows = sorted(prof.key_averages(), key=lambda e: getattr(e, key, 0), reverse=True)
    return [
        {
            "name": row.key,
            "count": row.count,
            "self_time_us": round(getattr(row, key, 0) / 1000, 3),
        }
        for row in rows[:limit]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a decoder stack with AKO kernels")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--intermediate-dim", type=int, default=5632)
    parser.add_argument("--rope-style", default="neox", choices=["neox", "interleaved"])
    parser.add_argument(
        "--model-id",
        default=None,
        help=(
            "Hugging Face LLaMA-family checkpoint to load instead of random weights "
            "(e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0). Requires 'ako[hf]' and network access; "
            "the shape flags above are then read from the checkpoint config."
        ),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--no-profiler", action="store_true", help="Skip the per-operator table")
    args = parser.parse_args()

    device = torch.device(args.device)
    (dtype,) = parse_dtypes([args.dtype])
    if device.type == "cpu" and dtype is not torch.float32:
        console.print("[yellow]CPU run: falling back to fp32.[/yellow]")
        dtype = torch.float32

    if args.model_id:
        console.print(f"Loading [bold]{args.model_id}[/bold] from the Hugging Face hub…")
        models: dict[str, Any] = load_hf_stacks(
            args.model_id, ("torch", "ako"), device=device, dtype=dtype
        )
        config = models["ako"].config
    else:
        config = ModelConfig(
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            intermediate_dim=args.intermediate_dim,
            max_seq_len=args.seq_len,
            rope_style=args.rope_style,
        )
        models = {
            backend: build_stack(config, backend=backend, device=device, dtype=dtype, seed=0)
            for backend in ("torch", "ako")
        }

    # The honest end-to-end competitor is not eager PyTorch, it is the same stack
    # under torch.compile: Inductor fuses the residual add into the norm and the
    # rotation into its consumer, which is what the AKO kernels do by hand.
    if is_compile_enabled():
        try:
            models["torch_compile"] = torch.compile(models["torch"], dynamic=True)
        except Exception as exc:  # a compile failure must not lose the whole run
            console.print(f"[yellow]torch.compile baseline unavailable: {exc}[/yellow]")
    else:
        console.print(
            "[dim]torch.compile baseline disabled — set AKO_ENABLE_COMPILE=1 to include it.[/dim]"
        )

    device_info = get_device_info(device)
    x = torch.randn(args.batch_size, args.seq_len, config.hidden_dim, dtype=dtype, device=device)

    with torch.inference_mode():
        error = max_abs_error(models["torch"](x), models["ako"](x))
    console.print(f"Backend agreement: max_abs_error = {error:.3e}")

    measurements: dict[str, dict[str, Any]] = {}
    for label, use_cache in (("prefill", False), ("decode", True)):
        results = {
            backend: _bench_stack(
                model,
                x,
                use_cache=use_cache,
                warmup=args.warmup,
                repeats=args.repeats,
                name=f"{label}_{backend}",
            )
            for backend, model in models.items()
        }
        latencies = {b: r.latency_p50_ms for b, r in results.items()}
        ako_ms = latencies["ako"]
        baselines = {b: ms for b, ms in latencies.items() if b != "ako"}
        best_baseline = min(baselines, key=baselines.get)

        row: dict[str, Any] = {f"{backend}_ms": round(ms, 4) for backend, ms in latencies.items()}
        for backend, res in results.items():
            if res.gpu_p50_ms is not None:
                row[f"{backend}_gpu_ms"] = round(res.gpu_p50_ms, 4)
                row[f"{backend}_launch_overhead_ms"] = round(res.launch_overhead_ms or 0.0, 4)
        row["speedup"] = round(latencies["torch"] / ako_ms, 3) if ako_ms else 0.0
        row["best_baseline"] = best_baseline
        row["speedup_vs_best_baseline"] = (
            round(baselines[best_baseline] / ako_ms, 3) if ako_ms else 0.0
        )
        measurements[label] = row

    table = Table(title="Decoder stack — PyTorch baselines vs AKO dispatch")
    table.add_column("Pass")
    for backend in models:
        table.add_column(f"{backend} (ms)")
    table.add_column("vs eager")
    table.add_column("vs best baseline")
    for label, row in measurements.items():
        table.add_row(
            label,
            *[str(row[f"{backend}_ms"]) for backend in models],
            f"{row['speedup']:.2f}x",
            f"{row['speedup_vs_best_baseline']:.2f}x ({row['best_baseline']})",
        )
    console.print(table)

    report = {
        "model_id": args.model_id or "random-weights",
        "model": asdict(config),
        "device": device_info.to_dict(),
        "parameters": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "dtype": str(dtype).replace("torch.", ""),
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "max_abs_error": error,
        "measurements": measurements,
        "top_ops": [] if args.no_profiler else _top_ops(models["ako"], x),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "model_profile.json"
    report_path.write_text(json.dumps(report, indent=2))

    console.print(f"\n[green]Profile written to {report_path}[/green]")


if __name__ == "__main__":
    main()
