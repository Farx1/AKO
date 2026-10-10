"""Run the whole measurement campaign with one command.

Chains the existing entrypoints (no measurement logic is re-implemented here)
and writes every artefact into ``--output`` as it goes, so a crash late in the
run never costs the earlier steps:

    python scripts/run_campaign.py --output reports/rtx3070

Steps: environment capture, bench health gate, RMSNorm sweep, RoPE sweep,
SwiGLU sweep, serving-policy comparison, end-to-end model profile, then a
6-number summary.

``--quick`` runs a sub-2-minute smoke pass and works without a GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from rich.console import Console
from rich.table import Table

from ako.benchmark.bench_op import bench_op
from ako.dispatcher.hardware import get_device_info
from ako.ops.rmsnorm_ref import rmsnorm_ref
from ako.ops.rmsnorm_triton import is_triton_available

console = Console()
ROOT = Path(__file__).resolve().parent.parent

STEPS = ("environment", "health", "rmsnorm", "rope", "swiglu", "serving", "profile")

#: Output file each step owns; --resume skips a step whose file already exists.
OUTPUTS = {
    "environment": "environment.json",
    "health": "health.json",
    "rmsnorm": "baseline_rmsnorm.json",
    "rope": "rope_benchmark.json",
    "swiglu": "swiglu_benchmark.json",
    "serving": "serving_simulation.json",
    "profile": "model_profile.json",
}

# ponytail: rough per-step wall-clock, measured order-of-magnitude on an RTX 3070
# laptop. Only used for the "estimated duration" line; tune if it lies to you.
ESTIMATE_S = {
    "full": {"environment": 5, "health": 20, "rmsnorm": 900, "rope": 300, "swiglu": 300,
             "serving": 400, "profile": 120},
    "quick": {"environment": 5, "health": 10, "rmsnorm": 25, "rope": 20, "swiglu": 20,
              "serving": 20, "profile": 20},
}

#: Prefill AND decode in the same grid: seq_len=1 is the shape a serving loop
#: actually spends its time in, and it is where fused kernels usually lose.
GRID = {
    "full": {
        "rmsnorm": ["--batches", "1", "8", "--seq-lens", "1", "512", "2048",
                    "--hidden-dims", "2048", "4096"],
        "rope": ["--batches", "1", "8", "--seq-lens", "1", "512", "2048",
                 "--num-heads", "32", "--head-dims", "128"],
        "swiglu": ["--batches", "1", "8", "--seq-lens", "1", "512", "2048",
                   "--intermediate-dims", "5632"],
        "serving": ["--preset", "small", "--requests", "64", "--max-batch", "8",
                    "--median-prompt", "256", "--median-decode", "64", "--max-prompt", "1024"],
        "profile": ["--batch-size", "1", "--seq-len", "512", "--hidden-dim", "2048",
                    "--n-layers", "4", "--n-heads", "32", "--n-kv-heads", "4"],
    },
    "quick": {
        "rmsnorm": ["--batches", "1", "--seq-lens", "1", "128", "--hidden-dims", "512"],
        "rope": ["--batches", "1", "--seq-lens", "1", "128", "--num-heads", "8",
                 "--head-dims", "64"],
        "swiglu": ["--batches", "1", "--seq-lens", "1", "128", "--intermediate-dims", "512"],
        "serving": ["--preset", "tiny", "--requests", "4", "--max-batch", "2",
                    "--median-prompt", "16", "--median-decode", "4", "--max-prompt", "32",
                    "--policies", "eager", "adaptive"],
        "profile": ["--batch-size", "1", "--seq-len", "32", "--hidden-dim", "128",
                    "--n-layers", "1", "--n-heads", "4", "--n-kv-heads", "2",
                    "--intermediate-dim", "256", "--no-profiler"],
    },
}


# --------------------------------------------------------------------------- env


def _nvidia_smi() -> dict[str, Any] | None:
    """Clock and power limits, which torch does not expose. Absent binary is fine."""
    if shutil.which("nvidia-smi") is None:
        return None
    fields = ["name", "driver_version", "clocks.max.sm", "clocks.current.sm",
              "power.limit", "power.draw", "memory.total", "temperature.gpu"]
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    gpus = [dict(zip(fields, [v.strip() for v in line.split(",")]))
            for line in out.splitlines() if line.strip()]
    return {"gpus": gpus}


def step_environment(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except Exception:
        triton_version = None

    info = get_device_info(torch.device(args.device))
    payload: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "triton": triton_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "triton_usable": is_triton_available(),
        "device": info.to_dict(),
        "compile_baseline": os.environ.get("AKO_ENABLE_COMPILE") == "1",
        "nvidia_smi": _nvidia_smi(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        payload["cuda_device"] = {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "total_memory_gb": round(props.total_memory / 1024**3, 2),
            "multi_processor_count": props.multi_processor_count,
        }
    (out / OUTPUTS["environment"]).write_text(json.dumps(payload, indent=2))
    return {"gpu": info.device_name}


# ------------------------------------------------------------------------ health


def step_health(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    """Measure the same thing twice; a drifting bench invalidates everything after it."""
    device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    x = torch.randn(4 if args.quick else 8, 512, 2048, dtype=dtype, device=device)
    weight = torch.ones(x.shape[-1], dtype=dtype, device=device)

    runs = [
        bench_op(rmsnorm_ref, (x, weight), warmup=5, repeats=args.repeats,
                 kernel_name="health_probe").latency_p50_ms
        for _ in range(2)
    ]
    drift = abs(runs[0] - runs[1]) / min(runs) if min(runs) > 0 else float("inf")
    healthy = drift <= args.health_tolerance
    payload = {
        "probe": "rmsnorm_ref",
        "shape": list(x.shape),
        "dtype": str(dtype).replace("torch.", ""),
        "run_p50_ms": [round(v, 5) for v in runs],
        "drift": round(drift, 4),
        "tolerance": args.health_tolerance,
        "healthy": healthy,
        "enforced": device.type == "cuda",
    }
    (out / OUTPUTS["health"]).write_text(json.dumps(payload, indent=2))

    if healthy:
        console.print(f"  bench drift {drift:.1%} (tolerance {args.health_tolerance:.0%}) — OK")
        return payload
    message = (
        f"bench drift {drift:.1%} > tolerance {args.health_tolerance:.0%} "
        f"({runs[0]:.4f} ms then {runs[1]:.4f} ms). The GPU is not stabilized or "
        "something else is using it. Close other workloads, let the clocks settle, "
        "and re-run; --health-tolerance raises the bar if you accept the noise."
    )
    if device.type != "cuda":
        # CPU timings are noisy by nature and no CUDA measurement depends on them.
        console.print(f"[yellow]  {message}[/yellow]")
        return payload
    raise RuntimeError(message)


# ---------------------------------------------------------------- script wrappers


def _run_script(name: str, script: str, argv: list[str], out: Path) -> dict[str, Any]:
    log = out / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), *argv],
        cwd=ROOT, capture_output=True, text=True,
    )
    log.write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
        raise RuntimeError(f"{script} exited {proc.returncode}; log: {log}\n{tail}")
    return {"log": str(log)}


def step_rmsnorm(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    return _run_script("rmsnorm", "benchmark_rmsnorm.py", [
        "--device", args.device, "--output-dir", str(out),
        "--repeats", str(args.repeats), "--dtypes", *args.dtypes,
        *GRID[args.mode]["rmsnorm"],
    ], out)


def step_rope(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    return _run_script("rope", "benchmark_rope.py", [
        "--device", args.device, "--output-dir", str(out),
        "--repeats", str(args.repeats), "--dtypes", *args.dtypes,
        *GRID[args.mode]["rope"],
    ], out)


def step_swiglu(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    return _run_script("swiglu", "benchmark_swiglu.py", [
        "--device", args.device, "--output-dir", str(out),
        "--repeats", str(args.repeats), "--dtypes", *args.dtypes,
        *GRID[args.mode]["swiglu"],
    ], out)


def step_serving(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    return _run_script("serving", "simulate_serving.py", [
        "--device", args.device, "--output-dir", str(out), "--dtype", args.dtypes[0],
        *GRID[args.mode]["serving"],
    ], out)


def step_profile(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    return _run_script("profile", "profile_model.py", [
        "--device", args.device, "--output-dir", str(out), "--dtype", args.dtypes[0],
        "--repeats", str(args.repeats), *GRID[args.mode]["profile"],
    ], out)


RUNNERS = {
    "environment": step_environment,
    "health": step_health,
    "rmsnorm": step_rmsnorm,
    "rope": step_rope,
    "swiglu": step_swiglu,
    "serving": step_serving,
    "profile": step_profile,
}


# ----------------------------------------------------------------------- summary


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _median_speedup(report: dict[str, Any] | None, *, decode: bool | None = None) -> float | None:
    """Median gain over the *fastest baseline*, which is the only quotable number."""
    if not report:
        return None
    values = []
    for row in report.get("comparisons", []):
        shape = row.get("shape") or [0, 0]
        if decode is not None and (len(shape) > 1 and shape[1] == 1) != decode:
            continue
        # No fallback to "speedup": that field is measured against the unfused
        # reference, so mixing it in here would quietly change the denominator.
        value = row.get("speedup_vs_best_baseline")
        if value:
            values.append(value)
    return statistics.median(values) if values else None


def summarize(out: Path) -> dict[str, Any]:
    rms = _load(out / OUTPUTS["rmsnorm"])
    rope = _load(out / OUTPUTS["rope"])
    swiglu = _load(out / OUTPUTS["swiglu"])
    profile = _load(out / OUTPUTS["profile"])
    serving = _load(out / OUTPUTS["serving"])

    decode_e2e = None
    if profile:
        decode_e2e = profile.get("measurements", {}).get("decode", {}).get(
            "speedup_vs_best_baseline"
        )

    adaptive_gain = None
    if serving:
        policies = {p: r for p, r in serving.get("policies", {}).items() if not r.get("failed")}
        adaptive = policies.pop("adaptive", None)
        others = [r["total_ms"] for r in policies.values() if r.get("total_ms")]
        if adaptive and adaptive.get("total_ms") and others:
            adaptive_gain = min(others) / adaptive["total_ms"]

    summary = {
        "rmsnorm_decode_speedup_vs_best_baseline": _median_speedup(rms, decode=True),
        "rmsnorm_prefill_speedup_vs_best_baseline": _median_speedup(rms, decode=False),
        "rope_speedup_vs_best_baseline": _median_speedup(rope),
        "swiglu_speedup_vs_best_baseline": _median_speedup(swiglu),
        "decode_end_to_end_speedup_vs_best_baseline": decode_e2e,
        "adaptive_vs_best_fixed_policy": adaptive_gain,
    }

    table = Table(title="Campaign summary — the six numbers")
    table.add_column("Metric")
    table.add_column("Value")
    labels = {
        "rmsnorm_decode_speedup_vs_best_baseline": "RMSNorm decode (seq=1) vs best baseline",
        "rmsnorm_prefill_speedup_vs_best_baseline": "RMSNorm prefill vs best baseline",
        "rope_speedup_vs_best_baseline": "RoPE vs best baseline",
        "swiglu_speedup_vs_best_baseline": "SwiGLU vs best baseline",
        "decode_end_to_end_speedup_vs_best_baseline": "Decoder decode step, end to end",
        "adaptive_vs_best_fixed_policy": "Adaptive vs best fixed policy (trace)",
    }
    for key, label in labels.items():
        value = summary[key]
        table.add_row(label, f"{value:.2f}x" if value else "n/a")
    console.print(table)
    console.print("[dim]Below 1.00x means the baseline wins on that shape class.[/dim]")
    return summary


# -------------------------------------------------------------------------- main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("reports/rtx3070"))
    parser.add_argument("--quick", action="store_true",
                        help="Sub-2-minute smoke pass; runs without a GPU")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Timed batches per measurement (default 30, 5 with --quick)")
    parser.add_argument("--dtypes", nargs="+", default=["fp16", "bf16"])
    parser.add_argument("--skip", nargs="+", default=[], choices=STEPS, metavar="STEP",
                        help=f"Steps to skip: {', '.join(STEPS)}")
    parser.add_argument("--resume", action="store_true",
                        help="Skip steps whose JSON already exists in --output")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--health-tolerance", type=float, default=0.10,
                        help="Max relative drift between two identical bench runs")
    args = parser.parse_args()

    args.mode = "quick" if args.quick else "full"
    if args.repeats is None:
        args.repeats = 5 if args.quick else 30
    if torch.device(args.device).type != "cuda":
        # The whole grid degrades to PyTorch baselines on CPU, and fp16/bf16 CPU
        # kernels are not representative of anything: measure fp32 only.
        args.dtypes = ["fp32"]
        console.print("[yellow]No CUDA device: CPU dry-run, fp32 only, Triton steps degrade "
                      "to PyTorch baselines.[/yellow]")
    # The honest end-to-end competitor is torch.compile, so enable it unless the
    # user decided otherwise — except in --quick, where compiling dominates the run.
    if not args.quick:
        os.environ.setdefault("AKO_ENABLE_COMPILE", "1")

    out: Path = args.output
    out.mkdir(parents=True, exist_ok=True)
    campaign_path = out / "campaign.json"

    planned = [s for s in STEPS if s not in args.skip]
    if args.resume:
        planned = [s for s in planned if not (out / OUTPUTS[s]).exists()]
    estimate = sum(ESTIMATE_S[args.mode][s] for s in planned)

    console.print(f"[bold]AKO campaign[/bold] — mode {args.mode}, device {args.device}, "
                  f"dtypes {' '.join(args.dtypes)}, output {out}")
    console.print(f"Steps: {', '.join(planned) or '(nothing to do)'}")
    console.print(f"Estimated duration: ~{estimate // 60} min {estimate % 60} s\n")

    campaign: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "device": args.device,
        "dtypes": args.dtypes,
        "repeats": args.repeats,
        "estimated_seconds": estimate,
        "planned": planned,
        "steps": {s: {"status": "skipped"} for s in STEPS if s not in planned},
    }

    def flush() -> None:
        campaign_path.write_text(json.dumps(campaign, indent=2))

    flush()
    for index, name in enumerate(planned, 1):
        console.print(f"[bold cyan][{index}/{len(planned)}] {name}[/bold cyan] "
                      f"(~{ESTIMATE_S[args.mode][name]}s)")
        started = time.perf_counter()
        try:
            extra = RUNNERS[name](args, out) or {}
            campaign["steps"][name] = {
                "status": "ok",
                "seconds": round(time.perf_counter() - started, 2),
                "output": OUTPUTS[name],
                **{k: v for k, v in extra.items() if k != "healthy"},
            }
        except Exception as exc:
            campaign["steps"][name] = {
                "status": "failed",
                "seconds": round(time.perf_counter() - started, 2),
                "error": f"{type(exc).__name__}: {exc}",
            }
            flush()
            console.print(f"[red]  {name} failed: {exc}[/red]")
            if name == "health":
                # A drifting bench poisons every later number: stop, keep what exists.
                console.print("[red]Aborting: fix the bench, then re-run "
                              "(--resume keeps what is already measured).[/red]")
                return 2
            console.print("[yellow]  continuing with the remaining steps[/yellow]")
            continue
        flush()
        console.print(f"  done in {campaign['steps'][name]['seconds']}s")

    campaign["summary"] = summarize(out)
    flush()

    failed = [s for s, v in campaign["steps"].items() if v["status"] == "failed"]
    console.print(f"\n[green]Campaign written to {campaign_path}[/green]")
    if failed:
        console.print(f"[red]Failed steps: {', '.join(failed)} (see {out / 'logs'})[/red]")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
