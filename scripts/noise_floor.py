"""M1 — can this card resolve the effect at all?

Re-times the fixed 4096-cubed fp16 canary at a steady cadence while keeping the
GPU loaded in between, and records the timestamp, clock, temperature, power and
throttle reasons of every point. It answers one question: is the measurement
noise on this board smaller than the effects the search is supposed to rank, or
does it eat them?

    python scripts/noise_floor.py --minutes 15

The defaults are M1's protocol: 10 s preheat, then 20 measurements over 15 min.
``--minutes 480`` turns the same script into an overnight drift trace at the same
45 s cadence.

Safe to leave unattended: the JSON is rewritten atomically after every point, so
Ctrl-C, a crash or a power cut keeps everything measured so far, verdict
included. The card is kept under load between points on purpose — a noise floor
measured on an idle GPU is not the thermal state a sweep runs in.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from rich.console import Console

from ako.benchmark.bench_op import bench_op
from ako.benchmark.canary import CANARY_N, canary_result
from ako.benchmark.env import preflight, sample_env
from ako.dispatcher.hardware import get_device_info

console = Console()

#: M1's gate. Drift is |last - first| / first on the canary p50; cv is the
#: within-measurement coefficient of variation.
DRIFT_PASS, DRIFT_STOP = 0.05, 0.15
CV_PASS, CV_STOP = 0.05, 0.10

ACTIONS = {
    "pass": "Ranking is publishable on this card: absolute latencies are stable "
            "enough to compare across measurements.",
    "paired_only": "Drift exceeds the effects being ranked. Paired interleaved "
                   "ratios only, never absolute rankings; M8 fits "
                   "log(speedup_vs_incumbent), not log(wall_p50_ms).",
    "stop": "No ranking is publishable on this card, and renting a stabler one is "
            "out of scope: the project reduces to the CPU-side composer, the "
            "schema and the feasibility map, and says so.",
    "insufficient": "Fewer than two measurements: no drift can be computed.",
}


def verdict(points: list[dict[str, Any]]) -> dict[str, Any]:
    """The M1 stop gate, plus the numbers that justify it.

    Pure function of the recorded points so it can be checked on simulated ones.
    """
    p50s = [p["p50_ms"] for p in points]
    if len(p50s) < 2:
        return {"verdict": "insufficient", "action": ACTIONS["insufficient"], "points": len(p50s)}

    cvs = sorted(p["cv"] for p in points)
    # p95 rather than max: over an 8 h trace a single contaminated point would
    # otherwise decide the project. At M1's 20 points the two are the same value.
    cv_p95 = cvs[min(int(len(cvs) * 0.95), len(cvs) - 1)]
    drift = (p50s[-1] - p50s[0]) / p50s[0] if p50s[0] else float("inf")
    median = statistics.median(p50s)

    if abs(drift) >= DRIFT_STOP or cv_p95 > CV_STOP:
        name = "stop"
    elif abs(drift) >= DRIFT_PASS or cv_p95 >= CV_PASS:
        name = "paired_only"
    else:
        name = "pass"

    return {
        "verdict": name,
        "action": ACTIONS[name],
        "criterion": "|drift(first, last)| < 0.05 and per-measurement cv < 0.05",
        "points": len(p50s),
        "p50_first_ms": round(p50s[0], 5),
        "p50_last_ms": round(p50s[-1], 5),
        "drift_first_last": round(drift, 5),
        "cv_p95": round(cv_p95, 5),
        "cv_max": round(cvs[-1], 5),
        "cv_median": round(statistics.median(cvs), 5),
        "p50_median_ms": round(median, 5),
        "p50_min_ms": round(min(p50s), 5),
        "p50_max_ms": round(max(p50s), 5),
        # The smallest speedup this card can tell apart from its own noise.
        "p50_spread_pct": round(100 * (max(p50s) - min(p50s)) / median, 3) if median else None,
        "thresholds": {"drift_pass": DRIFT_PASS, "drift_stop": DRIFT_STOP,
                       "cv_pass": CV_PASS, "cv_stop": CV_STOP},
    }


def flush_probe(device: str, n: int = 1024) -> dict[str, Any]:
    """Same matmul with and without the L2 flush, reported not asserted.

    A zero delta means the flush is not working, and every wave-quantization
    number downstream would be measured on a hot L2.
    """
    dtype = torch.float16 if torch.device(device).type == "cuda" else torch.float32
    a, b = (torch.randn(n, n, dtype=dtype, device=device) for _ in range(2))
    common: dict[str, Any] = dict(warmup=10, repeats=30, kernel_name="flush_probe")
    hot = bench_op(torch.matmul, (a, b), flush_l2=False, **common)
    cold = bench_op(torch.matmul, (a, b), flush_l2=True, **common)
    delta = (cold.latency_p50_ms - hot.latency_p50_ms) / hot.latency_p50_ms if hot.latency_p50_ms else 0.0
    return {
        "shape": [n, n, n],
        "unflushed_p50_ms": round(hot.latency_p50_ms, 5),
        "flushed_p50_ms": round(cold.latency_p50_ms, 5),
        "flush_cost_ms": round(cold.l2_flush_ms, 5),
        "delta_pct": round(100 * delta, 3),
        "l2_state": [hot.l2_state, cold.l2_state],
        "flush_effective": abs(delta) > 0.01,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Atomic rewrite: an interrupt mid-write must not eat the night's data."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _load_until(deadline: float, device: str, n: int) -> None:
    """Keep the card busy until ``deadline`` with the canary's own workload."""
    while time.perf_counter() < deadline:
        canary_result(device, n=n, repeats=3)


def _measure(index: int, started: float, device: str, n: int, repeats: int) -> dict[str, Any]:
    result = canary_result(device, n=n, repeats=repeats)
    return {
        "index": index,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "p50_ms": result.latency_p50_ms,
        "gpu_p50_ms": result.gpu_p50_ms,
        "cv": result.cv,
        "iqr_ms": round(result.iqr_ms, 5),
        "inner_iters": result.inner_iters,
        **sample_env(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("reports/rtx3070/noise_floor.json"))
    parser.add_argument("--minutes", type=float, default=15.0,
                        help="Target wall clock of the whole run (default 15, M1's protocol)")
    parser.add_argument("--interval", type=float, default=45.0,
                        help="Seconds between two measurement starts (default 45)")
    parser.add_argument("--preheat", type=float, default=10.0,
                        help="Seconds of untimed load before the first measurement")
    parser.add_argument("--repeats", type=int, default=20, help="Timed batches per measurement")
    parser.add_argument("--n", type=int, default=CANARY_N,
                        help=f"Canary matrix size; leave at {CANARY_N} for any recorded run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Before anything touches CUDA (see ako.benchmark.env for why the order matters).
    try:
        gate = {**preflight(), "ok": True}
    except RuntimeError as exc:
        # Recorded and carried into the report rather than aborting: an overnight
        # run that refuses to start at second 0 costs the whole night.
        gate = {"checked": True, "ok": False, "error": str(exc)}
        console.print(f"[yellow]preflight: {exc}[/yellow]")

    points: list[dict[str, Any]] = []
    planned = max(2, round((args.minutes * 60 - args.preheat) / args.interval))
    payload: dict[str, Any] = {
        "milestone": "M1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {"minutes": args.minutes, "interval_s": args.interval,
                   "preheat_s": args.preheat, "repeats": args.repeats, "n": args.n,
                   "device": args.device, "planned_points": planned},
        "device": get_device_info(args.device).to_dict(),
        "preflight": gate,
        "status": "running",
        "measurements": points,
        "verdict": verdict(points),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, payload)

    console.print(f"[bold]M1 noise floor[/bold] — {planned} points, {args.n} cubed on "
                  f"{args.device}, ~{args.minutes:.0f} min, output {args.output}")

    status = "ok"
    try:
        payload["l2_flush_probe"] = flush_probe(args.device)
        console.print(f"  L2 flush delta {payload['l2_flush_probe']['delta_pct']:+.1f}% "
                      f"at 1024 cubed")
        started = time.perf_counter()
        _load_until(started + args.preheat, args.device, args.n)
        for index in range(planned):
            _load_until(started + args.preheat + index * args.interval, args.device, args.n)
            point = _measure(index, started, args.device, args.n, args.repeats)
            points.append(point)
            payload["verdict"] = verdict(points)
            _write(args.output, payload)
            console.print(
                f"  [{index + 1}/{planned}] t={point['elapsed_s']:.0f}s "
                f"p50={point['p50_ms']:.4f} ms cv={point['cv']:.3f} "
                f"clock={point['sm_clock_mhz']} MHz temp={point['temp_c']} C "
                f"power={point['power_w']} W throttle={point['throttle_reasons']}"
            )
    except KeyboardInterrupt:
        status = "interrupted"
        console.print("[yellow]interrupted; keeping what is measured[/yellow]")
    except Exception as exc:
        status = f"failed: {type(exc).__name__}: {exc}"
        console.print(f"[red]{status}[/red]")
    finally:
        payload["status"] = status
        payload["verdict"] = verdict(points)
        _write(args.output, payload)

    final = payload["verdict"]
    console.print(f"\n[bold]verdict: {final['verdict']}[/bold] — {final['action']}")
    if final["verdict"] != "insufficient":
        console.print(f"drift {final['drift_first_last']:+.2%}, cv p95 {final['cv_p95']:.3f}, "
                      f"p50 spread {final['p50_spread_pct']:.2f}% of median")
    return 0 if final["verdict"] in ("pass", "paired_only") else 1


if __name__ == "__main__":
    raise SystemExit(main())
