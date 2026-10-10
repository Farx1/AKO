#!/usr/bin/env python3
"""Generate docs/RESULTS.md tables from reports/rtx3070/*.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _fmt_speedup(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}×"


def _fmt_err(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2e}"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _shape_cell(shape: Any) -> str:
    if isinstance(shape, list):
        return "(" + ", ".join(str(x) for x in shape) + ")"
    return str(shape)


def _comparison_table(data: Any) -> list[str]:
    lines = [
        "| Shape | Dtype | Baseline p50 (ms) | Triton p50 (ms) | Speedup | Max error |",
        "|-------|-------|-------------------|-----------------|---------|-----------|",
    ]
    if not data:
        lines.append("| *(missing report)* | | | | | |")
        return lines

    rows = data.get("comparisons") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        lines.append("| *(no comparisons in report)* | | | | | |")
        return lines

    for row in rows:
        lines.append(
            f"| `{_shape_cell(row.get('shape'))}` | {row.get('dtype', '?')} | "
            f"{_fmt_ms(_as_float(row.get('baseline_ms')))} | "
            f"{_fmt_ms(_as_float(row.get('candidate_ms')))} | "
            f"{_fmt_speedup(_as_float(row.get('speedup')))} | "
            f"{_fmt_err(_as_float(row.get('max_error')))} |"
        )
    return lines


def _selected_table(data: dict[str, Any] | None, label: str) -> list[str]:
    lines = [
        f"### {label}",
        "",
        "| Operation | Selected kernel | Backend | p50 (ms) | Speedup | Max error | Rejected |",
        "|-----------|-----------------|---------|----------|---------|-----------|----------|",
    ]
    if not data:
        lines.append("| *(missing file)* | | | | | | |")
        return lines

    ops = data.get("operations") or {}
    for op_name, sel in ops.items():
        rejected = sel.get("rejected") or []
        rejected_s = "[]" if not rejected else json.dumps(rejected, ensure_ascii=False)
        lines.append(
            f"| `{op_name}` | `{sel.get('selected_kernel')}` | {sel.get('backend', '—')} | "
            f"{_fmt_ms(_as_float(sel.get('latency_p50_ms')))} | "
            f"{_fmt_speedup(_as_float(sel.get('speedup')))} | "
            f"{_fmt_err(_as_float(sel.get('max_error')))} | `{rejected_s}` |"
        )
    return lines


def _profile_table(data: dict[str, Any] | None) -> list[str]:
    lines = [
        "| Pass | eager (ms) | torch.compile (ms) | AKO (ms) | vs eager | vs best baseline |",
        "|------|-----------|--------------------|----------|----------|------------------|",
    ]
    if not data:
        lines.append("| *(missing model_profile.json)* | | | | | |")
        return lines

    measurements = data.get("measurements") or data
    for pass_name in ("prefill", "decode"):
        block = measurements.get(pass_name) or {}
        best = block.get("best_baseline")
        vs_best = _fmt_speedup(_as_float(block.get("speedup_vs_best_baseline")))
        lines.append(
            f"| {pass_name} | {_fmt_ms(_as_float(block.get('torch_ms')))} | "
            f"{_fmt_ms(_as_float(block.get('torch_compile_ms')))} | "
            f"{_fmt_ms(_as_float(block.get('ako_ms')))} | "
            f"{_fmt_speedup(_as_float(block.get('speedup')))} | "
            f"{vs_best}{f' ({best})' if best else ''} |"
        )
    return lines


def _simulation_table(data: dict[str, Any] | None) -> list[str]:
    """Policy comparison over a simulated request trace."""
    lines = [
        "| Policy | Total (ms) | Prefill (ms) | Decode (ms) | TTFT p50 | TTFT p95 | vs best |",
        "|--------|-----------:|-------------:|------------:|---------:|---------:|--------:|",
    ]
    if not data:
        lines.append("| *(missing serving_simulation.json)* | | | | | | |")
        return lines

    policies = data.get("policies") or {}
    done = {p: r for p, r in policies.items() if not r.get("failed") and r.get("total_ms")}
    best = min((r["total_ms"] for r in done.values()), default=0.0)
    for name, row in sorted(policies.items(), key=lambda kv: kv[1].get("total_ms") or float("inf")):
        if row.get("failed"):
            lines.append(f"| {name} (failed) | — | — | — | — | — | — |")
            continue
        total = _as_float(row.get("total_ms"))
        lines.append(
            f"| {name} | {_fmt_ms(total)} | {_fmt_ms(_as_float(row.get('prefill_ms_total')))} | "
            f"{_fmt_ms(_as_float(row.get('decode_ms_total')))} | "
            f"{_fmt_ms(_as_float(row.get('ttft_p50_ms')))} | "
            f"{_fmt_ms(_as_float(row.get('ttft_p95_ms')))} | "
            f"{(f'{total / best:.2f}x' if best and total else '—')} |"
        )

    trace = data.get("trace") or {}
    lines += [
        "",
        f"Trace: {trace.get('requests')} requests in {trace.get('batches')} batches, "
        f"prompt lengths {trace.get('prompt_len_min')}-{trace.get('prompt_len_max')} "
        f"(median {trace.get('median_prompt')}), max batch {trace.get('max_batch')}, "
        f"seed {trace.get('seed')}.",
    ]
    crossover = data.get("crossover_batch") or {}
    if crossover:
        parts = [
            f"{k} at batch {v}" if v is not None else f"{k} never"
            for k, v in crossover.items()
        ]
        lines.append("")
        lines.append(f"Adaptive selection overtakes: {', '.join(parts)} (autotuning amortization).")
    return lines


def build_markdown(report_dir: Path) -> str:
    env_txt = report_dir / "environment.txt"
    env_block = (
        env_txt.read_text(encoding="utf-8").strip()
        if env_txt.exists()
        else "*(missing environment.txt)*"
    )
    pytest_txt = report_dir / "pytest_gpu.txt"
    pytest_block = (
        pytest_txt.read_text(encoding="utf-8").strip()
        if pytest_txt.exists()
        else "pytest -q -m gpu\n# expected: 18 passed"
    )

    rms = _load(report_dir / "baseline_rmsnorm.json")
    rope = _load(report_dir / "rope_benchmark.json")
    sel_fp16 = _load(report_dir / "selected_config_fp16.json")
    sel_bf16 = _load(report_dir / "selected_config_bf16.json")
    profile = _load(report_dir / "model_profile.json")
    simulation = _load(report_dir / "serving_simulation.json")

    parts = [
        "# AKO Results — NVIDIA GeForce RTX 3070 Laptop GPU",
        "",
        "Measured on a local WSL2 + CUDA setup. Numbers below come from",
        "`reports/rtx3070/` and are regenerated by `python scripts/summarize_reports.py`.",
        "",
        "## Methodology",
        "",
        "- **Warmup / repeats**: `optimize_model` defaults (`--warmup 10`, `--repeats 30`).",
        "- **Timing**: calls are timed in batches sized to ~1 ms of work, with CUDA",
        "  events for device time alongside amortized wall-clock. Reported latency is",
        "  per single call; `launch_overhead_ms` is the wall-clock minus device time.",
        "- **Baselines**: `rmsnorm_ref` (unfused reference), `torch.nn.functional.rms_norm`",
        "  (fused in PyTorch >= 2.4) and, when `AKO_ENABLE_COMPILE=1`, `torch.compile` —",
        "  for every operation, and for the decoder stack as a whole. PyTorch has no",
        "  fused rotary embedding, so `torch.compile` is the only real RoPE baseline.",
        "- **Dtypes**: fp16 and bf16 for selection; RMSNorm/RoPE benches cover both.",
        "- **Shapes**: representative decoder-block sizes (full grid in the JSON files).",
        "- **Numerical budget**: dtype-scaled via `ako.validation.error_budget()` (ulps in rejections).",
        "- **Seed**: PyTorch default RNG unless noted in a report file.",
        "",
        "## How to read the speedups",
        "",
        "`speedup` is measured against `rmsnorm_ref`, which upcasts to fp32 and",
        "materializes every intermediate — roughly 40 bytes of traffic per element",
        "against ~4 for the fused kernel. A large ratio against it is close to",
        "arithmetic, not a result.",
        "",
        "**`speedup_vs_best_baseline` is the number worth quoting**: the gain over the",
        "fastest baseline available on the machine, which is normally PyTorch's own",
        "fused RMSNorm rather than the reference. Rows without it predate the stronger",
        "baselines and are only comparable to the naive reference.",
        "",
        "## Known measurement limits",
        "",
        "- **Absolute latencies below ~0.15 ms** in reports produced before the batched",
        "  timing are dominated by the instrumentation, not the kernel. The symptom is",
        "  visible in the data: 8 points where doubling `hidden_dim` *lowered* the",
        "  measured Triton time, which is not physical. Ratios held; absolute numbers",
        "  at small shapes did not.",
        "- **The largest shapes stress an 8 GB laptop GPU.** `(16, 4096, 4096)` bf16",
        "  showed a 126 ms baseline against 40.9 ms in fp16 at identical element width —",
        "  allocator pressure or clock throttling late in the sweep, not a dtype effect.",
        "  The sweep now frees each shape and records `peak_memory_mb` per measurement.",
        "- **Decode is a regression, not a win.** The AKO path is slower than a",
        "  compiled stack at decode shapes. The selector benchmarks at those shapes and",
        "  still picks Triton, so the per-call benchmark disagrees with the loop: the",
        "  kernels are microseconds of device work, and whatever dominates is not the",
        "  kernel. `launch_overhead_ms` is now recorded per backend to settle it; the",
        "  operation sweep does not yet cover decode-sized shapes (batch x 1 token).",
        "",
        "## Environment",
        "",
        "```",
        env_block,
        "```",
        "",
        "## Kernel selection",
        "",
    ]
    parts.extend(_selected_table(sel_fp16 if isinstance(sel_fp16, dict) else None, "fp16"))
    parts.append("")
    parts.extend(_selected_table(sel_bf16 if isinstance(sel_bf16, dict) else None, "bf16"))
    parts.extend(
        [
            "",
            "## RMSNorm benchmark",
            "",
            *_comparison_table(rms),
            "",
            "## RoPE benchmark",
            "",
            *_comparison_table(rope),
            "",
            "## Serving simulation",
            "",
            "Policies replayed on one request trace: variable prompt lengths and dynamic",
            "batching keep the shapes moving, which is the situation a fixed kernel choice",
            "cannot cover.",
            "",
            *_simulation_table(simulation if isinstance(simulation, dict) else None),
            "",
            "## End-to-end decoder profile",
            "",
            *_profile_table(profile if isinstance(profile, dict) else None),
            "",
            "## GPU tests",
            "",
            "```",
            pytest_block,
            "```",
            "",
        ]
    )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/rtx3070"))
    parser.add_argument("--output", type=Path, default=Path("docs/RESULTS.md"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_markdown(args.report_dir), encoding="utf-8", newline="\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
