#!/usr/bin/env python3
"""Generate the README figures from the measured reports.

Every figure is derived from `reports/<hardware>/*.json`. Nothing is drawn that
was not measured, and re-running this after a new campaign refreshes the images.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Muted palette, readable in both GitHub themes.
BASELINE = "#8c8c8c"
TRITON = "#3b7dd8"
ADAPTIVE = "#2a9d63"
WARN = "#d1495b"
GRID = "#d9d9d9"

plt.rcParams.update(
    {
        "figure.dpi": 200,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
    }
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _latencies_per_config(report: Any) -> dict[tuple, dict[str, float]]:
    """``(shape, dtype) -> {kernel_name: latency}`` from a benchmark report."""
    per: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in (report or {}).get("results", []):
        per[(tuple(row["shape"]), row.get("dtype", "?"))][row["kernel_name"]] = row[
            "latency_p50_ms"
        ]
    return {k: v for k, v in per.items() if len(v) > 1}


def plot_penalty(reports: dict[str, Any], out: Path) -> None:
    """Penalty of each fixed strategy against the best available choice.

    This is the project's central claim: no fixed kernel choice is safe, and the
    cost of being wrong shows up in the tail, not the median.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0))

    for ax, (op_name, report) in zip(axes, reports.items()):
        per = _latencies_per_config(report)
        ratios: dict[str, list[float]] = defaultdict(list)
        for lat in per.values():
            best = min(lat.values())
            for kernel, ms in lat.items():
                ratios[kernel].append(ms / best)

        # Only the two real contenders. The unfused reference (5-9x off) and
        # PyTorch's own fused kernel (6.7x median here) are never competitive
        # and would flatten the scale; they are named in the caption instead.
        keep = {k: v for k, v in ratios.items() if k.endswith(("_triton", "_compiled"))}
        order = sorted(keep, key=lambda k: max(keep[k]))
        labels = ["always Triton" if k.endswith("_triton") else "always torch.compile" for k in order]

        medians = [sorted(keep[k])[len(keep[k]) // 2] for k in order]
        worst = [max(keep[k]) for k in order]
        y = list(range(len(order)))

        ax.barh(y, worst, color=WARN, alpha=0.30, label="worst case", height=0.5)
        ax.barh(y, medians, color=TRITON, label="median", height=0.5)
        ax.axvline(1.0, color=ADAPTIVE, linewidth=2.2, label="select per shape")

        for i, (w, m) in enumerate(zip(worst, medians)):
            ax.text(w + max(worst) * 0.015, i, f"{w:.2f}×", va="center", fontsize=8.5,
                    color=WARN, fontweight="bold")
            ax.text(1.01, i - 0.28, f"median {m:.2f}×", va="center", fontsize=7.5, color=TRITON)

        ax.set_yticks(y, labels, fontsize=9)
        ax.set_xlim(0.95, max(worst) * 1.16)
        ax.set_xlabel("slowdown vs the best choice for that shape")
        ax.set_title(f"{op_name} — {len(per)} shape × dtype configs", fontsize=10)

    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        "No fixed kernel choice is safe — the cost shows up in the tail",
        fontsize=12,
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def plot_baseline_gap(reports: dict[str, Any], out: Path) -> None:
    """Speedup against the unfused reference vs against the best baseline.

    The honest number is the right-hand column; the left one mostly measures the
    intermediates the reference materializes.
    """
    fig, ax = plt.subplots(figsize=(9.5, 3.4))

    labels, vs_ref, vs_best = [], [], []
    for op_name, report in reports.items():
        for dtype in ("float16", "bfloat16"):
            rows = [
                c
                for c in (report or {}).get("comparisons", [])
                if c.get("dtype") == dtype and c.get("speedup_vs_best_baseline")
            ]
            if not rows:
                continue
            labels.append(f"{op_name}\n{dtype.replace('float', 'fp').replace('bfp', 'bf')}")
            vs_ref.append(sorted(c["speedup"] for c in rows)[len(rows) // 2])
            vs_best.append(
                sorted(c["speedup_vs_best_baseline"] for c in rows)[len(rows) // 2]
            )

    x = range(len(labels))
    width = 0.38
    ax.bar([i - width / 2 for i in x], vs_ref, width, color=BASELINE, label="vs unfused reference")
    ax.bar([i + width / 2 for i in x], vs_best, width, color=TRITON, label="vs best baseline")
    ax.axhline(1.0, color=WARN, linewidth=1, linestyle="--")

    for i, (a, b) in enumerate(zip(vs_ref, vs_best)):
        ax.text(i - width / 2, a + 0.15, f"{a:.1f}×", ha="center", fontsize=8, color=BASELINE)
        ax.text(i + width / 2, b + 0.15, f"{b:.2f}×", ha="center", fontsize=8, color=TRITON)

    ax.set_xticks(list(x), labels)
    ax.set_ylabel("median speedup")
    ax.set_ylim(0, max(vs_ref) * 1.22)
    ax.legend(fontsize=8, framealpha=0.95)
    ax.set_title(
        "Which baseline you pick decides the headline number", fontsize=11
    )
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def plot_shape_sensitivity(report: Any, out: Path) -> None:
    """Where the Triton kernel wins and where it loses, by problem size."""
    fig, ax = plt.subplots(figsize=(9.5, 3.4))

    for dtype, marker, color in (("float16", "o", TRITON), ("bfloat16", "^", ADAPTIVE)):
        points = [
            (
                math.prod(c["shape"]) * 2 / (1024**2),  # MiB moved, 2 bytes/element
                c["speedup_vs_best_baseline"],
            )
            for c in (report or {}).get("comparisons", [])
            if c.get("dtype") == dtype and c.get("speedup_vs_best_baseline")
        ]
        if not points:
            continue
        xs, ys = zip(*sorted(points))
        ax.scatter(xs, ys, s=22, marker=marker, color=color, alpha=0.8,
                   label=dtype.replace("float", "fp").replace("bfp", "bf"))

    ax.axhline(1.0, color=WARN, linewidth=1.2, linestyle="--")
    ax.text(0.98, 0.06, "below the line: the Triton kernel loses", transform=ax.transAxes,
            ha="right", fontsize=8, color=WARN)
    ax.set_xscale("log")
    ax.set_xlabel("tensor size (MiB)")
    ax.set_ylabel("speedup vs best baseline")
    ax.legend(fontsize=8, framealpha=0.95)
    ax.set_title("The same kernel wins and loses depending on size", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("reports/rtx3070"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/images"))
    args = parser.parse_args()

    rms = _load(args.report_dir / "baseline_rmsnorm.json")
    rope = _load(args.report_dir / "rope_benchmark.json")
    if rms is None or rope is None:
        raise SystemExit(f"No reports in {args.report_dir}; run scripts/run_gpu_validation.sh first")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {"RMSNorm": rms, "RoPE": rope}
    plot_penalty(reports, args.output_dir / "fixed-strategy-penalty.png")
    plot_baseline_gap(reports, args.output_dir / "baseline-gap.png")
    plot_shape_sensitivity(rms, args.output_dir / "shape-sensitivity.png")


if __name__ == "__main__":
    main()
