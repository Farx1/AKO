"""Bootstrap 95 % intervals of every speedup in a benchmark report.

Reads a report written by ``scripts/benchmark_rmsnorm.py`` or
``scripts/benchmark_rope.py`` (``results`` + ``comparisons``) and, for each
comparison row, recomputes ``med(best baseline) / med(candidate)`` from the raw
per-call samples with a seeded percentile bootstrap. A speedup is called
significant only when its interval excludes 1 (docs/THEORY.md, section 5).

Reports written before ``samples_ms`` was stored carry no raw samples: their
rows print ``n/a`` instead of an invented interval.

    python scripts/speedup_ci.py reports/rtx3070/baseline_rmsnorm.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ako.benchmark.bench_op import bootstrap_ratio_ci


def _key(row: dict[str, Any], kernel: str) -> tuple[Any, ...]:
    # One RoPE style per report file, so (kernel, shape, dtype) is unique.
    return (kernel, tuple(row["shape"]), row.get("dtype"))


def speedup_rows(report: dict[str, Any], *, seed: int = 0) -> list[dict[str, Any]]:
    samples = {_key(r, r["kernel_name"]): r.get("samples_ms") for r in report.get("results", [])}
    rows = []
    for comp in report.get("comparisons", []):
        cand = samples.get(_key(comp, comp["selected_kernel"]))
        # dtype / best_baseline_kernel are optional in older reports: no interval then.
        base = samples.get(_key(comp, comp.get("best_baseline_kernel")))
        row = {
            "shape": comp["shape"],
            "dtype": comp.get("dtype"),
            "candidate": comp["selected_kernel"],
            "baseline": comp.get("best_baseline_kernel"),
            "speedup_p50": comp.get("speedup_vs_best_baseline"),
            "speedup_median": None,
            "ci95": None,
            "significant": None,
        }
        if cand and base and min(cand) > 0:
            ratio, low, high = bootstrap_ratio_ci(base, cand, seed=seed)
            row.update(
                speedup_median=round(ratio, 4),
                ci95=[round(low, 4), round(high, 4)],
                significant=low > 1.0 or high < 1.0,
            )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("report", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="also write the rows as JSON")
    args = parser.parse_args()

    rows = speedup_rows(json.loads(args.report.read_text()), seed=args.seed)
    for r in rows:
        ci = "n/a (no raw samples)" if r["ci95"] is None else f"[{r['ci95'][0]}, {r['ci95'][1]}]"
        sig = {True: "significant", False: "not significant", None: ""}[r["significant"]]
        print(f"{str(r['shape']):>18} {r['dtype']!s:>8} {r['speedup_p50']!s:>7}  {ci}  {sig}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
