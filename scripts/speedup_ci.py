"""Bootstrap 95 % intervals of every speedup in a benchmark report.

Reads a report written by ``scripts/benchmark_rmsnorm.py`` or
``scripts/benchmark_rope.py`` (``results`` + ``comparisons``) and, for each
comparison row, recomputes ``med(best baseline) / med(candidate)`` from the raw
per-call samples with a seeded percentile bootstrap. A speedup is called
significant only when its interval excludes 1 (docs/THEORY.md, section 5).

Several reports of the same grid (one per fresh process) are pooled by
concatenating their samples, so run-to-run variance enters the interval;
the comparison rows come from the first report and the best baseline is
re-picked on the pooled medians.

Reports written before ``samples_ms`` was stored carry no raw samples: their
rows print ``n/a`` instead of an invented interval.

    python scripts/speedup_ci.py reports/rtx3070/baseline_rmsnorm.json
    python scripts/speedup_ci.py reports/rtx3070/samples/run*/rope_benchmark.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from ako.benchmark.bench_op import bootstrap_ratio_ci


def _key(row: dict[str, Any], kernel: str) -> tuple[Any, ...]:
    # One RoPE style per report file, so (kernel, shape, dtype) is unique.
    return (kernel, tuple(row["shape"]), row.get("dtype"))


def speedup_rows(*reports: dict[str, Any], seed: int = 0) -> list[dict[str, Any]]:
    samples: dict[tuple[Any, ...], list[float]] = {}
    for report in reports:
        for r in report.get("results", []):
            samples.setdefault(_key(r, r["kernel_name"]), []).extend(r.get("samples_ms") or [])
    rows = []
    for comp in reports[0].get("comparisons", []):
        cand_key = _key(comp, comp["selected_kernel"])
        cand = samples.get(cand_key)
        # Best baseline = other kernel with the lowest median over the pooled samples
        # (the per-report pick flips between runs). Old reports: the stored pick, no samples.
        others = {k[0]: v for k, v in samples.items() if k[1:] == cand_key[1:] and k != cand_key and v}
        baseline = (
            min(others, key=lambda k: statistics.median(others[k]))
            if others
            else comp.get("best_baseline_kernel")
        )
        base = others.get(baseline)
        row = {
            "shape": comp["shape"],
            "dtype": comp.get("dtype"),
            "candidate": comp["selected_kernel"],
            "baseline": baseline,
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
    parser.add_argument("report", type=Path, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="also write the rows as JSON")
    args = parser.parse_args()

    rows = speedup_rows(*(json.loads(p.read_text()) for p in args.report), seed=args.seed)
    for r in rows:
        ci = "n/a (no raw samples, or one <= 0)" if r["ci95"] is None else str(r["ci95"])
        sig = {True: "significant", False: "not significant", None: ""}[r["significant"]]
        med = r["speedup_median"] if r["speedup_median"] is not None else "n/a"
        print(f"{str(r['shape']):>18} {r['dtype']!s:>8} {r['baseline']!s:>20} {med!s:>7}"
              f"  {ci}  {sig}")
    lows = [r["ci95"][0] for r in rows if r["ci95"]]
    highs = [r["ci95"][1] for r in rows if r["ci95"]]
    print(f"faster (CI > 1): {sum(lo > 1 for lo in lows)}, slower (CI < 1): "
          f"{sum(hi < 1 for hi in highs)}, with interval: {len(lows)} / {len(rows)}")
    if args.output:
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
