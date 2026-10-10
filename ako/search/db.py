"""The trial database: the artifact the whole project points at.

ARCHITECTURE.md section 3.5. SQLite from the stdlib, append-only, two tables so
a whole sweep is excludable by a join instead of a delete. Six methods, no more.

Two properties are load-bearing and easy to lose:

- an ``infeasible`` row is written *before* anything is compiled, at zero GPU
  cost, which is what turns "39% of the grid never compiles" into a measured
  artifact instead of a claim;
- ``blacklist()`` is deliberately not ``verdict != 'ok'``. ``infeasible``,
  ``oom`` and ``oracle_overflow`` are shape- or oracle-dependent, and pruning on
  them would permanently delete candidates that are valid at other shapes.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ako.search.spec import Candidate, DeviceCaps, Problem

VERDICTS = (
    "ok",
    "incorrect",
    "nan",
    "infeasible",
    "compile_error",
    "oom",
    "oracle_overflow",
)
#: Proven *wrong* on this GPU+dtype+toolchain at any shape. Never the others.
BLACKLIST_VERDICTS = ("incorrect", "nan", "compile_error")

SWEEP_STATUS = ("ok", "budget_exhausted", "drifted", "aborted")

#: Columns record() derives itself; callers may not pass them in **cols.
_DERIVED = (
    "ts",
    "sweep_id",
    "op",
    "shape_json",
    "m",
    "n",
    "k",
    "dtype",
    "act",
    "bias",
    "layout",
    "cand_hash",
    "composition",
    "blocks",
    "params",
    "gpu",
    "cc",
    "sm_count",
    "torch_version",
    "triton_version",
    "driver",
    "toolchain_key",
    "verdict",
)
#: Everything else a caller may fill in.
_OPTIONAL = (
    "reason",
    "max_abs_err",
    "err_ulps",
    "budget",
    "tol_scale",
    "seeds",
    "gate_runs",
    "stochastic",
    "magnitudes_json",
    "wall_p50_ms",
    "wall_p95_ms",
    "gpu_p50_ms",
    "cv",
    "iqr_ms",
    "peak_mem_mb",
    "compile_ms",
    "inner_iters",
    "l2_state",
    "graph_captured",
    "incumbent_hash",
    "incumbent_p50_ms",
    "speedup_vs_incumbent",
    "ci_lo",
    "ci_hi",
    "significant",
    "sm_clock_mhz",
    "temp_c",
    "power_w",
)
#: Timing columns. Invariant 1: nothing is timed before it passes the gate, so a
#: row carrying these with a failing verdict is a bug in the caller, not data.
_TIMING = ("wall_p50_ms", "wall_p95_ms", "gpu_p50_ms", "speedup_vs_incumbent")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sweep (
  id        TEXT PRIMARY KEY,
  started   TEXT NOT NULL, finished TEXT,
  budget_s  REAL, drift_pct REAL,
  status    TEXT,
  host TEXT, gpu TEXT, git_sha TEXT, notes TEXT
);

CREATE TABLE IF NOT EXISTS trial (
  id INTEGER PRIMARY KEY, ts TEXT NOT NULL,
  sweep_id TEXT NOT NULL REFERENCES sweep(id),

  op TEXT, shape_json TEXT NOT NULL, m INTEGER, n INTEGER, k INTEGER,
  dtype TEXT, act TEXT, bias INTEGER, layout TEXT,

  cand_hash TEXT NOT NULL, composition TEXT NOT NULL,
  blocks TEXT NOT NULL, params TEXT NOT NULL,

  gpu TEXT, cc TEXT, sm_count INTEGER,
  torch_version TEXT, triton_version TEXT, driver TEXT,
  toolchain_key TEXT NOT NULL,

  verdict TEXT NOT NULL,
  reason TEXT, max_abs_err REAL, err_ulps REAL, budget REAL,
  tol_scale REAL, seeds TEXT,
  gate_runs INTEGER, stochastic INTEGER,
  magnitudes_json TEXT,

  wall_p50_ms REAL, wall_p95_ms REAL, gpu_p50_ms REAL, cv REAL, iqr_ms REAL,
  peak_mem_mb REAL, compile_ms REAL, inner_iters INTEGER, l2_state TEXT,
  graph_captured INTEGER,
  incumbent_hash TEXT, incumbent_p50_ms REAL,
  speedup_vs_incumbent REAL, ci_lo REAL, ci_hi REAL, significant INTEGER,

  sm_clock_mhz INTEGER, temp_c INTEGER, power_w REAL
);

CREATE INDEX IF NOT EXISTS ix_trial_problem ON trial(op,dtype,act,m,n,k,gpu);
CREATE INDEX IF NOT EXISTS ix_trial_cand    ON trial(cand_hash,gpu,dtype);
CREATE INDEX IF NOT EXISTS ix_trial_verdict ON trial(verdict,gpu,dtype);
"""

DEFAULT_PATH = Path("reports/rtx3070/trials.sqlite")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TrialDB:
    """Append-only trial store. ``path=":memory:"`` for tests."""

    def __init__(self, path: str | Path = DEFAULT_PATH) -> None:
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        # One commit per trial, and a sweep is tens of thousands of trials, so
        # the default fsync-per-row costs more than the measurements. WAL +
        # synchronous=NORMAL still survives the failure mode that actually
        # happens here (process death: OOM, driver reset, kill).
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        # A sweep left with status NULL means the process died mid-run (OOM,
        # driver reset, kill). Stamping it here keeps partial rows out of every
        # query, all of which filter on status='ok'.
        self.conn.execute(
            "UPDATE sweep SET status='aborted', finished=? WHERE status IS NULL", (_now(),)
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ write

    def begin_sweep(self, prob: Problem, caps: DeviceCaps, budget_s: float) -> str:
        import platform
        import subprocess

        try:
            git_sha = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            ).stdout.strip()
        except Exception:
            git_sha = None
        sweep_id = uuid.uuid4().hex
        self.conn.execute(
            "INSERT INTO sweep (id,started,budget_s,host,gpu,git_sha,notes)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                sweep_id,
                _now(),
                budget_s,
                platform.node(),
                caps.name,
                git_sha,
                json.dumps(prob.as_dict(), sort_keys=True),
            ),
        )
        self.conn.commit()
        return sweep_id

    def end_sweep(self, sweep_id: str, *, drift_pct: float, status: str) -> None:
        if status not in SWEEP_STATUS:
            raise ValueError(f"unknown sweep status {status!r}")
        self.conn.execute(
            "UPDATE sweep SET finished=?, drift_pct=?, status=? WHERE id=?",
            (_now(), drift_pct, status, sweep_id),
        )
        self.conn.commit()

    def record(
        self,
        sweep_id: str,
        prob: Problem,
        cand: Candidate,
        caps: DeviceCaps,
        *,
        verdict: str,
        **cols: object,
    ) -> None:
        """Write one trial. ``verdict`` is mandatory: a row without the gate's
        result is not a measurement, and an ``infeasible`` row costs no GPU."""
        if verdict not in VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}")
        unknown = set(cols) - set(_OPTIONAL)
        if unknown:
            raise ValueError(f"unknown or derived trial columns: {sorted(unknown)}")
        if verdict == "ok" and cols.get("max_abs_err") is None:
            raise ValueError("verdict='ok' requires max_abs_err from the correctness gate")
        if verdict != "ok":
            timed = [c for c in _TIMING if cols.get(c) is not None]
            if timed:
                raise ValueError(f"verdict={verdict!r} must not carry timings: {timed}")

        row: dict[str, object] = {
            "ts": _now(),
            "sweep_id": sweep_id,
            "op": prob.op,
            "shape_json": json.dumps(prob.as_dict(), sort_keys=True),
            "m": prob.m,
            "n": prob.n,
            "k": prob.k,
            "dtype": prob.dtype,
            "act": prob.act,
            "bias": int(prob.bias),
            "layout": prob.layout,
            "cand_hash": cand.hash(),
            "composition": cand.composition,
            "blocks": json.dumps(list(cand.blocks)),
            "params": cand.params_json(),
            "gpu": caps.name,
            "cc": f"{caps.cc[0]}.{caps.cc[1]}",
            "sm_count": caps.sm_count,
            "torch_version": caps.torch_version,
            "triton_version": caps.triton_version,
            "driver": caps.driver,
            "toolchain_key": caps.toolchain_key,
            "verdict": verdict,
        }
        row.update(cols)
        names = ",".join(row)
        self.conn.execute(
            f"INSERT INTO trial ({names}) VALUES ({','.join('?' * len(row))})",
            tuple(row.values()),
        )
        self.conn.commit()

    # ------------------------------------------------------------------- read

    def blacklist(self, caps: DeviceCaps, dtype: str, toolchain_key: str) -> set[str]:
        """cand_hash with verdict in (incorrect, nan, compile_error) on this
        GPU+dtype+toolchain_key. Never 'infeasible', 'oom' or 'oracle_overflow'."""
        marks = ",".join("?" * len(BLACKLIST_VERDICTS))
        rows = self.conn.execute(
            "SELECT DISTINCT t.cand_hash FROM trial t JOIN sweep s ON s.id=t.sweep_id"
            f" WHERE s.status='ok' AND t.gpu=? AND t.dtype=? AND t.toolchain_key=?"
            f" AND t.verdict IN ({marks})",
            (caps.name, dtype, toolchain_key, *BLACKLIST_VERDICTS),
        ).fetchall()
        return {r["cand_hash"] for r in rows}

    def neighbours(
        self, prob: Problem, caps: DeviceCaps, *, limit: int = 3
    ) -> list[tuple[Problem, list[tuple[str, float]]]]:
        """Measured problems nearest in log-(m,n,k), with their ranked candidates."""
        where = (
            "s.status='ok' AND t.verdict='ok' AND t.gpu=? AND t.toolchain_key=?"
            " AND t.op=? AND t.dtype=? AND t.act=?"
        )
        args = (caps.name, caps.toolchain_key, prob.op, prob.dtype, prob.act)
        shapes = self.conn.execute(
            "SELECT DISTINCT t.shape_json, t.m, t.n, t.k FROM trial t"
            f" JOIN sweep s ON s.id=t.sweep_id WHERE {where}",
            args,
        ).fetchall()

        def dist(r: sqlite3.Row) -> float:
            return math.dist(
                [math.log(r["m"]), math.log(r["n"]), math.log(r["k"])],
                [math.log(prob.m), math.log(prob.n), math.log(prob.k)],
            )

        out = []
        for r in sorted(shapes, key=dist)[:limit]:
            ranked = self.conn.execute(
                "SELECT t.cand_hash, t.speedup_vs_incumbent AS sp FROM trial t"
                f" JOIN sweep s ON s.id=t.sweep_id WHERE {where}"
                " AND t.shape_json=? AND t.speedup_vs_incumbent IS NOT NULL"
                " ORDER BY sp DESC",
                (*args, r["shape_json"]),
            ).fetchall()
            out.append(
                (
                    Problem(**json.loads(r["shape_json"])),
                    [(x["cand_hash"], x["sp"]) for x in ranked],
                )
            )
        return out

    def fit_features(
        self, op: str, dtype: str, gpu: str, toolchain_key: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Design matrix and target for the deferred lstsq prior. The target is
        log(speedup_vs_incumbent) against the sweep's frozen incumbent, not
        log(wall_p50_ms): absolute latency is not a usable fit target on a card
        with unlockable clocks, a paired ratio is invariant to clock state.
        Rows whose incumbent_hash is not the sweep's frozen incumbent are
        excluded, so one sweep has one denominator."""
        rows = self.conn.execute(
            "WITH frozen AS ("
            "  SELECT sweep_id, incumbent_hash, MIN(id) FROM trial"
            "  WHERE incumbent_hash IS NOT NULL GROUP BY sweep_id)"
            " SELECT t.m, t.n, t.k, t.speedup_vs_incumbent AS sp FROM trial t"
            " JOIN sweep s ON s.id=t.sweep_id"
            " JOIN frozen f ON f.sweep_id=t.sweep_id"
            "  AND f.incumbent_hash=t.incumbent_hash"
            " WHERE s.status='ok' AND t.verdict='ok' AND t.op=? AND t.dtype=?"
            "  AND t.gpu=? AND t.toolchain_key=? AND t.speedup_vs_incumbent>0",
            (op, dtype, gpu, toolchain_key),
        ).fetchall()
        # ponytail: shape-only design matrix [1, log2 m, log2 n, log2 k]. M8 owns
        # the prior and adds its own candidate features by joining on params.
        x = np.array(
            [[1.0, math.log2(r["m"]), math.log2(r["n"]), math.log2(r["k"])] for r in rows],
            dtype=float,
        ).reshape(-1, 4)
        y = np.array([math.log(r["sp"]) for r in rows], dtype=float)
        return x, y
