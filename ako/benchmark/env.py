"""GPU state that torch does not expose: clocks, temperature, power, throttling.

Every field degrades to ``None`` when ``nvidia-smi`` is missing or refuses a
query, because a measurement without an environment sample is still a
measurement, while a crashed overnight run is not.

**Ordering.** :func:`preflight` must run **before ``torch.cuda.init()``**: once
CUDA is initialized this process owns a context and shows up in
``--query-compute-apps``, so the sweep would trip its own assertion. It filters
its own PID as a second line of defence, but the utilization check has no such
escape, so call it first.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from typing import Any, Sequence

#: Queried per measurement point. ``nounits`` keeps the values parseable.
GPU_FIELDS = ("clocks.sm", "temperature.gpu", "power.draw")

#: NVML clock-event bitmask. A raw hex mask is not a reason, so it is decoded.
THROTTLE_BITS = {
    0x1: "gpu_idle",
    0x2: "applications_clocks_setting",
    0x4: "sw_power_cap",
    0x8: "hw_slowdown",
    0x10: "sync_boost",
    0x20: "sw_thermal_slowdown",
    0x40: "hw_thermal_slowdown",
    0x80: "hw_power_brake_slowdown",
    0x100: "display_clock_setting",
}


def _smi(query: str, *extra: str) -> list[list[str]] | None:
    """Run one ``nvidia-smi`` query; ``None`` when the binary is absent or fails."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits", *extra],
            capture_output=True, text=True, timeout=15, check=True,
        ).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    return [[v.strip() for v in line.split(",")] for line in out.splitlines() if line.strip()]


def _num(raw: str | None) -> float | None:
    """Parse a value, tolerating the ``[N/A]`` a card prints for a sensor it lacks."""
    try:
        return float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=1)
def _throttle_field() -> str | None:
    """The driver-dependent spelling: renamed ``clocks_throttle`` -> ``clocks_event`` in 535."""
    for name in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        if _smi(f"--query-gpu={name}") is not None:
            return name
    return None


def _throttle_names(raw: str | None) -> list[str] | str | None:
    if not raw:
        return None
    try:
        mask = int(raw, 16)
    except ValueError:
        return raw
    return [name for bit, name in THROTTLE_BITS.items() if mask & bit]


def sample_env(index: int = 0) -> dict[str, Any]:
    """Clock, temperature, power and active throttle reasons for GPU ``index``."""
    throttle = _throttle_field()
    fields: Sequence[str] = (*GPU_FIELDS, throttle) if throttle else GPU_FIELDS
    rows = _smi(f"--query-gpu={','.join(fields)}")
    row = rows[index] if rows and index < len(rows) else []
    values = dict(zip(fields, row))
    return {
        "sm_clock_mhz": _num(values.get("clocks.sm")),
        "temp_c": _num(values.get("temperature.gpu")),
        "power_w": _num(values.get("power.draw")),
        "throttle_reasons": _throttle_names(values.get(throttle)) if throttle else None,
    }


def preflight() -> dict[str, Any]:
    """Refuse to measure on a busy GPU. See the module docstring for the ordering.

    Raises:
        RuntimeError: another process is computing on the card, or it is already
            above 3% utilization. Both make every latency after it meaningless.
    """
    apps = _smi("--query-compute-apps=pid,used_gpu_memory")
    utilization = _smi("--query-gpu=utilization.gpu")
    if apps is None or utilization is None:
        return {"checked": False, "compute_apps": [], "utilization_gpu": None}

    mine = str(os.getpid())
    foreign = [" ".join(row) for row in apps if row and row[0] != mine]
    used = _num(utilization[0][0]) if utilization else None
    state = {"checked": True, "compute_apps": foreign, "utilization_gpu": used}

    problems = []
    if foreign:
        problems.append(f"compute apps on the GPU: {foreign}")
    if used is not None and used >= 3:
        problems.append(f"utilization.gpu = {used:.0f}% (want < 3%)")
    if problems:
        raise RuntimeError("GPU is not idle: " + "; ".join(problems))
    return state
