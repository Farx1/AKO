"""Free filtering: the arithmetic that rejects a candidate without a compiler.

ARCHITECTURE.md section 1.6 and section 3.3. Every predicate here costs zero GPU
time, so a rejection is written straight to the trial DB as
``verdict='infeasible'`` with the returned ``reason`` -- which is what turns
"39% of the grid never compiles" into a measured artifact instead of a claim.

The limits come from ``DeviceCaps`` (i.e. from ``get_device_info``), never from a
card model hardcoded here. The ``CPU_FALLBACK_*`` values below are used only when
the running machine reports nothing at all, so the predicates stay testable and
measurable on a box without a GPU.
"""

from __future__ import annotations

from math import ceil

from ako.search.spec import DeviceCaps, Problem

Params = dict[str, int | str | bool]

#: cc 8.6 / RTX 3070 figures, used ONLY when the device reports 0 (CPU-only box).
#: With a GPU present these are never read: the values come from the device
#: properties through ``DeviceCaps``.
CPU_FALLBACK_SMEM_PER_BLOCK = 101376
CPU_FALLBACK_SM_COUNT = 46
CPU_FALLBACK_TOTAL_MEM_GB = 8.0

#: bench_op's L2 flush buffer, allocated for the whole measurement (section 4).
FLUSH_BUFFER_BYTES = 32 * 1024 * 1024
#: Left to the display on a card that also drives a monitor.
DISPLAY_HEADROOM_BYTES = 600 * 1024 * 1024
#: Architectural per-thread register file limit. NOT 64: at 128x128 with 4 warps
#: a candidate needs 128 accumulator registers, well under the limit, and a
#: tighter bound would silently delete the tile family most likely to win at
#: large shapes (ROADMAP.md M3, "known defect to avoid").
MAX_REGS_PER_THREAD = 255
#: tl.dot refuses tiles smaller than 16 in any dimension.
MIN_DOT_TILE = 16


def smem_limit(caps: DeviceCaps) -> int:
    return caps.smem_per_block or CPU_FALLBACK_SMEM_PER_BLOCK


def sm_count(caps: DeviceCaps) -> int:
    return caps.sm_count or CPU_FALLBACK_SM_COUNT


def total_mem_bytes(caps: DeviceCaps) -> int:
    return int((caps.total_mem_gb or CPU_FALLBACK_TOTAL_MEM_GB) * 1024**3)


def smem_bytes(params: Params, prob: Problem) -> int:
    """``num_stages * (BLOCK_M + BLOCK_N) * BLOCK_K * itemsize`` (section 1.6).

    The pipelined A and B tiles of every stage live in shared memory at once.
    """
    return int(
        params["num_stages"]
        * (params["BLOCK_M"] + params["BLOCK_N"])
        * params["BLOCK_K"]
        * prob.itemsize()
    )


def waves(params: Params, prob: Problem, caps: DeviceCaps) -> float:
    """Fractional number of SM waves the launch grid needs."""
    tiles = (
        ceil(prob.m / int(params["BLOCK_M"]))
        * ceil(prob.n / int(params["BLOCK_N"]))
        * int(params.get("SPLIT_K", 1))
    )
    return tiles / sm_count(caps)


def wave_eff(params: Params, prob: Problem, caps: DeviceCaps) -> float:
    """Occupancy of the last wave: 1.0 means the grid divides the SM count."""
    w = waves(params, prob, caps)
    return w / ceil(w) if w else 0.0


def global_mem_bytes(params: Params, prob: Problem) -> int:
    """Inputs + fp32 oracle + split-K workspace + the L2 flush buffer.

    Compared against the device memory minus display headroom so an over-large
    candidate is a free ``infeasible`` row instead of a GPU-side OOM that
    fragments the caching allocator for every candidate after it (section 3.5).
    """
    it = prob.itemsize()
    elems = prob.m * prob.k + prob.k * prob.n + prob.m * prob.n
    inputs = elems * it + (prob.n * it if prob.bias else 0)
    oracle = elems * 4  # invariant 2: the reference is computed in fp32
    split_k = int(params.get("SPLIT_K", 1))
    # One fp32 partial tile per split. The old budget was a single buffer because
    # atomics accumulated in place; stores need the full stack.
    workspace = prob.m * prob.n * 4 * split_k if split_k > 1 else 0
    # Conservative full storage per promoted boundary, above the old estimate.
    boundary_mask = params.get("FP32_BOUNDARIES", 0)
    if type(boundary_mask) is not int or boundary_mask < 0:
        raise ValueError("FP32_BOUNDARIES must be a nonnegative integer")
    boundary_storage = boundary_mask.bit_count() * prob.m * prob.n * 4
    return inputs + oracle + workspace + boundary_storage + FLUSH_BUFFER_BYTES


def feasible(params: Params, prob: Problem, caps: DeviceCaps) -> str | None:
    """``None`` to keep the candidate, else the ``reason`` of its infeasible row.

    Nothing here compiles, allocates or touches the GPU.
    """
    if global_mem_bytes(params, prob) > total_mem_bytes(caps) - DISPLAY_HEADROOM_BYTES:
        return "global_mem"
    if params.get("ACC_FP16") and prob.dtype != "fp16":
        return "acc_fp16_requires_fp16"
    if "BLOCK_M" not in params:
        return None  # non-tiled block (torch, Inductor): nothing else is free

    bm, bn, bk = int(params["BLOCK_M"]), int(params["BLOCK_N"]), int(params["BLOCK_K"])
    if min(bm, bn, bk) < MIN_DOT_TILE:
        return "tile_under_dot_minimum"
    if smem_bytes(params, prob) > smem_limit(caps):
        return "smem"
    if bm * bn // (int(params["num_warps"]) * 32) > MAX_REGS_PER_THREAD:
        return "regs"
    split_k = int(params.get("SPLIT_K", 1))
    if split_k > 1 and prob.k // split_k < bk:
        return "splitk_over_k"
    return None
