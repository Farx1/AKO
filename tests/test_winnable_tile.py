"""The fused-tile picker stays inside the free feasibility filter."""

from __future__ import annotations

from ako.search.constraints import feasible
from ako.search.spec import DeviceCaps, Problem
from scripts.winnable_region import best_fused_tile


def _cpu_caps() -> DeviceCaps:
    return DeviceCaps(
        name="cpu",
        cc=(0, 0),
        sm_count=0,
        smem_per_block=0,
        regs_per_sm=0,
        l2_bytes=0,
        total_mem_gb=0.0,
        torch_version="test",
        triton_version=None,
    )


def test_skinny_silu_tile_is_feasible_and_fused() -> None:
    prob = Problem(m=4096, n=4096, k=64, dtype="fp16", act="silu", bias=True)
    tile = best_fused_tile(prob, _cpu_caps())
    assert feasible(tile, prob, _cpu_caps()) is None
    assert tile["SPLIT_K"] == 1
    assert tile["ACC_FP16"] is False
