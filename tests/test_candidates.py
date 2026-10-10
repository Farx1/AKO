"""M3: candidate space and free filtering (no GPU, no compiler).

The headline number of the milestone is the fraction of the variant grid the
arithmetic predicates remove for zero compile cost. ROADMAP.md M3 says the
filtering is dropped if it removes less than 10% of the grid, so the fraction is
computed, reported and asserted here.
"""

from __future__ import annotations

import dataclasses
import itertools
import warnings

import torch

from ako.search.blocks import BLOCKS
from ako.search.constraints import (
    feasible,
    global_mem_bytes,
    smem_bytes,
    smem_limit,
    total_mem_bytes,
    wave_eff,
    waves,
)
from ako.search.space import composition_name, enumerate_candidates, gemm_grid, partitions
from ako.search.spec import DeviceCaps, Problem, caps

CC86 = DeviceCaps(
    name="NVIDIA GeForce RTX 3070",
    cc=(8, 6),
    sm_count=46,
    smem_per_block=101376,  # shared_memory_per_block_optin
    regs_per_sm=65536,
    l2_bytes=4 * 1024**2,
    total_mem_gb=8.0,
    torch_version="x",
    triton_version="y",
)
CC80 = dataclasses.replace(CC86, name="synthetic cc80", cc=(8, 0), smem_per_block=164 * 1024)

FP16 = Problem(m=4096, n=4096, k=4096, dtype="fp16")

#: The exact grid of ROADMAP.md M3, defined here on purpose: adding the GROUP_M
#: and SPLIT_K axes multiplies the count without changing the predicate.
GRID_720 = [
    {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "num_warps": w, "num_stages": s}
    for bm, bn, bk, w, s in itertools.product(
        (16, 32, 64, 128, 256), (32, 64, 128, 256), (32, 64, 128), (2, 4, 8), (2, 3, 4, 5)
    )
]


def _report(line: str) -> None:
    """pytest hides stdout of a passing test; the warnings summary survives -q."""
    warnings.warn(line, stacklevel=2)


def test_partitions_are_the_four_compositions():
    segments = partitions(("matmul", "bias", "act"))
    assert len(segments) == 4
    assert [composition_name(s) for s in segments] == [
        "mm+bias+act",
        "mm+bias|act",
        "mm|bias+act",
        "mm|bias|act",
    ]


def test_smem_predicate_is_the_device_limit_not_a_card_model():
    cfg = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "num_warps": 4, "num_stages": 4}
    assert smem_bytes(cfg, FP16) == 131072
    assert smem_bytes(cfg, FP16) > smem_limit(CC86)
    assert feasible(cfg, FP16, CC86) == "smem"
    # Same config, bigger card: kept. The limit comes from DeviceCaps.
    assert feasible(cfg, FP16, CC80) is None
    live = caps()
    assert live.smem_per_block > 0
    if not torch.cuda.is_available():
        # CPU-only box: the explicit fallback must match the cc 8.6 fixture.
        assert smem_limit(live) == smem_limit(CC86)


def test_register_pressure_keeps_the_128x128_family():
    # The tile family most likely to win at large shapes: 128 accumulator
    # registers per thread, far under the 255 architectural limit.
    cfg = {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
    assert feasible(cfg, FP16, CC86) is None


def test_720_point_grid_keeps_441_on_cc86():
    kept = [c for c in GRID_720 if smem_bytes(c, FP16) <= smem_limit(CC86)]
    assert len(GRID_720) == 720
    assert len(kept) == 441
    # No rejected config sneaks past the predicate, and no kept config fails it.
    rejected = [c for c in GRID_720 if c not in kept]
    assert all(smem_bytes(c, FP16) > smem_limit(CC86) for c in rejected)
    _report(f"smem predicate on the 720-point grid: {len(kept)} kept, {len(rejected)} rejected")


def test_free_filtering_removes_at_least_ten_percent_of_the_full_grid():
    grid = list(gemm_grid())
    reasons = [feasible(cfg, FP16, CC86) for cfg in grid]
    rejected = [(cfg, r) for cfg, r in zip(grid, reasons) if r is not None]
    fraction = len(rejected) / len(grid)
    by_reason = {r: sum(1 for _, x in rejected if x == r) for _, r in rejected}
    _report(
        f"free filtering on {len(grid)} variants of {FP16}: "
        f"{len(rejected)} rejected = {fraction:.1%} {by_reason}"
    )
    # ROADMAP.md M3 kill criterion.
    assert fraction >= 0.10
    # Every rejection is arithmetically justified, and nothing kept violates a
    # predicate: the filter never guesses.
    for cfg, reason in rejected:
        if reason == "smem":
            assert smem_bytes(cfg, FP16) > smem_limit(CC86)
        elif reason == "acc_fp16_requires_fp16":
            assert cfg["ACC_FP16"] and FP16.dtype != "fp16"
    for cfg, reason in zip(grid, reasons):
        if reason is None:
            assert smem_bytes(cfg, FP16) <= smem_limit(CC86)
            assert global_mem_bytes(cfg, FP16) <= total_mem_bytes(CC86)


def test_bf16_rejects_fp16_accumulation_for_free():
    bf16 = dataclasses.replace(FP16, dtype="bf16")
    cfg = {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "num_warps": 4, "num_stages": 2}
    assert feasible(cfg | {"ACC_FP16": True}, bf16, CC86) == "acc_fp16_requires_fp16"
    assert feasible(cfg | {"ACC_FP16": False}, bf16, CC86) is None


def test_oversized_problem_is_infeasible_before_any_allocation():
    huge = dataclasses.replace(FP16, m=32768, n=32768, k=32768)
    assert feasible({}, huge, CC86) == "global_mem"
    assert feasible({}, FP16, CC86) is None


def test_wave_model_matches_the_numbers_the_plan_predicts():
    # ARCHITECTURE.md 1.6: at 1024**2, 128x128 gives 64 tiles over 46 SMs
    # (2 waves, 70%), 64x128 gives 128 tiles (93%).
    p = dataclasses.replace(FP16, m=1024, n=1024)
    big = {"BLOCK_M": 128, "BLOCK_N": 128}
    small = {"BLOCK_M": 64, "BLOCK_N": 128}
    assert waves(big, p, CC86) == 64 / 46
    assert round(wave_eff(big, p, CC86), 2) == 0.70
    assert round(wave_eff(small, p, CC86), 2) == 0.93


def test_enumeration_is_deterministic_and_complete():
    first = list(enumerate_candidates(FP16, CC86))
    second = list(enumerate_candidates(FP16, CC86))
    assert [c.hash() for c, _ in first] == [c.hash() for c, _ in second]
    assert len({c.hash() for c, _ in first}) == len(first)

    #   mm+bias+act : eager, addmm_act, compile x3                      =  5
    #   mm+bias|act : 6 gemm choices x torch_elementwise                =  6
    #   mm|bias+act : 6 gemm choices x torch_elementwise                =  6
    #   mm|bias|act : 6 gemm choices x elementwise x elementwise        =  6
    # plus the M4 template: 6 of those slots are a triton_mm* block, each
    # carrying the whole 14400-point variant grid.
    assert len(first) == 23 + 6 * len(list(gemm_grid()))
    assert {c.composition for c, _ in first} == {
        "mm+bias+act",
        "mm+bias|act",
        "mm|bias+act",
        "mm|bias|act",
    }
    # A candidate carries the verdict of the block that would run it -- and that
    # is the block's own predicate, not the shared one: triton_mm_fused adds the
    # M4 split-K rule on top of feasible().
    for cand, reason in first:
        verdicts = [
            BLOCKS[name].feasible(dict(p), FP16, CC86)
            for name, p in zip(cand.blocks, cand.params)
        ]
        assert reason == next((v for v in verdicts if v is not None), None)


def test_disabled_epilogue_shrinks_the_chain():
    plain = Problem(m=512, n=512, k=512, dtype="fp16", bias=False, act="none")
    assert plain.chain() == ("matmul",)
    cands = list(enumerate_candidates(plain, CC86))
    # torch_addmm_act needs a bias and relu/gelu_tanh, so it drops out here.
    assert {c.blocks[0] for c, _ in cands} == {
        "torch_eager",
        "torch_addmm",
        "torch_compile",
        "triton_mm",
        "triton_mm_fused",  # epilogue codes all at 0: the bare GEMM of the template
    }
    assert all(c.composition == "mm" for c, _ in cands)


def test_every_block_except_compile_has_a_builder() -> None:
    """``build`` is M5: enumeration must not import Triton, measurement must."""
    for name, block in BLOCKS.items():
        if name == "torch_compile":
            assert block.build is None
            continue
        assert block.build is not None, name
        prob = Problem(m=8, n=8, k=8, dtype="fp32", act="relu", bias=True)
        fn = block.build(prob, {})
        assert callable(fn)
