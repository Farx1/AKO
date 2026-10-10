"""The candidate space: compositions x blocks x variant grid.

ARCHITECTURE.md sections 3.1, 3.2 and 3.6 step 1. Enumeration is deterministic
and does not touch the GPU: it yields ``(candidate, reason)`` pairs where a
non-None reason is the free rejection recorded as ``verdict='infeasible'``.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

from ako.search.blocks import BLOCKS, Block
from ako.search.constraints import Params
from ako.search.spec import Candidate, DeviceCaps, Problem

#: ARCHITECTURE.md section 3.2: the params of the triton_mm* blocks and, at the
#: same time, the constexpr tuple Triton specializes on. Insertion order fixes
#: the enumeration order.
GEMM_AXES: dict[str, tuple[object, ...]] = {
    "BLOCK_M": (16, 32, 64, 128, 256),
    "BLOCK_N": (32, 64, 128, 256),
    "BLOCK_K": (32, 64, 128),
    "GROUP_M": (1, 8),
    "SPLIT_K": (1, 2, 4, 8, 16),
    "num_warps": (2, 4, 8),
    "num_stages": (2, 3, 4, 5),
    "ACC_FP16": (False, True),
}

_SHORT = {"matmul": "mm"}


def gemm_grid(axes: dict[str, tuple[object, ...]] = GEMM_AXES) -> Iterator[Params]:
    """Cartesian product of the variant axes, in declaration order."""
    for values in itertools.product(*axes.values()):
        yield dict(zip(axes, values))


def partitions(chain: tuple[str, ...]) -> list[tuple[tuple[str, ...], ...]]:
    """Every contiguous partition of the chain: 2**(n-1) compositions."""
    out = []
    for cuts in itertools.product((False, True), repeat=max(len(chain) - 1, 0)):
        segments, current = [], [chain[0]]
        for cut, item in zip(cuts, chain[1:]):
            if cut:
                segments.append(tuple(current))
                current = []
            current.append(item)
        segments.append(tuple(current))
        out.append(tuple(segments))
    return out


def composition_name(segments: tuple[tuple[str, ...], ...]) -> str:
    """("matmul","bias"),("act") -> "mm+bias|act"."""
    return "|".join("+".join(_SHORT.get(s, s) for s in seg) for seg in segments)


def segments_of(name: str) -> tuple[tuple[str, ...], ...]:
    """Inverse of ``composition_name``: "mm+bias|act" -> ("matmul","bias"),("act")."""
    long = {v: k for k, v in _SHORT.items()}
    return tuple(tuple(long.get(s, s) for s in seg.split("+")) for seg in name.split("|"))


def _serves(block: Block, segment: tuple[str, ...]) -> bool:
    """Segment matching is ``set(segment) <= covers`` (section 3.3), with one
    extra rule: a block that carries a GEMM cannot serve an epilogue-only
    segment -- ``torch_elementwise`` is the tail block for those."""
    return set(segment) <= block.covers and ("matmul" in segment or "matmul" not in block.covers)


def enumerate_candidates(
    prob: Problem,
    caps: DeviceCaps,
    blocks: Iterable[Block] | None = None,
) -> Iterator[tuple[Candidate, str | None]]:
    """Yield every candidate for ``prob`` with its free-filter verdict.

    ``reason is None`` means the candidate survives to the correctness gate;
    anything else is an infeasible row that cost no GPU time.
    """
    blocks = list(BLOCKS.values() if blocks is None else blocks)
    for segments in partitions(prob.chain()):
        choices = [
            [b for b in blocks if _serves(b, seg) and b.supports(prob, caps)] for seg in segments
        ]
        if not all(choices):
            continue
        name = composition_name(segments)
        for combo in itertools.product(*choices):
            for params in itertools.product(*(list(b.grid(prob, caps)) for b in combo)):
                cand = Candidate(
                    composition=name,
                    blocks=tuple(b.name for b in combo),
                    params=tuple(tuple(sorted(p.items())) for p in params),
                )
                reason = None
                for block, block_params in zip(combo, params):
                    reason = block.feasible(block_params, prob, caps)
                    if reason is not None:
                        break
                yield cand, reason
