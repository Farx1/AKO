"""Order candidates before they are measured.

ROADMAP.md M6: ``prior`` is ``none`` (enumeration order) or ``nn`` (nearest
measured shapes in the trial DB). No learned cost model at this milestone.
"""

from __future__ import annotations

from ako.search.db import TrialDB
from ako.search.spec import Candidate, DeviceCaps, Problem

Prior = str  # "none" | "nn"


def order(
    pending: list[tuple[Candidate, str | None]],
    db: TrialDB,
    prob: Problem,
    caps: DeviceCaps,
    *,
    prior: Prior = "none",
) -> list[tuple[Candidate, str | None]]:
    """Return ``pending`` in the order the search should visit it.

    Infeasible rows stay in the stream: they cost nothing to record and the
    feasibility map is part of the product. ``nn`` only reshuffles the
    still-legal ones to the front, using neighbour speedups when they exist.
    """
    if prior == "none":
        return list(pending)
    if prior != "nn":
        raise ValueError(f"unknown prior {prior!r}; expected 'none' or 'nn'")

    scores: dict[str, float] = {}
    for neighbour, ranked in db.neighbours(prob, caps):
        del neighbour
        for cand_hash, speedup in ranked:
            scores[cand_hash] = max(scores.get(cand_hash, 0.0), speedup)

    def key(item: tuple[Candidate, str | None]) -> tuple[int, float, str]:
        cand, reason = item
        # Infeasible last among equals, then higher neighbour score first.
        return (0 if reason is None else 1, -scores.get(cand.hash(), 0.0), cand.hash())

    return sorted(pending, key=key)
