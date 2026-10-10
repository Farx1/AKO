"""Blocks: metadata plus callables. Not a class hierarchy, not a DSL.

ARCHITECTURE.md section 3.3. ``covers`` is what makes a new op family a
registration rather than a composer edit: segment matching is
``set(segment) <= block.covers``.

The two Triton blocks are the *same* kernel (``ako.search.kernels.mm_fused``):
they differ only by the epilogue constexprs the composer hands it, which is why
M4 adds no code generator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Literal

import torch

from ako.search.constraints import Params, feasible
from ako.search.spec import DeviceCaps, Problem


@dataclass(frozen=True)
class Block:
    name: str
    covers: frozenset[str]
    supports: Callable[[Problem, DeviceCaps], bool]
    grid: Callable[[Problem, DeviceCaps], Iterator[Params]]
    feasible: Callable[[Params, Problem, DeviceCaps], str | None]
    # ponytail: build() is M5's. It needs the *segment* the block was picked for
    # (torch_eager serving {matmul,bias} must not apply the activation), and that
    # convention is decided by the measurement wrapper, not by enumeration.
    build: Callable[[Problem, Params], Callable[..., torch.Tensor]] | None = None
    compiles: bool = False
    cost_class: Literal["free", "jit", "autotune"] = "free"
    max_build_s: float = 30.0
    nondeterministic: bool = False


BLOCKS: dict[str, Block] = {}


def register(block: Block) -> Block:
    BLOCKS[block.name] = block
    return block


def _always(prob: Problem, caps: DeviceCaps) -> bool:
    return True


def _no_params(prob: Problem, caps: DeviceCaps) -> Iterator[Params]:
    yield {}


_MM_BIAS_ACT = frozenset({"matmul", "bias", "act"})

register(Block("torch_eager", _MM_BIAS_ACT, _always, _no_params, feasible))
register(Block("torch_addmm", frozenset({"matmul", "bias"}), _always, _no_params, feasible))
register(
    Block(
        # torch._addmm_activation: the cuBLASLt RELU_BIAS / GELU_BIAS epilogue,
        # the strongest vendor baseline. It has no other activation.
        "torch_addmm_act",
        _MM_BIAS_ACT,
        lambda prob, caps: prob.bias and prob.act in ("relu", "gelu_tanh"),
        _no_params,
        feasible,
    )
)
register(Block("torch_elementwise", frozenset({"bias", "act"}), _always, _no_params, feasible))
register(
    Block(
        "torch_compile",
        _MM_BIAS_ACT,
        _always,
        lambda prob, caps: iter(
            [{"mode": m} for m in ("default", "max-autotune", "reduce-overhead")]
        ),
        feasible,
        compiles=True,
        # Inductor runs its own exhaustive Triton autotune per new shape, so every
        # mode is pre-built in the unmeasured warm pass, never on the sweep clock.
        cost_class="autotune",
    )
)


def _gemm_grid(prob: Problem, caps: DeviceCaps) -> Iterator[Params]:
    from ako.search.space import gemm_grid  # noqa: PLC0415 - space imports blocks

    return gemm_grid()


def _triton_supports(prob: Problem, caps: DeviceCaps) -> bool:
    # Read from the caps, not from the running box: enumeration for a given
    # (problem, device) must not depend on where the enumerator runs.
    return caps.triton_version is not None


def _fused_feasible(params: Params, prob: Problem, caps: DeviceCaps) -> str | None:
    """``feasible`` plus the M4 rule: the search does not emit split-K on a fused GEMM.

    The kernel can reduce partials and then apply the epilogue once. The search
    still rejects that combination until a measurement shows it beats the
    unfused vendor path; opening it early would spend the sweep on an unproven cell.
    """
    if int(params.get("SPLIT_K", 1)) > 1:
        return "splitk_with_epilogue"
    return feasible(params, prob, caps)


# ponytail: serving a bare "matmul" segment, triton_mm_fused is triton_mm with the
# epilogue codes at 0 -- the same duplication the torch blocks already have there.
# Deduplicate in the sweep (by launch identity) only if the double compile cost
# shows up in M6.
register(
    Block(
        "triton_mm",
        frozenset({"matmul"}),
        _triton_supports,
        _gemm_grid,
        feasible,
        compiles=True,
        cost_class="jit",
        nondeterministic=True,  # SPLIT_K > 1 accumulates through atomic_add
    )
)
register(
    Block(
        "triton_mm_fused",
        _MM_BIAS_ACT,
        _triton_supports,
        _gemm_grid,
        _fused_feasible,
        compiles=True,
        cost_class="jit",
    )
)


def _bind_builders() -> None:
    """Attach ``build`` after registration so enumeration does not import Triton."""
    from ako.search import builders

    mapping: dict[str, Callable[[Problem, Params], Callable[..., torch.Tensor]]] = {
        "torch_eager": builders.torch_eager,
        "torch_addmm": builders.torch_addmm,
        "torch_addmm_act": builders.torch_addmm_act,
        "torch_elementwise": builders.torch_elementwise,
        "triton_mm": builders.triton_mm_fused,
        "triton_mm_fused": builders.triton_mm_fused,
    }
    for name, fn in mapping.items():
        BLOCKS[name] = replace(BLOCKS[name], build=fn)


_bind_builders()
