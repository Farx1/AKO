"""Turn a (problem, params) into a callable. Kept off :mod:`ako.search.blocks`
so enumeration does not import Triton kernels.

``build`` is M5's: the measurement wrapper decides which segment a block was
picked for, then this module materializes it.
"""

from __future__ import annotations

from typing import Callable

import torch

from ako.search.constraints import Params
from ako.search.correctness import apply_act
from ako.search.spec import Problem

KernelFn = Callable[..., torch.Tensor]


def torch_eager(prob: Problem, params: Params | None = None) -> KernelFn:
    """Readable unfused path: matmul, then bias, then activation."""
    del params

    def fn(
        a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = a @ b
        if prob.bias and bias is not None:
            out = out + bias.to(dtype=out.dtype)
        return apply_act(out, prob.act).to(dtype=a.dtype)

    return fn


def torch_addmm(prob: Problem, params: Params | None = None) -> KernelFn:
    """Vendor ``addmm``: matmul + bias, no activation."""
    del params

    def fn(
        a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if not prob.bias or bias is None:
            return a @ b
        return torch.addmm(bias.to(dtype=a.dtype), a, b)

    return fn


def torch_elementwise(prob: Problem, params: Params | None = None) -> KernelFn:
    """Bias and/or activation on an already-computed matmul result.

    Signature is ``(x, bias=None)`` — this block never owns the GEMM.
    """
    del params

    def fn(x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        out = x
        if prob.bias and bias is not None:
            out = out + bias.to(dtype=out.dtype)
        return apply_act(out, prob.act).to(dtype=x.dtype)

    return fn


def torch_addmm_act(prob: Problem, params: Params | None = None) -> KernelFn:
    """cuBLASLt fused epilogue. Only relu / gelu_tanh, matching the block filter.

    Torch 2.6 exposes this as ``use_gelu`` (bool). On CUDA the GELU is the tanh
    approximation, which is this project's ``gelu_tanh``. ReLU is ``use_gelu=False``.
    """
    del params
    use_gelu = prob.act != "relu"

    def fn(
        a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if bias is None:
            raise ValueError("torch_addmm_act requires a bias")
        return torch._addmm_activation(bias, a, b, use_gelu=use_gelu)

    return fn


def triton_mm_fused(prob: Problem, params: Params) -> KernelFn:
    from ako.search.kernels.mm_fused import launch

    segments = (prob.chain(),)

    def fn(
        a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return launch(a, b, bias, params, segments=segments, act=prob.act)

    return fn


def drop_last_k_block(prob: Problem, block_k: int = 1) -> KernelFn:
    """Near-miss negative control: skip the last ``block_k`` of the reduction.

    ROADMAP.md M5: at ``tol_scale = sqrt(K)/8`` this must be *rejected*. A gate
    that lets it through is too loose for ``ACC_FP16``.
    """

    def fn(
        a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        cut = max(a.shape[1] - block_k, 1)
        return torch_eager(prob)(a[:, :cut], b[:cut, :], bias)

    return fn
