"""PyTorch reference implementation of RMSNorm."""

from __future__ import annotations

import torch


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square layer normalization (LLaMA-style).

    Args:
        x: Input tensor of shape ``(..., hidden_dim)``.
        weight: Per-channel scale of shape ``(hidden_dim,)``.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor, same shape as ``x``.
    """
    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_c = x.to(compute_dtype)
    variance = x_c.pow(2).mean(dim=-1, keepdim=True)
    normed = x_c * torch.rsqrt(variance + eps)
    return (normed * weight.to(compute_dtype)).to(x.dtype)


def rmsnorm_residual_ref(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + RMSNorm (LLaMA decoder block pattern).

    Computes ``new_residual = x + residual`` then normalizes it, which is the
    shape every transformer block uses around attention and MLP. Returning both
    tensors lets the caller keep the un-normalized residual stream.

    Args:
        x: Sub-layer output ``(..., hidden_dim)``.
        residual: Incoming residual stream, same shape as ``x``.
        weight: Per-channel scale of shape ``(hidden_dim,)``.
        eps: Small constant for numerical stability.

    Returns:
        Tuple ``(normed, new_residual)``.
    """
    if x.shape != residual.shape:
        raise ValueError(f"Shape mismatch: x {tuple(x.shape)} vs residual {tuple(residual.shape)}")
    new_residual = x + residual
    return rmsnorm_ref(new_residual, weight, eps), new_residual
