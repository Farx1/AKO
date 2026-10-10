"""PyTorch reference implementation of rotary position embeddings (RoPE).

Tensor layout used across AKO: ``x`` is ``(batch, seq_len, n_heads, head_dim)``
and the rotation tables ``cos`` / ``sin`` are ``(seq_len, head_dim // 2)``.

Two pairing conventions are supported:

- ``"neox"`` (default, GPT-NeoX / HF LLaMA): pairs ``(i, i + head_dim // 2)``
- ``"interleaved"`` (GPT-J, and the formulation used in the AKO paper):
  pairs ``(2i, 2i + 1)``
"""

from __future__ import annotations

import torch

ROPE_STYLES = ("neox", "interleaved")


def build_rope_tables(
    seq_len: int,
    head_dim: int,
    *,
    base: float = 10000.0,
    offset: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``(cos, sin)`` tables of shape ``(seq_len, head_dim // 2)``.

    Args:
        seq_len: Number of positions to materialize.
        head_dim: Per-head dimension (must be even).
        base: Rotary base frequency (``theta``).
        offset: First position index — use the KV cache length when decoding.
        device: Target device.
        dtype: Table dtype (frequencies are always computed in fp32).

    Returns:
        Tuple ``(cos, sin)``.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")

    half_dim = head_dim // 2
    # Frequencies are built in at least fp32 (fp64 when asked for) so that the
    # tables are never the precision bottleneck of the rotation.
    compute_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half_dim, device=device, dtype=compute_dtype) * 2.0 / head_dim)
    )
    positions = torch.arange(offset, offset + seq_len, device=device, dtype=compute_dtype)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def _validate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, style: str) -> None:
    if style not in ROPE_STYLES:
        raise ValueError(f"Unknown RoPE style '{style}', expected one of {ROPE_STYLES}")
    if x.ndim != 4:
        raise ValueError(f"Expected x of shape (batch, seq_len, n_heads, head_dim), got {tuple(x.shape)}")
    head_dim = x.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    if cos.shape != sin.shape:
        raise ValueError(f"cos/sin shape mismatch: {tuple(cos.shape)} vs {tuple(sin.shape)}")
    if cos.shape != (x.shape[1], head_dim // 2):
        raise ValueError(
            f"Expected cos/sin of shape {(x.shape[1], head_dim // 2)}, got {tuple(cos.shape)}"
        )


def rope_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    style: str = "neox",
) -> torch.Tensor:
    """Apply RoPE to ``x`` using precomputed tables.

    Args:
        x: Input ``(batch, seq_len, n_heads, head_dim)``.
        cos: Cosine table ``(seq_len, head_dim // 2)``.
        sin: Sine table ``(seq_len, head_dim // 2)``.
        style: ``"neox"`` or ``"interleaved"``.

    Returns:
        Rotated tensor with the same shape and dtype as ``x``.
    """
    _validate(x, cos, sin, style)

    compute_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    x_c = x.to(compute_dtype)
    # (seq_len, half) -> broadcast over (batch, seq_len, n_heads, half)
    cos_b = cos.to(compute_dtype)[None, :, None, :]
    sin_b = sin.to(compute_dtype)[None, :, None, :]

    if style == "neox":
        half = x.shape[-1] // 2
        x0, x1 = x_c[..., :half], x_c[..., half:]
        out = torch.cat([x0 * cos_b - x1 * sin_b, x0 * sin_b + x1 * cos_b], dim=-1)
    else:
        x0, x1 = x_c[..., 0::2], x_c[..., 1::2]
        rotated = torch.stack([x0 * cos_b - x1 * sin_b, x0 * sin_b + x1 * cos_b], dim=-1)
        out = rotated.flatten(start_dim=-2)

    return out.to(x.dtype)


def rope_qk_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    style: str = "neox",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to a query/key pair (convenience wrapper)."""
    return rope_ref(q, cos, sin, style), rope_ref(k, cos, sin, style)
