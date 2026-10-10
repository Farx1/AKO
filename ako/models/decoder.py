"""A small LLaMA-style decoder stack wired on top of the AKO operations.

The point is not to reimplement a model zoo: it is to run the AKO kernels in
the shape they actually appear in — inside a transformer block, around
attention and MLP — so that a kernel speedup can be turned into an end-to-end
latency number (paper §14.6).

Two interchangeable backends make that comparison possible:

- ``"torch"``: the unfused eager path (separate residual add, RMSNorm, RoPE,
  SwiGLU activation)
- ``"ako"``:   the same math through the adaptive dispatcher, which fuses the
  residual add into RMSNorm, the SwiGLU activation into one pass, and selects
  the best RoPE kernel for the device

Both must produce the same tensors; only their kernels differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from ako.ops.rmsnorm import rmsnorm, rmsnorm_residual
from ako.ops.rmsnorm_ref import rmsnorm_ref
from ako.ops.rope import rope
from ako.ops.rope_ref import build_rope_tables, rope_ref
from ako.ops.swiglu import swiglu

Backend = Literal["ako", "torch"]


@dataclass
class ModelConfig:
    """Shape of the decoder stack (defaults are TinyLlama-1.1B-like)."""

    hidden_dim: int = 2048
    n_layers: int = 4
    n_heads: int = 32
    n_kv_heads: int = 4
    intermediate_dim: int = 5632
    max_seq_len: int = 2048
    rope_base: float = 10000.0
    rope_style: str = "neox"
    eps: float = 1e-5

    @property
    def head_dim(self) -> int:
        if self.hidden_dim % self.n_heads:
            raise ValueError(
                f"hidden_dim {self.hidden_dim} is not divisible by n_heads {self.n_heads}"
            )
        return self.hidden_dim // self.n_heads

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads {self.n_heads} is not divisible by n_kv_heads {self.n_kv_heads}"
            )
        if self.head_dim % 2:
            raise ValueError(f"head_dim {self.head_dim} must be even for RoPE")


@dataclass
class KVCache:
    """Per-layer key/value cache for autoregressive decoding."""

    n_layers: int
    entries: list[tuple[torch.Tensor, torch.Tensor] | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.entries:
            self.entries = [None] * self.n_layers

    @property
    def length(self) -> int:
        first = self.entries[0]
        return 0 if first is None else first[0].shape[1]

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``(k, v)`` — laid out ``(batch, seq, heads, dim)`` — and return the full cache."""
        previous = self.entries[layer_idx]
        if previous is not None:
            k = torch.cat([previous[0], k], dim=1)
            v = torch.cat([previous[1], v], dim=1)
        self.entries[layer_idx] = (k, v)
        return k, v


#: Force a specific kernel per operation, bypassing selection. Used to measure a
#: fixed policy ("always Triton") against the adaptive one on the same workload.
KernelOverride = dict[str, str]


def _norm(
    backend: Backend,
    x: torch.Tensor,
    residual: torch.Tensor | None,
    weight: torch.Tensor,
    eps: float,
    override: KernelOverride | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Residual add + RMSNorm, returning ``(normed, residual)``."""
    override = override or {}
    if residual is None:
        residual = x
        if backend == "ako":
            return rmsnorm(x, weight, eps, kernel=override.get("rmsnorm")), residual
        return rmsnorm_ref(x, weight, eps), residual
    if backend == "ako":
        return rmsnorm_residual(
            x, residual, weight, eps, kernel=override.get("rmsnorm_residual")
        )
    residual = x + residual
    return rmsnorm_ref(residual, weight, eps), residual


def _rope(
    backend: Backend,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    style: str,
    override: KernelOverride | None = None,
) -> torch.Tensor:
    if backend != "ako":
        return rope_ref(x, cos, sin, style)
    return rope(x, cos, sin, style, kernel=(override or {}).get("rope"))


class Attention(nn.Module):
    """Grouped-query attention with RoPE and SDPA."""

    def __init__(
        self,
        config: ModelConfig,
        backend: Backend = "ako",
        kernel_override: KernelOverride | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.kernel_override = kernel_override
        self.repeats = config.n_heads // config.n_kv_heads
        kv_dim = config.n_kv_heads * config.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        layer_idx: int = 0,
        kv_cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        cfg = self.config

        q = self.q_proj(x).view(batch, seq_len, cfg.n_heads, cfg.head_dim)
        k = self.k_proj(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim)
        v = self.v_proj(x).view(batch, seq_len, cfg.n_kv_heads, cfg.head_dim)

        q = _rope(self.backend, q, cos, sin, cfg.rope_style, self.kernel_override)
        k = _rope(self.backend, k, cos, sin, cfg.rope_style, self.kernel_override)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        if self.repeats > 1:
            k = k.repeat_interleave(self.repeats, dim=2)
            v = v.repeat_interleave(self.repeats, dim=2)

        # (batch, seq, heads, dim) -> (batch, heads, seq, dim) for SDPA.
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=seq_len > 1,
        )
        return self.o_proj(attn.transpose(1, 2).reshape(batch, seq_len, cfg.hidden_dim))


class MLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(
        self,
        config: ModelConfig,
        backend: Backend = "ako",
        kernel_override: KernelOverride | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.kernel_override = kernel_override
        self.gate_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.intermediate_dim, bias=False)
        self.down_proj = nn.Linear(config.intermediate_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_proj(x), self.up_proj(x)
        if self.backend == "ako":
            activated = swiglu(gate, up, kernel=(self.kernel_override or {}).get("swiglu"))
        else:
            activated = F.silu(gate) * up
        return self.down_proj(activated)


class DecoderLayer(nn.Module):
    """One transformer block: norm -> attention -> norm -> MLP, residual-carried."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        backend: Backend = "ako",
        kernel_override: KernelOverride | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.backend = backend
        self.kernel_override = kernel_override
        self.attention = Attention(config, backend=backend, kernel_override=kernel_override)
        self.mlp = MLP(config, backend=backend, kernel_override=kernel_override)
        self.input_norm_weight = nn.Parameter(torch.ones(config.hidden_dim))
        self.post_attention_norm_weight = nn.Parameter(torch.ones(config.hidden_dim))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        kv_cache: KVCache | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eps = self.config.eps
        x, residual = _norm(
            self.backend, x, residual, self.input_norm_weight, eps, self.kernel_override
        )
        x = self.attention(x, cos, sin, layer_idx=self.layer_idx, kv_cache=kv_cache)

        x, residual = _norm(
            self.backend, x, residual, self.post_attention_norm_weight, eps, self.kernel_override
        )
        x = self.mlp(x)
        return x, residual


class DecoderStack(nn.Module):
    """A stack of decoder layers operating on hidden states (no embeddings/head).

    Args:
        config: Model shape.
        backend: ``"ako"`` for dispatched kernels, ``"torch"`` for the eager path.
    """

    def __init__(
        self,
        config: ModelConfig,
        backend: Backend = "ako",
        kernel_override: KernelOverride | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.kernel_override = kernel_override
        self.layers = nn.ModuleList(
            DecoderLayer(config, idx, backend=backend, kernel_override=kernel_override)
            for idx in range(config.n_layers)
        )
        self.final_norm_weight = nn.Parameter(torch.ones(config.hidden_dim))
        # Rotation tables are position-dependent but input-independent: build
        # them once up to max_seq_len and slice, like a real runtime would.
        self._rope_tables: tuple[torch.Tensor, torch.Tensor] | None = None

    def rope_tables(
        self, seq_len: int, *, offset: int = 0, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` for positions ``[offset, offset + seq_len)``."""
        end = offset + seq_len
        device = torch.device(device)
        cached = self._rope_tables
        if cached is None or cached[0].shape[0] < end or cached[0].device != device:
            self._rope_tables = build_rope_tables(
                max(end, self.config.max_seq_len),
                self.config.head_dim,
                base=self.config.rope_base,
                device=device,
                dtype=torch.float32,
            )
        cos, sin = self._rope_tables
        return cos[offset:end], sin[offset:end]

    def forward(self, x: torch.Tensor, *, kv_cache: KVCache | None = None) -> torch.Tensor:
        offset = kv_cache.length if kv_cache is not None else 0
        cos, sin = self.rope_tables(x.shape[1], offset=offset, device=x.device)

        residual: torch.Tensor | None = None
        for layer in self.layers:
            x, residual = layer(x, residual, cos, sin, kv_cache=kv_cache)

        x, _ = _norm(
            self.backend, x, residual, self.final_norm_weight, self.config.eps, self.kernel_override
        )
        return x

    def new_kv_cache(self) -> KVCache:
        return KVCache(n_layers=self.config.n_layers)


def build_stack(
    config: ModelConfig,
    *,
    backend: Backend = "ako",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    kernel_override: KernelOverride | None = None,
) -> DecoderStack:
    """Create a stack with reproducible random weights, ready for benchmarking."""
    torch.manual_seed(seed)
    model = DecoderStack(config, backend=backend, kernel_override=kernel_override)
    return model.to(device=device, dtype=dtype).eval()
