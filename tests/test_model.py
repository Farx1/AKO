"""Tests for the decoder stack used to measure kernels end-to-end."""

from __future__ import annotations

import pytest
import torch

from ako.models.decoder import KVCache, ModelConfig, build_stack
from ako.validation.numerical import assert_close

SMALL = ModelConfig(
    hidden_dim=64,
    n_layers=2,
    n_heads=4,
    n_kv_heads=2,
    intermediate_dim=128,
    max_seq_len=64,
)


def test_config_validates_head_divisibility() -> None:
    with pytest.raises(ValueError):
        ModelConfig(hidden_dim=64, n_heads=6, n_kv_heads=3)
    with pytest.raises(ValueError):
        ModelConfig(hidden_dim=64, n_heads=4, n_kv_heads=3)


def test_forward_shape_and_finiteness() -> None:
    model = build_stack(SMALL, backend="ako")
    out = model(torch.randn(2, 8, SMALL.hidden_dim))
    assert out.shape == (2, 8, SMALL.hidden_dim)
    assert torch.isfinite(out).all()


def test_backends_agree() -> None:
    """The dispatched path must be numerically identical to the eager path."""
    x = torch.randn(2, 8, SMALL.hidden_dim)
    ako_out = build_stack(SMALL, backend="ako", seed=7)(x)
    torch_out = build_stack(SMALL, backend="torch", seed=7)(x)
    assert_close(ako_out, torch_out, atol=1e-5, rtol=1e-5)


def test_attention_is_causal() -> None:
    """Changing a later token must not change an earlier token's output."""
    model = build_stack(SMALL, backend="ako", seed=3)
    x = torch.randn(1, 8, SMALL.hidden_dim)
    perturbed = x.clone()
    perturbed[:, 5:] += 1.0

    base = model(x)
    changed = model(perturbed)

    assert_close(base[:, :5], changed[:, :5], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(base[:, 5:], changed[:, 5:])


def test_kv_cache_decode_matches_prefill() -> None:
    """Token-by-token decoding must reproduce the full-sequence forward pass."""
    model = build_stack(SMALL, backend="ako", seed=11)
    x = torch.randn(1, 6, SMALL.hidden_dim)

    prefill = model(x)

    cache = model.new_kv_cache()
    outputs = [model(x[:, i : i + 1], kv_cache=cache) for i in range(x.shape[1])]
    decoded = torch.cat(outputs, dim=1)

    assert cache.length == 6
    assert_close(decoded, prefill, atol=1e-4, rtol=1e-4)


def test_kv_cache_grows_per_layer() -> None:
    cache = KVCache(n_layers=2)
    assert cache.length == 0
    k = torch.randn(1, 1, 2, 8)
    cache.update(0, k, k)
    cache.update(0, k, k)
    assert cache.length == 2
    assert cache.entries[1] is None
