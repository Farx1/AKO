"""Validate the AKO decoder stack against the Hugging Face LLaMA reference.

These tests never touch the network: ``transformers`` can build a LLaMA from a
config alone, with random weights. That is enough to prove the AKO stack uses
the same conventions as the reference implementation — RMSNorm placement, RoPE
pairing, grouped-query repetition, SwiGLU — which is exactly what a real
checkpoint depends on.
"""

from __future__ import annotations

import pytest
import torch

from ako.models.decoder import DecoderStack
from ako.models.loader import (
    UnsupportedCheckpoint,
    load_hf_weights,
    model_config_from_hf,
    remap_hf_state_dict,
)
from ako.validation.numerical import max_abs_error

transformers = pytest.importorskip("transformers", reason="transformers is not installed")

from transformers import LlamaConfig  # noqa: E402
from transformers.models.llama.modeling_llama import LlamaModel  # noqa: E402


def _hf_config(**overrides) -> LlamaConfig:
    params = dict(
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        vocab_size=32,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
    )
    params.update(overrides)
    return LlamaConfig(**params)


@pytest.fixture(scope="module")
def hf_model() -> LlamaModel:
    torch.manual_seed(0)
    model = LlamaModel(_hf_config()).eval().to(torch.float32)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.05)
    return model


def test_config_translation() -> None:
    config = model_config_from_hf(_hf_config(rope_theta=500000.0))
    assert (config.hidden_dim, config.n_layers, config.n_heads, config.n_kv_heads) == (64, 2, 4, 2)
    assert config.rope_base == 500000.0
    assert config.rope_style == "neox"


def test_rejects_unsupported_checkpoints() -> None:
    with pytest.raises(UnsupportedCheckpoint, match="bias"):
        model_config_from_hf(_hf_config(attention_bias=True))
    with pytest.raises(UnsupportedCheckpoint, match="scaling"):
        model_config_from_hf(_hf_config(rope_scaling={"rope_type": "linear", "factor": 2.0}))


def test_remap_accepts_causal_lm_prefix(hf_model: LlamaModel) -> None:
    config = model_config_from_hf(_hf_config())
    prefixed = {f"model.{k}": v for k, v in hf_model.state_dict().items()}
    prefixed["lm_head.weight"] = torch.zeros(32, 64)

    remapped = remap_hf_state_dict(prefixed, config)

    assert "final_norm_weight" in remapped
    assert "layers.1.attention.q_proj.weight" in remapped
    assert not any("lm_head" in key for key in remapped)


def test_remap_reports_missing_weights(hf_model: LlamaModel) -> None:
    config = model_config_from_hf(_hf_config())
    incomplete = {k: v for k, v in hf_model.state_dict().items() if "q_proj" not in k}
    with pytest.raises(UnsupportedCheckpoint, match="missing"):
        remap_hf_state_dict(incomplete, config)


@pytest.mark.parametrize("backend", ["ako", "torch"])
def test_matches_transformers_forward(hf_model: LlamaModel, backend: str) -> None:
    """The AKO stack must reproduce the reference hidden states."""
    config = model_config_from_hf(_hf_config())
    stack = DecoderStack(config, backend=backend).eval()
    load_hf_weights(stack, hf_model.state_dict())

    x = torch.randn(2, 6, config.hidden_dim)
    with torch.inference_mode():
        reference = hf_model(inputs_embeds=x).last_hidden_state
        out = stack(x)

    assert max_abs_error(reference, out) < 1e-5


def test_matches_transformers_decode(hf_model: LlamaModel) -> None:
    """Same check for cached decoding, which is where RoPE offsets matter."""
    config = model_config_from_hf(_hf_config())
    stack = DecoderStack(config, backend="ako").eval()
    load_hf_weights(stack, hf_model.state_dict())

    x = torch.randn(1, 5, config.hidden_dim)
    cache = stack.new_kv_cache()
    with torch.inference_mode():
        reference = hf_model(inputs_embeds=x).last_hidden_state
        decoded = torch.cat([stack(x[:, i : i + 1], kv_cache=cache) for i in range(x.shape[1])], 1)

    assert max_abs_error(reference, decoded) < 1e-4
