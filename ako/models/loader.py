"""Load Hugging Face LLaMA-family checkpoints into an AKO decoder stack.

The stack in :mod:`ako.models.decoder` deliberately mirrors the LLaMA layout, so
a checkpoint only needs a name remap — no conversion of the tensors themselves.
That makes it possible to run real weights through the AKO kernels and to check
the two implementations agree.

Nothing here requires the network except :func:`load_hf_model`, which is a thin
wrapper over ``transformers``. The remap itself works on any ``state_dict``.
"""

from __future__ import annotations

from typing import Any

import torch

from ako.models.decoder import Backend, DecoderStack, ModelConfig

#: Per-layer weight names, ``ako name -> hugging face name``.
_LAYER_MAP = {
    "attention.q_proj.weight": "self_attn.q_proj.weight",
    "attention.k_proj.weight": "self_attn.k_proj.weight",
    "attention.v_proj.weight": "self_attn.v_proj.weight",
    "attention.o_proj.weight": "self_attn.o_proj.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
    "input_norm_weight": "input_layernorm.weight",
    "post_attention_norm_weight": "post_attention_layernorm.weight",
}


class UnsupportedCheckpoint(ValueError):
    """The checkpoint uses a feature the AKO stack does not implement."""


def _attr(config: Any, name: str, default: Any = None) -> Any:
    return getattr(config, name, default)


def _rope_parameters(config: Any) -> dict[str, Any]:
    """Rotary settings, across transformers layouts.

    transformers 4.x exposes ``rope_theta`` plus an optional ``rope_scaling``
    dict; 5.x groups both under ``rope_parameters``.
    """
    params = _attr(config, "rope_parameters") or _attr(config, "rope_scaling") or {}
    if not isinstance(params, dict):
        return {}
    merged = dict(params)
    merged.setdefault("rope_theta", _attr(config, "rope_theta", 10000.0))
    merged.setdefault("rope_type", merged.get("type", "default"))
    return merged


def model_config_from_hf(hf_config: Any) -> ModelConfig:
    """Translate a ``LlamaConfig``-like object into a :class:`ModelConfig`.

    Raises:
        UnsupportedCheckpoint: if the checkpoint needs behaviour the AKO stack
            does not implement (attention bias, RoPE scaling, a head dimension
            that is not ``hidden_size // num_attention_heads``).
    """
    hidden_dim = _attr(hf_config, "hidden_size")
    n_heads = _attr(hf_config, "num_attention_heads")
    if hidden_dim is None or n_heads is None:
        raise UnsupportedCheckpoint("Config is missing hidden_size / num_attention_heads")

    if _attr(hf_config, "attention_bias", False):
        raise UnsupportedCheckpoint("Attention projections with bias are not supported")

    rope = _rope_parameters(hf_config)
    if rope.get("rope_type", "default") != "default":
        raise UnsupportedCheckpoint(f"RoPE scaling {rope!r} is not implemented")

    head_dim = _attr(hf_config, "head_dim") or hidden_dim // n_heads
    if head_dim != hidden_dim // n_heads:
        raise UnsupportedCheckpoint(
            f"head_dim {head_dim} != hidden_size // num_attention_heads ({hidden_dim // n_heads})"
        )

    return ModelConfig(
        hidden_dim=hidden_dim,
        n_layers=_attr(hf_config, "num_hidden_layers"),
        n_heads=n_heads,
        n_kv_heads=_attr(hf_config, "num_key_value_heads") or n_heads,
        intermediate_dim=_attr(hf_config, "intermediate_size"),
        max_seq_len=_attr(hf_config, "max_position_embeddings", 2048),
        rope_base=float(rope.get("rope_theta", 10000.0)),
        rope_style="neox",
        eps=float(_attr(hf_config, "rms_norm_eps", 1e-5)),
    )


def _strip_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Accept both ``LlamaModel`` and ``LlamaForCausalLM`` key layouts."""
    if any(key.startswith("model.") for key in state_dict):
        return {
            key[len("model.") :]: value
            for key, value in state_dict.items()
            if key.startswith("model.")
        }
    return dict(state_dict)


def remap_hf_state_dict(
    state_dict: dict[str, torch.Tensor], config: ModelConfig
) -> dict[str, torch.Tensor]:
    """Rename Hugging Face weights to AKO ``DecoderStack`` parameter names.

    Embedding and language-model-head weights are ignored: the stack operates on
    hidden states, not tokens.

    Raises:
        UnsupportedCheckpoint: if a weight the stack needs is missing.
    """
    source = _strip_prefix(state_dict)
    remapped: dict[str, torch.Tensor] = {}
    missing: list[str] = []

    for layer_idx in range(config.n_layers):
        for ako_suffix, hf_suffix in _LAYER_MAP.items():
            hf_key = f"layers.{layer_idx}.{hf_suffix}"
            if hf_key not in source:
                missing.append(hf_key)
                continue
            remapped[f"layers.{layer_idx}.{ako_suffix}"] = source[hf_key]

    if "norm.weight" in source:
        remapped["final_norm_weight"] = source["norm.weight"]
    else:
        missing.append("norm.weight")

    if missing:
        raise UnsupportedCheckpoint(f"Checkpoint is missing {len(missing)} weights: {missing[:5]}")
    return remapped


def load_hf_weights(stack: DecoderStack, state_dict: dict[str, torch.Tensor]) -> DecoderStack:
    """Load a Hugging Face ``state_dict`` into ``stack`` in place."""
    remapped = remap_hf_state_dict(state_dict, stack.config)
    target_dtype = stack.final_norm_weight.dtype
    target_device = stack.final_norm_weight.device
    stack.load_state_dict(
        {k: v.to(device=target_device, dtype=target_dtype) for k, v in remapped.items()},
        strict=True,
    )
    return stack


def load_hf_stacks(
    model_id: str,
    backends: tuple[Backend, ...] = ("ako",),
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
    revision: str | None = None,
) -> dict[Backend, DecoderStack]:
    """Download a LLaMA-family checkpoint once and load it into several stacks.

    Loading the same weights into an ``"ako"`` and a ``"torch"`` stack is how the
    two paths get compared on identical parameters without downloading twice.

    Requires ``transformers`` (``pip install "ako[hf]"``) and network access to
    the Hugging Face hub. Offline, build the stack yourself and call
    :func:`load_hf_weights` with a ``state_dict`` you already have.

    Args:
        model_id: Hub id, e.g. ``"TinyLlama/TinyLlama-1.1B-Chat-v1.0"``.
        backends: Which stacks to build.
        device: Target device.
        dtype: Target dtype.
        revision: Optional git revision to pin.

    Returns:
        One stack per requested backend, in eval mode.
    """
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Loading from the hub requires transformers. Install with: pip install 'ako[hf]'"
        ) from exc

    config = model_config_from_hf(AutoConfig.from_pretrained(model_id, revision=revision))
    hf_model = AutoModel.from_pretrained(model_id, revision=revision, dtype=dtype)
    state_dict = hf_model.state_dict()

    stacks: dict[Backend, DecoderStack] = {}
    for backend in backends:
        stack = DecoderStack(config, backend=backend).to(device=device, dtype=dtype).eval()
        stacks[backend] = load_hf_weights(stack, state_dict)

    del hf_model
    return stacks


def load_hf_model(
    model_id: str,
    *,
    backend: Backend = "ako",
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float16,
    revision: str | None = None,
) -> DecoderStack:
    """Download a LLaMA-family checkpoint and load it into one AKO stack."""
    stacks = load_hf_stacks(
        model_id, (backend,), device=device, dtype=dtype, revision=revision
    )
    return stacks[backend]
