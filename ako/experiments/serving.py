"""Simulate an inference server and compare kernel-selection policies.

The shape sweeps answer "is this kernel faster at shape S". They do not answer
the objection that matters: *a real service runs one model, so tune once and be
done*. That objection only holds if the shapes are stable — and in serving they
are not. Prompt lengths vary per request, dynamic batching varies the batch
dimension, and prefill and decode are two different regimes on the same weights.

This module replays a request trace through the same stack under several
policies, so the comparison is on identical work:

- ``eager``    — unfused PyTorch, the naive reference
- ``compile``  — ``torch.compile`` over the same stack
- ``triton``   — always the AKO Triton kernel, never selected
- ``adaptive`` — AKO selection, benchmarking on first sight of a shape

It also measures what the paper (§6) asks for and nothing has measured yet: the
cost of autotuning, and how many requests it takes to pay for itself.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import torch

from ako.models.decoder import DecoderStack, KernelOverride, ModelConfig, build_stack

#: Kernels forced by the ``triton`` policy — the "I wrote a kernel, I use it
#: everywhere" strategy that adaptive selection is meant to beat.
TRITON_OVERRIDE: KernelOverride = {
    "rmsnorm": "rmsnorm_triton",
    "rmsnorm_residual": "rmsnorm_residual_triton",
    "rope": "rope_triton",
}


@dataclass(frozen=True)
class Request:
    """One inference request: a prompt to ingest, then tokens to generate."""

    prompt_len: int
    decode_len: int


@dataclass
class RequestBatch:
    """Requests served together by one dynamic-batching step."""

    requests: tuple[Request, ...]

    @property
    def size(self) -> int:
        return len(self.requests)

    @property
    def prompt_len(self) -> int:
        """Padded prefill length — the batch runs at its longest prompt."""
        return max(r.prompt_len for r in self.requests)

    @property
    def decode_steps(self) -> int:
        return max(r.decode_len for r in self.requests)


def generate_trace(
    n_requests: int,
    *,
    seed: int = 0,
    median_prompt: int = 256,
    median_decode: int = 128,
    max_prompt: int = 2048,
    spread: float = 1.0,
) -> list[Request]:
    """Draw a request trace with heavy-tailed prompt and generation lengths.

    Serving traces are strongly right-skewed — many short prompts, a long tail of
    large ones — so lengths are lognormal around the requested medians rather
    than uniform. That skew is the point: it is what keeps shapes moving.
    """
    generator = torch.Generator().manual_seed(seed)

    def draw(median: int, cap: int) -> torch.Tensor:
        values = torch.exp(torch.randn(n_requests, generator=generator) * spread) * median
        return values.clamp(1, cap).round().to(torch.int64)

    prompts = draw(median_prompt, max_prompt)
    decodes = draw(median_decode, max_prompt)
    return [Request(int(p), int(d)) for p, d in zip(prompts, decodes)]


def batch_trace(trace: Iterable[Request], *, max_batch: int = 8) -> list[RequestBatch]:
    """Group consecutive requests into batches, as a scheduler would."""
    if max_batch < 1:
        raise ValueError(f"max_batch must be >= 1, got {max_batch}")
    requests = list(trace)
    return [
        RequestBatch(tuple(requests[i : i + max_batch]))
        for i in range(0, len(requests), max_batch)
    ]


@dataclass
class PolicyResult:
    """Timings for one policy over the whole trace."""

    policy: str
    prefill_ms: list[float] = field(default_factory=list)
    decode_ms: list[float] = field(default_factory=list)
    batch_ms: list[float] = field(default_factory=list)
    cumulative_ms: list[float] = field(default_factory=list)
    failed: str | None = None

    @property
    def total_ms(self) -> float:
        return sum(self.batch_ms)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "policy": self.policy,
            "total_ms": round(self.total_ms, 3),
            "batches": len(self.batch_ms),
            "prefill_ms_total": round(sum(self.prefill_ms), 3),
            "decode_ms_total": round(sum(self.decode_ms), 3),
            "cumulative_ms": [round(v, 3) for v in self.cumulative_ms],
        }
        if self.batch_ms:
            payload["ttft_p50_ms"] = round(_percentile(self.prefill_ms, 0.5), 4)
            payload["ttft_p95_ms"] = round(_percentile(self.prefill_ms, 0.95), 4)
            payload["batch_p50_ms"] = round(_percentile(self.batch_ms, 0.5), 4)
            payload["batch_p95_ms"] = round(_percentile(self.batch_ms, 0.95), 4)
        if self.failed:
            payload["failed"] = self.failed
        return payload


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * p), len(ordered) - 1)]


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def serve_batch(
    model: Callable[..., torch.Tensor],
    batch: RequestBatch,
    *,
    hidden_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    new_cache: Callable[[], Any],
) -> tuple[float, float]:
    """Run one batch through prefill then decode. Returns ``(prefill_ms, decode_ms)``."""
    cache = new_cache()
    prompt = torch.randn(batch.size, batch.prompt_len, hidden_dim, dtype=dtype, device=device)

    _sync(device)
    start = time.perf_counter()
    model(prompt, kv_cache=cache)
    _sync(device)
    prefill_ms = (time.perf_counter() - start) * 1000

    token = torch.randn(batch.size, 1, hidden_dim, dtype=dtype, device=device)
    _sync(device)
    start = time.perf_counter()
    for _ in range(batch.decode_steps):
        model(token, kv_cache=cache)
    _sync(device)
    decode_ms = (time.perf_counter() - start) * 1000

    return prefill_ms, decode_ms


def build_policy_stacks(
    config: ModelConfig,
    *,
    policies: Iterable[str],
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> dict[str, Any]:
    """Instantiate one stack per policy, all with identical weights."""
    stacks: dict[str, Any] = {}
    for policy in policies:
        if policy == "eager":
            stacks[policy] = build_stack(
                config, backend="torch", device=device, dtype=dtype, seed=seed
            )
        elif policy == "compile":
            base = build_stack(config, backend="torch", device=device, dtype=dtype, seed=seed)
            stacks[policy] = torch.compile(base, dynamic=True)
        elif policy == "triton":
            stacks[policy] = build_stack(
                config,
                backend="ako",
                device=device,
                dtype=dtype,
                seed=seed,
                kernel_override=TRITON_OVERRIDE,
            )
        elif policy == "adaptive":
            stacks[policy] = build_stack(
                config, backend="ako", device=device, dtype=dtype, seed=seed
            )
        else:
            raise ValueError(f"Unknown policy '{policy}'")
    return stacks


def run_policy(
    policy: str,
    model: Any,
    batches: list[RequestBatch],
    *,
    config: ModelConfig,
    device: torch.device,
    dtype: torch.dtype,
    warmup_batches: int = 0,
) -> PolicyResult:
    """Replay the trace under one policy.

    ``warmup_batches`` is deliberately 0 by default: the cost a policy pays to
    become fast — compilation, autotuning — is part of what is being compared.
    """
    result = PolicyResult(policy=policy)
    stack: DecoderStack = getattr(model, "_orig_mod", model)

    with torch.inference_mode():
        for _ in range(warmup_batches):
            if batches:
                serve_batch(
                    model,
                    batches[0],
                    hidden_dim=config.hidden_dim,
                    device=device,
                    dtype=dtype,
                    new_cache=stack.new_kv_cache,
                )

        cumulative = 0.0
        for batch in batches:
            try:
                prefill_ms, decode_ms = serve_batch(
                    model,
                    batch,
                    hidden_dim=config.hidden_dim,
                    device=device,
                    dtype=dtype,
                    new_cache=stack.new_kv_cache,
                )
            except Exception as exc:  # OOM on a long prompt must not lose the run
                result.failed = f"{type(exc).__name__}: {exc}"
                break
            result.prefill_ms.append(prefill_ms)
            result.decode_ms.append(decode_ms)
            result.batch_ms.append(prefill_ms + decode_ms)
            cumulative += prefill_ms + decode_ms
            result.cumulative_ms.append(cumulative)

    return result


def crossover_batch(adaptive: PolicyResult, fixed: PolicyResult) -> int | None:
    """First batch index where ``adaptive`` is cumulatively ahead of ``fixed``.

    Adaptive selection pays its benchmarking cost up front, on the first sight of
    each shape. This is the amortization point the paper reasons about in §6 —
    ``None`` means it never catches up over the trace.
    """
    for index, (a, f) in enumerate(zip(adaptive.cumulative_ms, fixed.cumulative_ms)):
        if a < f:
            return index
    return None
