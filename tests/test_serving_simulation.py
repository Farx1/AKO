"""Tests for the serving simulation.

An experiment is only worth publishing if its inputs are reproducible and its
comparison is fair, so these pin down the trace generator, the batching, and the
fact that every policy runs identical work.
"""

from __future__ import annotations

import pytest
import torch

from ako.experiments.serving import (
    TRITON_OVERRIDE,
    PolicyResult,
    Request,
    RequestBatch,
    batch_trace,
    build_policy_stacks,
    crossover_batch,
    generate_trace,
    run_policy,
)
from ako.models.decoder import ModelConfig
from ako.validation.numerical import assert_close

TINY = ModelConfig(
    hidden_dim=64, n_layers=2, n_heads=4, n_kv_heads=2, intermediate_dim=128, max_seq_len=64
)


def test_trace_is_reproducible() -> None:
    assert generate_trace(32, seed=7) == generate_trace(32, seed=7)
    assert generate_trace(32, seed=7) != generate_trace(32, seed=8)


def test_trace_is_right_skewed_and_bounded() -> None:
    """Serving traces have many short prompts and a long tail — that skew is the
    reason shapes keep moving, so a uniform trace would beg the question."""
    trace = generate_trace(512, seed=0, median_prompt=256, max_prompt=2048)
    prompts = sorted(r.prompt_len for r in trace)

    assert min(prompts) >= 1
    assert max(prompts) <= 2048
    median = prompts[len(prompts) // 2]
    assert 150 < median < 400
    # Right-skewed: the mean sits above the median.
    assert sum(prompts) / len(prompts) > median
    assert len({r.prompt_len for r in trace}) > 50  # shapes genuinely vary


def test_batching_covers_every_request() -> None:
    trace = generate_trace(10, seed=1)
    batches = batch_trace(trace, max_batch=4)

    assert [b.size for b in batches] == [4, 4, 2]
    assert [r for b in batches for r in b.requests] == trace


def test_batch_runs_at_its_longest_prompt() -> None:
    batch = RequestBatch((Request(10, 3), Request(50, 8), Request(20, 1)))
    assert batch.prompt_len == 50
    assert batch.decode_steps == 8


def test_batching_rejects_bad_size() -> None:
    with pytest.raises(ValueError):
        batch_trace(generate_trace(4, seed=0), max_batch=0)


def test_policies_share_identical_weights() -> None:
    """The comparison is only fair if every policy runs the same model."""
    stacks = build_policy_stacks(
        TINY, policies=("eager", "adaptive"), device=torch.device("cpu"), dtype=torch.float32
    )
    x = torch.randn(1, 8, TINY.hidden_dim)
    with torch.inference_mode():
        assert_close(stacks["eager"](x), stacks["adaptive"](x), atol=1e-5, rtol=1e-5)


def test_triton_policy_forces_every_op() -> None:
    assert set(TRITON_OVERRIDE) == {"rmsnorm", "rmsnorm_residual", "rope"}
    assert all(name.endswith("_triton") for name in TRITON_OVERRIDE.values())


def test_run_policy_records_one_entry_per_batch() -> None:
    batches = batch_trace(generate_trace(6, seed=2, median_prompt=8, median_decode=2, max_prompt=16), max_batch=3)
    stacks = build_policy_stacks(
        TINY, policies=("eager",), device=torch.device("cpu"), dtype=torch.float32
    )

    result = run_policy(
        "eager", stacks["eager"], batches, config=TINY, device=torch.device("cpu"), dtype=torch.float32
    )

    assert result.failed is None
    assert len(result.batch_ms) == len(batches) == len(result.cumulative_ms)
    assert result.total_ms > 0
    # Cumulative time is monotonic by construction.
    assert result.cumulative_ms == sorted(result.cumulative_ms)


def test_run_policy_survives_a_broken_policy() -> None:
    """A policy that cannot run (missing kernel, OOM) must not lose the others."""
    batches = batch_trace(generate_trace(2, seed=0, median_prompt=8, median_decode=1, max_prompt=8), max_batch=2)
    stacks = build_policy_stacks(
        TINY,
        policies=("adaptive",),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    stacks["adaptive"].kernel_override = {"rmsnorm_residual": "does_not_exist"}

    result = run_policy(
        "adaptive", stacks["adaptive"], batches, config=TINY,
        device=torch.device("cpu"), dtype=torch.float32,
    )

    assert result.failed is not None
    assert "does_not_exist" in result.failed


def test_crossover_finds_the_amortization_point() -> None:
    """Adaptive pays its tuning cost first, then has to earn it back."""
    adaptive = PolicyResult("adaptive", cumulative_ms=[10.0, 18.0, 24.0, 29.0])
    fixed = PolicyResult("fixed", cumulative_ms=[5.0, 15.0, 25.0, 35.0])

    assert crossover_batch(adaptive, fixed) == 2

    never = PolicyResult("fixed", cumulative_ms=[1.0, 2.0, 3.0, 4.0])
    assert crossover_batch(adaptive, never) is None
