"""The key: what to compute (``Problem``), how (``Candidate``), on what (``DeviceCaps``).

ARCHITECTURE.md section 3.1. These three objects form the primary key of every
trial row: dropping the hardware part would make "tune once, ship everywhere"
expressible, which invariant 6 forbids.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Literal

import torch

from ako.dispatcher.hardware import get_device_info

DType = Literal["fp16", "bf16", "fp32"]
Act = Literal["none", "relu", "gelu_tanh", "silu"]

_ITEMSIZE = {"fp16": 2, "bf16": 2, "fp32": 4}


def _digest(obj: object, n: int) -> str:
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:n]


@dataclass(frozen=True, slots=True)
class Problem:
    """WHAT to compute. Deliberately not a graph: no producers, no consumers."""

    op: str = "matmul_bias_act"
    m: int = 1
    n: int = 4096
    k: int = 4096
    dtype: DType = "fp16"
    act: Act = "gelu_tanh"
    bias: bool = True
    layout: str = "nn"  # "nn" | "nt"

    def chain(self) -> tuple[str, ...]:
        """("matmul", "bias", "act"), minus the segments this problem disables."""
        skip = set()
        if not self.bias:
            skip.add("bias")
        if self.act == "none":
            skip.add("act")
        return tuple(s for s in self.op.split("_") if s not in skip)

    def as_dict(self) -> dict:
        return asdict(self)

    def key(self) -> str:
        return _digest(self.as_dict(), 16)

    def flops(self) -> int:
        return 2 * self.m * self.n * self.k

    def itemsize(self) -> int:
        return _ITEMSIZE[self.dtype]


@dataclass(frozen=True, slots=True)
class Candidate:
    """HOW to compute it. One compilable point in the search space."""

    composition: str  # "mm+bias+act" | "mm+bias|act" | "mm|bias+act" | "mm|bias|act"
    blocks: tuple[str, ...]  # one block name per segment
    params: tuple[tuple[tuple[str, object], ...], ...]  # per-block sorted kv pairs

    def hash(self) -> str:
        return _digest([self.composition, list(self.blocks), self.params_json()], 16)

    def params_json(self) -> str:
        """Canonical json, sorted keys, one object per block."""
        return json.dumps(
            [dict(sorted(p)) for p in self.params], sort_keys=True, separators=(",", ":")
        )


@dataclass(frozen=True, slots=True)
class DeviceCaps:
    """Built from torch.cuda.get_device_properties, wrapping ``get_device_info``."""

    name: str  # "NVIDIA GeForce RTX 3070"
    cc: tuple[int, int]  # (8, 6)
    sm_count: int  # 46
    smem_per_block: int  # 101376 -> shared_memory_per_block_optin
    regs_per_sm: int
    l2_bytes: int
    total_mem_gb: float
    torch_version: str
    triton_version: str | None
    driver: str | None = None  # section 3.5 hashes it into toolchain_key

    @property
    def key(self) -> str:
        return f"{self.name.replace(' ', '_')}_cc{self.cc[0]}{self.cc[1]}"

    @property
    def toolchain_key(self) -> str:
        """sha1(torch|triton|driver)[:12]. Invariant 10: rows from another
        toolchain stay readable but are excluded from pruning and ordering."""
        return _digest([self.torch_version, self.triton_version, self.driver], 12)


def caps(device: torch.device | str | None = None) -> DeviceCaps:
    """DeviceCaps for ``device``. On a machine without CUDA the GPU-only counters
    are zero, which keeps the CPU milestones runnable without faking a card."""
    info = get_device_info(device)
    try:
        import triton  # noqa: PLC0415 - optional, absent on CPU-only installs

        triton_version = triton.__version__
    except Exception:
        triton_version = None

    # No fallback to the 48 KB static limit: silently halving smem_per_block
    # would silently shrink the feasible grid (~130 points instead of 441) with
    # no trace in the DB. An unknown limit stops the sweep instead.
    if info.device_type == "cuda" and not info.smem_per_block:
        raise RuntimeError(
            f"{info.device_name}: neither torch nor libcudart report a "
            "per-block shared-memory limit, so feasibility cannot be decided"
        )

    return DeviceCaps(
        name=info.device_name,
        cc=info.cuda_capability or (0, 0),
        sm_count=info.sm_count or 0,
        smem_per_block=info.smem_per_block or 0,
        regs_per_sm=info.regs_per_sm or 0,
        l2_bytes=info.l2_bytes or 0,
        total_mem_gb=info.total_memory_gb or 0.0,
        torch_version=torch.__version__,
        triton_version=triton_version,
        driver=info.driver_version,
    )
