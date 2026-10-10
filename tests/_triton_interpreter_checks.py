"""Run the Triton kernel bodies on CPU and compare them to the references.

Executed as a subprocess by ``tests/test_triton_interpreter.py`` because
``TRITON_INTERPRET`` has to be set before Triton is imported. Results are
printed as one JSON object ``{check_name: value}``: an absolute error for the
elementwise kernels, an error/budget ratio for the ``mm_*`` GEMM checks (the M4
gate is "below its own dtype budget", not "below a number typed here"), and a
case count for the two ``gate_*`` entries.

This validates the kernel *logic* — indexing, masking, pair offsets, dtype
rounding — on a machine without a GPU. It says nothing about performance, and
the autotuner is bypassed (it needs a real driver to time configs).
"""

from __future__ import annotations

import json
import os
import sys

os.environ["TRITON_INTERPRET"] = "1"

import torch  # noqa: E402
import triton  # noqa: E402

from ako.ops.rmsnorm_ref import rmsnorm_ref, rmsnorm_residual_ref  # noqa: E402
from ako.ops.rmsnorm_triton import (  # noqa: E402
    _rmsnorm_kernel_impl,
    _rmsnorm_residual_kernel_impl,
)
from ako.ops.rope_ref import build_rope_tables, rope_ref  # noqa: E402
from ako.ops.rope_triton import _rope_kernel_impl  # noqa: E402
from ako.ops.swiglu_ref import swiglu_ref  # noqa: E402
from ako.ops.swiglu_triton import _swiglu_kernel_impl  # noqa: E402
from ako.search.blocks import BLOCKS  # noqa: E402
from ako.search.kernels.mm_fused import launch  # noqa: E402
from ako.search.space import composition_name, partitions  # noqa: E402
from ako.search.spec import Problem, caps  # noqa: E402
from ako.validation.numerical import error_budget, max_abs_error  # noqa: E402


def _run_rmsnorm(n_rows: int, hidden_dim: int) -> float:
    x = torch.randn(n_rows, hidden_dim)
    weight = torch.randn(hidden_dim)
    y = torch.empty_like(x)
    _rmsnorm_kernel_impl[(n_rows,)](
        x,
        weight,
        y,
        n_rows,
        hidden_dim,
        1e-6,
        x.stride(0),
        y.stride(0),
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
    )
    return max_abs_error(y, rmsnorm_ref(x, weight))


def _run_rmsnorm_residual(n_rows: int, hidden_dim: int) -> float:
    x = torch.randn(n_rows, hidden_dim)
    residual = torch.randn(n_rows, hidden_dim)
    weight = torch.randn(hidden_dim)
    y = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    _rmsnorm_residual_kernel_impl[(n_rows,)](
        x,
        residual,
        weight,
        y,
        residual_out,
        n_rows,
        hidden_dim,
        1e-6,
        x.stride(0),
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
    )
    ref_y, ref_residual = rmsnorm_residual_ref(x, residual, weight)
    return max(max_abs_error(y, ref_y), max_abs_error(residual_out, ref_residual))


def _run_rope(style: str, shape: tuple[int, int, int, int]) -> float:
    batch, seq_len, n_heads, head_dim = shape
    x = torch.randn(batch, seq_len, n_heads, head_dim)
    cos, sin = build_rope_tables(seq_len, head_dim)
    x_2d = x.contiguous().view(-1, head_dim)
    y = torch.empty_like(x_2d)
    _rope_kernel_impl[(x_2d.shape[0],)](
        x_2d,
        cos,
        sin,
        y,
        x_2d.shape[0],
        seq_len,
        n_heads,
        head_dim,
        head_dim // 2,
        x_2d.stride(0),
        y.stride(0),
        cos.stride(0),
        INTERLEAVED=(style == "interleaved"),
        BLOCK_SIZE=triton.next_power_of_2(head_dim // 2),
    )
    return max_abs_error(y.view(*shape), rope_ref(x, cos, sin, style))


def _run_swiglu(n_rows: int, hidden_dim: int) -> float:
    gate = torch.randn(n_rows, hidden_dim)
    up = torch.randn(n_rows, hidden_dim)
    y = torch.empty_like(gate)
    _swiglu_kernel_impl[(n_rows,)](
        gate,
        up,
        y,
        n_rows,
        hidden_dim,
        gate.stride(0),
        BLOCK_SIZE=triton.next_power_of_2(hidden_dim),
    )
    return max_abs_error(y, swiglu_ref(gate, up))


#: M4 case matrix: 4 shapes x 2 dtypes x 4 compositions x SPLIT_K in {1, 4}.
#: (65, 97, 129) has M % BLOCK_M, N % BLOCK_N and K % BLOCK_K all nonzero -- every
#: dimension of the M7 grid is a multiple of 8, so without it no masking bug in
#: the template would ever surface anywhere in the project (ROADMAP.md M4).
MM_SHAPES = ((32, 64, 128), (65, 97, 129), (1, 128, 128), (129, 33, 130))
MM_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
MM_SPLIT_K = (1, 4)
#: One point of the section 3.2 grid; the whole grid is M6's sweep, not the
#: interpreter's job (it is 100-1000x slower than the GPU).
MM_PARAMS = {
    "BLOCK_M": 32,
    "BLOCK_N": 32,
    "BLOCK_K": 32,
    "GROUP_M": 8,
    "num_warps": 4,
    "num_stages": 2,
    "ACC_FP16": False,
}
MM_ACT = "gelu_tanh"


def _mm_cases():
    """(shape, dtype name, segments, SPLIT_K) for the 64 cases of the milestone."""
    for shape in MM_SHAPES:
        for dtype in MM_DTYPES:
            for segments in partitions(("matmul", "bias", "act")):
                for split_k in MM_SPLIT_K:
                    yield shape, dtype, segments, split_k


def _run_mm(shape, dtype, segments, split_k) -> float:
    """Fused output vs the eager reference, as a fraction of its error budget.

    The reference is the same for all four compositions -- only the fusion
    boundary moves -- and is computed in fp32 then downcast, so an unfused
    candidate is not credited for the extra rounding it does not do.
    """
    m, n, k = shape
    torch_dtype = MM_DTYPES[dtype]
    a = torch.randn(m, k, dtype=torch_dtype)
    b = torch.randn(k, n, dtype=torch_dtype)
    bias = torch.randn(n, dtype=torch_dtype)

    reference = torch.nn.functional.gelu(
        torch.addmm(bias.float(), a.float(), b.float()), approximate="tanh"
    ).to(torch_dtype)
    out = launch(a, b, bias, MM_PARAMS | {"SPLIT_K": split_k}, segments, MM_ACT)
    return max_abs_error(out, reference) / error_budget(reference)


def _mm_gate() -> dict[str, int]:
    """Free-filter verdicts over the same 64 cases: no kernel runs, nothing compiles.

    SPLIT_K > 1 under a GEMM that carries an epilogue is a legality rule, not a
    bug to fix: atomic_add makes partial tiles visible, so a bias would be added
    SPLIT_K times and a non-linear activation applied per partial.
    """
    device = caps()
    counts: dict[str, int] = {}
    for shape, dtype, segments, split_k in _mm_cases():
        m, n, k = shape
        prob = Problem(m=m, n=n, k=k, dtype=dtype, act=MM_ACT)
        fused = len(segments[0]) > 1  # the GEMM segment carries an epilogue
        block = BLOCKS["triton_mm_fused" if fused else "triton_mm"]
        reason = block.feasible(MM_PARAMS | {"SPLIT_K": split_k}, prob, device)
        counts[reason or "feasible"] = counts.get(reason or "feasible", 0) + 1
    return counts


def main() -> int:
    torch.manual_seed(0)
    checks = {
        "rmsnorm": _run_rmsnorm(4, 64),
        # Not a power of two: exercises the masked tail of the block.
        "rmsnorm_ragged": _run_rmsnorm(3, 100),
        # Wider than the old autotune ceiling (8192), which used to truncate rows.
        "rmsnorm_wide": _run_rmsnorm(2, 8448),
        "rmsnorm_residual": _run_rmsnorm_residual(4, 64),
        "rope_neox": _run_rope("neox", (2, 3, 2, 32)),
        "rope_interleaved": _run_rope("interleaved", (2, 3, 2, 32)),
        # Several heads per position: checks the row -> position mapping.
        "rope_many_heads": _run_rope("neox", (1, 4, 8, 16)),
        "swiglu": _run_swiglu(4, 64),
        # Non power of two, and wider than the old autotune ceiling.
        "swiglu_wide_ragged": _run_swiglu(2, 8448),
    }
    checks.update({f"gate_{reason}": n for reason, n in _mm_gate().items()})
    for shape, dtype, segments, split_k in _mm_cases():
        # bf16 is enumerated by the gate above but cannot be *run* here: the
        # Triton 3.7.1 interpreter stores bf16 as uint16 and create_dot feeds it
        # straight to np.matmul, so tl.dot returns an integer product of the raw
        # bits (a 16x32 randn pair comes back at 1e10). Not our masking to fix;
        # bf16 numerics are covered on the GPU at M5.
        if dtype != "fp16" or (split_k > 1 and len(segments[0]) > 1):
            continue
        name = f"mm_{composition_name(segments)}_{'x'.join(map(str, shape))}_sk{split_k}"
        checks[name] = _run_mm(shape, dtype, segments, split_k)
    json.dump(checks, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
