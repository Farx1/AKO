"""One Triton GEMM template; the epilogue chain is a pair of constexpr integers.

ARCHITECTURE.md 3.2 (the variant parameters, which are exactly the constexpr
tuple Triton specializes on) and 3.4 (the epilogue vocabulary), ROADMAP.md M4.

There is **no code generation layer**. A composition is not a generated kernel:
it is the same `_mm_fused_impl` launched with different `EPI*` integers, and
Triton's specialization cache is the template engine. `apply_epi` is shared by
the GEMM epilogue and by the tail kernel. Materializing an intermediate still
adds rounding, so fused and unfused results need not coincide. The optional
FP32_BOUNDARIES bitmask exposes intermediate precision for research.

Unlike :mod:`ako.ops.rmsnorm_triton` there is no autotune decorator: `num_warps`
and `num_stages` come from the candidate being measured, not from a search
Triton runs behind our back. The kernels keep the `_impl` suffix of that module
anyway, because that is the name the CPU harness reaches under
``TRITON_INTERPRET=1`` (tests/_triton_interpreter_checks.py).
"""

from __future__ import annotations

import torch

from ako.search.constraints import Params

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised on Triton-less installs
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    TRITON_AVAILABLE = False

#: ARCHITECTURE.md 3.4. One dict, shared by the GEMM epilogue, the tail kernel
#: and the fp32 oracle -- so the reference and the kernel cannot drift apart.
EPI_CODES = {"none": 0, "bias": 1, "relu": 2, "gelu_tanh": 3, "silu": 4}

#: A segment carries at most bias and act; a third entry means the op chain grew
#: and the kernel signature has to grow with it (deliberately not open-ended).
MAX_EPI = 2


def epi_codes(segment: tuple[str, ...], act: str) -> tuple[int, int]:
    """Segment -> the constexpr integers that specialize the kernel for it."""
    codes = tuple(
        EPI_CODES["bias"] if item == "bias" else EPI_CODES[act]
        for item in segment
        if item != "matmul"
    )
    if len(codes) > MAX_EPI:
        raise ValueError(f"segment {segment} needs {len(codes)} epilogues, kernel takes {MAX_EPI}")
    return (codes + (0, 0))[:MAX_EPI]  # type: ignore[return-value]


if TRITON_AVAILABLE:

    @triton.jit
    def apply_epi(acc, bias_ptr, offs_n, mask_n, CODE: tl.constexpr):
        """One epilogue step, selected at specialization time by ``CODE``."""
        if CODE == 1:
            acc = acc + tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)[None, :]
        elif CODE == 2:
            acc = tl.maximum(acc, 0.0)
        elif CODE == 3:
            # tanh(x) == 2 * sigmoid(2x) - 1. Deliberate: tl.math.tanh does not
            # exist in Triton 3.7.1 and the libdevice tanh it moved to fails
            # under TRITON_INTERPRET=1 ("cannot convert None of type NoneType to
            # tensor"). gelu_tanh is the default act and the act of the whole M7
            # grid, so a libdevice epilogue would make the CPU harness -- the
            # thing that makes days without a GPU productive -- unable to run the
            # only activation this project measures (ARCHITECTURE.md 3.4).
            u = 0.7978845608 * (acc + 0.044715 * acc * acc * acc)
            acc = 0.5 * acc * (2.0 * tl.sigmoid(2.0 * u))  # 1 + tanh(u)
        elif CODE == 4:
            acc = acc * tl.sigmoid(acc)
        return acc

    @triton.jit
    def _mm_fused_impl(
        a_ptr,
        b_ptr,
        c_ptr,
        bias_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_pk,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EPI0: tl.constexpr,
        EPI1: tl.constexpr,
        ACC_FP16: tl.constexpr,
        NUM_K: tl.constexpr,
        EVEN_K: tl.constexpr,
        EVEN_MN: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid_k = tl.program_id(1)

        # L2 swizzle: walk GROUP_M rows of tiles before moving on, so the B tiles
        # a group reads stay resident.
        grid_m = tl.cdiv(M, BLOCK_M)
        grid_n = tl.cdiv(N, BLOCK_N)
        width = GROUP_M * grid_n
        group_id = pid // width
        group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
        pid_m = group_id * GROUP_M + (pid % group_size)
        pid_n = (pid % width) // group_size

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_m = offs_m < M
        mask_n = offs_n < N

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        if ACC_FP16:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)
        else:
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        # NUM_K is computed on the host so the bound is a Python int. The
        # interpreter cannot turn ``tl.cdiv(...)`` into ``range()`` (it is a
        # tensor), and pid_k is a runtime id so ``K - pid_k * BLOCK_K`` cannot
        # be constexpr either. Host-side ``cdiv(K, BLOCK_K * SPLIT_K)`` is the
        # same count the GPU would walk.
        # A predicated inner loop cannot use the wide vector loads. The shapes
        # this project cares about (multiples of the tile) take the unmasked
        # path; ragged M/N/K keep the mask, which the interpreter checks.
        for step in range(NUM_K):
            k_off = step * BLOCK_K * SPLIT_K
            if EVEN_K and EVEN_MN:
                a = tl.load(a_ptrs + k_off * stride_ak)
                b = tl.load(b_ptrs + k_off * stride_bk)
            elif EVEN_K:
                a = tl.load(
                    a_ptrs + k_off * stride_ak, mask=mask_m[:, None], other=0.0
                )
                b = tl.load(
                    b_ptrs + k_off * stride_bk, mask=mask_n[None, :], other=0.0
                )
            else:
                mask_k = offs_k + k_off < K
                a = tl.load(
                    a_ptrs + k_off * stride_ak, mask=mask_m[:, None] & mask_k[None, :], other=0.0
                )
                b = tl.load(
                    b_ptrs + k_off * stride_bk, mask=mask_k[:, None] & mask_n[None, :], other=0.0
                )
            if ACC_FP16:
                acc = tl.dot(a, b, acc, out_dtype=tl.float16)
            else:
                acc = tl.dot(a, b, acc)

        acc = acc.to(tl.float32)
        # Split-K stores the raw partial. The epilogue runs once, after the
        # reduction; applying it here would silu (or gelu) each partial and
        # then add those, which is a different function.
        if SPLIT_K == 1:
            acc = apply_epi(acc, bias_ptr, offs_n, mask_n, EPI0)
            acc = apply_epi(acc, bias_ptr, offs_n, mask_n, EPI1)
            c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
            if EVEN_MN:
                tl.store(c_ptrs, acc)
            else:
                tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])
        else:
            p_ptrs = (
                c_ptr
                + pid_k * stride_pk
                + offs_m[:, None] * stride_cm
                + offs_n[None, :] * stride_cn
            )
            if EVEN_MN:
                tl.store(p_ptrs, acc)
            else:
                tl.store(p_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _splitk_reduce_impl(
        partials_ptr,
        out_ptr,
        bias_ptr,
        M,
        N,
        stride_pk,
        stride_pm,
        stride_pn,
        stride_om,
        stride_on,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SPLIT_K: tl.constexpr,
        EPI0: tl.constexpr,
        EPI1: tl.constexpr,
        EVEN_MN: tl.constexpr,
    ):
        """Sum split-K partials, then the epilogue, once."""
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_m = offs_m < M
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for step in range(SPLIT_K):
            ptrs = (
                partials_ptr
                + step * stride_pk
                + offs_m[:, None] * stride_pm
                + offs_n[None, :] * stride_pn
            )
            if EVEN_MN:
                acc += tl.load(ptrs)
            else:
                acc += tl.load(ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        acc = apply_epi(acc, bias_ptr, offs_n, mask_n, EPI0)
        acc = apply_epi(acc, bias_ptr, offs_n, mask_n, EPI1)
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        if EVEN_MN:
            tl.store(out_ptrs, acc)
        else:
            tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.jit
    def _epi_kernel_impl(
        x_ptr,
        y_ptr,
        bias_ptr,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        EPI0: tl.constexpr,
        EPI1: tl.constexpr,
    ):
        """The tail kernel of an unfused composition: same `apply_epi`, no dot."""
        offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        mask = (offs_m < M)[:, None] & mask_n[None, :]

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask, other=0.0
        ).to(tl.float32)
        x = apply_epi(x, bias_ptr, offs_n, mask_n, EPI0)
        x = apply_epi(x, bias_ptr, offs_n, mask_n, EPI1)
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, x, mask=mask)


def launch(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None,
    params: Params,
    segments: tuple[tuple[str, ...], ...] = (("matmul", "bias", "act"),),
    act: str = "gelu_tanh",
) -> torch.Tensor:
    """Run one candidate: the GEMM segment, then one tail kernel per remaining one.

    ``segments`` is the composition as :func:`ako.search.space.partitions` yields
    it, so the caller never spells epilogue codes by hand.
    """
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed. Install with: pip install triton")
    if not segments or "matmul" not in segments[0]:
        raise ValueError(f"first segment must carry the matmul, got {segments}")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"shape mismatch: a {tuple(a.shape)} @ b {tuple(b.shape)}")

    boundary_mask = params.get("FP32_BOUNDARIES", 0)
    if type(boundary_mask) is not int or not 0 <= boundary_mask < (1 << (len(segments) - 1)):
        raise ValueError("FP32_BOUNDARIES must be an integer bitmask over materialized boundaries")

    m, k = a.shape
    n = b.shape[1]
    epi0, epi1 = epi_codes(segments[0], act)
    split_k = int(params.get("SPLIT_K", 1))
    # SPLIT_K > 1 stores raw partials and reduces them before the epilogue, so a
    # fused epilogue is applied once to the sum. The old atomic_add path could
    # not do that: it would have activated each partial.

    bm, bn = int(params["BLOCK_M"]), int(params["BLOCK_N"])
    bk = int(params["BLOCK_K"])
    warps, stages = int(params.get("num_warps", 4)), int(params.get("num_stages", 3))
    even_mn = (m % bm == 0) and (n % bn == 0)
    bias_arg = bias if bias is not None else torch.empty(1, device=a.device, dtype=a.dtype)
    if split_k > 1:
        partials = torch.empty((split_k, m, n), dtype=torch.float32, device=a.device)
        # A following tail still wants the sum in fp32. A fused epilogue is the
        # final write, in the input dtype, unless a boundary asked for fp32.
        out = torch.empty(
            (m, n),
            dtype=torch.float32 if len(segments) > 1 or (boundary_mask & 1) else a.dtype,
            device=a.device,
        )
        c_tensor, stride_pk = partials, partials.stride(0)
        stride_cm, stride_cn = partials.stride(1), partials.stride(2)
    else:
        partials = None
        out = torch.empty(
            (m, n),
            dtype=torch.float32 if boundary_mask & 1 else a.dtype,
            device=a.device,
        )
        c_tensor, stride_pk = out, 0
        stride_cm, stride_cn = out.stride(0), out.stride(1)
    _mm_fused_impl[(triton.cdiv(m, bm) * triton.cdiv(n, bn), split_k)](
        a,
        b,
        c_tensor,
        bias_arg,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        stride_cm,
        stride_cn,
        stride_pk,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        GROUP_M=int(params.get("GROUP_M", 1)),
        SPLIT_K=split_k,
        EPI0=epi0,
        EPI1=epi1,
        ACC_FP16=bool(params.get("ACC_FP16", False)),
        NUM_K=triton.cdiv(k, bk * split_k),
        EVEN_K=(k % (bk * split_k)) == 0,
        EVEN_MN=even_mn,
        num_warps=warps,
        num_stages=stages,
    )
    if split_k > 1:
        _splitk_reduce_impl[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            partials,
            out,
            bias_arg,
            m,
            n,
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            out.stride(0),
            out.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            SPLIT_K=split_k,
            EPI0=epi0,
            EPI1=epi1,
            EVEN_MN=even_mn,
            num_warps=warps,
        )

    for index, segment in enumerate(segments[1:], start=1):
        e0, e1 = epi_codes(segment, act)
        # Bit i controls segment i's output, never the final result.
        dtype = torch.float32 if boundary_mask & (1 << index) else a.dtype
        y = torch.empty((m, n), dtype=dtype, device=a.device)
        # ponytail: the tail reuses the GEMM tile so no second tiling axis enters
        # the grid. Give it its own axis only if M6 shows the tail is hot.
        _epi_kernel_impl[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
            out,
            y,
            bias,
            m,
            n,
            out.stride(0),
            out.stride(1),
            y.stride(0),
            y.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            EPI0=e0,
            EPI1=e1,
            num_warps=warps,
        )
        out = y

    return out.to(a.dtype)  # no-op unless the split-K fp32 workspace is still it
