"""Registration of every AKO kernel into the global registry.

Called lazily by the public op wrappers so that importing :mod:`ako` stays cheap
and CPU-only machines never touch Triton.
"""

from __future__ import annotations

from ako.dispatcher.registry import KernelEntry, KernelRegistry, get_registry

_registered = False

# Priority orders the fallback chain: the highest-priority kernel is used when
# autotuning is off (or on CPU), so the PyTorch baseline stays the safe default
# and specialized kernels only win through measured benchmarks.
_PYTORCH_PRIORITY = 10
_BASELINE_PRIORITY = 7
_TRITON_PRIORITY = 5


def register_all(registry: KernelRegistry | None = None, *, force: bool = False) -> KernelRegistry:
    """Register reference and Triton kernels for every supported operation."""
    global _registered
    target = registry or get_registry()
    if registry is None and _registered and not force:
        return target

    from ako.ops.rmsnorm_ref import rmsnorm_ref, rmsnorm_residual_ref
    from ako.ops.rope_ref import rope_ref
    from ako.ops.swiglu_ref import swiglu_ref

    references = [
        ("rmsnorm", "rmsnorm_ref", rmsnorm_ref),
        ("rmsnorm_residual", "rmsnorm_residual_ref", rmsnorm_residual_ref),
        ("rope", "rope_ref", rope_ref),
        ("swiglu", "swiglu_ref", swiglu_ref),
    ]
    for op_name, kernel_name, fn in references:
        target.register(
            KernelEntry(
                op_name=op_name,
                kernel_name=kernel_name,
                fn=fn,
                backend="pytorch",
                priority=_PYTORCH_PRIORITY,
            )
        )

    # Harder PyTorch baselines. They carry their own backend names so the
    # numerical reference stays `rmsnorm_ref` (picked by backend == "pytorch"),
    # and a lower priority so the safe default is unchanged when autotuning is
    # off — they have to win a benchmark like any other candidate.
    from ako.ops.rmsnorm_torch import (
        is_compile_enabled,
        is_native_available,
        rmsnorm_compiled,
        rmsnorm_torch_native,
    )

    if is_native_available():
        target.register(
            KernelEntry(
                op_name="rmsnorm",
                kernel_name="rmsnorm_torch_native",
                fn=rmsnorm_torch_native,
                backend="torch_native",
                priority=_BASELINE_PRIORITY,
            )
        )
    if is_compile_enabled():
        from ako.ops.rmsnorm_torch import rmsnorm_residual_compiled
        from ako.ops.rope_torch import rope_compiled
        from ako.ops.swiglu_torch import swiglu_compiled

        compiled_kernels = [
            ("rmsnorm", "rmsnorm_compiled", rmsnorm_compiled),
            ("rmsnorm_residual", "rmsnorm_residual_compiled", rmsnorm_residual_compiled),
            ("rope", "rope_compiled", rope_compiled),
            ("swiglu", "swiglu_compiled", swiglu_compiled),
        ]
        for op_name, kernel_name, fn in compiled_kernels:
            target.register(
                KernelEntry(
                    op_name=op_name,
                    kernel_name=kernel_name,
                    fn=fn,
                    backend="torch_compile",
                    priority=_BASELINE_PRIORITY,
                )
            )

    from ako.ops.rmsnorm_triton import is_triton_available as rmsnorm_triton_available

    if rmsnorm_triton_available():
        from ako.ops.rmsnorm_triton import rmsnorm_residual_triton, rmsnorm_triton
        from ako.ops.rope_triton import rope_triton
        from ako.ops.swiglu_triton import swiglu_triton

        triton_kernels = [
            ("rmsnorm", "rmsnorm_triton", rmsnorm_triton),
            ("rmsnorm_residual", "rmsnorm_residual_triton", rmsnorm_residual_triton),
            ("rope", "rope_triton", rope_triton),
            ("swiglu", "swiglu_triton", swiglu_triton),
        ]
        for op_name, kernel_name, fn in triton_kernels:
            target.register(
                KernelEntry(
                    op_name=op_name,
                    kernel_name=kernel_name,
                    fn=fn,
                    backend="triton",
                    priority=_TRITON_PRIORITY,
                )
            )

    if registry is None:
        _registered = True
        # The process-wide selector memoizes resolved kernels; a re-registration
        # (e.g. force=True) must not keep handing out the previous ones.
        from ako.dispatcher.selector import get_selector

        get_selector().clear_memo()
    return target


def reset_registration() -> None:
    """Allow :func:`register_all` to run again (tests)."""
    global _registered
    _registered = False
