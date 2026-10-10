# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
from `0.2.0` onward. While the major version is `0`, the public API
(`ako.rmsnorm`, `ako.rmsnorm_residual`, `ako.rope`, `ako.rope_qk`, `ako.swiglu`,
`ako.build_rope_tables`) may still change in a minor release; `ako.dispatcher`,
`ako.models` and `ako.experiments` carry no compatibility promise yet.

There are no released versions and no git tags: `0.2.0` is the version currently
declared in `pyproject.toml` and `ako/__init__.py`, and the package has never
been published. Install from git.

## [Unreleased]

### Added

- `docs/EXTENDING.md`: the public contract for registering a third-party kernel
  — `KernelEntry` / `KernelRegistry` / `KernelSelector` signatures, the calling
  convention of every dispatched operation, a runnable CPU example, what the
  harness guarantees and what it does not, and how to test a Triton kernel with
  `TRITON_INTERPRET=1` on a machine without a GPU.
- `ako/py.typed`, declared in the packaging, so the type hints already present
  in the source are visible to type checkers downstream (PEP 561).
- Packaging metadata for a PyPI release: project URLs, SPDX license declaration
  with `LICENSE` attached, `Typing :: Typed` and OS classifiers.

### Changed

- Build requirement raised to `setuptools>=77` (SPDX `license` / `license-files`
  support); the redundant `wheel` build dependency was dropped.

### Not done

- Not published to PyPI. There is no release workflow and no publish token; the
  name `ako` has not been checked for availability.
- No API stability guarantee yet, and no deprecation policy.

## State of the project

An honest summary of what exists today, for the first reader of this file.

- Dispatched operations: RMSNorm, fused residual + RMSNorm, RoPE (`neox` and
  `interleaved`), SwiGLU. Each has a PyTorch reference, a Triton kernel, and
  where applicable a `torch.compile` / `F.rms_norm` baseline.
- The paper's two heavier operations — quantized INT8/INT4 matmul and IO-aware
  attention — are not started. They are the ones that would make the end-to-end
  number matter, which is why the measured whole-model gain is around 1 %.
- Selection is validated then benchmarked, and cached per
  (GPU, operation, shape, dtype) in `~/.ako/kernel_cache.json`; a candidate
  outside the dtype-derived error budget is rejected and never dispatched.
- Measurements come from a single GPU (RTX 3070 Laptop, see `docs/RESULTS.md`).
  Nothing is claimed about other hardware.
- Triton kernel logic is tested on CPU through Triton's interpreter; performance
  is not, and cannot be, tested without a device.
- CI runs the non-GPU test suite on Python 3.11 and 3.12 with `compileall` as
  its only lint. There is no coverage gate, no GPU runner, no publish job.
