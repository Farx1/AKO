"""Execute the Triton kernels on CPU via ``TRITON_INTERPRET=1``.

The GPU tests need CUDA hardware; these do not. They run the actual kernel
bodies through Triton's interpreter and compare them to the PyTorch references,
which catches indexing, masking and rounding bugs on a laptop — long before a
GPU is available. Performance still has to be measured on a real device.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECKS_SCRIPT = Path(__file__).parent / "_triton_interpreter_checks.py"
REPO_ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-5

triton = pytest.importorskip("triton", reason="Triton is not installed")


@pytest.fixture(scope="module")
def interpreter_results() -> dict[str, float]:
    """Run every kernel check once, in a subprocess with the interpreter on."""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, str(CHECKS_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=900,
        env=env,
        cwd=str(REPO_ROOT),
    )
    if completed.returncode != 0:
        pytest.fail(f"Interpreter checks failed:\n{completed.stderr[-3000:]}")
    return json.loads(completed.stdout)


@pytest.mark.slow
@pytest.mark.parametrize(
    "check",
    [
        "rmsnorm",
        "rmsnorm_ragged",
        "rmsnorm_wide",
        "rmsnorm_residual",
        "rope_neox",
        "rope_interleaved",
        "rope_many_heads",
        "swiglu",
        "swiglu_wide_ragged",
    ],
)
def test_kernel_matches_reference(interpreter_results: dict[str, float], check: str) -> None:
    assert check in interpreter_results, f"missing result for {check}"
    assert interpreter_results[check] < TOLERANCE


@pytest.mark.slow
def test_mm_template_matches_eager_within_its_budget_fp16(
    interpreter_results: dict[str, float],
) -> None:
    """M4: every composition of the one template, against one eager reference.

    Values are error / error_budget(reference), so the gate is the dtype's own
    precision (ako.validation.numerical) and not a threshold invented here.

    fp16 only. The 24 legal bf16 cases are NOT TESTED on CPU (the Triton 3.7.1
    interpreter reads bf16 as uint16 in create_dot) and are first validated on
    the GPU at M5 — see ROADMAP M4.
    """
    ratios = {k: v for k, v in interpreter_results.items() if k.startswith("mm_")}
    # 4 shapes x 4 compositions at SPLIT_K=1, plus the 2 unfused ones at
    # SPLIT_K=4 = 24 legal fp16 cases (half of the 48 the gate below counts).
    assert len(ratios) == 24, sorted(ratios)
    assert {k: v for k, v in ratios.items() if v >= 1.0} == {}


@pytest.mark.slow
def test_splitk_with_a_fused_epilogue_is_rejected_for_free(
    interpreter_results: dict[str, float],
) -> None:
    """M4: the 64 cases split 48 legal / 16 infeasible, and the split is asserted.

    "Zero failures out of 64" would be unsatisfiable: SPLIT_K > 1 under a GEMM
    that carries an epilogue is declared illegal, so the test fails if
    ``feasible()`` lets any of the 16 through *or* rejects any of the 48.
    """
    gate = {k[len("gate_") :]: v for k, v in interpreter_results.items() if k.startswith("gate_")}
    assert gate == {"feasible": 48, "splitk_with_epilogue": 16}
