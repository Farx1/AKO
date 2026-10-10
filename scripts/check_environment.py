"""Print AKO runtime environment and hardware compatibility."""

from __future__ import annotations

import sys

import torch
from rich.console import Console
from rich.table import Table

from ako.dispatcher.hardware import get_device_info
from ako.ops.rmsnorm_triton import is_triton_available

console = Console()


def main() -> int:
    info = get_device_info()
    table = Table(title="AKO Environment")
    table.add_column("Key")
    table.add_column("Value")
    table.add_row("Python", sys.version.split()[0])
    table.add_row("PyTorch", torch.__version__)
    table.add_row("Compute backend", info.compute_backend)
    table.add_row("CUDA available", str(torch.cuda.is_available()))
    table.add_row("Triton available", str(is_triton_available()))
    table.add_row("Device info", info.device_name)
    table.add_row("GPU vendor", info.gpu_vendor or "n/a")
    console.print(table)

    if info.gpu_vendor == "amd" and info.compute_backend == "cpu":
        console.print(
            "\n[yellow]AMD GPU detected but PyTorch runs on CPU.[/yellow]\n"
            "AKO Triton/CUDA kernels target NVIDIA GPUs. On this machine the project\n"
            "runs in CPU mode (PyTorch baseline + tests + benchmarks).\n"
            "For GPU compute on AMD, a ROCm PyTorch build and supported discrete GPU\n"
            "are required — integrated Radeon 840M is not supported for AKO kernels."
        )

    if not torch.cuda.is_available():
        console.print("\n[green]CPU pipeline OK[/green] — run: pytest -q -m 'not gpu'")
        return 0

    console.print("\n[green]CUDA pipeline available[/green] — run: pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
