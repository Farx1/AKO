"""GPU and device introspection (NVIDIA CUDA, AMD ROCm, CPU fallback)."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache

import torch


# cudaDeviceGetAttribute codes (cuda_runtime_api.h). Torch 2.6 on this driver
# does not expose shared_memory_per_block* on the properties object at all.
_CUDA_ATTR_MAX_SHARED_MEMORY_PER_BLOCK = 8
_CUDA_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
_CUDA_ATTR_L2_CACHE_SIZE = 38
_CUDA_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR = 82


@lru_cache(maxsize=8)
def _cudart_attr(device_index: int, attribute: int) -> int | None:
    """Read a CUDA device attribute via libcudart, or None if the runtime is missing."""
    try:
        lib = ctypes.CDLL("libcudart.so")
    except OSError:
        try:
            lib = ctypes.CDLL("libcudart.so.12")
        except OSError:
            return None
    value = ctypes.c_int()
    err = lib.cudaDeviceGetAttribute(ctypes.byref(value), attribute, int(device_index or 0))
    if err != 0 or value.value <= 0:
        return None
    return int(value.value)


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of the active compute device."""

    device_type: str
    device_name: str
    device_index: int
    cuda_capability: tuple[int, int] | None
    total_memory_gb: float | None
    #: NVIDIA driver version ("580.65.06"), not the CUDA capability. Part of the
    #: toolchain key: a driver change can move every timing in the database.
    driver_version: str | None
    gpu_vendor: str | None = None
    compute_backend: str = "cpu"
    #: Occupancy and tiling limits, read once from the device properties. They are
    #: what makes a candidate feasible or not, so they belong next to the cache key.
    sm_count: int | None = None
    smem_per_block: int | None = None
    l2_bytes: int | None = None
    regs_per_sm: int | None = None

    @property
    def cache_key(self) -> str:
        safe_name = self.device_name.replace(" ", "_")
        return f"{safe_name}_{self.device_index}"

    def to_dict(self) -> dict:
        return asdict(self)


@lru_cache(maxsize=1)
def _read_amd_gpu_name() -> str | None:
    lspci = shutil.which("lspci")
    if not lspci:
        return None
    try:
        out = subprocess.run([lspci], capture_output=True, text=True, timeout=3, check=False)
        for line in out.stdout.splitlines():
            lower = line.lower()
            if "display controller" in lower or "vga compatible controller" in lower:
                if "amd" in lower or "ati" in lower or "radeon" in lower:
                    return line.split(": ", 1)[-1].strip()
    except Exception:
        return None
    return None


@lru_cache(maxsize=1)
def _read_driver_version() -> str | None:
    """Driver version from ``nvidia-smi``; torch only exposes the CUDA runtime."""
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except Exception:
        return None
    return out.splitlines()[0].strip() if out else None


@lru_cache(maxsize=1)
def _detect_compute_backend() -> str:
    if torch.cuda.is_available():
        if getattr(torch.version, "hip", None):
            return "rocm"
        return "cuda"
    return "cpu"


def get_device_info(device: torch.device | str | None = None) -> DeviceInfo:
    """Return metadata for ``device`` (defaults to best available accelerator or CPU).

    Results are memoized: this sits on the dispatch hot path, and probing the
    hardware on every kernel call would dominate the dispatch overhead ``δ``.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
    return _device_info(device.type, device.index)


def clear_device_cache() -> None:
    """Forget memoized hardware probes (tests, or after a device hot-plug)."""
    _device_info.cache_clear()
    _read_amd_gpu_name.cache_clear()
    _read_driver_version.cache_clear()
    _detect_compute_backend.cache_clear()


@lru_cache(maxsize=None)
def _device_info(device_type: str, device_index: int | None) -> DeviceInfo:
    backend = _detect_compute_backend()
    device = (
        torch.device(device_type, device_index)
        if device_index is not None
        else torch.device(device_type)
    )

    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        capability = (props.major, props.minor)
        total_gb = round(props.total_memory / (1024**3), 2)
        vendor = "amd" if backend == "rocm" else "nvidia"
        return DeviceInfo(
            device_type="cuda",
            device_name=props.name,
            device_index=device.index or 0,
            cuda_capability=capability,
            total_memory_gb=total_gb,
            driver_version=_read_driver_version(),
            gpu_vendor=vendor,
            compute_backend=backend,
            sm_count=getattr(props, "multi_processor_count", None),
            # Prefer the opt-in limit (101376 B on cc 8.6). Torch 2.6 on this
            # driver does not expose ``shared_memory_per_block_optin`` on the
            # properties object at all; libcudart still knows the number.
            smem_per_block=(
                getattr(props, "shared_memory_per_block_optin", None)
                or getattr(props, "shared_memory_per_block", None)
                or _cudart_attr(device.index or 0, _CUDA_ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
                or _cudart_attr(device.index or 0, _CUDA_ATTR_MAX_SHARED_MEMORY_PER_BLOCK)
            ),
            l2_bytes=(
                getattr(props, "L2_cache_size", None)
                or _cudart_attr(device.index or 0, _CUDA_ATTR_L2_CACHE_SIZE)
            ),
            regs_per_sm=(
                getattr(props, "regs_per_multiprocessor", None)
                or _cudart_attr(device.index or 0, _CUDA_ATTR_MAX_REGISTERS_PER_MULTIPROCESSOR)
            ),
        )

    # Only worth probing lspci when PyTorch found no accelerator: it tells the
    # user whether an unsupported (e.g. integrated AMD) GPU is present.
    amd_gpu = _read_amd_gpu_name()
    cpu_name = "cpu"
    if amd_gpu:
        cpu_name = f"cpu (amd_gpu={amd_gpu})"

    return DeviceInfo(
        device_type=device.type,
        device_name=cpu_name,
        device_index=-1,
        cuda_capability=None,
        total_memory_gb=None,
        driver_version=None,
        gpu_vendor="amd" if amd_gpu else None,
        compute_backend="cpu",
    )
