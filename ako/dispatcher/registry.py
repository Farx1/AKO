"""Kernel registry — maps operation names to implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

KernelFn = Callable[..., torch.Tensor]


@dataclass(frozen=True)
class KernelEntry:
    """A single registered kernel implementation."""

    op_name: str
    kernel_name: str
    fn: KernelFn
    backend: str
    priority: int = 0


class KernelRegistry:
    """In-memory registry of kernels grouped by operation."""

    def __init__(self) -> None:
        self._entries: dict[str, list[KernelEntry]] = {}

    def register(self, entry: KernelEntry) -> None:
        """Add a kernel, replacing any previous entry with the same name."""
        kernels = [
            e for e in self._entries.get(entry.op_name, []) if e.kernel_name != entry.kernel_name
        ]
        kernels.append(entry)
        kernels.sort(key=lambda e: e.priority, reverse=True)
        self._entries[entry.op_name] = kernels

    def list_ops(self) -> list[str]:
        return sorted(self._entries)

    def list_kernels(self, op_name: str) -> list[KernelEntry]:
        return list(self._entries.get(op_name, []))

    def get(self, op_name: str, kernel_name: str) -> KernelEntry | None:
        for entry in self.list_kernels(op_name):
            if entry.kernel_name == kernel_name:
                return entry
        return None

    def default(self, op_name: str) -> KernelEntry | None:
        kernels = self.list_kernels(op_name)
        return kernels[0] if kernels else None


_global_registry = KernelRegistry()


def get_registry() -> KernelRegistry:
    return _global_registry
