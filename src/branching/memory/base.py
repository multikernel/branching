# SPDX-License-Identifier: Apache-2.0
"""Memory branching backends.

This module defines the MemoryBackend ABC, a stub implementation, and
an mprotect-based implementation that enforces read-only parent memory
after branching.
"""

from abc import ABC, abstractmethod
from typing import Any

from ._mprotect import (
    MemoryRegion,
    PROT_READ,
    PROT_WRITE,
    mprotect,
)


class MemoryBackend(ABC):
    """Abstract base class for memory branching.

    Memory branching creates copy-on-write snapshots of process address
    space regions, enabling speculative computation with rollback.
    """

    @abstractmethod
    def snapshot(self, addr: int, size: int) -> Any:
        """Create a COW snapshot of a memory region.

        Args:
            addr: Start address of the region.
            size: Size of the region in bytes.

        Returns:
            Opaque handle for the snapshot.
        """
        pass

    @abstractmethod
    def restore(self, handle: Any) -> None:
        """Restore a memory region from a snapshot (abort).

        Args:
            handle: Snapshot handle from snapshot().
        """
        pass

    @abstractmethod
    def commit(self, handle: Any) -> None:
        """Commit a snapshot (make changes permanent, discard snapshot).

        Args:
            handle: Snapshot handle from snapshot().
        """
        pass


class StubMemoryBackend(MemoryBackend):
    """Stub implementation that raises NotImplementedError.

    Full memory branching (COW snapshots of address space regions)
    requires kernel support that is not yet available.
    """

    def snapshot(self, addr: int, size: int) -> Any:
        raise NotImplementedError(
            "Memory branching requires kernel support (not yet available)."
        )

    def restore(self, handle: Any) -> None:
        raise NotImplementedError(
            "Memory branching requires kernel support (not yet available)."
        )

    def commit(self, handle: Any) -> None:
        raise NotImplementedError(
            "Memory branching requires kernel support (not yet available)."
        )


class MprotectMemoryBackend(MemoryBackend):
    """Memory backend using mprotect(2) to enforce read-only parent state.

    After ``snapshot()``, the region is marked read-only so the parent
    cannot mutate branched state. ``restore()`` and ``commit()`` both
    re-enable write access (the parent regains control after branching).
    """

    def snapshot(self, addr: int, size: int) -> MemoryRegion:
        """Mark a memory region as read-only and return a handle.

        Args:
            addr: Page-aligned start address of the region.
            size: Size of the region in bytes.

        Returns:
            A MemoryRegion handle that can be passed to restore/commit.
        """
        region = MemoryRegion(addr=addr, size=size, original_prot=PROT_READ | PROT_WRITE)
        mprotect(addr, size, PROT_READ)
        return region

    def restore(self, handle: Any) -> None:
        """Restore write access to a previously snapshotted region.

        Args:
            handle: MemoryRegion from snapshot().
        """
        region: MemoryRegion = handle
        mprotect(region.addr, region.size, region.original_prot)

    def commit(self, handle: Any) -> None:
        """Commit the snapshot — parent regains write access.

        Args:
            handle: MemoryRegion from snapshot().
        """
        region: MemoryRegion = handle
        mprotect(region.addr, region.size, region.original_prot)
