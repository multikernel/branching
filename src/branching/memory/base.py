# SPDX-License-Identifier: Apache-2.0
"""Memory branching stub for future kernel support.

This module defines the MemoryBackend ABC and a stub implementation.
Real memory branching requires kernel support via the proposed branch()
syscall with BR_MEMORY flag.
"""

from abc import ABC, abstractmethod
from typing import Any


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

    Memory branching requires kernel support via the proposed branch()
    syscall. See the BranchFS paper for the design of BR_MEMORY.
    """

    def snapshot(self, addr: int, size: int) -> Any:
        raise NotImplementedError(
            "Memory branching requires kernel support via the branch() syscall. "
            "See the BranchFS paper for the proposed BR_MEMORY design."
        )

    def restore(self, handle: Any) -> None:
        raise NotImplementedError(
            "Memory branching requires kernel support via the branch() syscall."
        )

    def commit(self, handle: Any) -> None:
        raise NotImplementedError(
            "Memory branching requires kernel support via the branch() syscall."
        )
