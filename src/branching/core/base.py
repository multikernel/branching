# SPDX-License-Identifier: Apache-2.0
"""Abstract base class for branching filesystems."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .branch import Branch


class FSBackend(ABC):
    """
    Abstract base class for branching filesystems.

    Implementations should:
    1. Extend this class
    2. Implement all abstract methods
    3. Register themselves using registry.register()

    All methods are classmethods — backends are stateless.
    """

    @classmethod
    @abstractmethod
    def fstype(cls) -> str:
        """
        Return filesystem type name as it appears in /proc/mounts.

        Returns:
            Filesystem type string (e.g., 'daxfs', 'fuse.branchfs')
        """
        pass

    @classmethod
    @abstractmethod
    def create_branch(
        cls, name: str, mountpoint: Path, parent_mount: Path, parent_branch: str
    ) -> Optional[Path]:
        """
        Create and mount a new branch.

        Args:
            name: Branch name (e.g., 'feature')
            mountpoint: Where to mount the new branch
            parent_mount: Path to parent's mountpoint (for getting backing info)
            parent_branch: Parent branch path (e.g., '/main' or '/main/L1')

        Returns:
            The actual branch working directory if different from mountpoint
            (e.g. BranchFS returns mount/@branch_id), or None to keep mountpoint.
        """
        pass

    @classmethod
    @abstractmethod
    def commit(cls, mountpoint: Path) -> None:
        """
        Commit the branch at mountpoint.

        Args:
            mountpoint: Path to the mounted branch

        Raises:
            CommitError: If the commit operation fails
        """
        pass

    @classmethod
    @abstractmethod
    def abort(cls, mountpoint: Path) -> None:
        """
        Abort the branch at mountpoint.

        Args:
            mountpoint: Path to the mounted branch

        Raises:
            AbortError: If the abort operation fails
        """
        pass

    @classmethod
    def single_mount(cls) -> bool:
        """Whether this backend uses a single mount (view switches in-place).

        Returns True for ioctl-based backends like BranchFS where branching
        happens within a single mount. Returns False for mount-per-branch
        backends like DaxFS.
        """
        return False
