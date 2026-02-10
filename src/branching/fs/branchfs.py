# SPDX-License-Identifier: Apache-2.0
"""BranchFS (FUSE) implementation of the FSBackend interface.

Uses the named-branch model with per-branch virtual paths:
- CREATE ioctl on a ctl fd → daemon creates branch, returns name in 128-byte buf
- Each branch is accessible at /@{name}/ with its own .branchfs_ctl
- COMMIT/ABORT via per-branch ctl fd (/@{name}/.branchfs_ctl)
- First-winner-commit: ESTALE on conflict (sibling already committed)
"""

import ctypes
import errno
import fcntl
import os
from pathlib import Path
from typing import Optional

from ..core.base import FSBackend
from ..exceptions import MountError, CommitError, AbortError, ConflictError

# ioctl numbers — must match branchfs daemon (fs.rs)
FS_IOC_BRANCH_CREATE = 0x8080_6200  # _IOR('b', 0, [u8; 128])
FS_IOC_BRANCH_COMMIT = 0x0000_6201  # _IO ('b', 1)
FS_IOC_BRANCH_ABORT = 0x0000_6202   # _IO ('b', 2)

CTL_FILE = ".branchfs_ctl"


def _ctl_ioctl(ctl_path: Path, cmd: int, op_name: str) -> None:
    """Open a ctl file and issue a simple ioctl (no output buffer)."""
    fd = os.open(str(ctl_path), os.O_RDWR)
    try:
        fcntl.ioctl(fd, cmd, 0)
    except OSError as e:
        if e.errno == errno.ESTALE:
            raise ConflictError(
                f"{op_name} conflict at {ctl_path} (sibling already committed)"
            ) from e
        raise
    finally:
        os.close(fd)


def _ctl_create(ctl_path: Path) -> str:
    """Issue CREATE ioctl and return the new branch name."""
    fd = os.open(str(ctl_path), os.O_RDWR)
    try:
        buf = ctypes.create_string_buffer(128)
        fcntl.ioctl(fd, FS_IOC_BRANCH_CREATE, buf)
        return buf.value.decode()
    finally:
        os.close(fd)


class BranchFSBackend(FSBackend):
    """Named-branch model with per-branch virtual paths.

    Key properties:
    - Single mount, branches accessed via /@{name}/ virtual paths
    - CREATE ioctl returns branch name (UUID); parent from ctl fd context
    - COMMIT/ABORT via per-branch ctl (/@{name}/.branchfs_ctl)
    - Parallel sibling branches supported (first-winner-commit)
    """

    @classmethod
    def fstype(cls) -> str:
        return "fuse.branchfs"

    @classmethod
    def single_mount(cls) -> bool:
        return True

    @classmethod
    def create_branch(
        cls, name: str, mountpoint: Path, parent_mount: Path, parent_branch: str
    ) -> Optional[Path]:
        """Create a new branch via CREATE ioctl.

        The ioctl is issued on the parent's ctl fd:
        - Root ctl (/.branchfs_ctl) when parent is main
        - Per-branch ctl (/@{parent}/.branchfs_ctl) when parent is a branch

        Args:
            name: Requested branch name (informational; daemon assigns UUID)
            mountpoint: Mount root
            parent_mount: Parent's path (mount root for main, /@{id} for nested)
            parent_branch: Parent branch path (ignored; ctl fd determines parent)

        Returns:
            Path to the new branch's virtual directory (mount/@{uuid})
        """
        ctl_path = parent_mount / CTL_FILE
        try:
            branch_id = _ctl_create(ctl_path)
        except OSError as e:
            raise MountError(
                f"Failed to create branch '{name}' at {parent_mount}: {e}"
            ) from e

        return mountpoint / f"@{branch_id}"

    @classmethod
    def commit(cls, mountpoint: Path) -> None:
        """Commit via per-branch ctl (mountpoint is /@{name}/).

        Raises:
            ConflictError: If a sibling already committed (ESTALE).
            CommitError: If the commit fails for other reasons.
        """
        try:
            _ctl_ioctl(mountpoint / CTL_FILE, FS_IOC_BRANCH_COMMIT, "commit")
        except ConflictError:
            raise
        except OSError as e:
            raise CommitError(f"Failed to commit at {mountpoint}: {e}") from e

    @classmethod
    def abort(cls, mountpoint: Path) -> None:
        """Abort via per-branch ctl (mountpoint is /@{name}/)."""
        try:
            _ctl_ioctl(mountpoint / CTL_FILE, FS_IOC_BRANCH_ABORT, "abort")
        except OSError as e:
            raise AbortError(f"Failed to abort at {mountpoint}: {e}") from e



# Auto-register on import
from ..core.registry import register

register("fuse.branchfs", BranchFSBackend)
