# SPDX-License-Identifier: Apache-2.0
"""DaxFS implementation of the FSBackend interface."""

import os
from pathlib import Path
from typing import Optional

from ..core.base import FSBackend
from ..exceptions import MountError, CommitError, AbortError
from ._mount import find_any_mount
from ._syscalls import (
    fsopen,
    fsconfig_set_string,
    fsconfig_set_fd,
    fsconfig_cmd_create,
    fsmount,
    move_mount,
    mount,
    umount,
    ioctl,
    AT_FDCWD,
    MOVE_MOUNT_F_EMPTY_PATH,
    MS_REMOUNT,
)

# DaxFS ioctl command - from include/daxfs_format.h
# _IO('D', 1) = (0 << 30) | (ord('D') << 8) | 1 = 0x4401
DAXFS_IOC_GET_DMABUF = 0x4401


def _get_dmabuf_fd(mountpoint: Path) -> int:
    """
    Get dmabuf fd from an existing daxfs mount via ioctl.

    Args:
        mountpoint: Path to existing daxfs mount

    Returns:
        File descriptor for the dmabuf

    Raises:
        OSError: If the ioctl fails
    """
    fd = os.open(str(mountpoint), os.O_RDONLY)
    try:
        return ioctl(fd, DAXFS_IOC_GET_DMABUF)
    finally:
        os.close(fd)


class DaxFSBackend(FSBackend):
    """DaxFS implementation of the FSBackend interface."""

    @classmethod
    def fstype(cls) -> str:
        return "daxfs"

    @classmethod
    def create_branch(
        cls, name: str, mountpoint: Path, parent_mount: Path, parent_branch: str
    ) -> Optional[Path]:
        """
        Create and mount a new daxfs branch.

        Uses the new mount API (fsopen/fsconfig/fsmount/move_mount) because
        fd parameters can't be passed via the mount command.
        """
        # Ensure mountpoint directory exists
        mountpoint.mkdir(parents=True, exist_ok=True)

        # Find any existing daxfs mount to get backing store info
        existing_mount = find_any_mount("daxfs")
        if existing_mount is None:
            raise MountError("No existing daxfs mount found to get backing store")

        # Try to get dmabuf fd from existing mount
        dmabuf_fd = -1
        phys = None
        size = None

        try:
            dmabuf_fd = _get_dmabuf_fd(existing_mount.mountpoint)
        except OSError:
            # Fall back to phys/size from mount options
            phys = existing_mount.get_option("phys")
            size = existing_mount.get_option("size")
            if not phys or not size:
                raise MountError(
                    "Cannot get backing store: no dmabuf and no phys/size in options"
                )

        # Build the full branch path
        if parent_branch == "main" or parent_branch == "/main":
            branch_path = f"/main/{name}"
        else:
            branch_path = f"{parent_branch}/{name}"

        fs_fd = -1
        mnt_fd = -1

        try:
            # Open filesystem context
            fs_fd = fsopen("daxfs")

            # Set backing store
            if dmabuf_fd >= 0:
                fsconfig_set_fd(fs_fd, "dmabuf", dmabuf_fd)
            else:
                fsconfig_set_string(fs_fd, "phys", phys)
                fsconfig_set_string(fs_fd, "size", size)

            # Set branch and parent
            fsconfig_set_string(fs_fd, "branch", branch_path)
            fsconfig_set_string(fs_fd, "parent", parent_branch)

            # Create superblock
            fsconfig_cmd_create(fs_fd)

            # Create mount fd
            mnt_fd = fsmount(fs_fd)

            # Move mount to target
            move_mount(mnt_fd, "", AT_FDCWD, str(mountpoint), MOVE_MOUNT_F_EMPTY_PATH)

        except OSError as e:
            raise MountError(f"Failed to create branch '{name}': {e}") from e
        finally:
            if dmabuf_fd >= 0:
                os.close(dmabuf_fd)
            if fs_fd >= 0:
                os.close(fs_fd)
            if mnt_fd >= 0:
                os.close(mnt_fd)

        return None  # DaxFS mountpoint is already correct

    @classmethod
    def commit(cls, mountpoint: Path) -> None:
        """Commit the branch chain at mountpoint to main."""
        try:
            mount("", str(mountpoint), "", MS_REMOUNT, "commit")
        except OSError as e:
            raise CommitError(f"Failed to commit at {mountpoint}: {e}") from e

    @classmethod
    def abort(cls, mountpoint: Path) -> None:
        """Abort entire branch chain back to main."""
        try:
            mount("", str(mountpoint), "", MS_REMOUNT, "abort")
        except OSError as e:
            raise AbortError(f"Failed to abort at {mountpoint}: {e}") from e



# Auto-register on import
from ..core.registry import register

register("daxfs", DaxFSBackend)
