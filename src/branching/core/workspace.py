# SPDX-License-Identifier: Apache-2.0
"""Workspace class - main user-facing API for branch management."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from .base import FSBackend
from .branch import Branch, OnSuccessAction, OnErrorAction
from .registry import detect_fs_for_mount
from ..exceptions import MountError


class _OwnedMount:
    """Manages a branchfs mount lifecycle started by Workspace.mount()."""

    def __init__(
        self,
        base: Path,
        mountpoint: Optional[Path],
        storage: Optional[Path],
        binary: Path,
    ):
        self.base = base
        self.binary = binary
        self._tmp_mountpoint = mountpoint is None
        self._tmp_storage = storage is None
        self.mountpoint = Path(mountpoint) if mountpoint else Path(
            tempfile.mkdtemp(prefix="branchfs_mnt_")
        )
        self.storage = Path(storage) if storage else Path(
            tempfile.mkdtemp(prefix="branchfs_storage_")
        )

    def start(self) -> None:
        """Run `branchfs mount` which starts the daemon and mounts."""
        self.mountpoint.mkdir(parents=True, exist_ok=True)
        self.storage.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                str(self.binary), "mount",
                "--base", str(self.base),
                "--storage", str(self.storage),
                str(self.mountpoint),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            self._cleanup_dirs()
            raise MountError(
                f"branchfs mount failed: {result.stderr.strip() or result.stdout.strip()}"
            )

        # Wait for FUSE to be ready
        ctl = self.mountpoint / ".branchfs_ctl"
        for _ in range(50):
            if ctl.exists():
                return
            time.sleep(0.1)

        self.stop()
        raise MountError("branchfs mount timed out waiting for FUSE")

    def stop(self) -> None:
        """Unmount via branchfs CLI and clean up temp dirs."""
        subprocess.run(
            [
                str(self.binary), "unmount",
                str(self.mountpoint),
                "--storage", str(self.storage),
            ],
            capture_output=True,
        )
        self._cleanup_dirs()

    def _cleanup_dirs(self) -> None:
        if self._tmp_mountpoint:
            shutil.rmtree(self.mountpoint, ignore_errors=True)
        if self._tmp_storage:
            shutil.rmtree(self.storage, ignore_errors=True)


class Workspace:
    """
    Entry point for agents - wraps a branching filesystem mount.

    A Workspace represents the main branch of a branching filesystem.
    Use it to create branches for speculative execution.

    Example:
        from branching import Workspace

        # Mount and use (handles setup and teardown)
        with Workspace.mount("./my_project") as ws:
            with ws.branch("attempt1") as b:
                (b.path / "result.txt").write_text("done")
                # auto-commits on success, auto-aborts on exception

        # Or open an already-mounted workspace
        ws = Workspace("/mnt/main")

    Attributes:
        path: Path to the main branch mountpoint
        fstype: Detected filesystem type name
    """

    def __init__(self, path: str | Path):
        """
        Open a workspace from an existing mount.

        Args:
            path: Path to the mounted branching filesystem

        Raises:
            BranchingError: If no mount found or filesystem not supported
        """
        self._path = Path(path).resolve()
        self._fs, self._mount_root = detect_fs_for_mount(self._path)
        self._owned_mount: Optional[_OwnedMount] = None

    @classmethod
    def mount(
        cls,
        base: str | Path,
        *,
        mountpoint: str | Path | None = None,
        storage: str | Path | None = None,
        branchfs_bin: str | Path | None = None,
    ) -> Workspace:
        """
        Mount a branchfs workspace and return a ready Workspace.

        Handles daemon startup and FUSE mount automatically.
        Use as a context manager for automatic cleanup::

            with Workspace.mount("./project") as ws:
                with ws.branch("fix") as b:
                    ...

        Or call ``ws.close()`` manually when done.

        Args:
            base: Directory to branch from (your project root).
            mountpoint: Where to mount FUSE. Auto-created temp dir if None.
            storage: Directory for daemon/branch data. Temp dir if None.
            branchfs_bin: Path to branchfs binary. Searches PATH if None.

        Raises:
            MountError: If branchfs binary not found or mount fails.
        """
        base = Path(base).resolve()
        if not base.is_dir():
            raise MountError(f"Base directory does not exist: {base}")

        binary = _find_branchfs(branchfs_bin)

        owned = _OwnedMount(
            base,
            Path(mountpoint) if mountpoint else None,
            Path(storage) if storage else None,
            binary,
        )
        owned.start()

        ws = cls(owned.mountpoint)
        ws._owned_mount = owned
        return ws

    def close(self) -> None:
        """Unmount and clean up if this workspace was created via mount().

        No-op if the workspace was opened from an existing mount.
        """
        if self._owned_mount is not None:
            self._owned_mount.stop()
            self._owned_mount = None

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.close()
        return False

    @property
    def path(self) -> Path:
        """Working directory for main branch."""
        return self._path

    @property
    def fstype(self) -> str:
        """Detected filesystem type name."""
        return self._fs.fstype()

    def _generate_mountpoint(self, name: str) -> Path:
        """
        Generate a mountpoint path for a branch.

        Args:
            name: Branch name

        Returns:
            Path for the new branch mount
        """
        return self._mount_root

    def branch(
        self,
        name: str,
        on_success: OnSuccessAction = "commit",
        on_error: OnErrorAction = "abort",
    ) -> Branch:
        """
        Create a branch for speculative execution.

        Args:
            name: Branch name (used for mount path generation)
            on_success: Action on clean exit - "commit" | None
            on_error: Action on exception - "abort" | None

        Returns:
            Branch context manager with .path for agent working directory
        """
        mount = self._generate_mountpoint(name)
        return Branch(
            fs=self._fs,
            name=name,
            mount=mount,
            parent=self,
            parent_branch="/main",
            on_success=on_success,
            on_error=on_error,
            mount_root=self._mount_root,
        )

    def __repr__(self) -> str:
        return f"Workspace(path={self._path!r}, fstype={self.fstype!r})"


def _find_branchfs(hint: str | Path | None) -> Path:
    """Locate the branchfs binary."""
    if hint is not None:
        p = Path(hint)
        if p.is_file():
            return p
        raise MountError(f"branchfs binary not found at {p}")

    found = shutil.which("branchfs")
    if found:
        return Path(found)

    raise MountError(
        "branchfs binary not found in PATH. "
        "Install branchfs or pass branchfs_bin= to Workspace.mount()."
    )
