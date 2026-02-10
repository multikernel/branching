# SPDX-License-Identifier: Apache-2.0
"""Workspace class - main user-facing API for branch management."""

from pathlib import Path

from .base import FSBackend
from .branch import Branch, OnSuccessAction, OnErrorAction
from .registry import detect_fs_for_mount


class Workspace:
    """
    Entry point for agents - wraps a branching filesystem mount.

    A Workspace represents the main branch of a branching filesystem.
    Use it to create branches for speculative execution.

    Example:
        from branching import Workspace

        # Open workspace from existing mount
        ws = Workspace("/mnt/main")

        # Simple speculative execution
        with ws.branch("attempt1") as b:
            subprocess.run(["agent", "--workdir", str(b.path)])
            # auto-commits on success, auto-aborts on exception

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
        self._fs = detect_fs_for_mount(self._path)

    @property
    def path(self) -> Path:
        """Working directory for main branch."""
        return self._path

    @property
    def fstype(self) -> str:
        """Detected filesystem type name."""
        return self._fs.fstype()

    @property
    def single_mount(self) -> bool:
        """Whether the backend uses stack-based branching on a single mount.

        When True, branches are serialized (no concurrent siblings).
        """
        return self._fs.single_mount()

    def _generate_mountpoint(self, name: str) -> Path:
        """
        Generate a mountpoint path for a branch.

        Backend-aware: single-mount backends (BranchFS) reuse the mount root;
        mount-per-branch backends (DaxFS) create a sibling directory.

        Args:
            name: Branch name

        Returns:
            Path for the new branch mount
        """
        if self._fs.single_mount():
            # BranchFS: view switches in-place, same mount root
            return self._path
        else:
            # DaxFS: each branch gets its own mount
            return self._path.parent / f"{self._path.name}_{name}"

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
            mount_root=self._path,
        )

    def __repr__(self) -> str:
        return f"Workspace(path={self._path!r}, fstype={self.fstype!r})"
