# SPDX-License-Identifier: Apache-2.0
"""Branch class with context manager support for speculative execution."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Literal, Union

from .base import FSBackend

if TYPE_CHECKING:
    from .workspace import Workspace

OnSuccessAction = Optional[Literal["commit"]]
OnErrorAction = Optional[Literal["abort"]]


class Branch:
    """
    A branch in a workspace, usable as a context manager.

    Branches provide isolation for speculative execution. Changes made
    within a branch are isolated until committed. On success, changes
    can be committed (merged to parent). On error, changes can be
    aborted (rolled back to parent).

    Example:
        with workspace.branch("attempt") as b:
            subprocess.run(["agent", "--workdir", str(b.path)])
            # auto-commits on success, auto-aborts on exception

        # Nested branches
        with workspace.branch("L1") as b1:
            with b1.branch("L2") as b2:
                risky_operation(b2.path)

    Attributes:
        name: Branch name
        path: Path to the mounted branch directory
    """

    def __init__(
        self,
        fs: type[FSBackend],
        name: str,
        mount: Path,
        parent: Union["Workspace", "Branch"],
        parent_branch: str,
        on_success: OnSuccessAction = "commit",
        on_error: OnErrorAction = "abort",
        mount_root: Optional[Path] = None,
    ):
        """
        Initialize a branch.

        Args:
            fs: Filesystem implementation class
            name: Branch name
            mount: Path where branch will be mounted
            parent: Parent Workspace or Branch
            parent_branch: Parent's branch path (e.g., '/main' or '/main/L1')
            on_success: Action on clean exit - "commit" or None
            on_error: Action on exception - "abort" or None
            mount_root: Filesystem mount root (for single-mount backends where
                branch virtual paths are always relative to root).
        """
        self._fs = fs
        self._name = name
        self._path = mount
        self._parent = parent
        self._parent_branch = parent_branch
        self._on_success = on_success
        self._on_error = on_error
        self._mount_root = mount_root or mount
        self._finished = False
        self._mounted = False

    @property
    def name(self) -> str:
        """Branch name."""
        return self._name

    @property
    def path(self) -> Path:
        """Path to the mounted branch directory."""
        return self._path

    @property
    def branch_path(self) -> str:
        """Full branch path (e.g., '/main/feature')."""
        if self._parent_branch in ("main", "/main"):
            return f"/main/{self._name}"
        return f"{self._parent_branch}/{self._name}"

    def branch(
        self,
        name: str,
        on_success: OnSuccessAction = "commit",
        on_error: OnErrorAction = "abort",
    ) -> "Branch":
        """
        Create a child branch from this branch.

        Args:
            name: Child branch name
            on_success: Action on clean exit
            on_error: Action on exception

        Returns:
            New Branch instance
        """
        if self._fs.single_mount():
            # Single-mount backends: mountpoint is always the mount root
            # (create_branch will return the actual virtual path)
            child_mount = self._mount_root
        else:
            # Mount-per-branch backends: create sibling directory
            child_mount = self._path.parent / f"{self._path.name}_{name}"
        return Branch(
            fs=self._fs,
            name=name,
            mount=child_mount,
            parent=self,
            parent_branch=self.branch_path,
            on_success=on_success,
            on_error=on_error,
            mount_root=self._mount_root,
        )

    def commit(self) -> None:
        """
        Commit the branch.

        For DaxFS: commits the entire branch chain to main.
        For BranchFS: commits the leaf branch to its parent.
        """
        if self._finished:
            return
        self._fs.commit(self._path)
        self._finished = True

    def abort(self) -> None:
        """
        Abort the branch.

        For DaxFS: aborts the entire branch chain back to main.
        For BranchFS: aborts the leaf branch, returning to parent.
        """
        if self._finished:
            return
        self._fs.abort(self._path)
        self._finished = True

    def _mount(self) -> None:
        """Mount the branch (called by __enter__)."""
        if self._mounted:
            return
        parent_mount = self._parent.path
        branch_path = self._fs.create_branch(
            self._name, self._path, parent_mount, self._parent_branch
        )
        if branch_path is not None:
            self._path = branch_path
        self._mounted = True

    def __enter__(self) -> "Branch":
        """Enter the branch context, mounting the branch."""
        self._mount()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the branch context.

        Performs the configured action based on whether an exception occurred.
        Returns False to propagate exceptions.
        """
        if self._finished:
            return False

        if exc_type is not None:
            # Exception occurred
            if self._on_error == "abort":
                self.abort()
            # If None, leave branch as-is
        else:
            # Clean exit
            if self._on_success == "commit":
                self.commit()
            # If None, leave branch as-is

        return False  # Don't suppress exceptions

    def __repr__(self) -> str:
        return f"Branch(name={self._name!r}, path={self._path!r})"
