# SPDX-License-Identifier: Apache-2.0
"""BranchContext: userspace approximation of branch(BR_CREATE).

Each child is confined to its branch workspace via Landlock (write-only
access to the branch path, read access everywhere).  No namespaces or
capabilities required.  Isolation failures raise, not silently degrade.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence, TYPE_CHECKING

from ..exceptions import ForkError, MemoryProtectError, ProcessBranchError
from ..memory._mprotect import (
    MemoryRegion,
    PROT_READ,
    PROT_WRITE,
    mprotect as _mprotect,
)
from ._landlock import confine_to_branch
from ._prlimit import apply_limits as _apply_limits
from ._process_tracker import BpfProcessTracker

if TYPE_CHECKING:
    from .limits import ResourceLimits


class BranchContext:
    """Userspace approximation of branch(BR_CREATE).

    Each child is confined to its branch workspace via Landlock
    (write-only to the branch path, read everywhere).  Process tracking
    via BPF LSM, resource limits via setrlimit.  No namespaces or
    capabilities required.  Isolation failures raise, not silently
    degrade.

    The target callable follows Python conventions: return normally for
    success, raise an exception for failure.
    """

    def __init__(
        self,
        target: Callable[[Path], None],
        workspace: Path,
        *,
        mount_root: Path | None = None,
        close_fds: bool = False,
        limits: ResourceLimits | None = None,
        protected_regions: Sequence[tuple[int, int]] | None = None,
    ):
        """
        Args:
            target: Callable receiving workspace path. Return normally for
                    success, raise for failure.
            workspace: Branch workspace path (BranchFS virtual path).
            mount_root: Filesystem mount root.  Used by Landlock to
                confine the child to *workspace* while blocking reads
                to sibling branches.  Defaults to ``workspace.parent``.
            close_fds: Close inherited fds (3+) in child.
            limits: Optional resource limits applied via setrlimit(2)
                in the child process.
            protected_regions: Optional list of (addr, size) tuples.
                After fork, these regions are marked read-only in the
                parent via mprotect(2) to enforce the branch invariant.
        """
        self._target = target
        self._workspace = workspace
        self._mount_root = mount_root or workspace.parent
        self._close_fds = close_fds
        self._limits = limits
        self._protected_regions = protected_regions
        self._pid: Optional[int] = None
        self._exited = False
        self._branch_id: Optional[int] = None
        self._memory_regions: list[MemoryRegion] = []

    @property
    def pid(self) -> int:
        if self._pid is None:
            raise ProcessBranchError("Process not started")
        return self._pid

    @property
    def branch_id(self) -> Optional[int]:
        """The branch_id assigned by the process tracker, or ``None``."""
        return self._branch_id

    @property
    def alive(self) -> bool:
        if self._pid is None or self._exited:
            return False
        try:
            os.kill(self._pid, 0)
            return True
        except ProcessLookupError:
            return False

    def wait(self, timeout: Optional[float] = None) -> None:
        """Wait for the child process to exit.

        On completion (success or failure), restores write access to any
        protected memory regions so the parent can proceed.

        Args:
            timeout: Maximum seconds to wait (None = wait forever).

        Raises:
            ProcessBranchError: If the child exited with non-zero status.
            TimeoutError: If the child doesn't exit within timeout.
        """
        exit_code = self._wait_raw(timeout)
        self._restore_memory_regions()
        self._tracker_cleanup()
        if exit_code != 0:
            raise ProcessBranchError(
                f"Child {self._pid} exited with status {exit_code}"
            )

    def _wait_raw(self, timeout: Optional[float] = None) -> int:
        """Wait for the child and return the raw exit code."""
        if timeout is None:
            _, status = os.waitpid(self._pid, 0)
            self._exited = True
            return os.waitstatus_to_exitcode(status)

        deadline = time.monotonic() + timeout
        while True:
            try:
                result, status = os.waitpid(self._pid, os.WNOHANG)
                if result != 0:
                    self._exited = True
                    return os.waitstatus_to_exitcode(status)
            except ChildProcessError:
                self._exited = True
                return -1

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Process {self._pid} did not exit within {timeout}s"
                )
            time.sleep(min(0.05, remaining))

    def abort(self, timeout: float = 5.0) -> None:
        """Abort the child and all its descendants.

        Restores write access to protected memory regions, then kills the
        child.  Uses the BPF LSM process tracker for reliable descendant
        termination.  Escalates SIGTERM -> SIGKILL after timeout.
        """
        if self._pid is None or self._exited:
            return

        self._restore_memory_regions()

        # Kill via BPF tracker (catches escaped descendants)
        if self._branch_id is not None:
            self._tracker.kill_branch(self._branch_id)

        # SIGTERM the process group
        try:
            os.killpg(self._pid, signal.SIGTERM)
        except ProcessLookupError:
            self._exited = True
            self._reap()
            self._tracker_cleanup()
            return

        # Poll for exit
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result, _ = os.waitpid(self._pid, os.WNOHANG)
                if result != 0:
                    self._exited = True
                    self._tracker_cleanup()
                    return
            except ChildProcessError:
                self._exited = True
                self._tracker_cleanup()
                return
            time.sleep(0.05)

        # Escalate to SIGKILL
        try:
            os.killpg(self._pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        self._reap()
        self._exited = True
        self._tracker_cleanup()

    def _tracker_cleanup(self) -> None:
        """Clean up process tracker state for this branch."""
        if self._branch_id is not None:
            self._tracker.cleanup(self._branch_id)
            self._branch_id = None

    def _restore_memory_regions(self) -> None:
        """Restore write access to all protected memory regions."""
        for region in self._memory_regions:
            _mprotect(region.addr, region.size, region.original_prot)
        self._memory_regions.clear()

    def _reap(self) -> None:
        """Reap the child process (non-blocking, best-effort)."""
        try:
            os.waitpid(self._pid, os.WNOHANG)
        except ChildProcessError:
            pass

    def __enter__(self) -> "BranchContext":
        # Require BPF process tracker before forking — fail fast
        self._tracker = BpfProcessTracker.get()

        try:
            pid = os.fork()
        except OSError as e:
            raise ForkError(f"fork() failed: {e}") from e

        if pid == 0:
            # === Child process ===
            try:
                os.setpgid(0, 0)

                # Landlock: confine to branch (read+write branch, read
                # system paths, block sibling branches and workspace root)
                confine_to_branch(self._workspace, self._mount_root)

                # Apply resource limits via setrlimit (inherited by children)
                if self._limits is not None:
                    _apply_limits(self._limits)

                if self._close_fds:
                    max_fd = os.sysconf("SC_OPEN_MAX")
                    os.closerange(3, max_fd)

                os.chdir(self._workspace)
                self._target(self._workspace)
                os._exit(0)
            except SystemExit as e:
                os._exit(e.code if isinstance(e.code, int) else 1)
            except BaseException:
                os._exit(1)
        else:
            # === Parent process ===
            self._pid = pid

            # Race-free process group setup
            try:
                os.setpgid(pid, pid)
            except OSError:
                pass  # Child may have already set it

            # Register with BPF process tracker
            self._branch_id = self._tracker.register(pid)

            # Protect registered memory regions in the parent
            if self._protected_regions:
                for addr, size in self._protected_regions:
                    region = MemoryRegion(
                        addr=addr,
                        size=size,
                        original_prot=PROT_READ | PROT_WRITE,
                    )
                    _mprotect(addr, size, PROT_READ)
                    self._memory_regions.append(region)

            return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.abort()

        # If abort() was a no-op (child already waited/exited), ensure
        # regions are still restored.
        self._restore_memory_regions()

        return False

    @staticmethod
    @contextlib.contextmanager
    def create(
        targets: Sequence[Callable[[Path], None]],
        workspaces: Sequence[Path],
        *,
        mount_root: Path | None = None,
        close_fds: bool = False,
        limits: ResourceLimits | None = None,
        protected_regions: Sequence[tuple[int, int]] | None = None,
    ) -> Iterator[list["BranchContext"]]:
        """Create N branch contexts.

        Returns a context manager that cleans up all children on exit.

        Args:
            targets: Sequence of callables, each receiving a workspace Path.
            workspaces: Sequence of workspace Paths, one per target.
            mount_root: Filesystem mount root for Landlock confinement.
            close_fds: Close inherited fds in children.
            limits: Optional resource limits applied via setrlimit(2).
            protected_regions: Optional list of (addr, size) tuples to
                mark read-only in the parent after each fork.

        Yields:
            List of entered BranchContext instances (already forked).
        """
        if len(targets) != len(workspaces):
            raise ValueError("targets and workspaces must have the same length")

        contexts: list[BranchContext] = []
        try:
            for target, workspace in zip(targets, workspaces):
                ctx = BranchContext(
                    target, workspace, mount_root=mount_root,
                    close_fds=close_fds, limits=limits,
                    protected_regions=protected_regions,
                )
                ctx.__enter__()
                contexts.append(ctx)
            yield contexts
        finally:
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)
