# SPDX-License-Identifier: Apache-2.0
"""BranchContext: userspace approximation of branch(BR_CREATE).

Each child is forked into its own process group with the branch
workspace as its working directory.  No sandboxing or resource limits
are applied — use sandlock (or similar) for confinement if needed.
"""

from __future__ import annotations

import contextlib
import os
import signal
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

from ..exceptions import ForkError, MemoryProtectError, ProcessBranchError
from ..memory._mprotect import (
    MemoryRegion,
    PROT_READ,
    PROT_WRITE,
    mprotect as _mprotect,
)


class BranchContext:
    """Userspace approximation of branch(BR_CREATE).

    Each child is forked into its own process group with the branch
    workspace as its working directory.  Memory protection via
    mprotect(2) enforces copy-on-write invariants.

    For sandboxing or resource limits, combine with sandlock.

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
        protected_regions: Sequence[tuple[int, int]] | None = None,
    ):
        """
        Args:
            target: Callable receiving workspace path. Return normally for
                    success, raise for failure.
            workspace: Branch workspace path (BranchFS virtual path).
            mount_root: Filesystem mount root (kept for API compatibility
                with callers that pass it through).
            close_fds: Close inherited fds (3+) in child.
            protected_regions: Optional list of (addr, size) tuples.
                After fork, these regions are marked read-only in the
                parent via mprotect(2) to enforce the branch invariant.
        """
        self._target = target
        self._workspace = workspace
        self._mount_root = mount_root or workspace.parent
        self._close_fds = close_fds
        self._protected_regions = protected_regions
        self._pid: Optional[int] = None
        self._exited = False
        self._memory_regions: list[MemoryRegion] = []

    @property
    def pid(self) -> int:
        if self._pid is None:
            raise ProcessBranchError("Process not started")
        return self._pid

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
        child process group.  Escalates SIGTERM -> SIGKILL after timeout.
        """
        if self._pid is None or self._exited:
            return

        self._restore_memory_regions()

        # SIGTERM the process group
        try:
            os.killpg(self._pid, signal.SIGTERM)
        except ProcessLookupError:
            self._exited = True
            self._reap()
            return

        # Poll for exit
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                result, _ = os.waitpid(self._pid, os.WNOHANG)
                if result != 0:
                    self._exited = True
                    return
            except ChildProcessError:
                self._exited = True
                return
            time.sleep(0.05)

        # Escalate to SIGKILL
        try:
            os.killpg(self._pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        self._reap()
        self._exited = True

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
        try:
            pid = os.fork()
        except OSError as e:
            raise ForkError(f"fork() failed: {e}") from e

        if pid == 0:
            # === Child process ===
            try:
                os.setpgid(0, 0)

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
        protected_regions: Sequence[tuple[int, int]] | None = None,
    ) -> Iterator[list["BranchContext"]]:
        """Create N branch contexts.

        Returns a context manager that cleans up all children on exit.

        Args:
            targets: Sequence of callables, each receiving a workspace Path.
            workspaces: Sequence of workspace Paths, one per target.
            mount_root: Filesystem mount root.
            close_fds: Close inherited fds in children.
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
                    close_fds=close_fds,
                    protected_regions=protected_regions,
                )
                ctx.__enter__()
                contexts.append(ctx)
            yield contexts
        finally:
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)
