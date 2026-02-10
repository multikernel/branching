# SPDX-License-Identifier: Apache-2.0
"""BranchContext: userspace approximation of branch(BR_CREATE).

Each child runs in its own user + mount namespace with the branch
workspace bind-mounted. Isolation failures raise, not silently degrade.
"""

import contextlib
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

from ..exceptions import ForkError, ProcessBranchError
from . import _cgroup
from ._namespace import setup_user_ns, bind_mount


class BranchContext:
    """Userspace approximation of branch(BR_CREATE).

    Each child runs in its own user + mount namespace with the branch
    workspace bind-mounted. Isolation failures raise, not silently degrade.

    The target callable follows Python conventions: return normally for
    success, raise an exception for failure.
    """

    def __init__(
        self,
        target: Callable[[Path], None],
        workspace: Path,
        *,
        isolate: bool = False,
        close_fds: bool = False,
    ):
        """
        Args:
            target: Callable receiving workspace path. Return normally for
                    success, raise for failure.
            workspace: Branch workspace path to bind-mount.
            isolate: BR_ISOLATE — separate user ns per child (always-on
                     since each child needs its own user ns for bind-mount).
            close_fds: BR_CLOSE_FDS — close inherited fds (3+) in child.
        """
        self._target = target
        self._workspace = workspace
        self._isolate = isolate
        self._close_fds = close_fds
        self._pid: Optional[int] = None
        self._exited = False
        self._cgroup_scope: Optional[Path] = None

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

        Args:
            timeout: Maximum seconds to wait (None = wait forever).

        Raises:
            ProcessBranchError: If the child exited with non-zero status.
            TimeoutError: If the child doesn't exit within timeout.
        """
        exit_code = self._wait_raw(timeout)
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

        Uses both cgroup kill (catches escapees) and killpg (POSIX standard).
        Escalates SIGTERM -> SIGKILL after timeout.
        """
        if self._pid is None or self._exited:
            return

        # Cgroup kill — catches descendants that escaped the process group
        if self._cgroup_scope is not None:
            _cgroup.kill_scope(self._cgroup_scope)

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

    def _reap(self) -> None:
        """Reap the child process (non-blocking, best-effort)."""
        try:
            os.waitpid(self._pid, os.WNOHANG)
        except ChildProcessError:
            pass

    def __enter__(self) -> "BranchContext":
        # Create cgroup scope (best-effort — don't fail if cgroups unavailable)
        try:
            self._cgroup_scope = _cgroup.create_scope(f"{os.getpid()}")
        except OSError:
            self._cgroup_scope = None

        # Create a private mount directory for the child
        self._private_dir = tempfile.mkdtemp(prefix="branchctx-")

        try:
            pid = os.fork()
        except OSError as e:
            raise ForkError(f"fork() failed: {e}") from e

        if pid == 0:
            # === Child process ===
            try:
                os.setpgid(0, 0)

                # User namespace + mount namespace (mandatory)
                setup_user_ns()

                # Bind-mount the workspace into our private dir
                bind_mount(self._workspace, self._private_dir)

                if self._close_fds:
                    max_fd = os.sysconf("SC_OPEN_MAX")
                    os.closerange(3, max_fd)

                os.chdir(self._private_dir)
                self._target(Path(self._private_dir))
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

            # Add to cgroup scope
            if self._cgroup_scope is not None:
                try:
                    _cgroup.add_pid(self._cgroup_scope, pid)
                except OSError:
                    pass  # Best-effort

            return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.abort()

        # Clean up private mount dir (best-effort)
        try:
            os.rmdir(self._private_dir)
        except OSError:
            pass

        return False

    @staticmethod
    @contextlib.contextmanager
    def create(
        targets: Sequence[Callable[[Path], None]],
        workspaces: Sequence[Path],
        *,
        isolate: bool = False,
        close_fds: bool = False,
    ) -> Iterator[list["BranchContext"]]:
        """Create N branch contexts, mirroring branch(BR_CREATE, n_branches=N).

        Returns a context manager that cleans up all children on exit.

        Args:
            targets: Sequence of callables, each receiving a workspace Path.
            workspaces: Sequence of workspace Paths, one per target.
            isolate: BR_ISOLATE — separate user ns per child.
            close_fds: BR_CLOSE_FDS — close inherited fds in children.

        Yields:
            List of entered BranchContext instances (already forked).
        """
        if len(targets) != len(workspaces):
            raise ValueError("targets and workspaces must have the same length")

        contexts: list[BranchContext] = []
        try:
            for target, workspace in zip(targets, workspaces):
                ctx = BranchContext(
                    target, workspace, isolate=isolate, close_fds=close_fds
                )
                ctx.__enter__()
                contexts.append(ctx)
            yield contexts
        finally:
            for ctx in reversed(contexts):
                ctx.__exit__(None, None, None)
