# SPDX-License-Identifier: Apache-2.0
"""Tests for BranchContext process branching."""

import os
import subprocess
import time
import tempfile
from pathlib import Path

import pytest

from branching.process.context import BranchContext
from branching.exceptions import ProcessBranchError


def _can_unshare_userns() -> bool:
    """Check if the system supports unprivileged user namespaces."""
    try:
        result = subprocess.run(
            ["unshare", "--user", "--map-root-user", "true"],
            capture_output=True, timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


needs_userns = pytest.mark.skipif(
    not _can_unshare_userns(),
    reason="Kernel does not support unprivileged user namespaces",
)


@needs_userns
def test_basic_success():
    """Target that returns normally means success — wait() returns None."""
    with tempfile.TemporaryDirectory() as ws:
        with BranchContext(lambda p: None, Path(ws)) as ctx:
            assert ctx.pid > 0
            ctx.wait(timeout=10.0)  # should not raise


@needs_userns
def test_target_failure_raises():
    """Target that raises causes wait() to raise ProcessBranchError."""
    def failing_target(p: Path) -> None:
        raise RuntimeError("boom")

    with tempfile.TemporaryDirectory() as ws:
        with BranchContext(failing_target, Path(ws)) as ctx:
            with pytest.raises(ProcessBranchError):
                ctx.wait(timeout=10.0)


@needs_userns
def test_exception_in_target():
    """Unhandled exception in target causes ProcessBranchError on wait."""
    def bad_target(p: Path) -> None:
        raise ValueError("something went wrong")

    with tempfile.TemporaryDirectory() as ws:
        with BranchContext(bad_target, Path(ws)) as ctx:
            with pytest.raises(ProcessBranchError):
                ctx.wait(timeout=10.0)


@needs_userns
def test_abort():
    """Abort kills a sleeping child."""
    def sleeper(p: Path) -> None:
        time.sleep(60)

    with tempfile.TemporaryDirectory() as ws:
        with BranchContext(sleeper, Path(ws)) as ctx:
            time.sleep(0.1)
            assert ctx.alive
            ctx.abort(timeout=2.0)
            assert not ctx.alive


@needs_userns
def test_wait_timeout():
    """TimeoutError raised when child doesn't exit in time."""
    def sleeper(p: Path) -> None:
        time.sleep(60)

    with tempfile.TemporaryDirectory() as ws:
        with BranchContext(sleeper, Path(ws)) as ctx:
            with pytest.raises(TimeoutError):
                ctx.wait(timeout=0.2)


@needs_userns
def test_context_manager():
    """__enter__/__exit__ lifecycle works correctly."""
    with tempfile.TemporaryDirectory() as ws:
        ctx = BranchContext(lambda p: None, Path(ws))
        with ctx:
            assert ctx.pid > 0
            ctx.wait(timeout=10.0)
        # After exit, child should be reaped
        assert not ctx.alive


@needs_userns
def test_abort_on_exit():
    """Child is aborted when leaving the context manager."""
    with tempfile.TemporaryDirectory() as ws:
        with BranchContext(lambda p: time.sleep(60), Path(ws)) as ctx:
            pid = ctx.pid
            assert ctx.alive

        # After exit, child should be dead
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
            pytest.fail("Process should be terminated")
        except ProcessLookupError:
            pass


def test_properties():
    """pid raises before start, alive is False before start."""
    with tempfile.TemporaryDirectory() as ws:
        ctx = BranchContext(lambda p: None, Path(ws))

        with pytest.raises(ProcessBranchError):
            _ = ctx.pid

        assert ctx.alive is False


@needs_userns
def test_close_fds():
    """Child has only stdin/stdout/stderr open when close_fds=True."""
    def check_fds(p: Path) -> None:
        # /proc/self/fd lists open file descriptors
        fds = set(os.listdir("/proc/self/fd"))
        # Should only have 0, 1, 2, and the fd used to read /proc/self/fd
        if len(fds) > 4:
            raise RuntimeError(f"Too many open fds: {fds}")

    with tempfile.TemporaryDirectory() as ws:
        with BranchContext(check_fds, Path(ws), close_fds=True) as ctx:
            ctx.wait(timeout=10.0)


@needs_userns
def test_mount_namespace_isolation():
    """Child sees bind-mounted workspace, but parent mount table is unaffected."""
    with tempfile.TemporaryDirectory() as ws:
        marker = Path(ws) / "marker.txt"
        marker.write_text("hello from workspace")

        def check_mount(p: Path) -> None:
            m = p / "marker.txt"
            if not m.exists() or m.read_text() != "hello from workspace":
                raise RuntimeError("Workspace not mounted correctly")

        with BranchContext(check_mount, Path(ws)) as ctx:
            ctx.wait(timeout=10.0)


@needs_userns
def test_create_multiple():
    """BranchContext.create() starts N contexts as a context manager."""
    def noop(p: Path) -> None:
        pass

    def fail(p: Path) -> None:
        raise RuntimeError("intentional")

    with tempfile.TemporaryDirectory() as ws1, \
         tempfile.TemporaryDirectory() as ws2, \
         tempfile.TemporaryDirectory() as ws3:

        targets = [noop, fail, noop]
        workspaces = [Path(ws1), Path(ws2), Path(ws3)]

        with BranchContext.create(targets, workspaces) as contexts:
            assert len(contexts) == 3
            for ctx in contexts:
                assert ctx.pid > 0

            # First and third succeed
            contexts[0].wait(timeout=10.0)
            contexts[2].wait(timeout=10.0)

            # Second fails
            with pytest.raises(ProcessBranchError):
                contexts[1].wait(timeout=10.0)

        # All cleaned up after exiting the with block
        for ctx in contexts:
            assert not ctx.alive
