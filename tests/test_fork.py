# SPDX-License-Identifier: Apache-2.0
"""Tests verifying actual fork behavior via run_in_process."""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

from branching.process.runner import run_in_process


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
def test_fork_runs_in_child_process():
    """Task runs in a child process with a different PID."""
    parent_pid = os.getpid()

    def task(workspace):
        pid_file = workspace / "child_pid"
        pid_file.write_text(str(os.getpid()))
        return True

    with tempfile.TemporaryDirectory() as ws:
        ws_path = Path(ws)
        run_in_process(task, (ws_path,), workspace=ws_path)
        child_pid = int((ws_path / "child_pid").read_text())
        assert child_pid != parent_pid


@needs_userns
def test_fork_inherits_parent_memory():
    """Forked child inherits parent's global state via COW."""
    import test_fork as _self_module

    _self_module._inherited_value = "hello from parent"

    def task(workspace):
        import test_fork as mod
        val = getattr(mod, "_inherited_value", None)
        (workspace / "inherited").write_text(str(val))
        return True

    with tempfile.TemporaryDirectory() as ws:
        ws_path = Path(ws)
        run_in_process(task, (ws_path,), workspace=ws_path)
        result = (ws_path / "inherited").read_text()
        assert result == "hello from parent"

    # Clean up
    del _self_module._inherited_value


@needs_userns
def test_fork_without_resource_limits():
    """run_in_process works with limits=None (fork without cgroup)."""
    def task(workspace):
        return 42

    with tempfile.TemporaryDirectory() as ws:
        ws_path = Path(ws)
        result = run_in_process(task, (ws_path,), workspace=ws_path, limits=None)
        assert result == 42
