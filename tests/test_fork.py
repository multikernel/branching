# SPDX-License-Identifier: Apache-2.0
"""Tests verifying actual fork behavior via run_in_process."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from branching.process.runner import run_in_process


# Mock BPF tracker and Landlock for all tests that fork.
@pytest.fixture(autouse=True)
def _mock_bpf_and_landlock():
    mock_tracker = MagicMock()
    mock_tracker.register.return_value = 1
    with patch(
        "branching.process.context.BpfProcessTracker.get",
        return_value=mock_tracker,
    ), patch(
        "branching.process.context.confine_to_branch",
    ):
        yield


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


def test_fork_without_resource_limits():
    """run_in_process works with limits=None (fork without resource limits)."""
    def task(workspace):
        return 42

    with tempfile.TemporaryDirectory() as ws:
        ws_path = Path(ws)
        result = run_in_process(task, (ws_path,), workspace=ws_path, limits=None)
        assert result == 42
