# SPDX-License-Identifier: Apache-2.0
"""Tests for Branch context manager."""

from pathlib import Path
from unittest.mock import MagicMock, call
import pytest

from branching.core.base import FSBackend
from branching.core.branch import Branch


class MockFSBackend(FSBackend):
    """Mock backend for testing."""

    create_branch_calls = []
    commit_calls = []
    abort_calls = []

    @classmethod
    def reset(cls):
        cls.create_branch_calls = []
        cls.commit_calls = []
        cls.abort_calls = []

    @classmethod
    def fstype(cls) -> str:
        return "mock"

    @classmethod
    def create_branch(cls, name, mountpoint, parent_mount, parent_branch):
        cls.create_branch_calls.append((name, mountpoint, parent_mount, parent_branch))

    @classmethod
    def commit(cls, mountpoint):
        cls.commit_calls.append(mountpoint)

    @classmethod
    def abort(cls, mountpoint):
        cls.abort_calls.append(mountpoint)


class MockParent:
    """Mock parent workspace/branch."""

    def __init__(self, path: Path):
        self._path = path

    @property
    def path(self) -> Path:
        return self._path


@pytest.fixture(autouse=True)
def reset_mock():
    MockFSBackend.reset()


def test_branch_properties():
    parent = MockParent(Path("/mnt/main"))
    b = Branch(MockFSBackend, "feat", Path("/mnt/main_feat"), parent, "/main")
    assert b.name == "feat"
    assert b.path == Path("/mnt/main_feat")
    assert b.branch_path == "/main/feat"


def test_branch_path_nested():
    parent = MockParent(Path("/mnt/main_l1"))
    b = Branch(MockFSBackend, "l2", Path("/mnt/main_l1_l2"), parent, "/main/l1")
    assert b.branch_path == "/main/l1/l2"


def test_context_manager_commit_on_success():
    parent = MockParent(Path("/mnt/main"))
    b = Branch(MockFSBackend, "test", Path("/mnt/main_test"), parent, "/main")

    with b:
        pass  # Clean exit

    assert len(MockFSBackend.create_branch_calls) == 1
    assert len(MockFSBackend.commit_calls) == 1
    assert len(MockFSBackend.abort_calls) == 0


def test_context_manager_abort_on_error():
    parent = MockParent(Path("/mnt/main"))
    b = Branch(MockFSBackend, "test", Path("/mnt/main_test"), parent, "/main")

    with pytest.raises(ValueError):
        with b:
            raise ValueError("oops")

    assert len(MockFSBackend.create_branch_calls) == 1
    assert len(MockFSBackend.commit_calls) == 0
    assert len(MockFSBackend.abort_calls) == 1


def test_context_manager_no_action():
    parent = MockParent(Path("/mnt/main"))
    b = Branch(
        MockFSBackend, "test", Path("/mnt/main_test"), parent, "/main",
        on_success=None, on_error=None,
    )

    with b:
        pass

    assert len(MockFSBackend.commit_calls) == 0
    assert len(MockFSBackend.abort_calls) == 0


def test_manual_commit():
    parent = MockParent(Path("/mnt/main"))
    b = Branch(
        MockFSBackend, "test", Path("/mnt/main_test"), parent, "/main",
        on_success=None,
    )

    with b:
        b.commit()

    # Commit called once (manual), not twice (auto)
    assert len(MockFSBackend.commit_calls) == 1


def test_manual_abort():
    parent = MockParent(Path("/mnt/main"))
    b = Branch(
        MockFSBackend, "test", Path("/mnt/main_test"), parent, "/main",
        on_success=None, on_error=None,
    )

    with b:
        b.abort()

    assert len(MockFSBackend.abort_calls) == 1
    assert len(MockFSBackend.commit_calls) == 0


def test_child_branch():
    parent = MockParent(Path("/mnt/main"))
    b = Branch(MockFSBackend, "l1", Path("/mnt/main_l1"), parent, "/main")
    child = b.branch("l2")
    assert child.name == "l2"
    assert child.branch_path == "/main/l1/l2"
    assert child.path == Path("/mnt/main_l1_l2")


def test_repr():
    parent = MockParent(Path("/mnt/main"))
    b = Branch(MockFSBackend, "feat", Path("/mnt/main_feat"), parent, "/main")
    assert "Branch" in repr(b)
    assert "feat" in repr(b)
