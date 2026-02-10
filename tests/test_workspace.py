# SPDX-License-Identifier: Apache-2.0
"""Tests for Workspace."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from branching.core.base import FSBackend
from branching.core.workspace import Workspace


class MockFSBackend(FSBackend):
    @classmethod
    def fstype(cls) -> str:
        return "mockfs"

    @classmethod
    def create_branch(cls, name, mountpoint, parent_mount, parent_branch):
        pass

    @classmethod
    def commit(cls, mountpoint):
        pass

    @classmethod
    def abort(cls, mountpoint):
        pass


class SingleMountMockFSBackend(MockFSBackend):
    @classmethod
    def fstype(cls) -> str:
        return "fuse.mock"

    @classmethod
    def single_mount(cls) -> bool:
        return True


@patch("branching.core.workspace.detect_fs_for_mount")
def test_workspace_creation(mock_detect):
    mock_detect.return_value = MockFSBackend
    ws = Workspace("/tmp/test_ws")
    assert ws.path == Path("/tmp/test_ws").resolve()
    assert ws.fstype == "mockfs"


@patch("branching.core.workspace.detect_fs_for_mount")
def test_workspace_branch_mountpoint_generation(mock_detect):
    mock_detect.return_value = MockFSBackend
    ws = Workspace("/tmp/test_ws")
    b = ws.branch("feat")
    # Mount-per-branch: sibling directory
    assert b.path == ws.path.parent / f"{ws.path.name}_feat"


@patch("branching.core.workspace.detect_fs_for_mount")
def test_workspace_single_mount_branch(mock_detect):
    mock_detect.return_value = SingleMountMockFSBackend
    ws = Workspace("/tmp/test_ws")
    b = ws.branch("feat")
    # Single mount: same path
    assert b.path == ws.path


@patch("branching.core.workspace.detect_fs_for_mount")
def test_workspace_repr(mock_detect):
    mock_detect.return_value = MockFSBackend
    ws = Workspace("/tmp/test_ws")
    assert "Workspace" in repr(ws)
    assert "mockfs" in repr(ws)
