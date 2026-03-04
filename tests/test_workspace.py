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


@patch("branching.core.workspace.detect_fs_for_mount")
def test_workspace_creation(mock_detect):
    mock_detect.return_value = (MockFSBackend, Path("/tmp/test_ws"))
    ws = Workspace("/tmp/test_ws")
    assert ws.path == Path("/tmp/test_ws").resolve()
    assert ws.fstype == "mockfs"


@patch("branching.core.workspace.detect_fs_for_mount")
def test_workspace_branch_mountpoint_generation(mock_detect):
    mock_detect.return_value = (MockFSBackend, Path("/tmp/test_ws"))
    ws = Workspace("/tmp/test_ws")
    b = ws.branch("feat")
    # Branch path is mount root
    assert b.path == Path("/tmp/test_ws")


@patch("branching.core.workspace.detect_fs_for_mount")
def test_workspace_repr(mock_detect):
    mock_detect.return_value = (MockFSBackend, Path("/tmp/test_ws"))
    ws = Workspace("/tmp/test_ws")
    assert "Workspace" in repr(ws)
    assert "mockfs" in repr(ws)
