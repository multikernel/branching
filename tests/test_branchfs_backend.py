# SPDX-License-Identifier: Apache-2.0
"""Tests for BranchFS FUSE backend.

Integration tests require a live branchfs mount.
Unit tests mock the ioctl layer.
"""

import ctypes
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
import pytest

from branching.fs.branchfs import BranchFSBackend, FS_IOC_BRANCH_CREATE, FS_IOC_BRANCH_COMMIT, FS_IOC_BRANCH_ABORT


def test_fstype():
    assert BranchFSBackend.fstype() == "fuse.branchfs"


def test_single_mount():
    assert BranchFSBackend.single_mount() is True


@patch("branching.fs.branchfs.os.open", return_value=3)
@patch("branching.fs.branchfs.os.close")
@patch("branching.fs.branchfs.fcntl.ioctl")
def test_create_branch(mock_ioctl, mock_close, mock_open):
    # Simulate daemon writing branch name into the 128-byte buffer
    def fill_buf(fd, cmd, buf):
        ctypes.memmove(buf, b"test-branch-id\x00", 15)

    mock_ioctl.side_effect = fill_buf

    result = BranchFSBackend.create_branch(
        "test", Path("/mnt/ws"), Path("/mnt/ws"), "/main"
    )
    mock_open.assert_called_once()
    mock_ioctl.assert_called_once_with(3, FS_IOC_BRANCH_CREATE, ANY)
    mock_close.assert_called_once_with(3)
    assert result == Path("/mnt/ws/@test-branch-id")


@patch("branching.fs.branchfs.os.open", return_value=3)
@patch("branching.fs.branchfs.os.close")
@patch("branching.fs.branchfs.fcntl.ioctl")
def test_commit(mock_ioctl, mock_close, mock_open):
    BranchFSBackend.commit(Path("/mnt/ws"))
    mock_ioctl.assert_called_once_with(3, FS_IOC_BRANCH_COMMIT, 0)


@patch("branching.fs.branchfs.os.open", return_value=3)
@patch("branching.fs.branchfs.os.close")
@patch("branching.fs.branchfs.fcntl.ioctl")
def test_abort(mock_ioctl, mock_close, mock_open):
    BranchFSBackend.abort(Path("/mnt/ws"))
    mock_ioctl.assert_called_once_with(3, FS_IOC_BRANCH_ABORT, 0)


@patch("branching.fs.branchfs.os.open", side_effect=OSError(2, "No such file"))
def test_create_branch_no_ctl_file(mock_open):
    with pytest.raises(Exception):  # MountError wraps OSError
        BranchFSBackend.create_branch(
            "test", Path("/mnt/ws"), Path("/mnt/ws"), "/main"
        )


@patch("branching.fs.branchfs.os.open", return_value=3)
@patch("branching.fs.branchfs.os.close")
@patch("branching.fs.branchfs.fcntl.ioctl", side_effect=OSError(5, "I/O error"))
def test_commit_ioctl_failure(mock_ioctl, mock_close, mock_open):
    from branching.exceptions import CommitError
    with pytest.raises(CommitError):
        BranchFSBackend.commit(Path("/mnt/ws"))
