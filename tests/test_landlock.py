# SPDX-License-Identifier: Apache-2.0
"""Tests for Landlock filesystem confinement."""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from branching.process._landlock import (
    confine_to_branch,
    landlock_abi_version,
    _WRITE_ACCESS,
    _READ_ACCESS,
    LANDLOCK_ACCESS_FS_REFER,
    LANDLOCK_ACCESS_FS_TRUNCATE,
)
from branching.exceptions import ProcessBranchError


def test_abi_version_returns_int():
    """landlock_abi_version returns a non-negative integer."""
    ver = landlock_abi_version()
    assert isinstance(ver, int)
    assert ver >= 0


def test_confine_to_branch_raises_when_unavailable():
    """confine_to_branch raises ProcessBranchError if Landlock is unavailable."""
    with patch("branching.process._landlock.landlock_abi_version", return_value=0):
        with pytest.raises(ProcessBranchError, match="Landlock not available"):
            confine_to_branch(Path("/mnt/main/@uuid"), Path("/mnt/main"))


def test_confine_to_branch_calls_syscalls(tmp_path):
    """confine_to_branch creates ruleset, adds sibling rules + branch rule, restricts self."""
    branch = tmp_path / "workspace" / "@branch"
    mount_root = tmp_path / "workspace"
    branch.mkdir(parents=True)

    mock_libc = MagicMock()
    # create_ruleset returns fd 42, then add_rule returns 0 for each rule,
    # then restrict_self returns 0.  We don't know exact count of rules
    # (depends on fs layout), so use a generous default.
    mock_libc.syscall.return_value = 0
    # First syscall (create_ruleset) returns fd 42
    mock_libc.syscall.side_effect = None
    mock_libc.syscall.return_value = 42  # all syscalls return 42/0 uniformly
    mock_libc.prctl.return_value = 0

    # We can't easily test the exact syscalls without a real kernel,
    # so just verify it doesn't raise with mocked libc.
    with patch("branching.process._landlock.landlock_abi_version", return_value=3), \
         patch("branching.process._landlock._libc", mock_libc), \
         patch("branching.process._landlock._add_path_rule") as mock_add, \
         patch("branching.process._landlock._set_no_new_privs"), \
         patch("branching.process._landlock._restrict_self"), \
         patch("branching.process._landlock._create_ruleset", return_value=42), \
         patch("branching.process._landlock.os.close"):
        confine_to_branch(branch, mount_root)

    # Verify the branch path got a full-access rule (last call)
    last_call = mock_add.call_args_list[-1]
    assert last_call[0][2] == branch  # path arg
    full_access = _READ_ACCESS | _WRITE_ACCESS
    assert last_call[0][1] == full_access  # allowed_access arg


def test_write_access_mask_covers_all_mutating_ops():
    """The _WRITE_ACCESS mask includes all filesystem-mutating operations."""
    from branching.process._landlock import (
        LANDLOCK_ACCESS_FS_WRITE_FILE,
        LANDLOCK_ACCESS_FS_REMOVE_DIR,
        LANDLOCK_ACCESS_FS_REMOVE_FILE,
        LANDLOCK_ACCESS_FS_MAKE_CHAR,
        LANDLOCK_ACCESS_FS_MAKE_DIR,
        LANDLOCK_ACCESS_FS_MAKE_REG,
        LANDLOCK_ACCESS_FS_MAKE_SOCK,
        LANDLOCK_ACCESS_FS_MAKE_FIFO,
        LANDLOCK_ACCESS_FS_MAKE_BLOCK,
        LANDLOCK_ACCESS_FS_MAKE_SYM,
    )
    for flag in (
        LANDLOCK_ACCESS_FS_WRITE_FILE,
        LANDLOCK_ACCESS_FS_REMOVE_DIR,
        LANDLOCK_ACCESS_FS_REMOVE_FILE,
        LANDLOCK_ACCESS_FS_MAKE_CHAR,
        LANDLOCK_ACCESS_FS_MAKE_DIR,
        LANDLOCK_ACCESS_FS_MAKE_REG,
        LANDLOCK_ACCESS_FS_MAKE_SOCK,
        LANDLOCK_ACCESS_FS_MAKE_FIFO,
        LANDLOCK_ACCESS_FS_MAKE_BLOCK,
        LANDLOCK_ACCESS_FS_MAKE_SYM,
        LANDLOCK_ACCESS_FS_REFER,
        LANDLOCK_ACCESS_FS_TRUNCATE,
    ):
        assert _WRITE_ACCESS & flag == flag, f"Missing flag 0x{flag:x}"


def test_sibling_read_rules_skip_mount_path(tmp_path):
    """_add_sibling_read_rules adds rules for siblings, skips path to mount_root."""
    from branching.process._landlock import _add_sibling_read_rules

    # Create a directory structure:
    #   tmp_path/a/  tmp_path/b/  tmp_path/workspace/
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ws = tmp_path / "workspace"
    ws.mkdir()

    added_paths = []

    def mock_add_rule(fd, access, path):
        added_paths.append(path)

    with patch("branching.process._landlock._add_path_rule", side_effect=mock_add_rule):
        _add_sibling_read_rules(42, 0x0F, ws)

    # "a" and "b" should have rules, "workspace" should NOT
    added_names = {p.name for p in added_paths}
    assert "a" in added_names
    assert "b" in added_names
    assert "workspace" not in added_names
