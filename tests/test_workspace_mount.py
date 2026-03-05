# SPDX-License-Identifier: Apache-2.0
"""Integration tests for Workspace.mount() — requires branchfs binary and FUSE."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from branching.core.workspace import Workspace, _find_branchfs
from branching.exceptions import MountError


# Skip all tests if branchfs binary not available or no FUSE
pytestmark = pytest.mark.skipif(
    shutil.which("branchfs") is None
    or not (Path("/dev/fuse").exists() and os.access("/dev/fuse", os.R_OK | os.W_OK)),
    reason="branchfs binary not in PATH or /dev/fuse not accessible",
)


@pytest.fixture
def base_dir():
    d = Path(tempfile.mkdtemp(prefix="branchfs_test_base_"))
    (d / "file1.txt").write_text("base content")
    (d / "subdir").mkdir()
    (d / "subdir" / "nested.txt").write_text("nested")
    yield d
    shutil.rmtree(d, ignore_errors=True)


class TestWorkspaceMount:
    def test_mount_and_close(self, base_dir):
        ws = Workspace.mount(base_dir)
        try:
            assert ws.path.exists()
            assert (ws.path / ".branchfs_ctl").exists()
            assert (ws.path / "file1.txt").read_text() == "base content"
            assert ws.fstype == "fuse.branchfs"
        finally:
            ws.close()

    def test_context_manager(self, base_dir):
        with Workspace.mount(base_dir) as ws:
            assert (ws.path / "file1.txt").read_text() == "base content"
        # After exit, mount should be cleaned up
        assert not (ws.path / ".branchfs_ctl").exists()

    def test_branch_commit(self, base_dir):
        with Workspace.mount(base_dir) as ws:
            with ws.branch("test_branch") as b:
                (b.path / "new_file.txt").write_text("from branch")
            # auto-committed
            assert (base_dir / "new_file.txt").read_text() == "from branch"

    def test_branch_abort(self, base_dir):
        with Workspace.mount(base_dir) as ws:
            try:
                with ws.branch("fail_branch") as b:
                    (b.path / "bad_file.txt").write_text("should vanish")
                    raise ValueError("simulated failure")
            except ValueError:
                pass
            # auto-aborted — file should not be in base
            assert not (base_dir / "bad_file.txt").exists()

    def test_explicit_mountpoint_and_storage(self, base_dir, tmp_path):
        mnt = tmp_path / "mnt"
        stor = tmp_path / "stor"
        with Workspace.mount(base_dir, mountpoint=mnt, storage=stor) as ws:
            assert ws.path == mnt.resolve()
            assert (ws.path / "file1.txt").exists()

    def test_mount_nonexistent_base_raises(self):
        with pytest.raises(MountError, match="does not exist"):
            Workspace.mount("/nonexistent/path/abc123")

    def test_mount_bad_binary_raises(self, base_dir):
        with pytest.raises(MountError, match="not found"):
            Workspace.mount(base_dir, branchfs_bin="/no/such/binary")
