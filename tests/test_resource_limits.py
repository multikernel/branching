# SPDX-License-Identifier: Apache-2.0
"""Tests for resource limits: dataclass, parse_memory_size, set_limits, and
pattern integration."""

import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from branching.process.limits import ResourceLimits, parse_memory_size


# ---------------------------------------------------------------
# ResourceLimits dataclass
# ---------------------------------------------------------------

class TestResourceLimits:
    def test_defaults_are_none(self):
        rl = ResourceLimits()
        assert rl.memory is None
        assert rl.cpu is None
        assert rl.memory_high is None
        assert rl.oom_group is False

    def test_memory_only(self):
        rl = ResourceLimits(memory=512 * 1024 * 1024)
        assert rl.memory == 512 * 1024 * 1024
        assert rl.cpu is None

    def test_cpu_only(self):
        rl = ResourceLimits(cpu=0.5)
        assert rl.cpu == 0.5
        assert rl.memory is None

    def test_both(self):
        rl = ResourceLimits(memory=1024, cpu=1.0)
        assert rl.memory == 1024
        assert rl.cpu == 1.0

    def test_frozen(self):
        rl = ResourceLimits(memory=1024)
        with pytest.raises(AttributeError):
            rl.memory = 2048  # type: ignore[misc]

    def test_equality(self):
        a = ResourceLimits(memory=100, cpu=0.5)
        b = ResourceLimits(memory=100, cpu=0.5)
        assert a == b

    def test_inequality(self):
        a = ResourceLimits(memory=100)
        b = ResourceLimits(memory=200)
        assert a != b

    def test_memory_high(self):
        rl = ResourceLimits(memory_high=256 * 1024 * 1024)
        assert rl.memory_high == 256 * 1024 * 1024

    def test_oom_group(self):
        rl = ResourceLimits(oom_group=True)
        assert rl.oom_group is True

    def test_all_fields(self):
        rl = ResourceLimits(
            memory=1024, cpu=0.5, memory_high=512, oom_group=True,
        )
        assert rl.memory == 1024
        assert rl.cpu == 0.5
        assert rl.memory_high == 512
        assert rl.oom_group is True


# ---------------------------------------------------------------
# parse_memory_size
# ---------------------------------------------------------------

class TestParseMemorySize:
    def test_plain_bytes(self):
        assert parse_memory_size("1024") == 1024

    def test_kilobytes(self):
        assert parse_memory_size("100K") == 100 * 1024

    def test_megabytes(self):
        assert parse_memory_size("512M") == 512 * 1024 ** 2

    def test_gigabytes(self):
        assert parse_memory_size("1G") == 1024 ** 3

    def test_terabytes(self):
        assert parse_memory_size("2T") == 2 * 1024 ** 4

    def test_case_insensitive(self):
        assert parse_memory_size("512m") == 512 * 1024 ** 2
        assert parse_memory_size("1g") == 1024 ** 3

    def test_with_spaces(self):
        assert parse_memory_size("  512 M ") == 512 * 1024 ** 2

    def test_fractional(self):
        assert parse_memory_size("1.5G") == int(1.5 * 1024 ** 3)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="invalid memory size"):
            parse_memory_size("abc")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="invalid memory size"):
            parse_memory_size("")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="invalid memory size"):
            parse_memory_size("-512M")


# ---------------------------------------------------------------
# set_limits (mocked filesystem)
# ---------------------------------------------------------------

class TestSetLimits:
    def test_memory_limit(self, tmp_path):
        from branching.process._cgroup import set_limits
        mem_file = tmp_path / "memory.max"
        mem_file.write_text("max")

        limits = ResourceLimits(memory=512 * 1024 * 1024)
        set_limits(tmp_path, limits)

        assert mem_file.read_text() == str(512 * 1024 * 1024)

    def test_cpu_limit(self, tmp_path):
        from branching.process._cgroup import set_limits
        cpu_file = tmp_path / "cpu.max"
        cpu_file.write_text("max 100000")

        limits = ResourceLimits(cpu=0.5)
        set_limits(tmp_path, limits)

        assert cpu_file.read_text() == "50000 100000"

    def test_both_limits(self, tmp_path):
        from branching.process._cgroup import set_limits
        mem_file = tmp_path / "memory.max"
        cpu_file = tmp_path / "cpu.max"
        mem_file.write_text("max")
        cpu_file.write_text("max 100000")

        limits = ResourceLimits(memory=1024 ** 3, cpu=0.25)
        set_limits(tmp_path, limits)

        assert mem_file.read_text() == str(1024 ** 3)
        assert cpu_file.read_text() == "25000 100000"

    def test_none_fields_skipped(self, tmp_path):
        from branching.process._cgroup import set_limits
        # No files created, nothing written
        limits = ResourceLimits()
        set_limits(tmp_path, limits)

        assert not (tmp_path / "memory.max").exists()
        assert not (tmp_path / "cpu.max").exists()

    def test_missing_cgroup_files_ignored(self):
        """set_limits is best-effort — missing files don't raise."""
        from branching.process._cgroup import set_limits
        # Non-existent path
        scope = Path("/nonexistent/cgroup/scope")
        limits = ResourceLimits(memory=1024, cpu=0.5)
        # Should not raise
        set_limits(scope, limits)

    def test_memory_high_limit(self, tmp_path):
        from branching.process._cgroup import set_limits
        mem_high_file = tmp_path / "memory.high"
        mem_high_file.write_text("max")

        limits = ResourceLimits(memory_high=256 * 1024 * 1024)
        set_limits(tmp_path, limits)

        assert mem_high_file.read_text() == str(256 * 1024 * 1024)

    def test_oom_group(self, tmp_path):
        from branching.process._cgroup import set_limits
        oom_file = tmp_path / "memory.oom.group"
        oom_file.write_text("0")

        limits = ResourceLimits(oom_group=True)
        set_limits(tmp_path, limits)

        assert oom_file.read_text() == "1"

    def test_oom_group_false_skipped(self, tmp_path):
        from branching.process._cgroup import set_limits
        limits = ResourceLimits(oom_group=False)
        set_limits(tmp_path, limits)

        assert not (tmp_path / "memory.oom.group").exists()


# ---------------------------------------------------------------
# _enable_subtree_controllers (mocked filesystem)
# ---------------------------------------------------------------

class TestEnableSubtreeControllers:
    def test_enables_available_controllers(self, tmp_path):
        from branching.process._cgroup import _enable_subtree_controllers
        (tmp_path / "cgroup.controllers").write_text("memory cpu io")
        (tmp_path / "cgroup.subtree_control").write_text("")

        _enable_subtree_controllers(tmp_path)

        assert (tmp_path / "cgroup.subtree_control").read_text() == "+memory +cpu"

    def test_only_enables_available(self, tmp_path):
        from branching.process._cgroup import _enable_subtree_controllers
        (tmp_path / "cgroup.controllers").write_text("cpu io")
        (tmp_path / "cgroup.subtree_control").write_text("")

        _enable_subtree_controllers(tmp_path)

        assert (tmp_path / "cgroup.subtree_control").read_text() == "+cpu"

    def test_no_matching_controllers(self, tmp_path):
        from branching.process._cgroup import _enable_subtree_controllers
        (tmp_path / "cgroup.controllers").write_text("io pids")
        (tmp_path / "cgroup.subtree_control").write_text("")

        _enable_subtree_controllers(tmp_path)

        # subtree_control should not be modified
        assert (tmp_path / "cgroup.subtree_control").read_text() == ""

    def test_missing_controllers_file(self, tmp_path):
        from branching.process._cgroup import _enable_subtree_controllers
        # No cgroup.controllers file — should not raise
        _enable_subtree_controllers(tmp_path)


# ---------------------------------------------------------------
# create_group (mocked filesystem)
# ---------------------------------------------------------------

class TestCreateGroup:
    def test_creates_group_directory(self, tmp_path):
        from branching.process._cgroup import create_group
        # Mock _own_cgroup to use tmp_path
        with patch("branching.process._cgroup._own_cgroup", return_value=tmp_path):
            group = create_group("test-group")

        assert group.is_dir()
        assert group.name == "branching-test-group.scope"
        assert group.parent == tmp_path

    def test_creates_under_parent(self, tmp_path):
        from branching.process._cgroup import create_group
        parent = tmp_path / "parent_cg"
        parent.mkdir()
        # Write controller files so subtree enable works
        (parent / "cgroup.controllers").write_text("memory cpu")
        (parent / "cgroup.subtree_control").write_text("")

        group = create_group("child", parent=parent)

        assert group.is_dir()
        assert group.parent == parent

    def test_applies_limits(self, tmp_path):
        from branching.process._cgroup import create_group
        with patch("branching.process._cgroup._own_cgroup", return_value=tmp_path):
            group = create_group(
                "limited",
                limits=ResourceLimits(memory=1024 * 1024),
            )

        assert (group / "memory.max").read_text() == str(1024 * 1024)

    def test_enables_subtree_control_on_self(self, tmp_path):
        from branching.process._cgroup import create_group
        with patch("branching.process._cgroup._own_cgroup", return_value=tmp_path):
            group = create_group("test")
            # Write controllers file to the created group
            (group / "cgroup.controllers").write_text("memory cpu")
            (group / "cgroup.subtree_control").write_text("")

            # Re-run to see the subtree enable effect
            group2 = create_group("test2")

        # The first group should exist
        assert group.is_dir()


# ---------------------------------------------------------------
# create_scope with parent
# ---------------------------------------------------------------

class TestCreateScopeWithParent:
    def test_default_no_parent(self, tmp_path):
        from branching.process._cgroup import create_scope
        with patch("branching.process._cgroup._own_cgroup", return_value=tmp_path):
            scope = create_scope("test-1234")

        assert scope.is_dir()
        assert scope.parent == tmp_path
        assert scope.name == "branching-test-1234.scope"

    def test_with_parent(self, tmp_path):
        from branching.process._cgroup import create_scope
        parent = tmp_path / "group_cg"
        parent.mkdir()
        (parent / "cgroup.controllers").write_text("memory cpu")
        (parent / "cgroup.subtree_control").write_text("")

        scope = create_scope("child-5678", parent=parent)

        assert scope.is_dir()
        assert scope.parent == parent
        assert scope.name == "branching-child-5678.scope"
        # subtree_control should have been enabled on parent
        assert "+memory +cpu" in (parent / "cgroup.subtree_control").read_text()


# ---------------------------------------------------------------
# kill_scope (recursive)
# ---------------------------------------------------------------

class TestKillScopeRecursive:
    def test_removes_empty_scope(self, tmp_path):
        from branching.process._cgroup import kill_scope
        scope = tmp_path / "branching-test.scope"
        scope.mkdir()

        kill_scope(scope)

        assert not scope.exists()

    def test_removes_nested_children(self, tmp_path):
        from branching.process._cgroup import kill_scope
        root = tmp_path / "branching-root.scope"
        root.mkdir()
        child1 = root / "branching-child1.scope"
        child1.mkdir()
        child2 = root / "branching-child2.scope"
        child2.mkdir()
        grandchild = child1 / "branching-gc.scope"
        grandchild.mkdir()

        kill_scope(root)

        assert not grandchild.exists()
        assert not child1.exists()
        assert not child2.exists()
        assert not root.exists()

    def test_ignores_nonexistent_scope(self):
        from branching.process._cgroup import kill_scope
        # Should not raise
        kill_scope(Path("/nonexistent/cgroup/scope"))

    def test_handles_files_in_scope(self, tmp_path):
        from branching.process._cgroup import kill_scope
        root = tmp_path / "branching-root.scope"
        root.mkdir()
        child = root / "branching-child.scope"
        child.mkdir()
        # Create a file (like cgroup.kill) in the child
        (child / "cgroup.kill").write_text("")

        # kill_scope should handle non-empty dirs gracefully (files present)
        # rmdir will fail if files present, but that's caught by OSError
        kill_scope(root)


# ---------------------------------------------------------------
# Pattern integration (resource_limits parameter acceptance)
# ---------------------------------------------------------------

class TestPatternResourceLimitsAcceptance:
    """Verify all patterns accept resource_limits without error."""

    def _make_workspace(self):
        from branching.core.base import FSBackend
        from branching.core.workspace import Workspace

        class MockFSBackend(FSBackend):
            @classmethod
            def fstype(cls):
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

        with patch("branching.core.workspace.detect_fs_for_mount") as mock:
            mock.return_value = MockFSBackend
            return Workspace("/tmp/test_ws")

    def test_speculate_accepts_resource_limits(self):
        from branching.agent.speculate import Speculate
        rl = ResourceLimits(memory=512 * 1024 * 1024)
        spec = Speculate([lambda p: True], resource_limits=rl)
        assert spec._resource_limits is rl

    def test_best_of_n_accepts_resource_limits(self):
        from branching.agent.patterns import BestOfN
        rl = ResourceLimits(cpu=0.5)
        bon = BestOfN(lambda p, i: (True, 1.0), n=2, resource_limits=rl)
        assert bon._resource_limits is rl

    def test_reflexion_accepts_resource_limits(self):
        from branching.agent.patterns import Reflexion
        rl = ResourceLimits(memory=1024)
        refl = Reflexion(lambda p, a, f: True, resource_limits=rl)
        assert refl._resource_limits is rl

    def test_tree_of_thoughts_accepts_resource_limits(self):
        from branching.agent.patterns import TreeOfThoughts
        rl = ResourceLimits(cpu=1.0)
        tot = TreeOfThoughts([lambda p: True], resource_limits=rl)
        assert tot._resource_limits is rl

    def test_beam_search_accepts_resource_limits(self):
        from branching.agent.patterns import BeamSearch
        rl = ResourceLimits(memory=2048, cpu=0.25)
        bs = BeamSearch(
            [lambda p: True],
            expand=lambda p, d: [],
            resource_limits=rl,
        )
        assert bs._resource_limits is rl

    def test_tournament_accepts_resource_limits(self):
        from branching.agent.patterns import Tournament
        rl = ResourceLimits(memory=4096)
        t = Tournament(
            lambda p, i: True, n=2,
            judge=lambda a, b: 0,
            resource_limits=rl,
        )
        assert t._resource_limits is rl


# ---------------------------------------------------------------
# Pattern group_limits acceptance
# ---------------------------------------------------------------

class TestPatternGroupLimitsAcceptance:
    """Verify all patterns accept and store group_limits."""

    def test_speculate_accepts_group_limits(self):
        from branching.agent.speculate import Speculate
        rl = ResourceLimits(memory=512 * 1024 * 1024)
        gl = ResourceLimits(memory=1024 * 1024 * 1024)
        spec = Speculate([lambda p: True], resource_limits=rl, group_limits=gl)
        assert spec._group_limits is gl

    def test_best_of_n_accepts_group_limits(self):
        from branching.agent.patterns import BestOfN
        gl = ResourceLimits(cpu=2.0)
        bon = BestOfN(lambda p, i: (True, 1.0), n=2, group_limits=gl)
        assert bon._group_limits is gl

    def test_reflexion_accepts_group_limits(self):
        from branching.agent.patterns import Reflexion
        gl = ResourceLimits(memory=2048)
        refl = Reflexion(lambda p, a, f: True, group_limits=gl)
        assert refl._group_limits is gl

    def test_tree_of_thoughts_accepts_group_limits(self):
        from branching.agent.patterns import TreeOfThoughts
        gl = ResourceLimits(cpu=4.0)
        tot = TreeOfThoughts([lambda p: True], group_limits=gl)
        assert tot._group_limits is gl

    def test_beam_search_accepts_group_limits(self):
        from branching.agent.patterns import BeamSearch
        gl = ResourceLimits(memory=8192)
        bs = BeamSearch(
            [lambda p: True],
            expand=lambda p, d: [],
            group_limits=gl,
        )
        assert bs._group_limits is gl

    def test_tournament_accepts_group_limits(self):
        from branching.agent.patterns import Tournament
        gl = ResourceLimits(memory=4096, cpu=2.0)
        t = Tournament(
            lambda p, i: True, n=2,
            judge=lambda a, b: 0,
            group_limits=gl,
        )
        assert t._group_limits is gl


# ---------------------------------------------------------------
# BranchContext accepts limits and parent_cgroup
# ---------------------------------------------------------------

class TestBranchContextLimits:
    def test_accepts_limits_kwarg(self):
        from branching.process.context import BranchContext
        rl = ResourceLimits(memory=1024)
        ctx = BranchContext(lambda p: None, workspace=Path("/tmp"), limits=rl)
        assert ctx._limits is rl

    def test_create_accepts_limits_kwarg(self):
        """BranchContext.create() signature accepts limits."""
        from branching.process.context import BranchContext
        import inspect
        sig = inspect.signature(BranchContext.create)
        assert "limits" in sig.parameters

    def test_accepts_parent_cgroup_kwarg(self):
        from branching.process.context import BranchContext
        parent = Path("/sys/fs/cgroup/test")
        ctx = BranchContext(
            lambda p: None, workspace=Path("/tmp"),
            parent_cgroup=parent,
        )
        assert ctx._parent_cgroup is parent

    def test_parent_cgroup_default_none(self):
        from branching.process.context import BranchContext
        ctx = BranchContext(lambda p: None, workspace=Path("/tmp"))
        assert ctx._parent_cgroup is None

    def test_create_accepts_parent_cgroup(self):
        from branching.process.context import BranchContext
        import inspect
        sig = inspect.signature(BranchContext.create)
        assert "parent_cgroup" in sig.parameters


# ---------------------------------------------------------------
# run_in_process accepts parent_cgroup
# ---------------------------------------------------------------

class TestRunInProcessParentCgroup:
    def test_signature_has_parent_cgroup(self):
        from branching.process.runner import run_in_process
        import inspect
        sig = inspect.signature(run_in_process)
        assert "parent_cgroup" in sig.parameters


# ---------------------------------------------------------------
# CLI resource limit parsing
# ---------------------------------------------------------------

class TestCLIResourceLimits:
    def test_parse_resource_limits_none(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit=None, cpu_limit=None,
            memory_high=None, oom_group=False,
        )
        assert _parse_resource_limits(args) is None

    def test_parse_resource_limits_memory(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit="512M", cpu_limit=None,
            memory_high=None, oom_group=False,
        )
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory == 512 * 1024 ** 2
        assert rl.cpu is None

    def test_parse_resource_limits_cpu(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit=None, cpu_limit=0.5,
            memory_high=None, oom_group=False,
        )
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory is None
        assert rl.cpu == 0.5

    def test_parse_resource_limits_both(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit="1G", cpu_limit=0.25,
            memory_high=None, oom_group=False,
        )
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory == 1024 ** 3
        assert rl.cpu == 0.25

    def test_parse_resource_limits_memory_high(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit=None, cpu_limit=None,
            memory_high="256M", oom_group=False,
        )
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory_high == 256 * 1024 ** 2

    def test_parse_resource_limits_oom_group(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit=None, cpu_limit=None,
            memory_high=None, oom_group=True,
        )
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.oom_group is True

    def test_parse_group_limits_none(self):
        import argparse
        from cli import _parse_group_limits
        args = argparse.Namespace(group_memory_limit=None, group_cpu_limit=None)
        assert _parse_group_limits(args) is None

    def test_parse_group_limits_memory(self):
        import argparse
        from cli import _parse_group_limits
        args = argparse.Namespace(group_memory_limit="2G", group_cpu_limit=None)
        gl = _parse_group_limits(args)
        assert gl is not None
        assert gl.memory == 2 * 1024 ** 3

    def test_parse_group_limits_cpu(self):
        import argparse
        from cli import _parse_group_limits
        args = argparse.Namespace(group_memory_limit=None, group_cpu_limit=4.0)
        gl = _parse_group_limits(args)
        assert gl is not None
        assert gl.cpu == 4.0

    def test_parser_has_resource_limit_flags(self):
        from cli import build_parser
        parser = build_parser()
        # Verify run subcommand accepts --memory-limit and --cpu-limit
        args = parser.parse_args(["run", "--memory-limit", "512M", "--cpu-limit", "0.5", "--", "echo", "hi"])
        assert args.memory_limit == "512M"
        assert args.cpu_limit == 0.5

    def test_parser_has_memory_high_flag(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "--memory-high", "256M", "--", "echo", "hi"])
        assert args.memory_high == "256M"

    def test_parser_has_oom_group_flag(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "--oom-group", "--", "echo", "hi"])
        assert args.oom_group is True

    def test_parser_speculate_has_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["speculate", "-c", "echo a", "--memory-limit", "256M"])
        assert args.memory_limit == "256M"

    def test_parser_speculate_has_group_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "speculate", "-c", "echo a",
            "--group-memory-limit", "2G", "--group-cpu-limit", "4.0",
        ])
        assert args.group_memory_limit == "2G"
        assert args.group_cpu_limit == 4.0

    def test_parser_best_of_n_has_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["best-of-n", "--cpu-limit", "0.75", "--", "echo", "hi"])
        assert args.cpu_limit == 0.75

    def test_parser_best_of_n_has_group_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "best-of-n", "--group-memory-limit", "1G", "--", "echo", "hi",
        ])
        assert args.group_memory_limit == "1G"

    def test_parser_reflexion_has_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["reflexion", "--memory-limit", "1G", "--", "echo", "hi"])
        assert args.memory_limit == "1G"

    def test_parser_reflexion_has_group_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "reflexion", "--group-cpu-limit", "2.0", "--", "echo", "hi",
        ])
        assert args.group_cpu_limit == 2.0


# ---------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# BranchContext.cgroup_scope property
# ---------------------------------------------------------------

class TestBranchContextCgroupScope:
    def test_cgroup_scope_none_before_enter(self):
        from branching.process.context import BranchContext
        ctx = BranchContext(lambda p: None, workspace=Path("/tmp"))
        assert ctx.cgroup_scope is None

    def test_property_exists(self):
        from branching.process.context import BranchContext
        assert isinstance(
            type(ctx := BranchContext(lambda p: None, workspace=Path("/tmp"))).cgroup_scope,
            property,
        )


# ---------------------------------------------------------------
# run_in_process scope_callback
# ---------------------------------------------------------------

class TestRunInProcessScopeCallback:
    def test_signature_has_scope_callback(self):
        from branching.process.runner import run_in_process
        import inspect
        sig = inspect.signature(run_in_process)
        assert "scope_callback" in sig.parameters

    def test_scope_callback_default_none(self):
        from branching.process.runner import run_in_process
        import inspect
        sig = inspect.signature(run_in_process)
        assert sig.parameters["scope_callback"].default is None


# ---------------------------------------------------------------
# Speculate._run_with_limits passes scope_callback
# ---------------------------------------------------------------

class TestSpeculateScopeCallback:
    def test_passes_scope_callback_to_run_in_process(self):
        from branching.agent.speculate import Speculate
        spec = Speculate(
            [lambda p: True],
            resource_limits=ResourceLimits(memory=1024),
        )
        with patch("branching.process.runner.run_in_process") as mock_rip:
            mock_rip.return_value = True
            callback = MagicMock()
            spec._run_with_limits(Path("/tmp"), 0, scope_callback=callback)
            assert mock_rip.call_args.kwargs["scope_callback"] is callback

    def test_scope_callback_none_by_default(self):
        from branching.agent.speculate import Speculate
        spec = Speculate(
            [lambda p: True],
            resource_limits=ResourceLimits(memory=1024),
        )
        with patch("branching.process.runner.run_in_process") as mock_rip:
            mock_rip.return_value = True
            spec._run_with_limits(Path("/tmp"), 0)
            assert mock_rip.call_args.kwargs["scope_callback"] is None


# ---------------------------------------------------------------
# Speculate cgroup kill on losers
# ---------------------------------------------------------------

class TestSpeculateCgroupKill:
    @staticmethod
    def _mock_workspace():
        """Create a mock workspace whose .branch() yields mock branch objects."""
        @contextmanager
        def mock_branch(name, on_success=None, on_error=None):
            b = MagicMock()
            b.path = Path(f"/tmp/ws/{name}")
            yield b
        ws = MagicMock()
        ws.branch = mock_branch
        return ws

    def test_kills_loser_scopes_on_first_wins(self):
        """first_wins + resource_limits: winner kills losers' cgroups."""
        from branching.agent.speculate import Speculate

        # Sync: both candidates register scopes before either returns.
        count = [0]
        lock = threading.Lock()
        both_registered = threading.Event()
        killed = []

        def mock_rip(fn, args, *, workspace, limits,
                     timeout=None, parent_cgroup=None, scope_callback=None):
            scope = workspace / ".scope"
            if scope_callback:
                scope_callback(scope)
            with lock:
                count[0] += 1
                if count[0] >= 2:
                    both_registered.set()
            both_registered.wait(timeout=5)
            # Candidate 0 wins immediately; candidate 1 stays alive briefly.
            if "speculate_0" in str(workspace):
                return True
            else:
                time.sleep(0.2)
                return False

        ws = self._mock_workspace()
        with patch("branching.process.runner.run_in_process", side_effect=mock_rip), \
             patch("branching.process._cgroup.kill_scope",
                   side_effect=lambda s: killed.append(s)):
            spec = Speculate(
                [lambda p: True, lambda p: True],
                first_wins=True,
                resource_limits=ResourceLimits(memory=1024),
            )
            spec._run(ws, None)

        loser_scope = Path("/tmp/ws/speculate_1/.scope")
        winner_scope = Path("/tmp/ws/speculate_0/.scope")
        assert loser_scope in killed
        assert winner_scope not in killed

    def test_no_kill_when_first_wins_false(self):
        """first_wins=False: all branches run to completion, no kill."""
        from branching.agent.speculate import Speculate

        killed = []

        def mock_rip(fn, args, *, workspace, limits,
                     timeout=None, parent_cgroup=None, scope_callback=None):
            scope = workspace / ".scope"
            if scope_callback:
                scope_callback(scope)
            return True

        ws = self._mock_workspace()
        with patch("branching.process.runner.run_in_process", side_effect=mock_rip), \
             patch("branching.process._cgroup.kill_scope",
                   side_effect=lambda s: killed.append(s)):
            spec = Speculate(
                [lambda p: True, lambda p: True],
                first_wins=False,
                resource_limits=ResourceLimits(memory=1024),
            )
            spec._run(ws, None)

        assert killed == []

    def test_no_kill_without_resource_limits(self):
        """Without resource_limits, no cgroup tracking or killing."""
        from branching.agent.speculate import Speculate

        killed = []

        def mock_rip(fn, args, *, workspace, limits,
                     timeout=None, parent_cgroup=None, scope_callback=None):
            return fn(*args)

        ws = self._mock_workspace()
        with patch("branching.process.runner.run_in_process", side_effect=mock_rip), \
             patch("branching.process._cgroup.kill_scope",
                   side_effect=lambda s: killed.append(s)):
            spec = Speculate(
                [lambda p: True, lambda p: True],
                first_wins=True,
            )
            spec._run(ws, None)

        assert killed == []


# ---------------------------------------------------------------
# Pattern cgroup kill on losers (BestOfN, TreeOfThoughts,
# BeamSearch, Tournament)
# ---------------------------------------------------------------

def _mock_workspace():
    """Create a mock workspace whose .branch() yields mock branch objects."""
    @contextmanager
    def mock_branch(name, on_success=None, on_error=None):
        b = MagicMock()
        b.path = Path(f"/tmp/ws/{name}")
        # .branch() on the mock branch itself (for ToT sub-branches)
        b.branch = mock_branch
        yield b
    ws = MagicMock()
    ws.branch = mock_branch
    return ws


class TestBestOfNCgroupKill:
    def test_no_kill_when_all_finish(self):
        """BestOfN: all tasks complete → no kill (nothing stuck)."""
        from branching.agent.patterns import BestOfN

        killed = []

        def mock_rip(fn, args, *, workspace, limits,
                     timeout=None, parent_cgroup=None, scope_callback=None):
            scope = workspace / ".scope"
            if scope_callback:
                scope_callback(scope)
            idx = args[1]
            return (True, float(idx))

        ws = _mock_workspace()
        with patch("branching.agent.patterns.run_in_process", side_effect=mock_rip), \
             patch("branching.process._cgroup.kill_scope",
                   side_effect=lambda s: killed.append(s)):
            bon = BestOfN(
                lambda p, i: (True, float(i)), n=2,
                resource_limits=ResourceLimits(memory=1024),
            )
            bon._run(ws, None)

        assert killed == []

    def test_kills_on_timeout(self):
        """BestOfN: timeout leaves tasks stuck → kill fires."""
        from branching.agent.patterns import BestOfN

        killed = []

        def mock_rip(fn, args, *, workspace, limits,
                     timeout=None, parent_cgroup=None, scope_callback=None):
            scope = workspace / ".scope"
            if scope_callback:
                scope_callback(scope)
            idx = args[1]
            if idx == 1:
                time.sleep(2)  # simulate stuck task
            return (True, float(idx))

        ws = _mock_workspace()
        with patch("branching.agent.patterns.run_in_process", side_effect=mock_rip), \
             patch("branching.process._cgroup.kill_scope",
                   side_effect=lambda s: killed.append(s)):
            bon = BestOfN(
                lambda p, i: (True, float(i)), n=2,
                timeout=0.05,
                resource_limits=ResourceLimits(memory=1024),
            )
            bon._run(ws, None)

        # Candidate 1 timed out and should have been killed.
        assert Path("/tmp/ws/best_of_n_1/.scope") in killed


class TestTreeOfThoughtsCgroupKill:
    def test_no_kill_when_all_finish(self):
        """TreeOfThoughts: all tasks complete → no kill."""
        from branching.agent.patterns import TreeOfThoughts

        killed = []

        def mock_rip(fn, args, *, workspace, limits,
                     timeout=None, parent_cgroup=None, scope_callback=None):
            scope = workspace / ".scope"
            if scope_callback:
                scope_callback(scope)
            return fn(*args)

        ws = _mock_workspace()
        strats = [lambda p: (True, 0.5), lambda p: (True, 0.9)]
        with patch("branching.agent.patterns.run_in_process", side_effect=mock_rip), \
             patch("branching.process._cgroup.kill_scope",
                   side_effect=lambda s: killed.append(s)):
            tot = TreeOfThoughts(
                strats, resource_limits=ResourceLimits(memory=1024),
            )
            tot._single_level(ws, strats, depth=0)

        assert killed == []


class TestTournamentCgroupKill:
    def test_no_kill_when_all_finish(self):
        """Tournament: all tasks complete → no kill."""
        from branching.agent.patterns import Tournament

        killed = []

        def mock_rip(fn, args, *, workspace, limits,
                     timeout=None, parent_cgroup=None, scope_callback=None):
            scope = workspace / ".scope"
            if scope_callback:
                scope_callback(scope)
            return True

        ws = _mock_workspace()
        with patch("branching.agent.patterns.run_in_process", side_effect=mock_rip), \
             patch("branching.process._cgroup.kill_scope",
                   side_effect=lambda s: killed.append(s)):
            t = Tournament(
                lambda p, i: True, n=2,
                judge=lambda a, b: 0,
                resource_limits=ResourceLimits(memory=1024),
            )
            t._run(ws, None)

        assert killed == []

    def test_no_kill_without_resource_limits(self):
        from branching.agent.patterns import Tournament
        killed = []
        ws = _mock_workspace()
        with patch("branching.process._cgroup.kill_scope",
                   side_effect=lambda s: killed.append(s)):
            t = Tournament(
                lambda p, i: True, n=2,
                judge=lambda a, b: 0,
            )
            t._run(ws, None)
        assert killed == []


class TestExports:
    def test_resource_limits_in_all(self):
        import branching
        assert "ResourceLimits" in branching.__all__

    def test_resource_limits_importable(self):
        from branching import ResourceLimits
        rl = ResourceLimits(memory=1024)
        assert rl.memory == 1024
