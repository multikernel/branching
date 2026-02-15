# SPDX-License-Identifier: Apache-2.0
"""Tests for resource limits: dataclass, parse_memory_size, set_limits, and
pattern integration."""

import json
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
# BranchContext accepts limits
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


# ---------------------------------------------------------------
# CLI resource limit parsing
# ---------------------------------------------------------------

class TestCLIResourceLimits:
    def test_parse_resource_limits_none(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(memory_limit=None, cpu_limit=None)
        assert _parse_resource_limits(args) is None

    def test_parse_resource_limits_memory(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(memory_limit="512M", cpu_limit=None)
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory == 512 * 1024 ** 2
        assert rl.cpu is None

    def test_parse_resource_limits_cpu(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(memory_limit=None, cpu_limit=0.5)
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory is None
        assert rl.cpu == 0.5

    def test_parse_resource_limits_both(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(memory_limit="1G", cpu_limit=0.25)
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory == 1024 ** 3
        assert rl.cpu == 0.25

    def test_parser_has_resource_limit_flags(self):
        from cli import build_parser
        parser = build_parser()
        # Verify run subcommand accepts --memory-limit and --cpu-limit
        args = parser.parse_args(["run", "--memory-limit", "512M", "--cpu-limit", "0.5", "--", "echo", "hi"])
        assert args.memory_limit == "512M"
        assert args.cpu_limit == 0.5

    def test_parser_speculate_has_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["speculate", "-c", "echo a", "--memory-limit", "256M"])
        assert args.memory_limit == "256M"

    def test_parser_best_of_n_has_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["best-of-n", "--cpu-limit", "0.75", "--", "echo", "hi"])
        assert args.cpu_limit == 0.75

    def test_parser_reflexion_has_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["reflexion", "--memory-limit", "1G", "--", "echo", "hi"])
        assert args.memory_limit == "1G"


# ---------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------

class TestExports:
    def test_resource_limits_in_all(self):
        import branching
        assert "ResourceLimits" in branching.__all__

    def test_resource_limits_importable(self):
        from branching import ResourceLimits
        rl = ResourceLimits(memory=1024)
        assert rl.memory == 1024
