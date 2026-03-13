# SPDX-License-Identifier: Apache-2.0
"""Tests for resource limits: dataclass, parse_memory_size, and
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
        assert rl.cpu_time is None
        assert rl.nproc is None

    def test_memory_only(self):
        rl = ResourceLimits(memory=512 * 1024 * 1024)
        assert rl.memory == 512 * 1024 * 1024
        assert rl.cpu_time is None

    def test_cpu_time_only(self):
        rl = ResourceLimits(cpu_time=60)
        assert rl.cpu_time == 60
        assert rl.memory is None

    def test_nproc_only(self):
        rl = ResourceLimits(nproc=128)
        assert rl.nproc == 128
        assert rl.memory is None

    def test_all_fields(self):
        rl = ResourceLimits(memory=1024, cpu_time=60, nproc=128)
        assert rl.memory == 1024
        assert rl.cpu_time == 60
        assert rl.nproc == 128

    def test_frozen(self):
        rl = ResourceLimits(memory=1024)
        with pytest.raises(AttributeError):
            rl.memory = 2048  # type: ignore[misc]

    def test_equality(self):
        a = ResourceLimits(memory=100, cpu_time=30)
        b = ResourceLimits(memory=100, cpu_time=30)
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
# Pattern integration (resource_limits parameter acceptance)
# ---------------------------------------------------------------

class TestPatternResourceLimitsAcceptance:
    """Verify all patterns pass resource_limits through to run_in_process."""

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
            mock.return_value = (MockFSBackend, Path("/tmp/test_ws"))
            return Workspace("/tmp/test_ws")

    def test_speculate_passes_resource_limits(self):
        from branching.agent.speculate import Speculate
        rl = ResourceLimits(memory=512 * 1024 * 1024)
        ws = self._make_workspace()
        captured = []
        def mock_rip(fn, args, *, workspace, limits, **kw):
            captured.append(limits)
            return fn(*args)
        with patch("branching.process.runner.run_in_process", mock_rip):
            Speculate([lambda p: True], resource_limits=rl)(ws)
        assert captured == [rl]

    def test_best_of_n_passes_resource_limits(self):
        from branching.agent.patterns import BestOfN
        rl = ResourceLimits(cpu_time=30)
        ws = self._make_workspace()
        captured = []
        def mock_rip(fn, args, *, workspace, limits, **kw):
            captured.append(limits)
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip):
            BestOfN([lambda p: (True, 1.0)] * 2, resource_limits=rl)(ws)
        assert all(l is rl for l in captured)

    def test_reflexion_passes_resource_limits(self):
        from branching.agent.patterns import Reflexion
        rl = ResourceLimits(memory=1024)
        ws = self._make_workspace()
        captured = []
        def mock_rip(fn, args, *, workspace, limits, **kw):
            captured.append(limits)
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip):
            Reflexion(lambda p, a, f: True, resource_limits=rl)(ws)
        assert captured == [rl]

    def test_tree_of_thoughts_passes_resource_limits(self):
        from branching.agent.patterns import TreeOfThoughts
        rl = ResourceLimits(cpu_time=60)
        ws = self._make_workspace()
        captured = []
        def mock_rip(fn, args, *, workspace, limits, **kw):
            captured.append(limits)
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip):
            TreeOfThoughts([lambda p: True], resource_limits=rl)(ws)
        assert captured == [rl]

    def test_beam_search_passes_resource_limits(self):
        from branching.agent.patterns import BeamSearch
        rl = ResourceLimits(memory=2048, cpu_time=15)
        ws = self._make_workspace()
        captured = []
        def mock_rip(fn, args, *, workspace, limits, **kw):
            captured.append(limits)
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip):
            BeamSearch(
                [lambda p: True],
                expand=lambda p, d: [],
                resource_limits=rl,
            )(ws)
        assert captured == [rl]

    def test_tournament_passes_resource_limits(self):
        from branching.agent.patterns import Tournament
        rl = ResourceLimits(memory=4096)
        ws = self._make_workspace()
        captured = []
        def mock_rip(fn, args, *, workspace, limits, **kw):
            captured.append(limits)
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip):
            Tournament(
                [lambda p: True] * 2,
                judge=lambda a, b: 0,
                resource_limits=rl,
            )(ws)
        assert all(l is rl for l in captured)


# ---------------------------------------------------------------
# Pattern group_limits acceptance
# ---------------------------------------------------------------

class TestPatternGroupLimitsAcceptance:
    """Verify all patterns apply group_limits via apply_limits."""

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
            mock.return_value = (MockFSBackend, Path("/tmp/test_ws"))
            return Workspace("/tmp/test_ws")

    def test_speculate_passes_group_limits(self):
        from branching.agent.speculate import Speculate
        rl = ResourceLimits(memory=512 * 1024 * 1024)
        gl = ResourceLimits(memory=1024 * 1024 * 1024)
        ws = self._make_workspace()
        def mock_rip(fn, args, *, workspace, limits, **kw):
            return fn(*args)
        with patch("branching.process.runner.run_in_process", mock_rip), \
             patch("branching.process._prlimit.apply_limits") as mock_al:
            Speculate([lambda p: True], resource_limits=rl, group_limits=gl)(ws)
        mock_al.assert_called_once_with(gl)

    def test_best_of_n_passes_group_limits(self):
        from branching.agent.patterns import BestOfN
        rl = ResourceLimits(cpu_time=30)
        gl = ResourceLimits(cpu_time=120)
        ws = self._make_workspace()
        def mock_rip(fn, args, *, workspace, limits, **kw):
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip), \
             patch("branching.process._prlimit.apply_limits") as mock_al:
            BestOfN([lambda p: (True, 1.0)] * 2, resource_limits=rl, group_limits=gl)(ws)
        mock_al.assert_called_once_with(gl)

    def test_reflexion_passes_group_limits(self):
        from branching.agent.patterns import Reflexion
        rl = ResourceLimits(memory=1024)
        gl = ResourceLimits(memory=2048)
        ws = self._make_workspace()
        def mock_rip(fn, args, *, workspace, limits, **kw):
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip), \
             patch("branching.process._prlimit.apply_limits") as mock_al:
            Reflexion(lambda p, a, f: True, resource_limits=rl, group_limits=gl)(ws)
        mock_al.assert_called_once_with(gl)

    def test_tree_of_thoughts_passes_group_limits(self):
        from branching.agent.patterns import TreeOfThoughts
        rl = ResourceLimits(cpu_time=60)
        gl = ResourceLimits(cpu_time=240)
        ws = self._make_workspace()
        def mock_rip(fn, args, *, workspace, limits, **kw):
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip), \
             patch("branching.process._prlimit.apply_limits") as mock_al:
            TreeOfThoughts([lambda p: True], resource_limits=rl, group_limits=gl)(ws)
        mock_al.assert_called_once_with(gl)

    def test_beam_search_passes_group_limits(self):
        from branching.agent.patterns import BeamSearch
        rl = ResourceLimits(memory=2048)
        gl = ResourceLimits(memory=8192)
        ws = self._make_workspace()
        def mock_rip(fn, args, *, workspace, limits, **kw):
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip), \
             patch("branching.process._prlimit.apply_limits") as mock_al:
            BeamSearch(
                [lambda p: True],
                expand=lambda p, d: [],
                resource_limits=rl, group_limits=gl,
            )(ws)
        mock_al.assert_called_with(gl)

    def test_tournament_passes_group_limits(self):
        from branching.agent.patterns import Tournament
        rl = ResourceLimits(memory=4096)
        gl = ResourceLimits(memory=4096, cpu_time=120)
        ws = self._make_workspace()
        def mock_rip(fn, args, *, workspace, limits, **kw):
            return fn(*args)
        with patch("branching.agent.patterns.run_in_process", mock_rip), \
             patch("branching.process._prlimit.apply_limits") as mock_al:
            Tournament(
                [lambda p: True] * 2,
                judge=lambda a, b: 0,
                resource_limits=rl, group_limits=gl,
            )(ws)
        mock_al.assert_called_once_with(gl)


# ---------------------------------------------------------------
# BranchContext accepts limits
# ---------------------------------------------------------------

class TestBranchContextLimits:
    def test_accepts_limits_kwarg(self):
        from branching.process.context import BranchContext
        rl = ResourceLimits(memory=1024)
        ctx = BranchContext(lambda p: None, workspace=Path("/tmp"), limits=rl)
        assert ctx._limits is rl

    def test_create_passes_limits_to_contexts(self):
        """BranchContext.create() threads limits through to each context."""
        from branching.process.context import BranchContext
        rl = ResourceLimits(memory=1024)
        with patch.object(BranchContext, '__enter__', lambda self: self), \
             patch.object(BranchContext, '__exit__', lambda self, *a: False):
            with BranchContext.create(
                targets=[lambda p: None, lambda p: None],
                workspaces=[Path("/tmp/a"), Path("/tmp/b")],
                limits=rl,
            ) as contexts:
                assert len(contexts) == 2
                for ctx in contexts:
                    assert ctx._limits is rl


# ---------------------------------------------------------------
# BranchContext.branch_id property
# ---------------------------------------------------------------

class TestBranchContextBranchId:
    def test_branch_id_none_before_enter(self):
        from branching.process.context import BranchContext
        ctx = BranchContext(lambda p: None, workspace=Path("/tmp"))
        assert ctx.branch_id is None


# ---------------------------------------------------------------
# run_in_process pid_callback
# ---------------------------------------------------------------

class TestRunInProcessPidCallback:
    def test_pid_callback_invoked_with_pid(self):
        """run_in_process invokes pid_callback with the child PID."""
        from branching.process import runner
        from branching.exceptions import ProcessBranchError
        callback = MagicMock()
        with patch.object(runner, 'BranchContext') as MockBC:
            mock_ctx = MockBC.return_value
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_ctx.pid = 12345
            mock_ctx.wait = MagicMock()
            with pytest.raises(ProcessBranchError):
                runner.run_in_process(
                    lambda: 42, (), Path("/tmp"),
                    pid_callback=callback,
                )
        callback.assert_called_once_with(12345)

    def test_pid_callback_not_invoked_when_omitted(self):
        """run_in_process with default pid_callback=None works."""
        from branching.process import runner
        from branching.exceptions import ProcessBranchError
        with patch.object(runner, 'BranchContext') as MockBC:
            mock_ctx = MockBC.return_value
            mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
            mock_ctx.__exit__ = MagicMock(return_value=False)
            mock_ctx.pid = 12345
            mock_ctx.wait = MagicMock()
            with pytest.raises(ProcessBranchError):
                runner.run_in_process(lambda: 42, (), Path("/tmp"))


# ---------------------------------------------------------------
# CLI resource limit parsing
# ---------------------------------------------------------------

class TestCLIResourceLimits:
    def test_parse_resource_limits_none(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit=None, cpu_time_limit=None,
            nproc_limit=None,
        )
        assert _parse_resource_limits(args) is None

    def test_parse_resource_limits_memory(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit="512M", cpu_time_limit=None,
            nproc_limit=None,
        )
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory == 512 * 1024 ** 2
        assert rl.cpu_time is None

    def test_parse_resource_limits_cpu_time(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit=None, cpu_time_limit=60,
            nproc_limit=None,
        )
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory is None
        assert rl.cpu_time == 60

    def test_parse_resource_limits_nproc(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit=None, cpu_time_limit=None,
            nproc_limit=128,
        )
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.nproc == 128

    def test_parse_resource_limits_all(self):
        import argparse
        from cli import _parse_resource_limits
        args = argparse.Namespace(
            memory_limit="1G", cpu_time_limit=120,
            nproc_limit=256,
        )
        rl = _parse_resource_limits(args)
        assert rl is not None
        assert rl.memory == 1024 ** 3
        assert rl.cpu_time == 120
        assert rl.nproc == 256

    def test_parse_group_limits_none(self):
        import argparse
        from cli import _parse_group_limits
        args = argparse.Namespace(
            group_memory_limit=None, group_cpu_time_limit=None,
            group_nproc_limit=None,
        )
        assert _parse_group_limits(args) is None

    def test_parse_group_limits_memory(self):
        import argparse
        from cli import _parse_group_limits
        args = argparse.Namespace(
            group_memory_limit="2G", group_cpu_time_limit=None,
            group_nproc_limit=None,
        )
        gl = _parse_group_limits(args)
        assert gl is not None
        assert gl.memory == 2 * 1024 ** 3

    def test_parse_group_limits_cpu_time(self):
        import argparse
        from cli import _parse_group_limits
        args = argparse.Namespace(
            group_memory_limit=None, group_cpu_time_limit=240,
            group_nproc_limit=None,
        )
        gl = _parse_group_limits(args)
        assert gl is not None
        assert gl.cpu_time == 240

    def test_parse_group_limits_nproc(self):
        import argparse
        from cli import _parse_group_limits
        args = argparse.Namespace(
            group_memory_limit=None, group_cpu_time_limit=None,
            group_nproc_limit=512,
        )
        gl = _parse_group_limits(args)
        assert gl is not None
        assert gl.nproc == 512

    def test_parser_has_resource_limit_flags(self):
        from cli import build_parser
        parser = build_parser()
        # Verify run subcommand accepts --memory-limit and --cpu-time-limit
        args = parser.parse_args(["run", "--memory-limit", "512M", "--cpu-time-limit", "60", "--", "echo", "hi"])
        assert args.memory_limit == "512M"
        assert args.cpu_time_limit == 60

    def test_parser_has_nproc_limit_flag(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["run", "--nproc-limit", "128", "--", "echo", "hi"])
        assert args.nproc_limit == 128

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
            "--group-memory-limit", "2G", "--group-cpu-time-limit", "240",
        ])
        assert args.group_memory_limit == "2G"
        assert args.group_cpu_time_limit == 240

    def test_parser_best_of_n_has_flags(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["best-of-n", "--cpu-time-limit", "30", "--", "echo", "hi"])
        assert args.cpu_time_limit == 30

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
            "reflexion", "--group-cpu-time-limit", "120", "--", "echo", "hi",
        ])
        assert args.group_cpu_time_limit == 120


# ---------------------------------------------------------------
# __init__.py exports
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# Speculate._run_with_limits passes pid_callback
# ---------------------------------------------------------------

class TestSpeculatePidCallback:
    def test_passes_pid_callback_to_run_in_process(self):
        from branching.agent.speculate import Speculate
        spec = Speculate(
            [lambda p: True],
            resource_limits=ResourceLimits(memory=1024),
        )
        with patch("branching.process.runner.run_in_process") as mock_rip:
            mock_rip.return_value = True
            callback = MagicMock()
            spec._run_with_limits(Path("/tmp"), Path("/"), 0, pid_callback=callback)
            assert mock_rip.call_args.kwargs["pid_callback"] is callback

    def test_pid_callback_none_by_default(self):
        from branching.agent.speculate import Speculate
        spec = Speculate(
            [lambda p: True],
            resource_limits=ResourceLimits(memory=1024),
        )
        with patch("branching.process.runner.run_in_process") as mock_rip:
            mock_rip.return_value = True
            spec._run_with_limits(Path("/tmp"), Path("/"), 0)
            assert mock_rip.call_args.kwargs["pid_callback"] is None


# ---------------------------------------------------------------
# Speculate process tracker kill on losers
# ---------------------------------------------------------------

class TestSpeculateProcessTrackerKill:
    @staticmethod
    def _mock_workspace():
        """Create a mock workspace whose .branch() yields mock branch objects."""
        @contextmanager
        def mock_branch(name, on_success=None, on_error=None):
            b = MagicMock()
            b.path = Path(f"/tmp/ws/{name}")
            b.mount_root = Path("/tmp/ws")
            yield b
        ws = MagicMock()
        ws.branch = mock_branch
        return ws

    def test_kills_loser_scopes_on_first_wins(self):
        """first_wins + resource_limits: winner kills losers via process tracker."""
        from branching.agent.speculate import Speculate

        # Sync: both candidates register pids before either returns.
        count = [0]
        lock = threading.Lock()
        both_registered = threading.Event()
        mock_tracker = MagicMock()

        def mock_rip(fn, args, *, workspace, mount_root=None, limits,
                     timeout=None, pid_callback=None):
            workspace_pid = hash(str(workspace)) % 10000
            if pid_callback:
                pid_callback(workspace_pid)
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
             patch("branching.process._process_tracker.BpfProcessTracker.get",
                   return_value=mock_tracker):
            spec = Speculate(
                [lambda p: True, lambda p: True],
                first_wins=True,
                resource_limits=ResourceLimits(memory=1024),
            )
            spec._run(ws)

        loser_pid = hash(str(Path("/tmp/ws/speculate_1"))) % 10000
        mock_tracker.kill_branch.assert_called()
        killed_pids = [c.args[0] for c in mock_tracker.kill_branch.call_args_list]
        assert loser_pid in killed_pids

    def test_no_kill_when_first_wins_false(self):
        """first_wins=False: all branches run to completion, no kill."""
        from branching.agent.speculate import Speculate

        mock_tracker = MagicMock()

        def mock_rip(fn, args, *, workspace, mount_root=None, limits,
                     timeout=None, pid_callback=None):
            workspace_pid = hash(str(workspace)) % 10000
            if pid_callback:
                pid_callback(workspace_pid)
            return True

        ws = self._mock_workspace()
        with patch("branching.process.runner.run_in_process", side_effect=mock_rip), \
             patch("branching.process._process_tracker.BpfProcessTracker.get",
                   return_value=mock_tracker):
            spec = Speculate(
                [lambda p: True, lambda p: True],
                first_wins=False,
                resource_limits=ResourceLimits(memory=1024),
            )
            spec._run(ws)

        mock_tracker.kill_branch.assert_not_called()

    def test_no_kill_without_resource_limits(self):
        """Without resource_limits, no process tracking or killing."""
        from branching.agent.speculate import Speculate

        mock_tracker = MagicMock()

        def mock_rip(fn, args, *, workspace, mount_root=None, limits,
                     timeout=None, pid_callback=None):
            return fn(*args)

        ws = self._mock_workspace()
        with patch("branching.process.runner.run_in_process", side_effect=mock_rip), \
             patch("branching.process._process_tracker.BpfProcessTracker.get",
                   return_value=mock_tracker):
            spec = Speculate(
                [lambda p: True, lambda p: True],
                first_wins=True,
            )
            spec._run(ws)

        mock_tracker.kill_branch.assert_not_called()


# ---------------------------------------------------------------
# Pattern process tracker kill on losers (BestOfN, TreeOfThoughts,
# BeamSearch, Tournament)
# ---------------------------------------------------------------

def _mock_workspace():
    """Create a mock workspace whose .branch() yields mock branch objects."""
    @contextmanager
    def mock_branch(name, on_success=None, on_error=None):
        b = MagicMock()
        b.path = Path(f"/tmp/ws/{name}")
        b.mount_root = Path("/tmp/ws")
        # .branch() on the mock branch itself (for ToT sub-branches)
        b.branch = mock_branch
        yield b
    ws = MagicMock()
    ws.branch = mock_branch
    return ws


class TestBestOfNProcessTrackerKill:
    def test_no_kill_when_all_finish(self):
        """BestOfN: all tasks complete -> no kill (nothing stuck)."""
        from branching.agent.patterns import BestOfN

        mock_tracker = MagicMock()

        def mock_rip(fn, args, *, workspace, mount_root=None, limits,
                     timeout=None, pid_callback=None):
            workspace_pid = hash(str(workspace)) % 10000
            if pid_callback:
                pid_callback(workspace_pid)
            return fn(*args)

        ws = _mock_workspace()
        with patch("branching.agent.patterns.run_in_process", side_effect=mock_rip), \
             patch("branching.process._process_tracker.BpfProcessTracker.get",
                   return_value=mock_tracker):
            bon = BestOfN(
                [lambda p, i=i: (True, float(i)) for i in range(2)],
                resource_limits=ResourceLimits(memory=1024),
            )
            bon._run(ws)

        mock_tracker.kill_branch.assert_not_called()

    def test_kills_on_timeout(self):
        """BestOfN: timeout leaves tasks stuck -> kill fires."""
        from branching.agent.patterns import BestOfN

        mock_tracker = MagicMock()

        def mock_rip(fn, args, *, workspace, mount_root=None, limits,
                     timeout=None, pid_callback=None):
            workspace_pid = hash(str(workspace)) % 10000
            if pid_callback:
                pid_callback(workspace_pid)
            return fn(*args)

        def fast(p):
            return (True, 0.0)

        def stuck(p):
            time.sleep(2)
            return (True, 1.0)

        ws = _mock_workspace()
        with patch("branching.agent.patterns.run_in_process", side_effect=mock_rip), \
             patch("branching.process._process_tracker.BpfProcessTracker.get",
                   return_value=mock_tracker):
            bon = BestOfN(
                [fast, stuck],
                timeout=0.05,
                resource_limits=ResourceLimits(memory=1024),
            )
            bon._run(ws)

        # Candidate 1 timed out and should have been killed.
        stuck_pid = hash(str(Path("/tmp/ws/best_of_n_1"))) % 10000
        killed_pids = [c.args[0] for c in mock_tracker.kill_branch.call_args_list]
        assert stuck_pid in killed_pids


class TestTreeOfThoughtsProcessTrackerKill:
    def test_no_kill_when_all_finish(self):
        """TreeOfThoughts: all tasks complete -> no kill."""
        from branching.agent.patterns import TreeOfThoughts

        mock_tracker = MagicMock()

        def mock_rip(fn, args, *, workspace, mount_root=None, limits,
                     timeout=None, pid_callback=None):
            workspace_pid = hash(str(workspace)) % 10000
            if pid_callback:
                pid_callback(workspace_pid)
            return fn(*args)

        ws = _mock_workspace()
        strats = [lambda p: (True, 0.5), lambda p: (True, 0.9)]
        with patch("branching.agent.patterns.run_in_process", side_effect=mock_rip), \
             patch("branching.process._process_tracker.BpfProcessTracker.get",
                   return_value=mock_tracker):
            tot = TreeOfThoughts(
                strats, resource_limits=ResourceLimits(memory=1024),
            )
            tot._single_level(ws, strats, depth=0)

        mock_tracker.kill_branch.assert_not_called()


class TestTournamentProcessTrackerKill:
    def test_no_kill_when_all_finish(self):
        """Tournament: all tasks complete -> no kill."""
        from branching.agent.patterns import Tournament

        mock_tracker = MagicMock()

        def mock_rip(fn, args, *, workspace, mount_root=None, limits,
                     timeout=None, pid_callback=None):
            workspace_pid = hash(str(workspace)) % 10000
            if pid_callback:
                pid_callback(workspace_pid)
            return fn(*args)

        ws = _mock_workspace()
        with patch("branching.agent.patterns.run_in_process", side_effect=mock_rip), \
             patch("branching.process._process_tracker.BpfProcessTracker.get",
                   return_value=mock_tracker):
            t = Tournament(
                [lambda p: True] * 2,
                judge=lambda a, b: 0,
                resource_limits=ResourceLimits(memory=1024),
            )
            t._run(ws)

        mock_tracker.kill_branch.assert_not_called()

    def test_no_kill_without_resource_limits(self):
        from branching.agent.patterns import Tournament
        mock_tracker = MagicMock()
        ws = _mock_workspace()
        with patch("branching.process._process_tracker.BpfProcessTracker.get",
                   return_value=mock_tracker):
            t = Tournament(
                [lambda p: True] * 2,
                judge=lambda a, b: 0,
            )
            t._run(ws)
        mock_tracker.kill_branch.assert_not_called()


class TestExports:
    def test_resource_limits_in_all(self):
        import branching
        assert "ResourceLimits" in branching.__all__

    def test_resource_limits_importable(self):
        from branching import ResourceLimits
        rl = ResourceLimits(memory=1024)
        assert rl.memory == 1024
