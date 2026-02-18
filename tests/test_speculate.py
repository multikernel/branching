# SPDX-License-Identifier: Apache-2.0
"""Tests for speculation patterns (mocked backends)."""

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest

from branching.core.base import FSBackend
from branching.core.workspace import Workspace
from branching.agent.speculate import Speculate
from branching.agent.patterns import BestOfN, Reflexion, TreeOfThoughts, BeamSearch, Tournament, Cascaded
from branching.agent.result import SpeculationResult, SpeculationOutcome


@pytest.fixture(autouse=True)
def mock_run_in_process():
    """Run tasks in-process for unit tests (skip fork/namespace)."""
    def _run_inline(fn, args, workspace, **kwargs):
        return fn(*args)
    with patch("branching.agent.patterns.run_in_process", _run_inline):
        with patch("branching.process.runner.run_in_process", _run_inline):
            yield


class MockFSBackend(FSBackend):
    """Mock FS backend that tracks operations."""

    _branches_created = []
    _commits = []
    _aborts = []

    @classmethod
    def reset(cls):
        cls._branches_created = []
        cls._commits = []
        cls._aborts = []

    @classmethod
    def fstype(cls) -> str:
        return "mockfs"

    @classmethod
    def create_branch(cls, name, mountpoint, parent_mount, parent_branch):
        cls._branches_created.append(name)

    @classmethod
    def commit(cls, mountpoint):
        cls._commits.append(mountpoint)

    @classmethod
    def abort(cls, mountpoint):
        cls._aborts.append(mountpoint)


@pytest.fixture(autouse=True)
def reset_mock():
    MockFSBackend.reset()


def _make_workspace():
    with patch("branching.core.workspace.detect_fs_for_mount") as mock_detect:
        mock_detect.return_value = MockFSBackend
        return Workspace("/tmp/test_ws")


class TestSpeculate:
    def test_first_wins(self):
        ws = _make_workspace()

        def success(path: Path) -> bool:
            return True

        def failure(path: Path) -> bool:
            return False

        spec = Speculate([success, failure], first_wins=True)
        outcome = spec(ws)

        assert outcome.committed
        assert outcome.winner is not None
        assert outcome.winner.success

    def test_all_fail(self):
        ws = _make_workspace()

        def failure(path: Path) -> bool:
            return False

        spec = Speculate([failure, failure], first_wins=True)
        outcome = spec(ws)

        assert not outcome.committed
        assert outcome.winner is None
        assert len(outcome.all_results) == 2

    def test_candidate_exception(self):
        ws = _make_workspace()

        def exploding(path: Path) -> bool:
            raise RuntimeError("boom")

        spec = Speculate([exploding])
        outcome = spec(ws)

        assert not outcome.committed
        assert len(outcome.all_results) == 1


class TestBestOfN:
    def test_picks_highest_score(self):
        ws = _make_workspace()

        def task(path: Path, attempt: int) -> tuple[bool, float]:
            scores = [0.3, 0.9, 0.6]
            return True, scores[attempt]

        outcome = BestOfN(task, n=3)(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 1
        assert outcome.winner.score == 0.9
        assert len(outcome.all_results) == 3

    def test_skips_failures(self):
        ws = _make_workspace()

        def task(path: Path, attempt: int) -> tuple[bool, float]:
            if attempt == 0:
                return False, 0.0  # Fail with score 0
            return True, 0.5

        outcome = BestOfN(task, n=2)(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 1

    def test_all_fail(self):
        ws = _make_workspace()

        def task(path: Path, attempt: int) -> tuple[bool, float]:
            return False, 0.0

        outcome = BestOfN(task, n=3)(ws)
        assert not outcome.committed
        assert outcome.winner is None
        assert len(outcome.all_results) == 3

    def test_exception_in_candidate(self):
        ws = _make_workspace()

        def task(path: Path, attempt: int) -> tuple[bool, float]:
            if attempt == 0:
                raise RuntimeError("boom")
            return True, 1.0

        outcome = BestOfN(task, n=2)(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 1
        assert outcome.all_results[0].exception is not None

    def test_commits_exactly_one(self):
        """Only the winner should be committed; all others aborted."""
        ws = _make_workspace()

        def task(path: Path, attempt: int) -> tuple[bool, float]:
            return True, float(attempt)

        outcome = BestOfN(task, n=3)(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 2  # highest score
        # Exactly 1 commit, rest aborted
        assert len(MockFSBackend._commits) == 1
        assert len(MockFSBackend._aborts) == 2

    def test_runs_in_parallel(self):
        """Verify candidates actually run concurrently."""
        import time
        ws = _make_workspace()
        start = time.monotonic()

        def task(path: Path, attempt: int) -> tuple[bool, float]:
            time.sleep(0.2)
            return True, 1.0

        outcome = BestOfN(task, n=3)(ws)
        elapsed = time.monotonic() - start
        assert outcome.committed
        # 3 tasks @ 0.2s each; parallel should be ~0.2s, sequential ~0.6s
        assert elapsed < 0.5


class TestReflexion:
    def test_succeeds_first_try(self):
        ws = _make_workspace()

        def task(path, attempt, feedback):
            return True

        outcome = Reflexion(task, max_retries=3)(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 0
        assert len(outcome.all_results) == 1

    def test_succeeds_after_retry(self):
        ws = _make_workspace()

        def task(path, attempt, feedback):
            return attempt >= 1  # Fail first, succeed second

        def critique(path):
            return "try harder"

        outcome = Reflexion(task, max_retries=3, critique=critique)(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 1
        assert len(outcome.all_results) == 2

    def test_all_retries_fail(self):
        ws = _make_workspace()

        def task(path, attempt, feedback):
            return False

        outcome = Reflexion(task, max_retries=2)(ws)
        assert not outcome.committed
        assert len(outcome.all_results) == 2


class TestTreeOfThoughts:
    def test_picks_highest_score(self):
        ws = _make_workspace()

        def strat_a(path):
            return True, 0.3

        def strat_b(path):
            return True, 0.9

        def strat_c(path):
            return True, 0.6

        outcome = TreeOfThoughts([strat_a, strat_b, strat_c])(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 1
        assert outcome.winner.score == 0.9
        assert len(outcome.all_results) == 3

    def test_bool_strategies(self):
        """Strategies returning bare bool get score 1.0/0.0."""
        ws = _make_workspace()

        def strat_a(path):
            return False

        def strat_b(path):
            return True

        outcome = TreeOfThoughts([strat_a, strat_b])(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 1
        assert len(outcome.all_results) == 2

    def test_all_fail(self):
        ws = _make_workspace()

        def strat_a(path):
            return False

        def strat_b(path):
            return False

        outcome = TreeOfThoughts([strat_a, strat_b])(ws)
        assert not outcome.committed
        assert len(outcome.all_results) == 2

    def test_commits_exactly_one(self):
        ws = _make_workspace()

        def strat_a(path):
            return True, 1.0

        def strat_b(path):
            return True, 2.0

        def strat_c(path):
            return True, 0.5

        outcome = TreeOfThoughts([strat_a, strat_b, strat_c])(ws)
        assert outcome.committed
        assert len(MockFSBackend._commits) == 1
        assert len(MockFSBackend._aborts) == 2

    def test_runs_in_parallel(self):
        import time
        ws = _make_workspace()
        start = time.monotonic()

        def strat(path):
            time.sleep(0.2)
            return True

        outcome = TreeOfThoughts([strat, strat, strat])(ws)
        elapsed = time.monotonic() - start
        assert outcome.committed
        # 3 tasks @ 0.2s; parallel ~0.2s, sequential ~0.6s
        assert elapsed < 0.5

    def test_evaluate_overrides_score(self):
        ws = _make_workspace()

        def strat_a(path):
            return True, 10.0  # strategy says 10

        def strat_b(path):
            return True, 1.0   # strategy says 1

        # evaluate overrides: always returns index-based score
        calls = []
        def evaluate(path):
            calls.append(path)
            return float(len(calls))  # 1.0 for first, 2.0 for second

        outcome = TreeOfThoughts(
            [strat_a, strat_b], evaluate=evaluate
        )(ws)
        assert outcome.committed
        # evaluate was called for both successes
        assert len(calls) == 2

    def test_multi_level_with_expand(self):
        ws = _make_workspace()
        expand_calls = []

        def strat_a(path):
            return True, 0.8

        def strat_b(path):
            return False, 0.0

        def refine_x(path):
            return True, 0.9

        def refine_y(path):
            return True, 0.7

        def expand(path, depth):
            expand_calls.append(depth)
            return [refine_x, refine_y]

        outcome = TreeOfThoughts(
            [strat_a, strat_b],
            expand=expand,
            max_depth=2,
        )(ws)
        assert outcome.committed
        # Level 0: strat_a wins (0.8 > fail)
        # Level 1: expand called with depth=1, refine_x wins (0.9 > 0.7)
        assert expand_calls == [1]
        assert outcome.winner.score == 0.9

    def test_multi_level_abort_on_no_winner(self):
        ws = _make_workspace()

        def strat_a(path):
            return True, 1.0

        def dead_end(path):
            return False, 0.0

        def expand(path, depth):
            return [dead_end]  # next level always fails

        outcome = TreeOfThoughts(
            [strat_a],
            expand=expand,
            max_depth=2,
        )(ws)
        # Level 0 succeeds, level 1 fails → root aborts
        assert not outcome.committed

    def test_exception_in_strategy(self):
        ws = _make_workspace()

        def exploding(path):
            raise RuntimeError("boom")

        def good(path):
            return True

        outcome = TreeOfThoughts([exploding, good])(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 1
        assert outcome.all_results[0].exception is not None


class TestBeamSearch:
    def test_single_level(self):
        """beam_width >= N, all survive, best wins."""
        ws = _make_workspace()

        def strat_a(path):
            return True, 0.3

        def strat_b(path):
            return True, 0.9

        def strat_c(path):
            return True, 0.6

        outcome = BeamSearch(
            [strat_a, strat_b, strat_c],
            expand=lambda p, d: [],
            beam_width=3,
            max_depth=1,
        )(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 1
        assert outcome.winner.score == 0.9
        assert len(outcome.all_results) == 3

    def test_prunes_to_beam_width(self):
        """N > K, verifies only K survive level 0."""
        ws = _make_workspace()

        def strat(path):
            return True, 1.0

        outcome = BeamSearch(
            [strat, strat, strat, strat],
            expand=lambda p, d: [],
            beam_width=2,
            max_depth=1,
        )(ws)
        assert outcome.committed
        # 1 final commit + 2 aborted at level 0 + 1 aborted at final
        # (2 survive level 0, 1 commits, 1 aborts at final)
        assert len(MockFSBackend._commits) == 1
        assert len(MockFSBackend._aborts) == 3

    def test_multi_level_expand(self):
        """2 levels, beams expand and best wins."""
        ws = _make_workspace()
        expand_calls = []

        def strat_a(path):
            return True, 0.8

        def strat_b(path):
            return True, 0.5

        def strat_c(path):
            return True, 0.3

        def refine_x(path):
            return True, 0.9

        def refine_y(path):
            return True, 0.7

        def expand(path, depth):
            expand_calls.append(depth)
            return [refine_x, refine_y]

        outcome = BeamSearch(
            [strat_a, strat_b, strat_c],
            expand=expand,
            beam_width=2,
            max_depth=2,
        )(ws)
        assert outcome.committed
        # Level 0: strat_a (0.8) and strat_b (0.5) survive (top-2)
        # Level 1: expand called for each surviving beam at depth=1
        assert 1 in expand_calls
        # Winner should have the highest sub-branch score
        assert outcome.winner.score == 0.9

    def test_beam_dies_mid_search(self):
        """A beam's children all fail → beam is pruned."""
        ws = _make_workspace()

        def strat_a(path):
            return True, 0.8

        def strat_b(path):
            return True, 0.5

        call_count = [0]

        def expand(path, depth):
            call_count[0] += 1
            if call_count[0] == 1:
                # First beam gets good children
                return [lambda p: (True, 0.9)]
            else:
                # Second beam gets failing children
                return [lambda p: (False, 0.0)]

        outcome = BeamSearch(
            [strat_a, strat_b],
            expand=expand,
            beam_width=2,
            max_depth=2,
        )(ws)
        assert outcome.committed
        # One beam should survive with score 0.9
        assert outcome.winner.score == 0.9

    def test_all_fail(self):
        """No successes → committed=False."""
        ws = _make_workspace()

        def strat(path):
            return False

        outcome = BeamSearch(
            [strat, strat, strat],
            expand=lambda p, d: [],
            beam_width=2,
            max_depth=2,
        )(ws)
        assert not outcome.committed
        assert outcome.winner is None
        assert len(outcome.all_results) == 3

    def test_commits_exactly_one(self):
        """Only final winner committed to workspace."""
        ws = _make_workspace()

        def strat(path):
            return True, 1.0

        outcome = BeamSearch(
            [strat, strat, strat],
            expand=lambda p, d: [],
            beam_width=2,
            max_depth=1,
        )(ws)
        assert outcome.committed
        # 1 beam commits to workspace, 2 beams aborted
        assert len(MockFSBackend._commits) == 1
        assert len(MockFSBackend._aborts) == 2

    def test_runs_in_parallel(self):
        """Verify candidates actually run concurrently."""
        import time
        ws = _make_workspace()
        start = time.monotonic()

        def strat(path):
            time.sleep(0.2)
            return True

        outcome = BeamSearch(
            [strat, strat, strat],
            expand=lambda p, d: [],
            beam_width=3,
            max_depth=1,
        )(ws)
        elapsed = time.monotonic() - start
        assert outcome.committed
        # 3 tasks @ 0.2s; parallel ~0.2s, sequential ~0.6s
        assert elapsed < 0.5

    def test_evaluate_overrides_score(self):
        """External evaluator overrides strategy scores."""
        ws = _make_workspace()
        eval_calls = []

        def strat_a(path):
            return True, 10.0  # strategy says 10

        def strat_b(path):
            return True, 1.0   # strategy says 1

        def evaluate(path):
            eval_calls.append(path)
            return float(len(eval_calls))  # 1.0, 2.0, ...

        outcome = BeamSearch(
            [strat_a, strat_b],
            expand=lambda p, d: [],
            evaluate=evaluate,
            beam_width=2,
            max_depth=1,
        )(ws)
        assert outcome.committed
        assert len(eval_calls) == 2

    def test_single_depth_fallback(self):
        """max_depth=1 behaves like scored BestOfN."""
        ws = _make_workspace()

        def strat_a(path):
            return True, 0.3

        def strat_b(path):
            return True, 0.9

        def strat_c(path):
            return False, 0.0

        outcome = BeamSearch(
            [strat_a, strat_b, strat_c],
            expand=lambda p, d: [],
            beam_width=5,
            max_depth=1,
        )(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 1
        assert outcome.winner.score == 0.9
        # Failed strategy not committed
        assert len(outcome.all_results) == 3


class TestTournament:
    def test_basic_bracket(self):
        """4 candidates, judge always picks second → candidate 3 wins."""
        ws = _make_workspace()

        def task(path: Path, index: int) -> bool:
            return True

        def judge(path_a: Path, path_b: Path) -> int:
            return 1  # always pick b

        outcome = Tournament(task, n=4, judge=judge)(ws)
        assert outcome.committed
        assert outcome.winner is not None
        # Bracket: (0v1→1), (2v3→3), (1v3→3)
        assert outcome.winner.branch_index == 3
        assert len(outcome.all_results) == 4

    def test_all_fail(self):
        """No survivors means nothing committed."""
        ws = _make_workspace()

        def task(path: Path, index: int) -> bool:
            return False

        def judge(path_a, path_b):
            raise AssertionError("judge should not be called")

        outcome = Tournament(task, n=3, judge=judge)(ws)
        assert not outcome.committed
        assert outcome.winner is None
        assert len(outcome.all_results) == 3

    def test_single_survivor(self):
        """Only 1 succeeds → auto-wins without judge call."""
        ws = _make_workspace()
        judge_calls = []

        def task(path: Path, index: int) -> bool:
            return index == 2

        def judge(path_a, path_b):
            judge_calls.append(1)
            return 0

        outcome = Tournament(task, n=4, judge=judge)(ws)
        assert outcome.committed
        assert outcome.winner.branch_index == 2
        assert len(judge_calls) == 0

    def test_odd_candidates(self):
        """3 candidates: one gets a bye in round 1."""
        ws = _make_workspace()

        def task(path: Path, index: int) -> bool:
            return True

        def judge(path_a: Path, path_b: Path) -> int:
            return 0  # always pick a

        outcome = Tournament(task, n=3, judge=judge)(ws)
        assert outcome.committed
        # Bracket: (0v1→0), bye 2, then (0v2→0)
        assert outcome.winner.branch_index == 0

    def test_commits_exactly_one(self):
        """Only the winner should be committed; all others aborted."""
        ws = _make_workspace()

        def task(path: Path, index: int) -> bool:
            return True

        def judge(path_a, path_b):
            return 0

        outcome = Tournament(task, n=4, judge=judge)(ws)
        assert outcome.committed
        assert len(MockFSBackend._commits) == 1
        assert len(MockFSBackend._aborts) == 3

    def test_runs_in_parallel(self):
        """Verify candidates actually run concurrently."""
        import time
        ws = _make_workspace()
        start = time.monotonic()

        def task(path: Path, index: int) -> bool:
            time.sleep(0.2)
            return True

        def judge(path_a, path_b):
            return 0

        outcome = Tournament(task, n=3, judge=judge)(ws)
        elapsed = time.monotonic() - start
        assert outcome.committed
        # 3 tasks @ 0.2s each; parallel ~0.2s, sequential ~0.6s
        assert elapsed < 0.5

    def test_exception_in_candidate(self):
        """Exception in one candidate → eliminated, others proceed."""
        ws = _make_workspace()

        def task(path: Path, index: int) -> bool:
            if index == 0:
                raise RuntimeError("boom")
            return True

        def judge(path_a, path_b):
            return 0

        outcome = Tournament(task, n=3, judge=judge)(ws)
        assert outcome.committed
        assert outcome.all_results[0].exception is not None
        assert outcome.winner.branch_index != 0


class TestCascaded:
    def test_succeeds_wave_0(self):
        """Task succeeds on first try — only 1 branch used."""
        ws = _make_workspace()

        def task(path, feedback):
            return True, ""

        outcome = Cascaded(task, fan_out=(1, 2, 4))(ws)
        assert outcome.committed
        assert outcome.winner is not None
        assert outcome.winner.branch_index == 0
        assert len(outcome.all_results) == 1

    def test_succeeds_wave_1(self):
        """Task fails wave 0, succeeds in wave 1."""
        ws = _make_workspace()

        def task(path, feedback):
            if not feedback:
                return False, "tests failed"
            return True, ""

        outcome = Cascaded(task, fan_out=(1, 2))(ws)
        assert outcome.committed
        # 1 result from wave 0 + 2 from wave 1
        assert len(outcome.all_results) == 3
        # Winner should be from wave 1 (global index 1 or 2)
        assert outcome.winner.branch_index >= 1

    def test_succeeds_wave_2(self):
        """Task fails waves 0 and 1, succeeds in wave 2."""
        ws = _make_workspace()

        def task(path, feedback):
            if len(feedback) < 2:
                return False, "not enough context"
            return True, ""

        outcome = Cascaded(task, fan_out=(1, 1, 4))(ws)
        assert outcome.committed
        # 1 + 1 + 4 = 6 total results
        assert len(outcome.all_results) == 6
        # Winner from wave 2 (global index 2..5)
        assert outcome.winner.branch_index >= 2

    def test_all_waves_fail(self):
        """All waves fail — nothing committed."""
        ws = _make_workspace()

        def task(path, feedback):
            return False, "nope"

        outcome = Cascaded(task, fan_out=(1, 2))(ws)
        assert not outcome.committed
        assert outcome.winner is None
        assert len(outcome.all_results) == 3

    def test_feedback_passed(self):
        """Error context is passed as feedback to subsequent waves."""
        ws = _make_workspace()
        received = []

        def task(path, feedback):
            received.append(list(feedback))
            if not feedback:
                return False, "error: test failed"
            return True, ""

        outcome = Cascaded(task, fan_out=(1, 1))(ws)
        assert outcome.committed
        # Wave 0: empty feedback
        assert received[0] == []
        # Wave 1: feedback from wave 0's error context
        assert received[1] == ["error: test failed"]

    def test_feedback_accumulates(self):
        """Error context accumulates across waves."""
        ws = _make_workspace()
        received = []

        call_count = [0]

        def task(path, feedback):
            received.append(list(feedback))
            call_count[0] += 1
            if len(feedback) < 2:
                return False, f"fail_{call_count[0]}"
            return True, ""

        # Use width=1 per wave to avoid threading non-determinism
        outcome = Cascaded(task, fan_out=(1, 1, 1))(ws)
        assert outcome.committed
        assert received[0] == []
        assert len(received[1]) == 1  # 1 error from wave 0
        assert len(received[2]) == 2  # 1 from wave 0 + 1 from wave 1

    def test_empty_error_not_collected(self):
        """Empty error strings are not added to feedback."""
        ws = _make_workspace()
        received = []
        call_count = [0]

        def task(path, feedback):
            received.append(list(feedback))
            call_count[0] += 1
            if call_count[0] == 1:
                return False, ""  # fail with no error context
            return True, ""

        outcome = Cascaded(task, fan_out=(1, 1))(ws)
        assert outcome.committed
        assert received[0] == []
        # Empty error string should not be collected
        assert received[1] == []

    def test_fan_out_scaling(self):
        """Correct number of branches per wave."""
        ws = _make_workspace()
        call_count = [0]

        def task(path, feedback):
            call_count[0] += 1
            return False, "err"

        outcome = Cascaded(task, fan_out=(1, 3, 5))(ws)
        assert not outcome.committed
        assert call_count[0] == 9
        assert len(outcome.all_results) == 9

    def test_exception_in_candidate(self):
        """Exception in wave 0 doesn't crash cascade; wave 1 succeeds."""
        ws = _make_workspace()
        call_count = [0]

        def task(path, feedback):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("boom")
            return True, ""

        outcome = Cascaded(task, fan_out=(1, 2))(ws)
        assert outcome.committed
        assert outcome.all_results[0].exception is not None

    def test_commits_exactly_one(self):
        """Only the winner is committed."""
        ws = _make_workspace()

        def task(path, feedback):
            return True, ""

        outcome = Cascaded(task, fan_out=(1,))(ws)
        assert outcome.committed
        assert len(MockFSBackend._commits) == 1
        assert len(MockFSBackend._aborts) == 0

    def test_wave_1_commits_exactly_one(self):
        """In wave with multiple candidates, only winner commits."""
        ws = _make_workspace()

        def task(path, feedback):
            if not feedback:
                return False, "err"
            return True, ""

        outcome = Cascaded(task, fan_out=(1, 3))(ws)
        assert outcome.committed
        # At least 1 commit (the winner) and at least 1 abort (wave 0)
        assert len(MockFSBackend._commits) >= 1
        assert len(MockFSBackend._aborts) >= 1

    def test_global_branch_indices(self):
        """Branch indices are globally unique across waves."""
        ws = _make_workspace()

        def task(path, feedback):
            if not feedback:
                return False, "err"
            return True, ""

        outcome = Cascaded(task, fan_out=(1, 2))(ws)
        indices = [r.branch_index for r in outcome.all_results]
        assert len(indices) == len(set(indices))  # all unique
        assert sorted(indices) == [0, 1, 2]

    def test_single_wave(self):
        """fan_out=(1,) behaves like a single attempt."""
        ws = _make_workspace()

        def task(path, feedback):
            return False, "err"

        outcome = Cascaded(task, fan_out=(1,))(ws)
        assert not outcome.committed
        assert len(outcome.all_results) == 1

    def test_runs_waves_in_parallel_within_wave(self):
        """Candidates within a wave run concurrently."""
        import time
        ws = _make_workspace()
        start = time.monotonic()

        def task(path, feedback):
            if not feedback:
                return False, "err"
            time.sleep(0.2)
            return True, ""

        outcome = Cascaded(task, fan_out=(1, 3))(ws)
        elapsed = time.monotonic() - start
        assert outcome.committed
        # 3 tasks @ 0.2s in parallel should be ~0.2s, not ~0.6s
        assert elapsed < 0.5

    def test_feedback_is_snapshot(self):
        """Each wave gets a snapshot of feedback, not a live reference."""
        ws = _make_workspace()
        received = []

        def task(path, feedback):
            received.append(feedback)
            return False, "err"

        outcome = Cascaded(task, fan_out=(1, 1, 1))(ws)
        assert not outcome.committed
        # Each wave's feedback should be an independent list
        assert received[0] is not received[1]
        if len(received) > 2:
            assert received[1] is not received[2]


class TestSpeculationResult:
    def test_dataclass(self):
        r = SpeculationResult(branch_index=0, success=True, score=0.95)
        assert r.branch_index == 0
        assert r.success
        assert r.score == 0.95

    def test_outcome_defaults(self):
        o = SpeculationOutcome()
        assert o.winner is None
        assert o.all_results == []
        assert not o.committed
