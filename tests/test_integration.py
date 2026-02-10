# SPDX-License-Identifier: Apache-2.0
"""Integration tests against a live BranchFS mount.

These tests require a running branchfs daemon. They are skipped
automatically when no mount is available (e.g. unit-test-only runs).

The BRANCHFS_MOUNT environment variable specifies the mount point.
"""

import os
import subprocess
from pathlib import Path

import pytest

needs_branchfs = pytest.mark.skipif(
    not os.environ.get("BRANCHFS_MOUNT"),
    reason="BRANCHFS_MOUNT not set (no live branchfs)",
)


@pytest.fixture
def ws():
    from branching import Workspace

    return Workspace(os.environ["BRANCHFS_MOUNT"])


@needs_branchfs
def test_workspace_detects_branchfs(ws):
    assert ws.fstype == "fuse.branchfs"
    assert ws.path.exists()


@needs_branchfs
def test_write_and_commit(ws):
    """Write a file in a branch, commit, verify it appears in main."""
    marker = f"integration_test_{os.getpid()}"
    filename = f"{marker}.txt"

    with ws.branch("write_commit") as b:
        (b.path / filename).write_text("hello from branch")
        b.commit()

    assert (ws.path / filename).read_text() == "hello from branch"
    # Cleanup
    (ws.path / filename).unlink()


@needs_branchfs
def test_abort_discards_changes(ws):
    """Write a file in a branch, abort, verify it does not appear in main."""
    marker = f"integration_abort_{os.getpid()}"
    filename = f"{marker}.txt"

    with ws.branch("abort_test", on_success=None, on_error=None) as b:
        (b.path / filename).write_text("should not persist")
        b.abort()

    assert not (ws.path / filename).exists()


@needs_branchfs
def test_nested_branches(ws):
    """Create a nested branch, commit inner then outer."""
    marker = f"integration_nested_{os.getpid()}"
    filename = f"{marker}.txt"

    with ws.branch("outer") as outer:
        with outer.branch("inner") as inner:
            (inner.path / filename).write_text("nested")
            inner.commit()
        assert (outer.path / filename).read_text() == "nested"
        outer.commit()

    assert (ws.path / filename).read_text() == "nested"
    (ws.path / filename).unlink()


@needs_branchfs
def test_sibling_branches_first_wins(ws):
    """Two sibling branches race to commit; first wins, second gets ConflictError."""
    from branching.exceptions import ConflictError

    marker = f"integration_sibling_{os.getpid()}"

    with ws.branch("sibling_a", on_success=None, on_error=None) as a:
        with ws.branch("sibling_b", on_success=None, on_error=None) as b_branch:
            (a.path / f"{marker}_a.txt").write_text("a wins")
            (b_branch.path / f"{marker}_b.txt").write_text("b loses")

            a.commit()

            with pytest.raises(ConflictError):
                b_branch.commit()

    assert (ws.path / f"{marker}_a.txt").read_text() == "a wins"
    assert not (ws.path / f"{marker}_b.txt").exists()
    (ws.path / f"{marker}_a.txt").unlink()


@needs_branchfs
def test_speculate_parallel(ws):
    """Speculate runs candidates in parallel, commits the winner."""
    from branching import Speculate

    marker = f"integration_spec_{os.getpid()}"

    def candidate_pass(path: Path) -> bool:
        (path / f"{marker}.txt").write_text("speculate winner")
        return True

    def candidate_fail(path: Path) -> bool:
        return False

    outcome = Speculate([candidate_pass, candidate_fail], first_wins=True)(ws)
    assert outcome.committed
    assert outcome.winner.branch_index == 0
    assert (ws.path / f"{marker}.txt").read_text() == "speculate winner"
    (ws.path / f"{marker}.txt").unlink()


@needs_branchfs
def test_tree_of_thoughts_multi_level(ws):
    """TreeOfThoughts with expand commits accumulated state across levels."""
    from branching import TreeOfThoughts

    marker = f"integration_tot_{os.getpid()}"

    # Level 0: two strategies write different files; strat_a scores higher
    def strat_a(path: Path) -> tuple[bool, float]:
        (path / f"{marker}_level0.txt").write_text("strat_a")
        return True, 0.9

    def strat_b(path: Path) -> tuple[bool, float]:
        (path / f"{marker}_level0.txt").write_text("strat_b")
        return True, 0.3

    # Level 1: refinements that read the level-0 result and build on it
    def refine_x(path: Path) -> tuple[bool, float]:
        prev = (path / f"{marker}_level0.txt").read_text()
        (path / f"{marker}_level1.txt").write_text(f"{prev}+refine_x")
        return True, 0.8

    def refine_y(path: Path) -> tuple[bool, float]:
        prev = (path / f"{marker}_level0.txt").read_text()
        (path / f"{marker}_level1.txt").write_text(f"{prev}+refine_y")
        return True, 0.5

    outcome = TreeOfThoughts(
        [strat_a, strat_b],
        expand=lambda path, depth: [refine_x, refine_y],
        max_depth=2,
    )(ws)

    assert outcome.committed
    # Level 0 winner: strat_a (0.9); level 1 winner: refine_x (0.8)
    assert outcome.winner.score == 0.8
    assert (ws.path / f"{marker}_level0.txt").read_text() == "strat_a"
    assert (ws.path / f"{marker}_level1.txt").read_text() == "strat_a+refine_x"

    # Cleanup
    (ws.path / f"{marker}_level0.txt").unlink()
    (ws.path / f"{marker}_level1.txt").unlink()
