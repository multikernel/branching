#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parallel speculation example — first successful fix wins.

Requires a mounted branchfs or daxfs filesystem.

Usage:
    python parallel_speculation.py /mnt/workspace
"""

import subprocess
import sys
import time
from pathlib import Path

from branching import Workspace, Speculate


def try_fix_a(path: Path) -> bool:
    """Apply fix A and test."""
    print(f"  [Fix A] Working in {path}")
    (path / "fix_a.txt").write_text("applied fix A\n")
    time.sleep(0.5)  # Simulate work
    return True  # Pretend tests pass


def try_fix_b(path: Path) -> bool:
    """Apply fix B and test."""
    print(f"  [Fix B] Working in {path}")
    (path / "fix_b.txt").write_text("applied fix B\n")
    time.sleep(1.0)  # Slower fix
    return True


def try_fix_c(path: Path) -> bool:
    """Apply fix C — this one fails."""
    print(f"  [Fix C] Working in {path}")
    (path / "fix_c.txt").write_text("applied fix C\n")
    return False  # Tests fail


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <workspace_path>")
        sys.exit(1)

    ws = Workspace(sys.argv[1])
    print(f"Opened workspace: {ws}")

    # Run 3 candidates in parallel, first success wins
    print("\n--- Parallel speculation (first wins) ---")
    spec = Speculate(
        [try_fix_a, try_fix_b, try_fix_c],
        first_wins=True,
        timeout=30,
    )
    outcome = spec(ws)

    if outcome.committed:
        print(f"\nWinner: candidate {outcome.winner.branch_index}")
    else:
        print("\nAll candidates failed!")

    print(f"\nResults:")
    for r in outcome.all_results:
        status = "SUCCESS" if r.success else "FAILED"
        print(f"  Candidate {r.branch_index}: {status}")


if __name__ == "__main__":
    main()
