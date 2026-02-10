#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Simple branch example — speculative execution with auto-commit/abort.

Requires a mounted branchfs or daxfs filesystem.

Usage:
    python simple_branch.py /mnt/workspace
"""

import subprocess
import sys
from pathlib import Path

from branching import Workspace


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <workspace_path>")
        sys.exit(1)

    ws = Workspace(sys.argv[1])
    print(f"Opened workspace: {ws}")

    # Pattern 1: Auto-commit on success, auto-abort on exception
    print("\n--- Pattern 1: Auto-commit/abort ---")
    with ws.branch("attempt1") as b:
        print(f"  Branch path: {b.path}")
        # Write a test file
        (b.path / "result.txt").write_text("success\n")
        print("  Wrote result.txt — will auto-commit on clean exit")
    print("  Branch committed!")

    # Pattern 2: Manual control
    print("\n--- Pattern 2: Manual control ---")
    with ws.branch("attempt2", on_success=None, on_error=None) as b:
        print(f"  Branch path: {b.path}")
        (b.path / "speculative.txt").write_text("maybe\n")

        # Agent decides
        confident = True
        if confident:
            b.commit()
            print("  Manually committed")
        else:
            b.abort()
            print("  Manually aborted")

    # Pattern 3: Nested branches
    print("\n--- Pattern 3: Nested branches ---")
    with ws.branch("strategy_a") as a:
        print(f"  Strategy A path: {a.path}")
        (a.path / "base_change.txt").write_text("strategy A base\n")

        with a.branch("variant_1") as v1:
            print(f"    Variant 1 path: {v1.path}")
            (v1.path / "tweak.txt").write_text("variant 1\n")
            # Auto-commits into strategy_a
        print("    Variant 1 committed into strategy A")
    print("  Strategy A committed into main")


if __name__ == "__main__":
    main()
