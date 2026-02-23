#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fix a buggy module: one LLM call, multiple branches.

The natural pattern for connecting LLMs with speculative execution:

  1. One LLM call with n=N  -> N candidate fixes  (cheap)
  2. N isolated branches    -> test each candidate (safe)
  3. First passing branch   -> committed to workspace (fast)

The ``n=`` parameter in the OpenAI Chat Completions API samples N
independent completions from a single prompt.  Prompt tokens are
charged once; completion tokens are charged N times.  This is much
cheaper and faster than making N separate API calls.

Usage:
    # With OpenAI
    export OPENAI_API_KEY=sk-...
    python examples/speculate_fix.py /mnt/workspace

    # With a local model (Ollama)
    export OPENAI_BASE_URL=http://localhost:11434/v1
    export OPENAI_API_KEY=unused
    python examples/speculate_fix.py /mnt/workspace --model llama3

    # Without API keys (scripted candidates for demo)
    python examples/speculate_fix.py /mnt/workspace --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

from branching import Workspace, Speculate

# ---------------------------------------------------------------------------
# Buggy module + test suite
#
# Four functions, four bugs — two easy, two hard:
#   interleave:            zip truncates the longer list (easy)
#   uniq_count:            resets count to 0 instead of 1 (easy)
#   spiral_order:          missing boundary checks causes duplicates (hard)
#   max_subarray_circular: wrong answer when all elements negative (hard)
# ---------------------------------------------------------------------------

BUGGY_CODE = dedent("""\
    def interleave(a, b):
        \"\"\"Interleave two lists, appending any remainder from the longer list.\"\"\"
        result = []
        for x, y in zip(a, b):
            result.append(x)
            result.append(y)
        return result


    def uniq_count(items):
        \"\"\"Count consecutive runs.  Return list of (value, count) pairs.\"\"\"
        if not items:
            return []
        result = []
        current = items[0]
        count = 1
        for item in items[1:]:
            if item == current:
                count += 1
            else:
                result.append((current, count))
                current = item
                count = 0
        result.append((current, count))
        return result


    def spiral_order(matrix):
        \"\"\"Return elements of an M x N matrix in clockwise spiral order.\"\"\"
        if not matrix or not matrix[0]:
            return []
        result = []
        top, bottom = 0, len(matrix) - 1
        left, right = 0, len(matrix[0]) - 1
        while top <= bottom and left <= right:
            for col in range(left, right + 1):
                result.append(matrix[top][col])
            top += 1
            for row in range(top, bottom + 1):
                result.append(matrix[row][right])
            right -= 1
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1
        return result


    def max_subarray_circular(nums):
        \"\"\"Maximum sum of a contiguous subarray in a circular array.\"\"\"
        max_sum = cur_max = nums[0]
        min_sum = cur_min = nums[0]
        total = nums[0]
        for x in nums[1:]:
            cur_max = max(x, cur_max + x)
            max_sum = max(max_sum, cur_max)
            cur_min = min(x, cur_min + x)
            min_sum = min(min_sum, cur_min)
            total += x
        return max(max_sum, total - min_sum)
""")

TEST_CODE = dedent("""\
    from listutil import interleave, uniq_count, spiral_order, max_subarray_circular

    # -- interleave --

    def test_interleave_equal():
        assert interleave([1, 2], ['a', 'b']) == [1, 'a', 2, 'b']

    def test_interleave_first_longer():
        assert interleave([1, 2, 3], ['a', 'b']) == [1, 'a', 2, 'b', 3]

    def test_interleave_second_longer():
        assert interleave([1], ['a', 'b', 'c']) == [1, 'a', 'b', 'c']

    def test_interleave_empty():
        assert interleave([], [1, 2]) == [1, 2]

    # -- uniq_count --

    def test_uniq_basic():
        assert uniq_count([1, 1, 2, 3, 3, 3, 1]) == [(1, 2), (2, 1), (3, 3), (1, 1)]

    def test_uniq_single():
        assert uniq_count([5]) == [(5, 1)]

    def test_uniq_no_repeats():
        assert uniq_count([1, 2, 3]) == [(1, 1), (2, 1), (3, 1)]

    def test_uniq_empty():
        assert uniq_count([]) == []

    # -- spiral_order --

    def test_spiral_3x3():
        m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        assert spiral_order(m) == [1, 2, 3, 6, 9, 8, 7, 4, 5]

    def test_spiral_1x4():
        m = [[1, 2, 3, 4]]
        assert spiral_order(m) == [1, 2, 3, 4]

    def test_spiral_4x1():
        m = [[1], [2], [3], [4]]
        assert spiral_order(m) == [1, 2, 3, 4]

    def test_spiral_2x3():
        m = [[1, 2, 3], [4, 5, 6]]
        assert spiral_order(m) == [1, 2, 3, 6, 5, 4]

    def test_spiral_empty():
        assert spiral_order([]) == []

    # -- max_subarray_circular --

    def test_circular_basic():
        assert max_subarray_circular([-2, 1, 3, -1]) == 4

    def test_circular_wrap():
        assert max_subarray_circular([5, -3, 5]) == 10

    def test_circular_all_negative():
        assert max_subarray_circular([-3, -1, -2]) == -1

    def test_circular_single():
        assert max_subarray_circular([42]) == 42

    def test_circular_single_negative():
        assert max_subarray_circular([-5]) == -5
""")

# ---------------------------------------------------------------------------
# Step 1: One LLM call -> N candidate fixes
# ---------------------------------------------------------------------------


def generate_candidates_llm(code: str, n: int, model: str) -> list[str]:
    """One LLM call with n= completions -> N candidate fixes."""
    import openai

    prompt = (
        "Fix all bugs in this Python module.\n\n"
        f"```python\n{code}```\n\n"
        "Return ONLY the corrected Python code. No markdown fences."
    )

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        n=n,
        temperature=0.8,
    )

    candidates = []
    for choice in resp.choices:
        text = choice.message.content
        # Strip markdown fences if the model wraps its output anyway.
        m = re.search(r"```(?:python)?\s*\n(.+?)```", text, re.DOTALL)
        candidates.append(m.group(1) if m else text)
    return candidates


def generate_candidates_dry_run(code: str, n: int) -> list[str]:
    """Scripted candidates simulating varying LLM quality."""
    candidates = []

    def _fix_interleave(c):
        return c.replace(
            "        result.append(y)\n    return result",
            "        result.append(y)\n"
            "    longer = a[len(b):] if len(a) > len(b) else b[len(a):]\n"
            "    result.extend(longer)\n"
            "    return result",
        )

    def _fix_uniq(c):
        return c.replace("            count = 0", "            count = 1")

    def _fix_spiral(c):
        return c.replace(
            "        for col in range(right, left - 1, -1):\n"
            "            result.append(matrix[bottom][col])\n"
            "        bottom -= 1\n"
            "        for row in range(bottom, top - 1, -1):\n"
            "            result.append(matrix[row][left])",
            "        if top <= bottom:\n"
            "            for col in range(right, left - 1, -1):\n"
            "                result.append(matrix[bottom][col])\n"
            "        bottom -= 1\n"
            "        if left <= right:\n"
            "            for row in range(bottom, top - 1, -1):\n"
            "                result.append(matrix[row][left])",
        )

    def _fix_circular(c):
        return c.replace(
            "    return max(max_sum, total - min_sum)",
            "    if max_sum < 0:\n"
            "        return max_sum\n"
            "    return max(max_sum, total - min_sum)",
        )

    easy = _fix_uniq(_fix_interleave(code))

    # Candidate 0: fixes the two easy bugs only
    candidates.append(easy)

    # Candidate 1: fixes easy + spiral, misses circular edge case
    candidates.append(_fix_spiral(easy))

    # Candidate 2: fixes easy + circular, misses spiral
    candidates.append(_fix_circular(easy))

    # Candidate 3: fixes all four bugs
    candidates.append(_fix_circular(_fix_spiral(easy)))

    return candidates[:n]


# ---------------------------------------------------------------------------
# Step 2: Map each candidate to a branch-testable function
# ---------------------------------------------------------------------------


def make_candidate(fixed_code: str):
    """Wrap a candidate fix string into a callable for Speculate."""

    def test_fix(path: Path) -> bool:
        (path / "listutil.py").write_text(fixed_code)
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "--tb=short",
             "test_listutil.py"],
            cwd=path, capture_output=True, text=True, timeout=30,
        )
        return result.returncode == 0

    return test_fix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fix a buggy module: one LLM call, N branches.",
    )
    parser.add_argument("workspace", help="BranchFS/DaxFS workspace path")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model name (default: gpt-4o-mini)")
    parser.add_argument("-n", type=int, default=4,
                        help="Number of completions to request (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use scripted fixes instead of LLM")
    args = parser.parse_args()

    ws_path = Path(args.workspace)
    ws = Workspace(ws_path)

    # Seed workspace with buggy code and tests.
    (ws_path / "listutil.py").write_text(BUGGY_CODE)
    (ws_path / "test_listutil.py").write_text(TEST_CODE)

    mode = "dry-run (scripted)" if args.dry_run else f"LLM ({args.model}, n={args.n})"
    print(f"Workspace: {ws}")
    print(f"Mode: {mode}")
    print("Bugs: listutil.py has 4 bugs (2 easy, 2 hard)")
    print()

    # --- Step 1: One LLM call -> N candidate fixes ---
    print(f"Step 1: Generating {args.n} candidate fixes (1 API call)...")
    if args.dry_run:
        fixes = generate_candidates_dry_run(BUGGY_CODE, args.n)
    else:
        fixes = generate_candidates_llm(BUGGY_CODE, args.n, args.model)
    print(f"  Got {len(fixes)} candidates")
    print()

    # --- Step 2: Test each candidate in an isolated branch ---
    print(f"Step 2: Testing each candidate in an isolated branch...")
    candidates = [make_candidate(fix) for fix in fixes]
    outcome = Speculate(candidates, first_wins=True, timeout=60)(ws)

    # --- Results ---
    print()
    for r in outcome.all_results:
        status = "PASS" if r.success else "FAIL"
        committed = " (committed)" if outcome.winner is r else ""
        print(f"  candidate {r.branch_index}: {status}{committed}")

    if outcome.committed:
        print(f"\nFixed! Candidate {outcome.winner.branch_index} passed all tests.")
    else:
        print("\nNo candidate passed all tests.")
        sys.exit(1)


if __name__ == "__main__":
    main()
