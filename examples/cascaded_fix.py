#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fix a buggy module with cascaded speculation and an LLM.

Cascaded starts with one cheap LLM call and escalates on failure,
feeding accumulated test errors into the next wave.  Each attempt
asks the model to fix the code, informed by all prior errors.

  Wave 0: model sees buggy code, no errors    → might miss subtle bugs
  Wave 1: model sees code + wave 0's error    → more targeted fix
  Wave 2: model sees all accumulated errors   → most likely succeeds

Most bugs get fixed on wave 0 (one LLM call).  You only pay for
extra calls when the first attempt fails.

Supports any OpenAI-compatible API (OpenAI, Ollama, etc.).
Use --dry-run to test the pattern without API keys.

Usage:
    # With OpenAI
    export OPENAI_API_KEY=sk-...
    python examples/cascaded_fix.py /mnt/workspace

    # With a local model (Ollama)
    export OPENAI_BASE_URL=http://localhost:11434/v1
    export OPENAI_API_KEY=unused
    python examples/cascaded_fix.py /mnt/workspace --model llama3

    # Without API keys (scripted fixes for demo)
    python examples/cascaded_fix.py /mnt/workspace --dry-run
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

from branching import Workspace, Cascaded

# ---------------------------------------------------------------------------
# Buggy module + test suite
# ---------------------------------------------------------------------------

BUGGY_CODE = dedent("""\
    def safe_divide(a, b):
        \"\"\"Divide a by b, returning None if b is zero.\"\"\"
        return a / b

    def average(nums):
        \"\"\"Return the mean of nums, or 0.0 for an empty list.\"\"\"
        return sum(nums) / len(nums)

    def clamp(x, lo, hi):
        \"\"\"Clamp x to the range [lo, hi].\"\"\"
        if x < lo:
            return lo
        if x < hi:
            return hi
        return x
""")

TEST_CODE = dedent("""\
    from mathutil import safe_divide, average, clamp

    def test_divide():
        assert safe_divide(10, 2) == 5.0

    def test_divide_zero():
        assert safe_divide(10, 0) is None

    def test_average():
        assert average([1, 2, 3]) == 2.0

    def test_average_empty():
        assert average([]) == 0.0

    def test_clamp_low():
        assert clamp(1, 5, 10) == 5

    def test_clamp_high():
        assert clamp(15, 5, 10) == 10

    def test_clamp_mid():
        assert clamp(7, 5, 10) == 7
""")

# ---------------------------------------------------------------------------
# LLM + dry-run backends
# ---------------------------------------------------------------------------


def _call_llm(code: str, feedback: list[str], model: str) -> str:
    """Ask an LLM to fix the code, informed by prior test errors."""
    import openai

    prompt = f"Fix the bugs in this Python module:\n\n```python\n{code}```\n"
    if feedback:
        prompt += "\nPrevious fix attempts failed with these test errors:\n"
        for i, err in enumerate(feedback):
            prompt += f"\n--- attempt {i} ---\n{err}\n"
    prompt += "\nReturn ONLY the corrected Python code. No markdown fences, no explanation."

    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    text = resp.choices[0].message.content
    # Strip markdown fences if the model wraps its output anyway.
    m = re.search(r"```(?:python)?\s*\n(.+?)```", text, re.DOTALL)
    return m.group(1) if m else text


def _scripted_fix(code: str, feedback: list[str]) -> str:
    """Dry-run: apply deterministic fixes based on error keywords."""
    code = code.replace(
        "    return a / b",
        "    if b == 0:\n        return None\n    return a / b",
    )
    if any("average" in e or "ZeroDivisionError" in e for e in feedback):
        code = code.replace(
            "    return sum(nums) / len(nums)",
            "    if not nums:\n        return 0.0\n    return sum(nums) / len(nums)",
        )
    if any("clamp" in e for e in feedback):
        code = code.replace("    if x < hi:", "    if x > hi:")
    return code


# ---------------------------------------------------------------------------
# Cascaded task
# ---------------------------------------------------------------------------


def make_task(model: str, dry_run: bool):
    """Build a Cascaded task that fixes code via LLM (or scripted fallback)."""

    def task(path: Path, feedback: list[str]) -> tuple[bool, str]:
        code = (path / "mathutil.py").read_text()

        if dry_run:
            fixed = _scripted_fix(code, feedback)
        else:
            fixed = _call_llm(code, feedback, model)

        (path / "mathutil.py").write_text(fixed)

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "--tb=short", "test_mathutil.py"],
            cwd=path, capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stdout + result.stderr).strip()

    return task


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Fix a buggy module with Cascaded speculation + LLM.",
    )
    parser.add_argument("workspace", help="BranchFS/DaxFS workspace path")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model name (default: gpt-4o-mini)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use scripted fixes instead of LLM (no API keys needed)")
    args = parser.parse_args()

    ws_path = Path(args.workspace)
    ws = Workspace(ws_path)

    # Seed workspace with buggy code and tests.
    (ws_path / "mathutil.py").write_text(BUGGY_CODE)
    (ws_path / "test_mathutil.py").write_text(TEST_CODE)

    mode = "dry-run (scripted fixes)" if args.dry_run else f"LLM ({args.model})"
    print(f"Workspace: {ws}")
    print(f"Mode: {mode}")
    print("Bug: mathutil.py has 3 bugs (divide-by-zero, empty list, wrong comparison)")
    print("Pattern: Cascaded — try cheap, escalate with error feedback\n")

    task = make_task(model=args.model, dry_run=args.dry_run)
    outcome = Cascaded(task, fan_out=(1, 1, 1), timeout=120)(ws)

    for r in outcome.all_results:
        status = "pass" if r.success else "FAIL"
        committed = " (committed)" if r.success and outcome.committed else ""
        print(f"  attempt {r.branch_index}: {status}{committed}")

    if outcome.committed:
        print(f"\nFixed on attempt {outcome.winner.branch_index}")
    else:
        print("\nAll attempts failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
