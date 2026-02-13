#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Race different LLM models, first correct solution wins.

Sends the same coding problem to GPT-4o, Claude, and DeepSeek in
parallel via Speculate.  Each model's solution runs in its own
filesystem branch and gets validated against a test suite.  The first
model to produce a passing solution wins and gets committed.

Different models produce genuinely different code with different algorithms,
different edge-case handling, different performance characteristics.
This is the pattern used by Cursor 2.0 (up to 8 parallel agents, each
potentially a different model) and Agentless (multiple candidate patches).

The task: implement longest_increasing_subsequence(nums).

Requires a mounted branchfs or daxfs filesystem and at least one
LLM provider API key (ideally all three).

Usage:
    export DEEPSEEK_API_KEY=sk-...
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-...
    python examples/race_models.py /mnt/workspace

Pass --dummy to use hard-coded solutions (no API calls needed).
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

from branching import Workspace, Speculate

# ---------------------------------------------------------------------------
# Problem + tests + benchmark
# ---------------------------------------------------------------------------

PROBLEM = dedent("""\
    def lis(nums: list[int]) -> int:
        \"\"\"Return the length of the longest strictly increasing subsequence.

        >>> lis([10, 9, 2, 5, 3, 7, 101, 18])
        4   # e.g. [2, 3, 7, 101]
        >>> lis([0, 1, 0, 3, 2, 3])
        4   # [0, 1, 2, 3]
        >>> lis([7, 7, 7, 7])
        1
        \"\"\"
""")

TEST_CODE = dedent("""\
    from solution import lis

    def test_basic():
        assert lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4

    def test_increasing():
        assert lis([1, 2, 3, 4, 5]) == 5

    def test_decreasing():
        assert lis([5, 4, 3, 2, 1]) == 1

    def test_duplicates():
        assert lis([7, 7, 7, 7]) == 1

    def test_mixed():
        assert lis([0, 1, 0, 3, 2, 3]) == 4

    def test_single():
        assert lis([42]) == 1

    def test_empty():
        assert lis([]) == 0

    def test_longer():
        assert lis([1, 3, 6, 7, 9, 4, 10, 5, 6]) == 6

    def test_negative():
        assert lis([-2, -1, 0, 1]) == 4

    def test_two_elements():
        assert lis([2, 1]) == 1
        assert lis([1, 2]) == 2
""")

# Benchmark: time the solution on a large random input.
# O(n²) on 5000 elements ≈ seconds; O(n log n) ≈ milliseconds.
BENCHMARK_CODE = dedent("""\
    import time, random
    from solution import lis

    random.seed(42)
    data = [random.randint(1, 100000) for _ in range(5000)]

    start = time.perf_counter()
    for _ in range(3):
        lis(data)
    elapsed = time.perf_counter() - start
    print(f"{elapsed:.6f}")
""")

PROMPT = (
    "Implement this Python function. Reply with ONLY the function "
    "inside a ```python``` block.\n\n"
    f"```python\n{PROBLEM}```"
)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------


def _call_deepseek(prompt: str) -> str:
    import openai

    client = openai.OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def _call_openai(prompt: str) -> str:
    import openai

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content


def _call_anthropic(prompt: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text


# All models, keyed by display name.  Filtered at runtime by which
# API keys are set.
ALL_MODELS = {
    "deepseek": ("DEEPSEEK_API_KEY", _call_deepseek),
    "gpt-4o": ("OPENAI_API_KEY", _call_openai),
    "claude": ("ANTHROPIC_API_KEY", _call_anthropic),
}


def detect_models() -> dict[str, callable]:
    """Return {name: call_fn} for every model whose API key is set."""
    return {
        name: fn
        for name, (env_var, fn) in ALL_MODELS.items()
        if os.environ.get(env_var)
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def extract_python(llm_response: str) -> str | None:
    """Extract the first ```python ... ``` block from an LLM response."""
    m = re.search(r"```python\s*\n(.+?)```", llm_response, re.DOTALL)
    return m.group(1).strip() if m else None


def run_pytest(workdir: Path) -> tuple[bool, str]:
    """Run pytest in *workdir* and return (passed, output)."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "test_solution.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def run_benchmark(workdir: Path) -> float:
    """Run the benchmark script; return elapsed seconds (inf on failure)."""
    result = subprocess.run(
        [sys.executable, "benchmark.py"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    try:
        return float(result.stdout.strip())
    except (ValueError, AttributeError):
        return float("inf")


# ---------------------------------------------------------------------------
# Candidate builder — one per model
# ---------------------------------------------------------------------------


def make_candidate(model_name: str, call_fn):
    """Build a Speculate candidate that calls a specific model."""

    def candidate(path: Path) -> bool:
        (path / "test_solution.py").write_text(TEST_CODE)
        (path / "benchmark.py").write_text(BENCHMARK_CODE)

        # Generate solution
        print(f"  [{model_name}] Calling model ...")
        response = call_fn(PROMPT)
        code = extract_python(response)
        if not code:
            print(f"  [{model_name}] Could not extract code")
            return False

        (path / "solution.py").write_text(code + "\n")

        # Correctness check
        passed, output = run_pytest(path)
        if not passed:
            print(f"  [{model_name}] FAILED correctness tests")
            for line in output.splitlines():
                if "FAILED" in line or "assert" in line.lower():
                    print(f"  [{model_name}]   {line.strip()}")
                    break
            return False

        # Benchmark
        elapsed = run_benchmark(path)
        print(f"  [{model_name}] PASSED  {elapsed:.4f}s")
        return True

    return candidate


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <workspace_path> [--dummy]")
        sys.exit(1)

    workspace_path = sys.argv[1]
    use_dummy = "--dummy" in sys.argv

    if use_dummy:
        # Dummy mode: simulate 3 models with hard-coded solutions
        models = {
            "dummy-fast": lambda _: dedent("""\
                ```python
                from bisect import bisect_left
                def lis(nums: list[int]) -> int:
                    if not nums:
                        return 0
                    tails = []
                    for x in nums:
                        pos = bisect_left(tails, x)
                        if pos == len(tails):
                            tails.append(x)
                        else:
                            tails[pos] = x
                    return len(tails)
                ```
            """),
            "dummy-slow": lambda _: dedent("""\
                ```python
                def lis(nums: list[int]) -> int:
                    if not nums:
                        return 0
                    n = len(nums)
                    dp = [1] * n
                    for i in range(1, n):
                        for j in range(i):
                            if nums[j] < nums[i]:
                                dp[i] = max(dp[i], dp[j] + 1)
                    return max(dp)
                ```
            """),
        }
        print("Using dummy models (hard-coded solutions, no API calls)")
    else:
        models = detect_models()
        if not models:
            print(
                "No API keys found. Set at least one of:\n"
                "  DEEPSEEK_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY\n"
                "Or pass --dummy."
            )
            sys.exit(1)
        if len(models) < 2:
            print(
                f"Only 1 model available ({list(models)[0]}). "
                "Set more API keys to race multiple models."
            )

    ws = Workspace(workspace_path)
    print(f"Opened workspace: {ws}")

    names = list(models.keys())
    candidates = [make_candidate(name, fn) for name, fn in models.items()]

    print(f"\n--- Racing {len(candidates)} models in parallel ---")
    for name in names:
        print(f"  • {name}")

    spec = Speculate(candidates, first_wins=True, timeout=120)
    outcome = spec(ws)

    # Report
    print("\nResults:")
    for r in outcome.all_results:
        name = names[r.branch_index]
        status = "PASSED" if r.success else "FAILED"
        detail = f" — {r.exception}" if r.exception else ""
        print(f"  {name}: {status}{detail}")

    if outcome.committed:
        winner = names[outcome.winner.branch_index]
        print(f"\nWinner: {winner}")
        print("Solution committed to workspace.")
    else:
        print("\nAll models failed — nothing committed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
