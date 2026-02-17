# branching — Copy-on-Write Workspace Branching for AI Agents

Explore multiple strategies without consequences. Fork the workspace into
parallel copy-on-write branches, run speculative attempts in each, commit
the winner, and abort the rest — instantly.

## Workspace

A workspace is a directory backed by a copy-on-write filesystem (branchfs
or daxfs). All branching operations happen within a workspace.

**Prerequisite:** The workspace must be mounted before use. Two backends
are supported, auto-detected at runtime:

| Backend | Mount |
|---------|-------|
| [BranchFS](https://github.com/multikernel/branchfs) (FUSE) | `branchfs /mnt/workspace` |
| [DaxFS](https://github.com/multikernel/daxfs) (kernel) | `mount -t daxfs <device> /mnt/workspace` |

The CLI auto-detects the workspace from your current directory if you are
inside a branchfs/daxfs mount. Otherwise pass `-w /mnt/workspace`.

## CLI

### branching run — single branch execution

Run a command in an isolated branch. Commits on exit 0, aborts on non-zero.

```bash
branching run -- ./build.sh
branching run --ask -- make test          # prompt before commit/abort
branching run --on-error none -- python train.py  # keep branch on failure
branching run --memory-limit 512M -- ./agent.sh
```

### branching speculate — race N approaches

Race multiple commands in parallel. First success wins, rest are aborted.

Use when multiple approaches could work and you want the fastest success.

```bash
branching speculate -c "./fix_a.sh" -c "./fix_b.sh" -c "./fix_c.sh"
branching speculate --timeout 60 -c "python v1.py" -c "python v2.py"
```

### branching best-of-n — parallel with scoring

Run the same command N times in parallel. Commit the highest-scoring success.
Child writes score to fd 3 (`echo 0.95 >&3`). Gets `BRANCHING_ATTEMPT` env var.

Use when quality matters: pick the best result across multiple attempts.

```bash
branching best-of-n -n 5 -- ./solve.py
branching best-of-n -n 3 -- bash -c 'python run.py && echo "$SCORE" >&3'
```

### branching reflexion — retry with feedback

Sequential retry with critique. Child gets `BRANCHING_ATTEMPT` and
`BRANCHING_FEEDBACK` (empty on first try, critique output on retries).

Use when failure output helps improve the next attempt.

```bash
branching reflexion --retries 5 -- ./fix.sh
branching reflexion --retries 3 --critique "./review.sh" -- ./solve.py
```

### branching status — workspace info

```bash
branching status
branching status --json
```

### Common CLI flags

| Flag | Description |
|------|-------------|
| `-w PATH` | Workspace path (auto-detected from cwd if omitted) |
| `-b NAME` | Branch name (auto-generated if omitted) |
| `--json` | Machine-readable JSON output |
| `--timeout SEC` | Timeout for parallel commands |
| `--memory-limit SIZE` | Per-branch memory cap (e.g. `512M`, `1G`) |
| `--cpu-limit FRAC` | Per-branch CPU limit (e.g. `0.5` = 50%) |
| `--group-memory-limit SIZE` | Total memory budget for all branches |
| `--group-cpu-limit FRAC` | Total CPU budget for all branches |

## Python API

Install: `pip install branchcontext`

### Workspace and Branch

```python
from branching import Workspace

ws = Workspace("/mnt/workspace")

# Auto-commit on success, auto-abort on exception
with ws.branch("attempt") as b:
    # b.path is an isolated copy-on-write view of the workspace
    subprocess.run(["agent", "--workdir", str(b.path)], check=True)

# Manual control
with ws.branch("attempt", on_success=None, on_error=None) as b:
    result = run_agent(workdir=b.path)
    if result.confident:
        b.commit()
    else:
        b.abort()

# Nested branches
with ws.branch("strategy_a") as a:
    apply_strategy(a.path)
    with a.branch("variant_1") as v1:
        tweak(v1.path)
        # v1 auto-commits into a on success
    # a auto-commits into workspace on success
```

### Speculate — race N candidates, first wins

```python
from branching import Workspace, Speculate

def try_fix_a(path: Path) -> bool:
    apply_patch(path / "a.patch")
    return run_tests(path)

def try_fix_b(path: Path) -> bool:
    apply_patch(path / "b.patch")
    return run_tests(path)

outcome = Speculate([try_fix_a, try_fix_b], first_wins=True, timeout=60)(ws)
if outcome.committed:
    print(f"Fix {outcome.winner.branch_index} succeeded!")
```

### BestOfN — parallel attempts, highest score wins

```python
from branching import BestOfN

def scored_task(path: Path, attempt: int) -> tuple[bool, float]:
    result = run_agent(workdir=path, seed=attempt)
    return result.passed, result.quality_score

outcome = BestOfN(scored_task, n=5)(ws)
```

### Reflexion — retry with critique feedback

```python
from branching import Reflexion

def task(path: Path, attempt: int, feedback: str | None) -> bool:
    if feedback:
        (path / "critique.txt").write_text(feedback)
    return run_and_test(path)

def critique(path: Path) -> str:
    return analyze_failure(path / "test_output.log")

outcome = Reflexion(task, max_retries=3, critique=critique)(ws)
```

### TreeOfThoughts — hierarchical strategy exploration

```python
from branching import TreeOfThoughts

def strategy_a(path: Path) -> tuple[bool, float]:
    apply_approach_a(path)
    return run_tests(path), evaluate_quality(path)

outcome = TreeOfThoughts(
    [strategy_a, strategy_b],
    expand=lambda path, depth: generate_refinements(path),
    max_depth=2,
)(ws)
```

### BeamSearch — multi-level with top-K survival

```python
from branching import BeamSearch

outcome = BeamSearch(
    [strat_a, strat_b, strat_c, strat_d],
    expand=lambda path, depth: generate_refinements(path),
    beam_width=2,
    max_depth=3,
)(ws)
```

### Tournament — pairwise elimination

```python
from branching import Tournament

def generate_patch(path: Path, index: int) -> bool:
    return run_agent(workdir=path, seed=index)

def judge(path_a: Path, path_b: Path) -> int:
    # 0 = a wins, 1 = b wins
    return llm_compare(path_a / "diff.patch", path_b / "diff.patch")

outcome = Tournament(generate_patch, n=8, judge=judge)(ws)
```

### Process isolation and resource limits

```python
from branching import BranchContext, ResourceLimits

# Run untrusted code in a sandboxed child process
with ws.branch("sandboxed", on_success=None, on_error=None) as b:
    with BranchContext(run_untrusted, workspace=b.path) as ctx:
        ctx.wait(timeout=30)
        b.commit()

# Resource limits (automatically enables process isolation)
limits = ResourceLimits(memory=512 * 1024 * 1024, cpu=0.5)
outcome = BestOfN(scored_task, n=5, resource_limits=limits)(ws)

# All patterns accept resource_limits and group_limits
outcome = Speculate(candidates, resource_limits=limits, timeout=60)(ws)
```

### Result types

```python
# SpeculationOutcome — returned by all patterns
outcome.committed   # bool — did a winner get committed?
outcome.winner      # SpeculationResult | None
outcome.all_results # list[SpeculationResult]

# SpeculationResult — per-candidate result
result.branch_index  # int — which candidate
result.success       # bool
result.score         # float
result.return_value  # Any — raw return from the callable
result.exception     # Exception | None
result.branch_path   # Path | None
```

## When to use which pattern

| Situation | CLI | Python |
|-----------|-----|--------|
| Try one thing safely, rollback on failure | `branching run` | `ws.branch()` |
| Multiple fix strategies, any success is fine | `branching speculate` | `Speculate` |
| Same task N times, pick the best result | `branching best-of-n` | `BestOfN` |
| Iterative fix with error feedback | `branching reflexion` | `Reflexion` |
| Hierarchical exploration (pick strategy, then refine) | — | `TreeOfThoughts` |
| Multi-level with multiple survivors per level | — | `BeamSearch` |
| Pairwise comparison, no absolute scoring | — | `Tournament` |
