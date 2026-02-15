# BranchContext

Let AI agents try things without consequences.

When an agent explores multiple strategies - applying different patches,
trying different prompts, or testing alternative approaches - it normally
has to snapshot the workspace, run the attempt, then clean up the mess
before trying the next one. BranchContext eliminates that overhead.

Fork the workspace into parallel copy-on-write branches, run speculative
attempts in each, commit the winner, and abort the rest - instantly.
No snapshots, no cleanup, no leftover state.

Based on the paper [Fork, Explore, Commit: OS Primitives for Agentic Exploration](https://arxiv.org/abs/2602.08199).

## Install

```
pip install .
```

Requires Python >= 3.10. No external dependencies.

## Quick start

```python
from branching import Workspace

ws = Workspace("/mnt/workspace")

# Auto-commit on success, auto-abort on exception
with ws.branch("attempt") as b:
    subprocess.run(["agent", "--workdir", str(b.path)], check=True)
```

The agent writes to `b.path`, which is an isolated copy-on-write view.
If the command succeeds, changes are merged back into the workspace. If it
raises, everything is rolled back - the workspace is untouched.

## Agent patterns

BranchContext ships with six high-level patterns that cover the most common
agent workflows. Each is a callable class: instantiate with config, call with
a workspace.

### Parallel speculation (first wins)

Run multiple strategies in parallel. The first one that succeeds gets
committed; the rest are aborted.

Use when you have several plausible approaches and care about latency more
than optimality: bug fixes where any passing patch is good enough, tool
selection where multiple tools could work, or prompt variants where you
just need one that doesn't error out.

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

### Best-of-N with scoring

Run the same task N times (e.g. with different random seeds or temperatures)
and commit the highest-scoring success.

Use when quality matters more than speed: code generation where you want
the cleanest output across multiple temperatures, translation with a BLEU
scorer picking the best variant, or any task with a reliable quality metric
where the same prompt can produce varying results.

```python
from branching import BestOfN

def scored_task(path: Path, attempt: int) -> tuple[bool, float]:
    result = run_agent(workdir=path, seed=attempt)
    return result.passed, result.quality_score

outcome = BestOfN(scored_task, n=5)(ws)
```

### Reflexion (retry with feedback)

Run a task, and if it fails, generate a critique and feed it back into the
next attempt. The agent learns from its mistakes across retries.

Use when failures carry diagnostic signal: fixing test failures where the
error log tells you what went wrong, iterating on a solution where a
validator explains why it was rejected, or multi-step plans where each
failed attempt narrows the search space for the next one.

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

### Tree of Thoughts

Explore multiple strategies in parallel, optionally expanding the best one
into deeper sub-strategies across multiple levels.

Use when the problem has hierarchical structure: architectural decisions
where you first pick a framework then optimize within it, multi-stage
pipelines where each stage has variants worth exploring, or planning tasks
where high-level strategies each decompose into tactical choices.

```python
from branching import TreeOfThoughts

def strategy_a(path: Path) -> tuple[bool, float]:
    apply_approach_a(path)
    return run_tests(path), evaluate_quality(path)

def strategy_b(path: Path) -> tuple[bool, float]:
    apply_approach_b(path)
    return run_tests(path), evaluate_quality(path)

outcome = TreeOfThoughts(
    [strategy_a, strategy_b],
    max_depth=2,
    expand=lambda path, depth: generate_refinements(path),
)(ws)
```

### Beam Search

Keep the top-K branches alive at each depth level instead of just one
winner. Interpolates between BestOfN (all parallel, one level) and
TreeOfThoughts (one winner per level). At each level, all candidates
across all beams are scored globally and only the top-K survive.

Inspired by [EnCompass](https://arxiv.org/abs/2512.03571), which showed
that multi-level beam search outperforms both BestOfN and single-winner
tree search for hierarchical agent tasks.

Use when the problem has hierarchical structure *and* you want to hedge
across multiple promising directions: multi-step code migrations where
several rewrite strategies look viable at each stage, planning tasks where
pruning to one path too early loses good alternatives, or any setting where
TreeOfThoughts' single-winner-per-level is too aggressive.

```python
from branching import BeamSearch

def strategy_a(path: Path) -> tuple[bool, float]:
    apply_approach_a(path)
    return run_tests(path), evaluate_quality(path)

def strategy_b(path: Path) -> tuple[bool, float]:
    apply_approach_b(path)
    return run_tests(path), evaluate_quality(path)

outcome = BeamSearch(
    [strategy_a, strategy_b, strategy_c, strategy_d],
    expand=lambda path, depth: generate_refinements(path),
    beam_width=2,
    max_depth=3,
)(ws)
```

### Tournament (pairwise elimination)

Generate N candidates in parallel, then narrow to one through pairwise
elimination via a judge function. The convergent dual of Tree of Thoughts:
starts wide, narrows to one.

Use when you have a reliable pairwise comparator but no absolute scoring
function: patch selection where an LLM judge picks the better diff,
A/B-style evaluation where candidates are compared head-to-head, or
any setting where relative ranking is easier than absolute scoring.

```python
from branching import Tournament

def generate_patch(path: Path, index: int) -> bool:
    return run_agent(workdir=path, seed=index)

def judge(path_a: Path, path_b: Path) -> int:
    # 0 = a wins, 1 = b wins
    return llm_compare(path_a / "diff.patch", path_b / "diff.patch")

outcome = Tournament(generate_patch, n=8, judge=judge)(ws)
```

## Lower-level usage

The patterns above are built on two lower-level primitives you can use
directly when you need more control.

### Branching with manual control

```python
with ws.branch("attempt", on_success=None, on_error=None) as b:
    result = run_agent(workdir=b.path)
    if result.confident:
        b.commit()
    else:
        b.abort()
```

### Nested branches

Branches can nest - useful for hierarchical exploration (e.g. pick a
strategy, then explore variants within it).

```python
with ws.branch("strategy_a") as a:
    apply_strategy(a.path)

    with a.branch("variant_1") as v1:
        tweak(v1.path)
        # v1 auto-commits into a on success

    # a auto-commits into main on success
```

### Process isolation

For untrusted or crash-prone agent code, `BranchContext` runs each task in
a sandboxed child process with its own filesystem view. No root needed.

```python
from branching import BranchContext

with ws.branch("sandboxed", on_success=None, on_error=None) as fb:
    with BranchContext(run_untrusted, workspace=fb.path) as ctx:
        try:
            ctx.wait(timeout=30)
            fb.commit()
        except ProcessBranchError:
            fb.abort()
```

Run N tasks in parallel, each in its own sandbox:

```python
with BranchContext.create(
    targets=[task_a, task_b, task_c],
    workspaces=[ws_a.path, ws_b.path, ws_c.path],
) as contexts:
    for ctx in contexts:
        ctx.wait(timeout=60)
```

Any agent pattern can also opt into process isolation:

```python
outcome = Speculate(candidates, isolate_processes=True, timeout=60)(ws)
```

### Resource limits

Constrain per-branch memory and CPU via cgroup v2. Passing
`resource_limits` to any pattern automatically enables process isolation -
each branch runs in a forked child with cgroup enforcement.

```python
from branching import ResourceLimits, BestOfN

limits = ResourceLimits(memory=512 * 1024 * 1024, cpu=0.5)  # 512 MB, 50% CPU

outcome = BestOfN(scored_task, n=5, resource_limits=limits)(ws)
```

All patterns accept `resource_limits`: `Speculate`, `BestOfN`, `Reflexion`,
`TreeOfThoughts`, `BeamSearch`, and `Tournament`. Fields default to `None`
(unlimited). A `ResourceLimits()` with all `None` fields triggers process
isolation without applying any limits.

You can also pass limits directly to `BranchContext`:

```python
from branching import BranchContext, ResourceLimits

limits = ResourceLimits(memory=1024 * 1024 * 1024)  # 1 GB

with BranchContext(run_agent, workspace=branch.path, limits=limits) as ctx:
    ctx.wait(timeout=30)
```

## CLI

The `branching` command exposes the agent patterns as shell commands.
Auto-detects the workspace from your current directory, or pass `-w PATH`.
All commands support `--json` for machine-readable output.

### run

Run a command in a new branch. Commits on exit 0, aborts on non-zero.

```bash
branching run -- ./build.sh
branching run --on-error none -- python train.py
branching run --ask -- make test          # prompt before commit/abort
branching run --memory-limit 512M -- ./agent.sh   # cap memory at 512 MB
branching run --memory-limit 1G --cpu-limit 0.5 -- python train.py
```

### speculate

Race N commands in parallel branches. First success wins.

```bash
branching speculate -c "./fix_a.sh" -c "./fix_b.sh" -c "./fix_c.sh"
branching speculate --timeout 60 -c "python solve_v1.py" -c "python solve_v2.py"
branching speculate --memory-limit 256M -c "./a.sh" -c "./b.sh"
```

### best-of-n

Run CMD N times in parallel, commit the highest-scoring success.

The child process can write a score to fd 3 (`echo 0.95 >&3`).
If nothing is written, score defaults to 1.0 for success / 0.0 for failure.
Each child receives `BRANCHING_ATTEMPT` (0-indexed) in its environment.

```bash
branching best-of-n -n 5 -- ./solve.py
branching best-of-n -n 3 --timeout 120 --json -- python attempt.py
branching best-of-n -n 3 -- bash -c 'python run.py && echo "$SCORE" >&3'
branching best-of-n -n 5 --memory-limit 1G --cpu-limit 0.5 -- python attempt.py
```

### reflexion

Sequential retry with optional critique feedback loop.

The child receives `BRANCHING_ATTEMPT` (0-indexed) and `BRANCHING_FEEDBACK`
(empty on first attempt, critique output on retries) in its environment.

```bash
branching reflexion --retries 5 -- ./fix.sh
branching reflexion --retries 3 --critique "./review.sh" -- ./solve.py
branching reflexion --retries 3 --critique "python critique.py" --json -- python agent.py
branching reflexion --retries 3 --memory-limit 512M -- ./fix.sh
```

### status

Show workspace info and active branches.

```bash
branching status
branching status --json
```

## How it works

BranchContext uses copy-on-write filesystems to create instant, zero-cost
branches of your workspace. Two backends are supported, auto-detected at
runtime:

| Backend | How branches work |
|---|---|
| **[BranchFS](https://github.com/multikernel/branchfs)** (FUSE) | Single mount; branches are virtual paths within it. First-winner-commit semantics. |
| **[DaxFS](https://github.com/multikernel/daxfs)** (kernel) | Separate mount per branch. Fastest option for DAX-capable storage. |

You just create a `Workspace` pointed at a mounted path - the backend is
detected automatically.

Process isolation (`BranchContext`) uses unprivileged Linux user namespaces
to give each child its own filesystem view. No root required - works on any
Linux distribution with `unprivileged_userns_clone=1` (the default).

Resource limits (`ResourceLimits`) use cgroup v2 to enforce per-branch
memory and CPU constraints. Each branch gets its own cgroup scope with
limits applied before the child process starts. Requires cgroup v2 with
the memory and cpu controllers enabled (the default on modern systemd
distributions).

