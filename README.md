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

BranchContext ships with five high-level patterns that cover the most common
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

## API reference

The library has three layers. Imports are lazy - only the layer you use
gets loaded.

```python
from branching import Workspace      # FS layer only
from branching import BranchContext  # process layer only
from branching import Speculate      # agent layer (+ FS layer)
```

### Workspace and Branch

**`Workspace(path)`** - open a workspace from an existing mount.

| Property / Method | Description |
|---|---|
| `.path` | Mount root path |
| `.fstype` | Detected backend (`"branchfs"` or `"daxfs"`) |
| `.branch(name, on_success="commit", on_error="abort")` | Create a branch (returns `Branch` context manager) |

**`Branch`** - context manager for an isolated workspace view.

| Property / Method | Description |
|---|---|
| `.name` | Branch name |
| `.path` | Working directory for this branch |
| `.branch_path` | Full path (e.g. `"/main/feature"`) |
| `.branch(name, ...)` | Create a nested child branch |
| `.commit()` | Merge changes into parent |
| `.abort()` | Roll back to parent |

`on_success` accepts `"commit"` or `None`.
`on_error` accepts `"abort"` or `None`.

### BranchContext (process isolation)

**`BranchContext(target, workspace, *, close_fds=False)`** - run a function
in a sandboxed child process.

| Property / Method | Description |
|---|---|
| `target` | `Callable[[Path], None]` - return normally for success, raise for failure |
| `workspace` | Directory to bind-mount into the child |
| `.pid` | Child PID |
| `.alive` | Whether the child is still running |
| `.wait(timeout=None)` | Block until exit; raises `ProcessBranchError` on failure, `TimeoutError` on timeout |
| `.abort(timeout=5.0)` | Abort child and all its descendants |

**`BranchContext.create(targets, workspaces, *, close_fds=False)`** - fork N
children at once; cleans up all on exit.

### Agent patterns

All patterns: instantiate with config, call with a `Workspace`, get a
`SpeculationOutcome`.

| Pattern | Constructor | What it does |
|---|---|---|
| **`Speculate`** | `(candidates, *, first_wins=True, max_parallel=None, isolate_processes=False, timeout=None)` | Run candidates in parallel; first success wins |
| **`BestOfN`** | `(task, n=3, *, timeout=None)` | Run N copies; commit highest-scoring success |
| **`Reflexion`** | `(task, max_retries=3, *, critique=None)` | Retry with critique feedback loop |
| **`TreeOfThoughts`** | `(strategies, *, evaluate=None, expand=None, max_depth=1, timeout=None)` | Parallel strategy tree with optional depth expansion |
| **`Tournament`** | `(task, n=4, *, judge, timeout=None)` | Generate N candidates; pairwise elimination picks winner |

### Result types

**`SpeculationOutcome`** - returned by all agent patterns.

| Field | Description |
|---|---|
| `.committed` | Whether a winner was committed |
| `.winner` | `SpeculationResult` or `None` |
| `.all_results` | All candidate results |

**`SpeculationResult`** - one candidate's outcome.

| Field | Description |
|---|---|
| `.branch_index` | Candidate index |
| `.success` | Whether the candidate succeeded |
| `.score` | Quality score (default 0.0) |
| `.return_value` | Raw return value |
| `.exception` | Exception if failed, else `None` |
| `.branch_path` | Working directory used |

### Exceptions

```
BranchingError                  # base for all errors
├── MountError                  # filesystem mount/unmount failed
├── BranchError                 # branch operation failed
│   ├── BranchStaleError        # a sibling branch was already committed
│   ├── BranchNotFoundError     # branch does not exist
│   ├── CommitError             # commit failed
│   │   └── ConflictError       # another branch already committed
│   └── AbortError              # abort failed
├── ProcessBranchError          # sandboxed child process failed
│   ├── ForkError               # fork() failed
│   └── NamespaceError          # sandbox setup failed
└── MemoryBranchError           # memory branching failed
```
