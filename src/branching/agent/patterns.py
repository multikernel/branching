# SPDX-License-Identifier: Apache-2.0
"""High-level speculation patterns for AI agents."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Sequence

from ..core.workspace import Workspace
from .result import SpeculationResult, SpeculationOutcome


class BestOfN:
    """Run N copies of a task in parallel, commit the highest-scoring one.

    All candidates run concurrently. Each holds its branch open after
    finishing. The main thread picks the winner based on score, then
    signals each thread to commit (winner) or abort (losers).

    The task callable receives (path, attempt_index) and returns
    (success: bool, score: float).

    Example:
        outcome = BestOfN(scored_task, n=5)(ws)
        # Commits the highest-scoring successful attempt
    """

    def __init__(
        self,
        task: Callable[[Path, int], tuple[bool, float]],
        n: int = 3,
        *,
        timeout: float | None = None,
    ):
        self._task = task
        self._n = n
        self._timeout = timeout

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        n = self._n
        results: list[Optional[SpeculationResult]] = [None] * n
        task_done = [threading.Event() for _ in range(n)]
        decision_ready = [threading.Event() for _ in range(n)]
        decisions = ["abort"] * n  # default: abort; main overwrites winner

        def _run_candidate(index: int) -> None:
            result = SpeculationResult(branch_index=index, success=False)
            try:
                with workspace.branch(
                    f"best_of_n_{index}", on_success=None, on_error=None
                ) as b:
                    result.branch_path = b.path
                    try:
                        success, score = self._task(b.path, index)
                        result.success = bool(success)
                        result.score = score
                        result.return_value = (success, score)
                    except Exception as e:
                        result.exception = e

                    # Store result and signal main thread
                    results[index] = result
                    task_done[index].set()

                    # Wait for main thread to decide commit vs abort
                    decision_ready[index].wait()

                    if decisions[index] == "commit":
                        b.commit()
                    else:
                        b.abort()

            except Exception as e:
                # Branch creation itself failed
                result.exception = e
                results[index] = result
                task_done[index].set()

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_run_candidate, i) for i in range(n)]

            # Wait for all tasks to finish (branches stay open)
            deadline = (
                time.monotonic() + self._timeout
                if self._timeout is not None
                else None
            )
            for ev in task_done:
                remaining = (
                    max(0, deadline - time.monotonic())
                    if deadline is not None
                    else None
                )
                ev.wait(timeout=remaining)

            # Pick the highest-scoring success
            best_idx: Optional[int] = None
            best_score = float("-inf")
            for i, r in enumerate(results):
                if r is not None and r.success and r.score > best_score:
                    best_score = r.score
                    best_idx = i

            if best_idx is not None:
                decisions[best_idx] = "commit"

            # Release all threads to commit/abort
            for ev in decision_ready:
                ev.set()

            # Wait for threads to finish
            for f in futures:
                f.result()

        committed = best_idx is not None
        winner = results[best_idx] if best_idx is not None else None
        all_results = [
            r if r is not None else SpeculationResult(branch_index=i, success=False)
            for i, r in enumerate(results)
        ]

        return SpeculationOutcome(
            winner=winner,
            all_results=all_results,
            committed=committed,
        )


class Reflexion:
    """Sequential retry with critique feedback loop.

    Each attempt gets a fresh branch. On failure, the critique function
    analyzes the result and provides feedback for the next attempt.

    Example:
        outcome = Reflexion(task, max_retries=3, critique=critique)(ws)
    """

    def __init__(
        self,
        task: Callable[[Path, int, Optional[str]], bool],
        max_retries: int = 3,
        *,
        critique: Optional[Callable[[Path], str]] = None,
    ):
        """
        Args:
            task: Callable(path, attempt, feedback) -> success.
                feedback is None on first attempt, critique output thereafter.
            max_retries: Maximum number of attempts.
            critique: Optional callable(path) -> feedback_string.
        """
        self._task = task
        self._max_retries = max_retries
        self._critique = critique

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        results: list[SpeculationResult] = []
        feedback: Optional[str] = None
        winner: Optional[SpeculationResult] = None

        for attempt in range(self._max_retries):
            branch_name = f"reflexion_{attempt}"
            result = SpeculationResult(branch_index=attempt, success=False)

            try:
                with workspace.branch(
                    branch_name, on_success=None, on_error="abort"
                ) as b:
                    result.branch_path = b.path
                    success = self._task(b.path, attempt, feedback)
                    result.success = bool(success)
                    result.return_value = success

                    if result.success:
                        b.commit()
                        winner = result
                        results.append(result)
                        break
                    else:
                        # Get critique before aborting
                        if self._critique is not None:
                            try:
                                feedback = self._critique(b.path)
                            except Exception:
                                feedback = None
                        b.abort()

            except Exception as e:
                result.exception = e

            results.append(result)

        committed = winner is not None
        return SpeculationOutcome(
            winner=winner,
            all_results=results,
            committed=committed,
        )


class TreeOfThoughts:
    """Parallel strategy exploration with optional multi-level depth.

    All strategies run concurrently at each level. The highest-scoring
    success wins and is committed. For multi-level exploration, pass
    ``expand`` to generate sub-strategies from the winning state at
    each depth.

    Strategies return ``bool`` or ``(bool, float)`` — if a bare bool,
    the score defaults to 1.0 for success, 0.0 for failure.

    Single level (no expand):
        Strategies run in parallel, best commits, rest abort.

    Multi-level (with expand):
        Exploration is wrapped in a root branch. At each depth the
        winner commits into the root, then ``expand(path, depth)``
        generates the next level's strategies on the accumulated state.
        If any level has no winner the root branch aborts cleanly.

    Example:
        outcome = TreeOfThoughts([strat_a, strat_b, strat_c])(ws)

        # Multi-level:
        outcome = TreeOfThoughts(
            [broad_a, broad_b],
            expand=lambda path, depth: [refine_x, refine_y],
            max_depth=3,
        )(ws)
    """

    def __init__(
        self,
        strategies: Sequence[Callable[[Path], bool | tuple[bool, float]]],
        *,
        evaluate: Callable[[Path], float] | None = None,
        expand: Callable[
            [Path, int],
            Sequence[Callable[[Path], bool | tuple[bool, float]]],
        ] | None = None,
        max_depth: int = 1,
        timeout: float | None = None,
    ):
        """
        Args:
            strategies: Level-0 callables. Each takes a Path and returns
                bool or (success, score).
            evaluate: Optional external scorer called on each successful
                branch path. Overrides any score returned by the strategy.
            expand: Callable(path, depth) -> list of strategies for the
                next level. Called after each level's winner is committed.
                Only used when max_depth > 1.
            max_depth: Maximum exploration depth (1 = single level).
            timeout: Per-level timeout in seconds.
        """
        self._strategies = list(strategies)
        self._evaluate = evaluate
        self._expand = expand
        self._max_depth = max_depth
        self._timeout = timeout

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        if self._expand is None or self._max_depth <= 1:
            return self._single_level(workspace, self._strategies, depth=0)
        return self._multi_level(workspace)

    # ------------------------------------------------------------------
    # Single level: parallel exploration, main picks best
    # ------------------------------------------------------------------

    def _single_level(self, parent, strategies, depth) -> SpeculationOutcome:
        """Run strategies in parallel on parent. Winner commits, rest abort."""
        n = len(strategies)
        if n == 0:
            return SpeculationOutcome()

        results: list[Optional[SpeculationResult]] = [None] * n
        task_done = [threading.Event() for _ in range(n)]
        decision_ready = [threading.Event() for _ in range(n)]
        decisions = ["abort"] * n

        def _run(index: int) -> None:
            result = SpeculationResult(branch_index=index, success=False)
            try:
                with parent.branch(
                    f"tot_{depth}_{index}", on_success=None, on_error=None
                ) as b:
                    result.branch_path = b.path
                    try:
                        ret = strategies[index](b.path)
                        if isinstance(ret, tuple):
                            success, score = ret
                        else:
                            success = bool(ret)
                            score = 1.0 if success else 0.0
                        if self._evaluate and success:
                            score = self._evaluate(b.path)
                        result.success = bool(success)
                        result.score = score
                        result.return_value = ret
                    except Exception as e:
                        result.exception = e

                    results[index] = result
                    task_done[index].set()
                    decision_ready[index].wait()

                    if decisions[index] == "commit":
                        b.commit()
                    else:
                        b.abort()
            except Exception as e:
                result.exception = e
                results[index] = result
                task_done[index].set()

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_run, i) for i in range(n)]

            deadline = (
                time.monotonic() + self._timeout
                if self._timeout is not None
                else None
            )
            for ev in task_done:
                remaining = (
                    max(0, deadline - time.monotonic())
                    if deadline is not None
                    else None
                )
                ev.wait(timeout=remaining)

            best_idx: Optional[int] = None
            best_score = float("-inf")
            for i, r in enumerate(results):
                if r is not None and r.success and r.score > best_score:
                    best_score = r.score
                    best_idx = i

            if best_idx is not None:
                decisions[best_idx] = "commit"

            for ev in decision_ready:
                ev.set()

            for f in futures:
                f.result()

        committed = best_idx is not None
        winner = results[best_idx] if best_idx is not None else None
        all_results = [
            r if r is not None else SpeculationResult(branch_index=i, success=False)
            for i, r in enumerate(results)
        ]
        return SpeculationOutcome(
            winner=winner, all_results=all_results, committed=committed
        )

    # ------------------------------------------------------------------
    # Multi-level: root branch wraps sequential depths
    # ------------------------------------------------------------------

    def _multi_level(self, workspace: Workspace) -> SpeculationOutcome:
        """Multi-depth exploration wrapped in a root branch.

        At each depth the winner is committed into the root branch.
        expand(root.path, depth+1) generates the next level's strategies
        which see the accumulated winning state.
        """
        all_results: list[SpeculationResult] = []
        final_winner: Optional[SpeculationResult] = None

        try:
            with workspace.branch(
                "tot_root", on_success=None, on_error=None
            ) as root:
                current_strategies = list(self._strategies)

                for depth in range(self._max_depth):
                    if not current_strategies:
                        break

                    outcome = self._single_level(
                        root, current_strategies, depth
                    )
                    all_results.extend(outcome.all_results)

                    if not outcome.committed:
                        root.abort()
                        return SpeculationOutcome(
                            all_results=all_results, committed=False
                        )

                    final_winner = outcome.winner

                    # Generate next level
                    if depth < self._max_depth - 1 and self._expand:
                        current_strategies = list(
                            self._expand(root.path, depth + 1)
                        )
                    else:
                        break

                # All levels produced a winner — commit root to main
                root.commit()

        except Exception as e:
            if final_winner is not None:
                final_winner.exception = e
            return SpeculationOutcome(
                all_results=all_results, committed=False
            )

        return SpeculationOutcome(
            winner=final_winner,
            all_results=all_results,
            committed=True,
        )
