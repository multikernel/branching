# SPDX-License-Identifier: Apache-2.0
"""High-level speculation patterns for AI agents."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional, Sequence, TYPE_CHECKING

from ..core.workspace import Workspace
from ..process.runner import run_in_process
from ..exceptions import ConflictError
from .result import SpeculationResult, SpeculationOutcome

if TYPE_CHECKING:
    from ..process.limits import ResourceLimits


class BestOfN:
    """Run N candidates in parallel, commit the highest-scoring one.

    All candidates run concurrently. Each holds its branch open after
    finishing. The main thread picks the winner based on score, then
    signals each thread to commit (winner) or abort (losers).

    Each candidate callable receives (path,) and returns
    (success: bool, score: float).

    Example:
        candidates = [lambda p: (run_tests(p), score(p)) for _ in range(5)]
        outcome = BestOfN(candidates)(ws)
        # Commits the highest-scoring successful candidate
    """

    def __init__(
        self,
        candidates: Sequence[Callable[[Path], tuple[bool, float]]],
        *,
        timeout: float | None = None,
        resource_limits: ResourceLimits | None = None,
        group_limits: ResourceLimits | None = None,
    ):
        self._candidates = list(candidates)
        self._timeout = timeout
        self._resource_limits = resource_limits
        self._group_limits = group_limits

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        import os as _os

        root_cgroup: Optional[Path] = None
        if self._resource_limits is not None and self._group_limits is not None:
            try:
                from ..process._cgroup import create_group
                root_cgroup = create_group(
                    f"bestofn-{_os.getpid()}",
                    limits=self._group_limits,
                )
            except OSError:
                root_cgroup = None

        try:
            return self._run(workspace, root_cgroup)
        finally:
            if root_cgroup is not None:
                from ..process._cgroup import kill_scope
                kill_scope(root_cgroup)

    def _run(self, workspace: Workspace, root_cgroup: Optional[Path]) -> SpeculationOutcome:
        n = len(self._candidates)
        results: list[Optional[SpeculationResult]] = [None] * n
        task_done = [threading.Event() for _ in range(n)]
        decision_ready = [threading.Event() for _ in range(n)]
        decisions = ["abort"] * n  # default: abort; main overwrites winner

        branch_scopes: dict[int, Path] = {}

        def _kill_scopes(exclude: int = -1) -> None:
            from ..process._cgroup import kill_scope
            for idx, scope in list(branch_scopes.items()):
                if idx != exclude:
                    kill_scope(scope)

        def _run_candidate(index: int) -> None:
            result = SpeculationResult(branch_index=index, success=False)
            try:
                with workspace.branch(
                    f"best_of_n_{index}", on_success=None, on_error=None
                ) as b:
                    result.branch_path = b.path
                    try:
                        def _on_scope(sp: Path, _i: int = index) -> None:
                            branch_scopes[_i] = sp

                        ret = run_in_process(
                            self._candidates[index], (b.path,),
                            workspace=b.path,
                            limits=self._resource_limits,
                            parent_cgroup=root_cgroup,
                            scope_callback=_on_scope if self._resource_limits else None,
                        )
                        success, score = ret
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

            # Kill still-running tasks (only useful when timeout left
            # some workers stuck in ctx.wait — a no-op otherwise).
            if any(r is None for r in results):
                _kill_scopes(best_idx if best_idx is not None else -1)

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
        resource_limits: ResourceLimits | None = None,
        group_limits: ResourceLimits | None = None,
    ):
        """
        Args:
            task: Callable(path, attempt, feedback) -> success.
                feedback is None on first attempt, critique output thereafter.
            max_retries: Maximum number of attempts.
            critique: Optional callable(path) -> feedback_string.
            resource_limits: Optional per-branch resource limits.
            group_limits: Optional resource limits for the root cgroup.
        """
        self._task = task
        self._max_retries = max_retries
        self._critique = critique
        self._resource_limits = resource_limits
        self._group_limits = group_limits

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        import os as _os

        root_cgroup: Optional[Path] = None
        if self._resource_limits is not None and self._group_limits is not None:
            try:
                from ..process._cgroup import create_group
                root_cgroup = create_group(
                    f"reflexion-{_os.getpid()}",
                    limits=self._group_limits,
                )
            except OSError:
                root_cgroup = None

        try:
            return self._run(workspace, root_cgroup)
        finally:
            if root_cgroup is not None:
                from ..process._cgroup import kill_scope
                kill_scope(root_cgroup)

    def _run(self, workspace: Workspace, root_cgroup: Optional[Path]) -> SpeculationOutcome:
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
                    success = run_in_process(
                        self._task, (b.path, attempt, feedback),
                        workspace=b.path,
                        limits=self._resource_limits,
                        parent_cgroup=root_cgroup,
                    )
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
        resource_limits: ResourceLimits | None = None,
        group_limits: ResourceLimits | None = None,
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
            resource_limits: Optional per-branch resource limits.
            group_limits: Optional resource limits for the root cgroup.
        """
        self._strategies = list(strategies)
        self._evaluate = evaluate
        self._expand = expand
        self._max_depth = max_depth
        self._timeout = timeout
        self._resource_limits = resource_limits
        self._group_limits = group_limits

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        import os as _os

        root_cgroup: Optional[Path] = None
        if self._resource_limits is not None and self._group_limits is not None:
            try:
                from ..process._cgroup import create_group
                root_cgroup = create_group(
                    f"tot-{_os.getpid()}",
                    limits=self._group_limits,
                )
            except OSError:
                root_cgroup = None

        try:
            if self._expand is None or self._max_depth <= 1:
                return self._single_level(
                    workspace, self._strategies, depth=0,
                    parent_cgroup=root_cgroup,
                )
            return self._multi_level(workspace, root_cgroup)
        finally:
            if root_cgroup is not None:
                from ..process._cgroup import kill_scope
                kill_scope(root_cgroup)

    # ------------------------------------------------------------------
    # Single level: parallel exploration, main picks best
    # ------------------------------------------------------------------

    def _single_level(
        self, parent, strategies, depth, *, parent_cgroup=None,
    ) -> SpeculationOutcome:
        """Run strategies in parallel on parent. Winner commits, rest abort."""
        n = len(strategies)
        if n == 0:
            return SpeculationOutcome()

        results: list[Optional[SpeculationResult]] = [None] * n
        task_done = [threading.Event() for _ in range(n)]
        decision_ready = [threading.Event() for _ in range(n)]
        decisions = ["abort"] * n

        branch_scopes: dict[int, Path] = {}

        def _kill_scopes(exclude: int = -1) -> None:
            from ..process._cgroup import kill_scope
            for idx, scope in list(branch_scopes.items()):
                if idx != exclude:
                    kill_scope(scope)

        def _run(index: int) -> None:
            result = SpeculationResult(branch_index=index, success=False)
            try:
                with parent.branch(
                    f"tot_{depth}_{index}", on_success=None, on_error=None
                ) as b:
                    result.branch_path = b.path
                    try:
                        def _on_scope(sp: Path, _i: int = index) -> None:
                            branch_scopes[_i] = sp

                        ret = run_in_process(
                            strategies[index], (b.path,),
                            workspace=b.path,
                            limits=self._resource_limits,
                            parent_cgroup=parent_cgroup,
                            scope_callback=_on_scope if self._resource_limits else None,
                        )
                        if isinstance(ret, (tuple, list)):
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

            if any(r is None for r in results):
                _kill_scopes(best_idx if best_idx is not None else -1)

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

    def _multi_level(
        self, workspace: Workspace, root_cgroup: Path | None = None,
    ) -> SpeculationOutcome:
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
                        root, current_strategies, depth,
                        parent_cgroup=root_cgroup,
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


class BeamSearch:
    """Multi-level beam search: keep top-K branches alive at each depth.

    Interpolates between BestOfN (all parallel, one level) and
    TreeOfThoughts multi-level (one winner per level).  Multiple beams
    survive each level, each accumulating its own state independently.
    Pruning happens globally across all beams' candidates.

    Strategies return ``bool`` or ``(bool, float)`` — if a bare bool,
    the score defaults to 1.0 for success, 0.0 for failure.

    Example:
        outcome = BeamSearch(
            [strat_a, strat_b, strat_c, strat_d],
            expand=lambda path, depth: [refine_x, refine_y],
            beam_width=2,
            max_depth=3,
        )(workspace)
    """

    def __init__(
        self,
        strategies: Sequence[Callable[[Path], bool | tuple[bool, float]]],
        *,
        expand: Callable[
            [Path, int],
            Sequence[Callable[[Path], bool | tuple[bool, float]]],
        ],
        evaluate: Callable[[Path], float] | None = None,
        beam_width: int = 3,
        max_depth: int = 2,
        timeout: float | None = None,
        resource_limits: ResourceLimits | None = None,
        group_limits: ResourceLimits | None = None,
    ):
        self._strategies = list(strategies)
        self._expand = expand
        self._evaluate = evaluate
        self._beam_width = beam_width
        self._max_depth = max_depth
        self._timeout = timeout
        self._resource_limits = resource_limits
        self._group_limits = group_limits

    def _score(self, ret, path):
        """Parse strategy return and apply optional evaluator."""
        if isinstance(ret, (tuple, list)):
            success, score = ret
        else:
            success = bool(ret)
            score = 1.0 if success else 0.0
        if self._evaluate and success:
            score = self._evaluate(path)
        return bool(success), score

    def _top_k(self, results, k):
        """Return indices of top-k successful results by score."""
        scored = [
            (i, r) for i, r in enumerate(results)
            if r is not None and r.success
        ]
        scored.sort(key=lambda x: x[1].score, reverse=True)
        return [i for i, _ in scored[:k]]

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        import os as _os

        n = len(self._strategies)
        if n == 0:
            return SpeculationOutcome()

        root_cgroup: Optional[Path] = None
        if self._resource_limits is not None and self._group_limits is not None:
            try:
                from ..process._cgroup import create_group
                root_cgroup = create_group(
                    f"beamsearch-{_os.getpid()}",
                    limits=self._group_limits,
                )
            except OSError:
                root_cgroup = None

        try:
            return self._run(workspace, n, root_cgroup)
        finally:
            if root_cgroup is not None:
                from ..process._cgroup import kill_scope
                kill_scope(root_cgroup)

    def _run(
        self, workspace: Workspace, n: int, root_cgroup: Optional[Path],
    ) -> SpeculationOutcome:
        K = self._beam_width
        all_results: list[SpeculationResult] = []

        # Pre-allocate per-beam intermediate cgroups (thread-safe population)
        beam_cgroups: list[Optional[Path]] = [None] * n
        # Track leaf cgroup scopes for kill-on-prune.
        beam_task_scopes: dict[int, Path] = {}

        def _kill_beam_scopes(keep: set[int]) -> None:
            from ..process._cgroup import kill_scope
            for idx, scope in list(beam_task_scopes.items()):
                if idx not in keep:
                    kill_scope(scope)

        # -- Level 0: create beam branches from workspace ----------------
        beam_branches: list[Optional[object]] = [None] * n
        level0_results: list[Optional[SpeculationResult]] = [None] * n
        task_done = [threading.Event() for _ in range(n)]
        final_decision = [threading.Event() for _ in range(n)]
        final_actions = ["abort"] * n

        def _beam_worker(index: int) -> None:
            result = SpeculationResult(branch_index=index, success=False)
            # Create per-beam intermediate cgroup
            if root_cgroup is not None:
                try:
                    from ..process._cgroup import create_group as _cg
                    beam_cgroups[index] = _cg(
                        f"beam_{index}", parent=root_cgroup,
                    )
                except OSError:
                    pass
            try:
                with workspace.branch(
                    f"beam_{index}", on_success=None, on_error=None
                ) as b:
                    result.branch_path = b.path
                    beam_branches[index] = b
                    try:
                        def _on_scope(sp: Path, _i: int = index) -> None:
                            beam_task_scopes[_i] = sp

                        ret = run_in_process(
                            self._strategies[index], (b.path,),
                            workspace=b.path,
                            limits=self._resource_limits,
                            parent_cgroup=beam_cgroups[index],
                            scope_callback=_on_scope if self._resource_limits else None,
                        )
                        result.success, result.score = self._score(
                            ret, b.path
                        )
                        result.return_value = ret
                    except Exception as e:
                        result.exception = e

                    level0_results[index] = result
                    task_done[index].set()

                    # Hold branch open until final decision
                    final_decision[index].wait()
                    if final_actions[index] == "commit":
                        b.commit()
                    else:
                        b.abort()
            except Exception as e:
                result.exception = e
                level0_results[index] = result
                task_done[index].set()

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(_beam_worker, i) for i in range(n)]

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

            # Select top-K beams
            survivors = set(self._top_k(level0_results, K))

            beam_scores: dict[int, float] = {}
            for i in survivors:
                beam_scores[i] = level0_results[i].score

            all_results.extend(
                r if r is not None
                else SpeculationResult(branch_index=i, success=False)
                for i, r in enumerate(level0_results)
            )

            # Kill timed-out beams (no-op when all tasks finished).
            if any(r is None for r in level0_results):
                _kill_beam_scopes(survivors)
            for i in range(n):
                if i not in survivors:
                    final_actions[i] = "abort"
                    final_decision[i].set()

            # -- Deeper levels -------------------------------------------
            for depth in range(1, self._max_depth):
                if not survivors:
                    break

                sub_tasks: list[tuple[int, int, Callable]] = []
                for beam_idx in sorted(survivors):
                    sub_strats = list(
                        self._expand(beam_branches[beam_idx].path, depth)
                    )
                    for si, strat in enumerate(sub_strats):
                        sub_tasks.append((beam_idx, si, strat))

                if not sub_tasks:
                    break

                m = len(sub_tasks)
                sub_results: list[Optional[SpeculationResult]] = [None] * m
                sub_done = [threading.Event() for _ in range(m)]
                sub_decision_ready = [threading.Event() for _ in range(m)]
                sub_decisions = ["abort"] * m
                sub_scopes: dict[int, Path] = {}
                _depth = depth  # capture value for closure

                def _sub_worker(idx: int, _d: int = _depth) -> None:
                    beam_idx, strat_idx, strategy = sub_tasks[idx]
                    result = SpeculationResult(
                        branch_index=idx, success=False
                    )
                    try:
                        parent = beam_branches[beam_idx]
                        with parent.branch(
                            f"beam_{beam_idx}_d{_d}_{strat_idx}",
                            on_success=None,
                            on_error=None,
                        ) as sb:
                            result.branch_path = sb.path
                            try:
                                def _on_sub_scope(
                                    sp: Path, _j: int = idx,
                                ) -> None:
                                    sub_scopes[_j] = sp

                                ret = run_in_process(
                                    strategy, (sb.path,),
                                    workspace=sb.path,
                                    limits=self._resource_limits,
                                    parent_cgroup=beam_cgroups[beam_idx],
                                    scope_callback=_on_sub_scope if self._resource_limits else None,
                                )
                                result.success, result.score = self._score(
                                    ret, sb.path
                                )
                                result.return_value = ret
                            except Exception as e:
                                result.exception = e

                            sub_results[idx] = result
                            sub_done[idx].set()
                            sub_decision_ready[idx].wait()

                            if sub_decisions[idx] == "commit":
                                sb.commit()
                            else:
                                sb.abort()
                    except Exception as e:
                        result.exception = e
                        sub_results[idx] = result
                        sub_done[idx].set()

                with ThreadPoolExecutor(max_workers=m) as sub_pool:
                    sub_futures = [
                        sub_pool.submit(_sub_worker, i) for i in range(m)
                    ]

                    for ev in sub_done:
                        remaining = (
                            max(0, deadline - time.monotonic())
                            if deadline is not None
                            else None
                        )
                        ev.wait(timeout=remaining)

                    top_k_indices = set(self._top_k(sub_results, K))

                    all_results.extend(
                        r if r is not None
                        else SpeculationResult(branch_index=i, success=False)
                        for i, r in enumerate(sub_results)
                    )

                    # Kill timed-out sub-branches (no-op when all finished).
                    if any(r is None for r in sub_results):
                        from ..process._cgroup import kill_scope as _ks
                        for si, scope in list(sub_scopes.items()):
                            if si not in top_k_indices:
                                _ks(scope)

                    for i in top_k_indices:
                        sub_decisions[i] = "commit"
                    for ev in sub_decision_ready:
                        ev.set()
                    for f in sub_futures:
                        f.result()

                # Update surviving beams
                beams_alive: dict[int, float] = {}
                for i in top_k_indices:
                    beam_idx = sub_tasks[i][0]
                    score = sub_results[i].score
                    if (
                        beam_idx not in beams_alive
                        or score > beams_alive[beam_idx]
                    ):
                        beams_alive[beam_idx] = score

                for beam_idx in survivors - set(beams_alive):
                    final_actions[beam_idx] = "abort"
                    final_decision[beam_idx].set()

                survivors = set(beams_alive)
                beam_scores.update(beams_alive)

            # -- Final: pick best surviving beam -------------------------
            winner = None
            if survivors:
                best = max(survivors, key=lambda i: beam_scores[i])
                final_actions[best] = "commit"
                winner = SpeculationResult(
                    branch_index=best,
                    success=True,
                    score=beam_scores[best],
                    branch_path=(
                        level0_results[best].branch_path
                        if level0_results[best] is not None
                        else None
                    ),
                )

            # Release all remaining beam threads
            for i in range(n):
                final_decision[i].set()
            for f in futures:
                f.result()

        return SpeculationOutcome(
            winner=winner,
            all_results=all_results,
            committed=winner is not None,
        )


class Tournament:
    """Pairwise elimination bracket: run N candidates, compare
    pairwise via a judge function, commit the final winner.

    The convergent dual of TreeOfThoughts: starts wide, narrows to one.

    Example:
        outcome = Tournament(candidates, judge=judge)(ws)
        # Commits the bracket winner
    """

    def __init__(
        self,
        candidates: Sequence[Callable[[Path], bool]],
        *,
        judge: Callable[[Path, Path], int],
        timeout: float | None = None,
        resource_limits: ResourceLimits | None = None,
        group_limits: ResourceLimits | None = None,
    ):
        """
        Args:
            candidates: Callables that take a Path (branch working dir)
                  and return True on success. Each produces output in
                  the branch directory for the judge to compare.
            judge: Callable(path_a, path_b) → 0 (a wins) or 1 (b wins).
                   Compares two candidates' branches during elimination.
            timeout: Overall timeout in seconds.
            resource_limits: Optional per-branch resource limits.
            group_limits: Optional resource limits for the root cgroup.
        """
        self._candidates = list(candidates)
        self._judge = judge
        self._timeout = timeout
        self._resource_limits = resource_limits
        self._group_limits = group_limits

    @staticmethod
    def _run_bracket(
        survivors: list[int],
        branch_paths: list[Path],
        judge: Callable[[Path, Path], int],
    ) -> int:
        """Single-elimination bracket. Returns the winning candidate index."""
        while len(survivors) > 1:
            next_round: list[int] = []
            i = 0
            while i < len(survivors) - 1:
                a, b = survivors[i], survivors[i + 1]
                pick = judge(branch_paths[a], branch_paths[b])
                next_round.append(b if pick else a)
                i += 2
            # Odd candidate gets a bye
            if len(survivors) % 2 == 1:
                next_round.append(survivors[-1])
            survivors = next_round
        return survivors[0]

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        import os as _os

        root_cgroup: Optional[Path] = None
        if self._resource_limits is not None and self._group_limits is not None:
            try:
                from ..process._cgroup import create_group
                root_cgroup = create_group(
                    f"tournament-{_os.getpid()}",
                    limits=self._group_limits,
                )
            except OSError:
                root_cgroup = None

        try:
            return self._run(workspace, root_cgroup)
        finally:
            if root_cgroup is not None:
                from ..process._cgroup import kill_scope
                kill_scope(root_cgroup)

    def _run(self, workspace: Workspace, root_cgroup: Optional[Path]) -> SpeculationOutcome:
        n = len(self._candidates)
        results: list[Optional[SpeculationResult]] = [None] * n
        branch_paths: list[Optional[Path]] = [None] * n
        task_done = [threading.Event() for _ in range(n)]
        decision_ready = [threading.Event() for _ in range(n)]
        decisions = ["abort"] * n

        branch_scopes: dict[int, Path] = {}

        def _kill_scopes(exclude: int = -1) -> None:
            from ..process._cgroup import kill_scope
            for idx, scope in list(branch_scopes.items()):
                if idx != exclude:
                    kill_scope(scope)

        def _run_candidate(index: int) -> None:
            result = SpeculationResult(branch_index=index, success=False)
            try:
                with workspace.branch(
                    f"tournament_{index}", on_success=None, on_error=None
                ) as b:
                    result.branch_path = b.path
                    branch_paths[index] = b.path
                    try:
                        def _on_scope(sp: Path, _i: int = index) -> None:
                            branch_scopes[_i] = sp

                        success = run_in_process(
                            self._candidates[index], (b.path,),
                            workspace=b.path,
                            limits=self._resource_limits,
                            parent_cgroup=root_cgroup,
                            scope_callback=_on_scope if self._resource_limits else None,
                        )
                        result.success = bool(success)
                        result.return_value = success
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
            futures = [pool.submit(_run_candidate, i) for i in range(n)]

            # Wait for all tasks to finish
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

            # Filter to successful survivors
            survivors = [
                i for i, r in enumerate(results)
                if r is not None and r.success
            ]

            winner_idx: Optional[int] = None
            if len(survivors) == 1:
                winner_idx = survivors[0]
            elif len(survivors) > 1:
                winner_idx = self._run_bracket(
                    survivors, branch_paths, self._judge
                )

            if winner_idx is not None:
                decisions[winner_idx] = "commit"

            if any(r is None for r in results):
                _kill_scopes(winner_idx if winner_idx is not None else -1)

            # Release all threads
            for ev in decision_ready:
                ev.set()

            for f in futures:
                f.result()

        committed = winner_idx is not None
        winner = results[winner_idx] if winner_idx is not None else None
        all_results = [
            r if r is not None else SpeculationResult(branch_index=i, success=False)
            for i, r in enumerate(results)
        ]

        return SpeculationOutcome(
            winner=winner,
            all_results=all_results,
            committed=committed,
        )


class Cascaded:
    """Cascaded speculation: start narrow, widen on failure with feedback.

    Runs successive waves of parallel candidates with increasing fan-out.
    Wave 0 starts with few candidates (typically one).  If all fail, the
    next wave runs more candidates, each informed by error context from
    prior failures.  Within each wave, first-winner-commit semantics
    apply: the first successful candidate commits and siblings are aborted.

    The task callable receives ``(path, feedback)`` and returns
    ``(success, error_context)``.  *feedback* is always a list of error
    strings from prior failed attempts (empty on wave 0).

    Example::

        def solve(path: Path, feedback: list[str]) -> tuple[bool, str]:
            result = run_agent(path, prior_errors=feedback)
            if result.tests_pass:
                return True, ""
            return False, result.error_output

        outcome = Cascaded(solve, fan_out=(1, 2, 4))(ws)
    """

    def __init__(
        self,
        task: Callable[[Path, list[str]], tuple[bool, str]],
        *,
        fan_out: Sequence[int] = (1, 2, 4),
        timeout: float | None = None,
        wave_timeout: float | None = None,
        resource_limits: ResourceLimits | None = None,
        group_limits: ResourceLimits | None = None,
    ):
        """
        Args:
            task: Callable(path, feedback) -> (success, error_context).
                *feedback* is a list of error strings from all prior
                failed attempts (empty list on the first wave).
                Returns a tuple of (success_bool, error_string).  The
                error string is collected as feedback for subsequent
                waves when the attempt fails; ignored on success.
            fan_out: Number of parallel candidates per wave.
                Default ``(1, 2, 4)`` runs 1 candidate in wave 0,
                2 in wave 1, 4 in wave 2.  The number of waves equals
                ``len(fan_out)``.
            timeout: Overall timeout in seconds across all waves.
            wave_timeout: Per-wave timeout in seconds.
            resource_limits: Optional per-branch resource limits.
            group_limits: Optional resource limits for the root cgroup.
        """
        self._task = task
        self._fan_out = list(fan_out)
        self._timeout = timeout
        self._wave_timeout = wave_timeout
        self._resource_limits = resource_limits
        self._group_limits = group_limits

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        import os as _os

        root_cgroup: Optional[Path] = None
        if self._resource_limits is not None and self._group_limits is not None:
            try:
                from ..process._cgroup import create_group
                root_cgroup = create_group(
                    f"cascaded-{_os.getpid()}",
                    limits=self._group_limits,
                )
            except OSError:
                root_cgroup = None

        try:
            return self._run(workspace, root_cgroup)
        finally:
            if root_cgroup is not None:
                from ..process._cgroup import kill_scope
                kill_scope(root_cgroup)

    def _run(
        self, workspace: Workspace, root_cgroup: Optional[Path],
    ) -> SpeculationOutcome:
        all_results: list[SpeculationResult] = []
        feedback: list[str] = []
        base_index = 0

        deadline = (
            time.monotonic() + self._timeout
            if self._timeout is not None
            else None
        )

        for wave, width in enumerate(self._fan_out):
            if deadline is not None and deadline - time.monotonic() <= 0:
                break

            # Snapshot so concurrent/later mutations don't leak in.
            feedback_snapshot = list(feedback)
            wave_errors: list[str] = []

            outcome = self._run_wave(
                workspace, wave, width, feedback_snapshot, base_index,
                root_cgroup, deadline, wave_errors,
            )
            all_results.extend(outcome.all_results)

            if outcome.committed:
                return SpeculationOutcome(
                    winner=outcome.winner,
                    all_results=all_results,
                    committed=True,
                )

            feedback.extend(wave_errors)
            base_index += width

        return SpeculationOutcome(all_results=all_results, committed=False)

    def _run_wave(
        self,
        workspace: Workspace,
        wave: int,
        width: int,
        feedback: list[str],
        base_index: int,
        root_cgroup: Optional[Path],
        deadline: Optional[float],
        wave_errors: list[str],
    ) -> SpeculationOutcome:
        """Run a single wave of parallel candidates with first-wins."""
        results: list[Optional[SpeculationResult]] = [None] * width
        winner: Optional[SpeculationResult] = None
        committed = False
        cancel_event = threading.Event()

        branch_scopes: dict[int, Path] = {}

        def _kill_scopes(exclude: int = -1) -> None:
            from ..process._cgroup import kill_scope
            for idx, scope in list(branch_scopes.items()):
                if idx != exclude:
                    kill_scope(scope)

        # Compute effective wave timeout.
        wt = self._wave_timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            wt = min(wt, remaining) if wt is not None else remaining

        def _run_candidate(index: int) -> SpeculationResult:
            global_index = base_index + index
            branch_name = f"cascaded_w{wave}_{index}"
            result = SpeculationResult(
                branch_index=global_index, success=False,
            )

            if cancel_event.is_set():
                return result

            try:
                with workspace.branch(
                    branch_name, on_success=None, on_error="abort"
                ) as b:
                    result.branch_path = b.path

                    if cancel_event.is_set():
                        b.abort()
                        return result

                    def _on_scope(scope_path: Path, _i: int = index) -> None:
                        branch_scopes[_i] = scope_path

                    success, error_ctx = self._run_task(
                        b.path, feedback, root_cgroup,
                        scope_callback=_on_scope if self._resource_limits else None,
                        timeout=wt,
                    )

                    result.success = bool(success)
                    result.return_value = (success, error_ctx)

                    if result.success and not cancel_event.is_set():
                        cancel_event.set()
                        _kill_scopes(index)
                        try:
                            b.commit()
                        except ConflictError:
                            result.success = False
                            b.abort()
                        return result
                    else:
                        if (
                            not cancel_event.is_set()
                            and error_ctx
                        ):
                            wave_errors.append(error_ctx)
                        b.abort()
                        return result

            except Exception as e:
                result.exception = e
                return result

        with ThreadPoolExecutor(max_workers=width) as pool:
            futures: dict = {}
            for i in range(width):
                f = pool.submit(_run_candidate, i)
                futures[f] = i

            try:
                for f in as_completed(futures, timeout=wt):
                    idx = futures[f]
                    try:
                        result = f.result()
                    except Exception as e:
                        result = SpeculationResult(
                            branch_index=base_index + idx,
                            success=False,
                            exception=e,
                        )
                    results[idx] = result

                    if result.success and winner is None:
                        winner = result
                        committed = True
            except TimeoutError:
                cancel_event.set()
                _kill_scopes()

            for f in futures:
                if not f.done():
                    try:
                        f.result(timeout=5.0)
                    except Exception:
                        pass

        for i, r in enumerate(results):
            if r is None:
                results[i] = SpeculationResult(
                    branch_index=base_index + i, success=False,
                )

        return SpeculationOutcome(
            winner=winner,
            all_results=results,
            committed=committed,
        )

    def _run_task(
        self,
        path: Path,
        feedback: list[str],
        parent_cgroup: Optional[Path],
        scope_callback=None,
        timeout: Optional[float] = None,
    ) -> tuple[bool, str]:
        """Run the task in a forked child with resource limits."""
        from ..exceptions import ProcessBranchError

        try:
            ret = run_in_process(
                self._task,
                (path, feedback),
                workspace=path,
                limits=self._resource_limits,
                timeout=timeout,
                parent_cgroup=parent_cgroup,
                scope_callback=scope_callback,
            )
            if isinstance(ret, (tuple, list)):
                success, error_ctx = ret
                return bool(success), str(error_ctx) if error_ctx else ""
            return bool(ret), ""
        except ProcessBranchError:
            return False, ""
