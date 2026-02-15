# SPDX-License-Identifier: Apache-2.0
"""Speculate - parallel branch orchestration for AI agents."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Callable, Sequence, Optional, TYPE_CHECKING
import threading

from ..core.workspace import Workspace
from ..exceptions import ConflictError
from .result import SpeculationResult, SpeculationOutcome

if TYPE_CHECKING:
    from ..process.limits import ResourceLimits


class Speculate:
    """Parallel branch speculation: run N candidates, first success wins.

    Callable class — instantiate with config, call with workspace.

    Example:
        spec = Speculate([try_fix_a, try_fix_b], first_wins=True, timeout=60)
        outcome = spec(ws)
        if outcome.committed:
            print(f"Fix {outcome.winner.branch_index} succeeded!")
    """

    def __init__(
        self,
        candidates: Sequence[Callable[[Path], bool]],
        *,
        first_wins: bool = True,
        max_parallel: int | None = None,
        isolate_processes: bool = False,
        timeout: float | None = None,
        resource_limits: ResourceLimits | None = None,
        group_limits: ResourceLimits | None = None,
    ):
        """
        Args:
            candidates: Callables that take a Path (branch working dir) and
                return True on success, False on failure.
            first_wins: If True, commit the first successful candidate and
                abort siblings. If False, run all and commit the first success.
            max_parallel: Maximum parallel workers (default: len(candidates)).
            isolate_processes: Run each candidate in a forked process.
            timeout: Overall timeout in seconds for all candidates.
            resource_limits: Optional per-branch resource limits (implies
                process isolation).
            group_limits: Optional resource limits applied to the root
                cgroup that contains all branches.
        """
        self._candidates = list(candidates)
        self._first_wins = first_wins
        self._max_parallel = max_parallel or len(self._candidates)
        self._isolate_processes = isolate_processes
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
                    f"speculate-{_os.getpid()}",
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
        results: list[SpeculationResult] = [None] * len(self._candidates)  # type: ignore
        winner: Optional[SpeculationResult] = None
        committed = False
        cancel_event = threading.Event()

        # Track live cgroup scope paths so we can kill losers immediately.
        # Populated by scope_callback from run_in_process; read by
        # _kill_scopes.  dict ops are GIL-protected in CPython.
        branch_scopes: dict[int, Path] = {}

        def _kill_scopes(exclude: int = -1) -> None:
            """Kill all tracked cgroup scopes except *exclude*."""
            from ..process._cgroup import kill_scope
            for idx, scope in list(branch_scopes.items()):
                if idx != exclude:
                    kill_scope(scope)

        def _run_candidate(index: int) -> SpeculationResult:
            branch_name = f"speculate_{index}"
            result = SpeculationResult(branch_index=index, success=False)

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

                    if self._resource_limits is not None:
                        def _on_scope(scope_path: Path, _i: int = index) -> None:
                            branch_scopes[_i] = scope_path

                        success = self._run_with_limits(
                            b.path, index, root_cgroup,
                            scope_callback=_on_scope,
                        )
                    elif self._isolate_processes:
                        success = self._run_in_process(b.path, index)
                    else:
                        success = self._candidates[index](b.path)

                    result.success = bool(success)
                    result.return_value = success

                    if result.success and not cancel_event.is_set():
                        if self._first_wins:
                            cancel_event.set()
                            _kill_scopes(index)
                        try:
                            b.commit()
                        except ConflictError:
                            # Sibling already committed — we lost the race
                            result.success = False
                            b.abort()
                        return result
                    else:
                        b.abort()
                        return result

            except Exception as e:
                result.exception = e
                return result

        with ThreadPoolExecutor(max_workers=self._max_parallel) as pool:
            futures: dict[Future, int] = {}
            for i in range(len(self._candidates)):
                f = pool.submit(_run_candidate, i)
                futures[f] = i

            try:
                for f in as_completed(futures, timeout=self._timeout):
                    idx = futures[f]
                    try:
                        result = f.result()
                    except Exception as e:
                        result = SpeculationResult(
                            branch_index=idx, success=False, exception=e
                        )
                    results[idx] = result

                    if result.success and winner is None:
                        winner = result
                        committed = True
            except TimeoutError:
                # Signal remaining candidates to abort
                cancel_event.set()
                _kill_scopes()

            # Wait briefly for in-flight branches to finish cleanup
            for f in futures:
                if not f.done():
                    try:
                        f.result(timeout=5.0)
                    except Exception:
                        pass

        # Fill in any None results (from timeout or cancelled)
        for i, r in enumerate(results):
            if r is None:
                results[i] = SpeculationResult(branch_index=i, success=False)

        return SpeculationOutcome(
            winner=winner,
            all_results=results,
            committed=committed,
        )

    def _run_in_process(self, path: Path, index: int) -> bool:
        """Run a candidate in a forked child process."""
        from ..process.context import BranchContext
        from ..exceptions import ProcessBranchError

        def target(workspace: Path) -> None:
            if not self._candidates[index](workspace):
                raise ProcessBranchError("Candidate returned failure")

        per_candidate = (
            self._timeout / len(self._candidates)
            if self._timeout is not None
            else None
        )

        with BranchContext(target, workspace=path) as pb:
            try:
                pb.wait(timeout=per_candidate)
                return True
            except ProcessBranchError:
                return False

    def _run_with_limits(
        self, path: Path, index: int, parent_cgroup: Path | None = None,
        scope_callback=None,
    ) -> bool:
        """Run a candidate in a forked child with resource limits."""
        from ..process.runner import run_in_process
        from ..exceptions import ProcessBranchError

        per_candidate = (
            self._timeout / len(self._candidates)
            if self._timeout is not None
            else None
        )

        try:
            result = run_in_process(
                self._candidates[index],
                (path,),
                workspace=path,
                limits=self._resource_limits,
                timeout=per_candidate,
                parent_cgroup=parent_cgroup,
                scope_callback=scope_callback,
            )
            return bool(result)
        except ProcessBranchError:
            return False
