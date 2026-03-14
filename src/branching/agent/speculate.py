# SPDX-License-Identifier: Apache-2.0
"""Speculate - parallel branch orchestration for AI agents."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from pathlib import Path
from typing import Callable, Sequence, Optional
import threading

from ..core.workspace import Workspace
from ..process.runner import _default_runner, Runner
from ..exceptions import ConflictError
from .result import SpeculationResult, SpeculationOutcome


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
        timeout: float | None = None,
        runner: Runner | None = None,
    ):
        """
        Args:
            candidates: Callables that take a Path (branch working dir) and
                return True on success, False on failure.
            first_wins: If True, commit the first successful candidate and
                abort siblings. If False, run all and commit the first success.
            max_parallel: Maximum parallel workers (default: len(candidates)).
            timeout: Overall timeout in seconds for all candidates.
            runner: Execution strategy for candidates. Default forks via
                BranchContext. Pass a sandlock runner for confinement.
        """
        self._candidates = list(candidates)
        self._first_wins = first_wins
        self._max_parallel = max_parallel or len(self._candidates)
        self._timeout = timeout
        self._runner = runner or _default_runner

    def __call__(self, workspace: Workspace) -> SpeculationOutcome:
        return self._run(workspace)

    def _run(self, workspace: Workspace) -> SpeculationOutcome:
        results: list[SpeculationResult] = [None] * len(self._candidates)  # type: ignore
        winner: Optional[SpeculationResult] = None
        committed = False
        cancel_event = threading.Event()

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

                    success = self._run_in_branch(b.path, index)

                    result.success = bool(success)
                    result.return_value = success

                    if result.success and not cancel_event.is_set():
                        if self._first_wins:
                            cancel_event.set()
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

    def _run_in_branch(self, path: Path, index: int) -> bool:
        """Run a candidate via the configured runner."""
        from ..exceptions import ProcessBranchError

        per_candidate = (
            self._timeout / len(self._candidates)
            if self._timeout is not None
            else None
        )

        try:
            result = self._runner(
                self._candidates[index],
                (path,),
                timeout=per_candidate,
            )
            return bool(result)
        except ProcessBranchError:
            return False
