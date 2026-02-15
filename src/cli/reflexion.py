# SPDX-License-Identifier: Apache-2.0
"""Implementation of 'branching reflexion' command."""

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from branching import Reflexion, Workspace

from . import _parse_group_limits, _parse_resource_limits, _print_error, _resolve_workspace


def _make_task(cmd: list[str]):
    """Wrap a command into a Reflexion task callable.

    Returns a callable(path, attempt, feedback) -> bool.
    The child receives BRANCHING_ATTEMPT and BRANCHING_FEEDBACK env vars.
    """

    def task(workdir: Path, attempt: int, feedback: Optional[str]) -> bool:
        env = {
            **os.environ,
            "BRANCHING_ATTEMPT": str(attempt),
            "BRANCHING_FEEDBACK": feedback or "",
        }
        result = subprocess.run(cmd, cwd=workdir, env=env)
        return result.returncode == 0

    return task


def _make_critique(critique_cmd: str):
    """Wrap a critique shell command into a callable(path) -> str."""

    def critique(workdir: Path) -> str:
        result = subprocess.run(
            critique_cmd,
            shell=True,
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return critique


def cmd_reflexion(args) -> int:
    if not args.cmd:
        _print_error("no command specified; use -- CMD...", args)
        return 1

    ws_path = _resolve_workspace(args)
    ws = Workspace(ws_path)

    task = _make_task(args.cmd)
    critique_fn = _make_critique(args.critique) if args.critique else None
    limits = _parse_resource_limits(args)
    group_limits = _parse_group_limits(args)
    refl = Reflexion(
        task, max_retries=args.retries, critique=critique_fn,
        resource_limits=limits, group_limits=group_limits,
    )
    outcome = refl(ws)

    results_summary = []
    for r in outcome.all_results:
        entry = {
            "index": r.branch_index,
            "success": r.success,
        }
        if r.exception is not None:
            entry["error"] = str(r.exception)
        results_summary.append(entry)

    data = {
        "command": "reflexion",
        "committed": outcome.committed,
        "attempts": len(outcome.all_results),
        "winner": outcome.winner.branch_index if outcome.winner else None,
        "results": results_summary,
    }

    if getattr(args, "json", False):
        print(json.dumps(data))
    else:
        for r in outcome.all_results:
            status = "ok" if r.success else "fail"
            suffix = " (committed)" if r.success and outcome.committed else ""
            print(f"attempt {r.branch_index}: {status}{suffix}")
        if not outcome.committed:
            print("all attempts failed")

    return 0 if outcome.committed else 1
