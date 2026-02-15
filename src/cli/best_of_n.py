# SPDX-License-Identifier: Apache-2.0
"""Implementation of 'branching best-of-n' command."""

import json
import os
import subprocess
from pathlib import Path

from branching import BestOfN, Workspace

from . import _parse_resource_limits, _print_error, _resolve_workspace


def _make_task(cmd: list[str]):
    """Wrap a command into a BestOfN task callable.

    Returns a callable(path, index) -> (success, score).
    The child process can write a score float to fd 3.
    """

    def task(workdir: Path, index: int) -> tuple[bool, float]:
        # Create a pipe for the child to report its score.
        # Python 3.4+ creates pipe fds with CLOEXEC, so they are
        # automatically closed on exec — only fd 3 (dup2 clears
        # CLOEXEC) survives into the child.
        r_fd, w_fd = os.pipe()

        env = {**os.environ, "BRANCHING_ATTEMPT": str(index)}

        def _preexec():
            os.dup2(w_fd, 3)

        proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            close_fds=False,
            preexec_fn=_preexec,
        )
        # Close write end in the parent so read will EOF when child exits
        os.close(w_fd)

        proc.wait()
        success = proc.returncode == 0

        # Read score from the pipe
        with os.fdopen(r_fd, "r") as f:
            raw = f.read().strip()

        if raw:
            try:
                score = float(raw)
            except ValueError:
                score = 1.0 if success else 0.0
        else:
            score = 1.0 if success else 0.0

        return (success, score)

    return task


def cmd_best_of_n(args) -> int:
    if not args.cmd:
        _print_error("no command specified; use -- CMD...", args)
        return 1

    ws_path = _resolve_workspace(args)
    ws = Workspace(ws_path)

    task = _make_task(args.cmd)
    limits = _parse_resource_limits(args)
    best = BestOfN(task, n=args.n, timeout=args.timeout, resource_limits=limits)
    outcome = best(ws)

    results_summary = []
    for r in outcome.all_results:
        entry = {
            "index": r.branch_index,
            "success": r.success,
            "score": r.score,
        }
        if r.exception is not None:
            entry["error"] = str(r.exception)
        results_summary.append(entry)

    data = {
        "command": "best-of-n",
        "n": args.n,
        "committed": outcome.committed,
        "winner": {
            "index": outcome.winner.branch_index,
            "score": outcome.winner.score,
        } if outcome.winner else None,
        "results": results_summary,
    }

    if getattr(args, "json", False):
        print(json.dumps(data))
    else:
        if outcome.committed:
            print(
                f"winner: attempt {outcome.winner.branch_index}"
                f" (score: {outcome.winner.score:.2f})"
            )
        else:
            print("no attempt succeeded")
        for r in outcome.all_results:
            status = "ok" if r.success else "fail"
            print(f"  [{r.branch_index}] {status:<4} ({r.score:.2f})")

    return 0 if outcome.committed else 1
