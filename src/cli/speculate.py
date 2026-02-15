# SPDX-License-Identifier: Apache-2.0
"""Implementation of 'branching speculate' command."""

import subprocess
from pathlib import Path

from branching import Speculate, Workspace

from . import _print_error, _print_result, _resolve_workspace


def _make_candidate(cmd_str: str):
    """Wrap a shell command string into a Speculate candidate callable."""

    def candidate(workdir: Path) -> bool:
        result = subprocess.run(cmd_str, shell=True, cwd=workdir)
        return result.returncode == 0

    return candidate


def cmd_speculate(args) -> int:
    ws_path = _resolve_workspace(args)
    ws = Workspace(ws_path)

    candidates = [_make_candidate(c) for c in args.candidates]
    spec = Speculate(candidates, first_wins=True, timeout=args.timeout)
    outcome = spec(ws)

    results_summary = []
    for r in outcome.all_results:
        entry = {"index": r.branch_index, "success": r.success}
        if r.exception is not None:
            entry["error"] = str(r.exception)
        results_summary.append(entry)

    data = {
        "command": "speculate",
        "committed": outcome.committed,
        "winner": outcome.winner.branch_index if outcome.winner else None,
        "candidates": len(args.candidates),
        "results": results_summary,
    }

    if getattr(args, "json", False):
        import json
        print(json.dumps(data))
    else:
        if outcome.committed:
            print(f"winner: candidate {outcome.winner.branch_index}")
        else:
            print("no candidate succeeded")
        for r in outcome.all_results:
            status = "ok" if r.success else "fail"
            line = f"  [{r.branch_index}] {status}: {args.candidates[r.branch_index]}"
            if r.exception:
                line += f" ({r.exception})"
            print(line)

    return 0 if outcome.committed else 1
