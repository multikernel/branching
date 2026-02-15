# SPDX-License-Identifier: Apache-2.0
"""Implementation of 'branching run' command."""

import subprocess
import sys

from branching import Workspace

from . import (
    _generate_branch_name,
    _parse_resource_limits,
    _print_error,
    _print_result,
    _resolve_workspace,
)


def _interactive_commit_prompt() -> bool:
    """Prompt user to commit or abort. Returns True to commit."""
    while True:
        try:
            answer = input("Commit changes? [y/n] ").strip().lower()
        except EOFError:
            return False
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def cmd_run(args) -> int:
    if not args.cmd:
        _print_error("no command specified; use -- CMD...", args)
        return 1

    ws_path = _resolve_workspace(args)
    branch_name = args.branch or _generate_branch_name("run")
    on_error_action = None if args.on_error == "none" else "abort"

    ws = Workspace(ws_path)
    limits = _parse_resource_limits(args)

    with ws.branch(branch_name, on_success=None, on_error=None) as b:
        if limits is not None:
            from branching.process.context import BranchContext

            def _target(ws_path):
                import subprocess as sp
                r = sp.run(args.cmd, cwd=ws_path)
                if r.returncode != 0:
                    raise SystemExit(r.returncode)

            with BranchContext(_target, workspace=b.path, limits=limits) as ctx:
                try:
                    ctx.wait()
                    exit_code = 0
                except Exception:
                    exit_code = 1
        else:
            result = subprocess.run(args.cmd, cwd=b.path)
            exit_code = result.returncode

        if args.ask:
            action = "committed" if _interactive_commit_prompt() else "aborted"
            if action == "committed":
                b.commit()
            else:
                b.abort()
        elif exit_code == 0:
            b.commit()
            action = "committed"
        else:
            if on_error_action == "abort":
                b.abort()
                action = "aborted"
            else:
                action = "kept"

    _print_result(
        {
            "command": "run",
            "branch": branch_name,
            "exit_code": exit_code,
            "action": action,
        },
        args,
    )
    return exit_code
