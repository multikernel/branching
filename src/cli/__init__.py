# SPDX-License-Identifier: Apache-2.0
"""CLI for BranchContext — run commands in COW branches from the terminal."""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Optional

from branching.fs._mount import parse_mounts


def _find_branchfs_mounts():
    """Find all branchfs mounts (handles both 'fuse.branchfs' and bare 'fuse' types)."""
    results = []
    for m in parse_mounts():
        if m.fstype == "fuse.branchfs":
            results.append(m)
        elif m.fstype == "fuse" and m.source == "branchfs":
            results.append(m)
    return results


def _find_workspace_from_cwd() -> Optional[Path]:
    """Find a branchfs mount that contains the current working directory.

    Checks all branchfs mounts, returns the longest matching mountpoint
    (most specific ancestor of cwd).
    """
    cwd = Path.cwd().resolve()
    mounts = _find_branchfs_mounts()
    best: Optional[Path] = None
    for m in mounts:
        mp = m.mountpoint.resolve()
        try:
            cwd.relative_to(mp)
        except ValueError:
            continue
        if best is None or len(mp.parts) > len(best.parts):
            best = mp
    return best


def _resolve_workspace(args: argparse.Namespace) -> Path:
    """Resolve workspace path from --workspace arg or cwd auto-detection."""
    if hasattr(args, "workspace") and args.workspace:
        p = Path(args.workspace).resolve()
        if not p.is_dir():
            _print_error(f"workspace path does not exist: {p}", args)
            sys.exit(1)
        return p
    detected = _find_workspace_from_cwd()
    if detected is None:
        _print_error(
            "no branchfs mount found for current directory; use -w/--workspace",
            args,
        )
        sys.exit(1)
    return detected


def _generate_branch_name(prefix: str = "run") -> str:
    """Generate a branch name like 'run-a1b2c3d4'."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _print_result(data: dict, args: argparse.Namespace) -> None:
    """Print result as JSON (if --json) or human-readable text."""
    if getattr(args, "json", False):
        print(json.dumps(data))
    else:
        for key, value in data.items():
            print(f"{key}: {value}")


def _print_error(message: str, args: argparse.Namespace) -> None:
    """Print error as JSON (if --json) or plain text to stderr."""
    if getattr(args, "json", False):
        print(json.dumps({"error": message}), file=sys.stderr)
    else:
        print(f"error: {message}", file=sys.stderr)


def _add_resource_limit_args(parser: argparse.ArgumentParser) -> None:
    """Add per-branch resource limit flags to a subparser."""
    parser.add_argument(
        "--memory-limit",
        default=None,
        metavar="SIZE",
        help="Per-branch memory limit (e.g. 512M, 1G). Implies process isolation.",
    )
    parser.add_argument(
        "--cpu-limit",
        type=float,
        default=None,
        metavar="FRAC",
        help="Per-branch CPU limit as fraction of 1 CPU (e.g. 0.5 = 50%%). Implies process isolation.",
    )
    parser.add_argument(
        "--memory-high",
        default=None,
        metavar="SIZE",
        help="Per-branch soft memory throttle (memory.high). Reclaims aggressively but no OOM kill.",
    )
    parser.add_argument(
        "--oom-group",
        action="store_true",
        default=False,
        help="Enable atomic OOM termination (memory.oom.group). Kills entire branch on OOM.",
    )


def _add_group_limit_args(parser: argparse.ArgumentParser) -> None:
    """Add group-level resource limit flags to a subparser."""
    parser.add_argument(
        "--group-memory-limit",
        default=None,
        metavar="SIZE",
        help="Total memory budget for all branches (group cgroup memory.max).",
    )
    parser.add_argument(
        "--group-cpu-limit",
        type=float,
        default=None,
        metavar="FRAC",
        help="Total CPU budget for all branches (group cgroup cpu.max).",
    )


def _parse_resource_limits(args: argparse.Namespace):
    """Parse per-branch resource limit flags into a ResourceLimits or None."""
    mem_str = getattr(args, "memory_limit", None)
    cpu_val = getattr(args, "cpu_limit", None)
    mem_high_str = getattr(args, "memory_high", None)
    oom_group = getattr(args, "oom_group", False)
    if mem_str is None and cpu_val is None and mem_high_str is None and not oom_group:
        return None
    from branching.process.limits import ResourceLimits, parse_memory_size
    memory = parse_memory_size(mem_str) if mem_str is not None else None
    memory_high = parse_memory_size(mem_high_str) if mem_high_str is not None else None
    return ResourceLimits(memory=memory, cpu=cpu_val, memory_high=memory_high, oom_group=oom_group)


def _parse_group_limits(args: argparse.Namespace):
    """Parse group-level resource limit flags into a ResourceLimits or None."""
    mem_str = getattr(args, "group_memory_limit", None)
    cpu_val = getattr(args, "group_cpu_limit", None)
    if mem_str is None and cpu_val is None:
        return None
    from branching.process.limits import ResourceLimits, parse_memory_size
    memory = parse_memory_size(mem_str) if mem_str is not None else None
    return ResourceLimits(memory=memory, cpu=cpu_val)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="branching",
        description="CLI for BranchContext — COW branching for AI agent workflows.",
    )
    sub = parser.add_subparsers(dest="command")

    # --- run ---
    p_run = sub.add_parser(
        "run",
        help="Run a command in a new branch.",
        description=(
            "Run CMD in a new branch. Commits on exit 0, aborts on non-zero."
        ),
    )
    p_run.add_argument("-w", "--workspace", help="Workspace path (auto-detected from cwd)")
    p_run.add_argument("-b", "--branch", help="Branch name (auto-generated if omitted)")
    p_run.add_argument(
        "--on-error",
        choices=["abort", "none"],
        default="abort",
        help="Action on non-zero exit (default: abort)",
    )
    p_run.add_argument(
        "--ask",
        action="store_true",
        help="Prompt to commit/abort instead of auto-deciding",
    )
    p_run.add_argument("--json", action="store_true", help="JSON output")
    _add_resource_limit_args(p_run)
    p_run.add_argument("cmd", nargs=argparse.REMAINDER, metavar="CMD",
                        help="Command to run (after --)")

    # --- speculate ---
    p_spec = sub.add_parser(
        "speculate",
        help="Race N commands in parallel branches.",
        description="Race N commands in parallel; first success wins.",
    )
    p_spec.add_argument("-w", "--workspace", help="Workspace path (auto-detected from cwd)")
    p_spec.add_argument(
        "-c", "--candidate",
        action="append",
        required=True,
        dest="candidates",
        metavar="CMD",
        help="Command string to run as candidate (use multiple -c flags)",
    )
    p_spec.add_argument("--timeout", type=float, default=None, help="Timeout in seconds")
    p_spec.add_argument("--json", action="store_true", help="JSON output")
    _add_resource_limit_args(p_spec)
    _add_group_limit_args(p_spec)

    # --- best-of-n ---
    p_bon = sub.add_parser(
        "best-of-n",
        help="Run CMD N times in parallel, commit the highest-scoring success.",
        description="Run CMD N times in parallel branches, commit the highest-scoring success.",
    )
    p_bon.add_argument("-w", "--workspace", help="Workspace path (auto-detected from cwd)")
    p_bon.add_argument("-n", type=int, default=3, help="Number of parallel attempts (default: 3)")
    p_bon.add_argument("--timeout", type=float, default=None, help="Timeout in seconds")
    p_bon.add_argument("--json", action="store_true", help="JSON output")
    _add_resource_limit_args(p_bon)
    _add_group_limit_args(p_bon)
    p_bon.add_argument("cmd", nargs=argparse.REMAINDER, metavar="CMD",
                        help="Command to run (after --)")

    # --- reflexion ---
    p_refl = sub.add_parser(
        "reflexion",
        help="Sequential retry with optional critique feedback loop.",
        description="Sequential retry with optional critique feedback loop.",
    )
    p_refl.add_argument("-w", "--workspace", help="Workspace path (auto-detected from cwd)")
    p_refl.add_argument("--retries", type=int, default=3, help="Max retry attempts (default: 3)")
    p_refl.add_argument("--critique", type=str, default=None,
                         help="Shell command to generate feedback after failure")
    p_refl.add_argument("--json", action="store_true", help="JSON output")
    _add_resource_limit_args(p_refl)
    _add_group_limit_args(p_refl)
    p_refl.add_argument("cmd", nargs=argparse.REMAINDER, metavar="CMD",
                         help="Command to run (after --)")

    # --- status ---
    p_status = sub.add_parser(
        "status",
        help="Show workspace info and branch list.",
    )
    p_status.add_argument("-w", "--workspace", help="Workspace path (auto-detected from cwd)")
    p_status.add_argument("--json", action="store_true", help="JSON output")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Strip leading '--' from remainder args for commands that take CMD...
    if args.command in ("run", "best-of-n", "reflexion"):
        cmd = args.cmd
        if cmd and cmd[0] == "--":
            cmd = cmd[1:]
        args.cmd = cmd

    try:
        if args.command == "run":
            from .run import cmd_run
            sys.exit(cmd_run(args))
        elif args.command == "speculate":
            from .speculate import cmd_speculate
            sys.exit(cmd_speculate(args))
        elif args.command == "best-of-n":
            from .best_of_n import cmd_best_of_n
            sys.exit(cmd_best_of_n(args))
        elif args.command == "reflexion":
            from .reflexion import cmd_reflexion
            sys.exit(cmd_reflexion(args))
        elif args.command == "status":
            from .status import cmd_status
            sys.exit(cmd_status(args))
        else:
            parser.print_help()
            sys.exit(1)
    except KeyboardInterrupt:
        print(file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        _print_error(str(e), args)
        sys.exit(1)
