# SPDX-License-Identifier: Apache-2.0
"""Cgroup v2 scope management for reliable descendant termination."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .limits import ResourceLimits

CGROUP_BASE = Path("/sys/fs/cgroup")


def _own_cgroup() -> Path:
    """Return the cgroup v2 directory for the current process.

    Reads /proc/self/cgroup to find our position in the hierarchy.
    On cgroup v2 (unified) the line is "0::/path/to/cgroup".

    Returns:
        Absolute path under /sys/fs/cgroup.

    Raises:
        OSError: If /proc/self/cgroup cannot be read or parsed.
    """
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        # cgroup v2 unified: "0::<path>"
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0":
            return CGROUP_BASE / parts[2].lstrip("/")
    raise OSError("No cgroup v2 entry found in /proc/self/cgroup")


def _enable_subtree_controllers(cgroup_dir: Path) -> None:
    """Enable memory and cpu controllers for children of *cgroup_dir*.

    Reads ``cgroup.controllers`` to discover available controllers, then
    writes ``+memory +cpu`` (intersection with available) to
    ``cgroup.subtree_control``.  Best-effort — silently ignores errors.
    """
    try:
        available = (cgroup_dir / "cgroup.controllers").read_text().split()
    except OSError:
        return
    wanted = [c for c in ("memory", "cpu") if c in available]
    if not wanted:
        return
    payload = " ".join(f"+{c}" for c in wanted)
    try:
        (cgroup_dir / "cgroup.subtree_control").write_text(payload)
    except OSError:
        pass


def create_scope(name: str, *, parent: Path | None = None) -> Path:
    """Create a child cgroup under *parent* (or our own cgroup) for process grouping.

    When *parent* is given the scope is created as a child of that
    directory and ``_enable_subtree_controllers`` is called on the parent
    first so that memory/cpu controllers are available in the child.

    Args:
        name: Scope name suffix (e.g. PID).
        parent: Optional parent cgroup directory.  Defaults to
            ``_own_cgroup()`` when ``None``.

    Returns:
        Path to the cgroup directory.

    Raises:
        OSError: If cgroup creation fails.
    """
    if parent is not None:
        _enable_subtree_controllers(parent)
        parent_dir = parent
    else:
        parent_dir = _own_cgroup()
    scope_dir = parent_dir / f"branching-{name}.scope"
    scope_dir.mkdir(exist_ok=True)
    return scope_dir


def create_group(
    name: str,
    *,
    parent: Path | None = None,
    limits: "ResourceLimits | None" = None,
) -> Path:
    """Create an intermediate cgroup for nesting (never holds PIDs directly).

    Calls ``_enable_subtree_controllers`` on the new directory so that
    child scopes can use memory/cpu controllers.  Applies optional
    group-level limits (total budget for all children).

    Args:
        name: Group name suffix.
        parent: Optional parent cgroup directory.  Defaults to
            ``_own_cgroup()`` when ``None``.
        limits: Optional resource limits applied to the group itself.

    Returns:
        Path to the new group cgroup directory.
    """
    if parent is not None:
        _enable_subtree_controllers(parent)
        parent_dir = parent
    else:
        parent_dir = _own_cgroup()
    group_dir = parent_dir / f"branching-{name}.scope"
    group_dir.mkdir(exist_ok=True)
    _enable_subtree_controllers(group_dir)
    if limits is not None:
        set_limits(group_dir, limits)
    return group_dir


def add_pid(scope: Path, pid: int) -> None:
    """Add a PID to a cgroup scope.

    Args:
        scope: Path to the cgroup directory.
        pid: Process ID to add.

    Raises:
        OSError: If writing to cgroup.procs fails.
    """
    (scope / "cgroup.procs").write_text(str(pid))


def set_limits(scope: Path, limits: "ResourceLimits") -> None:
    """Apply resource limits to a cgroup scope.

    Best-effort — skips ``None`` fields and ignores write errors so that
    missing controllers don't crash the caller.
    """
    if limits.memory is not None:
        try:
            (scope / "memory.max").write_text(str(limits.memory))
        except OSError:
            pass
    if limits.memory_high is not None:
        try:
            (scope / "memory.high").write_text(str(limits.memory_high))
        except OSError:
            pass
    if limits.oom_group:
        try:
            (scope / "memory.oom.group").write_text("1")
        except OSError:
            pass
    if limits.cpu is not None:
        period = 100_000  # 100 ms
        quota = int(limits.cpu * period)
        try:
            (scope / "cpu.max").write_text(f"{quota} {period}")
        except OSError:
            pass


def kill_scope(scope: Path) -> None:
    """Kill all processes in a cgroup scope and remove it recursively.

    Iterates child directories bottom-up, kills each, then kills and
    removes self.  Handles the cgroup v2 requirement that ``rmdir``
    needs no child cgroups.

    Best-effort cleanup — ignores errors.
    """
    try:
        # Recurse into children first (bottom-up)
        if scope.is_dir():
            for child in sorted(scope.iterdir()):
                if child.is_dir():
                    kill_scope(child)
        # Kill processes in this scope
        kill_file = scope / "cgroup.kill"
        if kill_file.exists():
            kill_file.write_text("1")
        scope.rmdir()
    except OSError:
        pass
