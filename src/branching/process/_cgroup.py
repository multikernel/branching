# SPDX-License-Identifier: Apache-2.0
"""Cgroup v2 scope management for reliable descendant termination."""

import os
from pathlib import Path

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


def create_scope(name: str) -> Path:
    """Create a child cgroup under our own cgroup for process grouping.

    Creates under the current process's cgroup, which is compatible
    with systemd's delegated hierarchy — we never write into a level
    that systemd manages directly.

    Args:
        name: Scope name suffix (e.g. PID).

    Returns:
        Path to the cgroup directory.

    Raises:
        OSError: If cgroup creation fails.
    """
    parent = _own_cgroup()
    scope_dir = parent / f"branching-{name}.scope"
    scope_dir.mkdir(exist_ok=True)
    return scope_dir


def add_pid(scope: Path, pid: int) -> None:
    """Add a PID to a cgroup scope.

    Args:
        scope: Path to the cgroup directory.
        pid: Process ID to add.

    Raises:
        OSError: If writing to cgroup.procs fails.
    """
    (scope / "cgroup.procs").write_text(str(pid))


def kill_scope(scope: Path) -> None:
    """Kill all processes in a cgroup scope and remove it.

    Best-effort cleanup — ignores errors.
    """
    try:
        kill_file = scope / "cgroup.kill"
        if kill_file.exists():
            kill_file.write_text("1")
        scope.rmdir()
    except OSError:
        pass
