# SPDX-License-Identifier: Apache-2.0
"""Shared helper: run a callable in a forked child with optional cgroup limits."""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import Any, Callable

from ..exceptions import ProcessBranchError
from .context import BranchContext
from .limits import ResourceLimits


def run_in_process(
    fn: Callable,
    args: tuple,
    workspace: Path,
    *,
    limits: ResourceLimits | None = None,
    timeout: float | None = None,
    parent_cgroup: Path | None = None,
    scope_callback: Callable[[Path], None] | None = None,
) -> Any:
    """Run *fn(*args)* in a forked child process, optionally with cgroup limits.

    The child writes its return value (or exception info) to a result file
    inside *workspace*.  The parent reads the result after the child exits.

    Args:
        fn: Callable to execute.
        args: Positional arguments for *fn*.
        workspace: Branch workspace path (used for the result file and as the
            BranchContext workspace).
        limits: Optional resource limits applied to the child's cgroup.
        timeout: Maximum seconds to wait for the child.
        parent_cgroup: Optional parent cgroup for hierarchical nesting.
        scope_callback: Optional callback invoked with the cgroup scope path
            after the child's scope is created.  Allows callers to track live
            cgroup paths for external kill/throttle.

    Returns:
        Whatever *fn* returned.

    Raises:
        ProcessBranchError: If the child was killed (e.g. OOM) or exited
            abnormally without writing a result.
        Exception: Re-raised from the child if *fn* raised.
    """
    result_path = workspace / ".branching_result"

    def _target(ws_path: Path) -> None:
        try:
            value = fn(*args)
            result_path.write_text(json.dumps({"ok": True, "value": repr(value)}))
            # Store the actual value via a second file so we can return
            # JSON-safe primitives directly and fall back to repr for the rest.
            _write_result(result_path, value)
        except BaseException as exc:
            try:
                result_path.write_text(
                    json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                )
            except Exception:
                pass
            raise

    with BranchContext(
        _target, workspace=workspace, limits=limits,
        parent_cgroup=parent_cgroup,
    ) as ctx:
        if scope_callback is not None and ctx.cgroup_scope is not None:
            scope_callback(ctx.cgroup_scope)
        try:
            ctx.wait(timeout=timeout)
        except ProcessBranchError:
            pass  # handled below via result file

    # Read result
    if not result_path.exists():
        raise ProcessBranchError(
            "Child process did not produce a result (possibly OOM-killed)"
        )

    try:
        data = json.loads(result_path.read_text())
    finally:
        result_path.unlink(missing_ok=True)

    if data.get("ok"):
        return data.get("value")

    raise ProcessBranchError(data.get("error", "unknown child error"))


def _write_result(path: Path, value: Any) -> None:
    """Write a JSON result file.  Handles non-serializable values gracefully."""
    try:
        payload = json.dumps({"ok": True, "value": value})
    except (TypeError, ValueError):
        payload = json.dumps({"ok": True, "value": repr(value)})
    path.write_text(payload)
