# SPDX-License-Identifier: Apache-2.0
"""Shared helper: run a callable in a forked child."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from ..exceptions import ProcessBranchError
from .context import BranchContext


def run_in_process(
    fn: Callable,
    args: tuple,
    workspace: Path,
    *,
    mount_root: Path | None = None,
    timeout: float | None = None,
    pid_callback: Callable[[int], None] | None = None,
) -> Any:
    """Run *fn(*args)* in a forked child process.

    Results are passed back via an inherited pipe fd — no filesystem
    dependency, so this works even when the workspace is a FUSE mount.

    Args:
        fn: Callable to execute.
        args: Positional arguments for *fn*.
        workspace: Branch workspace path (passed to BranchContext).
        mount_root: Filesystem mount root.
        timeout: Maximum seconds to wait for the child.
        pid_callback: Optional callback invoked with the child PID
            after the branch is created.  Allows callers to track live
            branch PIDs for external kill.

    Returns:
        Whatever *fn* returned.

    Raises:
        ProcessBranchError: If the child was killed (e.g. OOM) or exited
            abnormally without writing a result.
        Exception: Re-raised from the child if *fn* raised.
    """
    read_fd, write_fd = os.pipe()

    def _target(ws_path: Path) -> None:
        os.close(read_fd)
        try:
            value = fn(*args)
            _write_result_fd(write_fd, {"ok": True, "value": value})
        except BaseException as exc:
            _write_result_fd(
                write_fd, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            raise
        finally:
            os.close(write_fd)

    try:
        with BranchContext(
            _target, workspace=workspace, mount_root=mount_root,
        ) as ctx:
            os.close(write_fd)
            write_fd = -1  # prevent double-close in except branch
            if pid_callback is not None:
                pid_callback(ctx.pid)
            try:
                ctx.wait(timeout=timeout)
            except ProcessBranchError:
                pass  # handled below via pipe
    except Exception:
        if write_fd >= 0:
            os.close(write_fd)
        raise

    # Read result from pipe
    data = _read_result_fd(read_fd)
    os.close(read_fd)

    if data is None:
        raise ProcessBranchError(
            "Child process did not produce a result (possibly OOM-killed)"
        )

    if data.get("ok"):
        return data.get("value")

    raise ProcessBranchError(data.get("error", "unknown child error"))


def _write_result_fd(fd: int, data: dict) -> None:
    """Write a JSON result dict to a pipe fd, handling non-serializable values."""
    try:
        payload = json.dumps(data).encode()
    except (TypeError, ValueError):
        # Fall back to repr for non-JSON-serializable values
        fallback = dict(data)
        if "value" in fallback:
            fallback["value"] = repr(fallback["value"])
        payload = json.dumps(fallback).encode()
    os.write(fd, payload)


def _read_result_fd(fd: int) -> dict | None:
    """Read a JSON result dict from a pipe fd.  Returns None on empty read."""
    chunks = []
    while True:
        chunk = os.read(fd, 65536)
        if not chunk:
            break
        chunks.append(chunk)
    if not chunks:
        return None
    return json.loads(b"".join(chunks))
