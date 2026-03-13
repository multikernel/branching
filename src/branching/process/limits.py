# SPDX-License-Identifier: Apache-2.0
"""Resource limits applied via setrlimit(2).

Replaces the former cgroup v2 controller approach with lightweight
per-process rlimits.  Limits are inherited by children on fork,
so setting them once on the parent provides group-level budgets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UNITS = {
    "K": 1024,
    "M": 1024 ** 2,
    "G": 1024 ** 3,
    "T": 1024 ** 4,
}

_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT])?\s*$", re.IGNORECASE)


def parse_memory_size(s: str) -> int:
    """Parse a human-friendly memory size string to bytes.

    Accepts plain integers (bytes) or suffixed values: ``'512M'``, ``'1G'``,
    ``'100K'``.  The suffix is case-insensitive.

    Returns:
        Size in bytes (integer).

    Raises:
        ValueError: If the string cannot be parsed.
    """
    m = _SIZE_RE.match(s)
    if m is None:
        raise ValueError(f"invalid memory size: {s!r}")
    value = float(m.group(1))
    suffix = m.group(2)
    if suffix is not None:
        value *= _UNITS[suffix.upper()]
    return int(value)


@dataclass(frozen=True)
class ResourceLimits:
    """Per-process resource limits applied via setrlimit(2).

    All fields default to ``None`` (unlimited).  Limits are inherited
    by children on fork, so setting them before forking provides
    group-level budgets without cgroup infrastructure.

    Note: ``memory`` maps to ``RLIMIT_AS`` (virtual address space),
    not physical RSS.  This is coarser than cgroup ``memory.max``
    but requires no kernel infrastructure.
    """

    memory: int | None = None
    """Maximum virtual address space in bytes (``RLIMIT_AS``)."""

    cpu_time: int | None = None
    """Maximum CPU time in seconds (``RLIMIT_CPU``).

    When exceeded the kernel sends ``SIGXCPU``, then ``SIGKILL``
    one second later.  This is a cumulative budget, not a rate
    limit (unlike cgroup ``cpu.max``).
    """

    nproc: int | None = None
    """Maximum number of processes for the real user ID (``RLIMIT_NPROC``)."""
