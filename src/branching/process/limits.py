# SPDX-License-Identifier: Apache-2.0
"""Resource limits for per-branch cgroup constraints."""

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
    """Per-branch resource limits applied via cgroup v2.

    All fields default to ``None`` (unlimited).  A ``ResourceLimits()``
    with all ``None`` fields still triggers process isolation but applies
    no cgroup limits.
    """

    memory: int | None = None
    """Maximum memory in bytes (written to ``memory.max``)."""

    cpu: float | None = None
    """Fraction of one CPU (0.5 = 50%).  Written to ``cpu.max``."""

    memory_high: int | None = None
    """Soft memory throttle in bytes (written to ``memory.high``).

    When usage exceeds this value the kernel reclaims aggressively but
    does *not* OOM-kill the process.
    """

    oom_group: bool = False
    """Atomic OOM termination (written to ``memory.oom.group``).

    When ``True``, all processes in the cgroup are killed together on
    OOM rather than picking a single victim.
    """
