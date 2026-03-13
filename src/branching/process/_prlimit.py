# SPDX-License-Identifier: Apache-2.0
"""Process resource limits via setrlimit(2).

Lightweight alternative to cgroup v2 controllers — no cgroupfs
interaction, no directory creation/teardown.  Limits are inherited
by children on fork, making them suitable for group-level budgets.
"""

from __future__ import annotations

import resource
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .limits import ResourceLimits


def apply_limits(limits: ResourceLimits) -> None:
    """Apply *limits* to the current process via setrlimit(2).

    Should be called in the child process after fork, before executing
    the target callable.  Limits are inherited by any further children.

    Args:
        limits: Resource limits to apply.  ``None`` fields are skipped.

    Raises:
        OSError: If a setrlimit call fails (e.g. exceeds hard limit).
    """
    if limits.memory is not None:
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory, limits.memory))
    if limits.cpu_time is not None:
        resource.setrlimit(
            resource.RLIMIT_CPU, (limits.cpu_time, limits.cpu_time),
        )
    if limits.nproc is not None:
        resource.setrlimit(
            resource.RLIMIT_NPROC, (limits.nproc, limits.nproc),
        )
