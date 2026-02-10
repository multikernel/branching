# SPDX-License-Identifier: Apache-2.0
"""Exception hierarchy for branching operations."""


class BranchingError(Exception):
    """Base exception for all branching errors."""

    pass


class MountError(BranchingError):
    """Mount or remount operation failed."""

    pass


class BranchError(BranchingError):
    """Branch operation failed."""

    pass


class BranchStaleError(BranchError):
    """Branch became stale (sibling was committed). Corresponds to ESTALE."""

    pass


class BranchNotFoundError(BranchError):
    """Branch does not exist."""

    pass


class CommitError(BranchError):
    """Commit operation failed."""

    pass


class ConflictError(CommitError):
    """Commit rejected — a sibling branch already committed (ESTALE)."""

    pass


class AbortError(BranchError):
    """Abort operation failed."""

    pass


class ProcessBranchError(BranchingError):
    """Process branching operation failed."""

    pass


class ForkError(ProcessBranchError):
    """Fork operation failed."""

    pass


class NamespaceError(ProcessBranchError):
    """Namespace setup failed."""

    pass


class MemoryBranchError(BranchingError):
    """Memory branching operation failed."""

    pass
