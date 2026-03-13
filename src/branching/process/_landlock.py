# SPDX-License-Identifier: Apache-2.0
"""Landlock LSM bindings for filesystem confinement.

Provides unprivileged, per-process filesystem confinement using
Linux Landlock (5.13+).  No namespaces or capabilities required — any
process can self-confine.

Usage::

    confine_to_branch(Path("/mnt/main/@uuid"), Path("/mnt/main"))

After this call the process can read the filesystem outside the
workspace mount but can only write under the branch path.  Sibling
branches and the mount root itself are fully denied.

LANDLOCK_ACCESS_FS_REFER is included in the handled set so that
rename/link across the branch boundary is denied — a child cannot
reparent files from siblings into its branch or vice versa.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

from ..exceptions import ProcessBranchError

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# Syscall numbers (same on x86_64 and aarch64 — asm-generic)
__NR_landlock_create_ruleset = 444
__NR_landlock_add_rule = 445
__NR_landlock_restrict_self = 446

# --- Access flags (ABI v1–v5) ---

LANDLOCK_ACCESS_FS_EXECUTE = 1 << 0
LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_READ_FILE = 1 << 2
LANDLOCK_ACCESS_FS_READ_DIR = 1 << 3
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13        # ABI v2
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14     # ABI v3
LANDLOCK_ACCESS_FS_IOCTL_DEV = 1 << 15    # ABI v5

# All write-like operations (anything that mutates the filesystem).
_WRITE_ACCESS = (
    LANDLOCK_ACCESS_FS_WRITE_FILE
    | LANDLOCK_ACCESS_FS_REMOVE_DIR
    | LANDLOCK_ACCESS_FS_REMOVE_FILE
    | LANDLOCK_ACCESS_FS_MAKE_CHAR
    | LANDLOCK_ACCESS_FS_MAKE_DIR
    | LANDLOCK_ACCESS_FS_MAKE_REG
    | LANDLOCK_ACCESS_FS_MAKE_SOCK
    | LANDLOCK_ACCESS_FS_MAKE_FIFO
    | LANDLOCK_ACCESS_FS_MAKE_BLOCK
    | LANDLOCK_ACCESS_FS_MAKE_SYM
    | LANDLOCK_ACCESS_FS_REFER
    | LANDLOCK_ACCESS_FS_TRUNCATE
)

LANDLOCK_RULE_PATH_BENEATH = 1
LANDLOCK_CREATE_RULESET_VERSION = 1 << 0


# --- Structs ---

class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
        ("scoped", ctypes.c_uint64),
    ]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
    ]


# --- Low-level wrappers ---

def _create_ruleset(handled_access_fs: int) -> int:
    """Create a Landlock ruleset and return its fd."""
    attr = _LandlockRulesetAttr(handled_access_fs=handled_access_fs)
    fd = _libc.syscall(
        __NR_landlock_create_ruleset,
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint32(0),
    )
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"landlock_create_ruleset: {os.strerror(err)}")
    return fd


def _add_path_rule(ruleset_fd: int, allowed_access: int, path: Path) -> None:
    """Add a path-beneath rule to a Landlock ruleset."""
    dir_fd = os.open(str(path), os.O_PATH | os.O_DIRECTORY)
    try:
        rule = _LandlockPathBeneathAttr(
            allowed_access=allowed_access,
            parent_fd=dir_fd,
        )
        ret = _libc.syscall(
            __NR_landlock_add_rule,
            ctypes.c_int(ruleset_fd),
            ctypes.c_int(LANDLOCK_RULE_PATH_BENEATH),
            ctypes.byref(rule),
            ctypes.c_uint32(0),
        )
        if ret < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"landlock_add_rule({path}): {os.strerror(err)}")
    finally:
        os.close(dir_fd)


def _restrict_self(ruleset_fd: int) -> None:
    """Apply a Landlock ruleset to the current process."""
    ret = _libc.syscall(
        __NR_landlock_restrict_self,
        ctypes.c_int(ruleset_fd),
        ctypes.c_uint32(0),
    )
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"landlock_restrict_self: {os.strerror(err)}")


def landlock_abi_version() -> int:
    """Return the highest Landlock ABI version supported by the kernel.

    Returns 0 if Landlock is not supported.
    """
    ver = _libc.syscall(
        __NR_landlock_create_ruleset,
        ctypes.c_void_p(None),
        ctypes.c_size_t(0),
        ctypes.c_uint32(LANDLOCK_CREATE_RULESET_VERSION),
    )
    if ver < 0:
        return 0
    return ver


# --- Read access flags ---

_READ_ACCESS = (
    LANDLOCK_ACCESS_FS_EXECUTE
    | LANDLOCK_ACCESS_FS_READ_FILE
    | LANDLOCK_ACCESS_FS_READ_DIR
)

# Full access (read + write) for the branch path.
_FULL_ACCESS = _READ_ACCESS | _WRITE_ACCESS


# --- Public API ---

def confine_to_branch(branch_path: Path, mount_root: Path) -> None:
    """Confine the current process to its branch within a workspace.

    After this call the process (and any children it forks) can:
    - Execute and read any file on the filesystem **except** the
      workspace mount (``mount_root``) and sibling branches
    - Read, write, create, and remove files only under ``branch_path``

    The algorithm walks from ``/`` down to ``mount_root``.  At each
    level, every sibling directory that is *not* on the path to the
    mount root gets a read-only rule.  The mount root itself gets no
    rule (blocking reads of sibling branches), and ``branch_path``
    gets full read+write access.

    This is irreversible for the calling process.

    Args:
        branch_path: Branch workspace directory (read+write allowed).
        mount_root: Workspace mount point (parent of all branch
            virtual paths).  Reads to mount_root and its other
            children are denied.

    Raises:
        ProcessBranchError: If Landlock is unavailable or setup fails.
    """
    abi = landlock_abi_version()
    if abi < 1:
        raise ProcessBranchError(
            "Landlock not available. Requires Linux 5.13+ with "
            "CONFIG_SECURITY_LANDLOCK=y and lsm=...,landlock,..."
        )

    # Build ABI-aware access masks.
    write = _WRITE_ACCESS
    if abi < 2:
        write &= ~LANDLOCK_ACCESS_FS_REFER
    if abi < 3:
        write &= ~LANDLOCK_ACCESS_FS_TRUNCATE

    read = _READ_ACCESS
    full = read | write
    handled = full

    ruleset_fd = _create_ruleset(handled)
    try:
        # Walk from / to mount_root, adding read rules for siblings.
        _add_sibling_read_rules(ruleset_fd, read, mount_root)
        # Allow full access to the branch path.
        _add_path_rule(ruleset_fd, full, branch_path)
        # Enforce.
        _set_no_new_privs()
        _restrict_self(ruleset_fd)
    except OSError as e:
        raise ProcessBranchError(f"Landlock confinement failed: {e}") from e
    finally:
        os.close(ruleset_fd)


def _add_sibling_read_rules(
    ruleset_fd: int, read_access: int, mount_root: Path,
) -> None:
    """Add read rules for every directory that is NOT on the path to *mount_root*.

    Walks each ancestor of *mount_root* starting from ``/``.  At each
    level, lists directory entries and adds a read rule for every entry
    that is not the next component on the path to the mount root.

    Example for mount_root=/mnt/main:
        /       → add rules for /usr, /lib, /etc, /proc, ... (skip /mnt)
        /mnt    → add rules for /mnt/foo, /mnt/bar, ...     (skip /mnt/main)
        /mnt/main → no rule (the mount root itself is blocked)
    """
    mount_root = mount_root.resolve()
    parts = mount_root.parts  # ('/', 'mnt', 'main')

    for depth in range(len(parts) - 1):
        # Current ancestor directory
        ancestor = Path(*parts[: depth + 1])
        # The child on the path to mount_root that we must skip
        skip = parts[depth + 1]

        try:
            entries = os.listdir(ancestor)
        except OSError:
            continue

        for entry in entries:
            if entry == skip:
                continue
            entry_path = ancestor / entry
            if not entry_path.is_dir():
                # Landlock path rules require directories; for regular
                # files at the ancestor level we skip (rare — /vmlinuz etc.)
                continue
            try:
                _add_path_rule(ruleset_fd, read_access, entry_path)
            except OSError:
                # Entry may have vanished or be inaccessible (e.g. /proc/1)
                continue


def _set_no_new_privs() -> None:
    """Set PR_SET_NO_NEW_PRIVS — required before landlock_restrict_self."""
    import ctypes as _ct
    PR_SET_NO_NEW_PRIVS = 38
    ret = _libc.prctl(
        _ct.c_int(PR_SET_NO_NEW_PRIVS),
        _ct.c_ulong(1),
        _ct.c_ulong(0),
        _ct.c_ulong(0),
        _ct.c_ulong(0),
    )
    if ret < 0:
        err = _ct.get_errno()
        raise OSError(err, f"prctl(PR_SET_NO_NEW_PRIVS): {os.strerror(err)}")
