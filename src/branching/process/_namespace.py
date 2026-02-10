# SPDX-License-Identifier: Apache-2.0
"""ctypes bindings for Linux namespace and mount operations.

Provides unprivileged user namespace creation with bind-mount capability
for process-level isolation (approximating branch(BR_CREATE)).
"""

import ctypes
import ctypes.util
import os
import platform
from pathlib import Path

from ..exceptions import NamespaceError, MountError

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# Clone flags for unshare(2)
CLONE_NEWNS = 0x00020000       # New mount namespace
CLONE_NEWUSER = 0x10000000     # New user namespace
CLONE_NEWPID = 0x20000000      # New PID namespace

# Mount flags
MS_BIND = 0x1000
MS_REC = 0x4000
MS_PRIVATE = 0x40000

# Syscall numbers by architecture
_SYSCALL_TABLE = {
    "x86_64":  {"unshare": 272, "mount": 165},
    "aarch64": {"unshare": 97,  "mount": 40},
}

_arch = platform.machine()
if _arch not in _SYSCALL_TABLE:
    raise RuntimeError(
        f"Unsupported architecture '{_arch}'. "
        f"Supported: {', '.join(_SYSCALL_TABLE)}"
    )
_nr = _SYSCALL_TABLE[_arch]

__NR_unshare = _nr["unshare"]
__NR_mount = _nr["mount"]


def unshare(flags: int) -> None:
    """Call unshare(2) to disassociate parts of the execution context.

    Args:
        flags: Combination of CLONE_NEW* flags.

    Raises:
        OSError: If the syscall fails (e.g., EPERM without CAP_SYS_ADMIN).
    """
    ret = _libc.syscall(__NR_unshare, ctypes.c_int(flags))
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"unshare(0x{flags:x}): {os.strerror(err)}")


def setup_user_ns() -> None:
    """Create a new user + mount namespace and configure uid/gid maps.

    Must be called after fork(), before bind_mount(). Performs:
    1. unshare(CLONE_NEWUSER | CLONE_NEWNS)
    2. Write "deny" to /proc/self/setgroups (kernel requirement)
    3. Map current uid/gid into the new namespace

    Raises:
        NamespaceError: If any step fails.
    """
    uid = os.getuid()
    gid = os.getgid()

    try:
        unshare(CLONE_NEWUSER | CLONE_NEWNS)
    except OSError as e:
        raise NamespaceError(
            f"unshare(CLONE_NEWUSER | CLONE_NEWNS) failed: {e}"
        ) from e

    try:
        Path("/proc/self/setgroups").write_text("deny\n")
    except OSError as e:
        raise NamespaceError(f"Failed to write /proc/self/setgroups: {e}") from e

    try:
        Path("/proc/self/uid_map").write_text(f"0 {uid} 1\n")
    except OSError as e:
        raise NamespaceError(f"Failed to write uid_map: {e}") from e

    try:
        Path("/proc/self/gid_map").write_text(f"0 {gid} 1\n")
    except OSError as e:
        raise NamespaceError(f"Failed to write gid_map: {e}") from e


def bind_mount(source: str | Path, target: str | Path) -> None:
    """Bind-mount source onto target using mount(2).

    Args:
        source: Source path to bind.
        target: Target mountpoint (must exist).

    Raises:
        MountError: If the mount syscall fails.
    """
    src = str(source).encode("utf-8")
    tgt = str(target).encode("utf-8")

    ret = _libc.syscall(
        __NR_mount,
        src,
        tgt,
        ctypes.c_void_p(None),
        ctypes.c_ulong(MS_BIND),
        ctypes.c_void_p(None),
    )
    if ret < 0:
        err = ctypes.get_errno()
        raise MountError(
            f"bind_mount({source} -> {target}): {os.strerror(err)}"
        )


def make_mount_private(target: str | Path) -> None:
    """Make a mount private + recursive to prevent propagation.

    Args:
        target: Mountpoint to make private.

    Raises:
        MountError: If the mount syscall fails.
    """
    tgt = str(target).encode("utf-8")

    ret = _libc.syscall(
        __NR_mount,
        b"none",
        tgt,
        ctypes.c_void_p(None),
        ctypes.c_ulong(MS_REC | MS_PRIVATE),
        ctypes.c_void_p(None),
    )
    if ret < 0:
        err = ctypes.get_errno()
        raise MountError(
            f"make_mount_private({target}): {os.strerror(err)}"
        )
