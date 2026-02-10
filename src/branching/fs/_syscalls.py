# SPDX-License-Identifier: Apache-2.0
"""
Low-level ctypes bindings for Linux mount syscalls.

Provides Python wrappers for:
- New mount API (fsopen, fsconfig, fsmount, move_mount)
- Classic mount API (mount, umount)
- ioctl for filesystem-specific operations
"""

import ctypes
import ctypes.util
import os
import platform
from typing import Optional

# Load libc
_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

# Syscall numbers by architecture
_SYSCALL_TABLE = {
    "x86_64": {
        "fsopen": 430, "fsconfig": 431, "fsmount": 432,
        "move_mount": 429, "mount": 165, "umount2": 166,
    },
    "aarch64": {
        "fsopen": 430, "fsconfig": 431, "fsmount": 432,
        "move_mount": 429, "mount": 40, "umount2": 39,
    },
}

_arch = platform.machine()
if _arch not in _SYSCALL_TABLE:
    raise RuntimeError(
        f"Unsupported architecture '{_arch}'. "
        f"Supported: {', '.join(_SYSCALL_TABLE)}"
    )
_nr = _SYSCALL_TABLE[_arch]

__NR_fsopen = _nr["fsopen"]
__NR_fsconfig = _nr["fsconfig"]
__NR_fsmount = _nr["fsmount"]
__NR_move_mount = _nr["move_mount"]
__NR_mount = _nr["mount"]
__NR_umount2 = _nr["umount2"]

# fsconfig commands
FSCONFIG_SET_FLAG = 0
FSCONFIG_SET_STRING = 1
FSCONFIG_SET_FD = 5
FSCONFIG_CMD_CREATE = 6

# move_mount flags
MOVE_MOUNT_F_EMPTY_PATH = 0x00000004

# mount flags
MS_REMOUNT = 32
MS_RDONLY = 1

# AT_FDCWD for path resolution relative to current directory
AT_FDCWD = -100


def _check_error(ret: int, msg: str) -> int:
    """Check syscall return value and raise OSError on failure."""
    if ret < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"{msg}: {os.strerror(err)}")
    return ret


def fsopen(fstype: str, flags: int = 0) -> int:
    """
    Open a filesystem context for the new mount API.

    Args:
        fstype: Filesystem type name (e.g., 'daxfs')
        flags: Optional flags

    Returns:
        File descriptor for the filesystem context

    Raises:
        OSError: If the syscall fails
    """
    ret = _libc.syscall(
        __NR_fsopen, fstype.encode("utf-8"), ctypes.c_uint(flags)
    )
    return _check_error(ret, f"fsopen({fstype})")


def fsconfig_set_string(fd: int, key: str, value: str) -> None:
    """
    Set a string parameter on a filesystem context.

    Args:
        fd: Filesystem context fd from fsopen()
        key: Parameter name
        value: Parameter value
    """
    ret = _libc.syscall(
        __NR_fsconfig,
        ctypes.c_int(fd),
        ctypes.c_uint(FSCONFIG_SET_STRING),
        key.encode("utf-8"),
        value.encode("utf-8"),
        ctypes.c_int(0),
    )
    _check_error(ret, f"fsconfig({key}={value})")


def fsconfig_set_fd(fd: int, key: str, value_fd: int) -> None:
    """
    Set a file descriptor parameter on a filesystem context.

    Args:
        fd: Filesystem context fd from fsopen()
        key: Parameter name
        value_fd: File descriptor to pass
    """
    ret = _libc.syscall(
        __NR_fsconfig,
        ctypes.c_int(fd),
        ctypes.c_uint(FSCONFIG_SET_FD),
        key.encode("utf-8"),
        ctypes.c_void_p(None),
        ctypes.c_int(value_fd),
    )
    _check_error(ret, f"fsconfig({key}=fd:{value_fd})")


def fsconfig_cmd_create(fd: int) -> None:
    """
    Create the superblock for a filesystem context.

    Args:
        fd: Filesystem context fd from fsopen()
    """
    ret = _libc.syscall(
        __NR_fsconfig,
        ctypes.c_int(fd),
        ctypes.c_uint(FSCONFIG_CMD_CREATE),
        ctypes.c_void_p(None),
        ctypes.c_void_p(None),
        ctypes.c_int(0),
    )
    _check_error(ret, "fsconfig(CMD_CREATE)")


def fsmount(fd: int, flags: int = 0, attr_flags: int = 0) -> int:
    """
    Create a mount fd from a filesystem context.

    Args:
        fd: Filesystem context fd from fsopen()
        flags: Mount flags
        attr_flags: Attribute flags

    Returns:
        Mount file descriptor
    """
    ret = _libc.syscall(
        __NR_fsmount,
        ctypes.c_int(fd),
        ctypes.c_uint(flags),
        ctypes.c_uint(attr_flags),
    )
    return _check_error(ret, "fsmount")


def move_mount(
    from_fd: int, from_path: str, to_fd: int, to_path: str, flags: int = 0
) -> None:
    """
    Move a mount to a new location.

    Args:
        from_fd: Source directory fd (or AT_FDCWD)
        from_path: Source path (empty string with MOVE_MOUNT_F_EMPTY_PATH)
        to_fd: Target directory fd (or AT_FDCWD)
        to_path: Target mountpoint path
        flags: Move mount flags
    """
    ret = _libc.syscall(
        __NR_move_mount,
        ctypes.c_int(from_fd),
        from_path.encode("utf-8"),
        ctypes.c_int(to_fd),
        to_path.encode("utf-8"),
        ctypes.c_uint(flags),
    )
    _check_error(ret, f"move_mount -> {to_path}")


def mount(
    source: str,
    target: str,
    fstype: Optional[str],
    flags: int,
    data: Optional[str] = None,
) -> None:
    """
    Classic mount syscall.

    Args:
        source: Source device or empty string
        target: Mount target path
        fstype: Filesystem type or None
        flags: Mount flags (e.g., MS_REMOUNT)
        data: Mount options string or None
    """
    ret = _libc.syscall(
        __NR_mount,
        source.encode("utf-8") if source else b"",
        target.encode("utf-8"),
        fstype.encode("utf-8") if fstype else ctypes.c_void_p(None),
        ctypes.c_ulong(flags),
        data.encode("utf-8") if data else ctypes.c_void_p(None),
    )
    _check_error(ret, f"mount({target})")


def umount(target: str, flags: int = 0) -> None:
    """
    Unmount a filesystem.

    Args:
        target: Mount target path to unmount
        flags: Unmount flags (MNT_FORCE, MNT_DETACH, etc.)
    """
    ret = _libc.syscall(
        __NR_umount2, target.encode("utf-8"), ctypes.c_int(flags)
    )
    _check_error(ret, f"umount({target})")


def ioctl(fd: int, request: int, arg: int = 0) -> int:
    """
    Perform an ioctl operation.

    Args:
        fd: File descriptor
        request: ioctl request code
        arg: ioctl argument (default 0)

    Returns:
        ioctl return value (interpretation depends on request)
    """
    ret = _libc.ioctl(ctypes.c_int(fd), ctypes.c_ulong(request), ctypes.c_long(arg))
    return _check_error(ret, f"ioctl(0x{request:x})")
