# SPDX-License-Identifier: Apache-2.0
"""Mount information parsing from /proc/mounts."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

PROC_MOUNTS = "/proc/mounts"


@dataclass
class MountInfo:
    """Information about a mounted filesystem."""

    mountpoint: Path
    source: str
    fstype: str
    options: str

    @property
    def options_dict(self) -> Dict[str, Optional[str]]:
        """Parse options into a dictionary.

        Options without values (flags) have None as value.
        Options with values have the value as string.
        """
        result: Dict[str, Optional[str]] = {}
        for opt in self.options.split(","):
            if "=" in opt:
                key, value = opt.split("=", 1)
                result[key] = value
            else:
                result[opt] = None
        return result

    def get_option(self, key: str) -> Optional[str]:
        """Get an option value by key, or None if not present."""
        return self.options_dict.get(key)

    def has_option(self, key: str) -> bool:
        """Check if an option flag is present."""
        return key in self.options_dict

    @property
    def is_writable(self) -> bool:
        """Check if the mount is read-write."""
        return self.has_option("rw")


def parse_mounts(path: str = PROC_MOUNTS) -> List[MountInfo]:
    """
    Parse /proc/mounts and return list of MountInfo.

    Args:
        path: Path to mounts file (default /proc/mounts)

    Returns:
        List of MountInfo for all mounts
    """
    mounts = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                mounts.append(
                    MountInfo(
                        source=parts[0],
                        mountpoint=Path(parts[1]),
                        fstype=parts[2],
                        options=parts[3],
                    )
                )
    return mounts


def find_mount(mountpoint: Path, fstype: Optional[str] = None) -> Optional[MountInfo]:
    """
    Find mount info for a specific mountpoint.

    Args:
        mountpoint: Path to the mountpoint
        fstype: Optional filesystem type filter

    Returns:
        MountInfo if found, None otherwise
    """
    mountpoint = mountpoint.resolve()
    for mount in parse_mounts():
        if mount.mountpoint.resolve() == mountpoint:
            if fstype is None or mount.fstype == fstype:
                return mount
    return None


def find_mounts_by_type(fstype: str) -> List[MountInfo]:
    """
    Find all mounts of a specific filesystem type.

    Args:
        fstype: Filesystem type name (e.g., 'daxfs')

    Returns:
        List of MountInfo for matching mounts
    """
    return [m for m in parse_mounts() if m.fstype == fstype]


def find_any_mount(fstype: str) -> Optional[MountInfo]:
    """
    Find any mount of the specified filesystem type.

    Useful for getting backing store info from an existing mount.

    Args:
        fstype: Filesystem type name

    Returns:
        First matching MountInfo or None
    """
    mounts = find_mounts_by_type(fstype)
    return mounts[0] if mounts else None
