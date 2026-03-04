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
    Find mount info for the nearest enclosing mountpoint.

    Walks up from *mountpoint* toward the root, returning the first
    (nearest) mount that matches.  This lets callers pass a path
    *inside* a mount (e.g. a branch virtual path) and still find the
    underlying filesystem.

    Args:
        mountpoint: Path to look up (exact mountpoint or path inside one)
        fstype: Optional filesystem type filter

    Returns:
        MountInfo if found, None otherwise
    """
    mountpoint = mountpoint.resolve()
    mounts = parse_mounts()
    # Build map: resolved mountpoint → MountInfo (last wins for overlays)
    mount_map: Dict[Path, MountInfo] = {}
    for m in mounts:
        mp = m.mountpoint.resolve()
        if fstype is None or m.fstype == fstype:
            mount_map[mp] = m
    # Walk up from path to root, return nearest enclosing mount
    path = mountpoint
    while True:
        if path in mount_map:
            return mount_map[path]
        parent = path.parent
        if parent == path:
            break
        path = parent
    return None


