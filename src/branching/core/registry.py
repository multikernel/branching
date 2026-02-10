# SPDX-License-Identifier: Apache-2.0
"""Filesystem registration and auto-detection."""

from pathlib import Path
from typing import Dict, Type, Optional

from .base import FSBackend
from ..exceptions import BranchingError
from ..fs._mount import find_mount

# Global registry of filesystem implementations
_registry: Dict[str, Type[FSBackend]] = {}


def register(fstype: str, fs_class: Type[FSBackend]) -> None:
    """
    Register a filesystem implementation.

    Called by filesystem modules during import to register themselves.

    Args:
        fstype: Filesystem type name (e.g., 'daxfs', 'fuse.branchfs')
        fs_class: FSBackend subclass that handles this filesystem
    """
    _registry[fstype] = fs_class


def get_fs(fstype: str) -> Optional[Type[FSBackend]]:
    """
    Get filesystem implementation by type name.

    Args:
        fstype: Filesystem type name

    Returns:
        FSBackend subclass or None if not registered
    """
    return _registry.get(fstype)


def detect_fs_for_mount(path: Path) -> Type[FSBackend]:
    """
    Auto-detect filesystem type for a mountpoint and return its implementation.

    Args:
        path: Path to mounted filesystem

    Returns:
        FSBackend subclass for the detected filesystem

    Raises:
        BranchingError: If no mount found or filesystem type not supported
    """
    # Ensure all implementations are loaded
    _ensure_implementations_loaded()

    mount_info = find_mount(path)
    if mount_info is None:
        raise BranchingError(f"No mount found at {path}")

    fstype = mount_info.fstype

    # FUSE mounts may report bare "fuse" without a subtype suffix.
    # Try to recover the full type from mount options (subtype=X) or
    # from the mount source (e.g. source "branchfs" → "fuse.branchfs").
    if fstype == "fuse" and fstype not in _registry:
        subtype = mount_info.get_option("subtype")
        if subtype:
            fstype = f"fuse.{subtype}"
        elif mount_info.source and f"fuse.{mount_info.source}" in _registry:
            fstype = f"fuse.{mount_info.source}"

    fs_class = _registry.get(fstype)
    if fs_class is None:
        raise BranchingError(
            f"Unsupported filesystem type '{fstype}' at {path}. "
            f"Supported types: {', '.join(_registry.keys()) or 'none'}"
        )

    return fs_class


def list_supported() -> list[str]:
    """Return list of supported filesystem types."""
    _ensure_implementations_loaded()
    return list(_registry.keys())


def _ensure_implementations_loaded() -> None:
    """Ensure all filesystem implementations are loaded.

    Imports backend modules which register themselves on import.
    """
    if "daxfs" not in _registry:
        try:
            from ..fs import daxfs as _  # noqa: F401
        except ImportError:
            pass
    if "fuse.branchfs" not in _registry:
        try:
            from ..fs import branchfs as _  # noqa: F401
        except ImportError:
            pass
