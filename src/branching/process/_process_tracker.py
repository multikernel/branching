# SPDX-License-Identifier: Apache-2.0
"""Process tracking and reliable kill for branch children.

Uses a pre-compiled BPF LSM program loaded via libbpf to inescapably
track all descendants of a branch and atomically deny new forks during
teardown.

Requires:
* ``libbpf.so`` (typically already installed on modern Linux)
* A kernel with ``CONFIG_BPF_LSM=y`` and ``lsm=...,bpf,...``
* A delegated bpffs mount (for unprivileged loading via BPF token),
  or ``CAP_BPF`` for privileged loading.

The default bpffs path is ``/sys/fs/bpf``.  Override with the
``BRANCHING_BPFFS`` environment variable or the
``LIBBPF_BPF_TOKEN_PATH`` environment variable (checked by libbpf
automatically).

Use :func:`get_tracker` to obtain the tracker instance.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import signal
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

_log = logging.getLogger(__name__)

# Pre-compiled BPF object path (next to this file, under bpf/)
_BPF_OBJ = Path(__file__).parent / "bpf" / "branch_tracker.bpf.o"

# Default bpffs mount point for BPF token delegation
_DEFAULT_BPFFS = "/sys/fs/bpf"


# -------------------------------------------------------------------
# Protocol
# -------------------------------------------------------------------

@runtime_checkable
class ProcessTracker(Protocol):
    """Interface for branch process tracking and kill."""

    def register(self, pid: int) -> int:
        """Start tracking *pid* and its future descendants.

        Returns a branch_id that identifies this branch for later
        :meth:`kill_branch` / :meth:`cleanup` calls.
        """
        ...

    def kill_branch(self, branch_id: int) -> None:
        """Kill all tracked processes belonging to *branch_id*."""
        ...

    def cleanup(self, branch_id: int) -> None:
        """Remove tracking state for *branch_id*."""
        ...


# -------------------------------------------------------------------
# Minimal libbpf ctypes bindings
# -------------------------------------------------------------------

class _BpfObjectOpenOpts(ctypes.Structure):
    """Mirrors ``struct bpf_object_open_opts`` from libbpf.h."""
    _fields_ = [
        ("sz", ctypes.c_size_t),                  # 0
        ("object_name", ctypes.c_char_p),          # 8
        ("relaxed_maps", ctypes.c_bool),           # 16
        ("_pad1", ctypes.c_char * 3),              # 17
        ("_pad1b", ctypes.c_char * 4),             # 20 (align to 8)
        ("pin_root_path", ctypes.c_char_p),        # 24
        ("_stub_attach_prog_fd", ctypes.c_uint32), # 32
        ("_pad2", ctypes.c_char * 4),              # 36
        ("kconfig", ctypes.c_char_p),              # 40
        ("btf_custom_path", ctypes.c_char_p),      # 48
        ("kernel_log_buf", ctypes.c_char_p),       # 56
        ("kernel_log_size", ctypes.c_size_t),      # 64
        ("kernel_log_level", ctypes.c_uint32),     # 72
        ("_pad3", ctypes.c_char * 4),              # 76
        ("bpf_token_path", ctypes.c_char_p),       # 80
    ]


# bpf_map_update_elem flags
_BPF_ANY = 0


class _Libbpf:
    """Thin ctypes wrapper around the libbpf functions we need."""

    def __init__(self) -> None:
        name = ctypes.util.find_library("bpf")
        if name is None:
            for candidate in ("libbpf.so.1", "libbpf.so.0", "libbpf.so"):
                try:
                    self._lib = ctypes.CDLL(candidate, use_errno=True)
                    break
                except OSError:
                    continue
            else:
                raise OSError("libbpf shared library not found")
        else:
            self._lib = ctypes.CDLL(name, use_errno=True)

        self._setup_prototypes()

    def _setup_prototypes(self) -> None:
        L = self._lib

        L.bpf_object__open_file.argtypes = [
            ctypes.c_char_p, ctypes.POINTER(_BpfObjectOpenOpts),
        ]
        L.bpf_object__open_file.restype = ctypes.c_void_p

        L.bpf_object__load.argtypes = [ctypes.c_void_p]
        L.bpf_object__load.restype = ctypes.c_int

        L.bpf_object__close.argtypes = [ctypes.c_void_p]
        L.bpf_object__close.restype = None

        L.bpf_object__find_program_by_name.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
        ]
        L.bpf_object__find_program_by_name.restype = ctypes.c_void_p

        L.bpf_program__attach_lsm.argtypes = [ctypes.c_void_p]
        L.bpf_program__attach_lsm.restype = ctypes.c_void_p

        L.bpf_object__find_map_by_name.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p,
        ]
        L.bpf_object__find_map_by_name.restype = ctypes.c_void_p

        L.bpf_map__fd.argtypes = [ctypes.c_void_p]
        L.bpf_map__fd.restype = ctypes.c_int

        L.bpf_map_update_elem.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint64,
        ]
        L.bpf_map_update_elem.restype = ctypes.c_int

        L.bpf_map_lookup_elem.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        L.bpf_map_lookup_elem.restype = ctypes.c_int

        L.bpf_map_delete_elem.argtypes = [
            ctypes.c_int, ctypes.c_void_p,
        ]
        L.bpf_map_delete_elem.restype = ctypes.c_int

        L.bpf_map_get_next_key.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        L.bpf_map_get_next_key.restype = ctypes.c_int

        L.bpf_link__destroy.argtypes = [ctypes.c_void_p]
        L.bpf_link__destroy.restype = ctypes.c_int

    # -- Object lifecycle --

    def open_file(
        self, path: str, bpf_token_path: str | None = None,
    ) -> ctypes.c_void_p:
        opts = _BpfObjectOpenOpts()
        opts.sz = ctypes.sizeof(opts)
        if bpf_token_path is not None:
            opts.bpf_token_path = bpf_token_path.encode()
        obj = self._lib.bpf_object__open_file(
            path.encode(), ctypes.byref(opts),
        )
        if not obj:
            raise OSError(f"bpf_object__open_file({path}) failed")
        return obj

    def load(self, obj: ctypes.c_void_p) -> None:
        ret = self._lib.bpf_object__load(obj)
        if ret < 0:
            raise OSError(f"bpf_object__load failed: {ret}")

    def close(self, obj: ctypes.c_void_p) -> None:
        self._lib.bpf_object__close(obj)

    # -- Program --

    def find_program(self, obj: ctypes.c_void_p, name: str) -> ctypes.c_void_p:
        prog = self._lib.bpf_object__find_program_by_name(
            obj, name.encode(),
        )
        if not prog:
            raise OSError(f"program '{name}' not found")
        return prog

    def attach_lsm(self, prog: ctypes.c_void_p) -> ctypes.c_void_p:
        link = self._lib.bpf_program__attach_lsm(prog)
        if not link:
            raise OSError("bpf_program__attach_lsm failed")
        return link

    # -- Map --

    def find_map_fd(self, obj: ctypes.c_void_p, name: str) -> int:
        m = self._lib.bpf_object__find_map_by_name(obj, name.encode())
        if not m:
            raise OSError(f"map '{name}' not found")
        fd = self._lib.bpf_map__fd(m)
        if fd < 0:
            raise OSError(f"bpf_map__fd('{name}') failed: {fd}")
        return fd

    def map_update(self, fd: int, key: ctypes.Array, val: ctypes.Array) -> None:
        ret = self._lib.bpf_map_update_elem(
            fd, ctypes.byref(key), ctypes.byref(val), _BPF_ANY,
        )
        if ret < 0:
            err = ctypes.get_errno()
            raise OSError(err, f"bpf_map_update_elem: {os.strerror(err)}")

    def map_lookup(self, fd: int, key: ctypes.Array, val: ctypes.Array) -> bool:
        return self._lib.bpf_map_lookup_elem(
            fd, ctypes.byref(key), ctypes.byref(val),
        ) == 0

    def map_delete(self, fd: int, key: ctypes.Array) -> None:
        self._lib.bpf_map_delete_elem(fd, ctypes.byref(key))

    def map_get_next_key(
        self, fd: int, key: ctypes.Array | None, next_key: ctypes.Array,
    ) -> bool:
        key_ptr = ctypes.byref(key) if key is not None else ctypes.c_void_p(None)
        return self._lib.bpf_map_get_next_key(
            fd, key_ptr, ctypes.byref(next_key),
        ) == 0

    def link_destroy(self, link: ctypes.c_void_p) -> None:
        self._lib.bpf_link__destroy(link)


# -------------------------------------------------------------------
# BPF LSM process tracker
# -------------------------------------------------------------------

class BpfProcessTracker:
    """BPF LSM-based process tracker (singleton).

    Loads a pre-compiled BPF object (``branch_tracker.bpf.o``) via
    libbpf and attaches LSM hooks for inescapable descendant tracking.

    When a delegated bpffs is available, loading works without
    ``CAP_BPF`` by obtaining a BPF token from the mount.
    """

    _instance: BpfProcessTracker | None = None

    @classmethod
    def get(cls) -> BpfProcessTracker:
        """Return the singleton instance.

        Raises:
            RuntimeError: If the BPF tracker cannot be initialized.
        """
        if cls._instance is not None:
            return cls._instance
        try:
            cls._instance = cls()
            return cls._instance
        except Exception as exc:
            raise RuntimeError(
                f"BPF process tracker unavailable: {exc}\n"
                "Ensure:\n"
                "  1. libbpf.so is installed\n"
                "  2. Kernel has CONFIG_BPF_LSM=y and lsm=...,bpf,...\n"
                "  3. Either CAP_BPF is available, or a delegated bpffs is\n"
                "     mounted (set BRANCHING_BPFFS to the mount point)"
            ) from exc

    def __init__(self) -> None:
        if not _BPF_OBJ.exists():
            raise FileNotFoundError(
                f"Pre-compiled BPF object not found: {_BPF_OBJ}"
            )

        self._lib = _Libbpf()

        # Determine bpffs path for token delegation.
        # BRANCHING_BPFFS overrides; otherwise use default.
        # libbpf also checks LIBBPF_BPF_TOKEN_PATH on its own.
        bpffs = os.environ.get("BRANCHING_BPFFS", _DEFAULT_BPFFS)

        # Open with token path so libbpf creates a BPF token
        # from the delegated bpffs mount (works unprivileged).
        # If the bpffs doesn't have delegation, libbpf falls back
        # to direct bpf() calls (which need CAP_BPF).
        self._obj = self._lib.open_file(str(_BPF_OBJ), bpf_token_path=bpffs)
        try:
            self._lib.load(self._obj)
        except OSError:
            self._lib.close(self._obj)
            raise

        # Attach LSM hooks
        self._links: list[ctypes.c_void_p] = []
        try:
            for prog_name in ("branch_task_alloc", "branch_task_free"):
                prog = self._lib.find_program(self._obj, prog_name)
                link = self._lib.attach_lsm(prog)
                self._links.append(link)
        except OSError:
            self._detach()
            raise

        # Get map file descriptors
        self._pids_fd = self._lib.find_map_fd(self._obj, "branch_pids")
        self._state_fd = self._lib.find_map_fd(self._obj, "branch_state")

    def _detach(self) -> None:
        """Detach all links and close the BPF object."""
        for link in self._links:
            self._lib.link_destroy(link)
        self._links.clear()
        self._lib.close(self._obj)

    def _pids_for_branch(self, branch_id: int) -> list[int]:
        """Iterate the branch_pids map and return all PIDs for *branch_id*."""
        pids: list[int] = []
        next_key = ctypes.c_uint32()

        if not self._lib.map_get_next_key(self._pids_fd, None, next_key):
            return pids

        while True:
            key_arr = (ctypes.c_uint32 * 1)(next_key.value)
            val_arr = (ctypes.c_uint64 * 1)()
            if self._lib.map_lookup(self._pids_fd, key_arr, val_arr):
                if val_arr[0] == branch_id:
                    pids.append(next_key.value)

            cur_key = (ctypes.c_uint32 * 1)(next_key.value)
            if not self._lib.map_get_next_key(self._pids_fd, cur_key, next_key):
                break

        return pids

    def register(self, pid: int) -> int:
        """Register *pid* as a branch root.  Returns *pid* as the branch_id."""
        key = (ctypes.c_uint32 * 1)(pid)
        val = (ctypes.c_uint64 * 1)(pid)  # branch_id == root pid
        self._lib.map_update(self._pids_fd, key, val)
        return pid

    def kill_branch(self, branch_id: int) -> None:
        """Block new forks, then kill all processes in the branch."""
        # 1. Set state to "killing" — BPF hook will deny new forks
        bid_key = (ctypes.c_uint64 * 1)(branch_id)
        state_val = (ctypes.c_uint32 * 1)(1)
        self._lib.map_update(self._state_fd, bid_key, state_val)

        # 2. Kill all tracked processes
        for pid in self._pids_for_branch(branch_id):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def cleanup(self, branch_id: int) -> None:
        """Remove all tracking state for *branch_id*."""
        for pid in self._pids_for_branch(branch_id):
            self._lib.map_delete(
                self._pids_fd, (ctypes.c_uint32 * 1)(pid),
            )

        self._lib.map_delete(
            self._state_fd, (ctypes.c_uint64 * 1)(branch_id),
        )


# -------------------------------------------------------------------
# Factory (convenience alias)
# -------------------------------------------------------------------

#: Convenience alias for ``BpfProcessTracker.get()``.
get_tracker = BpfProcessTracker.get
