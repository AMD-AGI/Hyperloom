# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared filesystem-classification helpers."""

from __future__ import annotations

import os

# Mounts that can be revoked mid-run: a process whose cwd lives on one sees
# relative-path writes fail with ENOENT after a flap, so callers place anything
# they must keep writing to on local disk instead.
_NETWORK_FS_TYPES: frozenset[str] = frozenset(
    {
        "nfs",
        "nfs4",
        "cifs",
        "smb3",
        "lustre",
        "glusterfs",
        "ceph",
        "fuse.weka",
        "wekafs",
        "wekafsgw",
        "fuse.juicefs",
        "fuse.s3fs",
        "fuse.sshfs",
        "9p",
    }
)


def _path_fstype(path: str) -> str:
    """Return the filesystem type backing ``path`` per ``/proc/mounts``.

    Picks the longest mountpoint that is a prefix of the resolved path, and
    ``""`` when it cannot be determined (non-Linux, unreadable
    ``/proc/mounts``), which reads as "assume local".
    """
    try:
        rp = os.path.realpath(path)
    except OSError:
        return ""
    best_mp = ""
    best_type = ""
    try:
        with open("/proc/mounts", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                # /proc/mounts octal-escapes spaces in the mountpoint.
                try:
                    mp = parts[1].encode("latin-1").decode("unicode_escape")
                except (UnicodeDecodeError, UnicodeEncodeError):
                    mp = parts[1]
                fstype = parts[2]
                norm = mp.rstrip("/") or "/"
                is_under = norm == "/" or rp == norm or rp.startswith(norm + "/")
                if is_under and len(norm) >= len(best_mp):
                    best_mp = norm
                    best_type = fstype
    except OSError:
        return ""
    return best_type


def is_network_fs(path: str) -> bool:
    """True when ``path`` is backed by a revocable network filesystem."""
    return _path_fstype(path).lower() in _NETWORK_FS_TYPES
