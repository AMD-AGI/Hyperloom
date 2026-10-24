# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Triton's file cache as its releases write and read it, for the tests that need a kernel's index of its files.

``put_kernel`` is what a compile does: every product in ``<cache dir>/<hash>/``, then ``__grp__<kernel>.json`` listing them.
Release ``3.2`` lists each file by absolute path and ``get_group`` drops a listed path that does not exist, so a kernel
whose metadata file is not among what is left is compiled again; release ``current`` lists bare names, joined to the
group's own directory by ``get_group``. Nothing else of Triton is played.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["PRODUCTS", "RELEASES", "find_kernel", "get_group", "put_kernel"]

PRODUCTS = ("kernel.json", "kernel.ttir", "kernel.amdgcn", "kernel.hsaco")
RELEASES = ("3.2", "current")
_GROUP = "__grp__kernel.json"


def put_kernel(
    cache_dir: Path, key: str, *, release: str = "3.2", extra: dict[str, bytes] | None = None
) -> dict[str, str]:
    """Compile a kernel into ``cache_dir/key``: its products, then the group that lists them; the files by absolute path."""
    directory = cache_dir / key
    directory.mkdir(parents=True, exist_ok=True)
    group = {}
    for name, data in {**{name: f"{key} {name}".encode() for name in PRODUCTS}, **(extra or {})}.items():
        (directory / name).write_bytes(data)
        group[name] = str(directory / name)
    listed = group if release == "3.2" else sorted(group)
    (directory / _GROUP).write_text(json.dumps({"child_paths": listed}))
    return group


def get_group(cache_dir: Path, key: str, *, release: str = "3.2") -> dict[str, str] | None:
    """The files of the kernel Triton finds, or ``None`` when there is no group (or, in ``current``, a file is missing)."""
    path = cache_dir / key / _GROUP
    if not path.exists():
        return None
    children = json.loads(path.read_text()).get("child_paths")
    if children is None:
        return None
    if release == "3.2":
        return {name: target for name, target in children.items() if Path(target).exists()}
    found = {name: str(path.parent / name) for name in children}
    return found if all(Path(target).exists() for target in found.values()) else None


def find_kernel(cache_dir: Path, key: str, *, release: str = "3.2") -> bool:
    """Whether Triton would find the kernel compiled (``True``) or compile it again (``False``, and it does, here)."""
    group = get_group(cache_dir, key, release=release)
    if group is not None and "kernel.json" in group:
        return True
    put_kernel(cache_dir, key, release=release)
    return False
