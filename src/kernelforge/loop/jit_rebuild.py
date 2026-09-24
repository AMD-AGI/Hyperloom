# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Force JIT-compiled kernels to rebuild from the CURRENT source."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from kernelforge.llm.git import git
from kernelforge.loop.aiter_cache import activate_aiter_cache_for_sources

_CPP_EXTS = (".cu", ".cuh", ".hip", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp")


def force_jit_rebuild(paths: Iterable[str]) -> None:
    """Make the framework recompile the kernel from the current source."""
    source_paths = [str(path) for path in paths if path]
    strs = [path.lower() for path in source_paths]
    if not strs:
        return
    # Only C/C++ HIP kernels have the prebuilt-.so shadowing problem; forcing a rebuild for a Triton (.py) task
    # would recompile aiter's C++ for nothing.
    if not any(s.endswith(_CPP_EXTS) for s in strs):
        return
    joined = " ".join(strs)

    if "aiter" in joined:
        # A fresh source digest selects an empty private shard and therefore rebuilds exactly once.
        activate_aiter_cache_for_sources(source_paths)


def tracked_source_changes(workspace: str | Path) -> list[str]:
    """Return existing tracked files changed from HEAD, as absolute paths."""

    root = Path(workspace).expanduser().resolve()
    result = git("diff", "--name-only", "-z", "HEAD", "--", ".", cwd=root, text=False)
    changed: list[str] = []
    for encoded in result.stdout.split(b"\0"):
        if not encoded:
            continue
        path = (root / encoded.decode(errors="surrogateescape")).resolve()
        if path.is_file() and str(path) not in changed:
            changed.append(str(path))
    return changed


def force_jit_rebuild_for_changes(
    workspace: str | Path,
    declared_paths: Iterable[str] = (),
) -> None:
    """Rebuild from declared entry points plus every actual tracked source edit."""

    paths = list(
        dict.fromkeys(
            [
                *(str(path) for path in declared_paths if path),
                *tracked_source_changes(workspace),
            ]
        )
    )
    force_jit_rebuild(paths)
