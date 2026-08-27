# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Force JIT-compiled kernels to rebuild from the CURRENT source.

forge-loop optimizes real kernels in place, but aiter ships a prebuilt in-tree
``.so`` and loads it WITHOUT re-checking the source. An agent edit to a HIP
``.cu``/``.cuh`` would then be silently ignored — validation would pass the
ORIGINAL kernel and the reported speedup would never move.

Some upper-layer frameworks apply this same forcing centrally, but forge-loop is
a general engine that other frameworks drive directly over the CLI. This module
is forge's OWN safety net so an agent's edits take effect regardless of the driver.

Scope: aiter HIP (C/C++) kernels only. Triton / Python kernels re-key their JIT
on the source and recompile on edit, so they are left untouched — and forcing an
aiter rebuild for a Triton task would trigger a slow, pointless C++ recompile.
sglang (tvm-ffi) tasks are intentionally NOT handled yet (deferred). Best-effort
and idempotent; unknown frameworks are no-ops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from kernelforge.llm.git import git
from kernelforge.loop.aiter_cache import activate_aiter_cache_for_sources

log = logging.getLogger(__name__)

_CPP_EXTS = (".cu", ".cuh", ".hip", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp")


def force_jit_rebuild(paths: Iterable[str]) -> None:
    """Make the framework recompile the kernel from the current source.

    ``paths`` are the kernel/source files (relative or absolute); the framework
    and language are inferred from them.
    """
    try:
        source_paths = [str(path) for path in paths if path]
        strs = [path.lower() for path in source_paths]
        if not strs:
            return
        # Only C/C++ HIP kernels have the prebuilt-.so shadowing problem; forcing
        # a rebuild for a Triton (.py) task would recompile aiter's C++ for nothing.
        if not any(s.endswith(_CPP_EXTS) for s in strs):
            return
        joined = " ".join(strs)

        if "aiter" in joined:
            # A fresh source digest selects an empty private shard and therefore
            # rebuilds exactly once. Repeated correctness/bench/profile
            # subprocesses for unchanged source reuse that shard instead of
            # deleting the entire build tree via AITER_REBUILD.
            activate_aiter_cache_for_sources(source_paths)
    except Exception as exc:  # noqa: BLE001 - best-effort safety net
        log.debug("force_jit_rebuild skipped: %r", exc)


def tracked_source_changes(workspace: str | Path) -> list[str]:
    """Return existing tracked files changed from HEAD, as absolute paths."""

    root = Path(workspace).expanduser().resolve()
    try:
        result = git(
            "diff",
            "--name-only",
            "-z",
            "HEAD",
            "--",
            ".",
            cwd=root,
            check=False,
            text=False,
        )
        if result.returncode != 0:
            return []
        changed: list[str] = []
        for encoded in result.stdout.split(b"\0"):
            if not encoded:
                continue
            path = (root / encoded.decode(errors="surrogateescape")).resolve()
            if path.is_file() and str(path) not in changed:
                changed.append(str(path))
        return changed
    except Exception as exc:  # noqa: BLE001 - best-effort safety net
        log.debug("tracked source change discovery skipped: %r", exc)
        return []


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
