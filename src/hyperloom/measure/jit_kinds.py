# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The kinds of compiled-code cache a test can use, and which files of each are ever stored.

A kind names the variable that moves one compiler's cache into a directory of the test's own, where inside that
directory the compiled products are, and the closed set of file names that may enter the shared cache. Everything else a
compiler leaves there (object files, generated sources, build state, tuning results) is not cached: a file that is not on
the list fails toward "not cached", never toward leaking into another test.

A kind may also say which of its files name the directory they were written in: those are stored without it and given the
reading test's own directory again (:mod:`hyperloom.measure.jit.locations`).

The cache holds one kind that is not compiled code: the AgentX client's tokenized dataset (:data:`AIPERF_DATASET`). It is
stored, verified, seeded and published the way the compiled kinds are; only the process that reads it is the client, not
the server.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "AIPERF_DATASET",
    "AITER_CPP",
    "AITER_JIT",
    "DISABLED",
    "HIT",
    "KINDS",
    "MAX_SNAPSHOT_BYTES",
    "MISS",
    "STATES",
    "TRITON",
    "VERIFY_FAILED",
    "CacheKind",
    "kind_named",
]

_NAME = r"[A-Za-z0-9_][A-Za-z0-9_.-]*"
#: What a test found in the cache of a kind: a verified snapshot, none, one that failed verification, or no use of it.
HIT, MISS, VERIFY_FAILED, DISABLED = "hit", "miss", "verify_failed", "disabled"
STATES = (HIT, MISS, VERIFY_FAILED, DISABLED)
#: A kind's snapshot (everything it holds, not only what a test added) over this size is not stored, unless the kind says
#: otherwise (``CacheKind.max_bytes``).
MAX_SNAPSHOT_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class CacheKind:
    """One compiler's cache.

    ``env_var`` is set to ``<private root>/<name>`` for the server; ``content`` is the directory below that where the
    compiled products live, and ``allowed`` the pattern a path below ``content`` must fully match to be stored.
    ``keyed_by`` says which key the kind is stored under (``aiter`` or ``triton``, see
    :mod:`hyperloom.measure.jit.key`). ``located`` is the pattern of the admitted files whose content holds paths below
    the directory that wrote them (a compiler's index of the files it made); ``None`` when no file of the kind does.
    ``max_bytes`` is the most a snapshot of the kind may hold.
    """

    name: str
    env_var: str
    content: str
    allowed: re.Pattern[str]
    keyed_by: str
    located: re.Pattern[str] | None = None
    max_bytes: int = MAX_SNAPSHOT_BYTES

    def admits(self, relative: str) -> bool:
        return self.allowed.fullmatch(relative) is not None

    def locates(self, relative: str) -> bool:
        """Whether the file holds paths of the directory it was written in."""
        return self.located is not None and self.located.fullmatch(relative) is not None


#: aiter's ``compile_ops`` modules: ``<md_name>.so`` at the top of ``AITER_JIT_DIR`` and nothing below it.
AITER_JIT = CacheKind("aiter_jit", "AITER_JIT_DIR", "", re.compile(rf"{_NAME}\.so"), "aiter")
#: aiter's cpp_itfs libraries: ``AITER_ROOT_DIR/build/<function>/lib.so`` is all ``run_lib`` loads.
AITER_CPP = CacheKind("aiter_cpp", "AITER_ROOT_DIR", "build", re.compile(rf"{_NAME}/lib\.so"), "aiter")
#: Triton's compiled kernels: ``<hash>/<kernel>.<product>``; autotuning records are a choice, not a product. A kernel's
#: ``__grp__<kernel>.json`` lists its files by absolute path and Triton ignores a path that does not exist, so a group
#: written in another test's directory is a miss; it is the one file of the kind that names its own directory.
TRITON = CacheKind(
    "triton",
    "TRITON_CACHE_DIR",
    "",
    re.compile(
        r"[A-Za-z0-9_-]{16,128}/(?!.*(?:autotun|best_config))[A-Za-z0-9_.+-]+\.(?:json|hsaco|ttir|ttgir|llir|amdgcn|source)"
    ),
    "triton",
    located=re.compile(r"[^/]+/__grp__[^/]+"),
)

#: aiperf's content-addressed dataset cache (``aiperf/dataset/mmap_cache.py`` of the AgentX client):
#: ``<cache key>/dataset.dat``, ``index.dat`` (``.dat.zst`` when compressed) and the ``manifest.json`` that makes an entry
#: complete, one directory per 32-hex key. Its lock files and half-written ``.<key>.tmp.<pid>`` directories are not
#: stored. One entry of a corpus is about 6.5 GB.
AIPERF_DATASET = CacheKind(
    "aiperf_dataset",
    "AIPERF_DATASET_MMAP_CACHE_DIR",
    "",
    re.compile(r"[0-9a-f]{32}/(?:(?:dataset|index)\.dat(?:\.zst)?|manifest\.json)"),
    "client",
    max_bytes=32 * 1024**3,
)

KINDS = (AITER_JIT, AITER_CPP, TRITON, AIPERF_DATASET)


def kind_named(name: str) -> CacheKind:
    """The kind called ``name``; a :class:`KeyError` for one this version does not know."""
    for kind in KINDS:
        if kind.name == name:
            return kind
    raise KeyError(name)
