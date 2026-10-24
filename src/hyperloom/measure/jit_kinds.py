# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The kinds of compiled-code cache a test can use, and which files of each are ever stored (design 4.21).

A kind names the variable that moves one compiler's cache into a directory of the test's own, where inside that
directory the compiled products are, and the closed set of file names that may enter the shared cache. Everything else a
compiler leaves there (object files, generated sources, build state, tuning results) is not cached: a file that is not on
the list fails toward "not cached", never toward leaking into another test.

A kind may also say which of its files name the directory they were written in: those are stored without it and given the
reading test's own directory again (:mod:`hyperloom.measure.jit.locations`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "AITER_CPP",
    "AITER_JIT",
    "DISABLED",
    "HIT",
    "KINDS",
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


@dataclass(frozen=True)
class CacheKind:
    """One compiler's cache.

    ``env_var`` is set to ``<private root>/<name>`` for the server; ``content`` is the directory below that where the
    compiled products live, and ``allowed`` the pattern a path below ``content`` must fully match to be stored.
    ``keyed_by`` says which key the kind is stored under (``aiter`` or ``triton``, see
    :mod:`hyperloom.measure.jit.key`). ``located`` is the pattern of the admitted files whose content holds paths below
    the directory that wrote them (a compiler's index of the files it made); ``None`` when no file of the kind does.
    """

    name: str
    env_var: str
    content: str
    allowed: re.Pattern[str]
    keyed_by: str
    located: re.Pattern[str] | None = None

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

KINDS = (AITER_JIT, AITER_CPP, TRITON)


def kind_named(name: str) -> CacheKind:
    """The kind called ``name``; a :class:`KeyError` for one this version does not know."""
    for kind in KINDS:
        if kind.name == name:
            return kind
    raise KeyError(name)
