###############################################################################
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Build (and cache) a kernel-name -> source-file index for the v2 resolver.

The index answers "which native source file (and line) defines kernel
``<base_name>`` in the *currently installed* tree?". It is built once per
container by scanning the discovered ``csrc`` dirs for ``__global__`` kernel
definitions, and cached keyed by a version fingerprint so later runs are ~free.

``symbol_index`` maps a base kernel name (the demangled ``__global__`` identifier)
to the list of ``{file, line, framework}`` records that define it. Finding the kernel
wherever the installed version put it is what makes moves/renames self-healing.

Triton/Python launchers (``.py``) are intentionally NOT indexed here; the
resolver resolves those lazily via ``ast`` (vLLM ships thousands of ``.py``).
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:  # package import (TraceLens route / tests)
    from . import source_env
except ImportError:  # flat top-level import (bypass route puts tools/ on sys.path)
    import source_env  # type: ignore[no-redef]

# ``FrameworkRoot`` is referenced via the module (``source_env.FrameworkRoot``) to
# keep a single import style for ``source_env`` across both branches above.

_NATIVE_EXTS = (".cu", ".cuh", ".hip", ".h", ".hpp")

# --- kernel-definition scanning ---------------------------------------------
# A definition head is ``__global__`` <attrs / return type> NAME ( params ).
# The tricky part is attributes that carry their own parentheses -- notably
# ``__launch_bounds__(NUM_THREADS)`` (on ~40% of aiter kernels) and
# ``__attribute__((...))``. A naive ``__global__[^()]*?NAME(`` regex stops at the
# attribute's ``(`` and captures the *attribute* as the kernel name. So we scan
# token by token from ``__global__``, skip any attribute call (balanced parens),
# and take the first remaining identifier that is directly followed by ``(``.
_GLOBAL_TOKEN_RE = re.compile(r"\b__global__\b")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_ATTR_KEYWORDS = frozenset(
    {"__launch_bounds__", "launch_bounds", "__attribute__", "__maxnreg__",
     "__cluster_dims__", "__grid_constant__"}
)


def _skip_balanced_parens(text: str, open_pos: int) -> int:
    """Return the index just past the ``)`` matching the ``(`` at ``open_pos``."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def _iter_global_defs(text: str):
    """Yield ``(name, name_pos)`` for each ``__global__`` kernel definition."""
    n = len(text)
    for gm in _GLOBAL_TOKEN_RE.finditer(text):
        pos = gm.end()
        while pos < n:
            if text[pos].isspace():
                pos += 1
                continue
            if text[pos] in ";{}":  # not a definition head we understand
                break
            m = _IDENT_RE.match(text, pos)
            if not m:  # punctuation (``*``, ``&``, ``<``, ``::`` ...)
                pos += 1
                continue
            ident, pos = m.group(0), m.end()
            after = pos
            while after < n and text[after].isspace():
                after += 1
            if after < n and text[after] == "(":
                if ident in _ATTR_KEYWORDS:
                    pos = _skip_balanced_parens(text, after)
                    continue
                yield ident, m.start()
                break
            # else: a qualifier / return-type token -- keep scanning.


def _scan_file(path: Path) -> list[tuple[str, int]]:
    """Return ``(base_name, def_line)`` for each kernel defined in ``path``."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    if "__global__" not in text:
        return []
    return [(name, text.count("\n", 0, pos) + 1) for name, pos in _iter_global_defs(text)]


def _native_files(csrc_roots: tuple[Path, ...]):
    """Yield every native source file under the given ``csrc`` roots."""
    for root in csrc_roots:
        if not root.is_dir():
            continue
        for dirpath, _dirs, names in os.walk(root):
            for nm in names:
                if nm.lower().endswith(_NATIVE_EXTS):
                    yield Path(dirpath) / nm


# --- index ------------------------------------------------------------------
@dataclass
class SourceIndex:
    """Cached kernel-name -> source records index, plus build metadata."""

    fingerprint: str
    version_tag: str
    symbol_index: dict[str, list[dict[str, object]]] = field(default_factory=dict)
    build_ms: float = 0.0
    file_count: int = 0
    symbol_count: int = 0

    def lookup(self, base_name: str) -> list[dict[str, object]]:
        """Return all definition records for a base kernel name (``[]`` if none)."""
        return self.symbol_index.get(base_name, [])


def build_index(frameworks: dict[str, source_env.FrameworkRoot]) -> SourceIndex:
    """Scan the discovered ``csrc`` trees and build the kernel index (timed)."""
    started = time.perf_counter()
    symbol_index: dict[str, list[dict[str, object]]] = {}
    file_count = 0
    for name in sorted(frameworks):
        for path in _native_files(frameworks[name].csrc_roots):
            defs = _scan_file(path)
            if defs:
                file_count += 1
            for base, line_no in defs:
                symbol_index.setdefault(base, []).append(
                    {"file": str(path), "line": line_no, "framework": name}
                )
    return SourceIndex(
        fingerprint=source_env.fingerprint(frameworks),
        version_tag=source_env.version_tag(frameworks),
        symbol_index=symbol_index,
        build_ms=round((time.perf_counter() - started) * 1000.0, 2),
        file_count=file_count,
        symbol_count=len(symbol_index),
    )


# --- cache ------------------------------------------------------------------
def _cache_path(fingerprint: str) -> Path:
    """Cache file path (dir from ``$HYPERLOOM_KSI_CACHE_DIR`` or a temp subdir)."""
    raw = os.environ.get("HYPERLOOM_KSI_CACHE_DIR", "").strip()
    d = Path(raw) if raw else Path(tempfile.gettempdir()) / "hyperloom_ksi"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except OSError:
        # Best-effort: the on-disk index cache is an optimization, not required.
        # If the dir can't be created, _save_cache no-ops and the index rebuilds.
        pass
    return d / f"ksi_{fingerprint}.json"


def _load_cache(fingerprint: str) -> SourceIndex | None:
    try:
        with open(_cache_path(fingerprint), encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict) and data.get("fingerprint") == fingerprint:
            return SourceIndex(**data)
    except (OSError, ValueError, TypeError):
        # Any read/parse/shape error is treated as a cache miss -> rebuild upstream.
        return None
    return None


def _save_cache(index: SourceIndex) -> None:
    path = _cache_path(index.fingerprint)
    try:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(index), fh)
        tmp.replace(path)
    except OSError:
        # Best-effort cache write; a failure here must not break index build/use.
        pass


def load_or_build(frameworks: dict[str, source_env.FrameworkRoot] | None = None) -> SourceIndex:
    """Return a cached index for the current versions, or build + cache one.

    ``build_ms`` is ``0.0`` on a cache hit and the real build time on a miss.
    """
    fw = frameworks if frameworks is not None else source_env.discover_frameworks()
    cached = _load_cache(source_env.fingerprint(fw))
    if cached is not None:
        cached.build_ms = 0.0
        return cached
    index = build_index(fw)
    _save_cache(index)
    return index


def _main(argv: list[str] | None = None) -> int:
    """CLI: build the index for this container and print its stats."""
    import argparse

    parser = argparse.ArgumentParser(description="Build/verify the kernel source index.")
    parser.add_argument("--rebuild", action="store_true", help="Ignore cache and rebuild.")
    args = parser.parse_args(argv)

    fw = source_env.discover_frameworks()
    if not fw:
        print("No frameworks (vllm/sglang/aiter) discovered.")
        return 1
    print(f"Frameworks: {source_env.version_tag(fw)}")
    for name, fr in sorted(fw.items()):
        print(f"  {name} v{fr.version or '?'} @ {fr.root}")
        for cr in fr.csrc_roots:
            print(f"      csrc: {cr}")
    index = build_index(fw) if args.rebuild else load_or_build(fw)
    print(
        f"Index: {index.symbol_count} symbols across {index.file_count} files "
        f"(build_ms={index.build_ms}, fingerprint={index.fingerprint})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = ["SourceIndex", "build_index", "load_or_build"]
