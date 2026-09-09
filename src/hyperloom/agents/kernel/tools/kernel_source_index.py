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

import ast
import getpass
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

try:  # package import (TraceLens route / tests)
    from . import source_env
except ImportError:  # flat top-level import (tools/ on sys.path)
    import source_env  # type: ignore[no-redef]

log = logging.getLogger(__name__)

# ``FrameworkRoot`` is referenced via the module (``source_env.FrameworkRoot``) to
# keep a single import style for ``source_env`` across both branches above.

# Native source extensions to scan, and the editability filter's native set (one
# and the same): a ``__global__`` def indexed from an extension
# ``is_editable_source`` would later reject as non-editable is dead weight, so the
# scan set and the editability set are a single tuple.
_NATIVE_EXTS = (".cu", ".cuh", ".hip", ".h")


def is_editable_source(path: str | None, kernel_kind: str | None = None) -> bool:
    """Return whether ``path`` is a source we can route a kernel rewrite at.

    Editable == native device code (``.cu``/``.cuh``/``.hip``/``.h``) or a
    repo-resident Triton/TileLang ``.py``. Generated Triton is excluded
    (``triton_inductor_generated`` kind and any ``torchinductor`` / ``/tmp/``
    path).

    Args:
        path: Candidate source path (from a trace ``kernel_file`` or the finder).
        kernel_kind: Optional kernel-kind hint.

    Returns:
        ``True`` when the path is an editable source, else ``False``.
    """
    if not path:
        return False
    low = path.lower()
    if low.endswith(_NATIVE_EXTS):
        return True
    if low.endswith(".py"):
        if kernel_kind == "triton_inductor_generated":
            return False
        if "torchinductor" in path or path.startswith("/tmp/"):  # nosec B108 - marker for generated compiler artifacts.
            return False
        return True
    return False


# --- Triton .py AST pinning -------------------------------------------------
# Triton decorators marking a device-kernel def (``@triton.jit`` / ``@jit`` and
# the autotune/heuristics wrappers that sit on top of a jit'd kernel).
_TRITON_DECORATORS = frozenset({"jit", "autotune", "heuristics"})


def _is_triton_kernel_def(node: ast.AST) -> bool:
    """Return whether an AST function node carries a Triton kernel decorator."""
    for dec in getattr(node, "decorator_list", []):
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        if name in _TRITON_DECORATORS:
            return True
    return False


def _normalize_symbol(symbol: str) -> str:
    """Reduce a device kernel symbol to a bare identifier core for matching.

    Triton device symbols often wrap the ``@triton.jit`` function name with a
    leading ``triton_``/``_`` prefix and a trailing autotune/hash suffix
    (e.g. ``_fwd_kernel_0d1d2``). Strip the common decorations so a fuzzy match
    against the def name has a chance.
    """
    core = re.sub(r"[^0-9A-Za-z_].*$", "", str(symbol or "").strip())
    core = re.sub(r"_+\d[\dA-Za-z]*$", "", core)  # drop trailing autotune/hash suffix
    return core.strip("_").lower()


def triton_def_line(py_path: str, *, func: str = "", symbol: str = "", require_name_match: bool = False) -> int | None:
    """Find a Triton kernel's ``def`` line in a ``.py`` via AST (no import).

    Matching precedence: (1) exact ``func`` name; (2) a ``@triton.jit`` def whose
    name matches the normalized device ``symbol`` (exact then substring); (3) the
    sole ``@triton.jit`` def when unambiguous and ``require_name_match`` is ``False``.
    """
    try:
        tree = ast.parse(Path(py_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        return None

    jit_defs: dict[str, int] = {}
    all_defs: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            all_defs.setdefault(node.name, node.lineno)
            if _is_triton_kernel_def(node):
                jit_defs.setdefault(node.name, node.lineno)

    if func and func in all_defs:
        return all_defs[func]

    core = _normalize_symbol(symbol)
    if core:
        for name, line in jit_defs.items():
            if name.lower() == core:
                return line
        for name, line in jit_defs.items():
            low = name.lower()
            if core in low or low in core:
                return line

    if not require_name_match and len(jit_defs) == 1:
        return next(iter(jit_defs.values()))
    return None


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
    {"__launch_bounds__", "launch_bounds", "__attribute__", "__maxnreg__", "__cluster_dims__", "__grid_constant__"}
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
    """Yield ``(name, name_pos)`` for each ``__global__`` kernel *definition*.

    Only definitions (a parameter list immediately followed by a ``{`` body) are
    yielded. Forward declarations (``... );``), and ``__global__`` text living
    inside comments or string literals, are rejected so the index never points a
    rewrite at a header declaration or dead code.
    """
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
                # Definition, not a declaration: the first non-space character
                # after the matching ``)`` must open a body ``{``. A ``;`` (fwd
                # decl) or anything else (a match inside a comment/string) is
                # skipped -- this ``__global__`` yields no name.
                cursor = _skip_balanced_parens(text, after)
                while cursor < n and text[cursor].isspace():
                    cursor += 1
                if cursor < n and text[cursor] == "{":
                    yield ident, m.start()
                break
            # else: a qualifier / return-type token -- keep scanning.


def _scan_file(path: Path) -> list[tuple[str, int]]:
    """Return ``(base_name, def_line)`` for each kernel defined in ``path``."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        log.debug("kernel index: cannot read %s: %s", path, exc)
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
                symbol_index.setdefault(base, []).append({"file": str(path), "line": line_no, "framework": name})
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
    """Cache file path (dir from ``$HYPERLOOM_KSI_CACHE_DIR`` or a temp subdir).

    When falling back to the system temp root (typically a shared, world-writable
    ``/tmp`` on a multi-user host), the subdir is scoped to the current user and
    created owner-only (0o700), so users cannot collide on or shadow each other's
    cache. An explicit ``$HYPERLOOM_KSI_CACHE_DIR`` is used verbatim.
    """
    raw = os.environ.get("HYPERLOOM_KSI_CACHE_DIR", "").strip()
    if raw:
        d = Path(raw)
    else:
        try:
            uid = str(os.getuid())  # POSIX: stable per-user, no PII.
        except AttributeError:  # non-POSIX platforms
            uid = getpass.getuser() or "shared"
        d = Path(tempfile.gettempdir()) / f"hyperloom_ksi_{uid}"
    try:
        d.mkdir(parents=True, exist_ok=True)
        if not raw:
            # Best-effort: restrict the user-scoped temp cache to its owner.
            os.chmod(d, 0o700)
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
    except (OSError, ValueError, TypeError) as exc:
        # Any read/parse/shape error is treated as a cache miss -> rebuild upstream.
        log.debug("kernel index: cache read miss (%s): %s", fingerprint, exc)
        return None
    return None


def _save_cache(index: SourceIndex) -> None:
    path = _cache_path(index.fingerprint)
    try:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(asdict(index), fh)
        tmp.replace(path)
    except OSError as exc:
        # Best-effort cache write; a failure here must not break index build/use.
        log.debug("kernel index: cache write failed (%s): %s", index.fingerprint, exc)


# Process-level singleton for the no-argument (production) call. The agent path
# otherwise re-runs ``discover_frameworks()`` + ``fingerprint()`` + ``json.load``
# on every resolve; memoizing here makes "built once per container, later resolves
# ~free" true.
_PROCESS_INDEX: SourceIndex | None = None


def load_or_build(frameworks: dict[str, source_env.FrameworkRoot] | None = None) -> SourceIndex:
    """Return a cached index for the current versions, or build + cache one.

    On the no-argument production path the result is memoized in-process, so
    repeated resolves within one run do not re-discover frameworks or re-read the
    on-disk cache. ``build_ms`` is ``0.0`` on a cache hit and the real build time
    on a miss.
    """
    global _PROCESS_INDEX
    if frameworks is None and _PROCESS_INDEX is not None:
        return _PROCESS_INDEX

    fw = frameworks if frameworks is not None else source_env.discover_frameworks()
    if not fw:
        log.warning(
            "kernel index: no kernel-source frameworks discovered (vllm/sglang/aiter); "
            "native symbol resolution is disabled this run"
        )
    fingerprint = source_env.fingerprint(fw)
    cached = _load_cache(fingerprint)
    if cached is not None:
        cached.build_ms = 0.0
        if frameworks is None:
            _PROCESS_INDEX = cached
        return cached

    index = build_index(fw)
    log.info(
        "kernel index: built %d symbols across %d files (version=%s, build_ms=%.1f)",
        index.symbol_count,
        index.file_count,
        index.version_tag,
        index.build_ms,
    )
    _save_cache(index)
    if frameworks is None:
        _PROCESS_INDEX = index
    return index


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover - standalone CLI driver
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


__all__ = ["SourceIndex", "build_index", "is_editable_source", "load_or_build", "triton_def_line"]
