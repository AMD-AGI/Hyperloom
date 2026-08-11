###############################################################################
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Discover installed kernel-source trees + versions for the v2 resolver.

The resolver never trusts a stored absolute path; it searches the *actually
installed* source trees for a kernel's definition. This module answers where
those trees are and which versions they are (so a built index can be cached per
version).

Discovery has two parts:

1. **Known serving frameworks** (``vllm`` / ``sglang`` / ``aiter``) are located
   by name so their versions are reported even when they ship no native source
   (e.g. pip-installed vLLM has no ``csrc``).
2. **Auto-enumeration**: every other importable package that ships GPU kernel
   source (a ``csrc``/``kernels`` dir containing ``.cu``/``.cuh``/... files) is
   discovered automatically, so new libraries need no code change.

``$HYPERLOOM_DISCOVER_ONLY`` (csv) restricts discovery to named packages (faster,
avoids probing unrelated trees). ``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` (``name=path``
csv) pins a package to an explicit directory.
"""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "FrameworkRoot",
    "discover_frameworks",
    "fingerprint",
    "version_tag",
]

# Serving frameworks always located by name (for version + JSON hints), even
# when they ship no native source.
_KNOWN = ("vllm", "sglang", "aiter")

# Subdirectories (relative to a package dir) that hold GPU kernel source.
_CSRC_DIRS = ("csrc", "sgl-kernel/csrc", "kernels", "3rdparty/composable_kernel/include")
_NATIVE_EXTS = (".cu", ".cuh", ".hip", ".h", ".hpp")

_VERSION_RE = re.compile(r"(?:__version__|version)\s*=\s*['\"]([^'\"]+)['\"]")


@dataclass(frozen=True)
class FrameworkRoot:
    """One discovered library: its package dir, version, and native roots."""

    name: str
    root: Path
    version: str
    csrc_roots: tuple[Path, ...] = field(default_factory=tuple)


# ----------------------------------------------------------------------------
# Locating packages
# ----------------------------------------------------------------------------
def _env_source_roots() -> dict[str, Path]:
    """Parse ``$HYPERLOOM_FRAMEWORK_SOURCE_ROOTS`` (``name=path`` csv)."""
    out: dict[str, Path] = {}
    for item in os.environ.get("HYPERLOOM_FRAMEWORK_SOURCE_ROOTS", "").split(","):
        name, _, path = item.strip().partition("=")
        p = Path(path.strip())
        if name.strip() and path.strip() and p.is_dir():
            out[name.strip().lower()] = p
    return out


def _discover_only() -> set[str]:
    """Parse ``$HYPERLOOM_DISCOVER_ONLY`` into a lowercase allowlist (may be empty)."""
    raw = os.environ.get("HYPERLOOM_DISCOVER_ONLY", "")
    return {n.strip().lower() for n in raw.split(",") if n.strip()}


def _spec_root(name: str) -> Path | None:
    """Best-effort package directory via ``find_spec`` (no import)."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError, ModuleNotFoundError):
        return None
    if spec is None:
        return None
    for loc in list(getattr(spec, "submodule_search_locations", None) or []):
        if Path(loc).is_dir():
            return Path(loc)
    origin = getattr(spec, "origin", None)
    if origin and origin not in ("built-in", "namespace") and Path(origin).parent.is_dir():
        return Path(origin).parent
    return None


def _locate(name: str, env_roots: dict[str, Path]) -> Path | None:
    """Locate a known framework: env override, then ``find_spec``."""
    return env_roots.get(name) or _spec_root(name)


def _package_dirs(env_roots: dict[str, Path]) -> dict[str, Path]:
    """All candidate top-level package dirs, keyed by dir name (first wins).

    Sources: every child of a ``site-packages``/``dist-packages`` dir on
    ``sys.path``, plus explicit override paths.
    """
    dirs: dict[str, Path] = {}
    for entry in sys.path:
        base = Path(entry)
        if base.name in ("site-packages", "dist-packages") and base.is_dir():
            for child in base.iterdir():
                if child.is_dir():
                    dirs.setdefault(child.name, child)
    for name, path in env_roots.items():
        dirs.setdefault(name, path)
    return dirs


# ----------------------------------------------------------------------------
# Version + native source
# ----------------------------------------------------------------------------
def _version(name: str, root: Path | None) -> str:
    """Installed version: dist metadata first, then a ``_version.py`` literal.

    The source fallback covers packages that ship no dist metadata (e.g. aiter
    in the pip-built vLLM ROCm images) but record a version file; it avoids
    importing the package (which can require a GPU at import time).
    """
    try:
        return importlib_metadata.version(name)
    except (importlib_metadata.PackageNotFoundError, ValueError, OSError):
        pass
    if root is not None:
        for cand in ("_version.py", "version.py"):
            try:
                m = _VERSION_RE.search((root / cand).read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                continue
            if m:
                return m.group(1)
    return ""


def _has_native(directory: Path) -> bool:
    """True if ``directory`` contains any native kernel-source file."""
    for dirpath, _dirs, names in os.walk(directory):
        if any(nm.lower().endswith(_NATIVE_EXTS) for nm in names):
            return True
    return False


def _find_csrc(pkg_dir: Path) -> tuple[Path, ...]:
    """Native-source dirs under ``pkg_dir`` (and its parent) that hold kernels."""
    roots: list[Path] = []
    for base in (pkg_dir, pkg_dir.parent):
        for sub in _CSRC_DIRS:
            cand = base / sub
            if cand.is_dir() and cand not in roots and _has_native(cand):
                roots.append(cand)
    return tuple(roots)


def _canonical(pkg_name: str) -> str:
    """Map a source-package name to its framework name (``aiter_meta`` -> ``aiter``)."""
    return pkg_name[: -len("_meta")] if pkg_name.endswith("_meta") else pkg_name


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------
def discover_frameworks() -> dict[str, FrameworkRoot]:
    """Discover installed kernel-source libraries + versions.

    Known serving frameworks are located by name (so their versions are always
    reported); any other package shipping native kernel source is auto-detected.
    ``$HYPERLOOM_DISCOVER_ONLY`` restricts the result to named packages.

    Returns:
        Mapping of framework name to :class:`FrameworkRoot` for each one found.
    """
    env_roots = _env_source_roots()
    only = _discover_only()
    out: dict[str, FrameworkRoot] = {}

    # 1) Known serving frameworks (by name) -- kept even without native source.
    for name in _KNOWN:
        if only and name not in only:
            continue
        root = _locate(name, env_roots)
        if root is not None:
            out[name] = FrameworkRoot(name, root, _version(name, root), _find_csrc(root))

    # 2) Auto-enumerate any other package that ships kernel source.
    for pkg_name, pkg_dir in _package_dirs(env_roots).items():
        name = _canonical(pkg_name)
        if only and name not in only:
            continue
        csrc = _find_csrc(pkg_dir)
        if not csrc:
            continue
        if name in out:
            # Merge (e.g. aiter_meta/csrc into the aiter located by name).
            existing = out[name]
            merged = tuple(dict.fromkeys(existing.csrc_roots + csrc))
            version = existing.version or _version(name, pkg_dir)
            out[name] = FrameworkRoot(name, existing.root, version, merged)
        else:
            out[name] = FrameworkRoot(name, pkg_dir, _version(name, pkg_dir), csrc)
    return out


# ----------------------------------------------------------------------------
# Cache key + reporting
# ----------------------------------------------------------------------------
def _dir_signature(path: Path) -> str:
    """Recursive change signature for a source dir.

    Walks the whole tree (not just the immediate children) and folds in the
    newest native-file ``mtime_ns`` plus the native-file count, so an edit to a
    file in a *nested* subdirectory -- the common case, and exactly what the
    kernel agent does when it rewrites a ``.cu`` mid-tree -- changes the
    signature and invalidates a stale cached index.
    """
    try:
        count = 0
        max_mtime_ns = 0
        for dirpath, _dirs, names in os.walk(path):
            for nm in names:
                if not nm.lower().endswith(_NATIVE_EXTS):
                    continue
                try:
                    st = os.stat(os.path.join(dirpath, nm))
                except OSError:
                    continue
                count += 1
                if st.st_mtime_ns > max_mtime_ns:
                    max_mtime_ns = st.st_mtime_ns
        return f"{path}:{max_mtime_ns}:{count}"
    except OSError:
        return f"{path}:missing"


def fingerprint(frameworks: dict[str, FrameworkRoot]) -> str:
    """Stable cache key over roots + versions + recursive csrc signatures.

    Changes when a different framework/version is installed or any native source
    file under a csrc root is added, removed, or modified (including in nested
    subdirectories), so a cached index is reused iff still valid.
    """
    parts: list[str] = []
    for name in sorted(frameworks):
        fr = frameworks[name]
        parts.append(f"{name}={fr.version}@{fr.root}")
        parts += [_dir_signature(cr) for cr in fr.csrc_roots]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]  # nosec B324 - cache key, not security.


def version_tag(frameworks: dict[str, FrameworkRoot]) -> str:
    """Human-readable version tag for logs/reports (e.g. ``aiter0.1.16_vllm0.26.0``)."""
    return "_".join(f"{n}{frameworks[n].version or 'unknown'}" for n in sorted(frameworks)) or "none"
