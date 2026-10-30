# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Code-localization logic: closure gate + PR diff scoping."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Callable

from hyperloom.common.github_urls import repo_slug

if TYPE_CHECKING:
    from .stack_actions import EnablementStackAction


# A closure that touches any of these is a compiled / build-backend change and must defer to a targeted build, never
# localized+booted here.
_COMPILED_SUFFIXES: frozenset[str] = frozenset(
    {".cpp", ".cc", ".cxx", ".c", ".cu", ".cuh", ".hip", ".pyx", ".pxd", ".h", ".hpp"}
)
_BUILD_BACKEND_FILES: frozenset[str] = frozenset(
    {"CMakeLists.txt", "setup.py", "pyproject.toml", "setup.cfg", "meson.build", "Makefile"}
)

_MAX_VENDOR_BYTES = 5_000_000

PYTHON_ONLY = "python_only"
NEEDS_RUNG5 = "needs_rung5"
OVERSIZE = "oversize"
EMPTY = "empty"


@dataclass(frozen=True)
class ClosureVerdict:
    """Outcome of classifying a localization closure."""

    kind: str
    reason: str = ""

    @property
    def is_localizable(self) -> bool:
        """True only for a non-empty Python-only closure."""
        return self.kind == PYTHON_ONLY


def _is_compiled_or_build(path: str) -> bool:
    """True when ``path`` is a compiled source or a build-backend file."""
    p = PurePosixPath(str(path or "").strip())
    if not p.name:
        return False
    if p.name in _BUILD_BACKEND_FILES:
        return True
    return p.suffix.lower() in _COMPILED_SUFFIXES


def classify_closure(paths: list[str]) -> ClosureVerdict:
    """Classify a set of touched paths as localizable or targeted-build-deferred."""
    clean = [str(p).strip() for p in (paths or []) if str(p).strip()]
    if not clean:
        return ClosureVerdict(kind=EMPTY, reason="no files in closure")
    compiled = tuple(p for p in clean if _is_compiled_or_build(p))
    if compiled:
        return ClosureVerdict(
            kind=NEEDS_RUNG5,
            reason=(
                "closure includes compiled / build-backend file(s) requiring a "
                "targeted build (Rung 5, deferred): " + ", ".join(compiled[:8])
            ),
        )
    return ClosureVerdict(kind=PYTHON_ONLY, reason=f"{len(clean)} python-only file(s)")


def parse_diff_paths(diff_text: str) -> list[str]:
    """Return the repo-relative paths a unified diff modifies or adds (first-seen order)."""
    from hyperloom.common.unified_diff import parse_unified_diff

    return [c.path for c in parse_unified_diff(diff_text or "") if c.path and not c.is_deleted]


def _scope_diff(diff_text: str, scope: str) -> str:
    """Keep only the file sections of a git diff under ``scope``."""
    sections = re.split(r"(?m)^(?=diff --git )", diff_text)
    return "".join(
        s for s in sections if s.startswith("diff --git ") and PurePosixPath(_section_path(s)).is_relative_to(scope)
    )


def _section_path(section: str) -> str:
    """The post-image path in a ``diff --git a/<old> b/<new>`` header."""
    header = section.split("\n", 1)[0]
    return header.rsplit(" b/", 1)[-1].strip()


def _closure_bytes(diff_text: str) -> int:
    """Byte size of the localization payload (added content proxy)."""
    return len((diff_text or "").encode("utf-8", "replace"))


def _classify_pr_diff(diff_text: str, scope: str) -> tuple[str, list[str], ClosureVerdict]:
    """Classify a fetched PR diff over all its files, then keep the ones under ``scope``."""
    paths = parse_diff_paths(diff_text)
    verdict = classify_closure(paths)
    if not verdict.is_localizable:
        return diff_text, paths, verdict
    diff_text = _scope_diff(diff_text, scope)
    paths = parse_diff_paths(diff_text)
    if not paths:
        return "", [], ClosureVerdict(kind=EMPTY, reason=f"no file of the PR is under {scope!r}")
    if _closure_bytes(diff_text) > _MAX_VENDOR_BYTES:
        verdict = ClosureVerdict(kind=OVERSIZE, reason="PR diff exceeds byte cap")
    return diff_text, paths, verdict


def build_localization_diff(
    action: EnablementStackAction,
    *,
    fetch_pr_patches: Callable[[str, int], str],
    scope: str,
) -> tuple[str, list[str], ClosureVerdict]:
    """Fetch the localization diff and classify its closure.

    A compiled or build file anywhere in a PR makes it a compiled gap; of the
    rest, only the files under ``scope`` (the package the server imports) are
    backported.

    Blocking (synchronous HTTP); call via ``asyncio.to_thread``.
    """
    kind = action.kind
    slug = _repo_slug_safe(action.repo_url)

    if kind == "pr_backport":
        pr_number = action.pr_number
        if not slug or pr_number <= 0:
            return "", [], ClosureVerdict(kind=EMPTY, reason="pr_backport missing repo_url/pr_number")
        diff_text = fetch_pr_patches(slug, pr_number) or ""
        if not diff_text.strip():
            return "", [], ClosureVerdict(kind=EMPTY, reason="empty PR diff (fetch failed?)")
        return _classify_pr_diff(diff_text, scope)

    return "", [], ClosureVerdict(kind=EMPTY, reason=f"unsupported localization kind {kind!r}")


def _repo_slug_safe(repo_url: str) -> str:
    """Best-effort ``owner/name`` slug from a git URL; ``""`` when unparseable."""
    try:
        return repo_slug(repo_url)
    except ValueError:
        return ""


__all__ = [
    "ClosureVerdict",
    "EMPTY",
    "NEEDS_RUNG5",
    "OVERSIZE",
    "PYTHON_ONLY",
    "build_localization_diff",
    "classify_closure",
    "parse_diff_paths",
]
