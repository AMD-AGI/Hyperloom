# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Code-localization logic: closure gate + diff synthesis.

Given a localization stack action (PR backport or vendored files), this module
decides whether the change is a Python-only dependency closure safe to localize
here, or a compiled / build-backend change that must defer to a targeted build.
Diff fetch/synthesis flows through injectable shims so the pure closure/synthesis
logic is CI-testable without network.

Pure-Python: no subprocess, no filesystem writes, no network here.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Callable


# A closure that touches any of these is a compiled / build-backend change and
# must defer to a targeted build, never localized+booted here.
_COMPILED_SUFFIXES: frozenset[str] = frozenset(
    {".cpp", ".cc", ".cxx", ".c", ".cu", ".cuh", ".hip", ".pyx", ".pxd", ".h", ".hpp"}
)
_BUILD_BACKEND_FILES: frozenset[str] = frozenset(
    {"CMakeLists.txt", "setup.py", "pyproject.toml", "setup.cfg", "meson.build", "Makefile"}
)

# Localization caps (safety + observability).
_MAX_VENDOR_FILES = 64
_MAX_VENDOR_BYTES = 5_000_000

PYTHON_ONLY = "python_only"
NEEDS_RUNG5 = "needs_rung5"
EMPTY = "empty"


@dataclass(frozen=True)
class ClosureVerdict:
    """Outcome of classifying a localization closure.

    Attributes:
        kind: ``python_only`` / ``needs_rung5`` / ``empty``.
        reason: Human-readable justification.
        compiled_paths: The compiled / build-backend paths that forced
            ``needs_rung5`` (empty otherwise).
    """

    kind: str
    reason: str = ""
    compiled_paths: tuple[str, ...] = ()

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
    """Classify a set of touched paths as localizable or targeted-build-deferred.

    Args:
        paths: Repo-relative paths the closure touches.

    Returns:
        ClosureVerdict: ``empty`` when no paths; ``needs_rung5`` when any path
        is a compiled source / build-backend file; ``python_only`` otherwise.
    """
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
            compiled_paths=compiled,
        )
    if len(clean) > _MAX_VENDOR_FILES:
        return ClosureVerdict(
            kind=NEEDS_RUNG5,
            reason=f"closure of {len(clean)} files exceeds cap {_MAX_VENDOR_FILES}",
        )
    return ClosureVerdict(kind=PYTHON_ONLY, reason=f"{len(clean)} python-only file(s)")


def parse_diff_paths(diff_text: str) -> list[str]:
    """Return the repo-relative paths a unified diff touches (first-seen order)."""
    from hyperloom.agents.framework._audit_common import parse_unified_diff

    return [c.path for c in parse_unified_diff(diff_text or "") if c.path]


def synthesize_vendor_diff(files: list[tuple[str, str, str]]) -> str:
    """Build a ``git apply``-ready unified diff from raw file contents.

    Args:
        files: ``(rel_path, old_text, new_text)`` triples. An empty ``old_text``
            with non-empty ``new_text`` is an add; a non-empty ``old_text`` with
            empty ``new_text`` is a delete.

    Returns:
        str: Concatenated unified-diff text (empty when nothing usable).
    """
    out: list[str] = []
    for rel, old, new in files or []:
        rel_s = str(rel or "").strip().lstrip("/")
        if not rel_s:
            continue
        old_s = old or ""
        new_s = new or ""
        if old_s == new_s:
            continue
        old_lines = old_s.splitlines(keepends=True)
        new_lines = new_s.splitlines(keepends=True)
        is_add = not old_s
        is_del = not new_s
        from_label = "/dev/null" if is_add else f"a/{rel_s}"
        to_label = "/dev/null" if is_del else f"b/{rel_s}"
        body = difflib.unified_diff(old_lines, new_lines, fromfile=from_label, tofile=to_label, lineterm="\n")
        hunk = "".join(body)
        if not hunk:
            continue
        out.append(f"diff --git a/{rel_s} b/{rel_s}")
        if is_add:
            out.append("new file mode 100644")
        elif is_del:
            out.append("deleted file mode 100644")
        out.append(hunk.rstrip("\n"))
    if not out:
        return ""
    return "\n".join(out) + "\n"


def _closure_bytes(diff_text: str) -> int:
    """Byte size of the localization payload (added content proxy)."""
    return len((diff_text or "").encode("utf-8", "replace"))


def build_localization_diff(
    action: object,
    *,
    fetch_pr_patches: Callable[[str, int], str],
    fetch_raw_file: Callable[[str, str, str], str],
    framework_root: object = None,
) -> tuple[str, list[str], ClosureVerdict]:
    """Fetch/synthesize the localization diff and classify its closure.

    Dispatches on ``action.kind``: ``pr_backport`` fetches the merged PR's diff;
    ``vendor_files`` fetches each ``localized_paths`` entry at ``action.ref`` and
    synthesizes an add/replace diff. Fetch flows through injectable shims.

    Args:
        action: An ``EnablementStackAction`` (duck-typed: ``kind`` / ``repo_url``
            / ``ref`` / ``pr_number`` / ``localized_paths``).
        fetch_pr_patches: ``(repo_slug, pr_number) -> unified_diff`` shim.
        fetch_raw_file: ``(repo_slug, ref, path) -> file_contents`` shim.
        framework_root: Unused here (reserved for old-content diffs); kept for
            call-site symmetry.

    Returns:
        tuple[str, list[str], ClosureVerdict]: ``(diff_text, touched_paths, verdict)``.
        ``diff_text`` is ``""`` on fetch failure (verdict then ``empty``).
    """
    kind = str(getattr(action, "kind", "") or "")
    repo_url = str(getattr(action, "repo_url", "") or "")
    slug = _repo_slug_safe(repo_url)

    if kind == "pr_backport":
        pr_number = int(getattr(action, "pr_number", 0) or 0)
        if not slug or pr_number <= 0:
            return "", [], ClosureVerdict(kind=EMPTY, reason="pr_backport missing repo_url/pr_number")
        diff_text = fetch_pr_patches(slug, pr_number) or ""
        if not diff_text.strip():
            return "", [], ClosureVerdict(kind=EMPTY, reason="empty PR diff (fetch failed?)")
        paths = parse_diff_paths(diff_text)
        verdict = classify_closure(paths)
        if verdict.is_localizable and _closure_bytes(diff_text) > _MAX_VENDOR_BYTES:
            verdict = ClosureVerdict(kind=NEEDS_RUNG5, reason="PR diff exceeds byte cap")
        return diff_text, paths, verdict

    if kind == "vendor_files":
        ref = str(getattr(action, "ref", "") or "")
        rels = [str(p) for p in (getattr(action, "localized_paths", ()) or [])]
        if not slug or not ref or not rels:
            return "", [], ClosureVerdict(kind=EMPTY, reason="vendor_files missing repo_url/ref/paths")
        verdict = classify_closure(rels)
        if not verdict.is_localizable:
            return "", rels, verdict
        triples: list[tuple[str, str, str]] = []
        for rel in rels:
            new_text = fetch_raw_file(slug, ref, rel) or ""
            if not new_text:
                return "", rels, ClosureVerdict(kind=EMPTY, reason=f"raw fetch failed for {rel}")
            triples.append((rel, "", new_text))
        diff_text = synthesize_vendor_diff(triples)
        if _closure_bytes(diff_text) > _MAX_VENDOR_BYTES:
            return "", rels, ClosureVerdict(kind=NEEDS_RUNG5, reason="vendored bytes exceed cap")
        return diff_text, rels, verdict

    return "", [], ClosureVerdict(kind=EMPTY, reason=f"unsupported localization kind {kind!r}")


def _repo_slug_safe(repo_url: str) -> str:
    """Best-effort ``owner/name`` slug from a git URL; ``""`` when unparseable."""
    try:
        from hyperloom.agents.framework.sources._shared import _repo_slug

        return _repo_slug(repo_url)
    except Exception:  # noqa: BLE001
        return ""


__all__ = [
    "ClosureVerdict",
    "EMPTY",
    "NEEDS_RUNG5",
    "PYTHON_ONLY",
    "build_localization_diff",
    "classify_closure",
    "parse_diff_paths",
    "synthesize_vendor_diff",
]
