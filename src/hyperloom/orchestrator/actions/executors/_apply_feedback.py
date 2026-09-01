# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structured apply-failure feedback for patch reauthoring.

Public surface:

* :class:`ApplyFeedback`            — structured patch-apply failure record.
* :func:`read_patch_source_context` — parse a unified diff and extract a line
  window near the first failing hunk; used by the patch-apply path.
* :func:`source_context_for_file`   — shared file-resolve + window primitive;
  used by the enablement source-context path.
* :func:`build_apply_feedback`      — convenience factory from raw error info.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ApplyFeedback:
    """Structured feedback from a failed patch apply attempt.

    Attached to every ``apply_failed`` executor result under the key
    ``"retry_feedback"`` (a list of :class:`ApplyFeedback`, one per patch).

    Attributes:
        patch: Absolute path string of the patch file that failed.
        channel: ``"git"`` or ``"nogit"`` — which apply channel was used.
        tried_levels: The ``-p`` strip levels that were tried (e.g. ``[0, 1, 2]``).
        stderr: Combined stderr from all tried apply attempts (newline-separated).
        rejected_hunks: Text of ``.rej`` reject files produced during apply
            (empty string when none were collected).
        source_context: A formatted source-code snippet from the target file
            near the first failing hunk (empty string when unavailable).
    """

    patch: str
    channel: str
    tried_levels: list[int] = field(default_factory=list)
    stderr: str = ""
    rejected_hunks: str = ""
    source_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for embedding in result payloads."""
        return {
            "patch": self.patch,
            "channel": self.channel,
            "tried_levels": self.tried_levels,
            "stderr": self.stderr,
            "rejected_hunks": self.rejected_hunks,
            "source_context": self.source_context,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ApplyFeedback":
        """Deserialize from a previously serialized dict."""
        return cls(
            patch=str(d.get("patch") or ""),
            channel=str(d.get("channel") or "nogit"),
            tried_levels=list(d.get("tried_levels") or []),
            stderr=str(d.get("stderr") or ""),
            rejected_hunks=str(d.get("rejected_hunks") or ""),
            source_context=str(d.get("source_context") or ""),
        )

    def format_for_mandate(self) -> str:
        """Return a human-readable block suitable for inclusion in a patch-author mandate."""
        parts: list[str] = [f"## Apply failure: {Path(self.patch).name}"]
        parts.append(f"Channel: {self.channel}")
        if self.tried_levels:
            parts.append(f"Tried strip levels: {self.tried_levels}")
        if self.stderr:
            parts.append(f"\n### stderr\n```\n{self.stderr.strip()}\n```")
        if self.rejected_hunks:
            parts.append(f"\n### Rejected hunks (.rej)\n```diff\n{self.rejected_hunks.strip()}\n```")
        if self.source_context:
            parts.append(f"\n### Source context\n```\n{self.source_context.strip()}\n```")
        return "\n".join(parts)


def read_patch_source_context(
    patch_text: str,
    framework_root: Path,
    *,
    radius: int = 25,
) -> str:
    """Extract a source-code window near the first failing hunk in a patch.

    Parses the first ``---`` target file and first ``@@`` hunk line from
    ``patch_text``, resolves the file against ``framework_root`` (trying the
    raw path as-is first, then stripping one leading path component as ``-p1``
    does), and returns ``radius`` lines centred on the hunk start line.

    Fully exception-guarded: any failure returns ``""`` so callers degrade
    gracefully when source context is unavailable.

    Args:
        patch_text: The unified-diff text to parse.
        framework_root: The source-tree root to resolve target files against.
        radius: Total number of lines to include in the snippet window.

    Returns:
        A formatted ``# file (lines N-M)`` header + line-numbered snippet,
        or ``""`` when unavailable.
    """
    try:
        return _read_source_context_impl(patch_text, framework_root, radius=radius)
    except Exception:  # noqa: BLE001 — best-effort
        log.debug("apply_feedback: source-context extraction failed", exc_info=True)
        return ""


def _read_source_context_impl(
    patch_text: str,
    framework_root: Path,
    *,
    radius: int,
) -> str:
    """Implementation of :func:`read_patch_source_context` (may raise)."""
    import re

    lines = patch_text.splitlines()

    # Find the first target file, preferring the +++ (new) side.
    target_raw: str | None = None
    hunk_start: int = 0

    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            plus = lines[i + 1][4:].strip().split("\t")[0]
            if plus and plus != "/dev/null":
                target_raw = plus
            else:
                # Deletion patch: use the --- side.
                minus = ln[4:].strip().split("\t")[0]
                if minus and minus != "/dev/null":
                    target_raw = minus
            i += 2
            continue
        if target_raw and ln.startswith("@@ "):
            # Parse the new-side start line from @@ -L,N +L2,N2 @@.
            m = re.search(r"\+(\d+)", ln)
            if m:
                hunk_start = max(0, int(m.group(1)) - 1)  # 0-indexed
            break
        i += 1

    if not target_raw:
        return ""

    target_path = _resolve_patch_target(target_raw, framework_root)
    if target_path is None:
        return ""

    file_lines = target_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not file_lines:
        return ""

    half = max(1, radius // 2)
    start = max(0, hunk_start - half)
    end = min(len(file_lines), start + radius)
    snippet = "\n".join(f"{n + 1:>5}| {file_lines[n]}" for n in range(start, end))
    return f"# {target_path} (lines {start + 1}-{end})\n{snippet}"


def _resolve_patch_target(target_raw: str, framework_root: Path) -> Path | None:
    """Resolve a raw patch header path to an existing file.

    Tries the path directly (possibly absolute), then strips one leading
    component (mimicking ``-p1``), then two (``-p2``), falling back to
    ``framework_root``-relative.

    Args:
        target_raw: The raw path from the ``+++`` or ``---`` header.
        framework_root: Root to resolve relative paths against.

    Returns:
        An existing :class:`~pathlib.Path`, or ``None`` when unresolvable.
    """
    candidates: list[Path] = []
    raw = Path(target_raw)

    if raw.is_absolute():
        candidates.append(raw)

    # Try stripping leading path components: -p0, -p1, -p2.
    parts = raw.parts
    # Remove leading git "a/"/"b/" prefixes.
    if parts and parts[0] in ("a", "b"):
        parts = parts[1:]
    for strip in range(min(3, len(parts))):
        rel = Path(*parts[strip:]) if len(parts) > strip else Path(parts[-1])
        candidates.append(framework_root / rel)

    return next((c for c in candidates if c.is_file()), None)


def source_context_for_file(
    filepath: str,
    *,
    symbol: str = "",
    window: int = 12,
    search_roots: "list[Path] | None" = None,
) -> str:
    """Extract a source window centred on the first occurrence of *symbol*.

    This is the shared "file resolve + window" primitive underlying both
    :func:`read_patch_source_context` and
    :meth:`~enablement.params.EnablementParams._read_enablement_source_context`.

    Args:
        filepath: Absolute or relative path to the target file.
        symbol: Optional string to search for within the file; when found the
            window is centred on the first matching line.
        window: Total number of lines to include in the snippet.
        search_roots: Additional directories to search when *filepath* is
            relative (tried in order after ``filepath`` itself).

    Returns:
        A formatted ``# file (lines N-M)`` header + line-numbered snippet,
        or ``""`` when unavailable.
    """
    try:
        return _source_context_for_file_impl(filepath, symbol=symbol, window=window, search_roots=search_roots)
    except Exception:  # noqa: BLE001 — grounding is best-effort
        log.debug("apply_feedback: source-context-for-file failed for %s", filepath, exc_info=True)
        return ""


def _source_context_for_file_impl(
    filepath: str,
    *,
    symbol: str,
    window: int,
    search_roots: "list[Path] | None",
) -> str:
    """Implementation of :func:`source_context_for_file` (may raise)."""
    offending_file = filepath.strip()
    if not offending_file:
        return ""

    candidates: list[Path] = []
    p = Path(offending_file)
    if p.is_absolute():
        candidates.append(p)
    else:
        if search_roots:
            for root in search_roots:
                candidates.append(Path(str(root)) / offending_file)
        # As-is relative to cwd, last resort.
        candidates.append(p)

    target: Path | None = next((c for c in candidates if c.is_file()), None)
    if target is None:
        return ""

    file_lines = target.read_text(errors="replace").splitlines()
    if not file_lines:
        return ""

    hit = 0
    if symbol:
        for idx, ln in enumerate(file_lines):
            if symbol in ln:
                hit = idx
                break

    half = max(1, window // 2)
    start = max(0, hit - half)
    end = min(len(file_lines), start + window)
    snippet = "\n".join(f"{n + 1:>5}| {file_lines[n]}" for n in range(start, end))
    return f"# {target} (lines {start + 1}-{end})\n{snippet}"


def build_apply_feedback(
    patch_path: "str | Path",
    *,
    channel: str,
    tried_levels: "list[int] | None" = None,
    stderr: str = "",
    rejected_hunks: str = "",
    framework_root: "Path | None" = None,
) -> ApplyFeedback:
    """Build an :class:`ApplyFeedback` record with optional source context.

    When *framework_root* is provided the patch text is parsed and a source
    context snippet is extracted automatically.

    Args:
        patch_path: Path to the patch file that failed.
        channel: ``"git"`` or ``"nogit"``.
        tried_levels: Strip levels that were attempted.
        stderr: Combined stderr from all apply attempts.
        rejected_hunks: Content of collected ``.rej`` files.
        framework_root: Source-tree root for resolving target files.

    Returns:
        A populated :class:`ApplyFeedback` instance.
    """
    patch_str = str(patch_path)
    source_ctx = ""
    if framework_root is not None:
        try:
            patch_text = Path(patch_str).read_text(encoding="utf-8", errors="replace")
            source_ctx = read_patch_source_context(patch_text, framework_root, radius=50)
        except Exception:  # noqa: BLE001
            pass

    return ApplyFeedback(
        patch=patch_str,
        channel=channel,
        tried_levels=tried_levels or [],
        stderr=stderr,
        rejected_hunks=rejected_hunks,
        source_context=source_ctx,
    )


__all__ = [
    "ApplyFeedback",
    "build_apply_feedback",
    "read_patch_source_context",
    "source_context_for_file",
]
