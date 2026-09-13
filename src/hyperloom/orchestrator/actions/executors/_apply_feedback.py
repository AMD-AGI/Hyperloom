# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Structured apply-failure feedback for patch reauthoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class ApplyFeedback:
    """Structured feedback from a failed patch apply attempt."""

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
    """Extract a source-code window near the first failing hunk in a patch."""
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
    """Resolve a raw patch header path to an existing file."""
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
    """Extract a source window centred on the first occurrence of *symbol*."""
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
    """Build an :class:`ApplyFeedback` record with optional source context."""
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
