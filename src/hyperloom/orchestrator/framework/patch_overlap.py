# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How much of a patch a session has already measured, read off the diffs rather than off what they were named."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from hashlib import blake2b
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: Lines kept per patch. A port of one upstream change is a few dozen; the cap bounds what a runaway diff can
#: write into the session state without changing the verdict on anything human-sized.
_MAX_LINES: int = 400

#: Patches kept in the measured ledger, newest last.
MAX_MEASURED: int = 64


def _line_key(body: str) -> str:
    """Short stable key for one changed line body."""
    return blake2b(body.encode("utf-8", "replace"), digest_size=6).hexdigest()


def changed_line_keys(diff_texts: Iterable[str]) -> list[str]:
    """Keys for the lines a unified diff adds or removes, whitespace-normalised and de-duplicated.

    Two specialists porting one upstream change produce diffs that differ in context, ordering and file paths but
    repeat most of the same added lines, which is what makes the line bodies a better identity than the patch name:
    one GLM-5.3 session measured the same upstream PR seven times under five different names.
    """
    keys: dict[str, None] = {}
    for text in diff_texts:
        for raw in str(text or "").splitlines():
            if raw.startswith(("+++", "---", "@@", "diff ", "index ")):
                continue
            if raw[:1] not in ("+", "-"):
                continue
            body = raw[1:].strip()
            if not body:
                continue
            keys.setdefault(_line_key(body), None)
            if len(keys) >= _MAX_LINES:
                return list(keys)
    return list(keys)


def touched_files(diff_texts: Iterable[str]) -> list[str]:
    """Paths a unified diff writes to, in first-seen order."""
    files: dict[str, None] = {}
    for text in diff_texts:
        for raw in str(text or "").splitlines():
            if not raw.startswith("+++ "):
                continue
            path = raw[4:].strip()
            if path and path != "/dev/null":
                files.setdefault(path.removeprefix("b/"), None)
    return list(files)


def overlap_ratio(candidate: Iterable[str], measured: Iterable[str]) -> float:
    """Jaccard overlap of two changed-line key sets; 0.0 when either is empty."""
    left, right = set(candidate), set(measured)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def overlapping_measured(
    line_keys: Iterable[str],
    measured: Iterable[dict[str, Any]] | None,
    *,
    min_ratio: float,
) -> list[dict[str, Any]]:
    """Measured entries whose changed lines overlap *line_keys* by at least *min_ratio*, strongest first.

    Reported, never enforced: a follow-up to an already-measured patch legitimately repeats much of it, and only
    the review that reads both can tell that apart from the same work arriving twice.
    """
    keys = list(line_keys)
    if not keys:
        return []
    hits: list[dict[str, Any]] = []
    for entry in measured or []:
        if not isinstance(entry, dict):
            continue
        ratio = overlap_ratio(keys, entry.get("line_keys") or [])
        if ratio >= min_ratio:
            hits.append({**{k: v for k, v in entry.items() if k != "line_keys"}, "overlap_ratio": round(ratio, 3)})
    hits.sort(key=lambda hit: hit["overlap_ratio"], reverse=True)
    return hits


def patch_set_identity(patch_paths: Sequence[Any] | None) -> tuple[list[str], list[str]]:
    """Read a patch set off disk and return ``(touched files, changed-line keys)``; unreadable paths are skipped."""
    texts: list[str] = []
    for raw in patch_paths or []:
        path = str(raw or "").strip()
        if not path:
            continue
        try:
            texts.append(Path(path).read_text(encoding="utf-8", errors="replace"))
        except OSError as exc:
            log.debug("patch_overlap: %s unreadable, not fingerprinted: %s", path, exc)
    if not texts:
        return [], []
    return touched_files(texts), changed_line_keys(texts)


__all__ = [
    "MAX_MEASURED",
    "changed_line_keys",
    "overlap_ratio",
    "overlapping_measured",
    "patch_set_identity",
    "touched_files",
]
