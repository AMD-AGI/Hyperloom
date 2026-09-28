# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether a KB row's ROCm/AITER build can back a config replay on this pod.

``framework_version`` is part of the ``canonical_id``, so a row never reaches a
pod on the wrong framework release. ROCm and AITER cannot join the identity (a
patch bump or a new commit would orphan every row), so they are compared here,
at replay time, as a soft signal: a proven mismatch stops the row being a config
source, and ``unknown`` on either side is no claim at all.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from packaging.version import InvalidVersion, Version

__all__ = ["StackComparison", "compare_stacks", "config_stack_fingerprint", "row_stack_fingerprint"]

_UNKNOWN = {"", "unknown", "none", "n/a"}
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_RELEASE_RE = re.compile(r"^v?(\d+(?:\.\d+)*)")


class StackComparison:
    """Proven mismatches (these stop a replay) and incomparable pairs (disclosed only)."""

    def __init__(self, conflicts: list[str], notes: list[str]) -> None:
        self.conflicts = conflicts
        self.notes = notes

    @property
    def blocks_replay(self) -> bool:
        return bool(self.conflicts)

    def to_dict(self) -> dict[str, list[str]]:
        return {"conflicts": list(self.conflicts), "notes": list(self.notes)}


def row_stack_fingerprint(row: Mapping[str, Any] | None) -> dict[str, str]:
    """The ``rocm``/``aiter`` a row was recorded on, in the live fingerprint's key names."""
    stored = row.get("stack_fingerprint") if isinstance(row, Mapping) else None
    stored = stored if isinstance(stored, Mapping) else {}
    return {
        "rocm": str(stored.get("rocm_version") or ""),
        "aiter": str(stored.get("aiter_commit") or ""),
    }


def config_stack_fingerprint(stored: Mapping[str, Any] | None, session: Mapping[str, Any] | None) -> dict[str, str]:
    """The fingerprint to store beside a newly written ``best_config``: this session's ROCm/AITER.

    An unknown session value clears the stored one rather than keeping it, because the old value names the build an
    earlier config was measured on, not this one. Other components are kept as stored.
    """
    out = (
        {str(k): str(v) for k, v in (stored.get("stack_fingerprint") or {}).items()}
        if isinstance(stored, Mapping)
        else {}
    )
    session = session if isinstance(session, Mapping) else {}
    out["rocm_version"] = _known(session.get("rocm"))
    out["aiter_commit"] = _known(session.get("aiter"))
    return out


def _known(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in _UNKNOWN else text


def _release(value: str) -> str:
    """The numeric release of a ROCm version string (``6.4.1-120`` -> ``6.4.1``), or ``""``."""
    match = _RELEASE_RE.match(value.strip().lower())
    return match.group(1) if match else ""


def _rocm(recorded: str, live: str) -> tuple[str, str]:
    """``(conflict, note)`` for ROCm, by the framework-version rule: same release line, recorded not newer."""
    from .recipe_kb_t0 import _framework_version_is_compatible

    rec_release, live_release = _release(recorded), _release(live)
    if not (rec_release and live_release):
        return "", f"rocm {recorded!r} recorded, pod {live!r}: not comparable"
    if _framework_version_is_compatible(live_release, rec_release):
        return "", ""
    return f"rocm {recorded} recorded, pod runs {live}", ""


def _aiter_form(value: str) -> tuple[str, Any]:
    """``("sha", str)``, ``("version", Version)`` or ``("ref", str)``."""
    lowered = value.lower()
    if _SHA_RE.match(lowered):
        return "sha", lowered
    try:
        return "version", Version(value)
    except InvalidVersion:
        return "ref", value


def _aiter(recorded: str, live: str) -> tuple[str, str]:
    """``(conflict, note)`` for AITER, whose recorded form is a commit, a version/tag, or neither."""
    rec_form, rec_value = _aiter_form(recorded)
    live_form, live_value = _aiter_form(live)
    if rec_form != live_form or rec_form == "ref":
        if rec_form == live_form and rec_value == live_value:
            return "", ""
        return "", f"aiter {recorded!r} recorded, pod {live!r}: not comparable"
    if rec_form == "sha":
        if rec_value.startswith(live_value) or live_value.startswith(rec_value):
            return "", ""
        return f"aiter commit {recorded} recorded, pod runs {live}", ""
    if rec_value == live_value:
        # A distribution version is coarser evidence than a commit; equal is the most it can say.
        return "", ""
    return f"aiter {recorded} recorded, pod runs {live}", ""


def compare_stacks(recorded: Mapping[str, Any], live: Mapping[str, Any]) -> StackComparison:
    """Compare a row's recorded ``rocm``/``aiter`` with the pod's.

    Args:
        recorded: The row's fingerprint, as :func:`row_stack_fingerprint` returns it.
        live: The pod's fingerprint (``detect_stack_fingerprint`` keys).
    """
    conflicts: list[str] = []
    notes: list[str] = []
    for component, compare in (("rocm", _rocm), ("aiter", _aiter)):
        rec, cur = _known(recorded.get(component)), _known(live.get(component))
        if not (rec and cur):
            continue
        conflict, note = compare(rec, cur)
        if conflict:
            conflicts.append(conflict)
        if note:
            notes.append(note)
    return StackComparison(conflicts, notes)
