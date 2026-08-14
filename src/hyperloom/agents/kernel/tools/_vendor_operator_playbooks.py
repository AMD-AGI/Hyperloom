###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Vendor-operator-playbook registry: route a closed-source hot kernel to a
validated KernelForge *task bundle* instead of a source rewrite.

Most of the forge-submission pipeline assumes a hot kernel has an editable
device source file (``kernel_url`` -> in-place rewrite). Some vendor
operators -- mori's EP dispatch/combine all-to-all is the first case -- are
pip-installed compiled libraries with no such source, but do have a small,
named set of launch-config knobs that a KernelForge forge-loop task bundle
has already been validated to tune (see KernelForge PR #88's "Making this
real" section for the design rationale this module implements).

This is deliberately a narrow, explicit carve-out (one JSON registry, sibling
to the retired ``op_to_source.json``) rather than a general "config-tuning"
system: a candidate only gets vendor-playbook treatment when it matches a
registry entry by name.
"""

from __future__ import annotations

import copy
import functools
import json
import os
from pathlib import Path
from typing import Any

_REGISTRY_PATH = Path(__file__).resolve().parent / "vendor_operator_playbooks.json"


@functools.lru_cache(maxsize=1)
def load_vendor_operator_playbooks() -> tuple[dict[str, Any], ...]:
    """Load and cache the vendor-operator-playbook registry.

    Returns:
        A tuple of playbook entry dicts (empty when the registry file is
        missing or malformed -- a missing registry must never be fatal to
        the rest of the classification pipeline).
    """
    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    playbooks = raw.get("playbooks") if isinstance(raw, dict) else None
    if not isinstance(playbooks, list):
        return ()
    return tuple(entry for entry in playbooks if isinstance(entry, dict) and entry.get("id"))


def _reset_vendor_operator_playbooks_cache() -> None:
    """Clear the cached registry (tests only, e.g. after monkeypatching the path)."""
    load_vendor_operator_playbooks.cache_clear()


def _candidate_haystack(candidate: dict[str, Any]) -> str:
    """Join every text field a playbook's ``any_marker`` may match against."""
    fields = (
        candidate.get("name"),
        candidate.get("operation"),
        candidate.get("library"),
        candidate.get("source_file"),
        candidate.get("kernel_repo"),
    )
    return " ".join(str(f or "") for f in fields).lower()


def _last_symbol_segment(value: str) -> str:
    """Return the trailing method/function segment of a qualified symbol.

    ``mori::EpDispatchCombineOp::combine`` -> ``combine``; a plain name with
    no separator is returned unchanged. Needed because a class name like
    ``EpDispatchCombineOp`` itself contains the substring "dispatch", so
    matching a role marker against the *whole* qualified name is ambiguous --
    only the actual called method disambiguates dispatch vs combine.
    """
    tail = value
    for sep in ("::", ".", "/"):
        tail = tail.rsplit(sep, 1)[-1]
    return tail


def _role_haystack(candidate: dict[str, Any]) -> str:
    """Return the field(s) a playbook's ``name_any`` (op-role pattern) should match.

    Prefers ``operation`` (already the specific call, e.g. ``"combine"``) over
    ``name`` (which may be a fully-qualified ``Class::method`` symbol whose
    class name can itself contain another role's marker); when only ``name``
    is available, matches its trailing symbol segment rather than the whole
    qualified string.
    """
    operation = str(candidate.get("operation") or "").strip()
    if operation:
        return operation.lower()
    return _last_symbol_segment(str(candidate.get("name") or "")).lower()


def match_vendor_operator_playbook(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Return a matched playbook entry for ``candidate``, or ``None``.

    A candidate matches a playbook when at least one of the playbook's
    ``any_marker`` strings appears somewhere in the candidate's identifying
    fields (name/operation/library/source_file/kernel_repo) AND at least one
    of its ``name_any`` strings appears in the candidate's name/operation --
    e.g. mori's playbook requires both "mori" (library/source evidence) and
    "dispatch" or "combine" (which op within mori this is).

    Args:
        candidate: The hot-kernel candidate dict (as built by
            ``tracelens_analysis``).

    Returns:
        A deep copy of the matched registry entry, augmented with a
        ``"role"`` key set to whichever ``name_any`` marker matched (e.g.
        ``"dispatch"`` or ``"combine"``), or ``None`` when nothing matches.
    """
    if not isinstance(candidate, dict):
        return None
    haystack = _candidate_haystack(candidate)
    role_haystack = _role_haystack(candidate)
    if not haystack or not role_haystack:
        return None
    for playbook in load_vendor_operator_playbooks():
        match = playbook.get("match")
        if not isinstance(match, dict):
            continue
        any_markers = [str(m).lower() for m in (match.get("any_marker") or [])]
        if any_markers and not any(marker in haystack for marker in any_markers):
            continue
        name_markers = [str(m).lower() for m in (match.get("name_any") or [])]
        matched_role = next((m for m in name_markers if m in role_haystack), None)
        if name_markers and matched_role is None:
            continue
        result = copy.deepcopy(playbook)
        result["role"] = matched_role or ""
        return result
    return None


def playbook_group_id(playbook: dict[str, Any]) -> str:
    """Return the stable group id a playbook's sibling roles share."""
    return str(playbook.get("id") or "")


def resolve_kernel_anchor_path(playbook: dict[str, Any]) -> str:
    """Return a stand-in ``source_file`` path for a vendor-playbook candidate.

    A vendor-playbook candidate has no rewritable device source, but
    downstream tooling (``kernel_optimization.py``'s CLI, in particular)
    still gates on a non-empty, path-shaped ``source_file`` before it will
    dispatch to a backend at all. Point that field at the task bundle's
    ``kernel_anchor`` file instead of leaving it empty -- resolved to an
    absolute path under ``$FORGE_PATH`` when that's set and the file exists
    on this host, else the bare bundle-relative path (still path-shaped, so
    it survives ``looks_like_source_path`` even when this analysis runs on a
    host without the KernelForge checkout).

    Args:
        playbook: A matched playbook entry (as returned by
            ``match_vendor_operator_playbook``).

    Returns:
        An absolute or bundle-relative path string; never empty as long as
        the playbook declares a ``kernel_anchor``.
    """
    anchor = str(playbook.get("kernel_anchor") or "").strip()
    bundle = str(playbook.get("task_bundle") or "").strip()
    if not anchor:
        return ""
    relative = f"{bundle}/{anchor}" if bundle else anchor
    forge_root = (os.environ.get("FORGE_PATH") or "").strip()
    if forge_root and bundle:
        candidate = Path(forge_root) / bundle / anchor
        if candidate.is_file():
            return str(candidate)
    return relative
