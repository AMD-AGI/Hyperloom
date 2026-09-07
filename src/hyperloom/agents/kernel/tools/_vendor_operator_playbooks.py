###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Vendor-operator-playbook registry: route a closed-source hot kernel to a"""

from __future__ import annotations

import copy
import functools
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).resolve().parent / "vendor_operator_playbooks.json"


@functools.lru_cache(maxsize=1)
def load_vendor_operator_playbooks() -> tuple[dict[str, Any], ...]:
    """Load and cache the vendor-operator-playbook registry."""
    try:
        raw = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    playbooks = raw.get("playbooks") if isinstance(raw, dict) else None
    if not isinstance(playbooks, list):
        return ()
    # ``kernel_anchor`` is required, not optional.
    usable: list[dict[str, Any]] = []
    for entry in playbooks:
        if not isinstance(entry, dict) or not entry.get("id"):
            continue
        if not str(entry.get("kernel_anchor") or "").strip():
            log.warning(
                "vendor operator playbook %r declares no kernel_anchor; ignoring it",
                entry.get("id"),
            )
            continue
        usable.append(entry)
    return tuple(usable)


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
        candidate.get("trace_launcher_file"),
    )
    return " ".join(str(f or "") for f in fields).lower()


def _last_symbol_segment(value: str) -> str:
    """Return the trailing method/function segment of a qualified symbol."""
    tail = value
    for sep in ("::", ".", "/"):
        tail = tail.rsplit(sep, 1)[-1]
    return tail


def _role_haystack(candidate: dict[str, Any]) -> str:
    """Return the field(s) a playbook's ``name_any`` (op-role pattern) should match."""
    operation = str(candidate.get("operation") or "").strip()
    if operation:
        return _last_symbol_segment(operation).lower()
    return _last_symbol_segment(str(candidate.get("name") or "")).lower()


def match_vendor_operator_playbook(candidate: dict[str, Any]) -> dict[str, Any] | None:
    """Return a matched playbook entry for ``candidate``, or ``None``."""
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


def resolve_kernel_anchor_path(playbook: dict[str, Any]) -> str:
    """Return a stand-in ``source_file`` path for a vendor-playbook candidate."""
    anchor = str(playbook.get("kernel_anchor") or "").strip()
    bundle = str(playbook.get("task_bundle") or "").strip()
    if not anchor:
        return ""
    relative = f"{bundle}/{anchor}" if bundle else anchor
    # The bundle is packaged, so this resolves to a real file.
    from kernelforge.resources import default_project_root, resource_path

    return str(resource_path(relative, default_project_root(), missing_ok=True))
