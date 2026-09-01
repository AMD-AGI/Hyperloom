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
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

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
    # ``kernel_anchor`` is required, not optional. Every consumer of a match
    # overrides ``source_file`` with the anchor, so an entry without one has the
    # override substitute nothing for whatever tier resolved the path and land
    # the candidate as ``missing_native_source``. Refusing it here keeps that
    # shape out of the pipeline entirely, rather than asking each consumer to
    # guard against it -- which is where a guard for it went wrong before.
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
    """Join every text field a playbook's ``any_marker`` may match against.

    Includes ``trace_launcher_file`` alongside the resolved-source fields:
    for a kernel whose device launch is hidden behind a CUDA/HIP-graph
    replay (TraceLens reconstructs it as a "Synthetic Op" with no surviving
    module chain -- mori's EP dispatch/combine hit this in practice), the
    normal ``library``/``source_file``/``kernel_repo`` trio all resolve
    empty and the Python call-stack frame that first launched the op (e.g.
    ``.../site-packages/mori/jit/hip_driver.py``) is the *only* place the
    vendor identity marker survives. Omitting it silently drops exactly the
    graph-captured candidates this registry exists to route.
    """
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

    Prefers ``operation`` (often already the specific call, e.g.
    ``"combine"``) over ``name`` (which may be a fully-qualified
    ``Class::method`` symbol whose class name can itself contain another
    role's marker); either way, only the trailing symbol segment is
    matched, never the whole qualified string. This repo's own convention
    (``_task_group_contract.logical_operator_name()``) is to set ``operation``
    to the fully-qualified name too, e.g. ``mori::EpDispatchCombineOp::combine``
    -- taking ``operation`` verbatim would silently reintroduce the exact
    dispatch/combine ambiguity this function exists to resolve the moment
    some producer starts populating that field on a candidate row (PR #1191
    review finding #6).
    """
    operation = str(candidate.get("operation") or "").strip()
    if operation:
        return _last_symbol_segment(operation).lower()
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
    ``kernel_anchor`` file instead of leaving it empty.

    The bundle now ships inside the installed ``kernelforge`` package, so the
    resolved path is a real file on this host rather than a placeholder. The
    previous fallback -- an absolute path under a synthetic
    ``/nonexistent-forge-path`` root -- existed only to keep the value
    path-shaped when no checkout was around; it dressed "the env var is unset"
    up as "the file is missing", which is a different and much quieter failure.
    An operator substituting a bundle points ``$KERNELFORGE_PROJECT_ROOT`` at a
    tree holding it; :func:`resource_path` honours that ahead of the package.

    Args:
        playbook: A matched playbook entry (as returned by
            ``match_vendor_operator_playbook``).

    Returns:
        An absolute path string; never empty as long as the playbook
        declares a ``kernel_anchor``, and never relative -- a relative
        string here would later be reinterpreted by ``Path(...).resolve()``
        against whatever the apply-stage process's CWD happens to be, not
        against this bundle (PR #1191 review finding #8).
    """
    anchor = str(playbook.get("kernel_anchor") or "").strip()
    bundle = str(playbook.get("task_bundle") or "").strip()
    if not anchor:
        return ""
    relative = f"{bundle}/{anchor}" if bundle else anchor
    # The bundle is packaged, so this resolves to a real file. ``missing_ok``
    # still yields an absolute, package-anchored path for a bundle this
    # installation does not carry, rather than a bare relative string that a
    # later ``Path(...).resolve()`` would reinterpret against some other CWD.
    from kernelforge.resources import default_project_root, resource_path

    return str(resource_path(relative, default_project_root(), missing_ok=True))
