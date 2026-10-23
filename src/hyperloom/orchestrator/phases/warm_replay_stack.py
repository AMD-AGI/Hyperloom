# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a promoted warm replay records on its optimization_stack entry about the overlays it applied."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..actions.executors._git import _git_head_sha
from ..actions.executors._patch_snapshot import _patch_touched_paths_split
from ..enablement.runtime.adapters import get_adapter
from ..source_snapshot import MANIFEST_NAME, snapshot_source_layer

log = logging.getLogger(__name__)

# A unified-diff overlay that landed. ``applied_nogit`` is what a nogit apply reports when it writes; on a
# pip-installed framework it is the only status an overlay can land under.
_LANDED_PATCH_STATUSES = frozenset({"applied", "applied_3way", "applied_nogit", "present_in_dirty_worktree"})

_SNAPSHOT_SLOT = "replay_warm_recipe"


def replayed_patch_refs(result: Mapping[str, Any], params: Mapping[str, Any]) -> list[str]:
    required = params.get("required_patch_timeline")
    refs: list[str] = []
    for item in result.get("warm_patches_applied") or []:
        if not isinstance(item, dict):
            continue
        if required and item.get("status") not in _LANDED_PATCH_STATUSES:
            continue
        ref = str(item.get("patch_file") or "")
        if ref:
            refs.append(ref)
    return refs


def _landed_overlays(result: Mapping[str, Any]) -> dict[str, list[Path]]:
    """Map each framework root to the overlay diffs this replay applied there, in apply order."""
    landed: dict[str, list[Path]] = {}
    for status in (result.get("warm_patch_result") or {}).get("patches") or []:
        if isinstance(status, Mapping) and status.get("applied_patch") and status.get("target_repo"):
            landed.setdefault(str(status["target_repo"]), []).append(Path(str(status["applied_patch"])))
    return landed


def warm_replay_source_layer(result: Mapping[str, Any], session_dir: str | Path, framework: str) -> dict[str, Any]:
    """The ``source_patch`` handles for the overlays a promoted warm replay left on its tree, or {} when none landed.

    The overlays stay on the framework tree as part of current_best, so they need the durable snapshot an
    integrate_patch KEEP records: a resume onto a stock tree installs the kept layers from it.
    """
    landed = _landed_overlays(result)
    if not landed:
        return {}
    root_text = next(iter(landed))
    handles: dict[str, Any] = {"scope": "source_patch", "framework_root": root_text}
    # A stack entry names one framework root, so a replay that patched several is recorded as not reproducible.
    if len(landed) > 1:
        return handles
    root = Path(root_text)
    upserted, deleted = _patch_touched_paths_split(root, landed[root_text])
    declared_ops = {**dict.fromkeys(upserted, "upsert"), **dict.fromkeys(deleted, "delete")}
    base_sha = _git_head_sha(root)
    try:
        snap = snapshot_source_layer(
            framework_root=root,
            base_sha=base_sha,
            rel_paths=upserted + deleted,
            dest_dir=Path(session_dir) / "optimization_stack" / "src" / _SNAPSHOT_SLOT,
            provenance="replay_warm_recipe",
            declared_ops=declared_ops,
            import_root=get_adapter(framework).source_import_root(root_text),
        )
    except OSError:
        log.exception("warm replay: snapshotting the applied overlays under %s failed", root)
        return handles
    if not snap:
        return handles
    return {
        **handles,
        "source_snapshot": snap["snapshot_dir"],
        "source_manifest": str(Path(snap["snapshot_dir"]) / MANIFEST_NAME),
        "source_snapshot_complete": bool(snap["complete"]),
        "source_import_root": snap["import_root"],
        "target_files": [str(row["rel"]) for row in snap["files"]],
        "base_sha": base_sha,
    }
