# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a promoted warm replay records on its optimization_stack entry about the overlays it applied."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from ..actions.executors._git import _git_head_sha, _run_git_cp
from ..actions.executors._nogit_patch import _is_git_tree
from ..actions.executors._patch_snapshot import _git_commit_kept, _patch_touched_paths, _patch_touched_paths_split
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


def warm_patch_trees(result: Mapping[str, Any], pending: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    trees = [
        tree
        for tree in result.get("warm_patch_trees") or pending.get("recipe_patch_trees") or []
        if isinstance(tree, Mapping)
    ]
    # A round that predates the list reports only its primary tree.
    return trees or [
        {
            "root": result.get("warm_patch_target") or pending.get("recipe_patch_target"),
            "pre_sha": result.get("warm_patch_pre_sha") or pending.get("recipe_patch_pre_sha"),
            "snapshot_manifest": (
                result.get("warm_patch_snapshot_manifest") or pending.get("recipe_patch_snapshot_manifest")
            ),
        }
    ]


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


_OVERLAY_HEADS_KEY = "overlay_commit_heads"


class OverlayCommit(NamedTuple):
    """What committing a promoted warm replay's overlays did."""

    #: The HEAD each tree had before the commit, for every tree a commit advanced.
    advanced: dict[str, str]
    #: Why a tree's overlays could not be committed, one line each.
    failed: list[str]

    def record(self) -> dict[str, dict[str, str]]:
        """The promotion fields :func:`uncommit_warm_replay_overlays` reads back."""
        return {_OVERLAY_HEADS_KEY: dict(self.advanced)}


def commit_warm_replay_overlays(result: Mapping[str, Any]) -> OverlayCommit:
    """Commit the overlays a promoted warm replay applied on each git framework tree.

    Every integrate_patch candidate stashes a dirty tree before it applies, so overlays left uncommitted would be
    benched away from every candidate after the promotion. A root that is not a git tree holds them as written.
    """
    commit = OverlayCommit({}, [])
    for root_text, overlays in _landed_overlays(result).items():
        root = Path(root_text)
        if not _is_git_tree(root):
            continue
        head = _git_head_sha(root)
        ok, note = _git_commit_kept(root, "hyperloom KEEP replay_warm_recipe", _patch_touched_paths(root, overlays))
        if not ok:
            commit.failed.append(f"{root}: {note or 'git commit failed'}")
        elif not note:
            commit.advanced[root_text] = head
    return commit


def uncommit_warm_replay_overlays(promotion: Mapping[str, Any]) -> None:
    """Move each tree the promotion committed back to its prior HEAD, leaving the overlays for the rollback."""
    for root, head in (promotion.get(_OVERLAY_HEADS_KEY) or {}).items():
        _run_git_cp(["-C", root, "reset", "-q", "--soft", head], timeout=30.0)
