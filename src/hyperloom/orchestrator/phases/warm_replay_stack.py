# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a promoted warm replay records on its optimization_stack entry about the overlays it applied."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..actions.executors._git import _git_head_sha, _run_git_cp
from ..actions.executors._nogit_patch import _is_git_tree
from ..actions.executors.baseline import _revert_warm_patch_state
from ..actions.executors._patch_snapshot import _git_commit_kept, _patch_touched_paths, _patch_touched_paths_split
from ..enablement.runtime.adapters import get_adapter
from ..source_snapshot import MANIFEST_NAME, SourceLayer, snapshot_source_layer

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
            SourceLayer(
                framework_root=root,
                base_sha=base_sha,
                rel_paths=upserted + deleted,
                declared_ops=declared_ops,
                import_root=get_adapter(framework).source_import_root(root_text),
            ),
            dest_dir=Path(session_dir) / "optimization_stack" / "src" / _SNAPSHOT_SLOT,
            provenance="replay_warm_recipe",
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


# Set on the pending replay record, and persisted, before the promotion commits the overlays.
_OVERLAYS_COMMITTED_KEY = "recipe_overlays_committed"


def committing_overlays(pending: Mapping[str, Any]) -> dict[str, Any]:
    """The pending replay record to persist before the promotion commits, so any later rollback undoes the commit."""
    return {**pending, _OVERLAYS_COMMITTED_KEY: True}


def commit_warm_replay_overlays(result: Mapping[str, Any]) -> list[str]:
    """Commit the overlays a promoted warm replay applied on each git framework tree; return why a tree could not be.

    Every integrate_patch candidate stashes a dirty tree before it applies, so overlays left uncommitted would be
    benched away from every candidate after the promotion. A root that is not a git tree holds them as written.
    """
    failed = []
    for root_text, overlays in _landed_overlays(result).items():
        root = Path(root_text)
        if not _is_git_tree(root):
            continue
        ok, note = _git_commit_kept(root, "hyperloom KEEP replay_warm_recipe", _patch_touched_paths(root, overlays))
        if not ok:
            failed.append(f"{root}: {note or 'git commit failed'}")
    return failed


def uncommit_promoted_overlays(pending: Mapping[str, Any], root: str, pre_sha: str) -> str:
    """Move ``root`` back to ``pre_sha`` when the promotion committed the overlays on it; return why it could not.

    The restore requires HEAD at ``pre_sha``, and a tree the replay did not write to keeps the overlays it already
    held as uncommitted changes. Only a single commit directly on ``pre_sha`` is undone; any other HEAD is left for
    the restore to refuse.
    """
    if not pending.get(_OVERLAYS_COMMITTED_KEY) or not pre_sha:
        return ""
    head = _git_head_sha(root)
    if not head:
        return f"recipe:{root}:promotion_commit_state_unreadable:HEAD"
    if head == pre_sha:
        return ""
    parent = _run_git_cp(["-C", root, "rev-parse", "HEAD^"], timeout=30.0)
    if parent is None:
        return f"recipe:{root}:promotion_commit_state_unreadable:HEAD^:spawn failed"
    if parent.returncode != 0:
        detail = (parent.stderr or "").strip()
        return f"recipe:{root}:promotion_commit_state_unreadable:HEAD^:{detail}"
    if parent.stdout.strip() != pre_sha:
        return ""
    reset = _run_git_cp(["-C", root, "reset", "-q", pre_sha], timeout=30.0)
    if reset is None or reset.returncode != 0:
        detail = "spawn failed" if reset is None else (reset.stderr or "").strip()
        return f"recipe:{root}:promotion_commit_not_undone:{detail}"
    return ""


def restore_recipe_trees(result: Mapping[str, Any], pending: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Undo the promotion commit and restore every tree the replay patched, one restore report per tree."""
    restores: list[dict[str, Any]] = []
    for tree in warm_patch_trees(result, pending):
        target = str(tree.get("root") or "")
        if not target:
            continue
        if undo_error := uncommit_promoted_overlays(pending, target, str(tree.get("pre_sha") or "")):
            restores.append({"ok": False, "errors": [undo_error]})
            continue
        recipe_manifest = tree.get("snapshot_manifest")
        backups = list(tree.get("nogit_backups") or [])
        if not bool(tree.get("mutated", True)):
            # An overlay the tree already carried applies as a no-op, writing
            # nothing and so owing nothing. A record predating the flag reads as
            # written-to, and still has to answer with a channel below.
            continue
        if not recipe_manifest and not backups:
            restores.append({"ok": False, "errors": [f"recipe:{target}:missing_snapshot_manifest"]})
            continue
        restores.append(
            _revert_warm_patch_state(
                target,
                pre_sha=str(tree.get("pre_sha") or ""),
                snapshot_manifest=recipe_manifest,
                nogit_backups=backups,
            )
        )
    return restores
