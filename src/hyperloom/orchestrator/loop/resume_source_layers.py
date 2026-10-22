# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The resume step that puts the kept source patches back on the framework tree."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON

from ..actions.executors._git import _run_git_cp
from ..actions.executors._nogit_patch import _is_git_tree
from ..actions.executors._patch_snapshot import _git_commit_kept
from ..source_snapshot import SourceLayerInstall, install_source_layers

log = logging.getLogger(__name__)


def _stageable(root: Path, rels: list[str]) -> list[str]:
    """The kept paths git can stage: present on disk or tracked. A deletion HEAD already carries matches neither."""
    cp = _run_git_cp(["-C", str(root), "ls-files", "--", *rels], timeout=30.0)
    if cp is None or cp.returncode != 0:
        return rels
    tracked = set(cp.stdout.splitlines())
    return [rel for rel in rels if rel in tracked or (root / rel).exists()]


def _commit_installed_layers(result: SourceLayerInstall) -> list[str]:
    """Commit the kept files on each git root so HEAD is the accepted stack; return why a root could not be.

    Every candidate that follows stashes a dirty git tree before it applies, so a layer left uncommitted would be
    benched away. A root that is not a git tree is left as installed.
    """
    failed = []
    for root, rels in result.kept.items():
        if not _is_git_tree(root):
            continue
        ok, note = _git_commit_kept(root, "hyperloom resume: reinstall kept source layer", _stageable(root, rels))
        if not ok:
            failed.append(f"{root}: {note or 'git commit failed'}")
    return failed


def install_kept_source_layers(state: Any, session_dir: Path, report: dict[str, Any]) -> str | None:
    """Install the kept source patches on a tree that lost them; return why the resume must halt, if it must.

    The patches live on the framework tree and the session records them in durable snapshots, so a resume onto a
    fresh tree would otherwise bench every candidate against the stock source.
    """
    result = install_source_layers(state.optimization_stack or [])
    report["warnings"].extend(
        {"kind": "source_layer_not_reproducible", "variant": variant} for variant in result.unreproducible
    )
    if result.installed:
        log.warning("resume: installed %d kept source file(s) the framework tree did not hold", len(result.installed))
        report["fixes"].append({"kind": "reinstalled_source_layer", "files": list(result.installed)})
    if result.failed:
        failures, verb = result.failed, "installed on"
    elif failures := _commit_installed_layers(result):
        verb = "committed to"
    else:
        return None
    report["warnings"].append({"kind": "source_layer_install_failed", "failures": list(failures)})
    state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
    state.save(session_dir)
    return f"the kept source layer cannot be {verb} the framework tree: " + "; ".join(failures)
