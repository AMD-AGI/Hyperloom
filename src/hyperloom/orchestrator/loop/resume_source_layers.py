# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The resume step that puts the kept source patches back on the framework tree and the multi-node pods."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.breakdown.stop_reasons import PATCH_RECOVERY_INCOMPLETE_STOP_REASON
from hyperloom.inference_optimizer.session.session_paths import kernel_patch_backup_roots

from ..actions.executors._git import _run_git_cp
from ..actions.executors._nogit_patch import _is_git_tree
from ..actions.executors._patch_snapshot import _git_commit_kept
from ..kernel.kept_patch_restore import KeptPatchRestoreError, restore_kept_patches_on_pods
from ..source_snapshot import SourceLayerInstall, install_source_layers

log = logging.getLogger(__name__)

_GIT_QUERY_TIMEOUT_S = 30.0


def _stageable(root: Path, rels: list[str]) -> list[str]:
    """The kept paths git can stage: present on disk or tracked. A deletion HEAD already carries matches neither."""
    cp = _run_git_cp(["-C", str(root), "ls-files", "--", *rels], timeout=_GIT_QUERY_TIMEOUT_S)
    if cp is None or cp.returncode != 0:
        return rels
    tracked = set(cp.stdout.splitlines())
    return [rel for rel in rels if rel in tracked or (root / rel).exists()]


def _committed_past(root: Path, rel: str, op: str, source: Path) -> bool:
    """Whether HEAD's history already carries ``rel``'s kept state, so the different content now is later work.

    A fresh tree never held the kept state, so this is False there and the layer is installed. A root that is not a
    git tree has no history to tell the two apart.
    """
    if not _is_git_tree(root):
        return False
    if op == "delete":
        query = ["--diff-filter=D"]
    else:
        blob = _run_git_cp(
            ["-C", str(root), "hash-object", f"--path={rel}", "--", str(source)], timeout=_GIT_QUERY_TIMEOUT_S
        )
        if blob is None or blob.returncode != 0:
            return False
        query = [f"--find-object={blob.stdout.strip()}"]
    cp = _run_git_cp(
        ["-C", str(root), "log", "-1", "--format=%H", *query, "HEAD", "--", rel], timeout=_GIT_QUERY_TIMEOUT_S
    )
    return cp is not None and cp.returncode == 0 and bool(cp.stdout.strip())


def _commit_installed_layers(result: SourceLayerInstall) -> list[str]:
    """Commit the kept files on each git root so HEAD is the accepted stack; return why a root could not be.

    Every candidate that follows stashes a dirty git tree before it applies, so a layer left uncommitted would be
    benched away. A root that is not a git tree is left as installed.
    """
    failed = []
    superseded = set(result.superseded)
    for root, rels in result.kept.items():
        if not _is_git_tree(root):
            continue
        rels = [rel for rel in rels if str(root / rel) not in superseded]
        ok, note = _git_commit_kept(root, "hyperloom resume: reinstall kept source layer", _stageable(root, rels))
        if not ok:
            failed.append(f"{root}: {note or 'git commit failed'}")
    return failed


def _halt_resume(state: Any, session_dir: Path, report: dict[str, Any], warning: dict[str, Any]) -> None:
    """Record why the resume cannot go on, and end the session as an incomplete patch recovery."""
    report["warnings"].append(warning)
    state.set_stop_reason(PATCH_RECOVERY_INCOMPLETE_STOP_REASON)
    state.save(session_dir)


def _restore_kept_kernel_files_on_pods(state: Any, session_dir: Path, report: dict[str, Any]) -> str | None:
    """Send the kept kernel files to the multi-node pods; return why the resume must halt, if it must."""
    try:
        restored = restore_kept_patches_on_pods(kernel_patch_backup_roots(session_dir))
    except KeptPatchRestoreError as exc:
        _halt_resume(state, session_dir, report, {"kind": "kept_kernel_file_restore_failed", "error": str(exc)})
        return f"the kept kernel patches cannot be restored on the multi-node pods: {exc}"
    if restored:
        log.warning("resume: restored %d kept kernel file(s) on the multi-node pods", len(restored))
        report["fixes"].append({"kind": "restored_kept_kernel_files", "files": [str(path) for path in restored]})
    return None


def install_kept_source_layers(state: Any, session_dir: Path, report: dict[str, Any]) -> str | None:
    """Put the kept source patches back where the resume measures; return why the resume must halt, if it must.

    The patches live on the framework tree and the session records them in durable snapshots, so a resume onto a
    fresh tree would otherwise bench every candidate against the stock source. Multi-node pods start from the image's
    sources, so they get the kept kernel files from this tree once it holds them.
    """
    result = install_source_layers(state.optimization_stack or [], superseded=_committed_past)
    report["warnings"].extend(
        {"kind": "source_layer_not_reproducible", "variant": variant} for variant in result.unreproducible
    )
    if result.superseded:
        report["warnings"].append({"kind": "source_layer_superseded", "files": list(result.superseded)})
    if result.installed:
        log.warning("resume: installed %d kept source file(s) the framework tree did not hold", len(result.installed))
        report["fixes"].append({"kind": "reinstalled_source_layer", "files": list(result.installed)})
    if result.failed:
        failures, verb = result.failed, "installed on"
    elif failures := _commit_installed_layers(result):
        verb = "committed to"
    else:
        return _restore_kept_kernel_files_on_pods(state, session_dir, report)
    _halt_resume(state, session_dir, report, {"kind": "source_layer_install_failed", "failures": list(failures)})
    return f"the kept source layer cannot be {verb} the framework tree: " + "; ".join(failures)
