# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Warm-replay patch application and the exact rollback that undoes it.

A warm replay re-applies the code overlays a recorded recipe was measured with, each into
the checkout it was taken from, and keeps enough per-tree evidence to put every one of
them back byte for byte.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from hyperloom.common.git_safety import safe_directory_args

from ._git import _git_head_sha
from ._patch_snapshot import (
    _create_patch_snapshot,
    _patch_touched_paths_from_text,
    _restore_patch_snapshot,
)

log = logging.getLogger(__name__)


def _git_toplevel(repo_path: str) -> str:
    """Return the work-tree root of ``repo_path``, or ``\"\"`` when it is not in one."""
    if not repo_path:
        return ""
    try:
        result = subprocess.run(
            ["git", *safe_directory_args(["rev-parse", "--show-toplevel"], cwd=repo_path)],
            cwd=repo_path,
            capture_output=True,
            timeout=5,
            check=True,
        )
        return result.stdout.decode().strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return ""


def _patch_present_in_committed_head(
    repo_path: str,
    patch_path: Path,
) -> bool:
    """Check reverse applicability against a temporary index loaded from HEAD."""
    fd, raw_index = tempfile.mkstemp(prefix="warm-head-index-")
    os.close(fd)
    index_path = Path(raw_index)
    index_path.unlink(missing_ok=True)
    env = dict(os.environ)
    env["GIT_INDEX_FILE"] = str(index_path)
    try:
        subprocess.run(
            ["git", "read-tree", "HEAD"],
            cwd=repo_path,
            env=env,
            capture_output=True,
            timeout=15,
            check=True,
        )
        reverse = subprocess.run(
            [
                "git",
                "apply",
                "-R",
                "--check",
                "--cached",
                str(patch_path),
            ],
            cwd=repo_path,
            env=env,
            capture_output=True,
            timeout=30,
            check=False,
        )
        return reverse.returncode == 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False
    finally:
        index_path.unlink(missing_ok=True)
        Path(f"{index_path}.lock").unlink(missing_ok=True)


def _revert_patches(
    repo_path: str,
    pre_sha: str = "",
    snapshot_manifest: Any = None,
) -> dict[str, Any]:
    """Restore exact patch-touched state without broad reset/clean."""
    manifest, error = _validated_restore_manifest(repo_path, pre_sha, snapshot_manifest)
    result = {"ok": False, "errors": [error]} if error else _restore_patch_snapshot(manifest)
    if not result["ok"]:
        log.warning("baseline_executor: exact patch restore failed: %s", result["errors"])
    return result


def _validated_restore_manifest(
    repo_path: str,
    pre_sha: str,
    snapshot_manifest: Any,
) -> tuple[dict[str, Any], str]:
    """Load the snapshot manifest and check it was taken of ``repo_path`` at ``pre_sha``.

    Returns ``(manifest, "")`` when the restore may proceed, else ``({}, error)``.
    """
    manifest = snapshot_manifest
    if isinstance(manifest, (str, Path)):
        try:
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            return {}, f"manifest_read:{exc}"
    if not isinstance(manifest, dict):
        return {}, "missing_manifest"
    manifest_repo_value = str(manifest.get("repo_path") or "").strip()
    if not manifest_repo_value:
        return {}, "missing_manifest_repo"
    try:
        caller_repo = Path(repo_path).resolve(strict=True)
        manifest_repo = Path(manifest_repo_value).resolve(strict=True)
    except (OSError, ValueError) as exc:
        return {}, f"repo_validation:{type(exc).__name__}:{exc}"
    if caller_repo != manifest_repo:
        return {}, f"repo_mismatch:caller={caller_repo}:manifest={manifest_repo}"
    if pre_sha:
        try:
            head = subprocess.run(
                [
                    "git",
                    *safe_directory_args(
                        ["rev-parse", "HEAD"],
                        cwd=caller_repo,
                    ),
                ],
                cwd=caller_repo,
                capture_output=True,
                text=True,
                timeout=15,
                check=True,
            ).stdout.strip()
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            return {}, f"head_validation:{type(exc).__name__}:{exc}"
        if head != pre_sha:
            return {}, f"head_mismatch:expected={pre_sha}:actual={head}"
    return manifest, ""


def _three_way_residue_snapshot(
    repo_path: str,
    touched: list[str],
) -> dict[str, Any]:
    """Capture pre-existing residue only for this patch's paths."""
    root = Path(repo_path).resolve()
    unmerged = (
        subprocess.run(
            [
                "git",
                *safe_directory_args(
                    ["ls-files", "-u", "--", *touched],
                    cwd=repo_path,
                ),
            ],
            cwd=repo_path,
            capture_output=True,
            timeout=15,
            check=True,
        )
        .stdout.decode(errors="replace")
        .splitlines()
    )
    markers = (b"<<<<<<< ", b"=======", b">>>>>>> ")
    rows: dict[str, Any] = {}
    for rel in touched:
        target = root / rel
        marker_lines: list[str] = []
        if target.is_file() and not target.is_symlink():
            marker_lines = [
                line.decode(errors="replace") for line in target.read_bytes().splitlines() if line.startswith(markers)
            ]
        rows[rel] = {
            "reject": (root / f"{rel}.rej").exists(),
            "markers": marker_lines,
        }
    return {"unmerged": unmerged, "paths": rows}


def _verify_three_way_clean(
    repo_path: str,
    touched: list[str],
    before: dict[str, Any],
) -> tuple[bool, str]:
    """Reject only residue newly introduced on this patch's paths."""
    try:
        unmerged = subprocess.run(
            [
                "git",
                *safe_directory_args(
                    ["ls-files", "-u", "--", *touched],
                    cwd=repo_path,
                ),
            ],
            cwd=repo_path,
            capture_output=True,
            timeout=15,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return False, f"post_3way_check_failed:{type(exc).__name__}"
    after_unmerged = unmerged.stdout.decode(errors="replace").splitlines()
    if set(after_unmerged) - set(before.get("unmerged") or []):
        return False, "new_unmerged_index_entries"
    root = Path(repo_path).resolve()
    markers = (b"<<<<<<< ", b"=======", b">>>>>>> ")
    for rel in touched:
        prior = (before.get("paths") or {}).get(rel) or {}
        if (root / f"{rel}.rej").exists() and not prior.get("reject"):
            return False, f"new_reject_file:{rel}.rej"
        target = root / rel
        marker_lines: list[str] = []
        if target.is_file() and not target.is_symlink():
            marker_lines = [
                line.decode(errors="replace") for line in target.read_bytes().splitlines() if line.startswith(markers)
            ]
        if set(marker_lines) - set(prior.get("markers") or []):
            return False, f"new_conflict_marker:{rel}"
    return True, ""


def _revert_warm_patch_state(
    target_repo: str,
    *,
    pre_sha: str = "",
    snapshot_manifest: Any = None,
    nogit_backups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Restore warm-replay patch mutations via git snapshot or nogit backups."""
    if nogit_backups:
        from ...delivery.ledger import restore_records

        _restored, errors = restore_records(nogit_backups)
        if errors:
            log.warning("baseline_executor: nogit patch restore failed: %s", errors)
        return {"ok": not errors, "errors": errors}
    return _revert_patches(target_repo, pre_sha, snapshot_manifest)


#: The per-tree fields that leave this module. ``use_nogit`` stays behind: it
#: describes how the apply ran, and the restore reads the channel off whether
#: backups are present. ``mutated`` says whether this round wrote to the tree at
#: all, which absent restore artifacts alone cannot: a no-op apply and an apply
#: whose artifacts were lost both record none, and only the first is safe to
#: leave standing.
_WARM_TREE_FIELDS = ("root", "pre_sha", "snapshot_manifest", "nogit_backups", "mutated")


def _warm_tree_records(
    trees: Mapping[str, Mapping[str, Any]],
    order: Sequence[str],
    *,
    before_mutation: bool = False,
) -> list[dict[str, Any]]:
    """Return one JSON-safe record per touched tree, in apply order.

    ``before_mutation`` stamps ``mutated`` true. A record persisted ahead of the apply is
    the one record that cannot know what the round went on to write, so a resume that finds
    it has to treat the tree as written-to and restore it from the snapshot taken with it.
    """
    records = [{field: trees[root][field] for field in _WARM_TREE_FIELDS} for root in order if root in trees]
    if before_mutation:
        for record in records:
            record["mutated"] = True
    return records


def _revert_warm_patch_trees(trees: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Restore every tree that was snapshotted, reporting the combined outcome."""
    errors: list[str] = []
    restored: list[str] = []
    for tree in trees:
        root = str(tree.get("root") or "")
        backups = list(tree.get("nogit_backups") or [])
        manifest = tree.get("snapshot_manifest")
        if not backups and not manifest:
            continue
        outcome = _revert_warm_patch_state(
            root,
            pre_sha=str(tree.get("pre_sha") or ""),
            snapshot_manifest=manifest,
            nogit_backups=backups,
        )
        if outcome.get("ok"):
            restored.append(root)
            continue
        errors.extend(f"{root}:{error}" for error in (outcome.get("errors") or ["restore_failed"]))
    return {"ok": not errors, "errors": errors, "restored": restored}


def _apply_warm_patches(
    params: dict[str, Any],
    target_repo: str,
    output_dir: Path,
    *,
    before_mutation: Any = None,
) -> list[dict[str, str]] | dict[str, Any]:
    """Apply warm-replay code patches, each into the checkout it was taken from."""
    patches = params.get("patches") or []
    required_timeline = bool(params.get("required_patch_timeline"))
    if not patches:
        return []
    # The recorded root is the tree the gain was measured on, so it outranks the locally resolved one rather than
    # being checked against it.
    patch_roots = [str((patch or {}).get("framework_root") or "").strip() or target_repo for patch in patches]
    if not any(patch_roots):
        if required_timeline:
            return {
                "required": True,
                "status": "failed",
                "patches": [],
                "applied": [],
                "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
                "failure": "missing_target_repo",
                "rolled_back": True,
            }
        return []

    applied: list[dict[str, str]] = []
    statuses: list[dict[str, Any]] = []
    patch_log_dir = output_dir / "warm_patches"
    patch_log_dir.mkdir(parents=True, exist_ok=True)
    from ._nogit_patch import _apply_patch_no_git, _is_git_tree

    # Two recorded roots inside one checkout are one tree: a git diff's paths are work-tree-root relative, so both
    # resolve there and applying them separately would ignore whatever falls outside the narrower one.
    for idx, root in enumerate(patch_roots):
        if root and (toplevel := _git_toplevel(root)) and Path(toplevel).is_dir():
            patch_roots[idx] = toplevel
    # Ordered by first use so the primary tree stays the one ``target_repo`` named, which is what the single-tree
    # result fields report.
    tree_order: list[str] = list(dict.fromkeys(root for root in patch_roots if root))
    trees: dict[str, dict[str, Any]] = {}
    for root in tree_order:
        if not Path(root).is_dir():
            if required_timeline:
                return {
                    "required": True,
                    "status": "failed",
                    "patches": [],
                    "applied": [],
                    "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
                    "failure": "apply_root_absent",
                    "pre_sha": "",
                    "target_repo": root,
                    "rolled_back": True,
                }
            continue
        git_tree = _is_git_tree(Path(root))
        pre_sha = _git_head_sha(root) if git_tree else ""
        trees[root] = {
            "root": root,
            "pre_sha": pre_sha,
            "use_nogit": not git_tree or not pre_sha,
            "snapshot_manifest": None,
            "nogit_backups": [],
            "mutated": False,
        }
    tree_order = [root for root in tree_order if root in trees]
    if not tree_order:
        return []
    from ...specialists.patch_safety import is_unified_diff, patch_escapes_tree

    primary = trees[tree_order[0]]
    resolved_contents: dict[int, str] = {}
    snapshot_contents: dict[str, list[str]] = {root: [] for root in tree_order}
    for idx, patch in enumerate(patches):
        patch_file = str(patch.get("patch_file") or "")
        content = str(patch.get("patch_content") or "")
        patch_ref = str(patch.get("patch_ref") or "")
        if not content and patch_ref:
            try:
                content = Path(patch_ref).read_text(
                    encoding="utf-8",
                    errors="replace",
                )
            except OSError:
                content = ""
        reason = ""
        if not content:
            reason = "missing_artifact"
        elif not is_unified_diff(content) or "GIT binary patch" in content:
            reason = "unsafe_or_non_text_diff"
        elif patch_escapes_tree(content) is not None:
            reason = "path_escapes_tree"
        elif not _patch_touched_paths_from_text(content):
            reason = "missing_touched_paths"
        if reason:
            if required_timeline:
                return {
                    "required": True,
                    "status": "failed",
                    "patches": [{"patch_ref": patch_file, "status": "failed", "reason": reason}],
                    "applied": [],
                    "failed_ref": patch_file,
                    "failure": reason,
                    "pre_sha": primary["pre_sha"],
                    "target_repo": primary["root"],
                    "rolled_back": False,
                }
            continue
        root = patch_roots[idx]
        if root not in trees:
            continue
        resolved_contents[idx] = content
        snapshot_contents[root].append(content)

    # Every tree is snapshotted before any of them is written to, so each snapshot holds pristine content.
    for position, root in enumerate(tree_order):
        tree = trees[root]
        if tree["use_nogit"] or not snapshot_contents[root]:
            continue
        try:
            # _create_patch_snapshot owns one fixed sub-directory per output dir and clears it, so trees past the
            # first are given their own.
            tree["snapshot_manifest"] = _create_patch_snapshot(
                root,
                snapshot_contents[root],
                output_dir if position == 0 else output_dir / "warm_patch_trees" / f"{position:02d}",
            )
        except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            restore = _revert_warm_patch_trees(trees.values())
            if required_timeline:
                return {
                    "required": True,
                    "status": "failed",
                    "patches": [],
                    "applied": [],
                    "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
                    "failure": f"snapshot_failed:{type(exc).__name__}",
                    "pre_sha": primary["pre_sha"],
                    "target_repo": primary["root"],
                    "trees": _warm_tree_records(trees, tree_order),
                    "rolled_back": bool(restore.get("ok")),
                }
            log.warning(
                "baseline_executor: skipping legacy warm patches because the "
                "rollback snapshot could not be created: %s",
                exc,
            )
            return []
    snapshot_manifest = primary["snapshot_manifest"]
    if any(tree["snapshot_manifest"] for tree in trees.values()):
        params["_warm_patch_trees"] = _warm_tree_records(trees, tree_order, before_mutation=True)
        if before_mutation is not None and not bool(
            before_mutation(_warm_tree_records(trees, tree_order, before_mutation=True))
        ):
            return {
                "required": required_timeline,
                "status": "failed",
                "patches": [],
                "applied": [],
                "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
                "failure": "pending_state_persist_failed",
                "pre_sha": primary["pre_sha"],
                "target_repo": primary["root"],
                "snapshot_manifest": snapshot_manifest,
                "trees": _warm_tree_records(trees, tree_order),
                "rollback": {"ok": True, "errors": []},
                "rolled_back": True,
            }
    failed_ref = ""
    failure = ""

    for idx, patch in enumerate(patches):
        patch_file = patch.get("patch_file") or ""
        patch_content = resolved_contents.get(idx) or patch.get("patch_content") or ""
        patch_ref = patch.get("patch_ref") or ""
        status: dict[str, Any] = {
            "patch_ref": patch_file,
            "timeline_index": patch.get("timeline_index", idx),
        }
        tree = trees.get(patch_roots[idx])
        if tree is None:
            status.update(status="failed", reason="apply_root_absent")
            statuses.append(status)
            continue
        # The patch goes into its own recorded tree, so these are per patch rather than fixed for the whole sequence.
        target_repo = tree["root"]
        target_path = Path(target_repo)
        use_nogit = tree["use_nogit"]
        nogit_backups = tree["nogit_backups"]
        status["target_repo"] = target_repo

        if not patch_content and not patch_ref:
            log.warning(
                "baseline_executor: patch entry has no content/ref, skipping: %s",
                patch_file,
            )
            status.update(status="failed", reason="missing_artifact")
            statuses.append(status)
            if required_timeline:
                failed_ref, failure = patch_file, "missing_artifact"
                break
            continue

        # Resolve patch content: prefer inline content, fallback to patch_ref file.
        if not patch_content and patch_ref:
            ref_path = Path(patch_ref)
            if ref_path.is_file():
                try:
                    patch_content = ref_path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    log.warning(
                        "baseline_executor: cannot read patch_ref %s: %s",
                        patch_ref,
                        exc,
                    )
                    status.update(status="failed", reason="artifact_read_failed")
                    statuses.append(status)
                    if required_timeline:
                        failed_ref, failure = patch_file, "artifact_read_failed"
                        break
                    continue
            else:
                log.warning(
                    "baseline_executor: patch_ref not found: %s",
                    patch_ref,
                )
                status.update(status="failed", reason="missing_artifact")
                statuses.append(status)
                if required_timeline:
                    failed_ref, failure = patch_file, "missing_artifact"
                    break
                continue

        # Structural safety gate on untrusted KB-sourced patch_content before it is git-applied to the live checkout:
        # reject non-diff blobs and any patch whose header path escapes the tree (absolute / ``..``).
        if not is_unified_diff(patch_content) or "GIT binary patch" in patch_content:
            log.warning(
                "baseline_executor: skipping warm patch %s — not a unified diff",
                patch_file,
            )
            status.update(status="failed", reason="unsafe_or_non_text_diff")
            statuses.append(status)
            if required_timeline:
                failed_ref, failure = patch_file, "unsafe_or_non_text_diff"
                break
            continue
        _escape = patch_escapes_tree(patch_content)
        if _escape is not None:
            log.warning(
                "baseline_executor: skipping warm patch %s — path escapes tree: %r",
                patch_file,
                _escape,
            )
            status.update(status="failed", reason="path_escapes_tree")
            statuses.append(status)
            if required_timeline:
                failed_ref, failure = patch_file, "path_escapes_tree"
                break
            continue

        # Write patch to temp file then apply.
        patch_path = patch_log_dir / f"{idx:03d}_{Path(patch_file).stem or 'patch'}.diff"
        patch_path.write_text(patch_content, encoding="utf-8")

        method = ""
        try:
            if use_nogit:
                backup_root = patch_log_dir / "patch_backups"
                ok, err, backups, _feedback = _apply_patch_no_git(
                    target_path,
                    patch_path,
                    backup_root,
                    seq_offset=len(nogit_backups),
                )
                if not ok:
                    raise RuntimeError(err or "nogit patch apply failed")
                nogit_backups.extend(backups)
                # A real apply backs up every file it writes, so an empty set is the
                # applier reporting an overlay the tree already carried.
                method = "applied_nogit" if backups else "already_present"
            else:
                checked = subprocess.run(
                    ["git", "apply", "--check", str(patch_path)],
                    cwd=target_repo,
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                if checked.returncode == 0:
                    subprocess.run(
                        ["git", "apply", str(patch_path)],
                        cwd=target_repo,
                        capture_output=True,
                        timeout=30,
                        check=True,
                    )
                    method = "applied"
                elif required_timeline:
                    reverse = subprocess.run(
                        ["git", "apply", "-R", "--check", str(patch_path)],
                        cwd=target_repo,
                        capture_output=True,
                        timeout=30,
                        check=False,
                    )
                    if reverse.returncode == 0:
                        method = (
                            "already_present"
                            if _patch_present_in_committed_head(
                                target_repo,
                                patch_path,
                            )
                            else "present_in_dirty_worktree"
                        )
                    else:
                        touched = _patch_touched_paths_from_text(patch_content)
                        before_residue = _three_way_residue_snapshot(
                            target_repo,
                            touched,
                        )
                        three_way = subprocess.run(
                            ["git", "apply", "--3way", str(patch_path)],
                            cwd=target_repo,
                            capture_output=True,
                            timeout=30,
                            check=False,
                        )
                        if three_way.returncode == 0:
                            clean, residue = _verify_three_way_clean(
                                target_repo,
                                touched,
                                before_residue,
                            )
                            if not clean:
                                raise RuntimeError(residue)
                            method = "applied_3way"
                        else:
                            detail = (
                                three_way.stderr.decode(errors="replace")[:500]
                                if three_way.stderr
                                else "git apply --3way failed"
                            )
                            raise RuntimeError(detail)
                else:
                    detail = (
                        checked.stderr.decode(errors="replace")[:500] if checked.stderr else "git apply --check failed"
                    )
                    raise RuntimeError(detail)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
            log.warning(
                "baseline_executor: warm patch apply failed for %s: %s",
                patch_file,
                exc,
            )
            reason = "nogit_apply_failed" if use_nogit else "git_apply_failed"
            status.update(status="failed", reason=reason, detail=str(exc)[:500])
            statuses.append(status)
            if required_timeline:
                failed_ref, failure = patch_file, reason
                break
            continue

        if method in ("applied", "applied_3way", "applied_nogit"):
            tree["mutated"] = True
        item = {
            "patch_file": patch_file,
            "idx": str(idx),
            "status": method,
        }
        applied.append(item)
        status["status"] = method
        statuses.append(status)

    records = _warm_tree_records(trees, tree_order)
    if any(tree["snapshot_manifest"] or tree["nogit_backups"] for tree in trees.values()):
        params["_warm_patch_trees"] = records
    # A best-effort timeline reports only the patches that landed, so without
    # this the per-patch reasons computed above would die with this frame --
    # and they are the only record of the ones that did not.
    params["_warm_patch_statuses"] = statuses

    if required_timeline:
        if failed_ref:
            # One overlay failing voids the whole replay, and the sequence may already have written to trees before
            # this one, so every tree is restored rather than just the one that failed.
            restore = _revert_warm_patch_trees(trees.values())
            return {
                "required": True,
                "status": "failed",
                "patches": statuses,
                "applied": applied,
                "failed_ref": failed_ref,
                "failure": failure,
                "pre_sha": primary["pre_sha"],
                "target_repo": primary["root"],
                "snapshot_manifest": primary["snapshot_manifest"],
                "trees": records,
                "rolled_back": bool(restore.get("ok")),
                "rollback": restore,
            }
        return {
            "required": True,
            "status": "prepared",
            "patches": statuses,
            "applied": applied,
            "failed_ref": "",
            "pre_sha": primary["pre_sha"],
            "target_repo": primary["root"],
            "snapshot_manifest": primary["snapshot_manifest"],
            "trees": records,
            "rolled_back": False,
        }
    return applied


def _stamp_warm_patch_trees(
    result: dict[str, Any],
    patch_application: Mapping[str, Any],
    pre_sha: str,
) -> None:
    """Record which trees this round patched, for prelude to promote or restore."""
    trees = list(patch_application.get("trees") or [])
    primary = trees[0] if trees else {}
    target = str(patch_application.get("target_repo") or primary.get("root") or "")
    result["warm_patch_result"] = dict(patch_application)
    result["warm_patch_trees"] = trees
    result["warm_patch_pre_sha"] = str(primary.get("pre_sha") or pre_sha)
    result["warm_patch_target"] = target
    result["warm_patch_snapshot_manifest"] = patch_application.get("snapshot_manifest") or primary.get(
        "snapshot_manifest"
    )
    result["warm_patch_canonical_target"] = target


def _stamp_warm_patch_outcome(
    result: dict[str, Any],
    patch_application: list[dict[str, str]] | Mapping[str, Any],
    params: Mapping[str, Any],
    pre_sha: str,
) -> None:
    """Carry the patch application's report into the round's result.

    A required timeline reports the structure prelude promotes or restores
    from. A best-effort one reports only the patches that landed, so its
    per-patch statuses are lifted out of ``params`` instead: a patch that
    silently failed to apply would otherwise leave nothing behind, and the
    round would be measured on a tree nobody downstream can describe.
    """
    if isinstance(patch_application, Mapping):
        _stamp_warm_patch_trees(result, patch_application, pre_sha)
        result["warm_kernel_apply_results"] = list(params.get("warm_kernel_apply_results") or [])
        return
    if statuses := list(params.get("_warm_patch_statuses") or []):
        result["warm_patch_result"] = {"required": False, "status": "prepared", "patches": statuses}


def _revert_legacy_warm_patch_trees(params: Mapping[str, Any]) -> dict[str, Any]:
    """Undo a legacy (non-required) apply so nothing leaks into the next task."""
    return _revert_warm_patch_trees(params.get("_warm_patch_trees") or [])


def restore_warm_kernel_snapshots(
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """Restore exact kernel target bytes captured before each mutation."""
    errors: list[str] = []
    for snapshot in reversed(snapshots):
        target = Path(str(snapshot.get("target") or ""))
        try:
            if snapshot.get("existed"):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(str(snapshot.get("backup") or "")).read_bytes())
                if snapshot.get("mode") is not None:
                    target.chmod(int(snapshot["mode"]))
            elif target.exists() or target.is_symlink():
                target.unlink()
            if snapshot.get("existed"):
                expected = Path(str(snapshot.get("backup") or "")).read_bytes()
                if not target.is_file() or target.read_bytes() != expected:
                    raise OSError("kernel restore verification failed")
            elif target.exists() or target.is_symlink():
                raise OSError("kernel target still exists after restore")
        except OSError as exc:
            errors.append(f"{target}:{type(exc).__name__}:{exc}")
    return {"ok": not errors, "errors": errors}


def revert_warm_kernel_patches(
    applied: list[dict[str, Any]],
    snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Rollback kernels exactly, reporting every failure."""
    from ._kernel_agent_tool import _maybe_revert_kernel_patch

    errors: list[str] = []
    for apply_result in reversed(applied):
        if not apply_result.get("manifest_path"):
            continue
        try:
            reverted = _maybe_revert_kernel_patch(apply_result)
            if reverted.get("status") != "ok":
                raise RuntimeError(
                    str(
                        reverted.get("error")
                        or reverted.get("reason")
                        or f"kernel revert status={reverted.get('status')}"
                    )
                )
        except Exception as exc:
            log.warning("warm-kernel KB: revert failed", exc_info=True)
            errors.append(f"{type(exc).__name__}:{exc}")
    if snapshots:
        restored = restore_warm_kernel_snapshots(snapshots)
        errors.extend(restored.get("errors") or [])
    return {"ok": not errors, "errors": errors}


def _rollback_warm_kernel_apply_results(
    results: Any,
    snapshots: Any = None,
) -> dict[str, Any]:
    """Rollback kernel mutations and report whether every restore succeeded."""
    if not isinstance(results, list):
        return {"ok": False, "errors": ["invalid_apply_results"]}

    return revert_warm_kernel_patches(
        [r for r in results if isinstance(r, dict)],
        list(snapshots) if isinstance(snapshots, list) else None,
    )
