# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply, snapshot and revert of the warm-kernel patch replay."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple

from hyperloom_common.git_safety import safe_directory_args
from .._git import _git_head_sha
from .._patch_snapshot import (
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
        after = _three_way_residue_snapshot(repo_path, touched)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        return False, f"post_3way_check_failed:{type(exc).__name__}"
    if set(after["unmerged"]) - set(before.get("unmerged") or []):
        return False, "new_unmerged_index_entries"
    for rel in touched:
        prior = (before.get("paths") or {}).get(rel) or {}
        now = after["paths"][rel]
        if now["reject"] and not prior.get("reject"):
            return False, f"new_reject_file:{rel}.rej"
        if set(now["markers"]) - set(prior.get("markers") or []):
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
        from ....delivery.ledger import restore_records

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
    *,
    before_mutation: bool = False,
) -> list[dict[str, Any]]:
    """Return one JSON-safe record per touched tree, in apply order.

    ``before_mutation`` stamps ``mutated`` true. A record persisted ahead of the apply is
    the one record that cannot know what the round went on to write, so a resume that finds
    it has to treat the tree as written-to and restore it from the snapshot taken with it.
    """
    records = [{field: tree[field] for field in _WARM_TREE_FIELDS} for tree in trees.values()]
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


class _OverlayText(NamedTuple):
    content: str
    reason: str


def _refused_warm_replay(patches: list[Any], failure: str) -> dict[str, Any]:
    return {
        "required": True,
        "status": "failed",
        "patches": [],
        "applied": [],
        "failed_ref": str((patches[0] or {}).get("patch_file") or ""),
        "failure": failure,
        "rolled_back": True,
    }


def _warm_patch_trees(roots: list[str]) -> dict[str, dict[str, Any]]:
    from .._nogit_patch import _is_git_tree

    # Two recorded roots inside one checkout are one tree: a git diff's paths are work-tree-root relative, so both
    # resolve there and applying them separately would ignore whatever falls outside the narrower one.
    for idx, root in enumerate(roots):
        if root and (toplevel := _git_toplevel(root)) and Path(toplevel).is_dir():
            roots[idx] = toplevel
    trees: dict[str, dict[str, Any]] = {}
    # Ordered by first use: the single-tree result fields report the first recorded root's tree, so that one has to
    # stay primary.
    for root in dict.fromkeys(root for root in roots if root):
        if not Path(root).is_dir():
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
    return trees


def _warm_patch_text(patch: Mapping[str, Any]) -> _OverlayText:
    content = str(patch.get("patch_content") or "")
    patch_ref = str(patch.get("patch_ref") or "")
    patch_file = str(patch.get("patch_file") or "")
    if content:
        return _OverlayText(content, "")
    if not patch_ref:
        log.warning("baseline_executor: patch entry has no content/ref, skipping: %s", patch_file)
        return _OverlayText("", "missing_artifact")
    try:
        return _OverlayText(Path(patch_ref).read_text(encoding="utf-8", errors="replace"), "")
    except FileNotFoundError:
        log.warning("baseline_executor: patch_ref not found: %s", patch_ref)
        return _OverlayText("", "missing_artifact")
    except OSError as exc:
        log.warning("baseline_executor: cannot read patch_ref %s: %s", patch_ref, exc)
        return _OverlayText("", "artifact_read_failed")


def _unusable_patch_reason(content: str, patch_file: str) -> str:
    from ....specialists.patch_safety import is_unified_diff, patch_escapes_tree

    if not is_unified_diff(content) or "GIT binary patch" in content:
        log.warning("baseline_executor: skipping warm patch %s — not a unified diff", patch_file)
        return "unsafe_or_non_text_diff"
    escape = patch_escapes_tree(content)
    if escape is not None:
        log.warning("baseline_executor: skipping warm patch %s — path escapes tree: %r", patch_file, escape)
        return "path_escapes_tree"
    if not _patch_touched_paths_from_text(content):
        return "missing_touched_paths"
    return ""


def _screened_overlays(patches: list[Any]) -> list[_OverlayText]:
    overlays: list[_OverlayText] = []
    for patch in patches:
        read = _warm_patch_text(patch)
        reason = read.reason or _unusable_patch_reason(read.content, str(patch.get("patch_file") or ""))
        overlays.append(_OverlayText(read.content, reason))
    return overlays


def _refuse_warm_replay_before_apply(
    patches: list[Any],
    roots: list[str],
    trees: dict[str, dict[str, Any]],
    overlays: list[_OverlayText],
) -> dict[str, Any] | None:
    if absent := next((root for root in roots if root and root not in trees), ""):
        return {**_refused_warm_replay(patches, "apply_root_absent"), "pre_sha": "", "target_repo": absent}
    rejected = next((idx for idx, overlay in enumerate(overlays) if overlay.reason), -1)
    if rejected < 0:
        return None
    primary = trees[next(iter(trees))]
    ref, reason = str(patches[rejected].get("patch_file") or ""), overlays[rejected].reason
    return {
        **_refused_warm_replay(patches, reason),
        "patches": [{"patch_ref": ref, "status": "failed", "reason": reason}],
        "failed_ref": ref,
        "pre_sha": primary["pre_sha"],
        "target_repo": primary["root"],
        "rolled_back": False,
    }


def _snapshot_warm_patch_trees(
    trees: dict[str, dict[str, Any]],
    roots: list[str],
    overlays: list[_OverlayText],
    output_dir: Path,
) -> None:
    by_tree: dict[str, list[str]] = {root: [] for root in trees}
    for idx, overlay in enumerate(overlays):
        if not overlay.reason and roots[idx] in trees:
            by_tree[roots[idx]].append(overlay.content)
    for position, root in enumerate(trees):
        tree = trees[root]
        if tree["use_nogit"] or not by_tree[root]:
            continue
        # _create_patch_snapshot owns one fixed sub-directory per output dir and clears it, so trees past the
        # first are given their own.
        tree["snapshot_manifest"] = _create_patch_snapshot(
            root,
            by_tree[root],
            output_dir if position == 0 else output_dir / "warm_patch_trees" / f"{position:02d}",
        )


def _git_apply_overlay(
    target_repo: str,
    patch_path: Path,
    patch_content: str,
    *,
    required_timeline: bool,
) -> str:
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
        return "applied"
    if not required_timeline:
        raise RuntimeError(
            checked.stderr.decode(errors="replace")[:500] if checked.stderr else "git apply --check failed"
        )
    reverse = subprocess.run(
        ["git", "apply", "-R", "--check", str(patch_path)],
        cwd=target_repo,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if reverse.returncode == 0:
        return (
            "already_present"
            if _patch_present_in_committed_head(target_repo, patch_path)
            else "present_in_dirty_worktree"
        )
    touched = _patch_touched_paths_from_text(patch_content)
    before_residue = _three_way_residue_snapshot(target_repo, touched)
    three_way = subprocess.run(
        ["git", "apply", "--3way", str(patch_path)],
        cwd=target_repo,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if three_way.returncode != 0:
        raise RuntimeError(
            three_way.stderr.decode(errors="replace")[:500] if three_way.stderr else "git apply --3way failed"
        )
    clean, residue = _verify_three_way_clean(target_repo, touched, before_residue)
    if not clean:
        raise RuntimeError(residue)
    return "applied_3way"


def _apply_warm_patch_sequence(
    patches: list[Any],
    patch_trees: list[dict[str, Any] | None],
    overlays: list[_OverlayText],
    patch_log_dir: Path,
    *,
    required_timeline: bool,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    applied: list[dict[str, str]] = []
    statuses: list[dict[str, Any]] = []
    for idx, patch in enumerate(patches):
        patch_file = str(patch.get("patch_file") or "")
        status: dict[str, Any] = {"patch_ref": patch_file, "timeline_index": patch.get("timeline_index", idx)}
        statuses.append(status)
        tree = patch_trees[idx]
        if tree is None:
            status.update(status="failed", reason="apply_root_absent")
            continue
        # The patch goes into its own recorded tree, so this is per patch rather than fixed for the whole sequence.
        status["target_repo"] = tree["root"]
        overlay = overlays[idx]
        if overlay.reason:
            status.update(status="failed", reason=overlay.reason)
            if required_timeline:
                break
            continue
        patch_path = patch_log_dir / f"{idx:03d}_{Path(patch_file).stem or 'patch'}.diff"
        patch_path.write_text(overlay.content, encoding="utf-8")
        try:
            if tree["use_nogit"]:
                from .._nogit_patch import _apply_patch_no_git

                backups_so_far = tree["nogit_backups"]
                nogit_apply = _apply_patch_no_git(
                    Path(tree["root"]),
                    patch_path,
                    patch_log_dir / "patch_backups",
                    seq_offset=len(backups_so_far),
                )
                if not nogit_apply.ok:
                    raise RuntimeError(nogit_apply.error or "nogit patch apply failed")
                backups_so_far.extend(nogit_apply.backups)
                method = "already_present" if nogit_apply.present_by else "applied_nogit"
            else:
                method = _git_apply_overlay(
                    tree["root"],
                    patch_path,
                    overlay.content,
                    required_timeline=required_timeline,
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError, RuntimeError) as exc:
            log.warning("baseline_executor: warm patch apply failed for %s: %s", patch_file, exc)
            reason = "nogit_apply_failed" if tree["use_nogit"] else "git_apply_failed"
            status.update(status="failed", reason=reason, detail=str(exc)[:500])
            if required_timeline:
                break
            continue
        if method in ("applied", "applied_3way", "applied_nogit"):
            tree["mutated"] = True
        applied.append({"patch_file": patch_file, "status": method})
        status.update(status=method, applied_patch=str(patch_path))
    return applied, statuses


def _warm_replay_report(
    trees: dict[str, dict[str, Any]],
    applied: list[dict[str, str]],
    statuses: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = trees[next(iter(trees))]
    last = statuses[-1] if statuses else {}
    report = {
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
    if last.get("status") != "failed":
        return report
    # One overlay failing voids the whole replay, and the sequence may already have written to trees before
    # this one, so every tree is restored rather than just the one that failed.
    restore = _revert_warm_patch_trees(trees.values())
    return {
        **report,
        "status": "failed",
        "failed_ref": last["patch_ref"],
        "failure": last["reason"],
        "rolled_back": bool(restore.get("ok")),
        "rollback": restore,
    }


def _apply_warm_patches(
    params: dict[str, Any],
    output_dir: Path,
    *,
    before_mutation: Any = None,
) -> list[dict[str, str]] | dict[str, Any]:
    """Apply warm-replay code patches, each into the checkout it was taken from."""
    if not (patches := params.get("patches")):
        return []
    required = bool(params.get("required_patch_timeline"))
    # Each overlay applies into the tree its gain was measured on.
    roots = [str(patch.get("framework_root") or "").strip() for patch in patches]
    trees = _warm_patch_trees(roots)
    overlays = _screened_overlays(patches)
    if required and (refusal := _refuse_warm_replay_before_apply(patches, roots, trees, overlays)) is not None:
        return refusal
    if not trees:
        return []
    patch_log_dir = output_dir / "warm_patches"
    patch_log_dir.mkdir(parents=True, exist_ok=True)
    try:
        _snapshot_warm_patch_trees(trees, roots, overlays, output_dir)
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        restore = _revert_warm_patch_trees(trees.values())
        if not required:
            log.warning(
                "baseline_executor: skipping legacy warm patches because the "
                "rollback snapshot could not be created: %s",
                exc,
            )
            return []
        primary = trees[next(iter(trees))]
        return {
            **_refused_warm_replay(patches, f"snapshot_failed:{type(exc).__name__}"),
            "pre_sha": primary["pre_sha"],
            "target_repo": primary["root"],
            "trees": _warm_tree_records(trees),
            "rolled_back": bool(restore.get("ok")),
        }
    if any(tree["snapshot_manifest"] for tree in trees.values()):
        params["_warm_patch_trees"] = pending = _warm_tree_records(trees, before_mutation=True)
        if before_mutation is not None and not bool(before_mutation(pending)):
            primary = trees[next(iter(trees))]
            return {
                **_refused_warm_replay(patches, "pending_state_persist_failed"),
                "required": required,
                "pre_sha": primary["pre_sha"],
                "target_repo": primary["root"],
                "snapshot_manifest": primary["snapshot_manifest"],
                "trees": _warm_tree_records(trees),
                "rollback": {"ok": True, "errors": []},
                "rolled_back": True,
            }
    applied, statuses = _apply_warm_patch_sequence(
        patches,
        [trees.get(root) for root in roots],
        overlays,
        patch_log_dir,
        required_timeline=required,
    )
    records = _warm_tree_records(trees)
    if any(tree["snapshot_manifest"] or tree["nogit_backups"] for tree in trees.values()):
        params["_warm_patch_trees"] = records
    # A best-effort timeline reports only the patches that landed, so without
    # this the per-patch reasons computed above would die with this frame --
    # and they are the only record of the ones that did not.
    params["_warm_patch_statuses"] = statuses
    if not required:
        return applied
    return _warm_replay_report(trees, applied, statuses, records)


def _warm_patches_applied(
    patch_application: list[dict[str, str]] | Mapping[str, Any],
) -> list[dict[str, str]]:
    if isinstance(patch_application, Mapping):
        return list(patch_application.get("applied") or [])
    return list(patch_application)


def _stamp_warm_patch_trees(
    result: dict[str, Any],
    patch_application: Mapping[str, Any],
) -> None:
    """Record which trees this round patched, for prelude to promote or restore."""
    trees = list(patch_application.get("trees") or [])
    primary = trees[0] if trees else {}
    result["warm_patch_result"] = dict(patch_application)
    result["warm_patch_trees"] = trees
    result["warm_patch_pre_sha"] = str(primary.get("pre_sha") or "")
    result["warm_patch_target"] = str(primary.get("root") or "")
    result["warm_patch_snapshot_manifest"] = primary.get("snapshot_manifest")


def _stamp_warm_patch_outcome(
    result: dict[str, Any],
    patch_application: list[dict[str, str]] | Mapping[str, Any],
    params: Mapping[str, Any],
) -> None:
    """Carry the patch application's report into the round's result.

    A required timeline reports the structure prelude promotes or restores
    from. A best-effort one reports only the patches that landed, so its
    per-patch statuses are lifted out of ``params`` instead: a patch that
    silently failed to apply would otherwise leave nothing behind, and the
    round would be measured on a tree nobody downstream can describe.
    """
    if applied := _warm_patches_applied(patch_application):
        result["warm_patches_applied"] = applied
    if isinstance(patch_application, Mapping):
        _stamp_warm_patch_trees(result, patch_application)
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
    from .._kernel_agent_tool import _maybe_revert_kernel_patch

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
        except Exception as exc:  # isolation boundary: one applied warm-kernel patch revert
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
