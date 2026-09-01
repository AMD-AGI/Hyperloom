# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Path-scoped patch snapshot / restore / commit primitives for git worktrees.

Public surface
--------------
* :func:`_create_patch_snapshot`  — snapshot patch-touched paths before apply.
* :func:`_restore_patch_snapshot` — restore exactly those paths, verified.
* :func:`_git_commit_kept`        — stage and commit only patch-touched paths.
* :func:`_commit_strip_level`     — the ``-p`` level a forward apply resolved at.
* :func:`_patch_touched_paths`    — touched paths of an already-applied patch set.
* :func:`_patch_touched_paths_from_text` — pre-apply candidates, from header text.

A KEEP commits the paths the apply resolved at; a snapshot, taken before the
level is knowable, covers the candidates for every level it could resolve at, so
the restore is a superset of what the commit would stage. Neither ever widens to
the whole tree, so unrelated working-tree state survives both.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hyperloom.common.git_safety import safe_directory_args

from ...specialists.patch_safety import patch_file_targets
from ._git import _run_git_cp
from ._nogit_patch import _P_LEVELS, _PATCH_DEV_NULL, _strip_path_prefix


def _commit_strip_level(framework_root: Path, pairs: list[tuple[str, str]]) -> int:
    """Pick the ``-p`` strip level resolving the most targets to existing files.

    The patch has already been applied, so modify/create targets exist in the
    tree; the level that maximises those hits is the one the forward apply used.

    Args:
        framework_root: The git checkout the patch was applied into.
        pairs: ``(old_path, new_path)`` header pairs from the patch.

    Returns:
        The ``-p`` strip level resolving the most targets to existing files.
    """
    best_lvl, best_hits = 1, -1
    for lvl in _P_LEVELS:
        hits = 0
        for old, new in pairs:
            for raw in (new, old):
                if not raw or raw == _PATCH_DEV_NULL:
                    continue
                try:
                    if (framework_root / _strip_path_prefix(raw, lvl)).exists():
                        hits += 1
                except OSError:
                    continue
        if hits > best_hits:
            best_hits, best_lvl = hits, lvl
    return best_lvl


def _patch_touched_paths_split(framework_root: Path, patches: list[Path]) -> tuple[list[str], list[str]]:
    """Classify applied patch targets as upserted or deleted.

    Per header pair (``old`` ``---``, ``new`` ``+++``):
      * The ``new`` target exists post-apply → upserted (created/modified).
      * The ``new`` target is ``/dev/null`` or absent post-apply → deleted;
        emit the ``old`` path.
      * A header that resolves to neither is dropped.

    Args:
        framework_root: The git checkout the patches were applied into.
        patches: The applied patch files to inspect.

    Returns:
        ``(upserted, deleted)`` each in first-seen order.
    """
    upserted: list[str] = []
    deleted: list[str] = []
    for patch in patches:
        try:
            text = patch.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        pairs = patch_file_targets(text)
        if not pairs:
            continue
        lvl = _commit_strip_level(framework_root, pairs)
        for old, new in pairs:
            rel_new = _strip_path_prefix(new, lvl) if new and new != _PATCH_DEV_NULL else None
            rel_old = _strip_path_prefix(old, lvl) if old and old != _PATCH_DEV_NULL else None
            try:
                new_exists = bool(rel_new) and (framework_root / rel_new).exists()
            except OSError:
                new_exists = False
            if rel_new and new_exists:
                if rel_new not in upserted:
                    upserted.append(rel_new)
            elif rel_old:
                if rel_old not in deleted:
                    deleted.append(rel_old)
    return upserted, deleted


def _patch_touched_paths(framework_root: Path, patches: list[Path]) -> list[str]:
    """Repo-relative paths to stage, for callers that need no upsert/delete split.

    Args:
        framework_root: The git checkout the patches were applied into.
        patches: The applied patch files to inspect.

    Returns:
        The repo-relative paths, upserted first.
    """
    upserted, deleted = _patch_touched_paths_split(framework_root, patches)
    return list(dict.fromkeys(upserted + deleted))


def _patch_touched_paths_from_text(patch_content: str) -> list[str]:
    """Repo-relative paths a diff's headers may resolve to, before it is applied.

    The strip level is not knowable here — what the patch creates does not exist
    yet, and in a multi-patch set a later patch only resolves once an earlier one
    has applied — so every level in :data:`_P_LEVELS` is emitted, the same ladder
    the apply is driven through. Over-broad is safe: an unused candidate is
    snapshotted absent and restored absent, or restored to the content it
    already has.

    Args:
        patch_content: Raw text of a unified diff.

    Returns:
        Deduplicated repo-relative candidate paths, in first-seen order.
    """
    paths: list[str] = []
    for line in patch_content.splitlines():
        if not line.startswith(("--- ", "+++ ")):
            continue
        raw = line[4:].split("\t", 1)[0].strip()
        if raw in (_PATCH_DEV_NULL, ""):
            continue
        for level in _P_LEVELS:
            path = Path(_strip_path_prefix(raw, level))
            if not path.is_absolute() and ".." not in path.parts:
                paths.append(path.as_posix())
    return list(dict.fromkeys(paths))


def _index_entries(repo_path: str, paths: list[str]) -> dict[str, str]:
    """Map each tracked path to its ``git ls-files -s`` record.

    One call for the whole candidate set, which carries a path per strip level
    and so grows with the depth of the diff headers. ``-z`` leaves paths with
    unusual bytes unquoted, as the restore compares them verbatim.

    Args:
        repo_path: The git worktree root to query.
        paths: Repo-relative candidate paths.

    Returns:
        Tracked path to its ``"<mode> <sha> <stage>\\t<path>"`` record; untracked
        paths are absent.
    """
    result = subprocess.run(
        ["git", *safe_directory_args(["ls-files", "-s", "-z", "--", *paths], cwd=repo_path)],
        cwd=repo_path,
        capture_output=True,
        timeout=30,
        check=True,
    )
    entries: dict[str, str] = {}
    for record in result.stdout.decode(errors="replace").split("\0"):
        if record:
            entries[record.partition("\t")[2]] = record
    return entries


def _create_patch_snapshot(
    repo_path: str,
    patch_contents: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Snapshot the worktree content, mode and index entry of patch-touched paths.

    Covers a candidate per strip level, so the apply cannot resolve to a path the
    restore does not hold. A symlink among them fails the snapshot rather than
    the restore, before anything has been mutated.

    Args:
        repo_path: The git worktree root the patches will be applied into.
        patch_contents: Raw text of each patch, to derive the paths to snapshot.
        output_dir: Directory the snapshot sub-directory is created under.

    Returns:
        The manifest, also written to ``<output_dir>/warm_patch_snapshot/manifest.json``.

    Raises:
        ValueError: When no candidate paths are derivable, or one is a symlink.
        subprocess.CalledProcessError: When ``git ls-files`` fails (not a git tree).
    """
    touched = list(
        dict.fromkeys(path for content in patch_contents for path in _patch_touched_paths_from_text(content))
    )
    if not touched:
        raise ValueError("patch has no touched text paths")
    snapshot_dir = output_dir / "warm_patch_snapshot"
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    root = Path(repo_path).resolve()
    index_entries = _index_entries(repo_path, touched)
    rows: list[dict[str, Any]] = []
    for index, rel in enumerate(touched):
        target = (root / rel).resolve()
        target.relative_to(root)
        if (root / rel).is_symlink():
            raise ValueError(f"patch target must not be a symlink: {rel}")
        backup = snapshot_dir / f"{index:04d}.bin"
        existed = target.is_file() and not target.is_symlink()
        mode = target.stat().st_mode & 0o7777 if existed else None
        if existed:
            backup.write_bytes(target.read_bytes())
        rows.append(
            {
                "path": rel,
                "existed": existed,
                "mode": mode,
                "backup": str(backup) if existed else "",
                "index_entry": index_entries.get(rel, ""),
            }
        )
    manifest_path = snapshot_dir / "manifest.json"
    manifest = {
        "repo_path": str(root),
        "paths": rows,
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _restore_patch_snapshot(manifest: Any) -> dict[str, Any]:
    """Restore exact touched paths/index entries; never reset unrelated work.

    Every path is re-read after restore and compared against its backup, so a
    partial restore is reported rather than mistaken for success.

    Args:
        manifest: The dict from :func:`_create_patch_snapshot`, or a path to the
            ``manifest.json`` it wrote.

    Returns:
        ``{"ok": bool, "errors": list[str]}``; ``ok`` only when every path
        restored and verified.
    """
    if isinstance(manifest, (str, Path)):
        try:
            manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            return {"ok": False, "errors": [f"manifest_read:{exc}"]}
    if not isinstance(manifest, dict):
        return {"ok": False, "errors": ["missing_manifest"]}
    repo = str(manifest.get("repo_path") or "")
    errors: list[str] = []
    for row in manifest.get("paths") or []:
        if not isinstance(row, dict):
            continue
        rel = str(row.get("path") or "")
        target = Path(repo) / rel
        try:
            if row.get("existed"):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(Path(str(row.get("backup") or "")).read_bytes())
                if row.get("mode") is not None:
                    target.chmod(int(row["mode"]))
            elif target.exists() or target.is_symlink():
                target.unlink()
            entry = str(row.get("index_entry") or "").strip()
            if entry:
                metadata, entry_path = entry.split("\t", 1)
                mode, blob, stage = metadata.split()
                if stage != "0" or entry_path != rel:
                    raise ValueError("unsupported pre-existing unmerged index entry")
                subprocess.run(
                    ["git", "update-index", "--cacheinfo", mode, blob, rel],
                    cwd=repo,
                    capture_output=True,
                    timeout=15,
                    check=True,
                )
            else:
                subprocess.run(
                    ["git", "update-index", "--force-remove", "--", rel],
                    cwd=repo,
                    capture_output=True,
                    timeout=15,
                    check=True,
                )
            if row.get("existed"):
                expected = Path(str(row.get("backup") or "")).read_bytes()
                if not target.is_file() or target.read_bytes() != expected:
                    raise OSError("worktree restore verification failed")
            elif target.exists() or target.is_symlink():
                raise OSError("removed path still exists after restore")
            actual_index = _index_entries(repo, [rel]).get(rel, "")
            if actual_index != entry:
                raise OSError("index restore verification failed")
        except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{rel}:{type(exc).__name__}:{exc}")
    return {"ok": not errors, "errors": errors}


def _git_commit_kept(
    framework_root: Path,
    message: str,
    paths: list[str],
) -> tuple[bool, str]:
    """Commit only the patch-touched ``paths`` to git for cross-cycle durability.

    Committing each KEEP makes wins survive a later cycle's revert. The commit is
    scoped to the exact paths the patch touched (never ``git add -A``), so bench
    by-products and concurrent edits stay out of the accepted stack.

    Args:
        framework_root: The git checkout to commit in.
        message: The commit message for the KEEP.
        paths: The repo-relative patch-touched paths to stage and commit.

    Returns:
        A ``(ok, note)`` tuple; ``ok`` is ``True`` on a successful commit or a
        benign no-op (nothing to commit), and ``note`` carries any detail.
    """
    if not paths:
        return True, "no patch-touched paths to commit"
    cp_add = _run_git_cp(["-C", str(framework_root), "add", "-A", "--", *paths], timeout=60.0)
    if cp_add is None:
        return False, "git add spawn failed"
    if cp_add.returncode != 0:
        return False, f"git add failed: {cp_add.stderr.strip()}"
    cp = _run_git_cp(
        [
            "-C",
            str(framework_root),
            "-c",
            "user.email=hyperloom@local",
            "-c",
            "user.name=Hyperloom",
            "commit",
            "-q",
            "-m",
            message,
        ],
        timeout=60.0,
    )
    if cp is None:
        return False, "git commit spawn failed"
    if cp.returncode == 0:
        return True, ""
    if "nothing to commit" in (cp.stdout + cp.stderr).lower():
        return True, "nothing to commit"
    return False, cp.stderr.strip()


def harvest_realized_diff(
    framework_root: Path,
    rel_paths: list[str],
    dest_path: Path,
) -> str:
    """Render what a KEEP actually landed as one canonical ``-p1`` diff.

    The patch a specialist delivered is what it *asked* for; this is what the
    tree ended up holding, which differs whenever the apply resolved at another
    strip level or the KEEP created a file the diff never named. Publishing the
    realized form is what lets a later session replay the change without having
    to re-derive the strip level.

    Read from the KEEP's own commit rather than the working tree: by this point
    :func:`_git_commit_kept` has already committed exactly these paths, so the
    commit *is* the realized change and nothing has to touch the index of a
    checkout other work is still using.

    Args:
        framework_root: The git checkout the KEEP was committed in.
        rel_paths: Repo-relative paths the KEEP touched.
        dest_path: Where to write the diff; written only when non-empty.

    Returns:
        The path written, or ``""`` when there is nothing to harvest, the commit
        has no parent to diff against, or git could not be run.
    """
    paths = [path for path in (str(raw or "").strip() for raw in rel_paths) if path]
    if not paths:
        return ""
    cp = _run_git_cp(
        ["-C", str(framework_root), "diff", "HEAD^", "HEAD", "--", *paths],
        timeout=120.0,
    )
    if cp is None or cp.returncode != 0:
        return ""
    text = cp.stdout or ""
    if not text.strip():
        return ""
    try:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(text, encoding="utf-8")
    except OSError:
        return ""
    return str(dest_path)


__all__ = [
    "_commit_strip_level",
    "_create_patch_snapshot",
    "_git_commit_kept",
    "_patch_touched_paths",
    "_patch_touched_paths_from_text",
    "_patch_touched_paths_split",
    "_restore_patch_snapshot",
    "harvest_realized_diff",
]
