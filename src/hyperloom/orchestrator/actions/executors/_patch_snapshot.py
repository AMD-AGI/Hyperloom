# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Path-scoped patch snapshot / restore / commit primitives for git worktrees."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Sequence

from hyperloom.common.git_safety import safe_directory_args

from ...specialists.patch_safety import patch_file_targets
from ._git import _run_git_cp
from ._nogit_patch import _P_LEVELS, _PATCH_DEV_NULL, _strip_path_prefix


def _commit_strip_level(framework_root: Path, pairs: list[tuple[str, str]]) -> int:
    """Pick the ``-p`` strip level resolving the most targets to existing files."""
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
    """Classify applied patch targets as upserted or deleted."""
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


def patch_declared_ops(framework_root: Path, patches: list[Path]) -> dict[str, str]:
    """Return ``{rel: "upsert" | "delete"}`` as the patches themselves declare it.

    The operation comes from the diff headers -- a ``/dev/null`` post-image is a
    deletion, any other post-image is an upsert, and a rename declares both --
    and never from whether the path happens to exist in the tree at capture
    time. :func:`_patch_touched_paths_split` asks the tree instead, which is the
    wrong question twice over for the accepted-stack capture:

    * A KEEP reached with its mutation inputs stripped still finds the *base*
      file present, so a tree probe declares a satisfied upsert over content
      that contains none of the stack's changes -- the exact case
      ``declared_targets`` exists to refuse.
    * Across a multi-round stack the tree only shows the final state, and the
      two accumulated lists are merged with deletions applied last. A file an
      early round deleted and a later round recreated therefore lands in both
      lists and is declared ``delete``, so the replay removes a file the
      accepted stack requires.

    Ordering is the caller's: later patches override earlier ones for the same
    path, which is the order ``kept_patches`` records and the order a consumer
    replays them in.

    Args:
        framework_root: Tree the patches were bound to; read only to choose the
            ``-p`` strip level.
        patches: The accepted stack's patch files for that root, in apply order.

    Returns:
        The declared operation per repo-relative path. Unreadable patches are
        skipped, which leaves their targets undeclared and therefore uncaptured,
        and the decision refuses the recipe rather than certifying a gap.
    """
    ops: dict[str, str] = {}
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
            if rel_new:
                ops[rel_new] = "upsert"
                # A rename declares its source gone; a plain modify has old == new
                # and must not declare a deletion of the file it just wrote.
                if rel_old and rel_old != rel_new:
                    ops[rel_old] = "delete"
            elif rel_old:
                ops[rel_old] = "delete"
    return ops


_GIT_DIFF_BLOCK_RE = re.compile(r"^diff --git ", re.MULTILINE)


def _declares_every_block(text: str, pairs: list[tuple[str, str]]) -> bool:
    """Whether every file block a git diff announces reached ``pairs``.

    ``patch_file_targets`` reads adjacent ``---``/``+++`` lines, which a pure
    rename, a mode-only change and a ``GIT binary patch`` block need not carry.
    A patch made only of those declares nothing and is refused for it; one that
    ALSO carries an ordinary text hunk declares a non-empty but partial map, and
    both the per-patch map and the accepted stack then omit the same files -- so
    nothing downstream can see the omission. Counting the announced blocks is
    what makes the partial case indistinguishable from the empty one.
    """
    blocks = len(_GIT_DIFF_BLOCK_RE.findall(text))
    return blocks == 0 or blocks == len(pairs)


def _extract_base_tree(framework_root: Path, base_sha: str, dest: Path) -> bool:
    """Materialise ``base_sha``'s tree into ``dest``. Never mutates the source."""
    try:
        archive = subprocess.run(
            ["git", *safe_directory_args(["archive", base_sha], cwd=str(framework_root))],
            cwd=str(framework_root),
            capture_output=True,
            timeout=300,
            check=False,
        )
        if archive.returncode != 0:
            return False
        done = subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, capture_output=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _apply_at_some_level(tree: Path, patch: Path) -> int | None:
    """Apply ``patch`` inside ``tree`` at the first level that takes it."""
    for level in _P_LEVELS:
        try:
            check = subprocess.run(
                ["git", "apply", "--check", f"-p{level}", str(patch)],
                cwd=str(tree),
                capture_output=True,
                timeout=60,
                check=False,
            )
            if check.returncode != 0:
                continue
            done = subprocess.run(
                ["git", "apply", f"-p{level}", str(patch)],
                cwd=str(tree),
                capture_output=True,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if done.returncode == 0:
            return level
    return None


def _ops_at_level(text: str, level: int) -> dict[str, str]:
    """The ops a diff declares once the strip level is known to be right."""
    ops: dict[str, str] = {}
    for old, new in patch_file_targets(text):
        rel_new = _strip_path_prefix(new, level) if new and new != _PATCH_DEV_NULL else None
        rel_old = _strip_path_prefix(old, level) if old and old != _PATCH_DEV_NULL else None
        if rel_new:
            ops[rel_new] = "upsert"
            if rel_old and rel_old != rel_new:
                ops[rel_old] = "delete"
        elif rel_old:
            ops[rel_old] = "delete"
    return ops


def _same_file(replayed: Path, captured: Path) -> bool:
    """Whether two paths hold the same bytes AND the same executability."""
    try:
        if replayed.is_file() != captured.is_file():
            return False
        if not replayed.is_file():
            return True
        if replayed.read_bytes() != captured.read_bytes():
            return False
        # A recipe that restores a script without its execute bit does not
        # reproduce the accepted stack, and git treats a mode disagreement as a
        # warning rather than a failure, so it has to be compared here.
        return bool(replayed.stat().st_mode & 0o111) == bool(captured.stat().st_mode & 0o111)
    except OSError:
        return False


def replayed_stack_ops(
    framework_root: Path,
    patches: Sequence[Path],
    *,
    base_sha: str,
) -> dict[str, dict[str, str]] | None:
    """Replay the ordered stack from ``base_sha`` and prove it rebuilds the tree.

    This is the only question worth asking of a replay contract: does
    ``base_sha`` plus these patches, in this order, produce the files the
    capture is about to ship? Everything cheaper answers a different question
    and gets it wrong in both directions.

    Reading the diff headers says what a patch DECLARES and nothing about the
    tree. Reverse-applying each patch against the FINAL tree is worse than it
    looks: git searches for the postimage with an offset, so a patch that was
    never applied can reverse against a similar block elsewhere in the file; a
    deletion reverses by creating a file, which succeeds whatever it writes; a
    mode disagreement is only a warning; and it is simultaneously too strict,
    because an earlier round's patch cannot reverse once a later round has
    rewritten the same region, and because a patch that edits a file an earlier
    round created has no preimage in the stack's base at all.

    Replaying forward has none of those problems, because every patch meets
    exactly the tree it was authored against: the level that applies is the
    level that was used, the ops it declares are then trustworthy, and the
    final comparison covers content and mode for every declared path.

    Args:
        framework_root: The captured tree, compared against the replay.
        patches: The accepted stack's patches for this root, in apply order.
        base_sha: The commit the stack applies to.

    Returns:
        ``{patch_path: {rel: op}}`` when the replay reproduces every declared
        path, else ``None`` -- which the caller must treat as an undeclared
        step rather than a satisfied one.
    """
    # No explicit git-tree probe: ``git archive`` below fails on a root that is
    # not one, which is the same answer for one fewer subprocess.
    if not base_sha or not patches:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="hl-replay-"))
    try:
        if not _extract_base_tree(framework_root, base_sha, tmp):
            return None
        by_patch: dict[str, dict[str, str]] = {}
        for patch in patches:
            try:
                text = patch.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
            pairs = patch_file_targets(text)
            if not pairs or not _declares_every_block(text, pairs):
                return None
            level = _apply_at_some_level(tmp, patch)
            if level is None:
                return None
            by_patch[str(patch)] = _ops_at_level(text, level)
        for ops in by_patch.values():
            for rel in ops:
                if not _same_file(tmp / rel, framework_root / rel):
                    return None
        return by_patch
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _patch_touched_paths(framework_root: Path, patches: list[Path]) -> list[str]:
    """Repo-relative paths to stage, for callers that need no upsert/delete split."""
    upserted, deleted = _patch_touched_paths_split(framework_root, patches)
    return list(dict.fromkeys(upserted + deleted))


def _patch_touched_paths_from_text(patch_content: str) -> list[str]:
    """Repo-relative paths a diff's headers may resolve to, before it is applied."""
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
    """Map each tracked path to its ``git ls-files -s`` record."""
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
    """Snapshot the worktree content, mode and index entry of patch-touched paths."""
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
    """Restore exact touched paths/index entries; never reset unrelated work."""
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
    """Commit only the patch-touched ``paths`` to git for cross-cycle durability."""
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
    """Render what a KEEP actually landed as one canonical ``-p1`` diff."""
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
    "patch_declared_ops",
    "replayed_stack_ops",
    "harvest_realized_diff",
]
