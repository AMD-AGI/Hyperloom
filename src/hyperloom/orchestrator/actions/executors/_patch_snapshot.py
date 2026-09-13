# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Path-scoped patch snapshot / restore / commit primitives for git worktrees."""

from __future__ import annotations

import json
import os
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


#: Git's own identity for the throwaway replay commits; the source repository's
#: config is never read or written.
_REPLAY_IDENTITY: tuple[str, ...] = ("-c", "user.email=replay@hyperloom.invalid", "-c", "user.name=hyperloom")

#: Tree entry modes the snapshot contract cannot represent. A symlink is
#: captured by ``shutil.copy2``, which follows it and writes a REGULAR FILE, so
#: a recipe certified over one restores the wrong kind of entry; a gitlink names
#: a submodule whose checked-out content no archive or apply reconstructs.
#: Neither can be shipped honestly, so a stack touching one is refused.
_UNREPRESENTABLE_MODES: frozenset[str] = frozenset({"120000", "160000"})


def _git(tree: Path, *args: str, timeout: int = 300) -> subprocess.CompletedProcess[bytes] | None:
    """Run git inside ``tree``; ``None`` when it could not be run at all."""
    try:
        return subprocess.run(["git", *args], cwd=str(tree), capture_output=True, timeout=timeout, check=False)
    except (OSError, subprocess.SubprocessError):
        return None


def _repo_prefix(framework_root: Path) -> tuple[Path, str] | None:
    """Return ``(toplevel, prefix)`` for a root that may sit inside a repo.

    An apply root is allowed to name a directory INSIDE a worktree -- the
    executor's own resolver accepts one -- and ``git clone`` of such a
    directory clones nothing. The enclosing repository is what carries the
    base commit; the prefix is where the stack actually lands inside it.
    """
    top = _git(framework_root, "rev-parse", "--show-toplevel", timeout=60)
    pre = _git(framework_root, "rev-parse", "--show-prefix", timeout=60)
    if top is None or pre is None or top.returncode != 0 or pre.returncode != 0:
        return None
    toplevel = top.stdout.decode("utf-8", errors="replace").strip()
    prefix = pre.stdout.decode("utf-8", errors="replace").strip()
    return (Path(toplevel), prefix) if toplevel else None


def _checkout_base_tree(framework_root: Path, base_sha: str, dest: Path) -> bool:
    """Check ``base_sha`` out into ``dest``, byte for byte.

    A real checkout rather than ``git archive``: archive honours
    ``export-ignore`` and ``export-subst``, so it can drop a tracked file the
    stack patches, or substitute a ``$Format:...$`` placeholder the recorded
    commit actually contains. Either makes the replay compare against bytes a
    consumer checking out that commit would never see -- once refusing a valid
    stack, once accepting an invalid one.

    ``--shared`` reads the source object database through an alternate and
    ``-n`` skips the checkout until the detached one below, so the captured
    repository is never written to.
    """
    clone = _git(Path("."), "clone", "--shared", "-n", "-q", str(framework_root), str(dest), timeout=600)
    if clone is None or clone.returncode != 0:
        return False
    # BEFORE the checkout, not after. ``$GIT_DIR/info/attributes`` outranks a
    # tracked ``.gitattributes``, and this turns off every transformation that
    # can sit between the object database and the working tree -- in BOTH
    # directions, which is why the order matters:
    #
    #   * on the way out, a ``smudge`` filter or an encoding declared by the
    #     base itself rewrites the checked-out file. Installed afterwards, the
    #     override cannot undo that, and the first replay commit then records
    #     the smudged bytes as an effect of ITS patch -- content the recorded
    #     base and patch do not produce for anyone without that filter
    #     configured. The preimage has to be the commit, not the commit as this
    #     host renders it.
    #   * on the way in, ``git add`` converts eol, runs ``clean`` filters,
    #     expands ``$Id$`` and applies ``working-tree-encoding``. Any of those
    #     can stage a genuinely changed file back to the blob it started from,
    #     so the commit shows no change, the inventory omits the file, and
    #     nothing compares or ships it.
    #
    # ``working-tree-encoding`` is listed explicitly: it is not implied by
    # ``-text`` and converts independently of it.
    try:
        info = dest / ".git" / "info"
        info.mkdir(parents=True, exist_ok=True)
        (info / "attributes").write_text(
            "* -text -eol -filter -diff -ident -merge -working-tree-encoding\n", encoding="utf-8"
        )
    except OSError:
        return False
    done = _git(dest, "checkout", "-q", "--detach", base_sha, timeout=600)
    return done is not None and done.returncode == 0


def _commit_replay_step(tree: Path) -> bool:
    """Commit whatever the last apply left, so its inventory can be read back.

    ``--force``, because ``git add -A`` skips a newly created file the base's
    ``.gitignore`` matches. A patch creating one applies fine and the replay
    creates it too, but an unforced stage drops it from the commit -- which
    would move the completeness gap from the diff parser to the index instead
    of closing it.
    """
    added = _git(tree, "add", "-A", "--force")
    if added is None or added.returncode != 0:
        return False
    done = _git(tree, *_REPLAY_IDENTITY, "commit", "--allow-empty", "-q", "-m", "replay step")
    return done is not None and done.returncode == 0


def _step_inventory(tree: Path, *, prefix: str = "") -> dict[str, str] | None:
    """What the last replay commit changed, as git itself reports it.

    Read from ``git diff --raw`` rather than parsed out of the diff text.
    ``patch_file_targets`` cannot tell a file header from a hunk body, so a
    removed line beginning ``-- `` followed by an added line beginning ``++ ``
    reads as another header pair -- which is enough to make a block count agree
    while a mode-only or binary block goes undeclared. Git knows exactly which
    entries the apply touched, and their modes, so nothing is inferred here.

    Returns ``None`` when a touched entry is of a kind the snapshot cannot
    represent, which the caller treats as an unverifiable stack.
    """
    done = _git(tree, "diff", "--raw", "--no-renames", "-z", "HEAD~1", "HEAD")
    if done is None or done.returncode != 0:
        return None
    # Strict, not ``errors="replace"``: git paths are byte strings, and
    # replacing an undecodable byte can turn one real filename into another
    # real filename -- verification then checks an unrelated file that happens
    # to exist, and capture ships it. The durable schema is UTF-8, so a name it
    # cannot carry is refused rather than approximated.
    try:
        raw = done.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None
    fields = raw.split("\0")
    ops: dict[str, str] = {}
    index = 0
    while index + 1 < len(fields):
        meta, rel = fields[index], fields[index + 1]
        index += 2
        if not meta.startswith(":") or not rel:
            continue
        parts = meta[1:].split()
        if len(parts) < 5:
            return None
        src_mode, dst_mode, status = parts[0], parts[1], parts[4]
        if src_mode in _UNREPRESENTABLE_MODES or dst_mode in _UNREPRESENTABLE_MODES:
            return None
        if prefix:
            if not rel.startswith(prefix):
                # The stack reached outside the apply root, which no snapshot of
                # that root can carry.
                return None
            rel = rel[len(prefix) :]
        ops[rel] = "delete" if status.startswith("D") else "upsert"
    return ops


def _apply_at_some_level(tree: Path, patch: Path) -> bool:
    """Apply ``patch`` inside ``tree`` at the first level that takes it."""
    for level in _P_LEVELS:
        check = _git(tree, "apply", "--check", f"-p{level}", str(patch), timeout=60)
        if check is None or check.returncode != 0:
            continue
        done = _git(tree, "apply", f"-p{level}", str(patch), timeout=60)
        if done is not None and done.returncode == 0:
            return True
    return False


def _same_entry(replayed: Path, captured: Path, *, op: str) -> bool:
    """Whether the replayed entry and the captured one are the same thing.

    Type-aware on purpose. ``is_file``/``read_bytes``/``stat`` all follow
    symlinks, so comparing through them establishes neither the kind of entry
    nor a link's target; and treating "neither is a regular file" as agreement
    lets a populated directory stand in for a deletion.
    """
    if op == "delete":
        # Absence, not merely "not a regular file": a directory or a dangling
        # link left where the stack deleted a path is not a reproduced deletion.
        return not os.path.lexists(replayed) and not os.path.lexists(captured)
    if replayed.is_symlink() or captured.is_symlink():
        return False
    try:
        if not (replayed.is_file() and captured.is_file()):
            return False
        if replayed.read_bytes() != captured.read_bytes():
            return False
        # The OWNER execute bit, which is the one git records as 100755. Asking
        # whether ANY execute bit is set accepts 0645, which git classifies as
        # non-executable and whose owner cannot run it.
        return (replayed.stat().st_mode & 0o100) == (captured.stat().st_mode & 0o100)
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
    ``base_sha`` plus these patches, in this order, produce the entries the
    capture is about to ship? Everything cheaper answers a different question
    and gets it wrong in both directions.

    Reading the diff headers says what a patch DECLARES and nothing about the
    tree. Reverse-applying each patch against the FINAL tree is worse than it
    looks: git locates a postimage with an offset, so a patch that was never
    applied reverses against a similar block elsewhere in the file; a deletion
    reverses by creating a file, whatever it writes and at whatever level; a
    mode disagreement is only a warning; and it is simultaneously too strict,
    because an earlier round's patch cannot reverse once a later round has
    rewritten the same region, and a patch editing a file an earlier round
    created has no preimage in the base at all.

    Replaying forward has none of those problems: every patch meets exactly the
    tree it was authored against, the inventory comes from git rather than from
    a text parse, and the final comparison covers existence, kind, bytes and
    the executable bit for every entry the stack touched.

    Args:
        framework_root: The captured tree, compared against the replay.
        patches: The accepted stack's patches for this root, in apply order.
        base_sha: The commit the stack applies to.

    Returns:
        ``{patch_path: {rel: op}}`` when the replay reproduces every touched
        entry, else ``None`` -- which the caller must treat as an undeclared
        step rather than a satisfied one.
    """
    if not base_sha or not patches:
        return None
    located = _repo_prefix(framework_root)
    if located is None:
        return None
    toplevel, prefix = located
    tmp = Path(tempfile.mkdtemp(prefix="hl-replay-"))
    try:
        if not _checkout_base_tree(toplevel, base_sha, tmp):
            return None
        # ``git apply`` resolves a patch's paths against the working directory,
        # so the stack is applied where it actually landed inside the repo.
        work_tree = tmp / prefix if prefix else tmp
        by_patch: dict[str, dict[str, str]] = {}
        # Folded in apply order: a step's own operation is an INTERMEDIATE
        # state, so comparing each of them against the final tree refuses a
        # round that deletes what a later round recreates -- the same mistake
        # the decision side already had to unlearn. The per-step inventories
        # still travel, because the recipe emits one step per patch.
        end_state: dict[str, str] = {}
        for patch in patches:
            # Git records no empty directory, so the apply root may be absent at
            # base -- an untracked directory the first patch populates -- and a
            # deletion can remove it again when it takes the last file under it.
            # Either way the next step's git invocations would run with a
            # missing cwd.
            try:
                work_tree.mkdir(parents=True, exist_ok=True)
            except OSError:
                return None
            if not _apply_at_some_level(work_tree, patch) or not _commit_replay_step(tmp):
                return None
            ops = _step_inventory(tmp, prefix=prefix)
            if ops is None:
                return None
            by_patch[str(patch)] = ops
            end_state.update(ops)
        for rel, op in end_state.items():
            if not _same_entry(work_tree / rel, framework_root / rel, op=op):
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
