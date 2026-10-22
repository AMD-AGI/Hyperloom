# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Moving the worktree a patch is applied in: applying it, reversing it, and the stash that protects what was already there."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...delivery.ledger import load_prepared_records, restore_records
from ...specialists.patch_safety import is_unified_diff, patch_escapes_tree
from ._apply_feedback import ApplyFeedback, build_apply_feedback
from ._git import _git_head_sha, _run_git_cp
from ._integrate_attempt import IntegrateAttempt
from ._nogit_patch import _P_LEVELS, _apply_patch_no_git, _is_git_tree

log = logging.getLogger(__name__)


_HYPERLOOM_AUTO_STASH_MSG = "hyperloom-auto-stash: preserving user changes before candidate run"

# Deliberately shares no substring with the auto-stash tag: _find_hyperloom_auto_stash
# matches by message, and a quarantined merge must never be picked up and popped back.
_HYPERLOOM_QUARANTINE_STASH_MSG = "hyperloom-quarantine: unresolved merge cleared before candidate run"


def _detect_p_level(framework_root: Path, patch_path: Path) -> int | None:
    """Return the first ``-p`` level whose ``--check`` applies cleanly.

    Args:
        framework_root: The git checkout to test against.
        patch_path: The patch file to probe.

    Returns:
        The first ``-p<N>`` level that applies cleanly, or ``None`` when none
        do.
    """
    for lvl in _P_LEVELS:
        ok, _ = _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=True)
        if ok:
            return lvl
    return None


def _run_git_apply(
    framework_root: Path,
    patch_path: Path,
    *,
    p_level: int,
    check_only: bool,
) -> tuple[bool, str]:
    """Single ``git apply`` invocation at an explicit strip level.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        p_level: The ``-p<N>`` strip level.
        check_only: Whether to pass ``--check`` (dry run, no mutation).

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True on a zero return code.
    """
    args = ["-C", str(framework_root), "apply", f"-p{p_level}"]
    if check_only:
        args.append("--check")
    args.append(str(patch_path))
    cp = _run_git_cp(args, timeout=120.0)
    if cp is None:
        return False, "git apply spawn failed"
    return cp.returncode == 0, cp.stderr.strip()


def _git_apply(
    framework_root: Path,
    patch_path: Path,
    *,
    check_only: bool = False,
) -> tuple[bool, str]:
    """Run ``git apply -p<auto> [--check] <patch>`` inside
    ``framework_root``, auto-detecting the strip level. Returns
    ``(ok, stderr)``.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.
        check_only: Whether to only check (dry run) rather than mutate.

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True when the apply (or check)
        succeeds.
    """
    lvl = _detect_p_level(framework_root, patch_path)
    if lvl is None:
        # Surface a representative error at the git-native default level.
        return _run_git_apply(framework_root, patch_path, p_level=1, check_only=check_only)
    if check_only:
        return True, ""
    return _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=False)


def _git_reverse_applies_cleanly(framework_root: Path, patch_path: Path) -> bool:
    """True when ``patch_path`` is already fully applied in ``framework_root``.

    The git-channel twin of
    :func:`._nogit_patch._reverse_applies_cleanly`. ``git apply -R --check``
    succeeds only when every hunk's *post*-state is already present, i.e. the
    tree already equals what a forward apply would produce. Read-only:
    ``--check`` never mutates the tree.

    Args:
        framework_root: The git checkout the patch targets.
        patch_path: The patch file to probe.

    Returns:
        ``True`` when some strip level reverse-checks cleanly.
    """
    from ._nogit_patch import _P_LEVELS

    for lvl in _P_LEVELS:
        cp = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", "--check", str(patch_path)],
            timeout=120.0,
        )
        if cp is not None and cp.returncode == 0:
            return True
    return False


def _git_apply_collect_feedback(
    framework_root: Path,
    patch_path: Path,
) -> "tuple[bool, str, ApplyFeedback | None]":
    """Like :func:`_git_apply` but also returns an :class:`ApplyFeedback` on failure.

    On success returns ``(True, "", None)``.  On failure returns
    ``(False, stderr, ApplyFeedback)`` where *ApplyFeedback* carries the
    stderr of every tried ``-p`` level and a source-context snippet.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.

    Returns:
        ``(ok, err, feedback)`` — feedback is ``None`` on success.
    """
    from ._nogit_patch import _P_LEVELS

    # Collect per-level check stderr for the feedback record.
    tried_levels: list[int] = []
    level_stderrs: list[str] = []
    for lvl in _P_LEVELS:
        ok_check, stderr_check = _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=True)
        tried_levels.append(lvl)
        if stderr_check:
            level_stderrs.append(f"-p{lvl}: {stderr_check}")
        if ok_check:
            # Level works; now apply for real.
            ok_apply, stderr_apply = _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=False)
            if ok_apply:
                return True, "", None
            feedback = build_apply_feedback(
                patch_path,
                channel="git",
                tried_levels=tried_levels,
                stderr=stderr_apply,
                framework_root=framework_root,
            )
            return False, stderr_apply, feedback

    # A subset of a patch already applied is a no-op that ``--check`` still
    # rejects; a clean reverse check proves every hunk is in the tree.
    if _git_reverse_applies_cleanly(framework_root, patch_path):
        log.info(
            "integrate_patch: %s is already fully applied (clean git apply -R --check); treating as a no-op",
            patch_path.name,
        )
        return True, "", None
    all_stderrs = "\n".join(level_stderrs)
    feedback = build_apply_feedback(
        patch_path,
        channel="git",
        tried_levels=tried_levels,
        stderr=all_stderrs,
        framework_root=framework_root,
    )
    return False, all_stderrs, feedback


def _git_apply_reverse(
    framework_root: Path,
    patch_path: Path,
) -> tuple[bool, str]:
    """Reverse-apply ``patch_path`` (``git apply -R -p<auto>``) as the REVERT
    path; caller falls back to ``git checkout`` on failure. Auto-detects the
    same strip level the forward apply used via ``-R --check``.

    Args:
        framework_root: The git checkout to reverse-apply into.
        patch_path: The patch file to reverse-apply.

    Returns:
        A ``(ok, stderr)`` tuple; ``ok`` is True when the reverse apply
        succeeds.
    """
    for lvl in _P_LEVELS:
        cp = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", "--check", str(patch_path)],
            timeout=120.0,
        )
        if cp is None:
            return False, "git apply -R spawn failed"
        if cp.returncode != 0:
            continue
        cp2 = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", str(patch_path)],
            timeout=120.0,
        )
        if cp2 is None:
            return False, "git apply -R spawn failed"
        if cp2.returncode == 0:
            return True, ""
        return False, cp2.stderr.strip()
    return False, f"git apply -R: no matching -p level for {patch_path}"


def _apply_patch_series(
    attempt: IntegrateAttempt,
    patch_paths: list[Path],
    *,
    git_head: str,
    backup_root: Path,
) -> tuple[list[dict[str, str]], tuple[ApplyFeedback, ...]]:
    """Apply each delivered patch to the attempt's tree, stopping at the first one that does not land.

    ``attempt.applied`` is appended to in place, so a cancellation mid-series still
    leaves the undo able to see what reached the tree.
    """
    framework_root = attempt.framework_root
    errors: list[dict[str, str]] = []
    feedback: list[ApplyFeedback] = []
    nogit_backups: list[dict[str, Any]] = []
    for patch in patch_paths:
        # ``vet_patches`` runs at authoring time inside the specialist runner, so a
        # patch from anywhere else -- ``params.patches``, every remotely-fetched
        # ``upstream_pr`` diff -- arrives unvetted.
        try:
            text = patch.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            errors.append({"patch": str(patch), "stderr": f"unreadable: {exc!r}"})
            break
        if not is_unified_diff(text):
            errors.append({"patch": str(patch), "stderr": "not a unified diff"})
            break
        escaping = patch_escapes_tree(text)
        if escaping is not None:
            errors.append({"patch": str(patch), "stderr": f"path escapes tree: {escaping!r}"})
            break
        if git_head:
            ok, err, fb = _git_apply_collect_feedback(framework_root, patch)
        else:
            ok, err, taken, fb = _apply_patch_no_git(framework_root, patch, backup_root, seq_offset=len(nogit_backups))
            nogit_backups.extend(taken)
        if not ok:
            errors.append({"patch": str(patch), "stderr": err})
            if fb is not None:
                feedback.append(fb)
            break
        attempt.applied.append(patch)
    return errors, tuple(feedback)


def _uncommitted_revert(attempt: IntegrateAttempt, failure: str, what: str, **evidence: Any) -> dict[str, Any]:
    """Revert accepted work that could not be committed, since the next revert to HEAD would erase it unseen."""
    log.error("integrate_patch: %s could not be committed (%s); reverting", what, failure)
    return {
        "status": "reverted",
        "error_class": "keep_commit_failed",
        "error": failure,
        "specialist_task_id": attempt.specialist_task_id,
        "patches_applied": [],
        "patches_reverted": [],
        "artifacts_reverted": [],
        "reason": f"{what} could not be committed: {failure}",
        "workspace": str(attempt.output_root),
        **evidence,
    }


def _find_hyperloom_auto_stash(framework_root: Path) -> str:
    """Return the newest Hyperloom auto-stash ref, or ``""`` if absent."""
    cp = _run_git_cp(
        ["-C", str(framework_root), "stash", "list", "--format=%gd:%gs"],
        timeout=30.0,
    )
    if cp is None:
        return ""
    if cp.returncode != 0:
        return ""
    for line in cp.stdout.splitlines():
        ref, _sep, msg = line.partition(":")
        if ref and _HYPERLOOM_AUTO_STASH_MSG in msg:
            return ref
    return ""


def _git_quarantine_unmerged(framework_root: Path, paths: list[str]) -> tuple[bool, str]:
    """Bank unresolved-merge paths in a stash entry that is never popped back.

    ``git stash`` refuses to run at all while the index carries unmerged
    entries, so they are staged first — staging is what marks a conflict
    resolved — and the working-tree content, conflict markers and all, is what
    gets banked. Recover it with ``git stash list | grep hyperloom-quarantine``.

    Emptying the index is not the same as ending the merge: ``MERGE_HEAD``
    outlives both the staging and the stash, and while it stands the next
    ``git commit`` is a *merge* commit. A KEEP would then carry a second parent
    and silently claim the whole of the other side as accepted work, which is
    the one thing "every KEEP is a commit, so HEAD is the accepted stack"
    cannot survive. ``git merge --quit`` forgets the merge without touching the
    index or the working tree.

    Args:
        framework_root (Path): The framework repo.
        paths (list[str]): Repo-relative paths in an unresolved merge state.

    Returns:
        tuple[bool, str]: ``(ok, error)``; ``error`` is empty on success.
    """
    add = _run_git_cp(["-C", str(framework_root), "add", "--", *paths], timeout=60.0)
    if add is None:
        return False, "git add failed"
    if add.returncode != 0:
        return False, f"git add rc={add.returncode}: {add.stderr.strip()}"
    cp = _run_git_cp(
        [
            "-C",
            str(framework_root),
            "stash",
            "push",
            "-m",
            _HYPERLOOM_QUARANTINE_STASH_MSG,
            "--",
            *paths,
        ],
        timeout=60.0,
    )
    if cp is None:
        return False, "git stash push failed"
    if cp.returncode != 0:
        return False, f"git stash push rc={cp.returncode}: {cp.stderr.strip()}"
    quit_cp = _run_git_cp(["-C", str(framework_root), "merge", "--quit"], timeout=30.0)
    if quit_cp is None or quit_cp.returncode != 0:
        # Nothing was banked that a later revert cannot reach, but leaving
        # MERGE_HEAD standing would mislabel the next KEEP, so refuse rather
        # than proceed on a repo that is still mid-merge.
        detail = "git merge --quit failed" if quit_cp is None else quit_cp.stderr.strip()
        return False, f"merge state could not be cleared: {detail}"
    return True, ""


def _git_unmerged_paths(framework_root: Path) -> list[str]:
    """Return repo-relative paths left in an unresolved merge state, if any."""
    cp = _run_git_cp(["-C", str(framework_root), "ls-files", "-u", "--full-name"], timeout=30.0)
    if cp is None or cp.returncode != 0:
        return []
    seen: list[str] = []
    for line in (cp.stdout or "").splitlines():
        _meta, _sep, path = line.partition("\t")
        path = path.strip()
        if path and path not in seen:
            seen.append(path)
    return seen


def _git_restore_to_head(framework_root: Path, paths: list[str] | None = None) -> tuple[bool, str]:
    """Force ``paths`` (or the whole tree) back to HEAD, clearing any merge state.

    Every KEEP is committed, so HEAD is the accepted stack: restoring to it drops
    candidate work and keeps everything the loop has accepted.

    Args:
        framework_root (Path): The git checkout to restore.
        paths (list[str] | None): Repo-relative paths, or ``None`` for the tree.

    Returns:
        tuple[bool, str]: ``(ok, stderr)``.
    """
    target = paths if paths else ["."]
    # `checkout --force HEAD --` resolves an unmerged index, which plain
    # `checkout -- .` refuses to touch.
    cp = _run_git_cp(
        ["-C", str(framework_root), "checkout", "--force", "HEAD", "--", *target],
        timeout=60.0,
    )
    if cp is None:
        return False, "git checkout spawn failed"
    if cp.returncode != 0:
        return False, (cp.stderr or "").strip()
    return True, ""


def _git_stash_if_dirty(framework_root: Path) -> tuple[str, str]:
    """Stash uncommitted user changes so destructive resets don't lose them.

    Only stashes when the working tree is dirty (``git status --porcelain``
    is non-empty). The stash message is tagged for easy retrieval via
    ``git stash list | grep hyperloom-auto-stash``.

    Returns:
        ``(state, note)`` where ``state`` is one of:

        - ``"clean"`` — working tree was already clean, safe to proceed.
        - ``"stashed"`` — dirty tree was successfully stashed; ``note`` is the
          stash ref to restore when the candidate finishes.
        - ``"failed"`` — tree is dirty but stash command failed; callers
          MUST NOT proceed with destructive operations.
    """
    cp = _run_git_cp(["-C", str(framework_root), "status", "--porcelain"], timeout=30.0)
    if cp is None:
        return "failed", "git status check failed"
    if cp.returncode != 0:
        # Non-git directory or other git status errors: treat as clean.
        log.debug(
            "integrate_patch: git status rc=%d in %s (not a git repo?), treating as clean",
            cp.returncode,
            framework_root,
        )
        return "clean", ""
    if not cp.stdout.strip():
        return "clean", ""
    # git refuses to stash while an unresolved merge stands, so every later
    # candidate would abort here forever. Clear it — but bank the content
    # instead of overwriting it. Usually this is wreckage from an earlier cycle;
    # nothing here can prove that, and a merge the operator started themselves
    # is not ours to throw away. Quarantine keeps both cases recoverable at the
    # cost of one stash entry. It is deliberately not the auto-stash tag, so it
    # is never popped back: restoring conflict markers into the source is what
    # made every benchmark fail to parse the model.
    unmerged = _git_unmerged_paths(framework_root)
    if unmerged:
        ok, err = _git_quarantine_unmerged(framework_root, unmerged)
        log.warning(
            "integrate_patch: %s had %d path(s) in an unresolved merge (%s); moved to a '%s' stash entry%s",
            framework_root,
            len(unmerged),
            ", ".join(unmerged[:5]),
            _HYPERLOOM_QUARANTINE_STASH_MSG,
            "" if ok else f" FAILED: {err}",
        )
        if not ok:
            return "failed", f"unresolved merge could not be quarantined: {err}"
        cp = _run_git_cp(["-C", str(framework_root), "status", "--porcelain"], timeout=30.0)
        if cp is None:
            return "failed", "git status check failed"
        if not cp.stdout.strip():
            return "clean", ""
    cp2 = _run_git_cp(
        ["-C", str(framework_root), "stash", "push", "-u", "-m", _HYPERLOOM_AUTO_STASH_MSG],
        timeout=60.0,
    )
    if cp2 is None:
        return "failed", "git stash push failed"
    if cp2.returncode == 0:
        stash_ref = _find_hyperloom_auto_stash(framework_root) or "stash@{0}"
        log.info(
            "integrate_patch: stashed user changes in %s as %s",
            framework_root,
            stash_ref,
        )
        return "stashed", stash_ref
    return "failed", f"git stash push rc={cp2.returncode}: {cp2.stderr.strip()}"


def _git_restore_stash_if_needed(
    framework_root: Path,
    stash_state: str,
    stash_ref: str,
) -> str:
    """Restore the user-change stash created before candidate mutation."""
    if stash_state != "stashed":
        return ""
    ref = stash_ref or _find_hyperloom_auto_stash(framework_root)
    if not ref:
        return "auto-stash ref not found; user changes remain in git stash"
    cp = _run_git_cp(["-C", str(framework_root), "stash", "pop", "--index", ref], timeout=120.0)
    if cp is None:
        return f"git stash pop failed; user changes remain in {ref}"
    if cp.returncode == 0:
        log.info("integrate_patch: restored user changes from %s", ref)
        return ""
    # A pop that cannot merge leaves conflict markers in source. Left there they
    # break every later measurement silently -- the benchmark fails to parse the
    # file, and git will not stash again while the merge is unresolved. Put the
    # tree back at HEAD instead; the stash is deliberately NOT dropped, so the
    # work is still there under `git stash list`.
    detail = (cp.stderr or "").strip()
    unmerged = _git_unmerged_paths(framework_root)
    if unmerged:
        ok, err = _git_restore_to_head(framework_root, unmerged)
        log.warning(
            "integrate_patch: %s did not merge back into %s; restored %d path(s) to HEAD and kept the stash%s",
            ref,
            framework_root,
            len(unmerged),
            "" if ok else f" (restore FAILED: {err})",
        )
        if not ok:
            detail = f"{detail}; tree left conflicted: {err}"
    return f"git stash pop {ref} rc={cp.returncode}: {detail}; user changes remain in git stash"


def _git_reset_clean(framework_root: Path, base: str, *, exclude: list[str] | None = None) -> tuple[bool, str]:
    """Return the tree to ``base`` exactly: tracked, staged and untracked candidate files all go.

    This is the REVERT: ``base`` is the HEAD recorded before the candidate,
    which is the accepted stack because every KEEP is committed. User changes
    must already have been stashed before the candidate was applied.

    Args:
        framework_root: Directory to run ``git reset`` in.
        base: The commit to restore.
        exclude: Paths ``git clean`` must keep.

    Returns:
        ``(ok, stderr)`` where ``ok`` is ``True`` when both steps succeed.
    """
    cp = _run_git_cp(["-C", str(framework_root), "reset", "--hard", base], timeout=60.0)
    if cp is None:
        return False, "git reset spawn failed"
    if cp.returncode != 0:
        return False, (cp.stderr or "").strip()
    exclusions = [arg for path in exclude or [] for arg in ("-e", path)]
    cp2 = _run_git_cp(["-C", str(framework_root), "clean", "-fd", *exclusions], timeout=60.0)
    if cp2 is None:
        return False, "git clean spawn failed"
    return cp2.returncode == 0, cp2.stderr.strip()


def _recorded_stash_ref(framework_root: Path | None, stash_oid: str) -> str:
    """The stash entry holding the operator's pre-candidate work, found by the oid recorded for it."""
    if not stash_oid:
        return ""
    if framework_root is None:
        raise ValueError("stash recovery root missing")
    cp = _run_git_cp(["-C", str(framework_root), "stash", "list", "--format=%H %gd"], timeout=30.0)
    if cp is None or cp.returncode != 0:
        raise OSError("cannot read stash identities")
    refs = [line.split(" ", 1)[1] for line in cp.stdout.splitlines() if line.startswith(stash_oid + " ")]
    if len(refs) != 1:
        raise ValueError("recorded user stash is absent or ambiguous; refusing to guess")
    return refs[0]


def _take_candidate_out(
    pending: dict[str, Any],
    recovery: dict[str, Any],
    *,
    framework_root: Path | None,
    workspace: Path,
) -> dict[str, Any]:
    """Undo this attempt's artifact installs and patch applies, reporting what came back out."""
    summary: dict[str, Any] = {"reversed": [], "artifacts_reverted": [], "failed": []}
    try:
        artifact_records: list[dict[str, Any]] = []
        if recovery.get("artifacts_prepared"):
            # The artifact phase commits its whole plan's preimages before it
            # installs anything, so the witness tells "never started" apart from
            # "started, evidence now gone".
            artifact_records = load_prepared_records(workspace / "artifact_backups")
            expected = {str(Path(row["target"]).resolve()) for row in pending.get("artifacts", [])}
            if not expected <= {str(Path(row["target"]).resolve()) for row in artifact_records}:
                raise ValueError("artifact recovery ledger does not cover the complete mutation plan")
        # A git attempt takes no backups, so a HEAD that has moved since means the
        # tree is no longer the one it mutated. A non-git attempt writes a preimage
        # record before every mutation, so an absent ledger there is proof that no
        # patch reached the tree.
        patches = pending.get("patches")
        base = ""
        undo_records: list[dict[str, Any]] = []
        if patches:
            base = str(recovery.get("git_head", ""))
            if base:
                if framework_root is None or _git_head_sha(framework_root) != base:
                    raise ValueError("framework HEAD differs from the recorded attempt base")
            elif framework_root is not None and _is_git_tree(framework_root):
                raise ValueError("git attempt recorded no base to restore to")
            else:
                undo_records = load_prepared_records(workspace / "patch_backups")
        # Both ledgers are read and the artifacts are undone before the git sweep,
        # which is irreversible: it has to run against a tree that already holds
        # every preimage, and an artifact target git will not touch -- ignored, or
        # outside this root -- has no second chance.
        _, errors = restore_records(artifact_records)
        if errors:
            summary["failed"].extend(errors)
            return summary
        summary["artifacts_reverted"] = [row["rel_target"] for row in artifact_records]
        if not patches:
            return summary
        if base:
            # Everything still untracked here was created by this attempt: the
            # pre-candidate auto-stash took the operator's untracked files with
            # ``push -u``, and ``clean`` without ``-x`` leaves ignored paths (an
            # ignored artifact target included) where they are. Only this
            # attempt's own recovery data has to survive the sweep.
            exclusions = (
                ["/" + workspace.relative_to(framework_root).as_posix()]
                if workspace.is_relative_to(framework_root)
                else []
            )
            ok, error = _git_reset_clean(framework_root, base, exclude=exclusions)
            if not ok:
                raise OSError(error)
        else:
            _, errors = restore_records(undo_records)
            if errors:
                summary["failed"].extend(errors)
                return summary
        summary["reversed"] = list(reversed(patches))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        summary["failed"].append(str(exc))
    return summary


def restore_pending_integrate(pending: dict[str, Any], *, keep: bool = False) -> dict[str, Any]:
    """Discharge one recorded integration's file and stash obligations, or retain them."""
    summary: dict[str, Any] = {"reversed": [], "artifacts_reverted": [], "failed": []}
    recovery = pending.get("recovery")
    if not isinstance(recovery, dict) or recovery.get("version") != 1:
        if pending.get("patches") or pending.get("artifacts") or pending.get("attempt_venv_root"):
            summary["failed"].append("pending integration has no complete recovery evidence")
        return summary
    phase = recovery.get("phase")
    if phase in ("restored", "accepted"):
        return summary
    if phase not in ("ready", "applied", "files_restored"):
        summary["failed"].append(
            "the operator's uncommitted work was merged back on top of the ungraded candidate and "
            "the stash was dropped, so nothing on disk separates the two; taking the candidate back "
            "out would destroy that work, and this refuses rather than guess"
            if phase == "applied_with_restored_stash"
            else f"integration interrupted before stash identity was durable: {phase!r}"
        )
        return summary
    root = recovery.get("root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        summary["failed"].append("attempt-owned recovery root missing")
        return summary
    root_text = str(pending.get("framework_source_root", ""))
    framework_root = Path(root_text) if root_text else None
    try:
        stash_ref = _recorded_stash_ref(framework_root, str(recovery.get("stash_oid", "")))
        if not keep and phase != "files_restored":
            summary = _take_candidate_out(pending, recovery, framework_root=framework_root, workspace=Path(root))
            if summary["failed"]:
                return summary
            recovery["phase"] = "files_restored"
        if stash_ref:
            note = _git_restore_stash_if_needed(framework_root, "stashed", stash_ref)
            if note:
                raise OSError(note)
        recovery["phase"] = "accepted" if keep else "restored"
    except (OSError, ValueError, TypeError, KeyError) as exc:
        summary["failed"].append(str(exc))
    return summary


__all__ = [
    "_detect_p_level",
    "_git_apply",
    "_git_apply_reverse",
    "_git_reset_clean",
    "_run_git_apply",
    "restore_pending_integrate",
]
