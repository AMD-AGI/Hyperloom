# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply an attempt's ordered patch set to its framework tree, and say what that changed.

A git checkout takes each patch through ``git apply`` here; any other tree goes
through :mod:`._nogit_patch`. Both channels report a patch whose every hunk the
tree already holds as present rather than failed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...specialists.patch_safety import is_unified_diff, patch_escapes_tree
from ._apply_feedback import ApplyFeedback, PatchApplyResult, PresenceCheck, build_apply_feedback
from ._git import _run_git_cp
from ._nogit_patch import _P_LEVELS, _apply_patch_no_git, _reverse_probe_levels

if TYPE_CHECKING:
    from ._integrate_attempt import IntegrateAttempt

log = logging.getLogger(__name__)


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
        ``True`` when some level :func:`._nogit_patch._reverse_probe_levels`
        names reverse-checks cleanly.
    """
    for lvl in _reverse_probe_levels(framework_root, patch_path):
        cp = _run_git_cp(
            ["-C", str(framework_root), "apply", "-R", f"-p{lvl}", "--check", str(patch_path)],
            timeout=120.0,
        )
        if cp is not None and cp.returncode == 0:
            return True
    return False


def _git_apply_failure(
    framework_root: Path, patch_path: Path, tried_levels: list[int], stderr: str
) -> PatchApplyResult:
    """A failed git apply, with the feedback a reauthoring round needs."""
    feedback = build_apply_feedback(
        patch_path,
        channel="git",
        tried_levels=tried_levels,
        stderr=stderr,
        framework_root=framework_root,
    )
    return PatchApplyResult(ok=False, error=stderr, feedback=feedback)


def _git_apply_collect_feedback(framework_root: Path, patch_path: Path) -> PatchApplyResult:
    """Apply ``patch_path`` at the first ``-p`` level whose ``git apply --check`` passes.

    With no level passing, a patch the tree already holds in full is reported
    present. A failure carries the stderr of every tried ``-p`` level and a
    source-context snippet.

    Args:
        framework_root: The git checkout to apply into.
        patch_path: The patch file to apply.

    Returns:
        The apply's result.
    """
    tried_levels: list[int] = []
    level_stderrs: list[str] = []
    for lvl in _P_LEVELS:
        ok_check, stderr_check = _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=True)
        tried_levels.append(lvl)
        if stderr_check:
            level_stderrs.append(f"-p{lvl}: {stderr_check}")
        if ok_check:
            ok_apply, stderr_apply = _run_git_apply(framework_root, patch_path, p_level=lvl, check_only=False)
            if ok_apply:
                return PatchApplyResult(ok=True)
            return _git_apply_failure(framework_root, patch_path, tried_levels, stderr_apply)

    # A subset of a patch already applied is a no-op that ``--check`` still
    # rejects; a clean reverse check proves every hunk is in the tree.
    if _git_reverse_applies_cleanly(framework_root, patch_path):
        log.info(
            "integrate_patch: %s is already fully applied (clean git apply -R --check); treating as a no-op",
            patch_path.name,
        )
        return PatchApplyResult(ok=True, present_by=PresenceCheck.GIT_REVERSE_APPLY)
    return _git_apply_failure(framework_root, patch_path, tried_levels, "\n".join(level_stderrs))


@dataclass
class PatchSetOutcome:
    """What applying an attempt's patch set did to its framework tree."""

    #: One row per patch the tree already held in full: the patch, and the check that showed it.
    already_present: list[dict[str, str]] = field(default_factory=list)
    #: The patch that stopped the set, and why; empty when every patch landed.
    errors: list[dict[str, str]] = field(default_factory=list)
    #: Reauthoring feedback for that failure.
    feedbacks: list[ApplyFeedback] = field(default_factory=list)
    #: Every patch landed, and the tree already held each one, so the tree is unchanged.
    changed_nothing: bool = False


def _unvetted_reason(patch: Path) -> str:
    """Why ``patch`` may not reach an apply channel, or ``""`` when it may.

    ``vet_patches`` runs at authoring time inside the specialist runner, so a
    patch from anywhere else -- ``params.patches``, and every remotely-fetched
    ``upstream_pr`` diff -- arrives unvetted.
    """
    try:
        patch_text = patch.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"unreadable: {exc!r}"
    if not is_unified_diff(patch_text):
        return "not a unified diff"
    escaping = patch_escapes_tree(patch_text)
    if escaping is not None:
        return f"path escapes tree: {escaping!r}"
    return ""


def apply_patch_set(
    attempt: IntegrateAttempt,
    patch_paths: list[Path],
    *,
    git_tree: bool,
    backup_root: Path,
) -> PatchSetOutcome:
    """Apply ``patch_paths`` in order into ``attempt.framework_root``, stopping at the first that fails.

    A patch the tree already holds still counts as landed, so one subset patch
    inside a larger set does not fail the set. Each landed patch is appended to
    ``attempt.applied`` as it lands, so a stage that never returns still
    reports what reached the tree.

    Args:
        attempt: The attempt whose tree the set lands in.
        patch_paths: The patches, in apply order.
        git_tree: Whether the tree is a git checkout, which goes back through
            its HEAD; any other tree goes through ``patch`` with backups.
        backup_root: Where the ``patch`` channel keeps its per-file backups.

    Returns:
        What the set did to the tree.
    """
    framework_root = attempt.framework_root
    outcome = PatchSetOutcome()
    backups: list[dict[str, Any]] = []
    for patch in patch_paths:
        refusal = _unvetted_reason(patch)
        if refusal:
            outcome.errors.append({"patch": str(patch), "stderr": refusal})
            break
        if git_tree:
            result = _git_apply_collect_feedback(framework_root, patch)
        else:
            result = _apply_patch_no_git(framework_root, patch, backup_root, seq_offset=len(backups))
            backups.extend(result.backups)
        if not result.ok:
            outcome.errors.append({"patch": str(patch), "stderr": result.error})
            if result.feedback is not None:
                outcome.feedbacks.append(result.feedback)
            break
        if result.present_by is not None:
            outcome.already_present.append({"patch": str(patch), "check": result.present_by.value})
        attempt.applied.append(patch)
    outcome.changed_nothing = (
        bool(patch_paths) and not outcome.errors and len(outcome.already_present) == len(patch_paths)
    )
    return outcome


__all__ = ["PatchSetOutcome", "apply_patch_set"]
