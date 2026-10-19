# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sequential Git and E2E integration of KernelForge Controller patches."""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NamedTuple

from kernelforge.kernel_rewrite_controller.task import load_task

from hyperloom.common.io import JsonFormat, atomic_write_json
from hyperloom.orchestrator.actions.executors._patch_snapshot import (
    _git_commit_kept,
    _patch_touched_paths,
)
from hyperloom.orchestrator.actions.executors.integrate_patch import (
    _git_apply_reverse,
    _git_restore_to_head,
)
from hyperloom.orchestrator.actions.stop_attribution import StoppedByTheRun

from .controller_publication import (
    ControllerPatchPublication,
    ControllerPublicationError,
    discover_controller_patch_dirs,
    load_controller_publication,
)
from .patch_conflict_merge import LandingContext, PatchMergeOutcome, apply_patch_resolving_conflicts

_CONTROLLER_SOURCE = "kernel_rewrite_controller"
_CONTROLLER_BACKEND = "forge"


@dataclass(frozen=True)
class PatchIntegrationResult:
    operator_id: str
    status: str
    reason: str = ""
    base_commit: str = ""
    best_commit: str = ""
    repo_root: str = ""
    integration_head_before: str = ""
    integration_head_after: str = ""
    keep_commit: str = ""
    new_tput: float = 0.0
    gain_pct: float = 0.0
    #: How the patch reached the worktree; anything but ``strict`` was rebuilt
    #: against the KEEPs that landed ahead of it.
    merge_strategy: str = ""


@dataclass(frozen=True)
class ControllerIntegrationSummary:
    status: str
    results: tuple[PatchIntegrationResult, ...]
    kept_count: int
    reverted_count: int
    skipped_count: int
    results_dir: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "results": [asdict(result) for result in self.results],
            "kept_count": self.kept_count,
            "reverted_count": self.reverted_count,
            "skipped_count": self.skipped_count,
            "results_dir": self.results_dir,
        }


PatchValidator = Callable[[ControllerPatchPublication], Awaitable[dict[str, Any]]]

#: Records one validated KEEP into SharedState.
KeepRecorder = Callable[[dict[str, Any]], Awaitable[None]]

#: Given how many E2E rounds already ran, says why the next one cannot start, or ``None`` to start it.
RoundAdmission = Callable[[int], StoppedByTheRun | None]


def _priority_ordered_patch_dirs(patches_root: str | Path) -> tuple[Path, ...]:
    """Order publications by their original Controller task priority."""
    root = Path(patches_root).resolve()
    tasks_root = root.parent.parent / "controller" / "tasks"

    def key(patch_dir: Path) -> tuple[int, str]:
        parsed = load_task(tasks_root / patch_dir.name, record_state=False)
        if parsed.task is None:
            return (2**31 - 1, patch_dir.name)
        return (parsed.task.priority, parsed.task.operator_id)

    return tuple(sorted(discover_controller_patch_dirs(root), key=key))


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return completed.stdout.strip()


def _head_against_base(repo: Path, base_commit: str) -> tuple[str, str]:
    """Return the integration HEAD and why patches built on ``base_commit`` cannot land on it (empty when they can)."""
    try:
        head = _git_output(repo, "rev-parse", "HEAD").lower()
    except (OSError, subprocess.SubprocessError) as error:
        return "", f"could not read integration Git HEAD: {error}"
    if head != base_commit:
        return head, f"integration HEAD {head} does not match controller base {base_commit}"
    return head, ""


def _head_commit(repo: Path) -> str:
    """Return the repository's HEAD, or an empty string when it cannot be read."""
    try:
        return _git_output(repo, "rev-parse", "HEAD").lower()
    except Exception:  # noqa: BLE001 - an unreadable HEAD reads as "no commit landed"
        return ""


def _tracked_at_head(repo: Path, relative: str) -> bool:
    """Whether HEAD carries ``relative``, i.e. whether it has a version to restore."""
    try:
        _git_output(repo, "cat-file", "-e", f"HEAD:{relative}")
    except Exception:  # noqa: BLE001 - anything but a hit means "no version at HEAD"
        return False
    return True


def _revert_patch(repo: Path, patch_path: Path) -> tuple[bool, str]:
    """Undo one applied patch without touching a path the patch never named."""
    touched = _patch_touched_paths(repo, [patch_path])
    if touched:
        # A commit attempt that failed after ``git add`` leaves the patched content staged, and reversing the working
        # tree does not unstage it -- which would make the next patch see a dirty index and skip.
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            _git_output(repo, "reset", "--quiet", "HEAD", "--", *touched)
    reversed_ok, reverse_error = _git_apply_reverse(repo, patch_path)
    if reversed_ok:
        return True, ""
    # A reverse apply refuses a partially applied patch, which is the state a failed forward apply leaves.
    tracked = [relative for relative in touched if _tracked_at_head(repo, relative)]
    if not tracked:
        return False, reverse_error or "patch could not be reversed"
    restored_ok, restore_error = _git_restore_to_head(repo, tracked)
    if not restored_ok:
        return False, restore_error or reverse_error
    untracked_residue = [relative for relative in touched if relative not in tracked]
    if untracked_residue:
        return True, f"left files the patch created in place: {', '.join(untracked_residue)}"
    return True, ""


def _revert_note(repo: Path, patch_path: Path) -> str:
    """Revert one patch and render what happened as a reason suffix."""
    reverted, note = _revert_patch(repo, patch_path)
    if not reverted:
        return f" (revert failed: {note})"
    return f" (revert: {note})" if note else ""


def _settle_apply_manifest(validation: dict[str, Any], *, kept: bool) -> str:
    """Release the apply's backups now that the KEEP's fate is settled.

    Integrate defers this for a pre-applied publication because only the commit
    here makes it durable: finalizing earlier would delete the pod-side backups
    a failed commit still needs.
    """
    from ..actions.executors._kernel_agent_tool import _maybe_finalize_kernel_patch, _maybe_revert_kernel_patch

    apply_result = validation.get("apply_result")
    if not isinstance(apply_result, dict) or not apply_result.get("manifest_path"):
        return ""
    stage = "finalize" if kept else "revert"
    outcome = _maybe_finalize_kernel_patch(apply_result) if kept else _maybe_revert_kernel_patch(apply_result)
    if str(outcome.get("status") or "") in {"ok", "skipped"}:
        return ""
    return f" (patch {stage} incomplete: {outcome.get('error') or outcome.get('status')})"


def _write_result(results_dir: Path, index: int, result: PatchIntegrationResult) -> None:
    atomic_write_json(results_dir / f"{index:04d}.json", asdict(result), fmt=JsonFormat(trailing_newline=True))


def _keep_result(
    publication: ControllerPatchPublication,
    validation: dict[str, Any],
    keep_commit: str,
) -> dict[str, Any]:
    """Preserve the validated measurement and committed source identity for writeback."""
    return {
        **validation,
        "kernel_id": publication.operator_id,
        "operator_id": publication.operator_id,
        "patch_path": str(publication.patch_path),
        "target_file": str(publication.repo_root / publication.kernel_path),
        # A Controller KEEP lands as a committed source layer, not a snapshot
        # overlay; ``scope`` is what the source-layer export keys on.
        "scope": "source_patch",
        "base_sha": publication.base_commit,
        "keep_commit": keep_commit,
        "source": _CONTROLLER_SOURCE,
        "backend": _CONTROLLER_BACKEND,
        "engine": _CONTROLLER_SOURCE,
    }


async def _default_validator(
    publication: ControllerPatchPublication,
    *,
    session_dir: Path,
) -> dict[str, Any]:
    from .request_handlers import integrate_handler

    return await integrate_handler(
        {
            "kernel_id": publication.operator_id,
            "patch_path": str(publication.patch_path),
            "target_file": str(publication.repo_root / publication.kernel_path),
            # Apply resolves the patch's paths and its final-content snapshot
            # against this root; without it the diff has no repo to land in.
            "repo": str(publication.repo_root),
            # The Controller's Git-derived scope when it has one; the optimizer's own manifest only as a fallback for
            # a publication without it.
            "patch_write_paths": list(publication.changed_files)
            or list(publication.manifest.get("changed_files") or []),
        },
        session_dir=session_dir,
        preapplied_git_patch=True,
    )


class _IntegrationSession(NamedTuple):
    """The session-owned collaborators one integration pass works through."""

    session_dir: Path
    shared_state: Any
    record_keep: KeepRecorder
    validate: PatchValidator
    admit_round: RoundAdmission


class _WorktreeInspection(NamedTuple):
    """What the integration repository looked like before a patch was applied."""

    head_before: str
    touched: list[str]
    refusal: PatchIntegrationResult | None


class _ValidationOutcome(NamedTuple):
    """The E2E verdict on one applied patch."""

    validation: dict[str, Any]
    refusal: PatchIntegrationResult | None


class _KeepCommit(NamedTuple):
    """The commit a KEEP landed on, or why it did not land."""

    commit: str
    failure_reason: str


def _refusal(
    publication: ControllerPatchPublication,
    status: str,
    reason: str,
    head_before: str = "",
) -> PatchIntegrationResult:
    """A result for a publication this run carries no further."""
    return PatchIntegrationResult(
        operator_id=publication.operator_id,
        status=status,
        reason=reason,
        base_commit=publication.base_commit,
        best_commit=publication.best_commit,
        repo_root=str(publication.repo_root),
        integration_head_before=head_before,
    )


def _e2e_kept(validation: dict[str, Any]) -> bool:
    """Whether the E2E run both completed and decided KEEP."""
    return (
        str(validation.get("status") or "ok").lower() == "ok"
        and str(validation.get("decision") or "").upper() == "KEEP"
    )


def _commit_keep(
    publication: ControllerPatchPublication,
    validation: dict[str, Any],
    touched: list[str],
    head_before: str,
) -> _KeepCommit:
    """Commit the validated patch into the integration repository, undoing it when the commit fails."""
    repo = publication.repo_root
    committed, commit_note = _git_commit_kept(
        repo,
        f"hyperloom: keep KernelForge rewrite {publication.operator_id}",
        touched,
    )
    # A KEEP is only durable once HEAD carries it, so ask Git rather than the note.
    keep_commit = _head_commit(repo)
    if committed and keep_commit and keep_commit != head_before:
        return _KeepCommit(keep_commit, "")
    # Pods, compiled artifacts and the JIT tree come back from the apply
    # manifest first; the local worktree is reversed after, because its
    # own backup holds the patched bytes.
    settle_note = _settle_apply_manifest(validation, kept=False)
    revert_note = _revert_note(repo, publication.patch_path)
    return _KeepCommit("", (commit_note or "git commit did not advance HEAD") + settle_note + revert_note)


def _integration_summary(
    results: list[PatchIntegrationResult],
    results_dir: Path,
) -> ControllerIntegrationSummary:
    """Roll the per-patch outcomes up into the run's verdict."""
    kept = sum(result.status == "kept" for result in results)
    reverted = sum(result.status.startswith("reverted_") for result in results)
    # "completed" says the loop ran, which is not the same as the loop having done anything.
    admitted_nothing = bool(results) and kept == 0 and reverted == 0
    return ControllerIntegrationSummary(
        status="no_patch_admitted" if admitted_nothing else "completed",
        results=tuple(results),
        kept_count=kept,
        reverted_count=reverted,
        skipped_count=len(results) - kept - reverted,
        results_dir=str(results_dir),
    )


class _ControllerIntegration:
    """One sequential pass over a Controller's published patches."""

    def __init__(self, session: _IntegrationSession, results_dir: Path) -> None:
        self._session = session
        self._results_dir = results_dir
        self.results: list[PatchIntegrationResult] = []
        # One base commit per repository rather than one repository per run.
        self._pinned_bases: dict[Path, str] = {}
        self._pinned_heads: dict[Path, str] = {}
        self._pin_errors: dict[Path, str] = {}
        # Patches already committed into each repository, so a merge can be held to
        # keeping what they added.
        self._landed: dict[Path, list[tuple[str, Path]]] = {}
        self._rounds_run = 0

    async def integrate(self, index: int, patch_dir: Path) -> None:
        """Apply, validate and record the publication in ``patch_dir``."""
        try:
            publication = load_controller_publication(patch_dir)
        except ControllerPublicationError as error:
            self._record(index, PatchIntegrationResult(patch_dir.name, "skipped_invalid", str(error)))
            return
        refusal = self._refuse_against_pinned_base(publication)
        if refusal is not None:
            self._record(index, refusal)
            return
        worktree = self._inspect_worktree(publication)
        if worktree.refusal is not None:
            self._record(index, worktree.refusal)
            return
        self._record(index, await self._apply_and_validate(publication, worktree))

    def _record(self, index: int, result: PatchIntegrationResult) -> None:
        """Keep one patch's outcome and publish it beside the others."""
        self.results.append(result)
        _write_result(self._results_dir, index, result)

    def _refuse_against_pinned_base(
        self,
        publication: ControllerPatchPublication,
    ) -> PatchIntegrationResult | None:
        """Pin the repository on first sight, and refuse a publication built on another base."""
        repo = publication.repo_root
        if repo not in self._pinned_bases:
            self._pinned_bases[repo] = publication.base_commit
            self._pinned_heads[repo], self._pin_errors[repo] = _head_against_base(repo, publication.base_commit)
        if self._pin_errors.get(repo):
            return _refusal(
                publication,
                "skipped_baseline_mismatch",
                self._pin_errors[repo],
                self._pinned_heads.get(repo, ""),
            )
        if publication.base_commit != self._pinned_bases[repo]:
            return _refusal(
                publication,
                "skipped_baseline_mismatch",
                f"repository {repo} is pinned to controller base {self._pinned_bases[repo]} for this integration",
            )
        return None

    def _inspect_worktree(self, publication: ControllerPatchPublication) -> _WorktreeInspection:
        """Read the integration repository's HEAD, and refuse a worktree already dirty on these paths."""
        repo = publication.repo_root
        # Scoped to the paths this patch touches, not to the whole repository.
        touched = _patch_touched_paths(repo, [publication.patch_path])
        try:
            head_before = _git_output(repo, "rev-parse", "HEAD").lower()
            # A patch whose paths cannot be read is one nothing can be scoped to, so it falls back to asking about the
            # whole tree.
            scope = ["--", *sorted(touched)] if touched else []
            clean = _git_output(repo, "status", "--porcelain", "--untracked-files=no", *scope)
        except (OSError, subprocess.SubprocessError) as error:
            reason = f"could not inspect integration repository: {error}"
            return _WorktreeInspection("", touched, _refusal(publication, "skipped_invalid", reason))
        if not clean:
            return _WorktreeInspection(head_before, touched, None)
        changed = ", ".join(sorted(touched)) if touched else "the repository"
        refusal = _refusal(
            publication,
            "skipped_dirty_worktree",
            f"uncommitted tracked changes on the paths this patch modifies: {changed}",
            head_before,
        )
        return _WorktreeInspection(head_before, touched, refusal)

    async def _merge_onto_landed(self, publication: ControllerPatchPublication) -> PatchMergeOutcome:
        """Apply the patch, rebuilt against the KEEPs already committed above it.

        Lanes all diff against the pinned base, so every KEEP committed above makes the diffs still
        queued stale on the files it touched.
        """
        landed = self._landed.get(publication.repo_root, [])
        return await apply_patch_resolving_conflicts(
            publication.repo_root,
            publication.patch_path,
            LandingContext(
                operator_id=publication.operator_id,
                landed_operator_ids=[operator for operator, _ in landed],
                landed_patches=[patch for _, patch in landed],
                intent=f"{publication.operator_name} in {publication.kernel_path}",
            ),
        )

    async def _validate_applied_patch(
        self,
        publication: ControllerPatchPublication,
        head_before: str,
    ) -> _ValidationOutcome:
        """Run the E2E decision on the patch now in the worktree, reverting it unless it KEEPs."""
        repo = publication.repo_root
        try:
            validation = await self._session.validate(publication)
        except Exception as error:  # noqa: BLE001 - translated into a skipped_invalid result
            reason = f"E2E validation raised: {error}" + _revert_note(repo, publication.patch_path)
            return _ValidationOutcome({}, _refusal(publication, "reverted_e2e_failed", reason, head_before))
        if _e2e_kept(validation):
            return _ValidationOutcome(validation, None)
        reason = str(validation.get("error") or validation.get("decision_reason") or "E2E did not KEEP")
        refusal = PatchIntegrationResult(
            operator_id=publication.operator_id,
            status="reverted_e2e_failed",
            reason=reason + _revert_note(repo, publication.patch_path),
            base_commit=publication.base_commit,
            best_commit=publication.best_commit,
            repo_root=str(repo),
            integration_head_before=head_before,
            new_tput=float(validation.get("new_tput") or 0.0),
            gain_pct=float(validation.get("gain_pct") or 0.0),
        )
        return _ValidationOutcome(validation, refusal)

    async def _record_keep_in_state(
        self,
        publication: ControllerPatchPublication,
        validation: dict[str, Any],
        keep_commit: str,
    ) -> str:
        """Record the committed KEEP into SharedState; says what stopped it (``""`` when nothing did)."""
        try:
            await self._session.record_keep(_keep_result(publication, validation, keep_commit))
            self._session.shared_state.save(Path(self._session.session_dir))
        except Exception as error:  # noqa: BLE001 - KEEP is already committed to Git
            return f"Git KEEP committed; SharedState recording failed: {error}"
        return ""

    async def _apply_and_validate(
        self,
        publication: ControllerPatchPublication,
        worktree: _WorktreeInspection,
    ) -> PatchIntegrationResult:
        """Land one publication as a validated commit, or say why this run keeps nothing of it."""
        if (stop := self._session.admit_round(self._rounds_run)) is not None:
            return PatchIntegrationResult(
                operator_id=publication.operator_id, status=stop.error_class, reason=stop.never_started
            )
        merge = await self._merge_onto_landed(publication)
        if not merge.applied:
            return _refusal(
                publication,
                "reverted_apply_conflict" if merge.conflicted else "reverted_apply_failed",
                merge.error + merge.note(),
                worktree.head_before,
            )
        self._rounds_run += 1
        outcome = await self._validate_applied_patch(publication, worktree.head_before)
        if outcome.refusal is not None:
            return outcome.refusal
        commit = _commit_keep(publication, outcome.validation, worktree.touched, worktree.head_before)
        if not commit.commit:
            return _refusal(publication, "reverted_commit_failed", commit.failure_reason, worktree.head_before)
        record_reason = await self._record_keep_in_state(publication, outcome.validation, commit.commit)
        self._landed.setdefault(publication.repo_root, []).append((publication.operator_id, publication.patch_path))
        return PatchIntegrationResult(
            operator_id=publication.operator_id,
            status="kept",
            merge_strategy=merge.strategy,
            reason=record_reason + merge.note() + _settle_apply_manifest(outcome.validation, kept=True),
            base_commit=publication.base_commit,
            best_commit=publication.best_commit,
            repo_root=str(publication.repo_root),
            integration_head_before=worktree.head_before,
            integration_head_after=commit.commit,
            keep_commit=commit.commit,
            new_tput=float(outcome.validation.get("new_tput") or 0.0),
            gain_pct=float(outcome.validation.get("gain_pct") or 0.0),
        )


async def integrate_controller_patches(
    *,
    patches_root: str | Path,
    session_dir: Path,
    shared_state: Any,
    record_keep: KeepRecorder,
    validator: PatchValidator | None = None,
    admit_round: RoundAdmission = lambda _rounds_run: None,
) -> ControllerIntegrationSummary:
    """Apply and E2E-validate complete Controller patches in task-priority order.

    Args:
        patches_root: The Controller's published patch directory.
        session_dir: The session whose state the KEEPs are recorded into.
        shared_state: The live session state, persisted after each recorded KEEP.
        record_keep: The session-owned writeback every KEEP is recorded through,
            so a promotion also lands on the stack ledger.
        validator: Runs the E2E decision for one publication; defaults to the
            optimizer's own integrate handler.
        admit_round: Asked before each patch is applied; a stop leaves the patch
            unapplied and unmeasured, recorded under the stop's class.
    """
    integration_root = Path(patches_root).resolve().parent.parent / "integration"
    results_dir = integration_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    session = _IntegrationSession(
        session_dir=Path(session_dir),
        shared_state=shared_state,
        record_keep=record_keep,
        validate=validator or (lambda publication: _default_validator(publication, session_dir=Path(session_dir))),
        admit_round=admit_round,
    )
    integration = _ControllerIntegration(session, results_dir)
    for index, patch_dir in enumerate(_priority_ordered_patch_dirs(patches_root)):
        await integration.integrate(index, patch_dir)

    summary = _integration_summary(integration.results, results_dir)
    atomic_write_json(integration_root / "summary.json", summary.to_dict(), fmt=JsonFormat(trailing_newline=True))
    return summary


__all__ = [
    "ControllerIntegrationSummary",
    "PatchIntegrationResult",
    "integrate_controller_patches",
]
