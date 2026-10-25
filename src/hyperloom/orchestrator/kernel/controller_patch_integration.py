# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sequential Git and E2E integration of KernelForge Controller patches."""

from __future__ import annotations

import contextlib
import logging
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
from .patch_conflict_merge import LandingContext, apply_patch_resolving_conflicts

log = logging.getLogger(__name__)

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


def _measured_kernel_saving(patch_dir: Path) -> float:
    """The share of its kernel's time a publication measured saving, 0 when it recorded no speedup."""
    try:
        speedup = float(load_controller_publication(patch_dir).manifest.get("total_speedup") or 0.0)
    except (ControllerPublicationError, TypeError, ValueError):
        return 0.0
    return 1.0 - 1.0 / speedup if speedup > 0.0 else 0.0


class _MeasurementOrder(NamedTuple):
    """Where a publication goes in the E2E queue: larger measured saving first, then task priority, then operator."""

    negated_saving: float
    priority: int
    operator_id: str


def _measurement_ordered_patch_dirs(patches_root: str | Path) -> tuple[Path, ...]:
    """Order publications by the kernel time each measured saving, the Controller's task priority breaking ties.

    The phase reserves one E2E round past the Controller and admits later ones only into what is left of it, so the
    first publication is the one surely measured. The task priority was set before any search ran.
    """
    root = Path(patches_root).resolve()
    tasks_root = root.parent.parent / "controller" / "tasks"

    def key(patch_dir: Path) -> _MeasurementOrder:
        saving = -_measured_kernel_saving(patch_dir)
        parsed = load_task(tasks_root / patch_dir.name, record_state=False)
        if parsed.task is None:
            return _MeasurementOrder(saving, 2**31 - 1, patch_dir.name)
        return _MeasurementOrder(saving, parsed.task.priority, parsed.task.operator_id)

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
    except (OSError, subprocess.SubprocessError):  # an unreadable HEAD reads as "no commit landed"
        return ""


def _tracked_at_head(repo: Path, relative: str) -> bool:
    """Whether HEAD carries ``relative``, i.e. whether it has a version to restore."""
    try:
        _git_output(repo, "cat-file", "-e", f"HEAD:{relative}")
    except (OSError, subprocess.SubprocessError):  # a failed lookup means "no version at HEAD"
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
    validate = validator or (
        lambda publication: _default_validator(
            publication,
            session_dir=Path(session_dir),
        )
    )
    results: list[PatchIntegrationResult] = []
    # One base commit per repository rather than one repository per run.
    pinned_bases: dict[Path, str] = {}
    pinned_heads: dict[Path, str] = {}
    pin_errors: dict[Path, str] = {}
    # Patches already committed into each repository, so a merge can be held to
    # keeping what they added.
    landed: dict[Path, list[tuple[str, Path]]] = {}
    rounds_run = 0

    for index, patch_dir in enumerate(_measurement_ordered_patch_dirs(patches_root)):
        try:
            publication = load_controller_publication(patch_dir)
        except ControllerPublicationError as error:
            result = PatchIntegrationResult(
                operator_id=patch_dir.name,
                status="skipped_invalid",
                reason=str(error),
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue
        repo = publication.repo_root
        if repo not in pinned_bases:
            pinned_bases[repo] = publication.base_commit
            pinned_heads[repo], pin_errors[repo] = _head_against_base(repo, publication.base_commit)

        if pin_errors.get(repo):
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="skipped_baseline_mismatch",
                reason=pin_errors[repo],
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=pinned_heads.get(repo, ""),
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        if publication.base_commit != pinned_bases[repo]:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="skipped_baseline_mismatch",
                reason=f"repository {repo} is pinned to controller base {pinned_bases[repo]} for this integration",
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        # Scoped to the paths this patch touches, not to the whole repository.
        touched = _patch_touched_paths(repo, [publication.patch_path])
        try:
            head_before = _git_output(repo, "rev-parse", "HEAD").lower()
            # A patch whose paths cannot be read is one nothing can be scoped to, so it falls back to asking about the
            # whole tree.
            scope = ["--", *sorted(touched)] if touched else []
            clean = _git_output(repo, "status", "--porcelain", "--untracked-files=no", *scope)
        except (OSError, subprocess.SubprocessError) as error:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="skipped_invalid",
                reason=f"could not inspect integration repository: {error}",
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue
        if clean:
            changed = ", ".join(sorted(touched)) if touched else "the repository"
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="skipped_dirty_worktree",
                reason=f"uncommitted tracked changes on the paths this patch modifies: {changed}",
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue
        if (stop := admit_round(rounds_run)) is not None:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id, status=stop.error_class, reason=stop.never_started
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        # Lanes all diff against the pinned base, so every KEEP committed above
        # makes the diffs still queued stale on the files it touched.
        merge = await apply_patch_resolving_conflicts(
            repo,
            publication.patch_path,
            LandingContext(
                operator_id=publication.operator_id,
                landed_operator_ids=[operator for operator, _ in landed.get(repo, [])],
                landed_patches=[patch for _, patch in landed.get(repo, [])],
                intent=f"{publication.operator_name} in {publication.kernel_path}",
            ),
        )
        if not merge.applied:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_apply_conflict" if merge.conflicted else "reverted_apply_failed",
                reason=merge.error + merge.note(),
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        rounds_run += 1
        try:
            validation = await validate(publication)
        except Exception as error:  # noqa: BLE001 - translated into a reverted_e2e_failed result
            log.warning("E2E validation raised for %s: %s", publication.operator_id, error)
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_e2e_failed",
                reason=f"E2E validation raised: {error}" + _revert_note(repo, publication.patch_path),
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        if (
            str(validation.get("status") or "ok").lower() != "ok"
            or str(validation.get("decision") or "").upper() != "KEEP"
        ):
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_e2e_failed",
                reason=str(validation.get("error") or validation.get("decision_reason") or "E2E did not KEEP")
                + _revert_note(repo, publication.patch_path),
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
                new_tput=float(validation.get("new_tput") or 0.0),
                gain_pct=float(validation.get("gain_pct") or 0.0),
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        committed, commit_note = _git_commit_kept(
            repo,
            f"hyperloom: keep KernelForge rewrite {publication.operator_id}",
            touched,
        )
        # A KEEP is only durable once HEAD carries it, so ask Git rather than the note.
        keep_commit = _head_commit(repo)
        if not committed or not keep_commit or keep_commit == head_before:
            # Pods, compiled artifacts and the JIT tree come back from the apply
            # manifest first; the local worktree is reversed after, because its
            # own backup holds the patched bytes.
            settle_note = _settle_apply_manifest(validation, kept=False)
            revert_note = _revert_note(repo, publication.patch_path)
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_commit_failed",
                reason=(commit_note or "git commit did not advance HEAD") + settle_note + revert_note,
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        try:
            await record_keep(_keep_result(publication, validation, keep_commit))
            shared_state.save(Path(session_dir))
        except Exception as error:  # noqa: BLE001 - KEEP is already committed to Git; record_keep runs the whole writeback
            log.warning("KEEP %s committed but recording it failed: %s", publication.operator_id, error)
            record_reason = f"Git KEEP committed; SharedState recording failed: {error}"
        else:
            record_reason = ""
        landed.setdefault(repo, []).append((publication.operator_id, publication.patch_path))
        result = PatchIntegrationResult(
            operator_id=publication.operator_id,
            status="kept",
            merge_strategy=merge.strategy,
            reason=record_reason + merge.note() + _settle_apply_manifest(validation, kept=True),
            base_commit=publication.base_commit,
            best_commit=publication.best_commit,
            repo_root=str(repo),
            integration_head_before=head_before,
            integration_head_after=keep_commit,
            keep_commit=keep_commit,
            new_tput=float(validation.get("new_tput") or 0.0),
            gain_pct=float(validation.get("gain_pct") or 0.0),
        )
        results.append(result)
        _write_result(results_dir, index, result)

    kept = sum(result.status == "kept" for result in results)
    reverted = sum(result.status.startswith("reverted_") for result in results)
    skipped = len(results) - kept - reverted
    # "completed" says the loop ran, which is not the same as the loop having done anything.
    if results and kept == 0 and reverted == 0:
        status = "no_patch_admitted"
    else:
        status = "completed"
    summary = ControllerIntegrationSummary(
        status=status,
        results=tuple(results),
        kept_count=kept,
        reverted_count=reverted,
        skipped_count=skipped,
        results_dir=str(results_dir),
    )
    atomic_write_json(integration_root / "summary.json", summary.to_dict(), fmt=JsonFormat(trailing_newline=True))
    return summary


__all__ = [
    "ControllerIntegrationSummary",
    "PatchIntegrationResult",
    "integrate_controller_patches",
]
