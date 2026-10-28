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
from typing import Any

from kernelforge.kernel_rewrite_controller import load_task, operator_directory_name

from hyperloom.common.io import JsonFormat, atomic_write_json
from hyperloom.orchestrator.actions.executors._patch_snapshot import (
    _git_commit_kept,
    _patch_touched_paths,
)
from hyperloom.orchestrator.actions.executors._patch_git import _git_apply_reverse, _git_restore_to_head
from hyperloom.orchestrator.actions.stop_attribution import StoppedByTheRun
from hyperloom.orchestrator.enablement.runtime.adapters import get_adapter
from hyperloom.orchestrator.source_snapshot import MANIFEST_NAME, SourceLayer, snapshot_source_layer

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


@dataclass(frozen=True)
class KeepSession:
    """The session a Controller integration records its KEEPs into.

    Attributes:
        session_dir: The session whose state the KEEPs are recorded into.
        shared_state: The live session state, persisted after each recorded KEEP.
        record_keep: The session-owned writeback every KEEP is recorded through,
            so a promotion also lands on the stack ledger.
    """

    session_dir: Path
    shared_state: Any
    record_keep: KeepRecorder


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


def _committed_source_layer(
    repo: Path,
    keep_commit: str,
    operator_id: str,
    session_dir: Path,
    framework: str,
) -> dict[str, Any]:
    """Snapshot the files the KEEP commit changed, so a resume onto a stock tree can reinstall them."""
    handles: dict[str, Any] = {"framework_root": str(repo), "source_layer_landed": True}
    try:
        listing = _git_output(
            repo, "-c", "core.quotepath=off", "diff-tree", "--no-commit-id", "--name-status", "-r", keep_commit
        )
        rows = [line.split("\t") for line in listing.splitlines() if "\t" in line]
        import_root = get_adapter(framework).source_import_root(str(repo))
        snap = snapshot_source_layer(
            SourceLayer(
                framework_root=repo,
                base_sha=keep_commit,
                rel_paths=[row[-1] for row in rows],
                declared_ops={row[-1]: "delete" for row in rows if row[0].startswith("D")},
                import_root=import_root,
            ),
            dest_dir=session_dir / "optimization_stack" / "src" / operator_directory_name(operator_id),
            provenance=_CONTROLLER_SOURCE,
        )
    except (OSError, subprocess.SubprocessError) as error:
        log.warning("KEEP %s committed but its source snapshot could not be written: %s", operator_id, error)
        return handles
    if not snap:
        return handles
    return {
        **handles,
        "source_snapshot": snap["snapshot_dir"],
        "source_manifest": str(Path(snap["snapshot_dir"]) / MANIFEST_NAME),
        "source_snapshot_complete": bool(snap["complete"]),
        "source_import_root": snap["import_root"],
        "target_files": [str(row["rel"]) for row in snap["files"]],
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


def _refusal(
    publication: ControllerPatchPublication,
    status: str,
    reason: str,
    head_before: str = "",
    **fields: Any,
) -> PatchIntegrationResult:
    """A result carrying the publication's identity, however this run ended for it."""
    return PatchIntegrationResult(
        operator_id=publication.operator_id,
        status=status,
        reason=reason,
        base_commit=publication.base_commit,
        best_commit=publication.best_commit,
        repo_root=str(publication.repo_root),
        integration_head_before=head_before,
        **fields,
    )


class _ControllerIntegration:
    """One sequential pass over a Controller's published patches."""

    def __init__(
        self,
        validate: PatchValidator,
        persist_keep: Callable[[ControllerPatchPublication, dict[str, Any], str], Awaitable[str]],
        admit_round: RoundAdmission,
    ) -> None:
        self._validate = validate
        self._persist_keep = persist_keep
        self._admit_round = admit_round
        # One base commit, the head it was pinned against and the pin's error per
        # repository, rather than one repository per run.
        self._pins: dict[Path, tuple[str, str, str]] = {}
        # Patches already committed into each repository, so a merge can be held to
        # keeping what they added.
        self._landed: dict[Path, list[tuple[str, Path]]] = {}
        self._rounds_run = 0

    async def integrate(self, publication: ControllerPatchPublication) -> PatchIntegrationResult:
        """Pin the repository, refuse a stale base or a dirty worktree, then land the patch."""
        repo = publication.repo_root
        if repo not in self._pins:
            self._pins[repo] = (publication.base_commit, *_head_against_base(repo, publication.base_commit))
        base, pinned_head, pin_error = self._pins[repo]
        if pin_error:
            return _refusal(publication, "skipped_baseline_mismatch", pin_error, pinned_head)
        if publication.base_commit != base:
            reason = f"repository {repo} is pinned to controller base {base} for this integration"
            return _refusal(publication, "skipped_baseline_mismatch", reason)

        # Scoped to the paths this patch touches, not to the whole repository.
        touched = _patch_touched_paths(repo, [publication.patch_path])
        try:
            head_before = _git_output(repo, "rev-parse", "HEAD").lower()
            # A patch whose paths cannot be read is one nothing can be scoped to, so it falls back to asking about the
            # whole tree.
            scope = ["--", *sorted(touched)] if touched else []
            clean = _git_output(repo, "status", "--porcelain", "--untracked-files=no", *scope)
        except (OSError, subprocess.SubprocessError) as error:
            return _refusal(publication, "skipped_invalid", f"could not inspect integration repository: {error}")
        if clean:
            changed = ", ".join(sorted(touched)) if touched else "the repository"
            reason = f"uncommitted tracked changes on the paths this patch modifies: {changed}"
            return _refusal(publication, "skipped_dirty_worktree", reason, head_before)
        return await self._apply_and_validate(publication, head_before, touched)

    async def _apply_and_validate(
        self,
        publication: ControllerPatchPublication,
        head_before: str,
        touched: list[str],
    ) -> PatchIntegrationResult:
        """Land one publication as a validated commit, or say why this run keeps nothing of it."""
        if (stop := self._admit_round(self._rounds_run)) is not None:
            return PatchIntegrationResult(
                operator_id=publication.operator_id, status=stop.error_class, reason=stop.never_started
            )
        repo = publication.repo_root
        # Lanes all diff against the pinned base, so every KEEP committed above
        # makes the diffs still queued stale on the files it touched.
        landed = self._landed.get(repo, [])
        merge = await apply_patch_resolving_conflicts(
            repo,
            publication.patch_path,
            LandingContext(
                operator_id=publication.operator_id,
                landed_operator_ids=[operator for operator, _ in landed],
                landed_patches=[patch for _, patch in landed],
                intent=f"{publication.operator_name} in {publication.kernel_path}",
            ),
        )
        if not merge.applied:
            return _refusal(
                publication,
                "reverted_apply_conflict" if merge.conflicted else "reverted_apply_failed",
                merge.error + merge.note(),
                head_before,
            )

        self._rounds_run += 1
        try:
            validation = await self._validate(publication)
        except Exception as error:  # isolation boundary: one publication's E2E validation
            log.warning("E2E validation raised for %s: %s", publication.operator_id, error, exc_info=True)
            reason = f"E2E validation raised: {error}" + _revert_note(repo, publication.patch_path)
            return _refusal(publication, "reverted_e2e_failed", reason, head_before)
        kept = (
            str(validation.get("status") or "ok").lower() == "ok" and str(validation.get("decision")).upper() == "KEEP"
        )
        if not kept:
            reason = str(validation.get("error") or validation.get("decision_reason") or "E2E did not KEEP")
            return _refusal(
                publication,
                "reverted_e2e_failed",
                reason + _revert_note(repo, publication.patch_path),
                head_before,
                **_measured(validation),
            )

        committed, commit_note = _git_commit_kept(
            repo, f"hyperloom: keep KernelForge rewrite {publication.operator_id}", touched
        )
        # A KEEP is only durable once HEAD carries it, so ask Git rather than the note.
        keep_commit = _head_commit(repo)
        if not committed or not keep_commit or keep_commit == head_before:
            # Pods, compiled artifacts and the JIT tree come back from the apply
            # manifest first; the local worktree is reversed after, because its
            # own backup holds the patched bytes.
            settled = _settle_apply_manifest(validation, kept=False) + _revert_note(repo, publication.patch_path)
            reason = (commit_note or "git commit did not advance HEAD") + settled
            return _refusal(publication, "reverted_commit_failed", reason, head_before)

        record_reason = await self._persist_keep(publication, validation, keep_commit)
        self._landed.setdefault(repo, []).append((publication.operator_id, publication.patch_path))
        return _refusal(
            publication,
            "kept",
            record_reason + merge.note() + _settle_apply_manifest(validation, kept=True),
            head_before,
            merge_strategy=merge.strategy,
            integration_head_after=keep_commit,
            keep_commit=keep_commit,
            **_measured(validation),
        )


def _measured(validation: dict[str, Any]) -> dict[str, float]:
    """The throughput the E2E run reported, as ``PatchIntegrationResult`` fields."""
    return {key: float(validation.get(key) or 0.0) for key in ("new_tput", "gain_pct")}


async def integrate_controller_patches(
    *,
    patches_root: str | Path,
    session: KeepSession,
    validator: PatchValidator | None = None,
    admit_round: RoundAdmission = lambda _rounds_run: None,
) -> ControllerIntegrationSummary:
    """Apply and E2E-validate complete Controller patches in task-priority order.

    Args:
        patches_root: The Controller's published patch directory.
        session: The session every KEEP is recorded into.
        validator: Runs the E2E decision for one publication; defaults to the
            optimizer's own integrate handler.
        admit_round: Asked before each patch is applied; a stop leaves the patch
            unapplied and unmeasured, recorded under the stop's class.
    """
    integration_root = Path(patches_root).resolve().parent.parent / "integration"
    results_dir = integration_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    async def persist_keep(
        publication: ControllerPatchPublication,
        validation: dict[str, Any],
        keep_commit: str,
    ) -> str:
        try:
            layer = _committed_source_layer(
                publication.repo_root,
                keep_commit,
                publication.operator_id,
                session.session_dir,
                str(session.shared_state.framework or ""),
            )
            await session.record_keep({**_keep_result(publication, validation, keep_commit), **layer})
            session.shared_state.save(Path(session.session_dir))
        except Exception as error:  # isolation boundary: one committed publication's KEEP recording
            log.warning("KEEP %s committed but recording it failed: %s", publication.operator_id, error, exc_info=True)
            return f"Git KEEP committed; SharedState recording failed: {error}"
        return ""

    run = _ControllerIntegration(
        validator or (lambda publication: _default_validator(publication, session_dir=Path(session.session_dir))),
        persist_keep,
        admit_round,
    )
    results: list[PatchIntegrationResult] = []
    for index, patch_dir in enumerate(_priority_ordered_patch_dirs(patches_root)):
        try:
            publication = load_controller_publication(patch_dir)
        except ControllerPublicationError as error:
            result = PatchIntegrationResult(patch_dir.name, "skipped_invalid", str(error))
        else:
            result = await run.integrate(publication)
        results.append(result)
        _write_result(results_dir, index, result)

    kept = sum(result.status == "kept" for result in results)
    reverted = sum(result.status.startswith("reverted_") for result in results)
    # "completed" says the loop ran, which is not the same as the loop having done anything.
    admitted_nothing = bool(results) and kept == 0 and reverted == 0
    summary = ControllerIntegrationSummary(
        status="no_patch_admitted" if admitted_nothing else "completed",
        results=tuple(results),
        kept_count=kept,
        reverted_count=reverted,
        skipped_count=len(results) - kept - reverted,
        results_dir=str(results_dir),
    )
    atomic_write_json(integration_root / "summary.json", summary.to_dict(), fmt=JsonFormat(trailing_newline=True))
    return summary


__all__ = [
    "ControllerIntegrationSummary",
    "KeepSession",
    "PatchIntegrationResult",
    "integrate_controller_patches",
]
