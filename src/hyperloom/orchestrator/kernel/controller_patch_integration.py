# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sequential Git and E2E integration of KernelForge Controller patches."""

from __future__ import annotations

import contextlib
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom.orchestrator.actions.executors._patch_snapshot import (
    _git_commit_kept,
    _patch_touched_paths,
)
from hyperloom.orchestrator.actions.executors.integrate_patch import (
    _git_apply,
    _git_apply_reverse,
    _git_restore_to_head,
)

from .controller_publication import (
    ControllerPatchPublication,
    ControllerPublicationError,
    discover_controller_patch_dirs,
    load_controller_publication,
)
from .kth_qualification import KthQualificationProvider, KthQualificationResult


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
    kth_verdict: str = ""
    kth_subject_digest: str = ""
    kth_primary_detector: str = ""
    kth_artifacts_dir: str = ""
    performance_reached: bool = False
    repair_feedback: dict[str, Any] = field(default_factory=dict)


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


def _kth_fields(result: KthQualificationResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        "kth_verdict": result.verdict,
        "kth_subject_digest": result.subject_digest,
        "kth_primary_detector": result.primary_detector,
        "kth_artifacts_dir": result.artifacts_dir,
        "performance_reached": result.performance_reached,
        "repair_feedback": dict(result.repair_feedback or {}),
    }


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return completed.stdout.strip()


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
        with contextlib.suppress(Exception):
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
    from .request_handlers import _maybe_finalize_kernel_patch, _maybe_revert_kernel_patch

    apply_result = validation.get("apply_result")
    if not isinstance(apply_result, dict) or not apply_result.get("manifest_path"):
        return ""
    stage = "finalize" if kept else "revert"
    outcome = _maybe_finalize_kernel_patch(apply_result) if kept else _maybe_revert_kernel_patch(apply_result)
    if str(outcome.get("status") or "") in {"ok", "skipped"}:
        return ""
    return f" (patch {stage} incomplete: {outcome.get('error') or outcome.get('status')})"


def _write_result(results_dir: Path, index: int, result: PatchIntegrationResult) -> None:
    atomic_write_json(
        results_dir / f"{index:04d}.json",
        asdict(result),
        trailing_newline=True,
    )


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
        "source": "kernel_rewrite_controller",
    }


def _record_keep(
    shared_state: Any,
    publication: ControllerPatchPublication,
    validation: dict[str, Any],
    keep_commit: str,
    session_dir: Path,
) -> None:
    new_tput = float(validation.get("new_tput") or 0.0)
    variant_name = f"kernel_rewrite_controller:{publication.operator_id}"
    entry = {
        "action": "integrate",
        "scope": "source_patch",
        "variant_name": variant_name,
        "kernel_id": publication.operator_id,
        "operator_id": publication.operator_id,
        "source_file": str(publication.repo_root / publication.kernel_path),
        "patch_path": str(publication.patch_path),
        "base_sha": publication.base_commit,
        "keep_commit": keep_commit,
        "tput": new_tput,
        "gain_pct": float(validation.get("gain_pct") or 0.0),
        "source": "kernel_rewrite_controller",
    }
    shared_state.optimization_stack = [
        *[
            item
            for item in (getattr(shared_state, "optimization_stack", None) or [])
            if not (isinstance(item, dict) and str(item.get("operator_id") or "") == publication.operator_id)
        ],
        entry,
    ]
    current_best = (
        dict(shared_state.current_best) if isinstance(getattr(shared_state, "current_best", None), dict) else {}
    )
    current_best.update(
        {
            "action": "integrate",
            "variant_name": variant_name,
            "tput": new_tput,
            "source_file": entry["source_file"],
            "patch_path": entry["patch_path"],
            "keep_commit": keep_commit,
        }
    )
    if validation.get("extra_server_args") is not None:
        current_best["extra_server_args"] = validation.get("extra_server_args")
    if isinstance(validation.get("extra_envs"), dict):
        current_best["extra_envs"] = dict(validation["extra_envs"])
    shared_state.current_best = current_best
    baseline = float(getattr(shared_state, "baseline_tput", 0.0) or 0.0)
    if baseline > 0 and new_tput > 0:
        shared_state.cumulative_gain_validated = (new_tput / baseline - 1.0) * 100.0
    shared_state.save(session_dir)


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
    record_keep: KeepRecorder | None = None,
    validator: PatchValidator | None = None,
    kth_provider: KthQualificationProvider | None = None,
) -> ControllerIntegrationSummary:
    """Apply and E2E-validate every complete Controller patch in filename order.

    Args:
        patches_root: The Controller's published patch directory.
        session_dir: The session whose state the KEEPs are recorded into.
        shared_state: The live session state, persisted after each recorded KEEP.
        record_keep: Session-owned writeback for AgentX; other workloads use the local recorder.
        validator: Runs the E2E decision for one publication; defaults to the
            optimizer's own integrate handler.
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

    for index, patch_dir in enumerate(discover_controller_patch_dirs(patches_root)):
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
            try:
                pinned_heads[repo] = _git_output(repo, "rev-parse", "HEAD").lower()
            except Exception as error:
                pinned_heads[repo] = ""
                pin_errors[repo] = f"could not read integration Git HEAD: {error}"
            else:
                pin_errors[repo] = (
                    ""
                    if pinned_heads[repo] == publication.base_commit
                    else (
                        f"integration HEAD {pinned_heads[repo]} does not match "
                        f"controller base {publication.base_commit}"
                    )
                )

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
        except Exception as error:
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

        applies, apply_error = _git_apply(
            repo,
            publication.patch_path,
            three_way=False,
            check_only=True,
        )
        if not applies:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_apply_conflict",
                reason=apply_error or "git apply check failed",
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue
        applied, apply_error = _git_apply(
            repo,
            publication.patch_path,
            three_way=False,
            check_only=False,
        )
        if not applied:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_apply_failed",
                reason=(apply_error or "git apply failed") + _revert_note(repo, publication.patch_path),
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        kth_result = None
        provider = kth_provider
        if publication.kth_plan_id:
            provider = provider or KthQualificationProvider()
            try:
                kth_result = provider.qualify(
                    publication,
                    artifacts_root=Path(session_dir) / "kth_qualification",
                )
            except Exception as error:  # noqa: BLE001
                kth_result = KthQualificationResult(
                    status="needs_review",
                    reason=f"KTH provider infrastructure failure: {error}",
                    verdict="Inconclusive",
                    request_id="",
                    subject_digest="",
                    primary_detector="",
                    artifacts_dir=str(Path(session_dir) / "kth_qualification"),
                    repair_feedback={"instruction": "Repair the KTH provider failure and replay qualification."},
                )
            if not kth_result.eligible:
                result = PatchIntegrationResult(
                    operator_id=publication.operator_id,
                    status=kth_result.status,
                    reason=kth_result.reason + _revert_note(repo, publication.patch_path),
                    base_commit=publication.base_commit,
                    best_commit=publication.best_commit,
                    repo_root=str(repo),
                    integration_head_before=head_before,
                    **_kth_fields(kth_result),
                )
                results.append(result)
                _write_result(results_dir, index, result)
                continue
            try:
                kth_result = provider.mark_performance_reached(kth_result)
            except Exception as error:  # noqa: BLE001
                failed_result = KthQualificationResult(
                    status="needs_review",
                    reason=f"Could not persist the KTH performance boundary: {error}",
                    verdict=kth_result.verdict,
                    request_id=kth_result.request_id,
                    subject_digest=kth_result.subject_digest,
                    primary_detector=kth_result.primary_detector,
                    artifacts_dir=kth_result.artifacts_dir,
                    repair_feedback={"instruction": "Repair artifact persistence and replay qualification."},
                )
                result = PatchIntegrationResult(
                    operator_id=publication.operator_id,
                    status=failed_result.status,
                    reason=failed_result.reason + _revert_note(repo, publication.patch_path),
                    base_commit=publication.base_commit,
                    best_commit=publication.best_commit,
                    repo_root=str(repo),
                    integration_head_before=head_before,
                    **_kth_fields(failed_result),
                )
                results.append(result)
                _write_result(results_dir, index, result)
                continue

        try:
            validation = await validate(publication)
        except Exception as error:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_e2e_failed",
                reason=f"E2E validation raised: {error}" + _revert_note(repo, publication.patch_path),
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
                **_kth_fields(kth_result),
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
                **_kth_fields(kth_result),
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
                **_kth_fields(kth_result),
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        try:
            if record_keep is None:
                _record_keep(shared_state, publication, validation, keep_commit, Path(session_dir))
            else:
                await record_keep(_keep_result(publication, validation, keep_commit))
                shared_state.save(Path(session_dir))
        except Exception as error:
            record_reason = f"Git KEEP committed; SharedState recording failed: {error}"
        else:
            record_reason = ""
        result = PatchIntegrationResult(
            operator_id=publication.operator_id,
            status="kept",
            reason=record_reason + _settle_apply_manifest(validation, kept=True),
            base_commit=publication.base_commit,
            best_commit=publication.best_commit,
            repo_root=str(repo),
            integration_head_before=head_before,
            integration_head_after=keep_commit,
            keep_commit=keep_commit,
            new_tput=float(validation.get("new_tput") or 0.0),
            gain_pct=float(validation.get("gain_pct") or 0.0),
            **_kth_fields(kth_result),
        )
        results.append(result)
        _write_result(results_dir, index, result)

    kept = sum(result.status == "kept" for result in results)
    reverted = sum(
        result.status.startswith("reverted_") or result.status in {"kth_blocked", "kth_inconclusive", "needs_review"}
        for result in results
    )
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
    atomic_write_json(
        integration_root / "summary.json",
        summary.to_dict(),
        trailing_newline=True,
    )
    return summary


__all__ = [
    "ControllerIntegrationSummary",
    "PatchIntegrationResult",
    "integrate_controller_patches",
]
