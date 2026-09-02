# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sequential Git and E2E integration of KernelForge Controller patches."""

from __future__ import annotations

import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hyperloom.common.io import atomic_write_json
from hyperloom.orchestrator.actions.executors._patch_snapshot import (
    _git_commit_kept,
    _patch_touched_paths,
)
from hyperloom.orchestrator.actions.executors.integrate_patch import (
    _git_apply,
    _git_checkout_clean,
)

from .controller_publication import (
    ControllerPatchPublication,
    ControllerPublicationError,
    discover_controller_patch_dirs,
    load_controller_publication,
)


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


def _git_output(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return completed.stdout.strip()


def _write_result(results_dir: Path, index: int, result: PatchIntegrationResult) -> None:
    atomic_write_json(
        results_dir / f"{index:04d}.json",
        asdict(result),
        trailing_newline=True,
    )


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
            "patch_write_paths": list(publication.manifest.get("changed_files") or []),
            "_preapplied_git_patch": True,
        },
        session_dir=session_dir,
    )


async def integrate_controller_patches(
    *,
    patches_root: str | Path,
    session_dir: Path,
    shared_state: Any,
    validator: PatchValidator | None = None,
) -> ControllerIntegrationSummary:
    """Apply and E2E-validate every complete Controller patch in filename order."""
    integration_root = Path(patches_root).resolve().parent.parent / "integration"
    results_dir = integration_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    validate = validator or (
        lambda publication: _default_validator(
            publication,
            session_dir=Path(session_dir),
        )
    )
    from hyperloom.orchestrator.framework.paths import resolve_patch_target_roots

    configured_roots = [Path(root).expanduser().resolve() for root in resolve_patch_target_roots() if str(root).strip()]
    state_root = str(getattr(shared_state, "framework_repo_path", "") or "").strip()
    if state_root:
        configured_roots.append(Path(state_root).expanduser().resolve())
    allowed_roots = tuple(dict.fromkeys(configured_roots))
    results: list[PatchIntegrationResult] = []
    shared_base = ""
    shared_repo: Path | None = None
    shared_baseline_error = ""
    initial_head = ""

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
        if not any(
            publication.repo_root == root or publication.repo_root.is_relative_to(root) for root in allowed_roots
        ):
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="skipped_invalid",
                reason="publication repo_root is outside the configured patch target roots",
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(publication.repo_root),
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        if not shared_base:
            shared_base = publication.base_commit
            shared_repo = publication.repo_root
            try:
                initial_head = _git_output(shared_repo, "rev-parse", "HEAD").lower()
            except Exception as error:
                shared_baseline_error = f"could not read integration Git HEAD: {error}"
            else:
                shared_baseline_error = (
                    ""
                    if initial_head == shared_base
                    else f"integration HEAD {initial_head} does not match controller base {shared_base}"
                )

        if shared_baseline_error:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="skipped_baseline_mismatch",
                reason=shared_baseline_error,
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(publication.repo_root),
                integration_head_before=initial_head,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        if publication.base_commit != shared_base or publication.repo_root != shared_repo:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="skipped_baseline_mismatch",
                reason="publication does not share the Controller integration baseline",
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(publication.repo_root),
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        repo = publication.repo_root
        try:
            head_before = _git_output(repo, "rev-parse", "HEAD").lower()
            clean = _git_output(repo, "status", "--porcelain", "--untracked-files=no")
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
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="skipped_dirty_worktree",
                reason="integration repository has uncommitted tracked changes",
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
            _git_checkout_clean(repo)
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_apply_failed",
                reason=apply_error or "git apply failed",
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        try:
            validation = await validate(publication)
        except Exception as error:
            _git_checkout_clean(repo)
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_e2e_failed",
                reason=f"E2E validation raised: {error}",
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
            _git_checkout_clean(repo)
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_e2e_failed",
                reason=str(validation.get("error") or validation.get("decision_reason") or "E2E did not KEEP"),
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

        touched = _patch_touched_paths(repo, [publication.patch_path])
        committed, commit_note = _git_commit_kept(
            repo,
            f"hyperloom: keep KernelForge rewrite {publication.operator_id}",
            touched,
        )
        if not committed or commit_note:
            _git_checkout_clean(repo)
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_commit_failed",
                reason=commit_note or "git commit failed",
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

        keep_commit = _git_output(repo, "rev-parse", "HEAD").lower()
        try:
            _record_keep(
                shared_state,
                publication,
                validation,
                keep_commit,
                Path(session_dir),
            )
        except Exception as error:
            record_reason = f"Git KEEP committed; SharedState recording failed: {error}"
        else:
            record_reason = ""
        result = PatchIntegrationResult(
            operator_id=publication.operator_id,
            status="kept",
            reason=record_reason,
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
    summary = ControllerIntegrationSummary(
        status="completed",
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
