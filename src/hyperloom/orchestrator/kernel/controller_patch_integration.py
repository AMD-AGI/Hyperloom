# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Sequential Git and E2E integration of KernelForge Controller patches."""

from __future__ import annotations

import contextlib
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
    _git_apply_reverse,
    _git_restore_to_head,
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
    """Undo one applied patch without touching a path the patch never named.

    ``_git_checkout_clean`` is not usable here. It ends in ``git clean -fd``,
    which deletes every untracked file in the repository, and the admission check
    above asks ``git status`` with ``--untracked-files=no`` -- so an operator's
    own untracked notes or scratch directory pass admission and would then be
    destroyed by the first patch that fails. The legacy integrate path can afford
    that clean because it stashes the working tree first; this path never
    stashes, it declines a dirty repository instead, so its revert has to be
    scoped to the patch the same way its commit already is.

    Reversing the diff is the scoped equivalent: it restores what the patch
    modified and removes what it created, and names nothing else.
    """
    touched = _patch_touched_paths(repo, [patch_path])
    if touched:
        # A commit attempt that failed after ``git add`` leaves the patched
        # content staged, and reversing the working tree does not unstage it --
        # which would make the next patch see a dirty index and skip.
        with contextlib.suppress(Exception):
            _git_output(repo, "reset", "--quiet", "HEAD", "--", *touched)
    reversed_ok, reverse_error = _git_apply_reverse(repo, patch_path)
    if reversed_ok:
        return True, ""
    # A reverse apply refuses a partially applied patch, which is the state a
    # failed forward apply leaves. Restore every path HEAD still has a version
    # of; a path HEAD does not know is one the patch created, and it is left in
    # place rather than removed, because at this point nothing can prove it was
    # not already the operator's own untracked file.
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
    """Revert one patch and render what happened as a reason suffix.

    Every caller drops into this on its way to a ``reverted_`` status, and a
    revert that could not finish changes what the next patch will see, so it
    belongs in the recorded reason rather than in a discarded return value.
    """
    reverted, note = _revert_patch(repo, patch_path)
    if not reverted:
        return f" (revert failed: {note})"
    return f" (revert: {note})" if note else ""


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
            # The Controller's Git-derived scope when it has one; the optimizer's
            # own manifest only as a fallback for a publication without it.
            "patch_write_paths": list(publication.changed_files)
            or list(publication.manifest.get("changed_files") or []),
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
    # One base commit per repository rather than one repository per run. A patch
    # only ever applies to its own repository, so two independent repositories
    # cannot conflict and each can carry its own baseline; a second base within
    # one repository still cannot. Keyed by resolved root, established by the
    # first publication naming that repository.
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
        # Hyperloom dirties the framework tree itself -- the TraceLens and
        # ck-blockscale instrumentation are patched in place and stay uncommitted
        # for the life of the session, and every other lane leaves its own KEEP
        # uncommitted too -- so a repository-wide check is unsatisfiable in a real
        # session and threw away every patch that reached it. The narrower
        # question is the one that matters anyway: apply, commit and revert are
        # each already scoped to these paths, so dirt anywhere else cannot be
        # confused with this patch's own change.
        touched = _patch_touched_paths(repo, [publication.patch_path])
        try:
            head_before = _git_output(repo, "rev-parse", "HEAD").lower()
            # A patch whose paths cannot be read is one nothing can be scoped to,
            # so it falls back to asking about the whole tree.
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
        # A KEEP is only durable once HEAD carries it, so ask Git rather than the
        # note. The callee reports a benign no-op -- nothing staged, or staged
        # content already matching HEAD -- as success with a note and no commit,
        # and it documents that note as carrying "any detail", so a note alone
        # must neither admit an uncommitted patch nor discard a committed one.
        keep_commit = _head_commit(repo)
        if not committed or not keep_commit or keep_commit == head_before:
            result = PatchIntegrationResult(
                operator_id=publication.operator_id,
                status="reverted_commit_failed",
                reason=(commit_note or "git commit did not advance HEAD") + _revert_note(repo, publication.patch_path),
                base_commit=publication.base_commit,
                best_commit=publication.best_commit,
                repo_root=str(repo),
                integration_head_before=head_before,
            )
            results.append(result)
            _write_result(results_dir, index, result)
            continue

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
    # "completed" says the loop ran, which is not the same as the loop having
    # done anything. A run whose every patch was refused before it was even
    # measured is an environment failure, and reporting it the same way as a run
    # that graded its patches and kept none hides hours of work having been
    # dropped at the door.
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
