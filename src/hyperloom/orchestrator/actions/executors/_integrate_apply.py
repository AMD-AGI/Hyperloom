# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Applying a resolved patch series to the live tree, and refusing the attempt when it will not apply."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.session.session_paths import runs_dir

from ._enablement_config import _established_enablement_config, _merge_established_server_args
from ._git import _git_head_sha, _run_git_cp
from ._integrate_attempt import IntegrateAttempt
from ._integrate_refusals import _preflight_missing_targets
from ._integrate_results import (
    _Deliverable,
    _KbLesson,
    _apply_failed,
    _no_patches_result,
    _pending_sentinel,
    _reauthor_fields,
    _unresolved_root_rejection,
)
from ._nogit_patch import _is_git_tree
from ._patch_git import _apply_patch_series, _git_stash_if_dirty
from ._patch_sources import (
    _candidate_mutation_roots,
    _note_pre_mutation_head,
    _resolve_framework_root,
    _sole_patch_root,
    resolved_explicit_root,
)
from ._setup_replay import replay_setup_commands
from .source_tree_identity import new_recovery_record

log = logging.getLogger(__name__)


class ApplyStageMixin:
    """Open the mutation window, apply the patch series to the live tree, and refuse the attempt when it does not apply."""

    async def _stage_apply(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Setup replay, patch/artifact apply, pending_integrate sentinel.

        Returns an early-exit result dict on failure/no-patches/apply_only,
        or None to continue to bench+gate. Records mutations on the attempt.
        """
        shared_state = attempt.shared_state
        # Read back what earlier rounds established before either publish point
        # below decides what this round launches with.
        inherited_args, inherited_envs = _established_enablement_config(params, shared_state)
        # The HEADs are recorded first because the setup commands are the round's
        # first mutation: an install writes into the same trees the patches land in.
        is_enablement = bool(params.get("enablement"))
        for candidate in _candidate_mutation_roots(params=params, done_payload=attempt.done_payload):
            _note_pre_mutation_head(attempt, candidate, enablement=is_enablement, session_dir=self.session_dir)
        if is_enablement:
            await self._record_closure_baseline(attempt, params)
            setup_refusal = await replay_setup_commands(attempt, params=params, session_dir=self.session_dir)
            if setup_refusal is not None:
                return setup_refusal

        deliverable = _Deliverable.resolve(attempt, params)
        attempt.switch_manifest = deliverable.switch_manifest
        if deliverable.artifact_runtime_errors:
            return await self._refuse_apply(
                attempt,
                params,
                extra,
                "artifact_not_runtime_ready",
                deliverable.artifact_runtime_errors,
                artifacts_applied=[],
                artifact_errors=deliverable.artifact_runtime_errors,
                reason=(
                    "AITER GEMM model-config artifacts must contain measured, "
                    "non-placeholder rows for the target GPU architecture and CU count"
                ),
            )

        if not (
            deliverable.patch_paths
            or deliverable.extra_server_args
            or deliverable.extra_envs
            or deliverable.artifact_specs
            or attempt.setup_commands_applied
            or (bool(attempt.attempt_venv_root) and not shared_state.enablement.kept_stack_action)
        ):
            if params.get("enablement_launch_only"):
                # Launch-only mode: skip the no-patches early-return and fall through to bench.
                output_root = runs_dir(self.session_dir, "integrate_patch", attempt.specialist_task_id)
                output_root.mkdir(parents=True, exist_ok=True)
                attempt.output_root = output_root
                attempt.extra_envs_applied = dict(inherited_envs)
                attempt.extra_server_args_applied = inherited_args
                return None
            return _no_patches_result(attempt, deliverable, params)

        # An acquired runtime is what the boot resolves, so the shared framework
        # tree is not what the server imports and the root the specialist authored
        # against does not describe the tree this round's patches land in.
        patch_paths = deliverable.patch_paths
        runtime_source_root = attempt.runtime_source_root
        explicit_root = runtime_source_root or str(params.get("framework_source_root") or "").strip() or None
        framework_root = (
            resolved_explicit_root(runtime_source_root)
            if runtime_source_root
            # Off the loop: settling a tie between matching trees runs git once per patch.
            else await asyncio.to_thread(
                _resolve_framework_root,
                explicit_root,
                patch_paths=patch_paths,
                recorded_root=_sole_patch_root(
                    attempt.done_payload, patch_paths, specialist_workspace=attempt.specialist_workspace
                ),
            )
        )
        if patch_paths and framework_root is None:
            unresolved = _apply_failed(
                attempt,
                **_unresolved_root_rejection(explicit_root, patch_paths),
                **_reauthor_fields(params, patch_paths),
            )
            # A set spanning two trees is exactly what fails root resolution here,
            # so this is the result that has to carry the reason onward.
            if (attempt.done_payload or {}).get("patches_span_multiple_roots"):
                unresolved["patches_span_multiple_roots"] = True
            return unresolved
        if patch_paths:
            missing_records = _preflight_missing_targets(framework_root, patch_paths)
            if missing_records:
                return await self._refuse_apply(
                    attempt,
                    params,
                    extra,
                    "patch_target_missing",
                    missing_records,
                    **_reauthor_fields(
                        params,
                        patch_paths,
                        advisory="patch target file(s) absent from framework_source_root "
                        f"{framework_root}; author patches only against files that "
                        "exist in the installed framework tree (inspect it with "
                        "Glob/Grep before writing the diff).",
                    ),
                )

        stash_refusal = self._begin_mutation(attempt, deliverable, params, extra, framework_root=framework_root)
        if stash_refusal is not None:
            return stash_refusal

        recovery = attempt.pending["recovery"]
        recovery_root = Path(recovery["root"])
        patch_paths = self._admit_localization(attempt, framework_root, deliverable.patch_paths)
        recovery["phase"] = "ready"
        if shared_state is not None:
            shared_state.save(self.session_dir)

        # An artifact preimage has to describe the tree as this attempt found
        # it, so it is taken before the patches run: a patch that creates the
        # very file an artifact then overwrites would otherwise be recorded as
        # pre-existing, and the recovery would leave it behind instead of
        # deleting it. Nothing is installed here, so a failure leaves the tree
        # exactly as the stash left it.
        if deliverable.artifact_specs:
            backup_errors = self._backup_artifacts(
                deliverable.artifact_specs,
                backup_root=recovery_root / "artifact_backups",
            )
            if backup_errors:
                return await self._refuse_apply(
                    attempt,
                    params,
                    extra,
                    "artifact_backup_failed",
                    deliverable.artifact_resolve_errors + backup_errors,
                    artifacts_applied=[],
                    workspace=str(attempt.output_root),
                )
            recovery["artifacts_prepared"] = True
            if shared_state is not None:
                shared_state.save(self.session_dir)

        apply_errors, apply_feedback = _apply_patch_series(
            attempt,
            patch_paths,
            git_head=str(recovery.get("git_head") or ""),
            backup_root=recovery_root / "patch_backups",
        )
        if apply_errors:
            return await self._refuse_apply(
                attempt,
                params,
                extra,
                "git_apply_failed",
                apply_errors,
                workspace=str(attempt.output_root),
                **_reauthor_fields(params, patch_paths, feedback=apply_feedback),
            )

        if deliverable.artifact_specs:
            applied_artifacts, artifact_apply_errors = self._apply_artifacts(
                deliverable.artifact_specs,
                backup_root=recovery_root / "artifact_backups",
            )
            attempt.applied_artifacts = applied_artifacts
            if artifact_apply_errors:
                return await self._refuse_apply(
                    attempt,
                    params,
                    extra,
                    "artifact_install_failed",
                    deliverable.artifact_resolve_errors + artifact_apply_errors,
                    artifacts_applied=[],
                    workspace=str(attempt.output_root),
                )

        # Inherited first, this round last: inheriting is not pinning.
        attempt.extra_server_args_applied = _merge_established_server_args(
            inherited_args, deliverable.extra_server_args
        )
        attempt.extra_envs_applied = {**inherited_envs, **deliverable.extra_envs}
        attempt.dropped_env_overrides = deliverable.dropped_env_overrides
        if params.get("apply_only"):
            return {
                "status": "applied_no_bench",
                "specialist_task_id": attempt.specialist_task_id,
                "patches_applied": [str(patch) for patch in attempt.applied],
                "patches_reverted": [],
                "artifacts_applied": attempt.applied_artifacts,
                "extra_server_args_applied": attempt.extra_server_args_applied,
                "extra_envs_applied": attempt.extra_envs_applied,
                "dropped_env_overrides": deliverable.dropped_env_overrides,
                "reason": "apply_only=True; benchmark skipped",
                "workspace": str(attempt.output_root),
            }
        return None

    def _begin_mutation(
        self,
        attempt: IntegrateAttempt,
        deliverable: _Deliverable,
        params: dict[str, Any],
        extra: dict[str, Any],
        *,
        framework_root: Path | None,
    ) -> dict[str, Any] | None:
        """Open this attempt's workspace and recovery evidence, then bank the operator's work.

        The sentinel and the recovery record are written before any framework tree
        mutation; the Coordinator clears the sentinel after promoting the final result.
        """
        shared_state = attempt.shared_state
        task_runs = runs_dir(self.session_dir, "integrate_patch", attempt.task_id)
        output_root = Path(params.get("output_dir") or extra.get("workspace") or task_runs)
        output_root.mkdir(parents=True, exist_ok=True)
        attempt.output_root = output_root
        git_head = ""
        if framework_root is not None and _is_git_tree(framework_root):
            git_head = _git_head_sha(framework_root)
            if deliverable.patch_paths and not git_head:
                # Patches on a git tree keep no per-file backups, so the recorded HEAD is their only way back.
                raise OSError(f"cannot read the HEAD of {framework_root} to restore to")
        task_runs.mkdir(parents=True, exist_ok=True)
        recovery_root = Path(tempfile.mkdtemp(prefix="recovery-", dir=task_runs)).resolve()
        recovery = new_recovery_record(framework_root, git_head=git_head, recovery_root=recovery_root)
        attempt.pending = _pending_sentinel(attempt, deliverable, framework_root=framework_root, recovery=recovery)
        if shared_state is not None:
            shared_state.pending_integrate = attempt.pending
            try:
                shared_state.save(self.session_dir)
            except OSError:
                log.exception("integrate_patch: failed to persist pending_integrate sentinel")

        # Normally already recorded before the setup commands; a root that
        # resolved only here is still recorded before the stash and the apply.
        _note_pre_mutation_head(
            attempt, framework_root, enablement=bool(params.get("enablement")), session_dir=self.session_dir
        )
        stash_state, stash_note = _git_stash_if_dirty(framework_root)
        if stash_state == "failed":
            recovery["phase"] = "restored"
            log.error(
                "integrate_patch: cannot stash user changes in %s: %s; aborting to avoid data loss",
                framework_root,
                stash_note,
            )
            return _apply_failed(
                attempt,
                "stash_failed",
                f"refusing to proceed: user changes could not be stashed ({stash_note})",
            )

        # The stash is on the stack and the tree is about to be mutated, so
        # ``__call__``'s undo has to be able to see both before anything writes.
        attempt.framework_root = framework_root
        if stash_state == "stashed":
            cp = _run_git_cp(["-C", str(framework_root), "rev-parse", stash_note], timeout=30.0)
            if cp is None or cp.returncode != 0:
                raise OSError("cannot persist the exact pre-candidate stash identity")
            recovery["stash_oid"] = cp.stdout.strip()
        return None

    async def _refuse_apply(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
        error_class: str,
        error: Any,
        **fields: Any,
    ) -> dict[str, Any]:
        """Refuse the round at the apply stage, filing the lesson against the patch's PR identity."""
        await self._maybe_write_framework_kb_record(
            _KbLesson("rejected_apply_fail", 0.0), done_payload=attempt.done_payload, params=params, extra=extra
        )
        return _apply_failed(attempt, error_class, error, **fields)
