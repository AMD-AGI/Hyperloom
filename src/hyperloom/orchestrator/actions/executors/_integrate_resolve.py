# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The front of one integration attempt: resolving what it integrates, provisioning its runtime and localizing the source."""

from __future__ import annotations

import asyncio
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import to_float
from hyperloom.inference_optimizer.session.session_paths import enablement_stacks_dir, runs_dir

from ...state.shared_state import inject_stack_base_params
from ._enablement_config import _enablement_gap, _stamp_framework_kb_provenance
from ._integrate_attempt import IntegrateAttempt
from ._integrate_refusals import _enablement_lane_closed_skip, _enforce_critic_gate
from ._patch_git import _detect_p_level
from ._patch_source_pr import _candidate_slug, materialize_candidate_patches
from ._integrate_vocabulary import PATCH_SOURCE_UPSTREAM_PR, resolve_patch_source
from ._patch_sources import (
    _acquiring_round,
    _done_field,
    _first_ref_in_repo,
    _read_done_payload,
    _resolve_framework_root,
)

log = logging.getLogger(__name__)

_LAUNCH_ONLY_MUTATION_FIELDS: tuple[str, ...] = (
    "patches",
    "artifacts",
    "config_changes",
    "enablement_setup_commands",
)


class ResolveStageMixin:
    """Resolve the patches an attempt integrates, provision the runtime it measures in, and localize the source it edits."""

    def _stage_resolve_launch_only(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        shared_state: Any,
    ) -> dict[str, Any] | None:
        """Give a launch-only round its own scratch workspace, or refuse a round that would mutate."""
        mutation_fields = [key for key in _LAUNCH_ONLY_MUTATION_FIELDS if params.get(key)]
        if mutation_fields:
            return {
                "status": "failed",
                "error_class": "launch_only_mutation_forbidden",
                "error": (
                    "enablement_launch_only accepts a pre-built runtime_override "
                    f"but no mutation fields; received {mutation_fields}"
                ),
                "patches_applied": [],
                "patches_reverted": [],
            }
        scratch = runs_dir(self.session_dir, "integrate_patch", attempt.task_id)
        scratch.mkdir(parents=True, exist_ok=True)
        attempt.specialist_task_id = attempt.task_id
        attempt.shared_state = shared_state
        attempt.specialist_workspace = scratch
        return None

    async def _stage_resolve(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        extra: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Guards (multi-node, task-id, workspace, Critic) + param normalisation.

        Returns an early-exit result dict on failure, or None to continue.
        Resolved inputs belong to this attempt, not to the runner context.
        """
        from ._multi_node_env import is_multi_node

        shared_state = extra.get("shared_state")
        if is_multi_node():
            return {
                "status": "skipped",
                "skipped_reason": "multi_node_unsupported",
                "specialist_task_id": str(params.get("specialist_task_id") or "").strip(),
                "patches_applied": [],
                "patches_reverted": [],
                "reason": (
                    "specialist integrate_patch is not supported in "
                    "multi-node mode (no git-diff pod fan-out); skipped "
                    "without applying any patch. Other actions "
                    "(baseline/profile/explore/conc_sweep/roofline) continue "
                    "normally. Use the kernel-agent integrate path (which "
                    "fans out via `multi_node apply-patch`) or run single-node."
                ),
            }
        pending = getattr(shared_state, "pending_integrate", None)
        if (
            isinstance(pending, dict)
            and pending
            and (pending.get("recovery") or {}).get("phase") not in ("restored", "accepted")
        ):
            return {
                "status": "failed",
                "error_class": "integrate_restore_incomplete",
                "error": "previous integration still requires recovery",
            }
        if (skipped := _enablement_lane_closed_skip(params, shared_state)) is not None:
            return skipped
        if shared_state is not None and not params.get("accuracy_baseline"):
            base_acc = to_float(getattr(shared_state, "baseline_accuracy", 0.0), 0.0)
            if base_acc > 0:
                params["accuracy_baseline"] = base_acc

        # Launch-only mode: pure bench of a pre-built runtime (no specialist, no Critic).
        if params.get("enablement_launch_only"):
            return self._stage_resolve_launch_only(attempt, params, shared_state)

        # Upstream-PR mode has no specialist to look up and no specialist
        # verdict to enforce: the Critic verdict on this proposal is the gate.
        if resolve_patch_source(params) == PATCH_SOURCE_UPSTREAM_PR:
            return self._stage_resolve_upstream_pr(attempt, params, shared_state)

        specialist_task_id = str(params.get("specialist_task_id") or "").strip()
        if not specialist_task_id:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": (
                    "integrate_patch requires params.specialist_task_id "
                    "(the completed specialist whose worktree carries "
                    "the patches to integrate)"
                ),
            }
        # Rebind from live current_best so a task queued before an Explore KEEP
        # still measures against the real stack top.
        if shared_state is not None:
            inject_stack_base_params(params, shared_state, anchor=True, overwrite=True)
        # Specialist workspace conventionally at runs/specialist/<id>/.
        specialist_workspace = runs_dir(self.session_dir, "specialist", specialist_task_id)
        if not specialist_workspace.is_dir():
            return {
                "status": "failed",
                "error_class": "missing_specialist",
                "error": (f"specialist workspace not found at {specialist_workspace}"),
                "specialist_task_id": specialist_task_id,
            }

        # Critic-verdict gate — enforced BEFORE any side effect (setup replay,
        # stash, patch/artifact apply, pod fan-out). Paths that bypass PolicyGate
        # (notably a queued/resume-dispatched task) are not re-validated there, so
        # a forged coordinator.db row with no genuine Critic verdict must be
        # rejected here, all-or-nothing, before it can install packages or mutate
        # the live framework tree. specialist_patch_verdicts is a Coordinator-only
        # field an LLM/forged row cannot write, and a legitimate
        # integrate_patch always has its verdict persisted before the queued task
        # is created (see intent_router._handle_single_verdict), so a genuine
        # task is unaffected. No-op when SharedState is absent.
        critic_reject = _enforce_critic_gate(shared_state, specialist_task_id)
        if critic_reject is not None:
            return critic_reject

        done_payload = _read_done_payload(specialist_workspace)
        _stamp_framework_kb_provenance(done_payload, params=params, shared_state=shared_state)

        attempt.specialist_task_id = specialist_task_id
        attempt.shared_state = shared_state
        attempt.specialist_workspace = specialist_workspace
        attempt.done_payload = done_payload
        return None

    def _stage_resolve_upstream_pr(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        shared_state: Any,
    ) -> dict[str, Any] | None:
        """Resolve an upstream-PR candidate into patches on the task's own scratch.

        Sets ``params['patches']`` so the shared apply stage below sees the same
        explicit-paths input a specialist lane produces, and points the
        workspace at this task's scratch dir since there is no specialist one.

        Args:
            attempt: The integration attempt receiving the resolved inputs.
            params: Task params, mutated with the resolved patch paths.
            shared_state: SharedState, or ``None``.

        Returns:
            A terminal result when the candidate has no permissive Critic
            verdict or could not be materialised, else ``None``.
        """
        candidate = params.get("candidate") or {}
        if not isinstance(candidate, dict) or not candidate:
            return {
                "status": "failed",
                "error_class": "missing_param",
                "error": "patch_source=upstream_pr requires params.candidate (the discovered PR row)",
                "patches_applied": [],
                "patches_reverted": [],
            }
        if shared_state is not None:
            # Same rebind the specialist lane does: a task queued before a KEEP
            # landed must still measure against the real stack top.
            inject_stack_base_params(params, shared_state, anchor=True, overwrite=True)

        # A queued or resume-dispatched task is not re-validated by PolicyGate,
        # and this lane fetches a remote diff into the live framework tree.
        critic_reject = _enforce_critic_gate(
            shared_state,
            str(params.get("framework_agent_candidate_id") or "").strip(),
        )
        if critic_reject is not None:
            return critic_reject

        task_id = attempt.task_id
        scratch = runs_dir(self.session_dir, "integrate_patch", task_id)
        scratch.mkdir(parents=True, exist_ok=True)
        framework_root = _resolve_framework_root(str(params.get("framework_source_root") or "").strip() or None)
        if framework_root is None:
            return {
                "status": "failed",
                "error_class": "framework_root_unresolved",
                "error": "cannot resolve a framework source root to apply the candidate to",
                "patches_applied": [],
                "patches_reverted": [],
            }

        materialized = materialize_candidate_patches(
            candidate=candidate,
            params=params,
            framework_root=framework_root,
            dest=scratch / f"{_candidate_slug(candidate)}.patch",
        )
        if materialized.failure is not None:
            return {
                **materialized.failure,
                "candidate": candidate,
                "patches_applied": [],
                "patches_reverted": [],
                "patch_source_mode": materialized.mode,
                "workspace": str(scratch),
            }
        params["patches"] = [str(p) for p in materialized.patches]
        params["patch_source_mode"] = materialized.mode

        attempt.specialist_task_id = task_id
        attempt.shared_state = shared_state
        attempt.specialist_workspace = scratch
        return None

    async def _stage_provision_attempt_runtime(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        specialist_task_id: str,
    ) -> dict[str, Any] | None:
        """Provision the attempt-scoped runtime this round boots on (Rung 3).

        The runtime the last kept round promoted is re-provisioned whatever this round asks for, since the rest
        of the stack was proven on it. With none kept, the framework adapter builds the runtime the specialist
        requested (``needs_runtime`` in its done payload); no-op when it requested none. Runs a disk preflight,
        delegates provision+probe to the framework adapter (off the event loop; an in-flight pip install is not
        killed if the await is cancelled), and on success stores the resolved runtime on the attempt for the gate
        to activate via the YAML-layer ``runtime_override``. Returns an early-exit ``reverted`` dict on any
        provision failure (no patch side effects yet), or ``None`` to continue.
        """
        if not _acquiring_round(params):
            return None

        from ...enablement.runtime.adapters import get_adapter
        from ...enablement.runtime.stack_actions import EnablementStackAction, RuntimeRequest

        request = RuntimeRequest.from_payload(_done_field(attempt.done_payload, "needs_runtime"))
        kept = attempt.shared_state.enablement.kept_stack_action
        if kept:
            action = EnablementStackAction.from_state(kept)
            if request is not None:
                log.info("integrate_patch: needs_runtime ignored; the kept %s runtime stays", action.framework)
        else:
            if request is None:
                return None
            framework = str(getattr(attempt.shared_state, "framework", "") or "").strip().lower()
            action = get_adapter(framework).build_stack_action(request)
            if action is None:
                return None

        attempt_dir = (
            enablement_stacks_dir(self.session_dir)
            / (action.framework or "unknown")
            / (specialist_task_id or "attempt")
        )

        from hyperloom.agents.framework.isolation import DiskPreflightError, disk_preflight

        try:
            disk_preflight(attempt_dir.parent, n_candidates=1)
        except DiskPreflightError as exc:
            return {
                "status": "reverted",
                "error_class": "disk_preflight_failed",
                "error": str(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "enablement": True,
                "reason": f"attempt-runtime provision aborted: {exc}",
            }
        except OSError as exc:
            log.warning("integrate_patch: disk preflight raised (%r); continuing", exc)

        adapter = get_adapter(action.framework)

        def _provision_and_probe():
            provisioned = adapter.provision(action, attempt_dir)
            if provisioned.ok and not adapter.probe(provisioned, action):
                from ...enablement.runtime.stack_actions import ProvisionResult as _PR

                return _PR(ok=False, log_path=provisioned.log_path, error="adapter probe failed after provision")
            return provisioned

        try:
            result = await asyncio.to_thread(_provision_and_probe)
        except (OSError, subprocess.SubprocessError) as exc:
            log.exception("integrate_patch: attempt-runtime provision raised")
            self._gc_attempt_dir(attempt_dir)
            return {
                "status": "reverted",
                "error_class": "provision_exception",
                "error": repr(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "enablement": True,
                "reason": f"attempt-runtime provision raised: {exc!r}",
            }

        if not result.ok:
            self._gc_attempt_dir(attempt_dir)
            return {
                "status": "reverted",
                "error_class": "provision_failed",
                "error": result.error,
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "enablement": True,
                "reason": f"attempt-runtime provision failed: {result.error}",
                "provision_result": result.to_state(),
            }

        # Record the attempt venv root on the action so KEEP can persist it and
        # resume/GC can find it.
        action = EnablementStackAction.from_state({**action.to_state(), "attempt_venv_root": result.runtime.venv_root})
        attempt.provision_result = result
        attempt.stack_action = action
        log.info(
            "integrate_patch: attempt runtime provisioned for %s (venv=%s, versions=%s)",
            action.framework,
            result.runtime.venv_root,
            result.installed_versions,
        )
        return None

    @staticmethod
    def _gc_attempt_dir(attempt_dir: Path) -> None:
        """Remove a half/failed attempt runtime's venv and checkout, keeping its provision log (best-effort)."""
        try:
            if attempt_dir.exists():
                for child in attempt_dir.iterdir():
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
        except OSError:
            log.debug("integrate_patch: attempt-dir GC failed for %s", attempt_dir, exc_info=True)

    async def _stage_localize_source(self, attempt: IntegrateAttempt, params: dict[str, Any]) -> None:
        """Fetch a localization diff and record it on ``attempt`` for ``_stage_apply`` (Rung 4).

        Does nothing when the round's gap needs no code acquisition, no
        bridging candidate in the framework's own repo was discovered, or the
        adapter cannot localize. A change that cannot be localized (compiled,
        oversize, outside the package, or not fetched) is skipped and the
        round runs on its own patches.
        """
        gap = _enablement_gap(params)
        if gap is None:
            return

        from hyperloom.inference_optimizer import framework_registry

        from ...enablement.runtime.adapters import get_adapter
        from ...enablement.runtime.localization import build_localization_diff

        refs = attempt.shared_state.enablement.candidate_refs
        if not refs:
            return
        framework = str(getattr(attempt.shared_state, "framework", "") or "").strip().lower()
        repo_url = framework_registry.repo_url(framework)
        if not repo_url:
            return
        ref = _first_ref_in_repo(refs, repo_url)
        if not ref:
            return

        adapter = get_adapter(framework)
        action = adapter.build_localization_action(gap, candidate_ref=ref, repo_url=repo_url)
        if action is None:
            return

        from hyperloom.agents.framework.sources import github as _gh

        diff_text, touched_paths, verdict = await asyncio.to_thread(
            build_localization_diff,
            action,
            fetch_pr_patches=lambda slug, num: _gh.pr_patches(slug, num),
            fetch_raw_file=lambda slug, ref, path: _gh.fetch_raw_file(slug, ref, path),
            scope=adapter.package_path,
        )
        if not verdict.is_localizable:
            log.info("integrate_patch: localization of %s skipped: %s", ref, verdict.reason)
            return

        loc_dir = runs_dir(self.session_dir, "integrate_patch", attempt.task_id) / "localization"
        loc_dir.mkdir(parents=True, exist_ok=True)
        gap_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", action.gap_id or "localization")
        patch_path = loc_dir / f"{gap_slug}.patch"
        patch_path.write_text(diff_text, encoding="utf-8")

        attempt.localization_patches = [patch_path]
        attempt.localization_action = action
        attempt.localization_touched = list(touched_paths)
        log.info(
            "integrate_patch: localization staged %s (%d file(s), kind=%s)",
            patch_path,
            len(touched_paths),
            action.kind,
        )

    @staticmethod
    def _admit_localization(attempt: IntegrateAttempt, framework_root: Path, patch_paths: list[Path]) -> list[Path]:
        """Put the localization ahead of the round's patches when it applies to the stashed tree, else drop it.

        The recorded patch set is what recovery sweeps, so it follows.
        """
        loc = attempt.localization_patches
        if not loc:
            return patch_paths
        if any(_detect_p_level(framework_root, p) is None for p in loc):
            log.info("integrate_patch: localization %s does not apply to %s; skipped", loc, framework_root)
            attempt.drop_localization()
            return patch_paths
        patch_paths = [*loc, *patch_paths]
        attempt.pending["patches"] = [str(p) for p in patch_paths]
        return patch_paths

    async def _record_closure_baseline(self, attempt: IntegrateAttempt, params: dict[str, Any]) -> None:
        """Observe the environment before the enablement's first setup command, through the KEEP's interpreter.

        The KEEP closure alone cannot say which versions the enablement changed;
        the setup script checks the difference between the two.
        """
        if attempt.shared_state.enablement.environment_closure_baseline:
            return

        from ...enablement.recipe.keep_probe import KeepEvidence, graded_framework, probe_keep_environment

        closure, _assertions = await asyncio.to_thread(
            probe_keep_environment,
            params,
            KeepEvidence(
                build_manifest=(),
                specialist_task_id=attempt.specialist_task_id,
                provision_result=attempt.provision_result,
            ),
            framework=graded_framework(params, ""),
        )
        if not attempt.shared_state.enablement.record_closure_baseline(closure):
            return
        try:
            attempt.shared_state.save(self.session_dir)
        except (OSError, AttributeError):
            # Held in memory either way, and the rearm saves again.
            log.debug("integrate_patch: save after closure baseline failed", exc_info=True)
