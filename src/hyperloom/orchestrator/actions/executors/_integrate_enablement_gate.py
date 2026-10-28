# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The enablement gate: whether an applied switch is correct, what it keeps, and the record it leaves in the knowledge base."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.coerce import first_text, to_float, to_str_list
from hyperloom.inference_optimizer import framework_paths

from ...bringup import load_boot_observation, observation_summary, verdict_of
from ...bringup.persist import LoadedObservation
from ...enablement.recipe import keep_probe, keep_records
from ...enablement.recipe.projections import project_launch_evidence
from ._accuracy_gate import enablement_correctness
from ._enablement_config import _FRAMEWORK_KB_PROVENANCE_PREFIX
from ._git import _git_head_sha
from ._integrate_attempt import IntegrateAttempt
from ._integrate_results import _KbLesson, _PatchBench
from ._nogit_patch import _is_git_tree
from ._patch_git import _uncommitted_revert
from ._patch_snapshot import _git_commit_kept, _patch_touched_paths
from ._integrate_vocabulary import PATCH_SOURCE_UPSTREAM_PR, resolve_patch_source
from ._patch_sources import _declared_stack_ops, _patch_stack_by_root
from ._server_argv import ConfigUnreadable
from ._workload_envs import FrameworkScriptMismatchError

log = logging.getLogger(__name__)


def _observation_evidence(half: str, loaded: LoadedObservation) -> dict[str, Any]:
    """The ``<half>_observation`` summary, path and degraded reason of one loaded boot observation."""
    return {
        f"{half}_observation": observation_summary(loaded.observation) if loaded.observation is not None else None,
        f"{half}_observation_path": loaded.path,
        f"{half}_observation_degraded": loaded.degraded,
    }


class EnablementGateMixin:
    """Grade an applied enablement switch, commit what it keeps, and write the framework record the next session reads."""

    async def _stage_gate(
        self,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Bench + enablement/perf KEEP/REVERT gate.

        Runs _bench_patch, applies the appropriate gate, and returns the
        final integration result. Never returns None.
        """
        specialist_task_id = attempt.specialist_task_id
        output_root = attempt.output_root
        assert output_root is not None
        params = attempt.benchmark_params(params)

        # Bound both bench legs by the session wall-clock, as the sweep and explore
        # arms already are: the declared cap answers "how long before this counts
        # as hung", not "how much budget is left", so without this a patch benched
        # near the end of a run could outlive the run itself. Resolved once here
        # and reused by the parity leg -- the deadline is an absolute monotonic
        # timestamp, so it stays correct as the gate progresses.
        try:
            bench_result, gate_evidence = await self._bench_patch(
                _PatchBench.for_attempt(attempt, params),
                extra_server_args_applied=attempt.extra_server_args_applied,
                extra_envs_applied=attempt.extra_envs_applied,
            )
        except FrameworkScriptMismatchError as exc:
            return {
                "status": "reverted",
                "error_class": "framework_script_mismatch",
                "error": str(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "artifacts_reverted": [],
                "reason": str(exc),
                "workspace": str(output_root),
            }
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "status": "reverted",
                "error_class": "bench_exception",
                "error": repr(exc),
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "reason": f"bench raised: {exc!r}",
                "workspace": str(output_root),
            }

        if params.get("enablement"):
            verdict = await self._gate_enablement(
                attempt=attempt,
                params=params,
                bench_result=bench_result,
                gate_evidence=gate_evidence,
            )
        else:
            verdict = await self._gate_perf(
                attempt=attempt,
                params=params,
                bench_result=bench_result,
                gate_evidence=gate_evidence,
            )
        if attempt.dropped_env_overrides:
            verdict["dropped_env_overrides"] = attempt.dropped_env_overrides
        return verdict

    async def _gate_enablement(
        self,
        *,
        attempt: IntegrateAttempt,
        params: dict[str, Any],
        bench_result: dict[str, Any],
        gate_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Enablement gate: runnability + minimal-correctness.

        A round is accepted when the boot runs or reaches a deeper wall, and its
        writes then stay in the tree for the next round to build on; ``kept`` and
        ``advanced`` differ only in whether the lane is finished. Anything short
        of that reverts to the state the previous round left.

        The verdict depends on the trigger origin, because the two origins have
        different evidence available:

        * ``accuracy >= floor`` -> ``correctness_ok=True`` (KEEP, verified).
          eval-origin additionally requires the score to carry a task + metric;
          without them it did not come from a real eval and fails closed.
        * present but below floor / non-positive / non-finite ->
          ``correctness_ok=False`` (REVERT, garbage output).
        * ``accuracy is None`` -> eval-origin fails closed
          (``correctness_ok=False``): the trigger *was* an accuracy failure, so a
          candidate that produces no score has not shown it fixed anything.
          boot-origin stays ``None`` (KEEP but provisional) — it only ever
          claimed to make the model boot, and eval-less runs must not be blocked.

        On KEEP the benched env/arg layers are reported as
        ``enablement_effective_config``, captured from the variant that ran: the
        materialized YAML holds only the base layer, so the revalidation baseline
        needs them to re-run the graded config.
        When an attempt runtime was provisioned, an accepted round (kept or
        advanced) records it in the result (``enablement_kept_stack_action``) so
        it survives rearm. On REVERT, the attempt runtime dir is GC'd.

        Every verdict carries ``framework_root`` (the source tree patches were applied
        against, needed to replay them on a fresh machine).
        """
        specialist_task_id = attempt.specialist_task_id
        done_payload = attempt.done_payload
        output_root = attempt.output_root
        assert output_root is not None
        framework_root = attempt.framework_root
        root_text = str(framework_root or "")
        applied = attempt.applied
        applied_artifacts = attempt.applied_artifacts
        extra_envs_applied = attempt.extra_envs_applied
        extra_server_args_applied = attempt.extra_server_args_applied

        from hyperloom.common.failure_signature import runnable_decision

        from ...bringup import round_advanced
        from .benchmark_result import is_valid_measurement

        new_tput = bench_result.get("output_throughput")
        patches_applied = [str(patch) for patch in applied]

        correctness_ok, eval_provenance = enablement_correctness(params, gate_evidence)

        after_loaded = load_boot_observation(bench_result.get("boot_observation_path"))
        before_loaded = load_boot_observation(params.get("enablement_before_observation_path"))
        evidence = {
            **_observation_evidence("before", before_loaded),
            **_observation_evidence("after", after_loaded),
            **eval_provenance,
        }

        # A measurement exists only where the client completed requests, so it
        # witnesses the serving instead of inferring it from a log marker.
        served = is_valid_measurement(bench_result)

        runs, run_reason = runnable_decision(served=served, correctness_ok=correctness_ok)
        advanced = not runs and not served and round_advanced(before_loaded.observation, after_loaded.observation)
        if not runs and not advanced:
            await self._maybe_write_framework_kb_record(
                _KbLesson("reverted_smoke_fail", 0.0), attempt=attempt, params=params
            )
            return {
                "status": "reverted",
                "specialist_task_id": specialist_task_id,
                "patches_applied": [],
                "patches_reverted": [],
                "artifacts_reverted": [],
                "framework_root": root_text,
                "output_throughput": new_tput,
                "enablement": True,
                "runnable": False,
                "correctness_verified": correctness_ok is True,
                "reason": f"enablement not runnable: {run_reason}",
                "setup_commands_applied": attempt.setup_commands_applied,
                "bench_result": bench_result,
                "workspace": str(output_root),
                **evidence,
            }

        # Accepted: the boot either runs or reached a deeper wall. Both keep the
        # work in the tree and differ only in whether the lane is finished.
        commit_failure, _ = self._commit_accepted_work(
            attempt, f"hyperloom enablement {'advanced' if advanced else 'kept'} {specialist_task_id}"
        )
        if commit_failure:
            return _uncommitted_revert(
                attempt,
                commit_failure,
                "enablement progress",
                enablement=True,
                framework_root=root_text,
            )

        if advanced:
            await self._maybe_write_framework_kb_record(_KbLesson("integrated", 0.0), attempt=attempt, params=params)
            # round_advanced is only true for a recorded after-observation.
            assert after_loaded.observation is not None
            wall = after_loaded.observation.stage_failed
            signature = verdict_of(after_loaded.observation).signature
            return {
                "status": "advanced",
                "specialist_task_id": specialist_task_id,
                "patches_applied": patches_applied,
                # An ADVANCED round stacks its patch and never reaches the KEEP capture,
                # so without this the tree its patch applied to is never recorded and a
                # later KEEP binds it to that round's framework root instead. The capture
                # proves a patch against the tree it is bound to, so a mis-binding refuses
                # the whole recipe -- for a legitimate multi-root stack a false refusal
                # rather than a false pass.
                "enablement_patch_roots": attempt.shared_state.enablement.accepted_patch_roots(
                    done_payload=done_payload,
                    applied=applied,
                    framework_root=root_text,
                ),
                "patches_reverted": [],
                "artifacts_applied": applied_artifacts,
                "extra_envs_applied": extra_envs_applied,
                "extra_server_args_applied": extra_server_args_applied,
                "framework_root": root_text,
                "output_throughput": new_tput,
                "enablement": True,
                "advanced": True,
                "runnable": False,
                "correctness_verified": False,
                "reason": (
                    f"enablement progressed: {run_reason}; boot advanced "
                    f"to a new gap ({wall.name if wall is not None else 'no wall recorded'})"
                ),
                "setup_commands_applied": attempt.setup_commands_applied,
                "after_signature": signature.to_dict(),
                "enablement_launch_log": str(bench_result.get("error") or ""),
                # The wall this round advanced to, for the next round's before half.
                "enablement_observation_path": after_loaded.path,
                "bench_result": bench_result,
                "workspace": str(output_root),
                **evidence,
                **attempt.accepted_runtime_records(),
            }

        provisional = correctness_ok is None
        reason = f"enablement runnable: {run_reason}"
        if provisional:
            reason += " (provisional: served but eval produced no accuracy; correctness not verified)"
        await self._maybe_write_framework_kb_record(_KbLesson("integrated", 0.0), attempt=attempt, params=params)
        kept_result: dict[str, Any] = {
            "status": "kept",
            "specialist_task_id": specialist_task_id,
            "patches_applied": patches_applied,
            "patches_reverted": [],
            "artifacts_applied": applied_artifacts,
            "extra_server_args_applied": extra_server_args_applied,
            "extra_envs_applied": extra_envs_applied,
            "framework_root": root_text,
            "output_throughput": new_tput,
            "enablement": True,
            "runnable": True,
            "correctness_verified": correctness_ok is True,
            "provisional": provisional,
            "reason": reason,
            "setup_commands_applied": attempt.setup_commands_applied,
            "bench_result": bench_result,
            "workspace": str(output_root),
            # Base YAML only; the env/arg layers live in enablement_effective_config.
            "enablement_accepted_config_path": str(bench_result.get("materialized_config") or ""),
            # Captured from the variant this leg launched, so a revalidation
            # replays the graded configuration rather than a re-derived one.
            "enablement_effective_config": dict(bench_result.get("effective_config") or {}),
            **evidence,
        }
        kept_result.update(attempt.accepted_runtime_records())
        # Snapshot a manifest of the localized closure that survives rearm, so
        # the closure is recorded and not re-fetched.
        manifest = await asyncio.to_thread(
            self._finalize_localization_keep,
            attempt,
            framework_root=framework_root,
            specialist_task_id=specialist_task_id,
        )
        if manifest:
            kept_result["enablement_localization_manifest"] = manifest
        try:
            kept_result.update(self._enablement_keep_records(attempt, params=params, bench_result=bench_result))
        except (OSError, subprocess.SubprocessError, ConfigUnreadable):
            # Every field this fills is one the decision refuses the replay for
            # when absent, so a capture that cannot read the tree or the graded
            # config, spawn the probe, or see the durable stack leaves the
            # recipe insufficient rather than failing the round.
            log.exception("integrate_patch: enablement KEEP record capture failed")
        return kept_result

    def _enablement_keep_records(
        self,
        attempt: IntegrateAttempt,
        *,
        params: dict[str, Any],
        bench_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Capture the per-root identity, payload and assertions of this KEEP."""
        root = str(attempt.framework_root or "")
        enablement = attempt.shared_state.enablement
        patch_roots = enablement.accepted_patch_roots(
            done_payload=attempt.done_payload,
            applied=attempt.applied,
            framework_root=root,
        )
        # Read from the durable stack: the round lifecycle does not dispatch the
        # base set with the round.
        stack_artifacts = keep_records.accepted_stack_artifacts(
            inherited=enablement.kept_artifacts,
            applied=attempt.applied_artifacts,
        )
        contributions = keep_records.collect_contributions(
            framework_root=root,
            patch_roots=patch_roots,
            artifacts=stack_artifacts,
        )
        git_roots = [tree for tree in contributions if tree and _is_git_tree(Path(tree))]
        # Only what was captured before this round touched each tree, and only
        # where no earlier round already named the tree the stack applies to. A
        # root left with no sha either way keeps none, which the decision refuses
        # rather than answering with a HEAD that has moved since.
        inherited_sha = enablement.inherited_base_shas()
        captured = attempt.base_sha_by_root
        base_sha_by_root = {tree: inherited_sha.get(tree, "") or captured.get(tree, "") for tree in git_roots}
        records = keep_records.build_root_records(
            contributions=contributions,
            base_sha_by_root=base_sha_by_root,
            git_roots=git_roots,
            session_framework_root=framework_paths.resolve_session_framework_root(),
        )
        # One classification per root: a rel path is only meaningful against the
        # tree its patch was bound to, and reading them all against this round's
        # root would name files that root does not have. Per patch as well as per
        # root: the decision cross-checks each patch step against the snapshot of
        # the files that step declares, and it cannot read the diffs itself.
        targets: dict[str, dict[str, str]] = {}
        patch_targets: dict[str, dict[str, str]] = {}
        for patch_root, patches in _patch_stack_by_root(enablement, patch_roots, attempt.applied).items():
            base_sha = str(base_sha_by_root.get(patch_root) or "")
            for patch_path, ops in _declared_stack_ops(patch_root, patches, base_sha=base_sha).items():
                if not ops:
                    continue
                patch_targets[patch_path] = ops
                targets.setdefault(patch_root, {}).update(ops)
        for declared_root, ops in keep_records.declared_targets(
            framework_root=root,
            upserted=(),
            deleted=(),
            artifacts=stack_artifacts,
        ).items():
            targets.setdefault(declared_root, {}).update(ops)
        snapshots = keep_records.capture_root_snapshots(
            records=records,
            targets=targets,
            dest_root=self.session_dir / "optimization_stack" / "enablement",
            session_dir=self.session_dir,
        )
        materialized_config = str(bench_result.get("materialized_config") or "")
        framework = keep_probe.graded_framework(params, materialized_config)
        closure, assertions = keep_probe.probe_keep_environment(
            params,
            keep_probe.KeepEvidence(
                build_manifest=enablement.build_manifest,
                specialist_task_id=attempt.specialist_task_id,
                provision_result=attempt.provision_result,
            ),
            framework=framework,
            materialized_config=materialized_config,
        )
        launch_evidence, argv_refused = project_launch_evidence(bench_result.get("launch_evidence"))
        return {
            "enablement_roots": records,
            "enablement_patch_roots": patch_roots,
            # Per root, and through the same inherited-then-captured resolution
            # every other root gets. Preferring the persisted scalar outright
            # hands this round's root the sha an earlier round read off a
            # *different* tree, and the snapshot is then captured against a base
            # commit that tree never had.
            "enablement_base_sha": str(base_sha_by_root.get(root) or ""),
            "enablement_source_snapshots": snapshots,
            "enablement_patch_targets": patch_targets,
            "enablement_accepted_stack_targets": {
                str(record["id"]): dict(targets.get(str(record["path"])) or {}) for record in records
            },
            "enablement_launch_evidence": launch_evidence or {},
            "enablement_launch_argv_refused": argv_refused,
            "enablement_environment_closure": closure,
            "enablement_installed_versions_at_keep": assertions,
            "enablement_build_extensions_not_carried": keep_records.build_extensions_not_carried(
                enablement,
                attempt.framework_root,
                specialist_task_id=attempt.specialist_task_id,
            ),
            "enablement_levers_without_readers": keep_records.levers_without_readers(
                enablement,
                attempt.framework_root,
                framework=framework,
                effective_config=bench_result.get("effective_config"),
            ),
        }

    @staticmethod
    def _commit_accepted_work(attempt: IntegrateAttempt, message: str) -> tuple[str, bool]:
        """Commit this round's patch and artifact writes so they survive a later revert.

        A non-git root needs no commit: nothing there resets the tree between
        rounds, so the writes are already durable.

        Returns:
            ``(failure_note, head_advanced)``. ``failure_note`` is empty on
            success. ``head_advanced`` is ``True`` only when a real commit
            landed, which is what makes ``HEAD^..HEAD`` this round's diff.
        """
        framework_root = attempt.framework_root
        if framework_root is None or not _is_git_tree(framework_root):
            return "", False
        root_resolved = framework_root.resolve()
        try:
            touched = _patch_touched_paths(framework_root, attempt.applied)
            # An artifact installed into a sibling root is not addressable by a
            # rel path under this one, so it belongs to no commit here.
            artifact_rels = [
                str(a["rel_target"])
                for a in attempt.applied_artifacts
                if a.get("rel_target") and Path(str(a["root"])).resolve() == root_resolved
            ]
            ok, note = _git_commit_kept(framework_root, message, list(dict.fromkeys(touched + artifact_rels)))
        except (OSError, RuntimeError) as exc:
            return f"commit raised: {exc!r}", False
        if not ok:
            return note or "git commit failed", False
        return "", note == ""

    def _finalize_localization_keep(
        self,
        attempt: IntegrateAttempt,
        *,
        framework_root: Path | None,
        specialist_task_id: str,
    ) -> dict[str, Any]:
        """Record a localization manifest via :func:`snapshot_source_layer`.

        Blocking (filesystem copies); call via ``asyncio.to_thread``. Returns the
        manifest dict, empty when no localization ran.
        """
        touched = attempt.localization_touched
        if not touched or framework_root is None:
            return {}
        action = attempt.localization_action
        try:
            from ...source_snapshot import SourceLayer, snapshot_source_layer

            base_sha = _git_head_sha(framework_root)
            dest = self.session_dir / "optimization_stack" / "localization" / (specialist_task_id or "keep")
            snap = snapshot_source_layer(
                SourceLayer(
                    framework_root=framework_root,
                    base_sha=base_sha,
                    rel_paths=touched,
                ),
                dest_dir=dest,
                provenance="localization",
                extra={
                    "specialist_task_id": specialist_task_id,
                    "kind": str(getattr(action, "kind", "") or ""),
                    "repo_url": str(getattr(action, "repo_url", "") or ""),
                    "pr_number": int(getattr(action, "pr_number", 0) or 0),
                },
            )
            return dict(snap) if snap else {}
        except (OSError, ValueError, TypeError):
            log.exception("integrate_patch: localization snapshot failed")
            return {}

    # Helpers
    @staticmethod
    def _find_framework_proposal(
        done_payload: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Return the first proposal whose provenance starts with
        ``specialist:serving:framework`` (F2-5); ``None`` otherwise so
        the KB writeback hook no-ops for legacy / kernel outputs.

        Args:
            done_payload: The parsed ``specialist_done.json`` payload, or
                ``None``.

        Returns:
            The matching framework proposal dict, or ``None`` when absent.
        """
        if not isinstance(done_payload, dict):
            return None
        proposal_set = done_payload.get("proposal_set") or []
        if not isinstance(proposal_set, list):
            return None
        for proposal in proposal_set:
            if not isinstance(proposal, dict):
                continue
            provenance = str(proposal.get("provenance") or "")
            if provenance.startswith(_FRAMEWORK_KB_PROVENANCE_PREFIX):
                return proposal
        return None

    @staticmethod
    def _upstream_pr_kb_proposal(params: Mapping[str, Any]) -> dict[str, Any] | None:
        """Present an upstream-PR candidate in the shape the KB writer reads.

        The KB writer reads the PR identity from ``fa_pr_url`` / ``fa_pr_sha``.
        A candidate row carries it directly, so map rather than relay it through
        a specialist's output.

        Args:
            params: Task params, read for ``candidate``.

        Returns:
            A proposal-shaped mapping, or ``None`` when this is not an
            upstream-PR task or the candidate carries no dedup key.
        """
        if resolve_patch_source(params) != PATCH_SOURCE_UPSTREAM_PR:
            return None
        candidate = params.get("candidate")
        if not isinstance(candidate, dict):
            return None
        pr_url = str(candidate.get("pr_url") or candidate.get("url") or "").strip()
        pr_sha = str(candidate.get("head_sha") or "").strip()
        if not pr_url and not pr_sha:
            return None
        return {
            "fa_pr_url": pr_url,
            "fa_pr_sha": pr_sha,
            "framework": candidate.get("framework") or "",
            "applicability": candidate.get("applicability") or "",
        }

    async def _maybe_write_framework_kb_record(
        self,
        lesson: _KbLesson,
        *,
        attempt: IntegrateAttempt,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        """Record this attempt's outcome against the patch's upstream PR identity.

        A patch with no such identity leaves no lesson: the ledger files every
        lesson under a PR URL or head SHA.
        """
        proposal = self._find_framework_proposal(attempt.done_payload)
        if proposal is None:
            # The upstream-PR lane carries the PR identity on the candidate
            # rather than in a specialist's ``fa_*`` markers.
            proposal = self._upstream_pr_kb_proposal(params or {})
            if proposal is None:
                return
        # Never the payload's: a payload-level ``fa_pr_*`` belongs to whichever
        # proposal put it there, so standing it in for one this proposal omits
        # would file the lesson under another patch's PR.
        pr_url = first_text(proposal.get("fa_pr_url"))
        pr_sha = first_text(proposal.get("fa_pr_sha"))
        if not pr_url and not pr_sha:
            log.warning("integrate_patch: framework proposal lacks both fa_pr_url and fa_pr_sha; KB writeback skipped")
            return
        # The ``if v`` filter is what makes the merge mean "the proposal's value,
        # or the payload's where the proposal left it empty".
        fields = {**(attempt.done_payload or {}), **{k: v for k, v in proposal.items() if v}}
        shared_state = attempt.shared_state
        patches_written = to_str_list(proposal.get("patches_written"))
        accuracy_delta_pct = lesson.accuracy_delta_pct
        if accuracy_delta_pct is None:
            accuracy_delta_pct = to_float(fields.get("accuracy_delta_pct"), 0.0)
        try:
            from ...knowledge.kb_writeback import FrameworkOutcome, write_framework_record

            written = await write_framework_record(
                FrameworkOutcome(
                    pr_url=pr_url,
                    pr_sha=pr_sha,
                    patch_path=patches_written[0] if patches_written else "",
                    outcome=lesson.outcome,
                    tps_delta_pct=float(lesson.tps_delta_pct),
                    session_id=str(getattr(shared_state, "recipe_kb_session_id", "") or ""),
                    framework=first_text(fields.get("framework")).lower(),
                    applicability=lesson.config_fingerprint or first_text(fields.get("applicability")),
                    accuracy_delta_pct=accuracy_delta_pct,
                )
            )
        except (OSError, ValueError, TypeError) as exc:
            log.warning("integrate_patch: framework KB writeback failed: %r", exc)
            return
        log.info(
            "integrate_patch: wrote framework KB record to %s (outcome=%s pr_url=%s tps_delta=%+.2f%%)",
            written,
            lesson.outcome,
            pr_url,
            float(lesson.tps_delta_pct),
        )
