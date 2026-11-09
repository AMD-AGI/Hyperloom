# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT candidate-discovery rounds and the autosubmit of specialist deliverables to Critic review."""

from __future__ import annotations

import logging as _logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom_common.coerce import first_str, first_text, to_float, to_list, to_mapping
from hyperloom.common.perf_metric import agentx_active
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.grid_server_args import dedupe_extra_server_args, merge_server_args
from hyperloom.orchestrator.lever import LEVER_UPSTREAM_PR
from ..actions.executors import content_fingerprint, controls_of, normalize_proposal
from ..specialists.profile import is_authoring_specialist
from ..state.task_registry import TaskSpec
from ..state.task_states import TERMINAL_STATES
from . import machine_state as _phase_state
from .framework_records import (
    _framework_config_levers_from_done,
    _IntegrateReview,
    _record_discovered,
    _record_run,
    _resolvable_artifacts_from_done,
    _resolved_patch_paths,
)

if TYPE_CHECKING:
    from ..state.task_registry import Task

log = _logging.getLogger(__name__)

#: Consecutive empty discovery rounds tolerated before the source arm declines.
DISCOVER_FAILURE_RETRY_LIMIT: int = 3
_OBSERVER = "coordinator"
# Share of changed lines a proposed patch repeats from an already-measured one before the review is told. Set where
# two ports of one upstream change land and a follow-up to a measured patch does not; it informs, never refuses.
_PATCH_OVERLAP_MIN_RATIO: float = 0.5

#: The one ledger outcome that reports a round which reached a measurement. The
#: others name a round that ended before one: a patch that would not apply, a
#: server that failed its smoke, a parity check that could not conclude. Their
#: ``tps_delta_pct`` is the zero the record defaults to, not a result.
_LEDGER_OUTCOME_MEASURED: str = "integrated"


def _measured_short_of_keep(rec: dict[str, Any], keep_threshold_pct: float) -> bool:
    """Whether a ledger record is a completed measurement that did not clear the KEEP bar.

    A round that never measured says nothing about its idea, so it does not retire the direction.
    """
    if str(rec.get("outcome") or "") != _LEDGER_OUTCOME_MEASURED:
        return False
    return to_float(rec.get("tps_delta_pct"), default=0.0) < keep_threshold_pct


def _measured_on_this_stack(rec: dict[str, Any], state: Any) -> bool:
    """Whether a ledger record's throughput was measured on the model, GPU and precision running now.

    The ledger is shared by every session under the workspace root, and a throughput number belongs to the stack
    as much as to the config; an accuracy regression travels with the config, so that gate matches the fingerprint.
    """
    return all(
        str(rec.get(field) or "").strip() == getattr(state, field).strip()
        for field in ("model_class", "gpu_type", "precision")
    )


def _autosubmit_observation(claimed: list[str], done_payload: Mapping[str, Any]) -> dict[str, Any]:
    """The bus observation a specialist patch autosubmit records: the patches it claimed, the artifacts it installs."""
    return {
        "kind": "specialist_patch_autosubmitted_for_review",
        "patches": claimed,
        # Artifact-only deliverables: record their install targets.
        "artifacts_written": [
            str((a or {}).get("target") or "")
            for a in (done_payload.get("artifacts_written") or [])
            if isinstance(a, dict)
        ][:8],
    }


def _config_lever_deliverable(config_levers: Mapping[str, Any]) -> dict[str, Any]:
    """The integrate deliverable a config-lever round is reviewed with."""
    deliverable: dict[str, Any] = {
        "extra_server_args": config_levers.get("extra_server_args", ""),
        "extra_envs": dict(config_levers.get("extra_envs", {})),
        "framework_agent_authoring": True,
    }
    if config_levers:
        # The round is named after the proposal whose lever it benches, not the deliverable's first proposal.
        deliverable["patch_name"] = config_levers["name"]
        deliverable.update(controls_of(normalize_proposal(config_levers)))
    return deliverable


def _specialist_resolve_bases(session_dir: Any, sid: str) -> list[Path]:
    """The specialist's worktree and run dir, which its ``patches_written`` paths are resolved against."""
    from hyperloom.inference_optimizer.session.session_paths import runs_dir as _runs_dir

    resolve_bases: list[Path] = []
    if session_dir is not None:
        spec_root = _runs_dir(Path(session_dir), "specialist", sid)
        resolve_bases = [spec_root / "worktree", spec_root]
    return resolve_bases


class _CandidateDiscoveryMixin:
    """Dispatches and harvests candidate-discovery rounds and routes specialist deliverables to Critic review."""

    async def _maybe_enqueue_candidate_discovery(self, *, reason: str) -> bool:
        """Dispatch the next candidate-discovery round when the pool is empty; True while a round is queued or running."""
        state = self.shared_state
        limit = int(DISCOVER_FAILURE_RETRY_LIMIT)
        empties = int(state.framework_agent_empty_discoveries or 0)
        failures = int(state.framework_agent_discover_failures or 0)
        if empties >= limit or failures >= limit or int(state.research_lane_capacity or 0) <= 0:
            return False
        key_prefix = f"candidate-discovery:{reason}{self._coord.dispatcher.cycle_idem_suffix()}:r"
        rounds = await self.tasks.with_key_prefix("specialist", key_prefix)
        if any(t.state not in TERMINAL_STATES for t in rounds):
            return True
        gap, keywords = self._compose_framework_local_explore_gap()
        framework = str(state.framework or "").strip().lower()
        params: dict[str, Any] = {
            "domain": "candidate_discovery_specialist",
            "source_phase": "FRAMEWORK_AGENT",
            "lever_kind": LEVER_UPSTREAM_PR,
            "gap_canonical_id": f"gap.framework.candidate_discovery.{framework or 'unknown'}",
            "gap_symptom": gap or "Find upstream work worth landing for the current bottleneck",
            "gap_keywords": list(keywords or []),
            "gap_layer": "framework",
            "framework": framework,
            "task_kind": "candidate_discovery",
            "candidate_discovery": True,
            "mode": "research",
            "reason": reason,
            "source": "coordinator_internal",
        }
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=1800)
        task, existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params=params,
                # The registry hands back the row a key already names, so the next round is numbered past every round
                # it holds.
                idempotency_key=f"{key_prefix}{len(rounds)}",
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            ),
        )
        if not existing:
            # A row the key already names was recorded when it was dispatched; recording it again would reopen it.
            _record_run(
                self,
                task,
                role="discovery",
                status="dispatched",
                domain=str(params.get("domain") or ""),
                gap_canonical_id=str(params.get("gap_canonical_id") or ""),
                reason=reason,
                parallelism=len(lanes or ()),
            )
            log.info("FRAMEWORK: dispatched candidate discovery round %d (reason=%s)", len(rounds), reason)
        return True

    def _ingest_candidate_discovery(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any],
        run_error: str = "",
    ) -> None:
        """Harvest a discovery specialist's candidates into a batch."""
        params = getattr(task, "params", None) or {}
        if not bool(params.get("candidate_discovery")):
            return
        state = self.shared_state
        if run_error:
            failures = int(state.framework_agent_discover_failures or 0) + 1
            state.framework_agent_discover_failures = failures
            _record_run(
                self,
                task,
                role="discovery",
                status="failed",
                reason=run_error[:200],
            )
            log.warning(
                "FRAMEWORK: discovery task=%s failed (streak=%d): %s",
                getattr(task, "task_id", ""),
                failures,
                run_error[:200],
            )
            state.save(self.session_dir)
            return
        proposals = done_payload.get("proposal_set") if isinstance(done_payload, dict) else None
        candidates = self._candidates_from_discovery_proposals(proposals or [])
        _record_run(
            self,
            task,
            role="discovery",
            status="succeeded",
            reason="" if candidates else "no_usable_candidates",
        )
        _record_discovered(self, task, raw=proposals or [], candidates=candidates)
        # A round that ran is proof the lane works, whatever it came back with.
        state.framework_agent_discover_failures = 0
        if not candidates:
            empties = int(state.framework_agent_empty_discoveries or 0) + 1
            state.framework_agent_empty_discoveries = empties
            log.info("FRAMEWORK: discovery returned no usable candidates (streak=%d)", empties)
        else:
            state.framework_agent_empty_discoveries = 0
            batches = state.framework_agent_batches
            if not isinstance(batches, list):
                batches = []
                state.framework_agent_batches = batches
            batch_id = f"discovery-{len(batches)}-{getattr(task, 'task_id', '')[:8]}"
            for cand in candidates:
                cand["batch_id"] = batch_id
            batches.append({"batch_id": batch_id, "candidates": candidates})
            log.info("FRAMEWORK: harvested %d candidate(s) into batch=%s", len(candidates), batch_id)
        state.save(self.session_dir)

    def _candidates_from_discovery_proposals(self, proposals: Any) -> list[dict[str, Any]]:
        """Map discovery ``proposal_set`` entries to candidate rows."""
        out: list[dict[str, Any]] = []
        if not isinstance(proposals, list):
            return out
        known = self._framework_known_candidate_ids()
        processed = self._framework_processed_candidate_keys()
        for entry in proposals:
            if not isinstance(entry, dict):
                continue
            verdict = first_text(entry.get("verdict")).lower()
            if verdict in {"already_present", "not_applicable"}:
                continue
            pr_url = first_text(entry.get("pr_url"), entry.get("url"))
            head_sha = first_text(entry.get("head_sha"))
            if not pr_url and not head_sha:
                continue
            framework = first_text(entry.get("framework")).lower()
            if framework and not framework_registry.is_supported(framework):
                log.warning(
                    "FRAMEWORK: dropping discovery proposal %s: framework=%r is not registered",
                    pr_url or head_sha,
                    framework,
                )
                continue
            cand: dict[str, Any] = {
                "pr_url": pr_url,
                "url": pr_url,
                "head_sha": head_sha,
                "title": first_text(entry.get("title")),
                "diff_url": first_text(entry.get("diff_url")),
                "repo": first_text(entry.get("repo")),
                "pr_number": entry.get("pr_number"),
                "ref": first_text(entry.get("ref")),
                "framework": framework,
                "changed_files": entry.get("changed_files") or [],
                "gap_canonical_id": first_text(entry.get("gap_canonical_id")),
                "gap_keywords": entry.get("gap_keywords") or [],
                "reasoning": first_text(entry.get("reasoning"), entry.get("rationale"), entry.get("why")),
                "route": first_text(entry.get("route"), "author_via_specialist"),
                "audit": {
                    "verdict": verdict,
                    "reason": first_text(entry.get("reason")),
                    "recommended_next_step": first_text(entry.get("route")),
                },
            }
            key = self._framework_candidate_key(cand)
            if not key or key in known or key in processed:
                continue
            known.add(key)
            out.append(cand)
        return out

    async def maybe_autosubmit_specialist_patches(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any],
    ) -> None:
        """Auto-surface a specialist's source patches to the Critic via a synthetic integrate_patch proposal; idempotent per specialist."""
        patches = to_list(done_payload.get("patches_written"))
        claimed = [str(x) for x in patches][:8]
        sid = first_str(task.task_id).strip()
        if not sid:
            return
        # Resolve patches_written; submit only when >=1 real file exists.
        resolve_bases = _specialist_resolve_bases(self.session_dir, sid)
        existing_patches = _resolved_patch_paths(patches, resolve_bases)
        # A non-diff tuned artifact is also a routable deliverable; route it like a patch.
        routable_artifacts = _resolvable_artifacts_from_done(done_payload, resolve_bases)
        if not existing_patches and not routable_artifacts:
            if patches:
                await self.bus.record_observation(
                    _OBSERVER,
                    "observation",
                    {
                        "kind": "specialist_patch_autosubmit_skipped_no_files",
                        "specialist_task_id": sid,
                        "claimed": claimed,
                    },
                )
            return
        spec_params = to_mapping(getattr(task, "params", None))
        identity = await self._identity_unless_repeat(sid, spec_params, existing_patches)
        if identity is None:
            return
        # The lane, not the deliverable's own ``atomic`` flag, decides whether a companion lever rides with the patch.
        is_enablement_round = bool(spec_params.get("enablement"))
        companion_levers = _framework_config_levers_from_done(
            done_payload, levers_ride_with_patches=is_enablement_round
        )
        round_args = first_str(companion_levers.get("extra_server_args"))
        round_envs = to_mapping(companion_levers.get("extra_envs"))
        if is_enablement_round:
            # A flag the architecture requires stays required once a later round moves on to a different blocker, so
            # the round launches from everything enablement has established, with its own values winning a collision.
            established = to_mapping(getattr(self.shared_state.enablement, "accepted_config", None))
            round_envs = {
                **{str(k): str(v) for k, v in to_mapping(established.get("extra_envs")).items()},
                **round_envs,
            }
            round_args = dedupe_extra_server_args(
                merge_server_args(first_str(established.get("extra_server_args")), round_args)
            )
        deliverable: dict[str, Any] = {}
        if round_args or round_envs:
            deliverable["extra_server_args"] = round_args
            deliverable["extra_envs"] = round_envs
        deliverable.update(self._integrate_markers(spec_params, identity, enablement=is_enablement_round))
        await self._submit_integrate_review(
            sid,
            spec_params,
            done_payload,
            deliverable,
            review=_IntegrateReview(
                observation=_autosubmit_observation(claimed, done_payload),
                overlaps_measured=self._measured_patch_overlaps(identity["line_keys"]),
            ),
        )

    def _measured_patch_overlaps(self, line_keys: list[str]) -> list[dict[str, Any]]:
        """Measured rounds whose patch changed much the same lines as *line_keys*."""
        from ..framework.patch_overlap import overlapping_measured

        return overlapping_measured(
            line_keys,
            self.shared_state.measured_patch_sets,
            min_ratio=_PATCH_OVERLAP_MIN_RATIO,
        )

    async def _identity_unless_repeat(
        self, sid: str, spec_params: dict[str, Any], patch_paths: list[str]
    ) -> dict[str, Any] | None:
        """The patch's identity, or ``None`` when it repeats a patch already queued for review or measurement, or one
        that stayed in the tree. An authoring round's repeat still goes to review: its lane owns a terminal row."""
        from ..framework.patch_overlap import overlapping_measured, patch_set_identity, queued_identities

        files, line_keys = patch_set_identity(patch_paths)
        identity = {"files": files, "line_keys": line_keys}
        if not line_keys or spec_params.get("framework_agent_authoring"):
            return identity
        tasks = [*await self.tasks.queued(), *await self.tasks.running()]
        queued = queued_identities(self.state.pending_proposals.values(), tasks)
        landed = [row for row in self.shared_state.measured_patch_sets if row.get("landed")]
        hits = overlapping_measured(line_keys, [*queued, *landed], min_ratio=_PATCH_OVERLAP_MIN_RATIO)
        if not hits:
            return identity
        await self.bus.record_observation(
            _OBSERVER,
            "observation",
            {"kind": "specialist_patch_repeat_not_measured", "specialist_task_id": sid, "repeats": hits[0]},
        )
        return None

    def _integrate_markers(
        self, spec_params: Mapping[str, Any], identity: dict[str, Any], *, enablement: bool
    ) -> dict[str, Any]:
        """The integrate params that say how a specialist patch is to be measured rather than what it applies."""
        markers: dict[str, Any] = {}
        if spec_params.get("framework_agent_authoring"):
            markers["framework_agent_authoring"] = True
        if identity["line_keys"]:
            markers["patch_identity"] = identity
        from ..actions.executors._integrate_vocabulary import SCREEN_CYCLE, SCREEN_HOST, SCREEN_PARAM
        from ..actions.executors.cycle_bench_arm import cycle_bench_enabled, cycle_bench_sees

        if not enablement and cycle_bench_enabled() and agentx_active(benchmark_mode=self.shared_state.benchmark_mode):
            # A seen patch is admitted dormant only if switched and held to a reference; an unseen one is dispatched first.
            markers[SCREEN_PARAM] = SCREEN_CYCLE if cycle_bench_sees(identity["files"]) else SCREEN_HOST
        return markers

    async def maybe_autosubmit_framework_config(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any],
    ) -> None:
        """Route a FRAMEWORK config-lever deliverable through integrate_patch.

        Companion to :meth:`maybe_autosubmit_specialist_patches`: fires when a
        FRAMEWORK authoring or ENABLEMENT specialist returns NO source patch but a config-lever
        ``proposal_set`` (extra_args / extra_envs). The levers go into
        integrate_patch's ``config_changes`` channel (apply + bench + accuracy
        gate + KEEP/REVERT), which owns the terminal FRAMEWORK row. Idempotent
        per specialist.

        Args:
            task: The completed authoring specialist task.
            done_payload: Its ``specialist_done`` payload.
        """
        spec_params = getattr(task, "params", None) or {}
        if not is_authoring_specialist(spec_params):
            return
        # A patch deliverable is handled by the patch autosubmit bridge.
        patches = done_payload.get("patches_written") or []
        if isinstance(patches, list) and patches:
            return
        config_levers = _framework_config_levers_from_done(done_payload)
        is_enablement = bool(spec_params.get("enablement"))
        build_request = done_payload.get("needs_targeted_build")
        if (
            is_enablement
            and isinstance(build_request, dict)
            and build_request
            and not config_levers
            and not done_payload.get("setup_commands")
            and not done_payload.get("artifacts_written")
        ):
            return
        # Route only when there are config levers to test, except for an
        # ENABLEMENT round, which always routes: a round that does not reach
        # integrate_patch never reports what its boot did.
        if not config_levers and not is_enablement:
            return
        sid = first_text(task.task_id)
        if not sid:
            return
        if config_levers and self._config_lever_known_bad(config_levers):
            return
        # Empty, or the str args and dict envs config_lever_from_proposal builds.
        extra_server_args = config_levers.get("extra_server_args", "")
        extra_envs = dict(config_levers.get("extra_envs", {}))
        candidate_id = str(spec_params.get("framework_agent_candidate_id") or "")
        pending = await self._submit_integrate_review(
            sid,
            spec_params,
            done_payload,
            _config_lever_deliverable(config_levers),
            review=_IntegrateReview(
                observation={
                    "kind": "framework_config_autosubmitted_for_review",
                    "candidate_id": candidate_id,
                    "extra_server_args": extra_server_args,
                    "extra_envs": dict(extra_envs),
                },
            ),
        )
        if pending is not None:
            log.info(
                "FRAMEWORK: config-lever deliverable routed to integrate_patch candidate=%s args=%s env_keys=%s",
                candidate_id or sid,
                extra_server_args,
                sorted(extra_envs),
            )

    def _config_lever_known_bad(self, config_levers: dict[str, Any]) -> bool:
        """True when this exact config already lost an accuracy gate, or already measured short of KEEP.

        Different upstream PRs often reduce to the same server args / envs, so
        the ledger is keyed by content fingerprint rather than by PR.
        """
        try:
            from hyperloom.agents.framework.kb import read_pr_ledger

            # The integrate record hashes the deliverable this lever is reviewed with.
            fingerprint = content_fingerprint(_config_lever_deliverable(config_levers))
            threshold = _phase_state.resolve_keep_threshold(self.shared_state)
            for rec in read_pr_ledger():
                if str(rec.get("applicability") or "") != fingerprint:
                    continue
                if to_float(rec.get("accuracy_delta_pct"), default=0.0) < 0.0:
                    log.info(
                        "FRAMEWORK: skipping config lever %s — accuracy %.2f%% on %s",
                        fingerprint,
                        to_float(rec.get("accuracy_delta_pct"), default=0.0),
                        rec.get("pr_url") or "a prior candidate",
                    )
                    return True
                if _measured_on_this_stack(rec, self.shared_state) and _measured_short_of_keep(rec, threshold):
                    log.info(
                        "FRAMEWORK: skipping config lever %s — already measured at %+.2f%% against a %.2f%% bar on %s",
                        fingerprint,
                        to_float(rec.get("tps_delta_pct"), default=0.0),
                        threshold,
                        rec.get("pr_url") or "a prior candidate",
                    )
                    return True
        except OSError:
            # Warning, not debug: swallowing this re-dispatches config levers
            # that already lost an accuracy gate, so it must be visible.
            log.warning("FRAMEWORK: config-lever ledger check failed", exc_info=True)
        return False
