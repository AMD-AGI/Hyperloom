# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT candidate-discovery rounds and the autosubmit of specialist deliverables to Critic review."""

from __future__ import annotations

import logging as _logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.coerce import first_str, first_text, to_float, to_list, to_mapping
from hyperloom.inference_optimizer import framework_registry
from hyperloom.inference_optimizer.grid_server_args import dedupe_extra_server_args, merge_server_args
from hyperloom.orchestrator.lever import LEVER_UPSTREAM_PR
from ..specialists.profile import is_authoring_specialist
from ..state.task_registry import TaskSpec
from .framework_records import (
    _framework_config_levers_from_done,
    _patches_on_disk,
    _record_discovered,
    _record_run,
    _resolvable_artifacts_from_done,
)

if TYPE_CHECKING:
    from ..state.task_registry import Task

log = _logging.getLogger(__name__)

#: Consecutive empty discovery rounds tolerated before the source arm declines.
DISCOVER_FAILURE_RETRY_LIMIT: int = 3


class _CandidateDiscoveryMixin:
    """Dispatches and harvests candidate-discovery rounds and routes specialist deliverables to Critic review."""

    async def _candidate_discovery_inflight(self) -> bool:
        """True while a discovery round this phase dispatched, the kind ``_ingest_candidate_discovery`` harvests, is queued or running."""
        return any(
            t.kind == "specialist" and bool((t.params or {}).get("candidate_discovery"))
            for t in (*await self.tasks.queued(), *await self.tasks.running())
        )

    async def _maybe_enqueue_candidate_discovery(self, *, reason: str) -> bool:
        """Dispatch the candidate-discovery specialist when the pool is empty."""
        state = self.shared_state
        limit = int(DISCOVER_FAILURE_RETRY_LIMIT)
        empties = int(state.framework_agent_empty_discoveries or 0)
        failures = int(state.framework_agent_discover_failures or 0)
        if empties >= limit or failures >= limit or int(state.research_lane_capacity or 0) <= 0:
            return False
        if await self._candidate_discovery_inflight():
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
                # The round count is part of the key: the registry returns the row a key already names, so a fixed key
                # would re-fetch the finished first attempt and neither streak could advance.
                idempotency_key=f"candidate-discovery:{reason}{self._coord.dispatcher.cycle_idem_suffix()}:r{empties + failures}",
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
        log.info("FRAMEWORK: dispatched candidate discovery (reason=%s)", reason)
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
        from hyperloom.inference_optimizer.session.session_paths import runs_dir as _runs_dir

        resolve_bases: list[Path] = []
        if self.session_dir is not None:
            spec_root = _runs_dir(Path(self.session_dir), "specialist", sid)
            resolve_bases = [spec_root / "worktree", spec_root]
        existing_patches = [str(p) for p in patches if _patches_on_disk(Path(str(p)), resolve_bases)]
        # A non-diff tuned artifact is also a routable deliverable; route it like a patch.
        routable_artifacts = _resolvable_artifacts_from_done(done_payload, resolve_bases)
        if not existing_patches and not routable_artifacts:
            if patches:
                await self.bus.record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "specialist_patch_autosubmit_skipped_no_files",
                        "specialist_task_id": sid,
                        "claimed": claimed,
                    },
                )
            return
        spec_params = to_mapping(getattr(task, "params", None))
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
        if spec_params.get("framework_agent_authoring"):
            deliverable["framework_agent_authoring"] = True
        await self._submit_integrate_review(
            sid,
            spec_params,
            done_payload,
            deliverable,
            observation={
                "kind": "specialist_patch_autosubmitted_for_review",
                "patches": claimed,
                # Artifact-only deliverables: record their install targets.
                "artifacts_written": [
                    str((a or {}).get("target") or "")
                    for a in (done_payload.get("artifacts_written") or [])
                    if isinstance(a, dict)
                ][:8],
            },
        )

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
        # Empty, or the str args and dict envs _config_lever_from_proposal builds.
        extra_server_args = config_levers.get("extra_server_args", "")
        extra_envs = dict(config_levers.get("extra_envs", {}))
        candidate_id = str(spec_params.get("framework_agent_candidate_id") or "")
        pending = await self._submit_integrate_review(
            sid,
            spec_params,
            done_payload,
            {"extra_server_args": extra_server_args, "extra_envs": extra_envs, "framework_agent_authoring": True},
            observation={
                "kind": "framework_config_autosubmitted_for_review",
                "candidate_id": candidate_id,
                "extra_server_args": extra_server_args,
                "extra_envs": dict(extra_envs),
            },
        )
        if pending is not None:
            log.info(
                "FRAMEWORK: config-lever deliverable routed to integrate_patch candidate=%s args=%s env_keys=%s",
                candidate_id or sid,
                extra_server_args,
                sorted(extra_envs),
            )

    def _config_lever_known_bad(self, config_levers: dict[str, Any]) -> bool:
        """True when this exact config already lost an accuracy gate.

        Different upstream PRs often reduce to the same server args / envs, so
        the ledger is keyed by content fingerprint rather than by PR.
        """
        from hyperloom.inference_optimizer.canonical_fingerprint import canonical_fingerprint

        try:
            from hyperloom.agents.framework.kb import read_pr_ledger

            fingerprint = canonical_fingerprint(
                config_levers.get("extra_server_args"),
                config_levers.get("extra_envs"),
            )
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
        except OSError:
            # Warning, not debug: swallowing this re-dispatches config levers
            # that already lost an accuracy gate, so it must be visible.
            log.warning("FRAMEWORK: config-lever ledger check failed", exc_info=True)
        return False
