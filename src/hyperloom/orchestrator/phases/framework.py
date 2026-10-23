# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT phase handler: authoring specialist dispatch, enablement repair, deliverable routing, and Critic-review submission/reauthor."""

from __future__ import annotations
import logging as _logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

from hyperloom.common.coerce import first_str, to_list, to_mapping
from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from . import machine_state as _phase_state
from ..state.task_registry import TaskNotFound, TaskSpec
from ..state.shared_state import resolve_grading_anchor_tput, inject_stack_base_params

if TYPE_CHECKING:
    from ..predictor.payload import PhaseBudget
    from ..predictor.view import SessionView
    from ..state.task_registry import Task
    from .machine import Transition
from ..loop.proposals import PendingProposal, integrate_review_in_flight, record_proposal
from ..actions.executors import _integrate_vocabulary
from hyperloom.common.framework_arm import LOCAL_EXPLORE_CANDIDATE_PREFIX
from hyperloom.orchestrator.lever import LEVER_SOURCE_PATCH, LEVER_UPSTREAM_PR, patch_owner_phase
from .proposal_bench import CUSTOM_PROPOSAL_DOMAIN, ProposalBench, kb_exposure_of
from .framework_discovery import DISCOVER_FAILURE_RETRY_LIMIT, _CandidateDiscoveryMixin
from .framework_records import (
    ProgressStamp,
    ProposalStep,
    _AuthoredOutcomeMixin,
    _record_review_outcome,
    _record_run,
    _record_step,
    _recorder,
    _settle,
)

log = _logging.getLogger(__name__)

# Specialist attempts a local-exploration candidate gets before the phase moves on.
_LOCAL_EXPLORE_MAX_ATTEMPTS: int = 3
# Unified authored-lane max attempts (apply-failure retries + Critic reauthor).
_AUTHORED_LANE_MAX_ATTEMPTS: int = 3


def _critic_feedback_lines(critic_feedback: Mapping[str, Any] | None) -> list[str]:
    """The prompt lines a re-author round seeds from the Critic's prior verdict."""
    if not critic_feedback:
        return []
    required = [str(x).strip() for x in to_list(critic_feedback.get("required_evidence")) if str(x).strip()]
    lines = ["", "PRIOR CRITIC FEEDBACK (re-author round — supply the evidence below this round):"]
    lines.extend(f"  • required evidence: {ev}" for ev in required[:10])
    advice = first_str(critic_feedback.get("advice_text")).strip()
    if advice:
        lines.append(f"- advice: {advice}")
    risks = [str(r).strip() for r in to_list(critic_feedback.get("risks")) if str(r).strip()]
    if risks:
        lines.append("- risks: " + "; ".join(risks[:6]))
    return lines


#: Progress-row status for a candidate the Critic rejected. The gate writes it
#: and the priors reader selects on it, so the ledger is the only record of a
#: denial and both sites agree by construction.
FRAMEWORK_CRITIC_DENIED_STATUS: str = "critic_denied"


def _forward_enablement_carriers(src: dict[str, Any], dst: dict[str, Any]) -> None:
    """Copy the round's enablement context from specialist params to the integrate task.

    The dispatched verdict and the pre-patch boot observation travel on every
    enablement round; the eval-origin trigger context only on an eval-origin one.
    """
    signature = src.get("enablement_failure_signature")
    if isinstance(signature, dict) and signature:
        dst["enablement_failure_signature"] = dict(signature)
    before_path = str(src.get("enablement_before_observation_path") or "")
    if before_path:
        dst["enablement_before_observation_path"] = before_path
    origin = str(src.get("enablement_origin") or "")
    if not origin:
        return
    dst["enablement_origin"] = origin
    dst["enablement_accuracy_floor"] = float(src.get("enablement_accuracy_floor") or 0.0)
    cfg = str(src.get("enablement_probe_config_path") or "")
    if cfg:
        dst["enablement_probe_config_path"] = cfg
        # Bench the candidate against the original workload/eval contract rather
        # than the shipped default config.
        dst.setdefault("config_path", cfg)


def _forward_integrate_source(
    src: dict[str, Any],
    dst: dict[str, Any],
    done_payload: Mapping[str, Any],
) -> None:
    """Preserve proposal ownership across delayed ``integrate_patch`` execution."""

    domain = str(src.get("domain") or "").strip()
    source_phase = patch_owner_phase(src)
    if source_phase:
        dst["source_phase"] = source_phase
    if domain:
        dst["domain"] = domain
        dst["provenance"] = f"specialist:{domain}"
    elif src.get("provenance"):
        # A domain-less dispatch such as a predictor mandate keeps the label it was dispatched under.
        dst["provenance"] = str(src["provenance"])
    # ``framework`` is intentionally not forwarded: integrate_patch consumes
    # that parameter when selecting accuracy parsing/gating behavior, whereas
    # proposal ownership only needs the gap metadata below.
    # ``lever_kind`` travels with the proposal: the patch that lands moved the
    # same lever the specialist was dispatched against, and re-deriving it at
    # writeback time is how attribution drifts. The KB exposure does too: it is
    # what the authoring specialist was shown, and the attempt is recorded later.
    for key in (
        "gap_canonical_id",
        "gap_layer",
        "lever_kind",
        "reauthor_attempt",
        "apply_retry_attempt",
        "kb_read_id",
        "kb_rendered_refs",
    ):
        value = src.get(key)
        if value not in (None, "", [], {}):
            dst[key] = value
    # What the specialist says those Experiences did to the patch it wrote, already checked against them.
    citations = done_payload.get("experience_citations")
    if citations:
        dst["experience_citations"] = list(citations)


def _integrate_review_params(
    sid: str,
    spec_params: dict[str, Any],
    done_payload: dict[str, Any],
    deliverable: dict[str, Any],
) -> dict[str, Any]:
    """Build the synthetic integrate_patch params that review a specialist's deliverable."""
    proposals = done_payload.get("proposal_set")
    first = proposals[0] if isinstance(proposals, list) and proposals else None
    params: dict[str, Any] = {
        "specialist_task_id": sid,
        "provenance": "specialist",
        "patch_name": str(first.get("name") or "") if isinstance(first, dict) else "",
        **deliverable,
    }
    _forward_integrate_source(spec_params, params, done_payload)
    # The authored-outcome bridge keys the progress row on the real candidate id.
    for key in ("framework_agent_candidate_id", "framework_batch_id"):
        value = str(spec_params.get(key) or "")
        if value:
            params[key] = value
    if spec_params.get("enablement"):
        # Without the marker the integrate result never carries
        # ``enablement=True``, so the round is never rearmed, settled or charged.
        params["enablement"] = True
        _forward_enablement_carriers(spec_params, params)
        # Earlier rounds' setup commands plus this deliverable's own, so the
        # round replays its install step before booting.
        setup_commands = [
            *(spec_params.get("enablement_setup_commands") or []),
            *(done_payload.get("setup_commands") or []),
        ]
        merged_setup = list(dict.fromkeys(str(c) for c in setup_commands if str(c)))
        if merged_setup:
            params["enablement_setup_commands"] = merged_setup
    return params


def predictor_budget(state: SessionView) -> PhaseBudget:
    """The FRAMEWORK clock a predictor request is conditioned on, read when the request goes out."""
    from ..predictor.payload import PhaseBudget

    remaining = _phase_state.phase_budget_remaining_seconds(state)
    return PhaseBudget(
        headroom_sec=None if remaining is None else float(remaining),
        elapsed_sec=_phase_state.phase_elapsed_seconds(state),
        keep_threshold_pct=_phase_state.resolve_keep_threshold(state),
    )


class AuthoringRetry(NamedTuple):
    """An apply-failure retry queued for a fresh authoring specialist."""

    lane: str
    candidate: dict[str, Any] | None = None
    batch_id: str = ""
    specialist_task_id: str = ""
    attempt: int = 1
    retry_feedback: list[dict[str, Any]] | None = None
    vetting_drops: list[str] | None = None


def _apply_failure_feedback_lines(retry: AuthoringRetry) -> list[str]:
    """The prompt lines telling an authoring retry why the previous round's patch never landed."""
    lines: list[str] = []
    if retry.vetting_drops:
        lines.append("")
        lines.append(
            "PATCH GROUNDING FAILURE: the prior round's patches were dropped by "
            "the safety gate before reaching integration. The worktree contains "
            "the correct framework tree — edit files there and the diff is "
            "harvested automatically. Do not switch to the artifacts_written "
            "channel to avoid the gate."
        )
        lines.append("Dropped targets: " + "; ".join(retry.vetting_drops[:4]))
        lines.append("")
    if retry.retry_feedback:
        from ..actions.executors._apply_feedback import ApplyFeedback

        lines.append("")
        lines.append(
            "APPLY FAILURE FEEDBACK (previous patch failed to apply; "
            "study the errors below and produce a corrected patch):"
        )
        # Capped for prompt brevity.
        for fb_dict in retry.retry_feedback[:5]:
            fb = ApplyFeedback.from_dict(fb_dict) if isinstance(fb_dict, dict) else None
            if fb is not None:
                lines.append("")
                lines.append(fb.format_for_mandate())
        lines.append("")
    return lines


class FrameworkPhase(_AuthoredOutcomeMixin, _CandidateDiscoveryMixin, ProposalBench):
    """The FRAMEWORK_AGENT phase: upstream candidates, authored patches, deliverable routing, and the enablement hand-off."""

    # Set for the span of one FRAMEWORK entry; ``None`` outside it and when the recorder is unbound.
    _framework_timeline_recorder: Any = None
    # The PrimaTune predictor's in-flight request, created on first use (``predictor.pump.PredictorPump``).
    _predictor: Any = None
    # Max tried-candidate rows fed into the ranker/discovery working memory.
    _FRAMEWORK_TRIED_MEMORY_CAP: int = 12
    # Tail of outcomes from the priors ledger to evaluate.
    _CRITIC_PRIORS_OUTCOME_TAIL: int = 5
    # Backstop: max Critic-review submissions for a single candidate before the pump force-stamps
    # ``repeated_review_abort`` and stops re-selecting it.
    _MAX_REPEATED_REVIEW_SUBMISSIONS: int = 3

    def on_specialist_settled(self, task: "Task", done_payload: dict[str, Any], *, run_error: str) -> None:
        """Stamp an empty authoring round's terminal row and harvest a discovery round's candidates."""
        # An authoring specialist that wrote no patch never spawns an integrate_patch; stamp its terminal row here.
        self._record_framework_agent_authoring_empty_outcome(task=task, done_payload=done_payload, run_error=run_error)
        self._ingest_candidate_discovery(task=task, done_payload=done_payload, run_error=run_error)

    async def on_integrate_patch_settled(self, task: "Task", result: Any, verdict: Verdict) -> None:
        """Record an authored patch's settled outcome, then re-arm or drain the authored lane."""
        if (task.params or {}).get("framework_agent_authoring"):
            self._record_framework_agent_authored_outcome(task=task, result=result, adopted=verdict is Verdict.ADOPTED)
        await self._maybe_rearm_authored_lane(result.result)
        await self._drain_apply_fail_retry_pending()

    def record_unpromoted_candidate(self, task: "Task", result_payload: dict[str, Any]) -> None:
        """Stamp ``no_result_failed`` for an upstream-PR candidate task that settled without a promotable result."""
        params = task.params or {}
        if task.kind != "integrate_patch" or not params.get("framework_agent_candidate_id"):
            return
        cand = params.get("candidate")
        cand_id = self._framework_candidate_key(cand if isinstance(cand, dict) else None)
        if not cand_id:
            return
        self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=str(params.get("batch_id") or ""),
                status="no_result_failed",
                rationale=str(result_payload.get("reason") or result_payload.get("error") or "")[:500],
                provenance="executor",
                extra={"status": str(result_payload.get("status") or "")},
            )
        )

    def timeline(self):
        """Return the recorder for this FRAMEWORK entry, or ``None``."""
        return _recorder(self)

    def _open_framework_timeline(self) -> None:
        """Open the timeline event for this FRAMEWORK entry and record its policy.

        The policy is recorded here because this is the first point at which
        every threshold the entry will run under is resolvable: the reprofile
        has settled the anchor and the macro cycle is fixed.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.framework_event import make_framework_recorder

        state = self.shared_state
        recorder = make_framework_recorder(macro_cycle=int(state.macro_cycle or 0))
        self._framework_timeline_recorder = recorder
        if recorder is None:
            return
        recorder.record_policy(**self._framework_policy_fields())

    def _framework_policy_fields(self) -> dict:
        """Resolve the ``record_policy`` fields this entry runs under."""
        state = self.shared_state
        overrides = state.plateau_overrides or {}
        if not isinstance(overrides, dict):
            overrides = {}
        return {
            "keep_threshold_pct": _phase_state.resolve_keep_threshold(state),
            "config": {
                "keep_gain_threshold_pct": overrides.get(
                    "explore_keep_gain_pct",
                    _phase_state.DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
                ),
                "empty_streak_threshold": overrides.get(
                    "explore_empty_streak",
                    _phase_state.DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
                ),
                "lookback": overrides.get(
                    "explore_lookback",
                    _phase_state.DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
                ),
            },
            "source": {
                "no_keep_streak_threshold": overrides.get(
                    "framework_no_keep_streak",
                    _phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK,
                ),
                "discovery_retry_limit": DISCOVER_FAILURE_RETRY_LIMIT,
            },
        }

    def close_framework_timeline(self, tr: "Transition") -> None:
        """Close the FRAMEWORK timeline event when the phase is left."""
        exit_reason = tr.reason
        evidence: dict | None = tr.evidence if tr.evidence else None
        recorder = self.timeline()
        if recorder is None:
            return
        self._framework_timeline_recorder = None
        facts = dict(evidence or {})
        self._record_framework_exit_plateau(recorder, facts)
        recorder.finish(
            exit_reason=exit_reason,
            trigger=str(facts.get("evidence") or ""),
            hint=str(facts.get("hint") or ""),
            switch_bottleneck=facts.get("switch_bottleneck"),
        )

    @staticmethod
    def _record_framework_exit_plateau(recorder, evidence: dict) -> None:
        """Record both arms' plateau readings as the exit rule saw them.

        Skipped when the evidence carries no plateau reading -- a transition
        that did not come from the optimize exit rule -- since writing rows
        then would report an evaluation that never ran.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.framework_event import PLATEAU_PATH_EXIT

        if "config_arm_plateaued" not in evidence and "source_arm_plateaued" not in evidence:
            return
        FrameworkPhase._record_plateau_rows(recorder, path=PLATEAU_PATH_EXIT, evidence=evidence)

    def _record_advisory_plateau(self) -> None:
        """Record the current per-lever dryness reading on the advisory path.

        Both arms are recorded whether or not either tripped, so the timeline
        carries the reading as it stood when the advisory was built.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.framework_event import PLATEAU_PATH_ADVISORY

        recorder = self.timeline()
        if recorder is None:
            return
        _, evidence = _phase_state.per_lever_dryness(self.shared_state)
        self._record_plateau_rows(recorder, path=PLATEAU_PATH_ADVISORY, evidence=evidence)

    @staticmethod
    def _record_plateau_rows(recorder, *, path: str, evidence: dict) -> None:
        """Record the config and source arms' plateau readings with the values they ruled on."""
        from hyperloom.inference_optimizer.breakdown.recorder.framework_event import ARM_CONFIG, ARM_SOURCE

        recorder.record_plateau(
            arm=ARM_CONFIG,
            path=path,
            triggered=evidence.get("config_arm_plateaued"),
            inputs={
                "recent_keep_gain_pct": evidence.get("recent_keep_gain_pct"),
                "empty_streak": evidence.get("empty_streak"),
                "winners_seen": evidence.get("winners_seen"),
                "specialist_rounds_seen": evidence.get("specialist_rounds_seen"),
            },
            thresholds={
                "keep_gain_threshold_pct": evidence.get("keep_gain_threshold_pct"),
                "empty_streak_threshold": evidence.get("empty_streak_threshold"),
                "lookback": evidence.get("lookback"),
            },
        )
        recorder.record_plateau(
            arm=ARM_SOURCE,
            path=path,
            triggered=evidence.get("source_arm_plateaued"),
            inputs={
                "consecutive_no_keep": evidence.get("source_consecutive_no_keep"),
                "candidates_exhausted": evidence.get("source_candidates_exhausted"),
            },
            thresholds={"no_keep_streak_threshold": evidence.get("source_threshold")},
        )

    async def on_enter_framework(self, tr: "Transition") -> None:
        """FRAMEWORK entry hook: trigger the per-batch pump once on entry; later batches are driven from the main tick."""
        log.info(
            "OPTIMIZE entry (from=%s): pumping initial batch",
            tr.from_phase or "<unknown>",
        )
        # A reopened macro-cycle re-measures before either arm spends anything.
        await self._coord.phase_macro_cycle.on_cycle_start_reprofile(from_phase=tr.from_phase)
        # Opened after the reprofile so the policy reads the settled anchor,
        # and before the pump so the entry's first dispatch is inside the event.
        self._open_framework_timeline()
        # Queued before orchestration's first turn so the tick's dispatch reaches it ahead of what that turn delegates.
        await self._coord.specialist_dispatch.ensure_custom_specialist()
        if self.shared_state.custom_specialist_only:
            return
        await self._pump_framework_agent_phase()
        # Before orchestration's first turn, so the first grid it composes can carry the answer; last, as on the
        # tick, so a predictor fault reaches the transition's guard after the entry's own work has run.
        await self._predictor_pump().settle(self.shared_state)

    def _predictor_pump(self) -> Any:
        from ..predictor.pump import PredictorPump

        if self._predictor is None:
            self._predictor = PredictorPump(budget=predictor_budget, on_filed=self._record_predictor_round)
        return self._predictor

    def _record_predictor_round(self, entry: dict[str, Any]) -> None:
        """Record a filed predictor answer on this entry's event, as the config-arm run that produced it.

        Keyed by the round id, since no task produced it; only an answer that
        queued something is filed, so the run never reads as a failure.
        """
        from hyperloom.inference_optimizer.breakdown.recorder.framework_event import ROLE_CONFIG

        rows = entry.get("proposal_set") or []
        mandate = str(entry.get("mandate_id") or "")
        _record_run(
            self,
            str(entry.get("round_id") or ""),
            role=ROLE_CONFIG,
            status="succeeded",
            domain=str(entry.get("domain") or ""),
            dispatched_at=str(entry.get("asked_at") or ""),
            completed_at=str(entry.get("completed_at") or ""),
            proposals_total=len(rows),
            summary=f"{len(rows)} config row(s)" + (f", source-change mandate {mandate}" if mandate else ""),
        )

    def _authoring_inflight_candidate_ids(self, queued: list[Any], running: list[Any]) -> set[str]:
        """Candidate ids with a live specialist / integrate_patch task or a review proposal awaiting its verdict."""
        in_flight = {
            str(t.params.get("framework_agent_candidate_id") or "")
            for t in (*queued, *running)
            if t.kind in ("specialist", "integrate_patch") and t.params
        }
        for p in self.state.pending_proposals.values():
            if p.action_name != "integrate_patch":
                continue
            payload = p.payload or {}
            in_flight.add(
                str(
                    payload.get("framework_agent_candidate_id")
                    or (payload.get("params") or {}).get("framework_agent_candidate_id")
                    or ""
                )
            )
        in_flight.discard("")
        return in_flight

    async def _pump_framework_agent_phase(self) -> None:
        """Drive the FRAMEWORK_AGENT phase: submit candidates for review in parallel.

        Up to ``research_lane_capacity`` candidates are in flight at once; landing still serialises on
        ``workspace_mutation``. The discovery → local-explore → phase-done fallback runs only once no candidate is
        left and none is in flight; the local-explore rung is skipped once the source lever reads dry. Idempotent; a
        discover failure flips ``framework_agent_phase_done`` so the phase advances rather than wedging.
        """
        state = self.shared_state
        if (state.phase or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return
        if bool(state.framework_agent_phase_done):
            return
        in_flight_ids = self._authoring_inflight_candidate_ids(await self.tasks.queued(), await self.tasks.running())
        submitted = False
        while len(in_flight_ids) < max(1, state.research_lane_capacity):
            candidate = self._select_next_framework_agent_candidate(exclude_ids=in_flight_ids)
            if candidate is None:
                break
            await self._submit_framework_agent_candidate_for_review(
                candidate,
                audit=dict(candidate.get("audit") or {}),
                audit_step=str(candidate.get("route") or "author_via_specialist"),
            )
            in_flight_ids.add(self._framework_candidate_key(candidate))
            submitted = True
        if submitted or in_flight_ids or await self._framework_agent_authoring_inflight():
            return
        # Minimum supply: with the pool empty and no discovery in flight, ask for one.
        if await self._maybe_enqueue_candidate_discovery(reason="candidate_pool_empty"):
            state.save(self.session_dir)
            return
        source_dry = _phase_state.per_lever_dryness(state)[1]["source_arm_plateaued"]
        if (
            self.shared_state.framework_local_explore_enabled
            and not source_dry
            and int(state.research_lane_capacity or 0) > 0
        ):
            gap, keywords = self._compose_framework_local_explore_gap()
            title = (
                f"local source exploration ({gap})"
                if gap
                else "local source exploration (author a throughput patch from live source + profile)"
            )
            dispatched = await self._enqueue_framework_agent_local_explore_specialist(
                {
                    "title": title,
                    "repo": "(local source)",
                    "framework": str(state.framework or "").strip().lower(),
                    "gap_description": gap,
                    "gap_keywords": keywords,
                },
                reason="no_new_candidates",
            )
            if dispatched:
                state.save(self.session_dir)
                return
        self._record_framework_agent_phase_done(
            reason="no_candidates_and_discovery_exhausted",
            failure_count=int(state.framework_agent_discover_failures or 0),
        )
        state.framework_agent_phase_done = True
        state.save(self.session_dir)

    async def _framework_agent_authoring_inflight(self) -> bool:
        """True while a FRAMEWORK-authored patch for an unprocessed candidate is still in flight."""
        processed_ids = self._framework_processed_candidate_keys()

        def _cand_pins_pump(cand_id: str) -> bool:
            """True when an authoring cand_id keeps the pump serialized: it has no terminal progress row yet.

            A candidate-free local-exploration id never appears in a PR batch, so
            the settled outcome -- not batch membership -- is what releases it.
            """
            return not cand_id or cand_id not in processed_ids

        queued = await self.tasks.queued()
        running = await self.tasks.running()
        for t in (*queued, *running):
            if getattr(t, "kind", "") not in ("specialist", "integrate_patch"):
                continue
            params = getattr(t, "params", None) or {}
            if not params.get("framework_agent_authoring"):
                continue
            if _cand_pins_pump(str(params.get("framework_agent_candidate_id") or "")):
                return True
        # An authored patch awaiting Critic review (or a candidate awaiting its pre-screen verdict) keeps the phase
        # open, but only while the proposal targets a still-unprocessed candidate.
        for p in self.state.pending_proposals.values():
            if getattr(p, "action_name", "") != "integrate_patch":
                continue
            payload = getattr(p, "payload", None) or {}
            # Both the candidate pre-screen and the authored patch are ``integrate_patch`` proposals now, so the
            # candidate marker -- not the action name -- says which candidate is pinned.
            iparams = payload.get("params") or {}
            cand_id = str(
                payload.get("framework_agent_candidate_id") or iparams.get("framework_agent_candidate_id") or ""
            )
            if not cand_id and not iparams.get("framework_agent_authoring"):
                continue
            if _cand_pins_pump(cand_id):
                return True
        return False

    @staticmethod
    def _framework_agent_audit_seed_lines(audit: dict[str, Any] | None) -> list[str]:
        """Render audit evidence as authoring-seed lines (empty when no audit)."""
        if not isinstance(audit, dict) or not audit:
            return []
        lines = [
            "",
            "CANDIDATE REVIEW (author against the LIVE source, not the raw diff):",
            f"- verdict: {audit.get('verdict') or 'unknown'}",
        ]
        reason = str(audit.get("reason") or "").strip()
        if reason:
            lines.append(f"- reason: {reason}")
        next_step = str(audit.get("recommended_next_step") or "").strip()
        if next_step:
            lines.append(f"- recommended next step: {next_step}")
        return lines

    async def _settle_finished_authoring_specialist(
        self, spec_task: Any, *, cand_id: str, batch_id: str, label: str
    ) -> bool:
        """Whether the reused specialist already finished; recovers its outcome once when no row carries it yet."""
        from ..state.task_registry import TERMINAL_STATES

        if str(getattr(spec_task, "state", "") or "") not in TERMINAL_STATES:
            return False
        already_rows = self._framework_processed_candidate_keys()
        authoring_inflight = await self._framework_agent_authoring_inflight()
        if not cand_id or cand_id in already_rows or authoring_inflight:
            return True
        recovered = await self._recover_framework_agent_authoring_outcome(specialist_task=spec_task)
        if not recovered:
            log.warning(
                "FRAMEWORK %s: terminal outcome unavailable candidate=%s state=%s",
                label,
                cand_id,
                getattr(spec_task, "state", ""),
            )
            # Stamp a terminal row so an unrecoverable outcome cannot make the pump re-select the same
            # finished specialist forever.
            self._stamp_framework_progress(
                ProgressStamp(
                    candidate_id=cand_id,
                    batch_id=batch_id,
                    status="recovery_failed",
                    rationale=f"{label} outcome unrecoverable from persisted results",
                    provenance="pump",
                )
            )
        return True

    def _map_authoring_specialist(self, spec_task: Any, *, cand_id: str, label: str) -> str:
        """Record specialist task -> candidate so the authored-outcome bridge can resolve the candidate id from the
        downstream integrate_patch; returns the specialist task id."""
        state = self.shared_state
        spec_tid = str(getattr(spec_task, "task_id", "") or "")
        try:
            if spec_tid and cand_id:
                if not isinstance(state.framework_agent_specialist_candidate_map, dict):
                    state.framework_agent_specialist_candidate_map = {}
                state.framework_agent_specialist_candidate_map[spec_tid] = cand_id
                state.save(self.session_dir)
        except OSError:
            log.debug("FRAMEWORK %s: specialist->candidate map write failed", label, exc_info=True)
        return spec_tid

    async def _enqueue_framework_agent_authoring_specialist(
        self,
        candidate: dict[str, Any],
        audit: dict[str, Any] | None = None,
        *,
        reauthor_attempt: int = 0,
        critic_feedback: dict[str, Any] | None = None,
    ) -> str:
        """Dispatch a write-capable specialist seeded with ``candidate`` (flows through autosubmit → Critic → integrate_patch → bench → KEEP/REVERT)."""
        state = self.shared_state
        cand_id = self._framework_candidate_key(candidate)
        batch_id = first_str(candidate.get("batch_id"))
        gap_cid = first_str(candidate.get("gap_canonical_id")).strip() or f"gap.framework.{cand_id}"
        title = first_str(candidate.get("title")).strip()
        pr_url = first_str(candidate.get("pr_url")).strip()
        notes_lines = [
            *self._framework_agent_audit_seed_lines(audit),
            *_critic_feedback_lines(critic_feedback),
        ]
        notes = "\n".join(notes_lines).strip()
        params: dict[str, Any] = {
            "domain": self._authoring_specialist_domain(),
            "gap_canonical_id": gap_cid,
            "gap_symptom": (title or f"Author a framework source patch inspired by {pr_url or cand_id}"),
            "gap_layer": "framework",
            "framework": first_str(candidate.get("framework"), state.framework).strip().lower(),
            "task_kind": "framework_authoring",
            "source_phase": "FRAMEWORK_AGENT",
            "pr_lead": {"title": title, "url": pr_url, "diff_url": first_str(candidate.get("diff_url")).strip()},
            "lever_kind": LEVER_UPSTREAM_PR,
            # Provenance markers for the dispatcher-side authored-patch bridge.
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand_id,
            "framework_batch_id": batch_id,
            "reauthor_attempt": int(reauthor_attempt),
            "framework_audit": to_mapping(audit),
            "source": "coordinator_internal",
            "notes": notes,
        }
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        reauthored = f":reauthor:{int(reauthor_attempt)}" if reauthor_attempt > 0 else ""
        idem = f"framework_agent_authoring:{batch_id}:{cand_id}{reauthored}"
        # This internal dispatch bypasses intent_router (adds gpu_research_lane + budget TTL).
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        spec_task, _spec_existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params=params,
                idempotency_key=idem,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        if _spec_existing and await self._settle_finished_authoring_specialist(
            spec_task, cand_id=cand_id, batch_id=batch_id, label="authoring"
        ):
            return ""
        spec_tid = self._map_authoring_specialist(spec_task, cand_id=cand_id, label="authoring")
        _record_run(
            self,
            spec_tid,
            role="authoring",
            status="dispatched",
            domain=str(params.get("domain") or ""),
            gap_canonical_id=gap_cid,
            parallelism=len(lanes or ()),
        )
        # A re-author is the same lifecycle step again rather than a counter on
        # the proposal: discovery already produced the candidate.
        _record_step(
            self,
            cand_id,
            ProposalStep(step="reauthored" if reauthored else "authored", run_ref=spec_tid, outcome="dispatched"),
        )
        log.info("FRAMEWORK: dispatched authoring specialist candidate=%s batch=%s gap=%s", cand_id, batch_id, gap_cid)
        return spec_tid

    async def _maybe_rearm_authored_lane(self, res: dict[str, Any] | None) -> None:
        """Route an authored lane's result to the handler that rearms it: enablement settles, a perf apply-failure retries under cap."""
        if not isinstance(res, dict):
            return
        lane = first_str(res.get("lane"))
        if lane == "enablement" or res.get("enablement"):
            await self._coord.enablement_lane.maybe_rearm_enablement(res)
            return
        # Every other perf-lane status goes through the writeback path.
        if first_str(res.get("status")) != "apply_failed" or lane not in ("perf_framework", "perf_explore"):
            return

        candidate = to_mapping(res.get("candidate"))
        cand_id = self._framework_candidate_key(candidate) or first_str(res.get("specialist_task_id")).strip()
        if not cand_id:
            return
        batch_id = first_str(candidate.get("batch_id"), res.get("batch_id"))

        state = self.shared_state
        apply_fail_attempts = to_mapping(state.apply_fail_reauthor_attempts)
        attempt = int(apply_fail_attempts.get(cand_id, 0) or 0) + 1
        apply_fail_attempts[cand_id] = attempt
        state.apply_fail_reauthor_attempts = apply_fail_attempts

        log.info(
            "AUTHORED_LANE rearm: lane=%s cand_id=%s apply_fail_attempt=%d cap=%d",
            lane,
            cand_id,
            attempt,
            _AUTHORED_LANE_MAX_ATTEMPTS,
        )

        if attempt > _AUTHORED_LANE_MAX_ATTEMPTS:
            self._stamp_framework_progress(
                ProgressStamp(
                    candidate_id=cand_id,
                    batch_id=batch_id,
                    status="apply_fail_cap",
                    rationale=f"apply_failed {attempt} times (cap={_AUTHORED_LANE_MAX_ATTEMPTS})",
                    provenance="apply_fail_retry",
                )
            )
            try:
                state.save(self.session_dir)
            except OSError:
                log.debug("authored_lane: save after cap stamp failed", exc_info=True)
            return

        retry_ctx: dict[str, Any] = {
            "cand_id": cand_id,
            "batch_id": batch_id,
            "lane": lane,
            "attempt": attempt,
            "retry_feedback": to_list(res.get("retry_feedback")),
            "candidate": candidate,
            "specialist_task_id": first_str(res.get("specialist_task_id")),
        }
        vetting_drops = to_list(res.get("patches_ungrounded"))
        if vetting_drops:
            retry_ctx["vetting_drops"] = [str(d) for d in vetting_drops[:8]]
        pending = to_list(state.apply_fail_retry_pending)
        pending.append(retry_ctx)
        state.apply_fail_retry_pending = pending
        try:
            state.save(self.session_dir)
        except OSError:
            log.debug("authored_lane: save after retry-pending failed", exc_info=True)

    async def _authoring_specialist_params(self, task_id: str) -> dict[str, Any]:
        """The dispatch params of the authoring specialist ``task_id`` names; empty when the registry has no such row."""
        if not task_id:
            return {}
        try:
            return to_mapping((await self.tasks.get(task_id)).params)
        except TaskNotFound:
            return {}

    async def _enqueue_author_specialist(self, retry: AuthoringRetry) -> str:
        """Dispatch a fresh authoring specialist for an apply-failure retry."""
        candidate = to_mapping(retry.candidate)
        if retry.batch_id and not candidate.get("batch_id"):
            candidate["batch_id"] = retry.batch_id
        feedback = "\n".join(_apply_failure_feedback_lines(retry))
        if retry.lane == "perf_framework":
            return await self._reauthor_framework_candidate(retry, candidate, feedback)
        if retry.lane == "perf_explore":
            return await self._reauthor_explore_patch(retry, feedback)
        log.warning("_enqueue_author_specialist: unsupported lane=%s — skipping", retry.lane)
        return ""

    async def _reauthor_framework_candidate(
        self, retry: AuthoringRetry, candidate: dict[str, Any], feedback: str
    ) -> str:
        """Re-dispatch the upstream-PR authoring specialist, seeding it with the apply failure as Critic feedback."""
        cand_id = self._framework_candidate_key(candidate)
        if not cand_id:
            log.warning("_enqueue_author_specialist: perf_framework missing cand_id")
            return ""
        spec_params = await self._authoring_specialist_params(retry.specialist_task_id)
        new_task_id = await self._enqueue_framework_agent_authoring_specialist(
            candidate,
            audit=to_mapping(spec_params.get("framework_audit")),
            reauthor_attempt=retry.attempt,
            critic_feedback={"advice_text": feedback} if feedback else None,
        )
        log.info(
            "AUTHORED_LANE: dispatched perf_framework retry specialist cand=%s attempt=%d task=%s",
            cand_id,
            retry.attempt,
            new_task_id,
        )
        return new_task_id

    async def _reauthor_explore_patch(self, retry: AuthoringRetry, feedback: str) -> str:
        """Re-dispatch a serving specialist on the gap the apply-failed local-exploration patch came from."""
        spec_params = await self._authoring_specialist_params(retry.specialist_task_id)
        gap_cid = first_str(spec_params.get("gap_canonical_id")).strip()
        if not gap_cid:
            gap_cid = f"gap.explore.retry.{retry.specialist_task_id or 'unknown'}"
        gap_symptom = first_str(spec_params.get("gap_symptom")).strip()
        framework_name = first_str(spec_params.get("framework"), self.shared_state.framework).strip().lower()
        notes = feedback.strip()
        params: dict[str, Any] = {
            "domain": "serving_specialist",
            "source_phase": "FRAMEWORK_AGENT",
            "lever_kind": LEVER_SOURCE_PATCH,
            "gap_canonical_id": gap_cid,
            "gap_symptom": gap_symptom or f"Retry apply-failed patch for {gap_cid}",
            "gap_layer": "perf_explore",
            "framework": framework_name,
            "task_kind": "explore_apply_retry",
            "source": "coordinator_internal",
            "notes": notes,
            "apply_retry_attempt": retry.attempt,
        }
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        # Gap id and attempt both repeat across cycles.
        idem = f"perf_explore_authoring:{gap_cid}:retry:{retry.attempt}{self._coord.dispatcher.cycle_idem_suffix()}"
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        spec_task, _ = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="specialist",
                params=params,
                idempotency_key=idem,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        new_tid = str(getattr(spec_task, "task_id", "") or "")
        log.info(
            "AUTHORED_LANE: dispatched perf_explore retry specialist gap=%s attempt=%d task=%s",
            gap_cid,
            retry.attempt,
            new_tid,
        )
        return new_tid

    async def _drain_apply_fail_retry_pending(self) -> None:
        """Dispatch authoring specialists for any queued apply-failure retries."""
        state = self.shared_state
        pending: list[dict[str, Any]] = state.apply_fail_retry_pending or []
        if not isinstance(pending, list) or not pending:
            return
        # Consume the list atomically so a concurrent call doesn't double-fire.
        to_dispatch = list(pending)
        state.apply_fail_retry_pending = []
        for ctx in to_dispatch:
            if not isinstance(ctx, dict):
                continue
            lane = str(ctx.get("lane") or "")
            attempt = int(ctx.get("attempt") or 1)
            candidate = ctx.get("candidate") or {}
            specialist_task_id = str(ctx.get("specialist_task_id") or "")
            batch_id = str(ctx.get("batch_id") or "")
            retry_feedback = list(ctx.get("retry_feedback") or [])
            vetting_drops = list(ctx.get("vetting_drops") or [])
            await self._enqueue_author_specialist(
                AuthoringRetry(
                    lane=lane,
                    candidate=candidate,
                    batch_id=batch_id,
                    specialist_task_id=specialist_task_id,
                    attempt=attempt,
                    retry_feedback=retry_feedback,
                    vetting_drops=vetting_drops or None,
                )
            )
        try:
            state.save(self.session_dir)
        except OSError:
            log.debug("drain_apply_fail: save failed", exc_info=True)

    @staticmethod
    def _framework_candidate_key(row: dict[str, Any] | None) -> str:
        """Canonical FRAMEWORK candidate dedup/progress key (see ``candidate_key``)."""
        from ..framework.artifacts import candidate_key

        return candidate_key(row)

    def _framework_processed_candidate_keys(self) -> set[str]:
        """Set of candidate keys that already carry a terminal progress row."""
        return {
            self._framework_candidate_key(p)
            for p in (self.shared_state.framework_agent_phase_progress or [])
            if isinstance(p, dict) and self._framework_candidate_key(p)
        }

    def _unprocessed_framework_agent_candidates(self) -> list[dict[str, Any]]:
        """Return all not-yet-processed candidates in the latest batch (order preserved)."""
        state = self.shared_state
        batches = state.framework_agent_batches or []
        if not batches:
            return []
        latest = batches[-1]
        if not isinstance(latest, dict):
            return []
        candidates = latest.get("candidates") or []
        if not isinstance(candidates, list):
            return []
        processed = self._framework_processed_candidate_keys()
        out: list[dict[str, Any]] = []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            cand_id = self._framework_candidate_key(cand)
            if cand_id and cand_id not in processed:
                out.append(cand)
        return out

    def _select_next_framework_agent_candidate(self, exclude_ids: Collection[str] = ()) -> dict[str, Any] | None:
        """Return the next unprocessed candidate whose id is not in ``exclude_ids``, in batch order."""
        for cand in self._unprocessed_framework_agent_candidates():
            if self._framework_candidate_key(cand) not in exclude_ids:
                return cand
        return None

    def _authoring_specialist_domain(self) -> str:
        """Pick the authoring domain that matches the session's framework kind."""
        from ..specialists.domains import authoring_domain_for_framework

        return authoring_domain_for_framework(self.shared_state.framework)

    def _render_rewrite_evidence_for_prompt(self) -> str:
        """Render the measured host-side rewrite evidence as prompt lines."""
        path = str(self.shared_state.last_framework_rewrite_evidence or "").strip()
        if not path:
            return ""
        try:
            import json as _json

            from ..actions.executors._framework_rewrite_evidence import summarize_for_prompt

            document = _json.loads(Path(path).read_text(encoding="utf-8"))
            return summarize_for_prompt(document)
        except (OSError, ValueError):
            log.warning("FRAMEWORK: rewrite-evidence render failed path=%s", path, exc_info=True)
            return ""

    def _rewrite_evidence_absence_note(self) -> str:
        """Explain an empty evidence block instead of implying there is nothing to find."""
        read_the_source = (
            "Locate the candidates by reading the source: find the denoising / "
            "rollout loop and ask, for each call inside it, whether the result "
            "can change across iterations."
        )
        status = str(self.shared_state.last_framework_rewrite_evidence_status or "").strip()
        if status == "no_candidates":
            return (
                "The host-side probe ran and found no rewrite candidates. Treat "
                "that as a measured negative for the patterns it covers "
                "(host round-trips, host syncs, device residency, collective "
                "fusion, memoization, loop hoisting) and look elsewhere. " + read_the_source
            )
        if status and status != "ok":
            return (
                "No host-side rewrite evidence is available because the probe did "
                f"not deliver any: {status}. This is a broken instrument, NOT a "
                "measured negative -- do not conclude the loop is clean. " + read_the_source
            )
        if str(self.shared_state.last_framework_rewrite_evidence or "").strip():
            # Reached only when a document is on record but rendering it produced nothing, so the evidence exists and
            # this prompt cannot show it.
            return (
                "Host-side rewrite evidence was collected but could not be "
                "rendered here, so its absence below means nothing. " + read_the_source
            )
        return "No host-side rewrite evidence has been collected yet for this session. " + read_the_source

    def _compose_framework_local_explore_gap(self) -> tuple[str, list[str]]:
        """Compose the ``(gap, keywords)`` steering the local-exploration arm."""
        state = self.shared_state
        from ..actions.executors._framework_gap_composer import compose_gap

        return compose_gap(
            framework=str(state.framework or ""),
            gpu_type=str(state.gpu_type or ""),
            model_class=str(state.model_class or ""),
            precision=str(state.precision or ""),
            rewrite_evidence_path=state.last_framework_rewrite_evidence or None,
        )

    def _record_local_explore_proposal(
        self, cand_id: str, candidate: Mapping[str, Any], params: Mapping[str, Any], *, gap_cid: str, reason: str
    ) -> None:
        """Record a local-exploration round as a proposal the orchestration itself produced."""
        recorder = _recorder(self)
        if recorder is None:
            return
        from hyperloom.inference_optimizer.breakdown.recorder.framework_event import (
            ARM_SOURCE,
            PRODUCER_ORCHESTRATION,
            STEP_PROPOSED,
        )

        recorder.record_proposal(
            cand_id,
            arm=ARM_SOURCE,
            producer=PRODUCER_ORCHESTRATION,
            title=first_str(candidate.get("title")) or cand_id,
            reasoning=str(params["gap_symptom"]),
            gap_canonical_id=gap_cid,
            route="author_via_specialist",
        )
        recorder.record_proposal_step(cand_id, step=STEP_PROPOSED, reason=reason)

    async def _claim_local_explore_specialist(self, params: dict[str, Any], cand_id: str) -> tuple[Any, bool] | None:
        """Claim the specialist row for a local-exploration round, keying past rows an earlier attempt left failed."""
        # The registry de-duplicates by key and hands back whatever row it finds, so a candidate whose specialist
        # failed keeps resolving to that failure: the phase re-selects the candidate every tick, logs a dispatch, and
        # nothing runs.
        lanes, ttl = self._coord.gpu_lanes.framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        base_idem = f"framework_agent_local_explore:{cand_id}{self._coord.dispatcher.cycle_idem_suffix()}"
        for attempt in range(_LOCAL_EXPLORE_MAX_ATTEMPTS):
            idem = base_idem if attempt == 0 else f"{base_idem}:r{attempt}"
            spec_task, existing = await self.tasks.create_or_return_existing(
                TaskSpec(
                    kind="specialist",
                    params=params,
                    idempotency_key=idem,
                    requires_lanes=lanes,
                    lease_ttl_sec=ttl,
                    dispatch_class="coordinator",
                )
            )
            if not (existing and first_str(getattr(spec_task, "state", "")) == "failed"):
                return spec_task, existing
            log.info(
                "FRAMEWORK local-explore: %s already failed under %s; retrying candidate %s",
                getattr(spec_task, "task_id", "?"),
                idem,
                cand_id,
            )
        log.warning(
            "FRAMEWORK local-explore: candidate %s exhausted %d specialist "
            "attempt(s); leaving it to the phase to select another",
            cand_id,
            _LOCAL_EXPLORE_MAX_ATTEMPTS,
        )
        return None

    async def _enqueue_framework_agent_local_explore_specialist(
        self,
        candidate: dict[str, Any],
        *,
        reason: str = "",
    ) -> str:
        """Dispatch a candidate-free authoring specialist (no upstream PR lead)."""
        state = self.shared_state
        # A local-exploration round has no upstream lead to key on, so its id counts the rounds already settled.
        settled = sum(
            1
            for p in to_list(state.framework_agent_phase_progress)
            if first_str(to_mapping(p).get("candidate_id")).startswith(LOCAL_EXPLORE_CANDIDATE_PREFIX)
        )
        cand_id = self._framework_candidate_key(candidate) or f"{LOCAL_EXPLORE_CANDIDATE_PREFIX}{settled}"
        gap = first_str(candidate.get("gap_description")).strip()
        gap_cid = first_str(candidate.get("gap_canonical_id")).strip() or f"gap.framework.local_explore.{cand_id}"
        framework = first_str(candidate.get("framework"), state.framework).strip().lower()
        domain = self._authoring_specialist_domain()
        notes = ""
        if domain == "framework_rewrite_specialist":
            notes = self._render_rewrite_evidence_for_prompt() or self._rewrite_evidence_absence_note()
        state.upsert_gap(
            {
                "canonical_id": gap_cid,
                "symptom": gap or "Author a throughput patch from live source + profiling evidence",
                "layer": "framework",
                "severity": "medium",
                "domain_hint": domain,
                "source": "coordinator_internal",
            }
        )
        memory = self._build_framework_working_memory()
        prior_attempts = [
            t for t in to_list(memory.get("tried_and_why")) if first_str(to_mapping(t).get("ref")).strip()
        ]
        params: dict[str, Any] = {
            "domain": domain,
            "source_phase": "FRAMEWORK_AGENT",
            "gap_canonical_id": gap_cid,
            # No ``lever_kind``: this arm names a gap, not a lever, and its specialist returns either.
            "gap_symptom": (gap or "Author a framework source patch from live source + profile evidence"),
            "gap_layer": "framework",
            "framework": framework,
            "task_kind": "framework_local_explore",
            "prior_attempts": prior_attempts,
            "notes": notes,
            # Same provenance markers as the PR-authoring track so the autosubmit -> integrate_patch ->
            # authored-outcome bridge applies.
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand_id,
            "framework_batch_id": "",
            "framework_audit": {},
            "framework_local_explore": True,
            "source": "coordinator_internal",
        }
        self._record_local_explore_proposal(cand_id, candidate, params, gap_cid=gap_cid, reason=reason)
        await self._coord.specialist_dispatch.warm_specialist_params(params)
        claimed = await self._claim_local_explore_specialist(params, cand_id)
        if claimed is None:
            return ""
        spec_task, _spec_existing = claimed
        if _spec_existing and await self._settle_finished_authoring_specialist(
            spec_task, cand_id=cand_id, batch_id=str(candidate.get("batch_id") or ""), label="local-explore"
        ):
            return ""
        spec_tid = self._map_authoring_specialist(spec_task, cand_id=cand_id, label="local-explore")
        log.info(
            "FRAMEWORK: dispatched local-exploration specialist candidate=%s gap=%s reason=%s",
            cand_id,
            gap_cid,
            reason or "resident",
        )
        return spec_tid

    def _framework_known_candidate_ids(self) -> set[str]:
        """All candidate ids already discovered into any prior batch (dedup for new batches)."""
        state = self.shared_state
        ids: set[str] = set()
        batches = state.framework_agent_batches or []
        if not isinstance(batches, list):
            return ids
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            for cand in batch.get("candidates") or []:
                if not isinstance(cand, dict):
                    continue
                # Canonical key (candidate_id/pr_url/ref); synthetic repo-PR fallback only when the candidate carries
                # no identity field.
                cid = self._framework_candidate_key(cand) or f"{cand.get('repo', '')}-{cand.get('pr_number', '')}"
                if cid:
                    ids.add(cid)
        # Fold in PR ids the research scout already mined so the two mechanisms never re-process a PR.
        for pid in state.research_scout_seen_pr_ids or []:
            pid = str(pid or "").strip()
            if pid:
                ids.add(pid)
        return ids

    def _build_framework_working_memory(self) -> dict[str, Any]:
        """Summarise the most recent tried candidates from the progress ledger (deterministic, zero-LLM)."""
        state = self.shared_state
        progress = state.framework_agent_phase_progress or []
        rows = [p for p in progress if isinstance(p, dict) and self._framework_candidate_key(p)]
        tried: list[dict[str, Any]] = []
        for row in rows[-self._FRAMEWORK_TRIED_MEMORY_CAP :]:
            status = str(row.get("status") or "").strip()
            gain = row.get("gain_pct")
            why = str(row.get("rationale") or row.get("reason") or "").strip()
            tried.append(
                {
                    "ref": self._framework_candidate_key(row),
                    "status": status,
                    "gain_pct": (float(gain) if isinstance(gain, (int, float)) else None),
                    "why": why[:200],
                }
            )
        return {
            "tried_and_why": tried,
        }

    def _record_framework_agent_phase_done(
        self,
        *,
        reason: str,
        failure_count: int,
    ) -> None:
        """Append a framework_agent_phase_done row to phase_history describing why the pump gave up."""
        state = self.shared_state
        from ..framework.artifacts import summarize_candidate_outcomes

        # Classify this phase's candidate outcomes so the report / robustness can tell "discovered nothing"
        # (empty_discovery) apart from "tested candidates but none kept" (tested_no_keep).
        summary = summarize_candidate_outcomes(state.framework_agent_phase_progress)
        outcome_class = summary["outcome_class"]

        # Consecutive empty-discovery tracking → advisory ("framework phase ineffective").
        empty = outcome_class == "empty_discovery"
        consecutive_empty = state.framework_consecutive_empty_discoveries + 1 if empty else 0
        state.framework_consecutive_empty_discoveries = consecutive_empty

        advisory = ""
        if empty and consecutive_empty >= 2:
            advisory = (
                f"framework phase ineffective: {consecutive_empty} consecutive macro-cycles discovered zero candidates"
            )
        elif outcome_class == "tested_no_keep":
            advisory = (
                f"framework phase tested candidates but none cleared the gate (tested={summary.get('tested')}, keeps=0)"
            )
        if advisory:
            log.warning("FRAMEWORK advisory: %s", advisory)

        _phase_state.append_phase_history_event(
            state,
            reason=reason,
            evidence={
                "event": "framework_agent_phase_done",
                "failure_count": failure_count,
                "empty_count": state.framework_agent_empty_discoveries,
                "retry_limit": DISCOVER_FAILURE_RETRY_LIMIT,
                "batches_discovered": len(state.framework_agent_batches),
                "outcome_class": outcome_class,
                "candidate_outcomes": summary["by_status"],
                "keeps": summary["keeps"],
                "tested": summary["tested"],
                "consecutive_empty_discoveries": consecutive_empty,
                "advisory": advisory,
            },
        )

    async def _enqueue_framework_agent_task(self, candidate: dict[str, Any]) -> None:
        """Enqueue an ``integrate_patch`` task that lands ``candidate``'s diff."""
        state = self.shared_state
        cand_id = self._framework_candidate_key(candidate)
        params = {
            "candidate": candidate,
            "batch_id": candidate.get("batch_id") or "",
            # One action lands every patch; this says where the diff comes from.
            "patch_source": _integrate_vocabulary.PATCH_SOURCE_UPSTREAM_PR,
            "lever_kind": LEVER_UPSTREAM_PR,
            # The authored-outcome bridge, the candidate-processed dedup, the batch max-gain roll-up and phase
            # attribution all key on these two markers.
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand_id,
            "framework_batch_id": str(candidate.get("batch_id") or ""),
            "source_phase": "FRAMEWORK_AGENT",
            "base_tput": resolve_grading_anchor_tput(state),
            # Same decaying bar the explore and integrate_patch dispatch paths inject.
            "keep_threshold_pct": _phase_state.resolve_keep_threshold(state),
            "framework": str(candidate.get("framework") or state.framework or "").strip().lower(),
            # Source patches require the accuracy gate for KEEP.
            "require_accuracy_for_keep": True,
            "accuracy_baseline": float(state.baseline_accuracy or 0.0),
        }
        idem = f"framework:{candidate.get('batch_id', '')}:{cand_id}"
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("integrate_patch")
        try:
            # A framework candidate rebuilds and benchmarks, so it cannot share the GPU.
            if not lanes:
                raise RuntimeError("integrate_patch resolved to no lanes; the task would run without GPU exclusivity.")
            await self.tasks.create_or_return_existing(
                TaskSpec(
                    kind="integrate_patch",
                    params=params,
                    idempotency_key=idem,
                    requires_lanes=lanes,
                    lease_ttl_sec=ttl,
                    dispatch_class="coordinator",
                )
            )
            log.info(
                "FRAMEWORK: enqueued candidate=%s batch=%s",
                cand_id,
                candidate.get("batch_id") or "",
            )
        except (RuntimeError, sqlite3.Error, TypeError, ValueError) as exc:
            log.warning(
                "FRAMEWORK: failed to enqueue candidate=%s: %r",
                cand_id,
                exc,
            )
            # Record enqueue_failed progress row so the candidate is skipped next tick (else the loop spins).
            self._stamp_framework_progress(
                ProgressStamp(
                    candidate_id=cand_id,
                    batch_id=str(candidate.get("batch_id") or ""),
                    status="enqueue_failed",
                    rationale=repr(exc),
                    provenance="pump",
                    extra={"error": repr(exc)},
                )
            )

    def _collect_framework_agent_candidate_priors(self) -> dict[str, Any]:
        """Return compact session-local priors for the Critic gate."""
        raw_progress = self.shared_state.framework_agent_phase_progress or []
        terminal = {
            "kept",
            "kept_inert",
            "reverted",
            "no_patch",
            "enqueue_failed",
            FRAMEWORK_CRITIC_DENIED_STATUS,
        }
        tail = [r for r in raw_progress if isinstance(r, dict) and str(r.get("status") or "") in terminal]
        outcomes: list[dict[str, Any]] = []
        for row in tail[-self._CRITIC_PRIORS_OUTCOME_TAIL :]:
            entry: dict[str, Any] = {
                "candidate_id": str(row.get("candidate_id") or ""),
                "status": str(row.get("status") or ""),
                "gain_pct": row.get("gain_pct"),
            }
            rationale = str(row.get("rationale") or "")[:200]
            if rationale:
                entry["rationale"] = rationale
            outcomes.append(entry)
        return {
            "recent_outcomes": outcomes,
        }

    def _review_already_pending(self, cand_id: str) -> bool:
        """True while ``cand_id`` is still awaiting the Critic's pre-screen verdict."""
        return bool(cand_id) and any(
            first_str(to_mapping(getattr(p, "payload", None)).get("framework_agent_candidate_id")) == cand_id
            for p in self.state.pending_proposals.values()
            if getattr(p, "action_name", "") == "integrate_patch"
        )

    def _charge_review_submission(self, cand_id: str, batch_id: str) -> bool:
        """Count one review submission against ``cand_id``; False once the phase budget's cap is passed."""
        if not cand_id:
            return True
        counts = to_mapping(self.shared_state.framework_agent_review_counts)
        count = int(counts.get(cand_id, 0) or 0) + 1
        counts[cand_id] = count
        self.shared_state.framework_agent_review_counts = counts
        if count <= self._MAX_REPEATED_REVIEW_SUBMISSIONS:
            return True
        log.warning(
            "FRAMEWORK: candidate=%s submitted for review %d times (> cap %d); aborting to protect the phase budget",
            cand_id,
            count,
            self._MAX_REPEATED_REVIEW_SUBMISSIONS,
        )
        self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="repeated_review_abort",
                rationale=f"submitted for review {count} times (> cap {self._MAX_REPEATED_REVIEW_SUBMISSIONS})",
                provenance="pump",
                extra={"review_submissions": count},
            )
        )
        return False

    async def _submit_framework_agent_candidate_for_review(
        self,
        candidate: dict[str, Any],
        *,
        audit: dict[str, Any] | None = None,
        audit_step: str = "",
    ) -> None:
        """Submit a FRAMEWORK candidate as a normal ``proposal`` for async Critic review."""
        cand_id = self._framework_candidate_key(candidate)
        batch_id = first_str(candidate.get("batch_id"))
        if self._review_already_pending(cand_id) or not self._charge_review_submission(cand_id, batch_id):
            return
        propose_payload: dict[str, Any] = {
            "action_name": "integrate_patch",
            "provenance": LEVER_UPSTREAM_PR,
            "predicted_gain_pct": 0.0,
            "candidate": dict(candidate),
            "batch_id": batch_id,
            "framework_agent_candidate_id": cand_id,
            "audit": to_mapping(audit),
            "audit_step": first_str(audit_step),
            "priors": self._collect_framework_agent_candidate_priors(),
        }
        pending = await record_proposal(
            self._coord,
            from_agent="coordinator",
            action_name="integrate_patch",
            predicted_gain_pct=0.0,
            payload=dict(propose_payload),
        )
        # The review-bus message id is a field, not a second identity: the
        # candidate is keyed by its own id throughout the phase.
        _record_step(
            self,
            cand_id,
            ProposalStep(
                step="routed", outcome=str(audit_step or ""), reason=f"submitted_for_review:{pending.proposal_msg_id}"
            ),
        )
        log.info(
            "FRAMEWORK: candidate submitted for Critic review msg_id=%s candidate=%s batch=%s audit_step=%s",
            pending.proposal_msg_id,
            cand_id,
            batch_id,
            audit_step or "<unknown>",
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "framework_agent_candidate_submitted_for_review",
                "proposal_msg_id": pending.proposal_msg_id,
                "candidate_id": cand_id,
                "batch_id": batch_id,
                "audit_step": str(audit_step or ""),
            },
        )
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception(
                "save after framework_agent candidate submit failed for candidate=%s",
                cand_id,
            )

    async def materialize_candidate(
        self,
        pending: "PendingProposal",
    ) -> None:
        """Route a Critic-approved FRAMEWORK candidate to the apply / author tracks."""
        payload = pending.payload or {}
        candidate = dict(payload.get("candidate") or {})
        audit = payload.get("audit") if isinstance(payload.get("audit"), dict) else {}
        audit_step = str(payload.get("audit_step") or "")
        cand_id = str(payload.get("framework_agent_candidate_id") or self._framework_candidate_key(candidate))
        batch_id = str(payload.get("batch_id") or candidate.get("batch_id") or "")
        _record_review_outcome(self, cand_id, materialized=True)
        want_raw = audit_step == "direct_framework"
        want_author = audit_step == "author_via_specialist"
        if audit_step not in ("direct_framework", "author_via_specialist"):
            want_raw = True
            want_author = True
        log.info(
            "FRAMEWORK: critic-approved candidate=%s batch=%s audit_step=%s raw=%s author=%s",
            cand_id,
            batch_id,
            audit_step or "<unknown>",
            want_raw,
            want_author,
        )
        if want_raw:
            # _enqueue_framework_agent_task owns its own enqueue_failed terminal row on failure, so a raw-track
            # candidate always ends up processed.
            await self._enqueue_framework_agent_task(candidate)
        if want_author:
            try:
                await self._enqueue_framework_agent_authoring_specialist(
                    candidate,
                    audit=audit if isinstance(audit, dict) else {},
                )
            except Exception as exc:  # isolation boundary: one approved candidate's authoring dispatch
                log.warning(
                    "FRAMEWORK: authoring specialist dispatch failed: %r",
                    exc,
                    exc_info=True,
                )
                # Author-only route (no raw track to own a terminal row): stamp materialize_failed so an
                # approved-but-undispatchable candidate is not re-selected every tick.
                if not want_raw:
                    self._stamp_framework_progress(
                        ProgressStamp(
                            candidate_id=cand_id,
                            batch_id=batch_id,
                            status="materialize_failed",
                            rationale=repr(exc),
                            provenance="pump",
                            extra={"error": repr(exc)},
                        )
                    )

    def _stamp_framework_progress(self, stamp: ProgressStamp) -> bool:
        """Idempotently stamp a terminal ``framework_agent_phase_progress`` row."""
        cand_id = str(stamp.candidate_id or "")
        if not cand_id:
            return False
        state = self.shared_state
        progress = state.framework_agent_phase_progress
        if not isinstance(progress, list):
            progress = []
            state.framework_agent_phase_progress = progress
        if cand_id in {self._framework_candidate_key(p) for p in progress if isinstance(p, dict)}:
            return False
        batch_id = str(stamp.batch_id or "")
        status = str(stamp.status or "")
        row: dict[str, Any] = {
            "candidate_id": cand_id,
            "batch_id": batch_id,
            "status": status,
            "kept": bool(stamp.kept),
            "rationale": str(stamp.rationale or ""),
            "gain_pct": (float(stamp.gain_pct) if isinstance(stamp.gain_pct, (int, float)) else 0.0),
            "provenance": str(stamp.provenance or ""),
            "ts": datetime.now(timezone.utc).isoformat(),
            "cycle": int(state.macro_cycle or 0),
        }
        # Merge caller-supplied extras (e.g. ``error`` / ``review_submissions``) onto the row too, without clobbering
        # the canonical fields above, so downstream consumers see the same detail the decision.json carries.
        if isinstance(stamp.extra, dict):
            for k, v in stamp.extra.items():
                row.setdefault(str(k), v)
        progress.append(row)
        # The single idempotent terminal-row writer for every path that ends a
        # candidate without a benched result: settling here rather than at each
        # caller keeps a new dead end from leaving its proposal ``pending``.
        _settle(
            self,
            cand_id,
            disposition="attempted" if (stamp.kept or status in {"kept", "reverted"}) else "dropped",
            reason=status,
        )
        try:
            state.save(self.session_dir)
        except OSError:
            log.exception(
                "FRAMEWORK: save after progress stamp failed candidate=%s status=%s",
                cand_id,
                stamp.status,
            )
        log.info(
            "FRAMEWORK: stamped terminal progress candidate=%s batch=%s status=%s",
            cand_id,
            batch_id,
            stamp.status,
        )
        return True

    async def record_critic_denial(
        self,
        pending: "PendingProposal",
        reasoning: str,
    ) -> None:
        """Record a ``critic_denied`` FRAMEWORK row when the async gate rejects a candidate."""
        payload = pending.payload or {}
        cand_id = str(
            payload.get("framework_agent_candidate_id")
            or self._framework_candidate_key(
                payload.get("candidate") if isinstance(payload.get("candidate"), dict) else None
            )
        )
        batch_id = str(payload.get("batch_id") or "")
        self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=batch_id,
                status=FRAMEWORK_CRITIC_DENIED_STATUS,
                rationale=str(reasoning or ""),
                provenance="critic",
            )
        )
        _record_review_outcome(
            self,
            cand_id,
            verdict="reject",
            reason=str(reasoning or ""),
            denied=True,
        )
        log.info(
            "FRAMEWORK: critic rejected candidate=%s batch=%s rationale=%r",
            cand_id,
            batch_id,
            str(reasoning or "")[:200],
        )

    async def maybe_reauthor_from_critic_feedback(
        self,
        pending: "PendingProposal",
        advisory: dict[str, Any] | None,
    ) -> None:
        """Re-author a framework_agent deliverable once, seeding the next authoring round with the Critic's ``required_evidence``."""
        advisory = to_mapping(advisory)
        if first_str(getattr(pending, "action_name", "")) != "integrate_patch":
            return
        # Resolve the candidate identity FIRST so the two dead-end returns below (no required_evidence / reauthor cap)
        # can stamp a terminal progress row — a needs_review verdict that neither re-authors nor materializes would
        # otherwise leave the candidate row-less and re-selected forever.
        payload = to_mapping(getattr(pending, "payload", None))
        authored = to_mapping(payload.get("params")) if not payload.get("framework_agent_candidate_id") else {}
        old_sid = first_str(authored.get("specialist_task_id")).strip()
        candidate, audit = await self._reauthor_target(payload)
        cand_id = self._framework_candidate_key(candidate)
        if not cand_id:
            return
        batch_id = first_str(candidate.get("batch_id"), payload.get("batch_id"))
        required_evidence = [str(x).strip() for x in to_list(advisory.get("required_evidence")) if str(x).strip()]
        if not required_evidence:
            # needs_review with nothing to act on: no re-author is possible, so this is terminal for the candidate.
            self._stamp_framework_progress(
                ProgressStamp(
                    candidate_id=cand_id,
                    batch_id=batch_id,
                    status="needs_review_no_evidence",
                    rationale=first_str(advisory.get("advice_text"))[:500],
                    provenance="critic",
                )
            )
            return
        if await self._candidate_is_materializing(cand_id):
            return
        attempts = to_mapping(self.shared_state.specialist_reauthor_attempts)
        prior = int(attempts.get(cand_id, 0) or 0)
        if prior >= _AUTHORED_LANE_MAX_ATTEMPTS:
            await self._close_out_spent_reauthor_budget(pending, cand_id=cand_id, batch_id=batch_id, attempts=prior)
            return
        attempt = prior + 1
        attempts[cand_id] = attempt
        self.shared_state.specialist_reauthor_attempts = attempts
        critic_feedback = {
            "required_evidence": required_evidence,
            "advice_text": first_str(advisory.get("advice_text")),
            "risks": [str(r).strip() for r in to_list(advisory.get("risks")) if str(r).strip()],
        }
        new_task_id = await self._enqueue_framework_agent_authoring_specialist(
            candidate, audit=audit, reauthor_attempt=attempt, critic_feedback=critic_feedback
        )
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception("save after re-author dispatch failed candidate=%s", cand_id)
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "specialist_reauthor_dispatched",
                "candidate_id": cand_id,
                "attempt": attempt,
                "proposal_msg_id": first_str(getattr(pending, "proposal_msg_id", "")),
                "old_specialist_task_id": old_sid,
                "new_specialist_task_id": new_task_id,
                "verdict": "needs_review",
                "required_evidence": required_evidence[:6],
            },
        )

    async def _reauthor_target(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """The candidate row and the audit a needs_review integrate_patch proposal re-authors from."""
        if payload.get("framework_agent_candidate_id"):
            return to_mapping(payload.get("candidate")), to_mapping(payload.get("audit"))
        params = to_mapping(payload.get("params"))
        if not params.get("framework_agent_authoring"):
            return {}, {}
        spec_params = await self._authoring_specialist_params(first_str(params.get("specialist_task_id")).strip())
        candidate = {
            "candidate_id": first_str(
                params.get("framework_agent_candidate_id"), spec_params.get("framework_agent_candidate_id")
            ),
            "batch_id": first_str(params.get("framework_batch_id"), spec_params.get("framework_batch_id")),
            "title": first_str(spec_params.get("gap_symptom")),
            "framework": first_str(spec_params.get("framework")),
            "gap_canonical_id": first_str(spec_params.get("gap_canonical_id")),
        }
        return candidate, to_mapping(spec_params.get("framework_audit"))

    async def _candidate_is_materializing(self, cand_id: str) -> bool:
        """True while an integrate_patch task for ``cand_id`` is already queued or running."""
        return any(
            first_str(to_mapping(t.params).get("framework_agent_candidate_id")) == cand_id
            for t in (*await self.tasks.queued(), *await self.tasks.running())
            if t.kind == "integrate_patch"
        )

    async def _close_out_spent_reauthor_budget(
        self, pending: "PendingProposal", *, cand_id: str, batch_id: str, attempts: int
    ) -> None:
        """Settle a candidate whose re-author budget is spent: the cap is terminal for it."""
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "reauthor_cap_reached",
                "candidate_id": cand_id,
                "attempts": attempts,
                "proposal_msg_id": first_str(getattr(pending, "proposal_msg_id", "")),
                "verdict": "needs_review",
            },
        )
        self._stamp_framework_progress(
            ProgressStamp(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="reauthor_cap",
                rationale=f"reauthor attempts >= cap ({_AUTHORED_LANE_MAX_ATTEMPTS})",
                provenance="pump",
            )
        )

    async def pump(self) -> None:
        """Best-effort FRAMEWORK pump wrapper run once per Coordinator tick.

        A pump that raises must not take the tick down -- the phase is driven
        again on the next one -- but it is filed like any other coordinator-side
        exception, because a pump that raises on every tick otherwise leaves a
        phase that never dispatched anything closing clean.
        """
        stage = "framework_pump"
        try:
            if self.shared_state.custom_specialist_only:
                await self._coord.specialist_dispatch.ensure_custom_specialist()
                await self._maybe_bench_untested_proposals()
                await self._maybe_finish_custom_only_cycle()
                return
            await self._pump_framework_agent_phase()
            await self._maybe_bench_untested_proposals()
            await self._coord.phase_internal.maybe_enqueue_explore_research_scout()
            await self._coord.phase_internal.maybe_enqueue_config_plateau_static_recon()
            await self._coord.specialist_dispatch.maybe_force_stalled_domain_specialist()
            await self._coord.specialist_dispatch.ensure_custom_specialist()
            self._record_advisory_plateau()
            # Last, as at entry, so a predictor fault is filed here after the phase's own work has run.
            stage = "framework_predictor"
            await self._predictor_pump().step(self.shared_state)
        except Exception as exc:  # isolation boundary: the FRAMEWORK pump within one tick
            log.exception("FRAMEWORK pump failed at %s", stage)
            self._coord.record_exception(stage=stage, exc=exc)

    async def _maybe_bench_untested_proposals(self) -> None:
        """Enqueue explore grids from ``untested_proposal_rows``: the custom specialist's alone, ahead of every gate;
        the rest once no explore is queued or running and the framework candidate track is open."""
        state = self.shared_state
        if (state.phase or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return
        dispatcher = self._coord.dispatcher
        if dispatcher.admission_frozen or dispatcher.dispatch_paused_for_phase_budget():
            return
        explores = [t for t in (*await self.tasks.queued(), *await self.tasks.running()) if t.kind == "explore"]
        custom = self._custom_proposal_rows()
        if custom and not self._has_operator_work(explores):
            if await self._enqueue_bench_grid(custom, operator_priority=True):
                return
        if state.custom_specialist_only or state.framework_agent_phase_done or explores:
            return
        rows = [r for r in state.untested_proposal_rows() if r["domain"] != CUSTOM_PROPOSAL_DOMAIN]
        if rows:
            await self._enqueue_bench_grid(rows[: self._AUTO_EXPLORE_GRID_CAP], operator_priority=False)

    async def _enqueue_bench_grid(self, rows: list[dict[str, Any]], *, operator_priority: bool) -> bool:
        """Enqueue one explore grid benching ``rows``; returns whether a new task was created."""
        from ..actions.executors._proposal_identity import controls_of

        state = self.shared_state
        grid = [
            {
                "name": row["name"],
                "extra_args": row["extra_args"],
                "extra_envs": dict(row["extra_envs"]),
                **controls_of(row),
                "provenance": row["provenance"],
                # The proposing specialist's whole reasoning and checked citations reach the measured Experience.
                "reasoning": row["reason"],
                **({"experience_citations": row["experience_citations"]} if row["experience_citations"] else {}),
            }
            for row in rows
        ]
        # The grid's attempts join their proposal row through this id, as a proposed or delegated grid's do.
        proposal_id = uuid.uuid4().hex
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": "auto_bench_untested_proposals",
            "grid": grid,
            "proposal_msg_id": proposal_id,
        }
        if operator_priority:
            params["operator_priority"] = True
        if state.baseline_config_path:
            params["config_path"] = state.baseline_config_path
        inject_stack_base_params(params, state, anchor=True)
        benchmark_script = state.accepted_baseline_script()
        if benchmark_script:
            params["benchmark_script"] = benchmark_script
        lanes, ttl = self._coord.dispatcher.registry_lanes_ttl("explore")
        etask, was_existing = await self.tasks.create_or_return_existing(
            TaskSpec(
                kind="explore",
                params=params,
                idempotency_key=self._bench_key(rows, operator_priority=operator_priority),
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
                dispatch_class="coordinator",
            )
        )
        if was_existing:
            return False
        log.info("auto_bench: enqueued explore task_id=%s (variants=%d)", etask.task_id, len(grid))
        from ..loop.proposals import record_config_proposal

        record_config_proposal(
            self._coord, proposal_id, "explore", {"params": params, **kb_exposure_of(rows)}, outcome="auto_bench"
        )
        return True

    async def _submit_integrate_review(
        self,
        sid: str,
        spec_params: dict[str, Any],
        done_payload: dict[str, Any],
        deliverable: dict[str, Any],
        *,
        observation: dict[str, Any],
    ) -> PendingProposal | None:
        """Propose a specialist deliverable to the Critic as a synthetic integrate_patch; idempotent per specialist.

        Args:
            sid: The specialist task id the review is keyed on.
            spec_params: The specialist's dispatch params, whose provenance and
                enablement context travel to integrate_patch.
            done_payload: The specialist's ``specialist_done`` payload.
            deliverable: The integrate_patch params that say what to apply.
            observation: The bus observation recorded with the proposal.

        Returns:
            The recorded proposal, or ``None`` when the Critic already ruled on
            this specialist or its review is still pending.
        """
        if self.shared_state.get_specialist_patch_verdict(sid) or integrate_review_in_flight(
            self.state.pending_proposals, sid
        ):
            return None
        params = _integrate_review_params(sid, spec_params, done_payload, deliverable)
        pending = await record_proposal(
            self._coord,
            from_agent="coordinator",
            action_name="integrate_patch",
            predicted_gain_pct=0.0,
            payload={
                "action_name": "integrate_patch",
                "provenance": "specialist",
                "predicted_gain_pct": 0.0,
                "params": params,
            },
        )
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                **observation,
                "specialist_task_id": sid,
                "proposal_msg_id": pending.proposal_msg_id,
                "patch_name": params["patch_name"],
            },
        )
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception("save after integrate-review autosubmit failed for task=%s", sid)
        return pending
