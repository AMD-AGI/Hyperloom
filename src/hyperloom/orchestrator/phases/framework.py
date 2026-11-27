# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT phase handler: authoring specialist dispatch, enablement repair, deliverable routing, and Critic-review submission/reauthor."""

from __future__ import annotations
import logging as _logging
import sqlite3
from datetime import datetime, timezone
from collections.abc import Collection, Mapping
from typing import TYPE_CHECKING, Any

from hyperloom_common.coerce import first_str, to_list, to_mapping
from hyperloom.inference_optimizer.session.optimization_journal import Verdict

from . import machine_state as _phase_state
from ..state.task_registry import Task, TaskSpec
from ..state.shared_state import resolve_grading_anchor_tput

if TYPE_CHECKING:
    from ..predictor.payload import PhaseBudget
    from ..predictor.view import SessionView
    from .machine import Transition
from ..state.pending_proposal import PendingProposal
from ..actions.executors.integrate import _integrate_vocabulary
from hyperloom.orchestrator.lever import LEVER_UPSTREAM_PR
from .proposal_bench import ProposalBench
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
from .framework_reauthor import _ReauthorMixin
from .framework_local_explore import _LocalExploreMixin
from .framework_proposals import _FrameworkProposalsMixin

log = _logging.getLogger(__name__)


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


def predictor_budget(state: SessionView) -> PhaseBudget:
    """The FRAMEWORK clock a predictor request is conditioned on, read when the request goes out."""
    from ..predictor.payload import PhaseBudget

    remaining = _phase_state.phase_budget_remaining_seconds(state)
    return PhaseBudget(
        headroom_sec=None if remaining is None else float(remaining),
        elapsed_sec=_phase_state.phase_elapsed_seconds(state),
        keep_threshold_pct=_phase_state.resolve_keep_threshold(state),
    )


def _coordinator_specialist_spec(params: dict[str, Any], idem: str, *, lanes: list[str] | None, ttl: int) -> TaskSpec:
    """The specialist task a coordinator-internal dispatch opens with ``params``."""
    return TaskSpec(
        kind="specialist",
        params=params,
        idempotency_key=idem,
        requires_lanes=lanes,
        lease_ttl_sec=ttl,
        dispatch_class="coordinator",
    )


class FrameworkPhase(
    _ReauthorMixin,
    _LocalExploreMixin,
    _FrameworkProposalsMixin,
    _AuthoredOutcomeMixin,
    _CandidateDiscoveryMixin,
    ProposalBench,
):
    """The FRAMEWORK_AGENT phase: upstream candidates, authored patches, deliverable routing, and the enablement hand-off."""

    # Set for the span of one FRAMEWORK entry; ``None`` outside it and when the recorder is unbound.
    _framework_timeline_recorder: Any = None
    # The PrimaTune predictor's in-flight request, created on first use (``predictor.pump.PredictorPump``).
    _predictor: Any = None

    def on_specialist_settled(self, task: "Task", done_payload: dict[str, Any], *, run_error: str) -> None:
        """Stamp an empty authoring round's terminal row and harvest a discovery round's candidates."""
        # An authoring specialist that wrote no patch never spawns an integrate_patch; stamp its terminal row here.
        self._record_framework_agent_authoring_empty_outcome(task=task, done_payload=done_payload, run_error=run_error)
        self._ingest_candidate_discovery(task=task, done_payload=done_payload, run_error=run_error)

    async def on_integrate_patch_settled(self, task: "Task", result: Any, verdict: Verdict) -> None:
        """Record an authored patch's settled outcome, then re-arm or drain the authored lane."""
        if (task.params or {}).get("framework_agent_authoring"):
            self._record_framework_agent_authored_outcome(task=task, result=result, adopted=verdict is Verdict.ADOPTED)
        await self._maybe_rearm_authored_lane(result.result, to_mapping(task.params))
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
                "no_keep_streak_threshold": _phase_state.DEFAULT_FRAMEWORK_PLATEAU_NO_KEEP_STREAK,
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
        if state.framework_local_explore_enabled and not source_dry and state.research_lane_capacity > 0:
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
            _coordinator_specialist_spec(params, idem, lanes=lanes, ttl=ttl)
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
        want_raw = want_raw and not self._skip_direct_route(cand_id, batch_id, authored=want_author)
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
            except (sqlite3.Error, OSError) as exc:
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

    def _skip_direct_route(self, cand_id: str, batch_id: str, *, authored: bool) -> bool:
        """Leave the pull request's own diff unqueued in a ``--no-source-changes`` session; True when it was left.

        That diff is a source change ``integrate_patch`` could only skip. The author track may still return
        configuration, so a candidate that has one goes on there; one without it ends here as ``skipped``.
        """
        if not self.shared_state.no_source_changes:
            return False
        if not authored:
            self._stamp_framework_progress(
                ProgressStamp(
                    candidate_id=cand_id,
                    batch_id=batch_id,
                    status="skipped",
                    rationale="--no-source-changes: the pull request's own diff is a source change; not queued",
                    provenance="pump",
                )
            )
        return True

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
            await self._coord.specialist_dispatch.ensure_custom_specialist()
            self._record_advisory_plateau()
            # Last, as at entry, so a predictor fault is filed here after the phase's own work has run.
            stage = "framework_predictor"
            await self._predictor_pump().step(self.shared_state)
        except Exception as exc:  # isolation boundary: the FRAMEWORK pump within one tick
            log.exception("FRAMEWORK pump failed at %s", stage)
            self._coord.record_exception(stage=stage, exc=exc)
