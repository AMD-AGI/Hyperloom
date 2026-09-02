# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FRAMEWORK_AGENT phase handler: candidate discovery/ranking/audit, authoring
specialist dispatch, enablement repair, and Critic-review submission/reauthor."""

from __future__ import annotations
import logging as _logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import machine_state as _phase_state
from ..bus.message_bus import Message
from ..state.shared_state import resolve_grading_anchor_tput

if TYPE_CHECKING:
    from ..state.task_registry import Task
from ..loop.coordinator import (
    PendingProposal,
    _AUTHORED_LANE_MAX_ATTEMPTS,
    _framework_config_levers_from_done,
)
from ..actions.executors.integrate_patch import PATCH_SOURCE_UPSTREAM_PR
from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    LEVER_SOURCE_PATCH,
    LEVER_UPSTREAM_PR,
)
from ..collaborator import CoordinatorCollaborator

log = _logging.getLogger(__name__)

# Specialist attempts a local-exploration candidate gets before the phase moves
# on. A candidate that keeps failing to author is not worth the wall clock, but
# one interrupted run — a process restart reclaims an in-flight specialist as
# failed — must not retire it.
_LOCAL_EXPLORE_MAX_ATTEMPTS: int = 3


#: Progress-row status for a candidate the Critic rejected. The gate writes it
#: and the working-memory and priors readers select on it, so the ledger is the
#: only record of a denial and the three sites agree by construction.
FRAMEWORK_CRITIC_DENIED_STATUS: str = "critic_denied"


class FrameworkPhase(CoordinatorCollaborator):
    """The source arm of the OPTIMIZE phase: upstream candidates, authored
    patches, and the enablement hand-off. Not a phase of its own -- it shares
    FRAMEWORK_AGENT with the configuration arm.
    """

    # Marker for the candidate-free local-exploration arm (a synthetic
    # "candidate" whose id is ``local_explore:<n>``): the ranker may pick it,
    # and it routes to a write-capable authoring specialist instead of a PR.
    _LOCAL_EXPLORE_KIND = "local_explore"

    async def _on_enter_framework(self, *, from_phase: str) -> None:
        """FRAMEWORK entry hook: trigger the per-batch pump once on entry (best-effort; later batches driven from the main tick).

        Args:
            from_phase: The phase being left, used only for logging.
        """
        log.info(
            "OPTIMIZE entry (from=%s): pumping initial batch",
            from_phase or "<unknown>",
        )
        # A reopened macro-cycle re-measures before either arm spends anything.
        await self._on_cycle_start_reprofile(from_phase=from_phase)
        # Ordered after the reprofile so the predictor sees fresh evidence, and
        # before the arms so its task takes the serving lease first. Going first
        # needs no suppression: an LLM-proposed explore only arrives on the next
        # tick's reactor pass, while this runs inside the entry hook.
        await self._pump_predictor(caller="entry")
        try:
            await self._pump_framework_agent_phase()
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning("FRAMEWORK entry pump failed: %r", exc)

    async def _pump_framework_agent_phase(self) -> None:
        """Drive the FRAMEWORK_AGENT phase: enqueue the next candidate. Idempotent; a discover failure flips framework_agent_phase_done so the phase advances rather than wedging."""
        state = self.shared_state
        if (state.phase or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return
        if bool(getattr(state, "framework_agent_phase_done", False)):
            return
        # Skip if a framework task is already queued or running.
        queued = await self.tasks.queued()
        running = await self.tasks.running()
        for t in (*queued, *running):
            # A candidate landing as ``integrate_patch`` with a candidate id.
            if getattr(t, "kind", "") == "integrate_patch" and (getattr(t, "params", None) or {}).get(
                "framework_agent_candidate_id"
            ):
                return
        # Serialize one candidate at a time: skip while a candidate proposal
        # awaits its (durable) Critic verdict, resolved on a later tick.
        try:
            if any(
                getattr(p, "action_name", "") == "integrate_patch"
                and not getattr(p, "decided", False)
                and (getattr(p, "payload", None) or {}).get("framework_agent_candidate_id")
                for p in self.state.pending_proposals.values()
            ):
                return
        except Exception:  # noqa: BLE001 — defensive
            pass
        # An authoring specialist (or its downstream integrate_patch) for the
        # current candidate may still be running; wait only on a live TASK
        # (queued/running), NOT on a pending Critic proposal. The
        # proposal-pending case is covered by the ``next_candidate is None``
        # inflight wait below.
        if getattr(state, "framework_agent_authoring_enabled", False):
            _q = await self.tasks.queued()
            _r = await self.tasks.running()
            if any(
                getattr(t, "kind", "") in ("specialist", "integrate_patch")
                and bool((getattr(t, "params", None) or {}).get("framework_agent_authoring"))
                for t in (*_q, *_r)
            ):
                return
            # Proposal-window guard: the task check above misses the interval
            # between a specialist completing and its integrate_patch becoming a
            # live TASK (the deliverable exists only as a pending Critic
            # proposal). ``_framework_agent_authoring_inflight`` covers pending
            # integrate_patch proposals, serializing one candidate's
            # author->integrate->KEEP/REVERT lifecycle before the next.
            if await self._framework_agent_authoring_inflight():
                return
        # Take the next un-dispatched candidate. Ordering and the
        # already-present / not-applicable / worth-a-bench judgement are the
        # discovery specialist's deliverable, so the pump takes them in the
        # order it was given rather than re-ranking or re-auditing here.
        next_candidate = self._select_next_framework_agent_candidate()
        if next_candidate is None:
            # Hold the phase open while authored patches are still benched or
            # reviewed; only when a batch was discovered (an LLM-proposed
            # integrate_patch must not keep FRAMEWORK open).
            discovered_batch = bool(getattr(state, "framework_agent_batches", None) or [])
            if (
                discovered_batch
                and getattr(state, "framework_agent_authoring_enabled", False)
                and await self._framework_agent_authoring_inflight()
            ):
                return
            # Minimum supply: with the pool empty and no discovery in flight,
            # ask for one. Orchestration may dispatch the same specialist
            # itself at any time; this only keeps the lane from idling when it
            # does not.
            if await self._maybe_enqueue_candidate_discovery(reason="candidate_pool_empty"):
                state.save(self.session_dir)
                return
            if await self._maybe_dispatch_local_explore(reason="no_new_candidates"):
                state.save(self.session_dir)
                return
            self._record_framework_agent_phase_done(
                reason="no_candidates_and_discovery_exhausted",
                failure_count=int(getattr(state, "framework_agent_discover_failures", 0) or 0),
            )
            state.framework_agent_phase_done = True
            state.save(self.session_dir)
            return
        # Local-exploration arm: a candidate-free authoring specialist has no
        # upstream diff, so it dispatches directly.
        if str(next_candidate.get("kind") or "") == self._LOCAL_EXPLORE_KIND:
            await self._enqueue_framework_agent_local_explore_specialist(next_candidate)
            state.save(self.session_dir)
            return
        # Submit the candidate as a proposal; the async Critic verdict drives the
        # apply/author enqueue or the critic_denied row on a later tick.
        await self._submit_framework_agent_candidate_for_review(
            next_candidate,
            audit=dict(next_candidate.get("audit") or {}),
            audit_step=str(next_candidate.get("route") or "author_via_specialist"),
        )

    async def _framework_agent_authoring_inflight(self) -> bool:
        """True while a FRAMEWORK-authored patch for an unprocessed candidate is still in flight.

        Counts only items with the ``framework_agent_authoring`` provenance marker whose candidate
        is still unprocessed (no terminal progress row yet). A KERNEL-phase specialist/integrate or
        an orphaned stale proposal therefore never pins the pump.

        Returns:
            ``True`` if a framework-owned specialist/integrate_patch task is queued or running,
            or a framework-owned undecided proposal targets an unprocessed candidate; else ``False``.
        """
        unprocessed_ids = {self._framework_candidate_key(c) for c in self._unprocessed_framework_agent_candidates()}
        # The local-exploration arm's synthetic candidate id never appears in a
        # PR batch, so it is "in flight" while it lacks a terminal progress row.
        processed_ids = self._framework_processed_candidate_keys()

        def _cand_pins_pump(cand_id: str) -> bool:
            """True when an authoring cand_id keeps the pump serialized."""
            if not cand_id or cand_id in unprocessed_ids:
                return True
            return cand_id.startswith("local_explore:") and cand_id not in processed_ids

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
        # An authored patch awaiting Critic review (or a candidate awaiting its
        # pre-screen verdict) keeps the phase open, but only while the proposal
        # targets a still-unprocessed candidate.
        try:
            for p in self.state.pending_proposals.values():
                if getattr(p, "decided", False):
                    continue
                if getattr(p, "action_name", "") != "integrate_patch":
                    continue
                payload = getattr(p, "payload", None) or {}
                # Both the candidate pre-screen and the authored patch are
                # ``integrate_patch`` proposals now, so the candidate marker --
                # not the action name -- says which candidate is pinned. The
                # pre-screen carries it at the top level, the authored patch
                # under ``params``.
                iparams = payload.get("params") or {}
                cand_id = str(
                    payload.get("framework_agent_candidate_id") or iparams.get("framework_agent_candidate_id") or ""
                )
                if not cand_id and not iparams.get("framework_agent_authoring"):
                    continue
                if _cand_pins_pump(cand_id):
                    return True
        except Exception:  # noqa: BLE001 — defensive
            pass
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

    async def _enqueue_framework_agent_authoring_specialist(
        self,
        candidate: dict[str, Any],
        audit: dict[str, Any] | None = None,
        *,
        reauthor_attempt: int = 0,
        critic_feedback: dict[str, Any] | None = None,
    ) -> str:
        """Dispatch a write-capable specialist seeded with ``candidate`` (flows through autosubmit → Critic → integrate_patch → bench → KEEP/REVERT).

        Args:
            candidate: The discovered FRAMEWORK candidate (PR url, title,
                diff url, batch/candidate ids) used to seed the specialist's
                authoring task and provenance markers.
            audit: Optional candidate-review verdict; injected into the seed
                so the specialist authors against the live source instead of
                re-discovering why the candidate was worth trying.
            reauthor_attempt: Re-author round; ``> 0`` adds a ``reauthor:{n}``
                idempotency-key suffix so the round gets a fresh task.
            critic_feedback: Prior-round Critic advisory (``required_evidence`` /
                ``advice_text`` / ``risks``) appended to the authoring seed.

        Returns:
            The dispatched specialist ``task_id`` (empty when a livelock-break
            short-circuits the re-dispatch).
        """
        state = self.shared_state
        cand_id = self._framework_candidate_key(candidate)
        batch_id = str(candidate.get("batch_id") or "")
        gap_cid = str(candidate.get("gap_canonical_id") or "").strip() or f"gap.framework.{cand_id}"
        title = str(candidate.get("title") or "").strip()
        pr_url = str(candidate.get("pr_url") or "").strip()
        diff_url = str(candidate.get("diff_url") or "").strip()
        notes_lines: list[str] = []
        notes_lines.extend(self._framework_agent_audit_seed_lines(audit))
        if critic_feedback:
            req_ev = [str(x).strip() for x in (critic_feedback.get("required_evidence") or []) if str(x).strip()]
            fb_lines = [
                "",
                "PRIOR CRITIC FEEDBACK (re-author round — supply the evidence below this round):",
            ]
            fb_lines.extend(f"  • required evidence: {ev}" for ev in req_ev[:10])
            advice = str(critic_feedback.get("advice_text") or "").strip()
            if advice:
                fb_lines.append(f"- advice: {advice}")
            risks = [str(r).strip() for r in (critic_feedback.get("risks") or []) if str(r).strip()]
            if risks:
                fb_lines.append("- risks: " + "; ".join(risks[:6]))
            notes_lines.extend(fb_lines)
        notes = "\n".join(notes_lines).strip()
        params: dict[str, Any] = {
            "domain": self._authoring_specialist_domain(),
            "gap_canonical_id": gap_cid,
            "gap_symptom": (title or f"Author a framework source patch inspired by {pr_url or cand_id}"),
            "gap_layer": "framework",
            "framework": str(candidate.get("framework") or getattr(state, "framework", "") or "").strip().lower(),
            "task_kind": "framework_authoring",
            "source_phase": "FRAMEWORK_AGENT",
            "pr_lead": {"title": title, "url": pr_url, "diff_url": diff_url},
            "lever_kind": LEVER_UPSTREAM_PR,
            # Provenance markers for the dispatcher-side authored-patch bridge.
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand_id,
            "framework_batch_id": batch_id,
            "reauthor_attempt": int(reauthor_attempt),
            "framework_audit": (audit if isinstance(audit, dict) else {}),
            "source": "coordinator_internal",
            "notes": notes,
            # Whole-machine GPU request. Empty on multi-node / no-GPU hosts.
            **self._framework_gpu_params(),
        }
        try:
            await self._warm_specialist_params(params)
        except Exception:  # noqa: BLE001 — best-effort warmup
            log.debug(
                "FRAMEWORK authoring: warm specialist params failed",
                exc_info=True,
            )
        idem = f"framework_agent_authoring:{batch_id}:{cand_id}"
        if reauthor_attempt > 0:
            idem = f"{idem}:reauthor:{int(reauthor_attempt)}"
        # This internal dispatch bypasses intent_router (adds gpu_research_lane + budget TTL).
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        spec_task, _spec_existing = await self.tasks.create_or_return_existing(
            kind="specialist",
            params=params,
            idempotency_key=idem,
            requires_lanes=lanes,
            side_effects=["writes_results", "writes_patches"],
            lease_ttl_sec=ttl,
        )
        from ..state.task_registry import TERMINAL_STATES as _TERMINAL_STATES

        if _spec_existing and str(getattr(spec_task, "state", "") or "") in _TERMINAL_STATES:
            already_rows = self._framework_processed_candidate_keys()
            authoring_inflight = await self._framework_agent_authoring_inflight()
            if cand_id and cand_id not in already_rows and not authoring_inflight:
                try:
                    recovered = await self._recover_framework_agent_authoring_outcome(
                        specialist_task=spec_task,
                    )
                except Exception:  # noqa: BLE001 — never wedge the pump
                    log.exception(
                        "FRAMEWORK: terminal authoring outcome recovery failed candidate=%s",
                        cand_id,
                    )
                    recovered = False
                if not recovered:
                    log.warning(
                        "FRAMEWORK: terminal authoring outcome unavailable candidate=%s state=%s",
                        cand_id,
                        getattr(spec_task, "state", ""),
                    )
                    # Stamp a terminal row so an unrecoverable outcome cannot make
                    # the pump re-select the same finished specialist forever.
                    self._stamp_framework_progress(
                        candidate_id=cand_id,
                        batch_id=batch_id,
                        status="recovery_failed",
                        rationale="authoring outcome unrecoverable from persisted results",
                        provenance="pump",
                    )
                return ""
        # Map specialist task -> candidate so the authored-outcome bridge can
        # resolve the PR-URL candidate id from the downstream integrate_patch.
        spec_tid = str(getattr(spec_task, "task_id", "") or "")
        try:
            if spec_tid and cand_id:
                if not isinstance(getattr(state, "framework_agent_specialist_candidate_map", None), dict):
                    state.framework_agent_specialist_candidate_map = {}
                state.framework_agent_specialist_candidate_map[spec_tid] = cand_id
                state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — best-effort provenance
            log.debug(
                "FRAMEWORK authoring: specialist->candidate map write failed",
                exc_info=True,
            )
        log.info(
            "FRAMEWORK: dispatched authoring specialist candidate=%s batch=%s gap=%s",
            cand_id,
            batch_id,
            gap_cid,
        )
        return spec_tid

    def _maybe_rearm_authored_lane(self, res: dict[str, Any] | None) -> None:
        """Unified rearm dispatcher for all authored lanes.

        Routes to the lane-specific handler:

        * ``enablement`` lane → :meth:`_maybe_rearm_enablement` (unchanged
          semantics; ``advanced`` / ``kept`` / stall logic preserved).
        * ``perf_framework`` / ``perf_explore`` lanes with ``apply_failed``
          status → increment per-candidate apply-fail retry counter; below cap
          clear the in-flight guard so :meth:`_enqueue_author_specialist` can
          be called from the dispatcher; at/above cap stamp a terminal
          progress row.

        All other statuses for perf lanes are not handled here (they go through
        the existing writeback / progress-stamp paths).

        Args:
            res: The ``integrate_patch`` or ``framework_agent`` result dict.
        """
        if not isinstance(res, dict):
            return
        lane = str(res.get("lane") or "")
        status = str(res.get("status") or "")

        if lane == "enablement" or res.get("enablement"):
            self._maybe_rearm_enablement(res)
            return

        if status != "apply_failed":
            # Non-apply-failed perf-lane results go through the writeback path.
            return
        if lane not in ("perf_framework", "perf_explore"):
            return

        # Determine the candidate key for tracking retry attempts.
        candidate = res.get("candidate")
        if not isinstance(candidate, dict):
            candidate = {}
        cand_id = self._framework_candidate_key(candidate)
        if not cand_id:
            cand_id = str(res.get("specialist_task_id") or "").strip()
        if not cand_id:
            return

        batch_id = str(
            (candidate.get("batch_id") if isinstance(candidate, dict) else None) or res.get("batch_id") or ""
        )

        state = self.shared_state

        existing = state.apply_fail_reauthor_attempts
        apply_fail_attempts: dict[str, int] = existing if isinstance(existing, dict) else {}
        prior = int(apply_fail_attempts.get(cand_id, 0) or 0)
        attempt = prior + 1
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
            # Cap reached: stamp a terminal progress row.
            self._stamp_framework_progress(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="apply_fail_cap",
                kept=False,
                rationale=f"apply_failed {attempt} times (cap={_AUTHORED_LANE_MAX_ATTEMPTS})",
                provenance="apply_fail_retry",
            )
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.debug("authored_lane: save after cap stamp failed", exc_info=True)
            return

        # Under cap: store the retry context for the dispatcher to pick up.
        vetting_drops_raw = res.get("patches_dropped_by_grounding")
        retry_ctx: dict[str, Any] = {
            "cand_id": cand_id,
            "batch_id": batch_id,
            "lane": lane,
            "attempt": attempt,
            "retry_feedback": res.get("retry_feedback") or [],
            "candidate": candidate,
            "specialist_task_id": str(res.get("specialist_task_id") or ""),
        }
        if isinstance(vetting_drops_raw, list) and vetting_drops_raw:
            retry_ctx["vetting_drops"] = [str(d) for d in vetting_drops_raw[:8]]
        pending = state.apply_fail_retry_pending or []
        if not isinstance(pending, list):
            pending = []
        pending.append(retry_ctx)
        state.apply_fail_retry_pending = pending
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.debug("authored_lane: save after retry-pending failed", exc_info=True)

    async def _enqueue_author_specialist(
        self,
        *,
        lane: str,
        candidate: "dict[str, Any] | None" = None,
        batch_id: str = "",
        specialist_task_id: str = "",
        attempt: int = 1,
        retry_feedback: "list[dict[str, Any]] | None" = None,
        critic_feedback: "dict[str, Any] | None" = None,
        vetting_drops: "list[str] | None" = None,
    ) -> str:
        """Dispatch a fresh authoring specialist for an apply-failure retry.

        Handles the ``perf_framework`` and ``perf_explore`` lanes only
        (enablement uses :meth:`_maybe_enqueue_enablement_specialist`).
        Injects structured apply-failure feedback into the specialist mandate
        and uses a cycle-scoped ``:retry:{n}`` idempotency suffix to get a
        fresh task that reuses the existing worktree via the
        idempotency-based worktree lookup.

        Args:
            lane: ``"perf_framework"`` or ``"perf_explore"``.
            candidate: The candidate dict (for perf_framework lane).
            batch_id: The failing round's batch, used when the candidate row
                does not carry one; it keys the retry's idempotency and its
                progress rows.
            specialist_task_id: The original specialist task that produced the
                failing patch (for worktree reuse + provenance).
            attempt: Retry attempt number (1-based; appended to idempotency key).
            retry_feedback: List of :class:`~._apply_feedback.ApplyFeedback`
                dicts from the failed apply, injected into the specialist mandate.
            critic_feedback: Optional prior Critic advisory (for reauthor retries
                that also had a Critic ``needs_review`` verdict).
            vetting_drops: Patch targets the safety gate dropped last round,
                named in the mandate so the retry does not re-author them.

        Returns:
            The dispatched specialist ``task_id`` (empty on failure / skip).
        """
        if lane not in ("perf_framework", "perf_explore"):
            log.warning("_enqueue_author_specialist: unsupported lane=%s — skipping", lane)
            return ""

        candidate = dict(candidate or {})
        if batch_id and not candidate.get("batch_id"):
            candidate["batch_id"] = batch_id
        retry_feedback = retry_feedback or []
        state = self.shared_state

        feedback_lines: list[str] = []
        if vetting_drops:
            feedback_lines.append("")
            feedback_lines.append(
                "PATCH GROUNDING FAILURE: the prior round's patches were dropped by "
                "the safety gate before reaching integration. The worktree contains "
                "the correct framework tree — edit files there and the diff is "
                "harvested automatically. Do not switch to the artifacts_written "
                "channel to avoid the gate."
            )
            feedback_lines.append("Dropped targets: " + "; ".join(vetting_drops[:4]))
            feedback_lines.append("")
        if retry_feedback:
            feedback_lines.append("")
            feedback_lines.append(
                "APPLY FAILURE FEEDBACK (previous patch failed to apply; "
                "study the errors below and produce a corrected patch):"
            )
            for fb_dict in retry_feedback[:5]:  # cap at 5 entries for prompt brevity
                from ..actions.executors._apply_feedback import ApplyFeedback

                fb = ApplyFeedback.from_dict(fb_dict) if isinstance(fb_dict, dict) else None
                if fb is not None:
                    feedback_lines.append("")
                    feedback_lines.append(fb.format_for_mandate())
            feedback_lines.append("")

        if lane == "perf_framework":
            # Re-dispatch a framework_agent authoring specialist with apply
            # failure context injected via a critic_feedback-style note.
            cand_id = self._framework_candidate_key(candidate)
            if not cand_id:
                log.warning("_enqueue_author_specialist: perf_framework missing cand_id")
                return ""
            # Look up original audit from the specialist task params if available.
            audit: dict[str, Any] = {}
            if specialist_task_id:
                try:
                    spec_task = await self.tasks.get(specialist_task_id)
                    spec_params = dict(getattr(spec_task, "params", None) or {})
                    raw_audit = spec_params.get("framework_audit")
                    if isinstance(raw_audit, dict):
                        audit = raw_audit
                except Exception:  # noqa: BLE001
                    pass
            # Merge apply feedback + critic feedback into a single note block.
            merged_feedback = dict(critic_feedback or {})
            if feedback_lines:
                existing_advice = str(merged_feedback.get("advice_text") or "")
                apply_advice = "\n".join(feedback_lines)
                merged_feedback["advice_text"] = apply_advice + ("\n\n" + existing_advice if existing_advice else "")
            try:
                new_task_id = await self._enqueue_framework_agent_authoring_specialist(
                    candidate,
                    audit=audit,
                    reauthor_attempt=attempt,
                    critic_feedback=merged_feedback if merged_feedback else None,
                )
            except Exception:  # noqa: BLE001
                log.exception(
                    "_enqueue_author_specialist: perf_framework dispatch failed cand=%s attempt=%d",
                    cand_id,
                    attempt,
                )
                return ""
            log.info(
                "AUTHORED_LANE: dispatched perf_framework retry specialist cand=%s attempt=%d task=%s",
                cand_id,
                attempt,
                new_task_id,
            )
            return new_task_id

        # perf_explore lane: reauthor from original specialist worktree.
        # Build a minimal candidate proxy from the specialist task params.
        gap_cid = ""
        gap_symptom = ""
        framework_name = str(getattr(state, "framework", "") or "").strip().lower()
        if specialist_task_id:
            try:
                spec_task = await self.tasks.get(specialist_task_id)
                spec_params = dict(getattr(spec_task, "params", None) or {})
                gap_cid = str(spec_params.get("gap_canonical_id") or "").strip()
                gap_symptom = str(spec_params.get("gap_symptom") or "").strip()
                framework_name = str(spec_params.get("framework") or framework_name).strip().lower()
            except Exception:  # noqa: BLE001
                pass
        if not gap_cid:
            gap_cid = f"gap.explore.retry.{specialist_task_id or 'unknown'}"
        notes_lines: list[str] = list(feedback_lines)
        if critic_feedback:
            req_ev = [str(x).strip() for x in (critic_feedback.get("required_evidence") or []) if str(x).strip()]
            if req_ev:
                notes_lines.append("")
                notes_lines.append("PRIOR CRITIC FEEDBACK (also address this):")
                notes_lines.extend(f"  • {ev}" for ev in req_ev[:10])
            advice = str(critic_feedback.get("advice_text") or "").strip()
            if advice:
                notes_lines.append(f"- advice: {advice}")
        notes = "\n".join(notes_lines).strip()
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
            "apply_retry_attempt": attempt,
            **self._framework_gpu_params(),
        }
        try:
            await self._warm_specialist_params(params)
        except Exception:  # noqa: BLE001
            pass
        # Gap id and attempt both repeat across cycles.
        idem = f"perf_explore_authoring:{gap_cid}:retry:{attempt}{self._cycle_idem_suffix()}"
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        try:
            spec_task, _ = await self.tasks.create_or_return_existing(
                kind="specialist",
                params=params,
                idempotency_key=idem,
                requires_lanes=lanes,
                side_effects=["writes_results", "writes_patches"],
                lease_ttl_sec=ttl,
            )
        except Exception:  # noqa: BLE001
            log.exception(
                "_enqueue_author_specialist: perf_explore dispatch failed gap=%s attempt=%d",
                gap_cid,
                attempt,
            )
            return ""
        new_tid = str(getattr(spec_task, "task_id", "") or "")
        log.info(
            "AUTHORED_LANE: dispatched perf_explore retry specialist gap=%s attempt=%d task=%s",
            gap_cid,
            attempt,
            new_tid,
        )
        return new_tid

    async def _drain_apply_fail_retry_pending(self) -> None:
        """Dispatch authoring specialists for any queued apply-failure retries.

        Called from the dispatcher after every ``integrate_patch`` completion.
        Drains :attr:`SharedState.apply_fail_retry_pending` (a list of retry
        context dicts populated by :meth:`_maybe_rearm_authored_lane`) by
        calling :meth:`_enqueue_author_specialist` for each entry, then
        clearing the list.

        Best-effort: errors are logged and the list is cleared regardless so
        stale contexts cannot accumulate.
        """
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
            try:
                await self._enqueue_author_specialist(
                    lane=lane,
                    candidate=candidate,
                    batch_id=batch_id,
                    specialist_task_id=specialist_task_id,
                    attempt=attempt,
                    retry_feedback=retry_feedback,
                    vetting_drops=vetting_drops or None,
                )
            except Exception:  # noqa: BLE001 — never wedge the dispatcher
                log.exception(
                    "_drain_apply_fail_retry_pending: dispatch failed lane=%s attempt=%d",
                    lane,
                    attempt,
                )
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.debug("drain_apply_fail: save failed", exc_info=True)

    @staticmethod
    def _framework_candidate_key(row: dict[str, Any] | None) -> str:
        """Canonical FRAMEWORK candidate dedup/progress key (see ``candidate_key``).

        Thin wrapper over
        :func:`~hyperloom.orchestrator.framework.artifacts.candidate_key` so every candidate
        selection / dedup / progress-row / idempotency site derives the key
        from one place (``candidate_id or pr_url or ref``). Prevents the
        asymmetry where a candidate carrying only a ``pr_url`` failed to dedup
        against its own progress row.

        Args:
            row: A candidate dict or ``framework_agent_phase_progress`` row.

        Returns:
            The candidate key, or ``""`` when no identity field is set.
        """
        from ..framework.artifacts import candidate_key

        return candidate_key(row)

    def _framework_processed_candidate_keys(self) -> set[str]:
        """Set of candidate keys that already carry a terminal progress row.

        A candidate is "processed" once any ``framework_agent_phase_progress``
        row is keyed on it; such a candidate must never be re-selected. Progress
        rows store the key in their ``candidate_id`` field, so this reuses
        :meth:`_framework_candidate_key`.

        Returns:
            The set of processed candidate keys (possibly empty).
        """
        return {
            self._framework_candidate_key(p)
            for p in (getattr(self.shared_state, "framework_agent_phase_progress", None) or [])
            if isinstance(p, dict) and self._framework_candidate_key(p)
        }

    def _unprocessed_framework_agent_candidates(self) -> list[dict[str, Any]]:
        """Return all not-yet-processed candidates in the latest batch (order preserved).

        Processed = the candidate key already appears in
        ``framework_agent_phase_progress`` (see
        :meth:`_framework_processed_candidate_keys`).

        Returns:
            The list of candidate dicts lacking a progress row (possibly empty).
        """
        state = self.shared_state
        batches = getattr(state, "framework_agent_batches", None) or []
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

    def _select_next_framework_agent_candidate(self) -> dict[str, Any] | None:
        """Return the next unprocessed candidate in the batch, in the order given.

        Linear on purpose: the discovery specialist ranks what it finds and
        judges each entry, so the batch arrives ordered. Re-ranking here would
        overrule a judgement made with the gap and the tried-ledger in view,
        using less context than the specialist had.

        Returns:
            The first candidate dict not yet recorded in the phase progress, or
            ``None`` when no batch exists or all are processed.
        """
        unprocessed = self._unprocessed_framework_agent_candidates()
        return unprocessed[0] if unprocessed else None

    def _authoring_specialist_domain(self) -> str:
        """Pick the authoring domain that matches the session's framework kind.

        Returns:
            ``"framework_rewrite_specialist"`` for a scriptable (server-less)
            framework, else ``"serving_specialist"``.
        """
        from ..specialists.domains import authoring_domain_for_framework

        return authoring_domain_for_framework(getattr(self.shared_state, "framework", ""))

    def _render_rewrite_evidence_for_prompt(self) -> str:
        """Render the measured host-side rewrite evidence as prompt lines.

        Returns:
            The evidence block, or ``""`` when no profile has produced one yet.
            Empty is normal on the first FRAMEWORK pass (the arm can run before
            any profile has landed), and the specialist's own reading of the
            source is the fallback.
        """
        path = str(getattr(self.shared_state, "last_framework_rewrite_evidence", "") or "").strip()
        if not path:
            return ""
        try:
            import json as _json

            from ..actions.executors._framework_rewrite_evidence import summarize_for_prompt

            document = _json.loads(Path(path).read_text(encoding="utf-8"))
            return summarize_for_prompt(document)
        except Exception:  # noqa: BLE001 — advisory only
            log.warning("FRAMEWORK: rewrite-evidence render failed path=%s", path, exc_info=True)
            return ""

    def _rewrite_evidence_absence_note(self) -> str:
        """Explain an empty evidence block instead of implying there is nothing to find.

        "No candidates" and "the probe never produced any" read identically to a
        specialist, and only the first one means the source is already clean.
        Saying which it is decides whether the specialist should trust the
        silence or go read the loop itself.

        Returns:
            str: The prompt note describing why no evidence is present.
        """
        read_the_source = (
            "Locate the candidates by reading the source: find the denoising / "
            "rollout loop and ask, for each call inside it, whether the result "
            "can change across iterations."
        )
        status = str(getattr(self.shared_state, "last_framework_rewrite_evidence_status", "") or "").strip()
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
        if str(getattr(self.shared_state, "last_framework_rewrite_evidence", "") or "").strip():
            # Reached only when a document is on record but rendering it produced
            # nothing, so the evidence exists and this prompt cannot show it.
            return (
                "Host-side rewrite evidence was collected but could not be "
                "rendered here, so its absence below means nothing. " + read_the_source
            )
        return "No host-side rewrite evidence has been collected yet for this session. " + read_the_source

    def _framework_local_explore_arm_enabled(self) -> bool:
        """True when the candidate-free local-exploration arm may run.

        Requires the authoring capability (``framework_agent_authoring_enabled``)
        and the dedicated toggle (``framework_local_explore_enabled``, default
        on; ``--no-framework-local-explore`` opts out).

        Returns:
            ``True`` when both toggles allow the arm.
        """
        state = self.shared_state
        return bool(getattr(state, "framework_agent_authoring_enabled", False)) and bool(
            getattr(state, "framework_local_explore_enabled", True)
        )

    def _compose_framework_local_explore_gap(self) -> tuple[str, list[str]]:
        """Compose the ``(gap, keywords)`` steering the local-exploration arm.

        Reuses the same bottleneck-aware composer as PR discovery so the
        specialist attacks the current hot path.

        Returns:
            A ``(gap_description, keywords)`` tuple (``("", [])`` on failure).
        """
        state = self.shared_state
        try:
            from ..actions.executors._framework_gap_composer import compose_gap

            return compose_gap(
                framework=str(getattr(state, "framework", "") or ""),
                gpu_type=str(getattr(state, "gpu_type", "") or ""),
                model_class=str(getattr(state, "model_class", "") or ""),
                precision=str(getattr(state, "precision", "") or ""),
                profile_kernel_breakdown_path=getattr(state, "last_profile_kernel_breakdown", None),
                rewrite_evidence_path=getattr(state, "last_framework_rewrite_evidence", None),
            )
        except Exception:  # noqa: BLE001 — advisory only
            log.debug("FRAMEWORK: local-explore gap compose failed", exc_info=True)
            return "", []

    def _next_local_explore_candidate_id(self) -> str:
        """Return the next unique local-exploration candidate id.

        The sequence is the count of local-exploration progress rows already
        recorded, so the id is stable while an attempt is in flight (no new row
        yet) and increments once it reaches a terminal row — yielding a fresh
        attempt each round until the phase plateaus.

        Returns:
            An id of the form ``local_explore:<n>``.
        """
        progress = getattr(self.shared_state, "framework_agent_phase_progress", None) or []
        n = sum(
            1 for p in progress if isinstance(p, dict) and str(p.get("candidate_id") or "").startswith("local_explore:")
        )
        return f"local_explore:{n}"

    def _make_local_explore_pseudo_candidate(self) -> dict[str, Any] | None:
        """Build the synthetic local-exploration candidate, or ``None`` when disabled.

        Returns:
            A candidate dict tagged ``kind="local_explore"`` for the ranker, or
            ``None`` when the arm is disabled.
        """
        if not self._framework_local_explore_arm_enabled():
            return None
        gap, keywords = self._compose_framework_local_explore_gap()
        cand_id = self._next_local_explore_candidate_id()
        title = (
            f"local source exploration ({gap})"
            if gap
            else "local source exploration (author a throughput patch from live source + profile)"
        )
        return {
            "kind": self._LOCAL_EXPLORE_KIND,
            "candidate_id": cand_id,
            "title": title,
            "repo": "(local source)",
            "framework": str(getattr(self.shared_state, "framework", "") or "").strip().lower(),
            "gap_description": gap,
            "gap_keywords": keywords,
            "gap_canonical_id": f"gap.framework.local_explore.{cand_id}",
        }

    async def _maybe_dispatch_local_explore(self, *, reason: str) -> bool:
        """Dispatch a local-exploration specialist when the arm is enabled.

        Used as the discovery-exhaustion fallback: rather than marking the phase
        done, author a patch from the live source. No-op when the arm is disabled.

        Args:
            reason: Short provenance tag recorded on the dispatch log line.

        Returns:
            ``True`` when a specialist was dispatched, else ``False``.
        """
        if not self._framework_local_explore_arm_enabled():
            return False
        pseudo = self._make_local_explore_pseudo_candidate()
        if pseudo is None:
            return False
        tid = await self._enqueue_framework_agent_local_explore_specialist(pseudo, reason=reason)
        return bool(tid)

    async def _enqueue_framework_agent_local_explore_specialist(
        self,
        candidate: dict[str, Any],
        *,
        reason: str = "",
    ) -> str:
        """Dispatch a candidate-free authoring specialist (no upstream PR lead).

        The specialist reads the live framework source + profiling evidence (and
        may web-search the latest upstream code) and authors the best throughput
        win, flowing through the same autosubmit -> Critic -> integrate_patch ->
        bench -> KEEP/REVERT path as the PR-authoring track. Empty deliverables
        stamp an ``author_empty`` progress row (counts as a no-KEEP for plateau).

        Args:
            candidate: The synthetic ``local_explore`` candidate (carries the
                candidate id + composed gap).
            reason: Short provenance tag for the dispatch log line.

        Returns:
            The dispatched specialist ``task_id`` (empty on a livelock-break
            short-circuit).
        """
        state = self.shared_state
        cand_id = self._framework_candidate_key(candidate) or self._next_local_explore_candidate_id()
        gap = str(candidate.get("gap_description") or "").strip()
        gap_cid = str(candidate.get("gap_canonical_id") or "").strip() or f"gap.framework.local_explore.{cand_id}"
        framework = str(candidate.get("framework") or getattr(state, "framework", "") or "").strip().lower()
        # Route by framework kind. An iterative model pipeline and a
        # request-serving framework share almost no optimization surface, so the
        # scriptable case goes to the rewrite domain; everything else resolves to
        # the serving domain this dispatch has always used. The static guidance
        # for each lives in its domain description, and the per-kind brief in
        # ``_TASK_KIND_BRIEFS``; only measured, per-dispatch evidence is passed
        # as notes below.
        domain = self._authoring_specialist_domain()
        rewrite_arm = domain == "framework_rewrite_specialist"
        notes = ""
        if rewrite_arm:
            notes = self._render_rewrite_evidence_for_prompt() or self._rewrite_evidence_absence_note()
        try:
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
        except Exception:  # noqa: BLE001
            log.debug("FRAMEWORK local-explore: upsert_gap failed", exc_info=True)
        prior_attempts: list[dict[str, Any]] = []
        try:
            memory = self._build_framework_working_memory()
            for t in memory.get("tried_and_why") or []:
                if isinstance(t, dict) and str(t.get("ref") or "").strip():
                    prior_attempts.append(t)
        except Exception:  # noqa: BLE001
            pass
        params: dict[str, Any] = {
            "domain": domain,
            "source_phase": "FRAMEWORK_AGENT",
            "gap_canonical_id": gap_cid,
            # No ``lever_kind``: this arm names a gap, not a lever, and its
            # specialist returns either. ``patch_lever_kind`` reads what came back.
            "gap_symptom": (gap or "Author a framework source patch from live source + profile evidence"),
            "gap_layer": "framework",
            "framework": framework,
            "task_kind": "framework_local_explore",
            "prior_attempts": prior_attempts,
            "notes": notes,
            # Same provenance markers as the PR-authoring track so the
            # autosubmit -> integrate_patch -> authored-outcome bridge applies.
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand_id,
            "framework_batch_id": "",
            "framework_audit": {},
            "framework_local_explore": True,
            "source": "coordinator_internal",
            **self._framework_gpu_params(),
        }
        try:
            await self._warm_specialist_params(params)
        except Exception:  # noqa: BLE001 — best-effort warmup
            log.debug("FRAMEWORK local-explore: warm specialist params failed", exc_info=True)
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=3600)
        create_kwargs: dict[str, Any] = {
            "kind": "specialist",
            "params": params,
            "requires_lanes": lanes,
            "side_effects": ["writes_results", "writes_patches"],
            "lease_ttl_sec": ttl,
        }
        # The registry de-duplicates by key and hands back whatever row it finds,
        # so a candidate whose specialist failed keeps resolving to that failure:
        # the phase re-selects the candidate every tick, logs a dispatch, and
        # nothing runs. A specialist that was mid-flight when the process died is
        # reclaimed as failed, so one interrupted run retires a candidate for
        # good. Retries take a fresh key, which is what this registry documents.
        base_idem = f"framework_agent_local_explore:{cand_id}{self._cycle_idem_suffix()}"
        spec_task = None
        _spec_existing = False
        for attempt in range(_LOCAL_EXPLORE_MAX_ATTEMPTS):
            idem = base_idem if attempt == 0 else f"{base_idem}:r{attempt}"
            spec_task, _spec_existing = await self.tasks.create_or_return_existing(
                idempotency_key=idem,
                **create_kwargs,
            )
            if not (_spec_existing and str(getattr(spec_task, "state", "") or "") == "failed"):
                break
            log.info(
                "FRAMEWORK local-explore: %s already failed under %s; retrying candidate %s",
                getattr(spec_task, "task_id", "?"),
                idem,
                cand_id,
            )
        else:
            log.warning(
                "FRAMEWORK local-explore: candidate %s exhausted %d specialist "
                "attempt(s); leaving it to the phase to select another",
                cand_id,
                _LOCAL_EXPLORE_MAX_ATTEMPTS,
            )
            return ""
        from ..state.task_registry import TERMINAL_STATES as _TERMINAL_STATES

        if _spec_existing and str(getattr(spec_task, "state", "") or "") in _TERMINAL_STATES:
            already_rows = self._framework_processed_candidate_keys()
            authoring_inflight = await self._framework_agent_authoring_inflight()
            if cand_id and cand_id not in already_rows and not authoring_inflight:
                try:
                    recovered = await self._recover_framework_agent_authoring_outcome(
                        specialist_task=spec_task,
                    )
                except Exception:  # noqa: BLE001 — never wedge the pump
                    log.exception(
                        "FRAMEWORK local-explore: terminal outcome recovery failed candidate=%s",
                        cand_id,
                    )
                    recovered = False
                if not recovered:
                    log.warning(
                        "FRAMEWORK local-explore: terminal outcome unavailable candidate=%s state=%s",
                        cand_id,
                        getattr(spec_task, "state", ""),
                    )
                    # Stamp a terminal row so an unrecoverable outcome cannot make
                    # the pump re-select the same finished specialist forever.
                    self._stamp_framework_progress(
                        candidate_id=cand_id,
                        batch_id=str(candidate.get("batch_id") or ""),
                        status="recovery_failed",
                        rationale="local-explore outcome unrecoverable from persisted results",
                        provenance="pump",
                    )
                return ""
        spec_tid = str(getattr(spec_task, "task_id", "") or "")
        try:
            if spec_tid and cand_id:
                if not isinstance(getattr(state, "framework_agent_specialist_candidate_map", None), dict):
                    state.framework_agent_specialist_candidate_map = {}
                state.framework_agent_specialist_candidate_map[spec_tid] = cand_id
                state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — best-effort provenance
            log.debug("FRAMEWORK local-explore: specialist->candidate map write failed", exc_info=True)
        log.info(
            "FRAMEWORK: dispatched local-exploration specialist candidate=%s gap=%s reason=%s",
            cand_id,
            gap_cid,
            reason or "resident",
        )
        return spec_tid

    def _framework_known_candidate_ids(self) -> set[str]:
        """All candidate ids already discovered into any prior batch (dedup for new batches).

        Returns:
            A set of candidate ids drawn from every prior batch plus PR ids the
            research scout has already mined.
        """
        state = self.shared_state
        ids: set[str] = set()
        batches = getattr(state, "framework_agent_batches", None) or []
        if not isinstance(batches, list):
            return ids
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            for cand in batch.get("candidates") or []:
                if not isinstance(cand, dict):
                    continue
                # Canonical key (candidate_id/pr_url/ref); synthetic repo-PR
                # fallback only when the candidate carries no identity field.
                cid = self._framework_candidate_key(cand) or f"{cand.get('repo', '')}-{cand.get('pr_number', '')}"
                if cid:
                    ids.add(cid)
        # Fold in PR ids the research scout already mined so the two mechanisms never re-process a PR.
        for pid in getattr(state, "research_scout_seen_pr_ids", None) or []:
            pid = str(pid or "").strip()
            if pid:
                ids.add(pid)
        return ids

    def _framework_tried_refs(self) -> list[str]:
        """Refs already discovered this phase (fed to compose_gap to bias away from prior PR categories).

        Returns:
            A list of already-known candidate id refs.
        """
        refs: list[str] = []
        for cid in self._framework_known_candidate_ids():
            if cid:
                refs.append(cid)
        return refs

    def _build_framework_working_memory(self) -> dict[str, Any]:
        """Aggregate the FRAMEWORK working memory from the three ledgers (deterministic, zero-LLM).

        Mirrors ``orchestration_memory`` for the framework layer: a structured,
        purely-derived view of "what this session already tried / excluded /
        learned", so discovery and the candidate ranker can be biased away from
        repeating failed candidates. No new data source — it aggregates
        ``framework_agent_phase_progress`` (terminal rows, reliably populated
        after the P0 fix) and the batch dedup set. Critic denials are part of
        that ledger: the gate stamps them as ``critic_denied`` rows carrying
        the rationale.

        Returns:
            A dict with:
              - ``tried_and_why``: recent terminal candidates as
                ``{ref, status, gain_pct, why}`` (most recent last, capped).
              - ``excluded_refs``: sorted union of known-candidate ids and
                candidates that already carry a terminal progress row (hard
                dedup source for discovery — Step B).
              - ``learnings``: deduped Critic rationales for denied candidates.
              - ``pending``: still-unprocessed candidate refs in the latest batch.
        """
        state = self.shared_state
        progress = getattr(state, "framework_agent_phase_progress", None) or []
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
        # Learnings: distinct Critic denial rationales (negative priors), capped.
        # Read from the progress ledger, the only place a denial is recorded:
        # ``_record_framework_agent_critic_denied`` stamps a ``critic_denied``
        # row carrying the Critic's reasoning.
        learnings: list[str] = []
        seen_learn: set[str] = set()
        for row in rows:
            if str(row.get("status") or "").strip().lower() != FRAMEWORK_CRITIC_DENIED_STATUS:
                continue
            rationale = str(row.get("rationale") or "").strip()
            if rationale and rationale not in seen_learn:
                seen_learn.add(rationale)
                learnings.append(rationale[:200])
            if len(learnings) >= self._FRAMEWORK_TRIED_MEMORY_CAP:
                break
        pending = [self._framework_candidate_key(c) for c in self._unprocessed_framework_agent_candidates()]
        excluded = self._framework_known_candidate_ids() | self._framework_processed_candidate_keys()
        return {
            "tried_and_why": tried,
            "excluded_refs": sorted(r for r in excluded if r),
            "learnings": learnings,
            "pending": [r for r in pending if r],
        }

    def _framework_agent_discover_repo_urls(self, framework: str) -> list[str]:
        """Repo URLs to query for the FRAMEWORK batch: framework's own repo + global PR_QUERY_REPOS allowlist, dedup preserving order.

        Args:
            framework: The framework name whose own repo seeds the query.

        Returns:
            An order-preserving, deduped list of repo URLs to query.
        """
        from hyperloom.inference_optimizer import framework_registry

        from ..framework import client as _fa_client
        from ..specialists.domains import PR_QUERY_REPOS

        urls: list[str] = []

        def _add(u: str) -> None:
            """Append a trimmed URL to ``urls`` if non-empty and not already present.

            Args:
                u (str): A candidate repo URL.
            """
            u = (u or "").strip()
            if u and u not in urls:
                urls.append(u)

        # Primary: the framework's own repo.
        primary_repo_url = _fa_client.repo_url_for_framework(framework)
        _add(primary_repo_url)

        # Serving/infra PRs cannot be git-applied to scriptable model repos, so
        # a scriptable session queries its own repo and nothing else. Scoping on
        # scriptability alone is deliberate: keying it on also having a repo URL
        # excluded the case that needs it most, since an operator-supplied
        # workload has no upstream repo by construction and so inherited the
        # whole serving allowlist. It queries nothing now rather than everything.
        if not framework_registry.is_scriptable(framework):
            for repo in PR_QUERY_REPOS:
                repo = str(repo or "").strip()
                if repo and "/" in repo:
                    _add(f"https://github.com/{repo}.git")
            if not urls:
                # Last-ditch: let phase_discover resolve from framework itself.
                _add(_fa_client.repo_url_for_framework(framework or "sglang"))
        return urls

    def _record_framework_agent_phase_done(
        self,
        *,
        reason: str,
        failure_count: int,
    ) -> None:
        """Append a framework_agent_phase_done row to phase_history describing why the pump gave up.

        Args:
            reason: Human-readable reason the phase ended.
            failure_count: Number of consecutive discover failures recorded.
        """
        state = self.shared_state
        try:
            from ..framework import client as _fa_client
            from ..framework.artifacts import summarize_candidate_outcomes

            # Classify this phase's candidate outcomes so the report /
            # robustness can tell "discovered nothing" (empty_discovery) apart
            # from "tested candidates but none kept" (tested_no_keep).
            summary = summarize_candidate_outcomes(
                getattr(state, "framework_agent_phase_progress", None),
            )
            outcome_class = str(summary.get("outcome_class") or "empty_discovery")

            # Consecutive empty-discovery tracking → advisory ("framework phase
            # ineffective"). Reset the streak the moment a phase tested anything.
            prev_empty = int(getattr(state, "framework_consecutive_empty_discoveries", 0) or 0)
            if outcome_class == "empty_discovery":
                consecutive_empty = prev_empty + 1
            else:
                consecutive_empty = 0
            state.framework_consecutive_empty_discoveries = consecutive_empty

            advisory = ""
            if outcome_class == "empty_discovery" and consecutive_empty >= 2:
                advisory = (
                    f"framework phase ineffective: {consecutive_empty} consecutive "
                    "macro-cycles discovered zero candidates"
                )
            elif outcome_class == "tested_no_keep":
                advisory = (
                    "framework phase tested candidates but none cleared the gate "
                    f"(tested={summary.get('tested')}, keeps=0)"
                )
            if advisory:
                log.warning("FRAMEWORK advisory: %s", advisory)

            state.append_phase_history_event(
                reason=reason,
                evidence={
                    "event": "framework_agent_phase_done",
                    "failure_count": int(failure_count),
                    "empty_count": int(getattr(state, "framework_agent_empty_discoveries", 0) or 0),
                    "retry_limit": int(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT),
                    "batches_discovered": len(getattr(state, "framework_agent_batches", None) or []),
                    "outcome_class": outcome_class,
                    "candidate_outcomes": summary.get("by_status") or {},
                    "keeps": int(summary.get("keeps") or 0),
                    "tested": int(summary.get("tested") or 0),
                    "consecutive_empty_discoveries": consecutive_empty,
                    "advisory": advisory,
                },
            )
        except Exception:  # noqa: BLE001 — defensive
            pass

    async def _enqueue_framework_agent_task(self, candidate: dict[str, Any]) -> None:
        """Enqueue an ``integrate_patch`` task that lands ``candidate``'s diff.

        Builds the task params (candidate, batch id, baseline throughput, KEEP
        threshold, framework, and the ownership markers the authored-outcome
        bridge keys on) and creates an idempotent ``integrate_patch`` task with
        ``patch_source=upstream_pr``, whose lanes and lease TTL come from the
        action catalogue. On enqueue failure,
        records an ``enqueue_failed`` progress row so the pump skips the
        candidate next tick instead of spinning.

        Args:
            candidate (dict[str, Any]): The discovered PR candidate to apply
                and benchmark.
        """
        state = self.shared_state
        cand_id = self._framework_candidate_key(candidate)
        params = {
            "candidate": candidate,
            "batch_id": candidate.get("batch_id") or "",
            # One action lands every patch; this says where the diff comes from.
            "patch_source": PATCH_SOURCE_UPSTREAM_PR,
            "lever_kind": LEVER_UPSTREAM_PR,
            # The authored-outcome bridge, the candidate-processed dedup, the
            # batch max-gain roll-up and phase attribution all key on these two
            # markers. Without them a KEEP lands with no progress row, the
            # plateau judge never advances, and the gain reports unattributed.
            "framework_agent_authoring": True,
            "framework_agent_candidate_id": cand_id,
            "framework_batch_id": str(candidate.get("batch_id") or ""),
            "source_phase": "FRAMEWORK_AGENT",
            "base_tput": resolve_grading_anchor_tput(state),
            # Same decaying bar the explore and integrate_patch dispatch paths inject.
            "keep_threshold_pct": _phase_state.resolve_keep_threshold(state),
            "framework": str(candidate.get("framework") or getattr(state, "framework", "") or "").strip().lower(),
            # Source patches require the accuracy gate for KEEP.
            "require_accuracy_for_keep": True,
            "accuracy_baseline": float(getattr(state, "baseline_accuracy", 0.0) or 0.0),
            # The lane templates from the shipped default config, which
            # materializes RUN_EVAL=true and would override the session's choice.
            "disable_run_eval": bool(getattr(state, "eval_disabled", False)),
        }
        idem = f"framework:{candidate.get('batch_id', '')}:{cand_id}"
        lanes, ttl = self._registry_lanes_ttl("integrate_patch")
        try:
            # A framework candidate rebuilds and benchmarks, so it cannot share
            # the GPU. Enqueueing without lanes would run it unserialised
            # against every other task; the handler below turns this into a
            # warning plus a progress row.
            if not lanes:
                raise RuntimeError("integrate_patch resolved to no lanes; the task would run without GPU exclusivity.")
            await self.tasks.create_or_return_existing(
                kind="integrate_patch",
                params=params,
                idempotency_key=idem,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
            )
            log.info(
                "FRAMEWORK: enqueued candidate=%s batch=%s",
                cand_id,
                candidate.get("batch_id") or "",
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "FRAMEWORK: failed to enqueue candidate=%s: %r",
                cand_id,
                exc,
            )
            # Record enqueue_failed progress row so the candidate is skipped next tick (else the loop spins).
            self._stamp_framework_progress(
                candidate_id=cand_id,
                batch_id=str(candidate.get("batch_id") or ""),
                status="enqueue_failed",
                kept=False,
                rationale=repr(exc),
                provenance="pump",
                extra={"error": repr(exc)},
            )

    def _collect_framework_agent_candidate_priors(self) -> dict[str, Any]:
        """Return compact session-local priors for the Critic gate.

        Everything the Critic needs about earlier candidates lives in the
        progress ledger, denials included, so the outcomes carry the rationale
        rather than being paired with a separate decision list. Only the rows
        that were stamped with a reason have one — bench results record their
        numbers instead — so the key is omitted rather than sent empty.

        Returns:
            A dict with ``recent_outcomes`` (recent terminal apply/bench
            results, each with the reason recorded for it, where there is one),
            bounded to a short tail.
        """
        raw_progress = getattr(self.shared_state, "framework_agent_phase_progress", None) or []
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

    async def _submit_framework_agent_candidate_for_review(
        self,
        candidate: dict[str, Any],
        *,
        audit: dict[str, Any] | None = None,
        audit_step: str = "",
    ) -> None:
        """Submit a FRAMEWORK candidate as a normal ``proposal`` for async Critic review.

        Mirrors :meth:`_maybe_autosubmit_specialist_patches`: writes a
        ``topic="proposal"`` message + registers a :class:`PendingProposal`,
        bypassing ``_handle_intent`` / ``PolicyGate`` (``framework_agent`` is a
        COORDINATOR_INTERNAL action that ``propose_action`` would deny). The
        Critic verdict arrives on a later tick and drives the apply/author
        enqueue (approve/advise) or the critic_denied row (reject) via
        ``_handle_single_verdict``. All context needed by ``replay_for_resume``
        and ``_materialize_framework_agent_candidate`` lives in the payload, so no
        separate persistent candidate map is kept.

        Args:
            candidate: The discovered PR candidate to gate.
            audit: The semantic-audit verdict (carried for the authoring seed).
            audit_step: The resolved route (``direct_framework`` /
                ``author_via_specialist`` / ``""`` for legacy both-tracks).
        """
        cand_id = self._framework_candidate_key(candidate)
        batch_id = str(candidate.get("batch_id") or "")
        # Dedup: a candidate is already awaiting its pre-screen verdict.
        for p in self.state.pending_proposals.values():
            try:
                if getattr(p, "action_name", "") != "integrate_patch":
                    continue
                if not (getattr(p, "payload", None) or {}).get("framework_agent_candidate_id"):
                    continue
                if getattr(p, "decided", False):
                    continue
                pl = getattr(p, "payload", {}) or {}
                if str(pl.get("framework_agent_candidate_id") or "") == cand_id and cand_id:
                    return
            except Exception:  # noqa: BLE001 — defensive
                continue
        # Repeated-review backstop: count how many times this candidate has
        # been sent for review. Under healthy operation a candidate is submitted
        # once (a terminal row makes it "processed"); repeated submissions mean a
        # terminal-row leak let it be re-selected. Past the abort threshold,
        # force a terminal row and stop — no single candidate can burn the phase.
        if cand_id:
            counts = getattr(self.shared_state, "framework_agent_review_counts", None)
            if not isinstance(counts, dict):
                counts = {}
                self.shared_state.framework_agent_review_counts = counts
            count = int(counts.get(cand_id, 0) or 0) + 1
            counts[cand_id] = count
            if count > self._MAX_REPEATED_REVIEW_SUBMISSIONS:
                log.warning(
                    "FRAMEWORK: candidate=%s submitted for review %d times "
                    "(> cap %d); aborting to protect the phase budget",
                    cand_id,
                    count,
                    self._MAX_REPEATED_REVIEW_SUBMISSIONS,
                )
                self._stamp_framework_progress(
                    candidate_id=cand_id,
                    batch_id=batch_id,
                    status="repeated_review_abort",
                    kept=False,
                    rationale=(f"submitted for review {count} times (> cap {self._MAX_REPEATED_REVIEW_SUBMISSIONS})"),
                    provenance="pump",
                    extra={"review_submissions": count},
                )
                return
        propose_payload: dict[str, Any] = {
            "action_name": "integrate_patch",
            "provenance": LEVER_UPSTREAM_PR,
            "predicted_gain_pct": 0.0,
            "candidate": dict(candidate),
            "batch_id": batch_id,
            "framework_agent_candidate_id": cand_id,
            "audit": dict(audit) if isinstance(audit, dict) else {},
            "audit_step": str(audit_step or ""),
            "priors": self._collect_framework_agent_candidate_priors(),
        }
        msg = Message.new(
            "coordinator",
            "*",
            "proposal",
            {**propose_payload, "needs_review": True},
            priority=1,
        )
        await self.bus.append_and_seq(msg)
        self.state.pending_proposals[msg.msg_id] = PendingProposal(
            proposal_msg_id=msg.msg_id,
            from_agent="coordinator",
            action_name="integrate_patch",
            predicted_gain_pct=0.0,
            payload=dict(propose_payload),
        )
        log.info(
            "FRAMEWORK: candidate submitted for Critic review msg_id=%s candidate=%s batch=%s audit_step=%s",
            msg.msg_id,
            cand_id,
            batch_id,
            audit_step or "<unknown>",
        )
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "framework_agent_candidate_submitted_for_review",
                "proposal_msg_id": msg.msg_id,
                "candidate_id": cand_id,
                "batch_id": batch_id,
                "audit_step": str(audit_step or ""),
            },
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "save after framework_agent candidate submit failed for candidate=%s",
                cand_id,
            )

    async def _materialize_framework_agent_candidate(
        self,
        pending: "PendingProposal",
    ) -> None:
        """Route a Critic-approved FRAMEWORK candidate to the apply / author tracks.

        Reads the audit route stamped on the proposal payload and reuses the
        existing ``direct_framework`` (raw-diff executor) vs
        ``author_via_specialist`` enqueue helpers, with the same
        authoring-disabled → raw-diff fallback the pump used to apply inline.

        Args:
            pending: The approved framework_agent pending proposal.
        """
        payload = pending.payload or {}
        candidate = dict(payload.get("candidate") or {})
        audit = payload.get("audit") if isinstance(payload.get("audit"), dict) else {}
        audit_step = str(payload.get("audit_step") or "")
        cand_id = str(payload.get("framework_agent_candidate_id") or self._framework_candidate_key(candidate))
        batch_id = str(payload.get("batch_id") or candidate.get("batch_id") or "")
        authoring_enabled = bool(getattr(self.shared_state, "framework_agent_authoring_enabled", False))
        want_raw = audit_step == "direct_framework"
        want_author = audit_step == "author_via_specialist"
        if audit_step not in ("direct_framework", "author_via_specialist"):
            want_raw = True
            want_author = True
        if want_author and not authoring_enabled:
            want_raw = True
            want_author = False
        log.info(
            "FRAMEWORK: critic-approved candidate=%s batch=%s audit_step=%s raw=%s author=%s",
            cand_id,
            batch_id,
            audit_step or "<unknown>",
            want_raw,
            want_author,
        )
        if want_raw:
            # _enqueue_framework_agent_task owns its own enqueue_failed terminal
            # row on failure, so a raw-track candidate always ends up processed.
            await self._enqueue_framework_agent_task(candidate)
        if want_author and authoring_enabled:
            try:
                await self._enqueue_framework_agent_authoring_specialist(
                    candidate,
                    audit=audit if isinstance(audit, dict) else {},
                )
            except Exception as exc:  # noqa: BLE001 — never wedge the phase
                log.warning(
                    "FRAMEWORK: authoring specialist dispatch failed: %r",
                    exc,
                )
                # Author-only route (no raw track to own a terminal row): stamp
                # materialize_failed so an approved-but-undispatchable candidate
                # is not re-selected every tick.
                if not want_raw:
                    self._stamp_framework_progress(
                        candidate_id=cand_id,
                        batch_id=batch_id,
                        status="materialize_failed",
                        kept=False,
                        rationale=repr(exc),
                        provenance="pump",
                        extra={"error": repr(exc)},
                    )

    def _stamp_framework_progress(
        self,
        *,
        candidate_id: str,
        batch_id: str = "",
        status: str,
        kept: bool = False,
        rationale: str = "",
        provenance: str = "",
        gain_pct: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> bool:
        """Idempotently stamp a terminal ``framework_agent_phase_progress`` row.

        The single terminal-row writer for every FRAMEWORK path that ends a
        candidate's life without a benched executor result (critic denial,
        needs_review dead-ends, materialize/enqueue failures, silent apply/bench
        failures, repeated-review aborts). Guarantees the P0 invariant: any
        candidate that can no longer advance carries exactly one terminal row,
        so the pump's plateau / phase-done early-exit accrues instead of relying
        on the budget-cap backstop.

        Idempotent: a candidate key that already has ANY progress row is left
        untouched (returns ``False``) so a later path can never double-stamp or
        overwrite an earlier verdict. Writes the row + a ``decision.json`` and
        persists SharedState. Best-effort on the artifact/save side (never
        raises into the pump).

        Args:
            candidate_id: The canonical candidate key (see
                :meth:`_framework_candidate_key`).
            batch_id: The discovery batch the candidate belonged to.
            status: The terminal status (e.g. ``critic_denied`` /
                ``no_result_failed`` / ``reauthor_cap`` …).
            kept: Whether the candidate was promoted (terminal rows are almost
                always ``False``).
            rationale: Human-readable reason recorded on the row + decision.json.
            provenance: Origin tag for the decision.json (``critic`` / ``pump``
                / ``executor`` …).
            gain_pct: Measured delta, when one exists.
            extra: Optional additional fields merged into the decision.json.

        Returns:
            ``True`` when a new row was appended; ``False`` when the candidate
            already had a row (idempotent no-op) or the key was empty.
        """
        cand_id = str(candidate_id or "")
        if not cand_id:
            return False
        state = self.shared_state
        progress = getattr(state, "framework_agent_phase_progress", None)
        if not isinstance(progress, list):
            progress = []
            state.framework_agent_phase_progress = progress
        if cand_id in {self._framework_candidate_key(p) for p in progress if isinstance(p, dict)}:
            return False
        row: dict[str, Any] = {
            "candidate_id": cand_id,
            "batch_id": str(batch_id or ""),
            "status": str(status or ""),
            "kept": bool(kept),
            "rationale": str(rationale or ""),
            "gain_pct": (float(gain_pct) if isinstance(gain_pct, (int, float)) else 0.0),
            "provenance": str(provenance or ""),
            "ts": datetime.now(timezone.utc).isoformat(),
            "cycle": int(getattr(state, "macro_cycle", 0) or 0),
        }
        # Merge caller-supplied extras (e.g. ``error`` / ``review_submissions``)
        # onto the row too, without clobbering the canonical fields above, so
        # downstream consumers see the same detail the decision.json carries.
        if isinstance(extra, dict):
            for k, v in extra.items():
                row.setdefault(str(k), v)
        progress.append(row)
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "FRAMEWORK: save after progress stamp failed candidate=%s status=%s",
                cand_id,
                status,
            )
        log.info(
            "FRAMEWORK: stamped terminal progress candidate=%s batch=%s status=%s",
            cand_id,
            str(batch_id or ""),
            status,
        )
        return True

    async def _record_framework_agent_critic_denied(
        self,
        pending: "PendingProposal",
        reasoning: str,
    ) -> None:
        """Record a ``critic_denied`` FRAMEWORK row when the async gate rejects a candidate.

        Stamps a ``framework_agent_phase_progress`` row + ``decision.json`` keyed
        on the candidate id so ``_select_best_framework_agent_candidate`` treats it
        as processed and the pump advances to the next candidate.

        Args:
            pending: The rejected framework_agent pending proposal.
            reasoning: The Critic's free-text rationale.
        """
        payload = pending.payload or {}
        cand_id = str(
            payload.get("framework_agent_candidate_id")
            or self._framework_candidate_key(
                payload.get("candidate") if isinstance(payload.get("candidate"), dict) else None
            )
        )
        batch_id = str(payload.get("batch_id") or "")
        self._stamp_framework_progress(
            candidate_id=cand_id,
            batch_id=batch_id,
            status=FRAMEWORK_CRITIC_DENIED_STATUS,
            kept=False,
            rationale=str(reasoning or ""),
            provenance="critic",
        )
        log.info(
            "FRAMEWORK: critic rejected candidate=%s batch=%s rationale=%r",
            cand_id,
            batch_id,
            str(reasoning or "")[:200],
        )

    async def _maybe_reauthor_from_critic_feedback(
        self,
        pending: "PendingProposal",
        advisory: dict[str, Any] | None,
    ) -> None:
        """Re-author a framework_agent deliverable once, seeding the next authoring round with the Critic's ``required_evidence``.

        Fires only for a ``needs_review`` verdict carrying non-empty
        ``required_evidence`` on a framework_agent candidate or authoring proposal,
        capped at :attr:`_MAX_REAUTHOR_ATTEMPTS` per candidate.

        Args:
            pending: The proposal the verdict targets.
            advisory: The serialized Critic advisory.
        """
        advisory = advisory or {}
        # Resolve the candidate identity FIRST so the two dead-end returns below
        # (no required_evidence / reauthor cap) can stamp a terminal progress row
        # — a needs_review verdict that neither re-authors nor materializes would
        # otherwise leave the candidate row-less and re-selected forever.
        action_name = str(getattr(pending, "action_name", "") or "")
        payload = getattr(pending, "payload", {}) or {}
        candidate: dict[str, Any] = {}
        audit: dict[str, Any] = {}
        old_sid = ""
        if action_name == "integrate_patch" and payload.get("framework_agent_candidate_id"):
            candidate = dict(payload.get("candidate") or {})
            raw_audit = payload.get("audit")
            audit = raw_audit if isinstance(raw_audit, dict) else {}
        elif action_name == "integrate_patch":
            # Rebuild the candidate from the originating authoring specialist.
            params = payload.get("params") or {}
            if not bool(params.get("framework_agent_authoring")):
                return
            sid = str(params.get("specialist_task_id") or "").strip()
            old_sid = sid
            spec_params: dict[str, Any] = {}
            if sid:
                try:
                    spec_task = await self.tasks.get(sid)
                    spec_params = dict(getattr(spec_task, "params", None) or {})
                except Exception:  # noqa: BLE001 — best-effort lookup
                    spec_params = {}
            candidate = {
                "candidate_id": str(
                    params.get("framework_agent_candidate_id") or spec_params.get("framework_agent_candidate_id") or ""
                ),
                "batch_id": str(params.get("framework_batch_id") or spec_params.get("framework_batch_id") or ""),
                "title": str(spec_params.get("gap_symptom") or ""),
                "framework": str(spec_params.get("framework") or ""),
                "gap_canonical_id": str(spec_params.get("gap_canonical_id") or ""),
            }
            raw_audit = spec_params.get("framework_audit")
            audit = raw_audit if isinstance(raw_audit, dict) else {}
        else:
            return
        cand_id = self._framework_candidate_key(candidate)
        if not cand_id:
            return
        batch_id = str(candidate.get("batch_id") or payload.get("batch_id") or "")
        required_evidence = [str(x).strip() for x in (advisory.get("required_evidence") or []) if str(x).strip()]
        if not required_evidence:
            # needs_review with nothing to act on: no re-author is possible, so
            # this is terminal for the candidate. Stamp it so the pump advances.
            self._stamp_framework_progress(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="needs_review_no_evidence",
                kept=False,
                rationale=str(advisory.get("advice_text") or "")[:500],
                provenance="critic",
            )
            return
        # Skip if the candidate is already materializing as a live integrate_patch task.
        try:
            live_tasks = [*await self.tasks.queued(), *await self.tasks.running()]
        except Exception:  # noqa: BLE001 — defensive
            live_tasks = []
        for t in live_tasks:
            if getattr(t, "kind", "") != "integrate_patch":
                continue
            tp = getattr(t, "params", None) or {}
            if str(tp.get("framework_agent_candidate_id") or "") == cand_id:
                return
        attempts = getattr(self.shared_state, "specialist_reauthor_attempts", None)
        if not isinstance(attempts, dict):
            attempts = {}
            self.shared_state.specialist_reauthor_attempts = attempts
        prior = int(attempts.get(cand_id, 0) or 0)
        if prior >= self._MAX_REAUTHOR_ATTEMPTS:
            await self._record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "reauthor_cap_reached",
                    "candidate_id": cand_id,
                    "attempts": prior,
                    "proposal_msg_id": str(getattr(pending, "proposal_msg_id", "") or ""),
                    "verdict": "needs_review",
                },
            )
            # Re-author budget exhausted: terminal for the candidate. Stamp so
            # the pump stops re-selecting it once this proposal drains.
            self._stamp_framework_progress(
                candidate_id=cand_id,
                batch_id=batch_id,
                status="reauthor_cap",
                kept=False,
                rationale=f"reauthor attempts >= cap ({self._MAX_REAUTHOR_ATTEMPTS})",
                provenance="pump",
            )
            return
        attempt = prior + 1
        attempts[cand_id] = attempt
        critic_feedback = {
            "required_evidence": required_evidence,
            "advice_text": str(advisory.get("advice_text") or ""),
            "risks": [str(r).strip() for r in (advisory.get("risks") or []) if str(r).strip()],
        }
        new_task_id = ""
        try:
            new_task_id = await self._enqueue_framework_agent_authoring_specialist(
                candidate,
                audit=audit,
                reauthor_attempt=attempt,
                critic_feedback=critic_feedback,
            )
        except Exception:  # noqa: BLE001 — never wedge the verdict handler
            log.exception(
                "re-author dispatch failed candidate=%s attempt=%s",
                cand_id,
                attempt,
            )
            return
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "save after re-author dispatch failed candidate=%s",
                cand_id,
            )
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "specialist_reauthor_dispatched",
                "candidate_id": cand_id,
                "attempt": attempt,
                "proposal_msg_id": str(getattr(pending, "proposal_msg_id", "") or ""),
                "old_specialist_task_id": old_sid,
                "new_specialist_task_id": new_task_id,
                "verdict": "needs_review",
                "required_evidence": required_evidence[:6],
            },
        )

    async def _pump_predictor(self, *, caller: str) -> None:
        """Consult the first-pass tuning predictor, when one is configured.

        Off unless an endpoint is set, and shadow-only unless the mode says
        otherwise, so a session that configures nothing behaves as before.

        Args:
            caller: Label identifying the caller ("entry" / "tick").
        """
        from ..predictor.pump import pump as _predictor_pump

        await _predictor_pump(self, caller=caller)

    async def _pump_framework_agent_phase_safely(self, *, caller: str) -> None:
        """Best-effort FRAMEWORK pump wrapper shared by tick and run.

        Args:
            caller: Label identifying the caller ("tick" / "run"), used only in
                the failure log.
        """
        # The chain's later steps land here: a KEEP deepens the stack, which
        # changes the idempotency key, which lets the next prediction enqueue.
        await self._pump_predictor(caller=caller)
        try:
            await self._pump_framework_agent_phase()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("FRAMEWORK pump (%s) failed", caller)

    def _record_framework_agent_authored_outcome(
        self,
        *,
        task: "Task",
        result: Any,
    ) -> None:
        """Bridge an authored-patch ``integrate_patch`` outcome into the FRAMEWORK progress ledger (else the gain is invisible). Attributed to the latest batch; every terminal status is recorded (empty/in-progress statuses and lane-owned ``apply_failed`` retries are skipped).

        Args:
            task: The integrate_patch task carrying the FRAMEWORK authoring
                provenance markers.
            result: The task result; any non-empty terminal status is recorded
                except ``apply_failed`` on a perf lane, which the retry loop owns.
        """
        params = getattr(task, "params", None) or {}
        if not bool(params.get("framework_agent_authoring")):
            return
        res = result if isinstance(result, dict) else getattr(result, "result", None)
        if not isinstance(res, dict):
            return
        status = str(res.get("status") or "")
        # Record EVERY terminal integrate_patch outcome — not just keep/revert.
        # A patch that fails to apply / bench (``apply_failed`` /
        # ``bench_reverted`` / ``error`` …) is still a terminal verdict for the
        # candidate; without a progress row the FRAMEWORK pump re-selects the
        # same candidate every tick and livelocks (the authoring specialist's
        # ``patches_written`` is non-empty so the empty-outcome bridge does not
        # fire either). Only an empty / in-progress status is skipped.
        if not status:
            return
        # apply_failed with a lane field means the unified retry loop will handle
        # this result (either re-dispatch or stamp a terminal row at the cap).
        # Do NOT stamp a progress row here — that would block the retry pump.
        if status == "apply_failed" and res.get("lane") in ("perf_framework", "perf_explore"):
            return
        # Resolve the FRAMEWORK candidate id (a PR URL) that this authored
        # patch belongs to. The integrate_patch task carries only
        # ``specialist_task_id``; map that back to the originating candidate via
        # the dispatch-time map so the progress row is keyed on the same PR URL
        # that ``_select_next_framework_agent_candidate`` checks. Falling back to a
        # task_id here would leave the candidate looking unprocessed forever.
        spec_tid = str(params.get("specialist_task_id") or "")
        cand_map = getattr(self.shared_state, "framework_agent_specialist_candidate_map", None)
        mapped_cand = ""
        if isinstance(cand_map, dict) and spec_tid:
            mapped_cand = str(cand_map.get(spec_tid) or "")
        cand_id = str(
            params.get("framework_agent_candidate_id") or mapped_cand or spec_tid or getattr(task, "task_id", "") or ""
        )
        batch_id = str(params.get("framework_batch_id") or "")
        if not batch_id:
            batches = getattr(self.shared_state, "framework_agent_batches", None) or []
            if isinstance(batches, list) and batches and isinstance(batches[-1], dict):
                batch_id = str(batches[-1].get("batch_id") or "")
        delta_pct = res.get("delta_pct")
        new_tput = res.get("output_throughput")
        gain = float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0
        progress = getattr(self.shared_state, "framework_agent_phase_progress", None)
        if not isinstance(progress, list):
            progress = []
            self.shared_state.framework_agent_phase_progress = progress
        matching = [row for row in progress if isinstance(row, dict) and self._framework_candidate_key(row) == cand_id]
        # A KEEP is the last word on a candidate; any other row is an outcome
        # a later attempt may better, and is replaced below.
        if any(str(row.get("status") or "") == "kept" for row in matching):
            return
        if matching:
            progress[:] = [
                row for row in progress if not (isinstance(row, dict) and self._framework_candidate_key(row) == cand_id)
            ]
        # Roll the batch max-gain stat the plateau judge reads.
        batches = getattr(self.shared_state, "framework_agent_batches", None) or []
        if isinstance(batches, list) and batch_id:
            for entry in reversed(batches):
                if isinstance(entry, dict) and str(entry.get("batch_id") or "") == batch_id:
                    prev = float(entry.get("max_gain_pct_observed_in_batch") or 0.0)
                    if gain > prev:
                        entry["max_gain_pct_observed_in_batch"] = gain
                    break
        recorded = self._stamp_framework_progress(
            candidate_id=cand_id,
            batch_id=batch_id,
            status=status,
            kept=status == "kept",
            rationale=str(res.get("reason") or ""),
            provenance="authored",
            gain_pct=gain,
            extra={
                # The anchor the executor graded against, so post/pre and
                # gain_pct in the same row agree once a stack has formed.
                "pre_tput": float(res.get("base_tput") or params.get("base_tput") or 0.0),
                "post_tput": float(new_tput) if isinstance(new_tput, (int, float)) else 0.0,
                "accuracy_pass": res.get("accuracy_pass"),
                "specialist_task_id": spec_tid,
                "integrate_task_id": str(getattr(task, "task_id", "") or ""),
                "reauthor_attempt": res.get("reauthor_attempt", params.get("reauthor_attempt")),
            },
        )
        if not recorded:
            return
        log.info(
            "FRAMEWORK: authored patch outcome candidate=%s batch=%s status=%s gain=%.2f%%",
            cand_id,
            batch_id,
            status,
            gain,
        )

    async def _recover_framework_agent_authoring_outcome(
        self,
        *,
        specialist_task: "Task",
    ) -> bool:
        """Recover a missed authoring outcome from persisted delegated results."""
        params = getattr(specialist_task, "params", None) or {}
        specialist_task_id = str(getattr(specialist_task, "task_id", "") or "")
        cand_id = str(params.get("framework_agent_candidate_id") or "")
        if not specialist_task_id or not cand_id:
            return False
        messages = await self.bus.tail(n=10000, topic="delegated_result")
        for message in messages:
            payload = getattr(message, "payload", None) or {}
            if (
                str(payload.get("task_id") or "") != specialist_task_id
                or str(payload.get("kind") or "") != "specialist"
            ):
                continue
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            done_payload = result.get("specialist_done")
            # A run that failed before delivering carries its error on the bus
            # entry rather than in the result, and must reach the recorder for
            # the same reason it does on the live path.
            run_error = str(payload.get("error") or "")
            if isinstance(done_payload, dict) or run_error:
                self._record_framework_agent_authoring_empty_outcome(
                    task=specialist_task,
                    done_payload=done_payload if isinstance(done_payload, dict) else {},
                    run_error=run_error,
                )
                if cand_id in self._framework_processed_candidate_keys():
                    return True
            break
        for message in messages:
            payload = getattr(message, "payload", None) or {}
            if str(payload.get("kind") or "") != "integrate_patch":
                continue
            task_id = str(payload.get("task_id") or "")
            if not task_id:
                continue
            try:
                integrate_task = await self.tasks.get(task_id)
            except Exception:  # noqa: BLE001 — stale bus entries are ignored
                continue
            integrate_params = getattr(integrate_task, "params", None) or {}
            if str(integrate_params.get("specialist_task_id") or "") != specialist_task_id or not bool(
                integrate_params.get("framework_agent_authoring")
            ):
                continue
            result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
            self._record_framework_agent_authored_outcome(
                task=integrate_task,
                result=result,
            )
            if cand_id in self._framework_processed_candidate_keys():
                return True
        return False

    def _record_framework_agent_dispatch_failure(
        self,
        *,
        task: "Task",
        run_error: str,
    ) -> None:
        """Stamp a terminal row for a candidate whose specialist never ran.

        The row is what stops the pump re-selecting the candidate every tick, so
        it must exist; ``dispatch_failed`` is what keeps the plateau streak from
        counting it as a search that came back empty.

        Args:
            task: The specialist task that failed before delivering.
            run_error: The dispatch error, recorded as the row's rationale.
        """
        params = getattr(task, "params", None) or {}
        cand_id = str(params.get("framework_agent_candidate_id") or "")
        if not cand_id:
            return
        recorded = self._stamp_framework_progress(
            candidate_id=cand_id,
            batch_id=str(params.get("framework_batch_id") or ""),
            status="dispatch_failed",
            kept=False,
            rationale=run_error[:500],
            provenance="dispatch_failed",
            gain_pct=0.0,
            extra={
                "specialist_task_id": str(getattr(task, "task_id", "") or ""),
                "reauthor_attempt": params.get("reauthor_attempt"),
            },
        )
        if not recorded:
            return
        log.warning(
            "FRAMEWORK: authoring specialist never delivered candidate=%s: %s",
            cand_id,
            run_error[:300],
        )

    def _record_framework_agent_authoring_empty_outcome(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any] | None,
        run_error: str = "",
    ) -> None:
        """Record a terminal FRAMEWORK row when an authoring specialist finishes WITHOUT a patch.

        The authored-patch bridge (`_record_framework_agent_authored_outcome`)
        only fires on a following ``integrate_patch`` task. An *empty*
        deliverable (specialist judged the PR already-present / not-applicable,
        ``patches_written == []``) never produces an ``integrate_patch``, so
        without this hook the candidate is never marked processed and
        `_select_next_framework_agent_candidate` re-selects it every tick (the
        FRAMEWORK pump livelocks re-dispatching the same candidate). Here we
        stamp a `framework_agent_phase_progress` row + `decision.json` so the pump
        advances. Idempotent: a candidate that already has a row is skipped.

        Args:
            task: The completed authoring specialist task (carries the
                ``framework_*`` provenance markers).
            done_payload: The specialist's ``specialist_done`` payload.
            run_error: The dispatch error when the run failed before delivering
                anything. Together with an absent payload it separates "the
                specialist ran and found nothing" from "the specialist never
                ran", which the plateau streak must not treat alike.
        """
        params = getattr(task, "params", None) or {}
        if not bool(params.get("framework_agent_authoring")):
            return
        payload = done_payload if isinstance(done_payload, dict) else {}
        # No payload at all plus an error means the run never delivered: there is
        # no deliverable to judge, so this is infrastructure, not a search result.
        if run_error and not payload:
            self._record_framework_agent_dispatch_failure(task=task, run_error=run_error)
            return
        inner = payload.get("payload") if isinstance(payload.get("payload"), dict) else payload
        # A downstream integrate_patch (owned by the authored-outcome bridge
        # that writes the terminal row) is created by
        # ``_maybe_autosubmit_specialist_patches`` when the deliverable is
        # routable: ``patches_written`` (post safety-vetting) non-empty, OR a
        # config-lever deliverable, OR a non-diff tuned artifact. This empty-
        # outcome bridge MUST mirror EACH of those signals so it never skip-
        # stamps a deliverable autosubmit will route (double terminal row) nor
        # stamps nothing for one autosubmit will NOT route (livelock). The three
        # guards below (patches / config levers / artifacts) are that mirror. If
        # none hold we stamp the terminal row here, REGARDLESS of
        # ``proposal_set`` — the dangerous case being a specialist that authors a
        # patch (proposal_set non-empty) which safety-vetting then DROPS as
        # unusable (missing_target / forbidden_fields), emptying
        # ``patches_written``: autosubmit creates no integrate_patch, so without
        # stamping here the candidate has no terminal row and the FRAMEWORK pump
        # re-dispatches it forever (gap-5 livelock).
        patches = inner.get("patches_written") or []
        if isinstance(patches, list) and patches:
            return
        # Relaxed FRAMEWORK rule: a config-lever deliverable (proposal_set
        # carrying extra_args / extra_envs) is a FULL result, not "empty". When
        # one exists, ``_maybe_autosubmit_framework_config`` routes it through
        # integrate_patch (which owns the terminal row), so do NOT stamp an
        # authored_empty row here.
        if _framework_config_levers_from_done(inner):
            return
        # Relaxed FRAMEWORK rule (parity with autosubmit): a non-diff tuned
        # artifact deliverable (``artifacts_written`` with a real source file) is
        # a FULL result — autosubmit routes it to integrate_patch, which owns the
        # terminal row. Use the SAME routable-signal as autosubmit so we never
        # skip-stamp a deliverable that autosubmit will NOT route (livelock).
        try:
            from ..loop.coordinator import _resolvable_artifacts_from_done
            from hyperloom.inference_optimizer.session.session_paths import (
                runs_dir as _runs_dir,
            )
            from pathlib import Path as _Path

            _sid_arts = str(getattr(task, "task_id", "") or "")
            if self.session_dir is not None and _sid_arts:
                _spec_root = _runs_dir(_Path(self.session_dir), "specialist", _sid_arts)
                if _resolvable_artifacts_from_done(inner, [_spec_root / "worktree", _spec_root]):
                    return
        except Exception:  # noqa: BLE001 — defensive; fall through to stamp
            log.debug("FRAMEWORK: artifacts routable-check failed", exc_info=True)
        cand_id = str(params.get("framework_agent_candidate_id") or "")
        if not cand_id:
            return
        batch_id = str(params.get("framework_batch_id") or "")
        # Map the cached audit verdict to a terminal status.
        audit = params.get("framework_audit") if isinstance(params.get("framework_audit"), dict) else {}
        sem = str((audit or {}).get("semantic_status") or "").strip().lower()
        if sem in ("already_equivalent", "already_superset"):
            status = "already_present"
        elif sem in ("not_present", "partially_present"):
            status = "not_applicable"
        else:
            status = "author_empty"
        reason = str(inner.get("summary") or "").strip()[:500]
        recorded = self._stamp_framework_progress(
            candidate_id=cand_id,
            batch_id=batch_id,
            status=status,
            kept=False,
            rationale=reason,
            provenance="authored_empty",
            gain_pct=0.0,
            extra={
                "specialist_task_id": str(getattr(task, "task_id", "") or ""),
                "reauthor_attempt": params.get("reauthor_attempt"),
            },
        )
        if not recorded:
            return
        log.info(
            "FRAMEWORK: authoring specialist empty deliverable candidate=%s batch=%s status=%s",
            cand_id,
            batch_id,
            status,
        )

    async def _candidate_discovery_inflight(self) -> bool:
        """True while a candidate-discovery specialist is queued or running."""
        queued = await self.tasks.queued()
        running = await self.tasks.running()
        return any(
            getattr(t, "kind", "") == "specialist"
            and bool((getattr(t, "params", None) or {}).get("candidate_discovery"))
            for t in (*queued, *running)
        )

    async def _maybe_enqueue_candidate_discovery(self, *, reason: str) -> bool:
        """Dispatch the candidate-discovery specialist when the pool is empty.

        Minimum supply only; Orchestration dispatches the same specialist
        whenever it judges the upstream lane worth pursuing.

        Declines once discovery has spent ``DISCOVER_FAILURE_RETRY_LIMIT`` on
        either counter: rounds that came back empty, or rounds that could not
        run at all. Without that the lane would always answer "asked again",
        the pump would return on every tick, and neither the local-exploration
        pivot below it nor ``framework_agent_phase_done`` could be reached --
        the source arm would have no way to report itself plateaued.

        Args:
            reason: Why discovery is being requested; carried into the mandate.

        Returns:
            True when a discovery task was created or is already in flight;
            False once either budget says there is nothing more to get.
        """
        from ..framework import client as _fa_client

        state = self.shared_state
        limit = int(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT)
        empties = int(getattr(state, "framework_agent_empty_discoveries", 0) or 0)
        failures = int(getattr(state, "framework_agent_discover_failures", 0) or 0)
        if empties >= limit or failures >= limit:
            return False
        if await self._candidate_discovery_inflight():
            return True
        gap, keywords = self._compose_framework_local_explore_gap()
        framework = str(getattr(state, "framework", "") or "").strip().lower()
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
            "discover_repo_urls": self._framework_agent_discover_repo_urls(framework),
            "tried_refs": self._framework_tried_refs(),
        }
        await self._warm_specialist_params(params)
        lanes, ttl = self._framework_authoring_lanes_ttl(params, base_ttl_sec=1800)
        await self.tasks.create_or_return_existing(
            kind="specialist",
            params=params,
            # The round count is part of the key: the registry returns the row
            # a key already names, so a fixed key would re-fetch the finished
            # first attempt and neither streak could advance.
            idempotency_key=f"candidate-discovery:{reason}{self._cycle_idem_suffix()}:r{empties + failures}",
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
            side_effects=["writes_results"],
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
        """Harvest a discovery specialist's candidates into a batch.

        No-op unless the task carries the ``candidate_discovery`` marker.
        Entries are appended in the order the specialist ranked them.

        Args:
            task: The completed specialist task.
            done_payload: Its ``specialist_done`` payload.
            run_error: The task's error, if it did not complete. A round that
                failed reports nothing about what is out there, so it counts
                against its own budget rather than the empty-result streak.
        """
        params = getattr(task, "params", None) or {}
        if not bool(params.get("candidate_discovery")):
            return
        state = self.shared_state
        if run_error:
            failures = int(getattr(state, "framework_agent_discover_failures", 0) or 0) + 1
            state.framework_agent_discover_failures = failures
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
        # A round that ran is proof the lane works, whatever it came back with.
        state.framework_agent_discover_failures = 0
        if not candidates:
            empties = int(getattr(state, "framework_agent_empty_discoveries", 0) or 0) + 1
            state.framework_agent_empty_discoveries = empties
            log.info("FRAMEWORK: discovery returned no usable candidates (streak=%d)", empties)
        else:
            state.framework_agent_empty_discoveries = 0
            batches = getattr(state, "framework_agent_batches", None)
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
        """Map discovery ``proposal_set`` entries to candidate rows.

        Entries verdicted ``already_present`` or ``not_applicable`` are dropped
        rather than dispatched, as are duplicates of known or processed
        candidates.

        Args:
            proposals: The specialist's ``proposal_set``.

        Returns:
            Candidate rows in the order the specialist returned them.
        """
        out: list[dict[str, Any]] = []
        if not isinstance(proposals, list):
            return out
        known = self._framework_known_candidate_ids()
        processed = self._framework_processed_candidate_keys()
        for entry in proposals:
            if not isinstance(entry, dict):
                continue
            verdict = str(entry.get("verdict") or "").strip().lower()
            if verdict in {"already_present", "not_applicable"}:
                continue
            pr_url = str(entry.get("pr_url") or entry.get("url") or "").strip()
            head_sha = str(entry.get("head_sha") or "").strip()
            if not pr_url and not head_sha:
                continue
            cand: dict[str, Any] = {
                "pr_url": pr_url,
                "url": pr_url,
                "head_sha": head_sha,
                "title": str(entry.get("title") or "").strip(),
                "diff_url": str(entry.get("diff_url") or "").strip(),
                "repo": str(entry.get("repo") or "").strip(),
                "pr_number": entry.get("pr_number"),
                "ref": str(entry.get("ref") or "").strip(),
                "framework": str(entry.get("framework") or "").strip().lower(),
                "changed_files": entry.get("changed_files") or [],
                "gap_canonical_id": str(entry.get("gap_canonical_id") or "").strip(),
                "gap_keywords": entry.get("gap_keywords") or [],
                "route": str(entry.get("route") or "author_via_specialist").strip(),
                "audit": {
                    "verdict": verdict,
                    "reason": str(entry.get("reason") or "").strip(),
                    "recommended_next_step": str(entry.get("route") or "").strip(),
                },
            }
            key = self._framework_candidate_key(cand)
            if not key or key in known or key in processed:
                continue
            known.add(key)
            out.append(cand)
        return out
