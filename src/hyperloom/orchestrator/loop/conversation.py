# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import json
import time
from typing import Any
from ..phases import machine_state as _phase_state
from ..roles.base import BackendTurnResult
from ..bus.message_bus import Message
from ..trace.conversation_trace import ConversationRecord, append_conversation

from .coordinator import (
    _format_inbox_event,
)
import logging as _logging
log = _logging.getLogger(__name__)


class ConversationCollaborator:
    """Extracted collaborator; delegates unknown attrs to its Coordinator."""

    def __init__(self, coordinator) -> None:
        self._coord = coordinator

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_coord"), name)

    def _orchestration_conversational(self) -> bool:
        """True when the orchestration backend runs in persistent-conversation mode.

        Returns:
            ``True`` if the orchestration backend exposes a truthy
            ``conversational`` attribute, else ``False``.
        """
        backend = self.backends.get("orchestration")
        return bool(getattr(backend, "conversational", False))

    def _reset_orchestration_conversation(self) -> None:
        """Force the next orchestration turn to re-seed a fresh conversation."""
        backend = self.backends.get("orchestration")
        reset = getattr(backend, "reset_conversation", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001
                log.exception("Coordinator: orchestration reset_conversation failed")
        self._coord._orchestration_seeded = False

    def _conversation_progress_signal(self) -> dict[str, Any]:
        """Compute the no-progress circuit-breaker signal.

        Returns:
            A dict with ``ticks_without_progress``, ``threshold``,
            ``severity`` ("ok" or "high"), and ``last_progress_tick``;
            progress is detected from stack growth, validated gain,
            current-best signature, or phase change.
        """
        state = self.shared_state
        cur_tick = int(getattr(state, "tick", 0) or 0)
        try:
            stack_len = len(state.optimization_stack or [])
        except Exception:  # noqa: BLE001
            stack_len = 0
        validated_gain = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
        cb = getattr(state, "current_best", None)
        try:
            current_best_sig = json.dumps(cb, sort_keys=True, default=str) if cb else ""
        except Exception:  # noqa: BLE001
            current_best_sig = str(cb)
        phase = str(getattr(state, "phase", "") or "")

        marker = self._progress_marker
        if not marker:
            self._coord._progress_marker = {
                "stack_len": stack_len,
                "validated_gain": validated_gain,
                "current_best_sig": current_best_sig,
                "phase": phase,
                "last_progress_tick": cur_tick,
            }
            return {
                "ticks_without_progress": 0,
                "threshold": self._no_progress_threshold,
                "severity": "ok",
                "last_progress_tick": cur_tick,
            }

        progressed = (
            stack_len > int(marker.get("stack_len", 0))
            or validated_gain > float(marker.get("validated_gain", 0.0)) + 1e-9
            or current_best_sig != marker.get("current_best_sig", "")
            or phase != marker.get("phase", "")
        )
        if progressed:
            marker["last_progress_tick"] = cur_tick
        marker["stack_len"] = stack_len
        marker["validated_gain"] = validated_gain
        marker["current_best_sig"] = current_best_sig
        marker["phase"] = phase

        gap = max(0, cur_tick - int(marker.get("last_progress_tick", cur_tick)))
        severity = "high" if gap >= self._no_progress_threshold else "ok"
        return {
            "ticks_without_progress": gap,
            "threshold": self._no_progress_threshold,
            "severity": severity,
            "last_progress_tick": int(marker.get("last_progress_tick", cur_tick)),
        }

    def _attach_orchestration_context_tools(self) -> None:
        """Bind a read-only ContextProvider to the orchestration backend (no-op without setter)."""
        backend = self.backends.get("orchestration")
        setter = getattr(backend, "set_context_provider", None)
        if setter is None:
            return
        try:
            from ..roles.mcp_context_tools import ContextProvider

            provider = ContextProvider(
                shared_state=self.shared_state,
                inbox_reader=self._context_inbox_reader,
                analysis_reader=self._context_analysis_reader,
                recent_outcomes_reader=self._context_recent_outcomes_reader,
                action_runner=self._run_action_now_sync,
            )
            setter(provider)
        except Exception:  # noqa: BLE001 — context pull is best-effort
            log.exception("Coordinator: failed to attach orchestration context tools")

    def _context_inbox_reader(self, since_seq: int = 0) -> str:
        """Synchronous projection of the orchestration inbox tail (sync SQLite path).

        Args:
            since_seq: Only events with a sequence number greater than this are
                included; defaults to ``0`` (all events).

        Returns:
            A newline-joined rendering of the last 40 matching inbox events, or
            a placeholder string when none are available.
        """
        try:
            rows = self.bus.db.fetchall_sync(
                "SELECT * FROM events WHERE seq > ? AND (to_agent = ? OR to_agent = '*') ORDER BY seq ASC",
                (int(since_seq or 0), "orchestration"),
            )
        except Exception as exc:  # noqa: BLE001
            return f"(inbox unavailable: {exc!r})"
        if not rows:
            return "(no inbox events)"

        msgs = [Message.from_row(r) for r in rows]
        lines = [_format_inbox_event(m) for m in msgs[-40:]]
        return "\n".join(lines)

    def _context_recent_outcomes_reader(self, top_k: int = 8) -> str:
        """Synchronous projection of recent action outcomes.

        Args:
            top_k: Number of recent outcome events to project; clamped to the
                range 1..50 (defaults to 8).

        Returns:
            A newline-joined, chronological (newest-last) rendering of recent
            delegated_result/review_verdict events, or a placeholder string.
        """
        try:
            k = max(1, min(int(top_k or 8), 50))
        except (TypeError, ValueError):
            k = 8
        try:
            rows = self.bus.db.fetchall_sync(
                "SELECT * FROM events WHERE topic IN ('delegated_result', 'review_verdict') ORDER BY seq DESC LIMIT ?",
                (k,),
            )
        except Exception as exc:  # noqa: BLE001
            return f"(recent outcomes unavailable: {exc!r})"
        if not rows:
            return "(no recent outcomes)"

        # Flip newest-first query to newest-last for chronological reading.
        # Flip newest-first query to newest-last for chronological reading.
        msgs = [Message.from_row(r) for r in rows][::-1]
        lines = ["=== Recent action outcomes (newest last) ==="]
        lines.extend(_format_inbox_event(m) for m in msgs)
        return "\n".join(lines)

    def _context_analysis_reader(self) -> str:
        """Return the latest TraceLens analysis.md snapshot text.

        Returns:
            The formatted analysis.md snapshot, the text read from the recorded
            ``analysis_md_path``, or a placeholder when none is available.
        """
        try:
            blob = self.shared_state._format_analysis_md_full()
            if blob and blob.strip():
                return blob
        except Exception:  # noqa: BLE001 — fall through to path read
            log.exception("Coordinator: _format_analysis_md_full failed")
        # Fallback: read the path recorded on last_trace_analyze.
        lta = getattr(self.shared_state, "last_trace_analyze", {}) or {}
        path = str(lta.get("analysis_md_path") or "")
        if path:
            try:
                from pathlib import Path as _Path

                return _Path(path).read_text(encoding="utf-8")
            except OSError as exc:
                return f"(analysis.md unreadable at {path}: {exc!r})"
        return "(no analysis.md snapshot yet)"

    def _record_reactor_conversation(
        self,
        agent_name: str,
        result: BackendTurnResult,
    ) -> None:
        """Append one ``conversations.jsonl`` row for a reactor turn.

        Persists the full (redacted) prompt + completion from the backend
        ``metadata`` (``prompt`` / ``response``). Only rows that carry
        conversation text are written. Best-effort: any failure degrades to a
        logged warning rather than breaking the tick loop.

        Args:
            agent_name: The reactor role; doubles as trace component and role.
            result: The backend turn result whose metadata carries the redacted
                prompt/response text.
        """
        try:
            metadata = result.metadata or {}
            prompt = metadata.get("prompt")
            response = metadata.get("response")
            if not prompt and not response:
                return
            record = ConversationRecord(
                session_id=self.session_dir.name,
                component=agent_name,
                role=agent_name,
                tick=int(self.shared_state.tick or 0),
                phase=(self.shared_state.phase or "") or None,
                model=metadata.get("model"),
                prompt=prompt or "",
                response=response or "",
            )
            append_conversation(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break the loop
            log.debug(
                "full-trace: reactor conversation append failed for %s",
                agent_name,
                exc_info=True,
            )

    async def _compose_prompt(self, agent_name: str) -> str:
        """Compose the orchestration prompt: SharedState summary + inbox tail (with canonical msg_id per inbox row).

        Args:
            agent_name: The agent role to compose the per-tick prompt for;
                selects which advisory/telemetry sections are included.

        Returns:
            The assembled prompt string for this agent's reactor turn.
        """
        sections: list[str] = []

        # SESSION_DIR contract — literal path for every agent.
        sections.append(f"SESSION_DIR={self.session_dir}")

        # Per-tick phase block for every agent, high in the prompt.
        try:
            phase_block = self.shared_state.to_phase_status_summary(
                budget_pct=self._phase_budget_pct,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: phase status summary failed")
            phase_block = ""
        if phase_block:
            sections.append("=== Phase ===")
            sections.append(phase_block)

        # Conversational delta gating: first turn gets full SEED, later turns thin DELTA.
        push_full = True
        if agent_name == "orchestration":
            push_full = not self._orchestration_conversational() or not self._orchestration_seeded
            if self._orchestration_conversational():
                log.info(
                    "orchestration prompt mode=%s seeded=%s tick=%s",
                    "SEED" if push_full else "DELTA",
                    self._orchestration_seeded,
                    getattr(self.shared_state, "tick", 0),
                )

        # On a full SEED push, inject recovered working memory.
        if (
            agent_name == "orchestration"
            and push_full
            and self._orchestration_conversational()
            and self._orchestration_seed_memory
        ):
            sections.append(self._orchestration_seed_memory)

        if agent_name == "orchestration":
            sections.append("=== Mission progress ===")
            sections.append(self.shared_state.to_mission_summary())
            if push_full:
                try:
                    cycle_strategy_block = self._cycle_strategy_seed_block()
                except Exception:  # noqa: BLE001 — advisory only
                    log.exception("Coordinator: cycle strategy seed render failed")
                    cycle_strategy_block = ""
                if cycle_strategy_block:
                    sections.append(cycle_strategy_block)
            if self._run_deadline is not None and self._run_started_monotonic is not None:
                remaining_min = max(
                    0.0,
                    (self._run_deadline - time.monotonic()) / 60.0,
                )
                elapsed_min = (time.monotonic() - self._run_started_monotonic) / 60.0
                budget_min = self.shared_state.max_minutes or 0
                sections.append("=== Time budget ===")
                sections.append(
                    f"elapsed={elapsed_min:.1f}min  remaining={remaining_min:.1f}min  "
                    f"budget={budget_min}min  "
                    f"closing_phase={self.shared_state.closing_phase}"
                )
                if remaining_min <= 5.0 and not self.shared_state.closing_phase:
                    sections.append(
                        "WARNING: < 5 min remaining. Prefer `report` next; new "
                        "`explore` rounds (which inline the stack rebench) "
                        "will likely be cut by the deadline."
                    )

        # Time budget for Robustness — fires deadline_imminent → delegate(report) wind-down.
        if agent_name == "robustness" and self._run_deadline is not None and self._run_started_monotonic is not None:
            remaining_min = max(
                0.0,
                (self._run_deadline - time.monotonic()) / 60.0,
            )
            elapsed_min = (time.monotonic() - self._run_started_monotonic) / 60.0
            budget_min = self.shared_state.max_minutes or 0
            sections.append("=== Time budget ===")
            sections.append(
                f"elapsed={elapsed_min:.1f}min  remaining={remaining_min:.1f}min  "
                f"budget={budget_min}min  "
                f"closing_phase={self.shared_state.closing_phase}"
            )

        # Shared session state; omitted on orchestration DELTA turns.
        if push_full:
            sections.append("=== Shared session state ===")
            sections.append(self.shared_state.to_prompt_summary())
        if agent_name == "orchestration":
            # target_gap_pct is the gain still needed for --target-gain.
            obj = getattr(self, "_current_objective", None)
            obj_kind = getattr(obj, "kind", "") if obj is not None else ""
            if obj_kind == "gain_pct":
                target_val = float(getattr(obj, "value", 0.0) or 0.0)
                self.shared_state.target_gap_pct = max(
                    0.0,
                    target_val - float(self.shared_state.cumulative_gain or 0.0),
                )
            else:
                self.shared_state.target_gap_pct = 0.0
            # Advisory/ledger blocks below are part of the full SEED push only.
            if push_full:
                denial_summary = self.shared_state.to_policy_denial_summary(top_k=6)
                if denial_summary:
                    sections.append(denial_summary)

        # Cortex T0 warm-start snapshot + structured gaps[] ledger.
        if agent_name == "orchestration" and push_full:
            try:
                warm_block = self.shared_state.to_warm_start_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: warm_start_summary failed")
                warm_block = ""
            if warm_block:
                sections.append("=== Warm start (Cortex T0) ===")
                sections.append(warm_block)
            try:
                gaps_block = self.shared_state.to_gaps_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: gaps_summary failed")
                gaps_block = ""
            if gaps_block:
                sections.append("=== Current gaps ===")
                sections.append(gaps_block)
            try:
                from ..knowledge import research_hints as _research_hints

                hints_block = _research_hints.summarise_for_prompt(
                    self.session_dir,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: research_hints summary failed")
                hints_block = ""
            if hints_block:
                sections.append("=== Research hints (advisory) ===")
                sections.append(hints_block)
            try:
                gap_block = self._target_gap_advisory_block()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: target gap advisory failed")
                gap_block = ""
            if gap_block:
                sections.append("=== External target gap (advisory) ===")
                sections.append(gap_block)
            # Advisory multi-model proposal scores (ProposalScorer); not a ranking directive.
            try:
                scores_block = self.shared_state.to_proposal_scores_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: proposal_scores_summary failed")
                scores_block = ""
            if scores_block:
                sections.append("=== Specialist proposal scores (advisory) ===")
                sections.append(scores_block)
            # Priors-match: recently proposed variants aligning with research hints/external gap (advisory only).
            try:
                priors_block = self._priors_match_advisory_block()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: priors-match advisory failed")
                priors_block = ""
            if priors_block:
                sections.append("=== Priors-match (advisory ordering) ===")
                sections.append(priors_block)

            # Surface the intervention-mix ledger (config vs code_patch counts) as neutral telemetry.
            try:
                mix_block = self.shared_state.to_intervention_mix_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: intervention_mix_summary failed")
                mix_block = ""
            if mix_block:
                sections.append("=== Intervention mix (telemetry) ===")
                sections.append(mix_block)

            try:
                plateau_block = self._plateau_advisory_block()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: plateau advisory failed")
                plateau_block = ""
            if plateau_block:
                sections.append("=== Plateau advisory ===")
                sections.append(plateau_block)

            # On a plateau, surface candidate directions (advisory).
            if plateau_block:
                try:
                    from ..knowledge import trajectory_reviewer as _trajectory_reviewer

                    trajectory_block = _trajectory_reviewer.build_trajectory_digest(
                        self.session_dir,
                        self.shared_state,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception("Coordinator: trajectory review failed")
                    trajectory_block = ""
                if trajectory_block:
                    sections.append("=== Trajectory review (advisory) ===")
                    sections.append(trajectory_block)

            # Cyclic bottleneck-redirect advisory (next-cycle re-targeting).
            try:
                redirect_block = self._bottleneck_redirect_advisory_block()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: bottleneck redirect advisory failed")
                redirect_block = ""
            if redirect_block:
                sections.append("=== Bottleneck redirect (advisory) ===")
                sections.append(redirect_block)

            # Decaying acceptance bar + prior variants now re-testable under it.
            try:
                accept_block = self._acceptance_threshold_advisory_block()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: acceptance threshold advisory failed")
                accept_block = ""
            if accept_block:
                sections.append("=== Acceptance threshold (advisory) ===")
                sections.append(accept_block)

        # Conversational DELTA turn: tell the agent verbose state was not re-pushed + how to pull it.
        if agent_name == "orchestration" and not push_full:
            sections.append("=== Context (pull on demand) ===")
            sections.append(
                "This is a continuation of our ongoing conversation; the "
                "full session state was NOT re-pasted. The Phase, Mission "
                "progress, Time budget, and new inbox events above are the "
                "delta since your last turn. Pull anything else you need "
                "with the read-only context tools: get_shared_state, "
                "get_gaps, get_warm_start, get_proposal_scores, "
                "get_intervention_mix, why_denied, show_analysis_md, "
                "get_inbox. Reason from your own running plan; do not "
                "re-derive it from scratch."
            )

        # Robustness gets phase budget telemetry + specialist health for medium-severity alerts.
        if agent_name == "robustness":
            try:
                budget_block = self.shared_state.to_phase_budget_telemetry(
                    budget_pct=self._phase_budget_pct,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: phase budget telemetry failed")
                budget_block = ""
            if budget_block:
                sections.append("=== Phase budget telemetry ===")
                sections.append(budget_block)
            try:
                stale = await self._scan_stale_specialists()
                running = await self.tasks.running()
                specialist_running = sum(1 for t in (running or []) if (t.kind or "").strip() == "specialist")
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: specialist health scan failed")
                stale, specialist_running = [], 0
            stale_lines = [f"  - task_id={row['task_id']} running_sec={int(row['running_seconds'])}" for row in stale]
            sections.append("=== Specialist health ===")
            sections.append(
                f"running={specialist_running} stale={len(stale)} stale_threshold_sec={int(self._specialist_stale_sec)}"
            )
            if stale_lines:
                sections.append("stale specialists (consider kill_task):")
                sections.extend(stale_lines)

            # Conversation no-progress circuit-breaker; Robustness is the external safety net.
            try:
                progress = self._conversation_progress_signal()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: conversation progress signal failed")
                progress = {}
            if progress:
                sections.append("=== Conversation progress ===")
                sections.append(
                    f"ticks_without_progress={progress.get('ticks_without_progress', 0)} "
                    f"threshold={progress.get('threshold', 0)} "
                    f"severity={progress.get('severity', 'ok')} "
                    f"last_progress_tick={progress.get('last_progress_tick', 0)}"
                )
                if progress.get("severity") == "high":
                    sections.append(
                        "WARNING: no observable progress (no new KEEP / stack "
                        "growth / validated-gain bump / phase advance) for "
                        f">= {progress.get('threshold', 0)} ticks. The "
                        "Orchestration conversation may be stuck. Consider "
                        "escalating: signal a wind-down (delegate `report`) or "
                        "raise a high-severity no_progress observation so the "
                        "operator can intervene."
                    )

        # 2. Inbox tail since this agent's last cursor.
        cursor = await self.cursors.load(agent_name)
        msgs = await self.bus.replay_for(agent_name, after_seq=cursor.last_processed_seq)
        rendered = list(msgs[-20:])
        # Durable at-least-once-until-decided delivery of proposals to the
        # Critic: the inbox tail is lossy, so re-present every still-undecided
        # proposal from the durable ``pending_proposals`` registry.
        if agent_name == "critic":
            rendered = await self._augment_critic_inbox_with_pending(rendered)
        if rendered:
            sections.append(f"=== Inbox for {agent_name} (newest last) ===")
            for m in rendered:
                sections.append(f"  {_format_inbox_event(m)}")
        else:
            sections.append(f"=== Inbox for {agent_name} ===")
            sections.append("(no new messages)")

        return "\n".join(sections)

    async def _augment_critic_inbox_with_pending(
        self, rendered: list["Message"]
    ) -> list["Message"]:
        """Ensure every undecided proposal awaiting a Critic verdict is present.

        The rendered tail can drop proposals that scrolled past the capped
        window. Source the review set from the durable ``pending_proposals``
        registry and merge any missing proposal messages into the rendered
        window (deduped by ``msg_id``, re-sorted by ``seq`` so "newest last"
        holds).

        Args:
            rendered: The tail-capped messages already selected for the inbox.

        Returns:
            The rendered list augmented with any undecided proposal messages
            not already present; unchanged on any error (best-effort).
        """
        try:
            pending = [
                p
                for p in self.state.pending_proposals.values()
                if not getattr(p, "decided", False)
            ]
        except Exception:  # noqa: BLE001 — never break prompt composition
            return rendered
        if not pending:
            return rendered
        seen = {getattr(m, "msg_id", None) for m in rendered}
        extra: list["Message"] = []
        for p in pending:
            pid = str(getattr(p, "proposal_msg_id", "") or "")
            if not pid or pid in seen:
                continue
            try:
                pm = await self.bus.lookup_by_id(pid)
            except Exception:  # noqa: BLE001 — defensive
                pm = None
            if pm is not None:
                extra.append(pm)
                seen.add(pid)
        if not extra:
            return rendered
        merged = list(rendered) + extra
        merged.sort(key=lambda m: int(getattr(m, "seq", 0) or 0))
        return merged

    async def _load_system_prompt(self, agent_name: str) -> str:
        """Load the system prompt for an agent, honoring overrides.

        Args:
            agent_name: Name of the agent/role whose prompt to load.

        Returns:
            The override prompt if configured, the role's prompt file
            contents, or a placeholder string when none exists.
        """
        override = getattr(self, "system_prompt_overrides", {}).get(agent_name)
        if override is not None:
            return override
        role = self.role_registry[agent_name]
        try:
            return role.load_system_prompt()
        except FileNotFoundError:
            return f"(no system prompt for {agent_name})"

    # Advisory prompt blocks (folded in from the former AdvisoryCollaborator).
    # Consumed by :meth:`_compose_prompt` above and by the phase handlers via the
    # coordinator's bare-name ``_DELEGATED`` resolution.
    def _plateau_advisory_block(self) -> str:
        """Render the plateau-judgment advisory block (EXPLORE/KERNEL/FRAMEWORK). Returns "" when no plateau signal is active.

        KERNEL / FRAMEWORK plateaus are advisory only (never auto-exit the
        phase). An EXPLORE plateau is advisory in non-cyclic mode, but in cyclic
        mode (default) it deterministically advances EXPLORE → KERNEL_AGENT via
        ``explore_no_more_leverage`` (a non-terminal lever switch); the rendered
        footer states which regime is active.

        Returns:
            The rendered plateau advisory text, or ``""`` when no plateau
            signal is active for the current phase.
        """
        state = self.shared_state
        phase = (getattr(state, "phase", "") or "").strip().upper()
        overrides = getattr(state, "plateau_overrides", None) or {}
        if not isinstance(overrides, dict):
            overrides = {}
        lines: list[str] = []
        if phase == _phase_state.PHASE_EXPLORE:
            triggered, evidence = _phase_state.compute_plateau_explore(
                state,
                lookback=int(
                    overrides.get(
                        "explore_lookback",
                        _phase_state.DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
                    )
                ),
                keep_gain_threshold_pct=float(
                    overrides.get(
                        "explore_keep_gain_pct",
                        _phase_state.DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
                    )
                ),
                empty_streak_threshold=int(
                    overrides.get(
                        "explore_empty_streak",
                        _phase_state.DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
                    )
                ),
            )
            if triggered:
                lines.append("EXPLORE plateau detected: low recent KEEP gain plus specialist empty streak.")
                lines.append(
                    "  recent_keep_gain_pct="
                    f"{evidence.get('recent_keep_gain_pct', 0.0)} "
                    f"threshold={evidence.get('keep_gain_threshold_pct', 0.0)} "
                    f"empty_streak={evidence.get('empty_streak', 0)} "
                    f"streak_threshold={evidence.get('empty_streak_threshold', 0)}"
                )
        elif phase == _phase_state.PHASE_KERNEL_AGENT:
            triggered, evidence = _phase_state.compute_plateau_kernel(
                state,
                lookback=int(
                    overrides.get(
                        "kernel_lookback",
                        _phase_state.DEFAULT_PLATEAU_KERNEL_LOOKBACK,
                    )
                ),
                revert_streak_threshold=int(
                    overrides.get(
                        "kernel_revert_streak",
                        _phase_state.DEFAULT_PLATEAU_KERNEL_REVERT_STREAK,
                    )
                ),
                keep_gain_threshold_pct=float(
                    overrides.get(
                        "kernel_keep_gain_pct",
                        _phase_state.DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT,
                    )
                ),
            )
            if triggered:
                lines.append("KERNEL_AGENT plateau detected: REVERT streak or low recent KEEP gain.")
                lines.append(
                    "  revert_streak="
                    f"{evidence.get('revert_streak', 0)} "
                    f"threshold={evidence.get('revert_streak_threshold', 0)} "
                    f"recent_keep_gain_pct={evidence.get('recent_keep_gain_pct', 0.0)} "
                    f"keep_gain_threshold_pct={evidence.get('keep_gain_threshold_pct', 0.0)}"
                )
        elif phase == _phase_state.PHASE_FRAMEWORK_AGENT:
            triggered, evidence = _phase_state.compute_plateau_framework_agent(
                state,
                lookback=int(
                    overrides.get(
                        "framework_lookback",
                        _phase_state.DEFAULT_FRAMEWORK_PLATEAU_LOOKBACK,
                    )
                ),
                keep_gain_threshold_pct=float(
                    overrides.get(
                        "framework_keep_gain_pct",
                        _phase_state.DEFAULT_FRAMEWORK_PLATEAU_KEEP_GAIN_PCT,
                    )
                ),
            )
            if triggered:
                lines.append("FRAMEWORK_AGENT plateau detected: recent batches all below keep-gain threshold.")
                lines.append(
                    "  lookback="
                    f"{evidence.get('lookback', 0)} "
                    f"keep_gain_pct_threshold={evidence.get('keep_gain_pct_threshold', 0.0)} "
                    f"batch_max_gains={evidence.get('batch_max_gains', [])}"
                )
        if not lines:
            return ""
        if phase == _phase_state.PHASE_EXPLORE and _phase_state.is_cyclic_phases_enabled():
            lines.append(
                "Note: in cyclic mode a detected EXPLORE plateau "
                "deterministically advances EXPLORE → KERNEL_AGENT (non-terminal "
                "lever switch, reason=explore_no_more_leverage); it does not "
                "end the run. You may still request an earlier advance with an "
                "escalate_strategy_change hint, or keep exploring until the "
                "plateau/budget gate fires."
            )
        else:
            lines.append(
                "Phase advance is driven only by hard limits (IR-6 force-exit, "
                "phase budget, terminal stop_reason) or explicit "
                "escalate_strategy_change hints; this block is informational."
            )
        return "\n".join(lines)

    def _dominant_roofline_direction(self) -> tuple[str, float]:
        """Return ``(direction, pct)`` for the most-saturated roofline direction
        in the latest snapshot; ``("", 0.0)`` when no snapshot is available.

        Returns:
            A ``(direction, pct)`` tuple for the dominant roofline direction, or
            ``("", 0.0)`` when no snapshot exists.
        """
        from ..kernel.roofline_snapshot import dominant_direction

        snaps = getattr(self.shared_state, "roofline_snapshots", None) or []
        if not snaps or not isinstance(snaps[-1], dict):
            return "", 0.0
        return dominant_direction(snaps[-1])

    def _bottleneck_redirect_advisory_block(self) -> str:
        """Render the R3 cyclic bottleneck-redirect advisory (EXPLORE only).

        Active only in cyclic mode when a prior cycle's plateau flagged
        ``pending_bottleneck_switch``. Names the bottleneck we plateaued on, the
        current dominant roofline direction, and a suggested specialist domain so
        Orchestration redirects the new cycle's dispatch. Advisory, never gates.

        Returns:
            The rendered bottleneck-redirect advisory text, or ``""`` when not
            applicable.
        """
        state = self.shared_state
        if not _phase_state.is_cyclic_phases_enabled():
            return ""
        if (getattr(state, "phase", "") or "").strip().upper() != _phase_state.PHASE_EXPLORE:
            return ""
        sat = getattr(state, "saturated_directions", {}) or {}
        saturated = {
            str(k): v
            for k, v in (sat.items() if isinstance(sat, dict) else [])
            if isinstance(v, dict) and bool(v.get("saturated"))
        }
        rows = [r for r in (getattr(state, "cycle_strategy_log", []) or []) if isinstance(r, dict)]
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        focus_row = next((r for r in reversed(rows) if int(r.get("cycle", -1) or -1) == cycle), {})
        has_switch = bool(getattr(state, "pending_bottleneck_switch", False))
        if not has_switch and not saturated and not focus_row:
            return ""
        prev = str(getattr(state, "last_cycle_bottleneck", "") or "")
        cur_top = state.current_top_bottleneck()
        direction, pct = self._dominant_roofline_direction()
        lines: list[str] = []
        if has_switch:
            lines.append(
                "The previous macro-cycle plateaued; redirect this cycle to a "
                "different bottleneck instead of re-mining the exhausted one."
            )
        if saturated:
            lines.append("Roofline ceiling signal: one or more lever families are saturated; deprioritize them.")
            for domain, row in sorted(saturated.items()):
                lines.append(
                    f"  saturated_domain={domain} direction={row.get('direction')} "
                    f"within={row.get('within_pct')}% threshold={row.get('threshold_pct')}%"
                )
        if focus_row:
            lines.append(
                f"  suggested_cycle_focus={focus_row.get('focus')} "
                f"score={focus_row.get('score')} rationale={focus_row.get('rationale')}"
            )
        if prev:
            lines.append(f"  plateaued_bottleneck={prev} (avoid re-targeting)")
        if cur_top:
            lines.append(f"  current_top_bottleneck={cur_top}")
        shift = getattr(state, "bottleneck_shift", {}) or {}
        if isinstance(shift, dict) and (shift.get("from") or shift.get("to")):
            lines.append(
                f"  bottleneck_shift: {shift.get('from') or 'unknown'} → {shift.get('to') or 'unknown'} "
                f"(within_delta={shift.get('within_delta')} gap_delta={shift.get('gap_delta')})"
            )
        if direction:
            from ..kernel.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

            hint = BOTTLENECK_DOMAIN_HINTS.get(direction)
            if hint:
                lines.append(
                    f"  dominant_direction={direction} ({pct:.1f}%) → "
                    f"suggested specialist domain={hint[0]} tag={hint[1]}"
                )
            else:
                lines.append(f"  dominant_direction={direction} ({pct:.1f}%)")
        lines.append(
            f"  macro_cycle={cycle}; KEEP'd variants stay de-duped permanently, "
            "but prior sub-threshold variants whose measured gain now meets the "
            "decayed KEEP bar are unblocked for re-test."
        )
        lines.append("Advisory only: pick the domain/tag yourself; this nudges focus, it does not gate dispatch.")
        return "\n".join(lines)

    def _acceptance_threshold_advisory_block(self) -> str:
        """Render the current decaying acceptance bar + re-testable prior variants.

        Active only in cyclic mode after at least one macro-cycle (when the bar
        has decayed below the first-cycle default). Surfaces the current KEEP /
        stack-stable thresholds and lists prior sub-threshold variants whose
        measured gain now meets the decayed bar (unblocked for re-test) plus a
        few still below it (reference only). Advisory; never gates dispatch.

        Returns:
            The rendered acceptance-threshold advisory text, or ``""`` when not
            applicable (non-cyclic mode or first cycle).
        """
        state = self.shared_state
        keep = self._decaying_keep_threshold_pct()
        if keep is None:
            return ""
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        if cycle < 1:
            return ""
        stable = keep / 2.0
        unlockable = {"REVERT", "KEEP_UNSTABLE", "no_promote"}
        search = getattr(state, "explore_search", None) or {}
        entries: list[dict[str, Any]] = []
        if isinstance(search, dict):
            tested = search.get("tested") or {}
            if isinstance(tested, dict):
                entries.extend(v for v in tested.values() if isinstance(v, dict))
            rejected = search.get("rejected") or []
            if isinstance(rejected, list):
                entries.extend(v for v in rejected if isinstance(v, dict))
        now_unblocked: list[tuple[str, float]] = []
        still_blocked: list[tuple[str, float]] = []
        for e in entries:
            if str(e.get("outcome") or "") not in unlockable:
                continue
            try:
                g = float(e.get("gain_pct"))
            except (TypeError, ValueError):
                continue
            name = str(e.get("name") or e.get("fingerprint") or "")[:48]
            (now_unblocked if g >= keep else still_blocked).append((name, g))
        lines: list[str] = [
            f"Current acceptance bar (macro_cycle={cycle}): KEEP>={keep:.2f}% stack_stable>={stable:.2f}%.",
            "KEEP'd variants stay de-duped permanently; prior sub-threshold "
            "variants are de-duped only while below the current KEEP bar.",
        ]
        if now_unblocked:
            now_unblocked.sort(key=lambda p: p[1], reverse=True)
            lines.append("Re-testable now (prior gain now clears the bar; re-propose if still relevant):")
            for name, g in now_unblocked[:8]:
                lines.append(f"  {name}: prior gain {g:+.2f}% >= {keep:.2f}%")
        if still_blocked:
            still_blocked.sort(key=lambda p: p[1], reverse=True)
            lines.append("Still below the bar (reference only, not re-tested):")
            for name, g in still_blocked[:5]:
                lines.append(f"  {name}: prior gain {g:+.2f}% < {keep:.2f}%")
        return "\n".join(lines)

    def _target_gap_advisory_block(self) -> str:
        """Build the advisory "External target gap" prompt block (current-best vs competitor target; never gates).

        Returns:
            The rendered external-target-gap advisory text, or ``""`` when
            disabled or no competitor target/current-best is available.
        """
        state = self.shared_state
        if not bool(getattr(state, "target_advisory_enabled", True)):
            return ""
        from ..knowledge import research_hints as _research_hints

        target = _research_hints.load_competitor_target(self.session_dir)
        if not target:
            return ""
        best = getattr(state, "current_best", None)
        if not isinstance(best, dict):
            return ""
        tput = best.get("tput")
        tpot = best.get("tpot_mean_ms")
        tp = int(getattr(state, "tp", 0) or 0)
        our_tput_per_gpu = float(tput) / tp if isinstance(tput, (int, float)) and tput > 0 and tp > 0 else None
        our_tpot_ms = float(tpot) if isinstance(tpot, (int, float)) and tpot > 0 else None
        conc = int(getattr(state, "conc", 0) or 0) or None
        gap = _research_hints.gap_analysis(
            target,
            our_tput_per_gpu=our_tput_per_gpu,
            our_tpot_ms=our_tpot_ms,
            conc=conc,
        )
        return _research_hints.full_gap_summary(gap)

    def _current_primary_gap(self) -> str | None:
        """Resolve the dominant external gap direction ('latency'/'throughput') from the competitor target, or None when advisory is off / no target. Fail-soft.

        Returns:
            The primary gap direction string, or ``None`` when the advisory is
            off, no target exists, or analysis fails.
        """
        state = self.shared_state
        if not bool(getattr(state, "target_advisory_enabled", True)):
            return None
        try:
            from ..knowledge import research_hints as _research_hints

            target = _research_hints.load_competitor_target(self.session_dir)
            if not target:
                return None
            best = getattr(state, "current_best", None)
            if not isinstance(best, dict):
                return None
            tput = best.get("tput")
            tpot = best.get("tpot_mean_ms")
            tp = int(getattr(state, "tp", 0) or 0)
            our_tput_per_gpu = float(tput) / tp if isinstance(tput, (int, float)) and tput > 0 and tp > 0 else None
            our_tpot_ms = float(tpot) if isinstance(tpot, (int, float)) and tpot > 0 else None
            conc = int(getattr(state, "conc", 0) or 0) or None
            gap = _research_hints.gap_analysis(
                target,
                our_tput_per_gpu=our_tput_per_gpu,
                our_tpot_ms=our_tpot_ms,
                conc=conc,
            )
        except Exception:  # noqa: BLE001 — defensive
            return None
        if not isinstance(gap, dict):
            return None
        return str(gap.get("primary_gap") or "").strip() or None

    def _recent_proposed_variants(
        self,
        *,
        max_rounds: int = 2,
    ) -> list[dict[str, Any]]:
        """Collect proposal_set rows from the most recent specialist rounds (deduped by name; fail-soft).

        Args:
            max_rounds: Number of most-recent specialist rounds to scan
                (default 2).

        Returns:
            A name-deduped list of proposal variant dicts.
        """
        rounds = [
            r
            for r in (getattr(self.shared_state, "specialist_rounds", []) or [])
            if isinstance(r, dict) and isinstance(r.get("proposal_set"), list)
        ]
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for r in rounds[-max_rounds:]:
            for variant in r.get("proposal_set") or []:
                if not isinstance(variant, dict):
                    continue
                name = str(variant.get("name") or "").strip()
                if name and name not in seen:
                    seen.add(name)
                    out.append(variant)
        return out

    def _priors_match_advisory_block(self) -> str:
        """Flag recently proposed variants aligning with proven priors / dominant external gap (advisory ordering, fail-soft).

        Returns:
            The rendered priors-match advisory text, or ``""`` when there are no
            recent variants or rendering fails.
        """
        try:
            from ..knowledge import research_hints as _research_hints

            variants = self._recent_proposed_variants()
            if not variants:
                return ""
            hints = _research_hints.load_hints(self.session_dir)
            primary_gap = self._current_primary_gap()
            return _research_hints.priors_match_summary(
                variants,
                hints,
                primary_gap=primary_gap,
            )
        except Exception:  # noqa: BLE001 — defensive
            return ""
