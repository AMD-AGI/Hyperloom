# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import json
import time
from typing import Any
from .backends.base import BackendTurnResult
from .message_bus import Message
from .trace.conversation_trace import ConversationRecord, append_conversation
from .coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_sglang_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _resolve_roofline_watermark_ratio,
    effective_closing_grace_sec,
    format_exc_brief,
    serialize_verdict_advisory,
)

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

    # Context-pull tools
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
            from .backends.mcp_context_tools import ContextProvider

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

        Persists the full (redacted) prompt + completion the backend put on
        ``metadata`` (``prompt`` / ``response``). Only rows that actually
        carry conversation text are written, so subprocess-backed reactors
        (critic / robustness) that don't surface text here don't emit empty
        rows — their conversation is captured by their own workdir artefacts.

        Best-effort: any failure degrades to a logged warning rather than
        breaking the tick loop.

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

        # 0. SESSION_DIR contract — literal path for every agent (pairs with PolicyGate path containment).
        sections.append(f"SESSION_DIR={self.session_dir}")

        # per-tick phase block for every agent, high in the prompt because R1 rejection is phase-driven.
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

        # 0a. Mission progress (Orchestration only), shown before the verbose dump.
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

        # On a full SEED push, inject recovered working memory so the agent re-anchors its plan.
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

        # 1. Shared session state — goal + progress context; omitted on orchestration DELTA turns.
        if push_full:
            sections.append("=== Shared session state ===")
            sections.append(self.shared_state.to_prompt_summary())
        if agent_name == "orchestration":
            # target_gap_pct is a fact (gain still needed for --target-gain); refresh to keep prompt current.
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
            # Advisory/ledger blocks below are part of the full SEED push; omitted on DELTA turns.
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
                from . import research_hints as _research_hints

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

            # On a plateau, review the whole lineage and surface candidate
            # directions (exhausted directions to avoid + under-exploited
            # bottleneck to push). Advisory; never gates phase advance.
            if plateau_block:
                try:
                    from . import trajectory_reviewer as _trajectory_reviewer

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

            # R3: cyclic bottleneck-redirect advisory (next-cycle re-targeting).
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
        if msgs:
            sections.append(f"=== Inbox for {agent_name} (newest last) ===")
            for m in msgs[-20:]:
                # Structured rendering for delegated_result/denial/verdict; compact dump otherwise.
                sections.append(f"  {_format_inbox_event(m)}")
        else:
            sections.append(f"=== Inbox for {agent_name} ===")
            sections.append("(no new messages)")

        return "\n".join(sections)

    async def _load_system_prompt(self, agent_name: str) -> str:
        """Load the system prompt for an agent, honoring overrides.

        Args:
            agent_name: Name of the agent/role whose prompt to load.

        Returns:
            The override prompt if configured, the role's prompt file
            contents, or a placeholder string when none exists.
        """
        # Demo/test override via self.system_prompt_overrides[agent_name].
        override = getattr(self, "system_prompt_overrides", {}).get(agent_name)
        if override is not None:
            return override
        role = self.role_registry[agent_name]
        try:
            return role.load_system_prompt()
        except FileNotFoundError:
            return f"(no system prompt for {agent_name})"
