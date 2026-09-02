# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations
import json
import time
from typing import Any
from ..phases import machine_state as _phase_state
from ..roles.base import BackendTurnResult
from ..roles.mcp_context_tools import CONTEXT_TOOL_NAMES as _CONTEXT_TOOL_NAMES
from ..bus.message_bus import Message
from ..trace.conversation_trace import ConversationRecord, append_conversation

from .coordinator import (
    _format_inbox_event,
)
from .coordinator_helpers import _parse_iso_unix
from ..state.task_registry import Task
from hyperloom.inference_optimizer.session.session_paths import runs_dir
import logging as _logging

log = _logging.getLogger(__name__)

# Per-variant failure lines expanded by get_recent_outcomes, and the total cap
# that keeps a wide top_k from flooding the turn.
_RECENT_OUTCOMES_VARIANT_ROWS = 12
_RECENT_OUTCOMES_LINE_CAP = 120


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

    def _orchestration_context_tools_mounted(self) -> bool:
        """True when the orchestration backend really exposes the pull tools.

        Returns:
            ``True`` when the backend reports the read-only context tools live;
            backends without them (and a failed MCP build) report ``False``.
        """
        return bool(getattr(self.backends.get("orchestration"), "context_tools_mounted", False))

    def _orchestration_needs_seed(self, system_prompt: str | None = None) -> bool:
        """True when the orchestration backend lost the history a delta assumes.

        Only the backend knows when the conversation underneath it was
        replaced — a session-scoped provider re-opens its thread on a re-scoped
        system prompt or after a turn that never landed. Backends that keep no
        conversation report nothing and the seeded flag alone decides.

        The answer has to describe the turn that is *about* to run: a re-scoped
        system prompt replaces the thread inside the turn, so a backend asked
        only about the thread as it stands would report history that this turn is
        going to discard. ``needs_seed_for`` answers for the pending prompt;
        ``needs_seed`` is the fallback for backends that cannot.

        Args:
            system_prompt: The system prompt this turn will carry, when known.

        Returns:
            ``True`` when the backend reports a conversation with no history.
        """
        backend = self.backends.get("orchestration")
        ask = getattr(backend, "needs_seed_for", None)
        if callable(ask):
            return bool(ask(system_prompt))
        return bool(getattr(backend, "needs_seed", False))

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

    def _count_prompt_mode(self, mode: str) -> None:
        """Tally one orchestration prompt push as SEED or DELTA.

        Args:
            mode: ``"seed"`` or ``"delta"``.
        """
        census = dict(self.shared_state.orchestration_prompt_modes or {})
        census[mode] = int(census.get(mode, 0)) + 1
        self.shared_state.orchestration_prompt_modes = census

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
                running_tasks_reader=self._context_running_tasks_reader,
                action_runner=self._run_action_now_sync,
                reference_reader=self._context_reference_reader,
            )
            setter(provider)
        except Exception:  # noqa: BLE001 — context pull is best-effort
            log.exception("Coordinator: failed to attach orchestration context tools")

    def _context_reference_reader(self, name: str = "") -> str:
        """Resolve a reference doc by stem; reject path traversal."""
        from hyperloom.inference_optimizer.session.paths import asset_prompt_references_dir

        refs_dir = asset_prompt_references_dir()
        stem = (name or "").strip()
        if not stem or "/" in stem or "\\" in stem or stem.startswith("."):
            available = sorted(p.stem for p in refs_dir.glob("*.md"))
            return f"(read_reference: invalid name {name!r}; available: {available})"
        candidate = (refs_dir / stem).with_suffix(".md").resolve()
        if candidate.parent != refs_dir.resolve():
            return f"(read_reference: path traversal rejected for {name!r})"
        if not candidate.exists():
            available = sorted(p.stem for p in refs_dir.glob("*.md"))
            return f"(read_reference: {name!r} not found; available: {available})"
        return candidate.read_text(encoding="utf-8")

    def _context_inbox_reader(self, since_seq: int = 0) -> str:
        """Synchronous projection of the orchestration inbox tail (sync SQLite path).

        Args:
            since_seq: Only events with a sequence number greater than this are
                included; defaults to ``0`` (all events).

        Returns:
            A newline-joined rendering of all matching inbox events, or
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
        lines = [_format_inbox_event(m) for m in msgs]
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
        header = "=== Recent action outcomes (newest last) ==="
        body_lines: list[str] = []
        body_lines.extend(_format_inbox_event(m, max_variant_rows=_RECENT_OUTCOMES_VARIANT_ROWS) for m in msgs)
        rendered = "\n".join(body_lines).splitlines()
        if len(rendered) > _RECENT_OUTCOMES_LINE_CAP:
            rendered = rendered[-_RECENT_OUTCOMES_LINE_CAP:]
            return "\n".join(
                [header]
                + rendered
                + [f"(truncated at {_RECENT_OUTCOMES_LINE_CAP} lines; re-query with a smaller top_k)"]
            )
        return "\n".join([header] + rendered)

    def _context_running_tasks_reader(self) -> str:
        """Synchronous projection of in-flight tasks with their held resources.

        Returns:
            One line per running task carrying elapsed time, lease expiry, held
            lanes, leased GPUs and heartbeat age, or a placeholder string.
        """
        try:
            rows = self.bus.db.fetchall_sync(
                "SELECT * FROM tasks WHERE state='running' ORDER BY updated_at ASC",
                (),
            )
        except Exception as exc:  # noqa: BLE001
            return f"(running tasks unavailable: {exc!r})"
        if not rows:
            return "(no tasks in flight)"

        lanes_by_task: dict[str, list[str]] = {}
        # Soonest lane expiry: the first one to lapse is when reclaim starts.
        expiry_by_task: dict[str, str] = {}
        for r in self.bus.db.fetchall_sync("SELECT lane, task_id, expires_at FROM leases", ()):
            tid = str(r["task_id"])
            lanes_by_task.setdefault(tid, []).append(str(r["lane"]))
            expires = str(r["expires_at"])
            prev = expiry_by_task.get(tid)
            if prev is None or expires < prev:
                expiry_by_task[tid] = expires
        gpus_by_task: dict[str, list[int]] = {}
        for r in self.bus.db.fetchall_sync("SELECT gpu_id, task_id FROM gpu_leases", ()):
            gpus_by_task.setdefault(str(r["task_id"]), []).append(int(r["gpu_id"]))

        now_unix = time.time()
        lines = ["=== Tasks in flight ==="]
        for row in rows:
            task = Task.from_row(row)
            params = task.params or {}
            started = _parse_iso_unix(task.updated_at)
            running_sec = max(0.0, now_unix - started) if started > 0 else 0.0
            parts = [
                f"  - task_id={task.task_id}",
                f"kind={task.kind!r}",
                f"running_sec={int(running_sec)}",
            ]
            domain = str(params.get("domain") or "")
            gap = str(params.get("gap_canonical_id") or "")
            if domain:
                parts.append(f"domain={domain!r}")
            if gap:
                parts.append(f"gap={gap!r}")
            parts.append(f"idempotency_key={task.idempotency_key!r}")
            if task.lease_ttl_sec:
                parts.append(f"lease_ttl_sec={task.lease_ttl_sec}")
            expires_at = expiry_by_task.get(task.task_id, "")
            if expires_at:
                exp_unix = _parse_iso_unix(expires_at)
                if exp_unix > 0:
                    parts.append(f"lease_expires_in_sec={int(exp_unix - now_unix)}")
            lanes = lanes_by_task.get(task.task_id)
            if lanes:
                parts.append(f"lanes={sorted(lanes)}")
            gpus = gpus_by_task.get(task.task_id)
            if gpus:
                parts.append(f"gpu_ids={sorted(gpus)}")
            hb_age = self._task_heartbeat_age_sec(task, now_unix=now_unix)
            if hb_age is not None:
                parts.append(f"heartbeat_age_sec={int(hb_age)}")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def _task_heartbeat_age_sec(self, task: "Task", *, now_unix: float) -> float | None:
        """Age of a specialist's freshest liveness file, mirroring the reaper.

        The reap loop treats either ``heartbeat.json`` or ``process.log`` as
        proof of life; this reports the same signal.

        Args:
            task: The running task to probe.
            now_unix: Current wall-clock epoch seconds.

        Returns:
            Seconds since the most recent liveness write, or ``None`` when no
            workspace file is readable.
        """
        if (task.kind or "").strip() != "specialist":
            return None
        ws = runs_dir(self.session_dir, "specialist", task.task_id)
        newest = 0.0
        for name in ("heartbeat.json", "process.log"):
            try:
                mtime = (ws / name).stat().st_mtime
            except OSError:
                continue
            newest = max(newest, mtime)
        if newest <= 0:
            return None
        return max(0.0, now_unix - newest)

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
                # Same turn metadata the token row is built from, so both halves
                # carry the backend's call_id when it stamped one.
                call_id=metadata.get("call_id"),
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

    async def _compose_prompt(self, agent_name: str, *, system_prompt: str | None = None) -> str:
        """Compose the orchestration prompt: SharedState summary + inbox tail (with canonical msg_id per inbox row).

        Args:
            agent_name: The agent role to compose the per-tick prompt for;
                selects which advisory/telemetry sections are included.
            system_prompt: The system prompt the turn will carry. The SEED/DELTA
                gate needs it because a re-scoped prompt empties the backend's
                conversation inside the turn this prompt is being built for.

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
            push_full = (
                not self._orchestration_conversational()
                or not self._orchestration_seeded
                or self._orchestration_needs_seed(system_prompt)
            )
            if self._orchestration_conversational():
                log.info(
                    "orchestration prompt mode=%s seeded=%s tick=%s",
                    "SEED" if push_full else "DELTA",
                    self._orchestration_seeded,
                    getattr(self.shared_state, "tick", 0),
                )
                self._count_prompt_mode("seed" if push_full else "delta")

        # On a full SEED push, inject recovered working memory.
        if (
            agent_name == "orchestration"
            and push_full
            and self._orchestration_conversational()
            and self._orchestration_seed_memory
        ):
            sections.append(self._orchestration_seed_memory)

        if agent_name == "orchestration":
            # Refresh before any section renders it.
            obj = self._current_objective
            self.shared_state.target_gap_pct = obj.gap_pct(self.shared_state) if obj is not None else 0.0
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
                        "`explore` rounds (which bench every variant on the "
                        "stack) will likely be cut by the deadline."
                    )

        # Time budget for Robustness — drives the deadline_imminent alert.
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
            # Resource pools are orchestration-only; robustness cannot schedule GPU work.
            if agent_name != "robustness":
                sections.append("=== Resource pools ===")
                sections.append(self.shared_state.to_resource_pools_summary())
        if agent_name == "orchestration":
            # Advisory/ledger blocks below are part of the full SEED push only.
            if push_full:
                denial_summary = self.shared_state.to_policy_denial_summary(top_k=6)
                if denial_summary:
                    sections.append(denial_summary)
            # Outside the SEED gate: a queue seen once is the amnesia it fixes.
            if (self.shared_state.phase or "").strip().upper() == _phase_state.PHASE_FRAMEWORK_AGENT:
                untested_block = self.shared_state.to_untested_proposals_summary()
                if untested_block:
                    sections.append("=== Untested proposals (current cycle) ===")
                    sections.append(untested_block)

        # Recipe KB T0 warm-start snapshot + structured gaps[] ledger.
        if agent_name == "orchestration" and push_full:
            try:
                warm_block = self.shared_state.to_warm_start_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: warm_start_summary failed")
                warm_block = ""
            if warm_block:
                sections.append("=== Warm start (Recipe KB T0) ===")
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
                research_block = self._research_scout_seed_block()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: research scout seed render failed")
                research_block = ""
            if research_block:
                sections.append(research_block)
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

            # A latency budget changes what a KEEP even means, so it is a
            # constraint the router has to see rather than telemetry it may skip.
            try:
                latency_block = self.shared_state.to_latency_budget_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: latency_budget_summary failed")
                latency_block = ""
            if latency_block:
                sections.append("=== Latency budget (constraint) ===")
                sections.append(latency_block)

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

        # Conversational DELTA turn: tell the agent verbose state was not
        # re-pushed. Where to find it depends on what the backend mounted —
        # pointing a tool-less session at the context tools is an instruction
        # it cannot follow, and the state is still in its conversation anyway.
        if agent_name == "orchestration" and not push_full:
            preamble = (
                "This is a continuation of our ongoing conversation; the "
                "full session state was NOT re-pasted. The Phase, Mission "
                "progress, Time budget, and new inbox events above are the "
                "delta since your last turn. "
            )
            if self._orchestration_context_tools_mounted():
                tool_list = ", ".join(_CONTEXT_TOOL_NAMES)
                sections.append("=== Context (pull on demand) ===")
                sections.append(
                    preamble + "Pull anything else you need "
                    f"with the read-only context tools: {tool_list} "
                    "(and `Read` for sandboxed files). Reason "
                    "from your own running plan; do not re-derive it from scratch."
                )
            else:
                sections.append("=== Context (delta turn) ===")
                sections.append(
                    preamble + "Everything omitted was pushed earlier in this "
                    "same conversation; re-read it above. Reason from your own "
                    "running plan; do not re-derive it from scratch."
                )

        # NOTE: there is deliberately no "=== Specialist health ===" block.
        # This prompt renders only on an agent's own turn, and a turn only
        # comes around between blocking actions — so a running specialist is
        # exactly what the agent is waiting on and is structurally absent from
        # any snapshot taken here. Measured over a full 11.6h session: 33
        # renders, 0 of them overlapped a live specialist, while specialists
        # held 41% of the wall clock. A block that always reports "none
        # running" is worse than no block, because it manufactures a false
        # belief. In-flight specialists reach the agent through
        # ``specialist_progress`` observations (pushed from the reap loop,
        # independent of turn timing) and are verified on demand with
        # ``get_running_tasks``.

        # Robustness gets phase budget telemetry for medium-severity alerts.
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
        rendered = list(msgs)
        if msgs:
            top = msgs[-1]
            self._coord._rendered_cursor[agent_name] = (int(top.seq), str(top.msg_id))
        if agent_name == "critic":
            rendered = await self._augment_critic_inbox_with_pending(rendered)
        if rendered:
            sections.append(f"=== Inbox for {agent_name} (newest last) ===")
            # Only Orchestration acts on variant-level failures; the reviewers do not.
            variant_rows = 3 if agent_name == "orchestration" else 0
            for m in rendered:
                sections.append(f"  {_format_inbox_event(m, max_variant_rows=variant_rows)}")
        else:
            sections.append(f"=== Inbox for {agent_name} ===")
            sections.append("(no new messages)")

        return "\n".join(sections)

    async def _advance_rendered_cursor(self, agent_name: str) -> None:
        """Advance an agent's read cursor to the last message its prompt rendered.

        Args:
            agent_name: The agent whose cursor to advance; a no-op when its
                last composed prompt carried no new messages.
        """
        entry = self._coord._rendered_cursor.get(agent_name)
        if entry is None:
            return
        seq, msg_id = entry
        await self.cursors.advance(agent_name, seq=seq, msg_id=msg_id)

    async def _augment_critic_inbox_with_pending(self, rendered: list["Message"]) -> list["Message"]:
        """Ensure every undecided proposal awaiting a Critic verdict is present.

        A rendered proposal whose verdict has not yet arrived will not appear in
        the next inbox because the cursor has legitimately moved past it. Source
        the review set from the durable ``pending_proposals`` registry and merge
        any missing proposal messages into the rendered window (deduped by
        ``msg_id``, re-sorted by ``seq`` so "newest last" holds).

        Args:
            rendered: The messages selected for the inbox.

        Returns:
            The rendered list augmented with any undecided proposal messages
            not already present; unchanged on any error (best-effort).
        """
        try:
            pending = [p for p in self.state.pending_proposals.values() if not getattr(p, "decided", False)]
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
            The override prompt if configured, ``""`` for roles that are not
            prompt-driven, the role's prompt file contents, or a placeholder
            string when the file is missing.
        """
        override = getattr(self, "system_prompt_overrides", {}).get(agent_name)
        if override is not None:
            return override
        role = self.role_registry[agent_name]
        if not role.prompt_driven:
            return ""
        try:
            return role.load_system_prompt()
        except FileNotFoundError:
            return f"(no system prompt for {agent_name})"

    # Advisory prompt blocks (folded in from the former AdvisoryCollaborator).
    # Consumed by :meth:`_compose_prompt` above and by the phase handlers via the
    # coordinator's bare-name ``_DELEGATED`` resolution.
    def _plateau_advisory_block(self) -> str:
        """Render the plateau-judgment advisory block for the current phase.

        In the optimisation phase both arms are always reported: the phase
        leaves only when both are dry, so naming one alone would say "plateau"
        about a phase still paying on the other lever. Both dry advances to
        KERNEL_AGENT via ``optimize_no_more_leverage`` (a non-terminal lever
        switch); a KERNEL plateau is advisory only.

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
        if phase == _phase_state.PHASE_FRAMEWORK_AGENT:
            # Both arms, always: the phase leaves only when both are dry, so
            # reporting one alone would say "plateau" about a phase that is
            # still paying on the other lever.
            config_dry, config_ev = _phase_state.compute_plateau_explore(
                state,
                lookback=int(
                    overrides.get("explore_lookback", _phase_state.DEFAULT_PLATEAU_EXPLORE_LOOKBACK),
                ),
                keep_gain_threshold_pct=float(
                    overrides.get("explore_keep_gain_pct", _phase_state.DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT),
                ),
                empty_streak_threshold=int(
                    overrides.get("explore_empty_streak", _phase_state.DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK),
                ),
            )
            source_dry, source_ev = _phase_state.source_arm_plateaued(state)
            if config_dry:
                lines.append("OPTIMIZE config arm plateaued: low recent KEEP gain plus specialist empty streak.")
                lines.append(
                    "  recent_keep_gain_pct="
                    f"{config_ev.get('recent_keep_gain_pct', 0.0)} "
                    f"threshold={config_ev.get('keep_gain_threshold_pct', 0.0)} "
                    f"empty_streak={config_ev.get('empty_streak', 0)} "
                    f"streak_threshold={config_ev.get('empty_streak_threshold', 0)}"
                )
            streak = int(source_ev.get("source_consecutive_no_keep", 0) or 0)
            if source_dry:
                lines.append("OPTIMIZE source arm plateaued: candidates exhausted or no KEEP on the trailing ones.")
            elif streak > 0:
                lines.append("OPTIMIZE source arm approaching plateau: no KEEP on the trailing candidates.")
            if source_dry or streak > 0:
                lines.append(
                    f"  consecutive_no_keep={streak} "
                    f"threshold={source_ev.get('source_threshold', 0)} "
                    f"candidates_exhausted={source_ev.get('source_candidates_exhausted', False)}"
                )
            if config_dry != source_dry:
                lines.append("  Only one arm is dry: the other lever is still live, and the phase stays open.")
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
        if not lines:
            return ""
        if phase == _phase_state.PHASE_FRAMEWORK_AGENT:
            lines.append(
                "Note: OPTIMIZE advances to KERNEL_AGENT only when BOTH arms are dry "
                "(reason=optimize_no_more_leverage) -- a non-terminal lever switch, not "
                "the end of the run. Either arm going quiet also flags the next "
                "macro-cycle to steer off this bottleneck. You may request an earlier "
                "advance with an escalate_strategy_change hint, or keep working the live "
                "arm until the plateau / budget gate fires."
            )
        else:
            lines.append(
                "Phase advance is driven only by hard limits (phase budget, "
                "terminal stop_reason) or explicit escalate_strategy_change "
                "hints; this block is informational."
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
        """Render the R3 cyclic bottleneck-redirect advisory (optimisation phase only).

        Applies when a prior cycle's plateau flagged
        ``pending_bottleneck_switch``. Names the bottleneck we plateaued on, the
        current dominant roofline direction, and a suggested specialist domain so
        Orchestration redirects the new cycle's dispatch. Advisory, never gates.

        Returns:
            The rendered bottleneck-redirect advisory text, or ``""`` when not
            applicable.
        """
        state = self.shared_state
        if (getattr(state, "phase", "") or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
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
        lines.append(f"  macro_cycle={cycle}")
        lines.append("Advisory only: pick the domain/tag yourself; this nudges focus, it does not gate dispatch.")
        return "\n".join(lines)

    def _acceptance_threshold_advisory_block(self) -> str:
        """Render the decaying acceptance bar and prior measured gains as evidence.

        Active only in cyclic mode after at least one macro-cycle. Shows the
        current KEEP threshold and lists prior measured results above and below
        it for decision context; the results never gate re-submission.

        Returns:
            The rendered advisory text, or ``""`` when not applicable.
        """
        state = self.shared_state
        keep = _phase_state.resolve_keep_threshold(state)
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        if cycle < 1:
            return ""
        stable = keep / 2.0
        search = getattr(state, "explore_search", None) or {}
        entries: list[dict[str, Any]] = []
        if isinstance(search, dict):
            tested = search.get("tested") or {}
            if isinstance(tested, dict):
                entries.extend(v for v in tested.values() if isinstance(v, dict))
            rejected = search.get("rejected") or []
            if isinstance(rejected, list):
                entries.extend(v for v in rejected if isinstance(v, dict))
        above_bar: list[tuple[str, float]] = []
        below_bar: list[tuple[str, float]] = []
        for e in entries:
            try:
                g = float(e.get("gain_pct"))
            except (TypeError, ValueError):
                continue
            name = str(e.get("name") or e.get("fingerprint") or "")[:48]
            (above_bar if g >= keep else below_bar).append((name, g))
        lines: list[str] = [
            f"Current acceptance bar (macro_cycle={cycle}): KEEP>={keep:.2f}% stack_stable>={stable:.2f}%.",
            "Historical results are evidence only — any fingerprint may be re-proposed.",
        ]
        if above_bar:
            above_bar.sort(key=lambda p: p[1], reverse=True)
            lines.append("Prior results above the bar (consider re-proposing if conditions changed):")
            for name, g in above_bar[:8]:
                lines.append(f"  {name}: prior gain {g:+.2f}% >= {keep:.2f}%")
        if below_bar:
            below_bar.sort(key=lambda p: p[1], reverse=True)
            lines.append("Prior results below the bar (reference):")
            for name, g in below_bar[:5]:
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

    def _research_scout_seed_block(self) -> str:
        """Render the persisted research-scout findings for an Orchestration SEED.

        The scout's executable proposals are not rendered here: they go through
        ``=== Untested proposals (current cycle) ===`` alongside every other
        domain's, which also drops the ones already benched.
        """
        from ..knowledge import research_hints as _research_hints

        hints = _research_hints.load_hints(self.session_dir)
        rounds = [
            row
            for row in (getattr(self.shared_state, "specialist_rounds", []) or [])
            if isinstance(row, dict) and row.get("domain") == "research_scout_specialist"
        ]
        if not hints and not rounds:
            return ""

        lines = ["=== Research Scout ==="]
        if hints:
            lines.append("Findings:")
            for hint in hints:
                lines.append(json.dumps(hint, sort_keys=True))

        questions: list[str] = []
        seen_questions: set[str] = set()
        for row in rounds:
            for question in row.get("residual_questions") or []:
                text = str(question).strip()
                if text and text not in seen_questions:
                    seen_questions.add(text)
                    questions.append(text)

        if questions:
            lines.append("Residual questions:")
            lines.extend(f"- {question}" for question in questions)
        return "\n".join(lines)

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
