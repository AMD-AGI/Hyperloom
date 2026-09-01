# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Configuration arm: macro-cycle strategy, specialist fan-out/retry, gap
tracking, and autosubmit of specialist patches / framework configs."""

from __future__ import annotations
from hashlib import sha1
import logging as _logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from . import machine_state as _phase_state
from hyperloom.common.coerce import to_float
from hyperloom.inference_optimizer.breakdown.agent_ownership import (
    patch_owner_phase,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from ..bus.message_bus import Message
from ..policy.gate import (
    SPECIALIST_FROM_AGENT_PREFIX,
    PolicyDenied,
    validate_freeform_wave_task,
)
from ..loop.maintenance import run_lease_and_db_reclaim
from ..loop.sub_agent_runner import SubAgentResult
from ..prompts import write_prompt_snapshot as _write_prompt_snapshot
from ..specialists.runner import SpecialistFailureType
from ..state.failure_evidence import UNMEASURED_OUTCOMES, failure_from_variant_outcome
from ..state.shared_state import inject_stack_base_params
from ..state.task_registry import Task
from ..loop.coordinator import (
    FORCE_STALLED_KEEP_ROUNDS,
    FORCE_STALLED_SPECIALIST_ROUNDS,
    PendingProposal,
    SPECIALIST_AUTO_RETRY_MAX,
    _framework_config_levers_from_done,
)
from ..collaborator import CoordinatorCollaborator

log = _logging.getLogger(__name__)

# Artifact references copied from a per-variant outcome onto its gap attempt.
_GAP_ATTEMPT_ARTIFACT_KEYS: tuple[str, ...] = (
    "failure_id",
    "fingerprint",
    "stage",
    "workspace",
    "server_log_path",
)


def _forward_enablement_carriers(src: dict[str, Any], dst: dict[str, Any]) -> None:
    """Copy eval-origin trigger context from specialist params to the integrate task."""
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
) -> None:
    """Preserve proposal ownership across delayed ``integrate_patch`` execution."""

    domain = str(src.get("domain") or src.get("source_domain") or "").strip()
    source_phase = patch_owner_phase(src)
    if source_phase:
        dst["source_phase"] = source_phase
    if domain:
        dst["domain"] = domain
        dst["provenance"] = f"specialist:{domain}"
    # ``framework`` is intentionally not forwarded: integrate_patch consumes
    # that parameter when selecting accuracy parsing/gating behavior, whereas
    # proposal ownership only needs the gap metadata below.
    # ``lever_kind`` travels with the proposal: the patch that lands moved the
    # same lever the specialist was dispatched against, and re-deriving it at
    # writeback time is how attribution drifts.
    for key in ("gap_canonical_id", "gap_layer", "lever_kind", "reauthor_attempt", "apply_retry_attempt"):
        value = src.get(key)
        if value not in (None, "", [], {}):
            dst[key] = value


class ExplorePhase(CoordinatorCollaborator):
    """The configuration arm of the OPTIMIZE phase: server-arg / env grids, the
    specialist fan-out that sources them, and the macro-cycle machinery that
    reopens a cycle. Not a phase of its own -- it shares FRAMEWORK_AGENT with
    the source arm, and ``exit_normal_optimize`` leaves only when both are dry.
    """

    def _negative_ledger_domain_counts(self, *, recent_cycles: int = 3) -> dict[str, int]:
        """Summarise recent negative explore-ledger pressure by specialist domain."""
        state = self.shared_state
        cur_cycle = int(getattr(state, "macro_cycle", 0) or 0)
        search = getattr(state, "explore_search", {}) or {}
        rows: list[Any] = []
        if isinstance(search, dict):
            tested = search.get("tested") or {}
            if isinstance(tested, dict):
                rows.extend(tested.values())
            rejected = search.get("rejected") or []
            if isinstance(rejected, list):
                rows.extend(rejected)
        counts: dict[str, int] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                cycle = int(row.get("cycle", cur_cycle) or 0)
            except (TypeError, ValueError):
                cycle = cur_cycle
            if cycle < max(0, cur_cycle - recent_cycles + 1):
                continue
            domain = str(
                row.get("domain")
                or row.get("specialist_domain")
                or row.get("source_domain")
                or row.get("provenance")
                or ""
            ).strip()
            if not domain:
                continue
            counts[domain] = counts.get(domain, 0) + 1
        return counts

    def _plan_cycle_focus(self) -> dict[str, Any]:
        """Pick an advisory specialist-domain focus for the current macro-cycle."""
        from ..kernel.roofline_snapshot import BOTTLENECK_DOMAIN_HINTS

        state = self.shared_state
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        domains = sorted({v[0] for v in BOTTLENECK_DOMAIN_HINTS.values()} | {"freeform_specialist"})
        scores: dict[str, float] = {d: 0.0 for d in domains}
        reasons: dict[str, list[str]] = {d: [] for d in domains}
        shift = getattr(state, "bottleneck_shift", {}) or {}
        to_domain = str(shift.get("to_domain") or "").strip()
        if to_domain:
            scores.setdefault(to_domain, 0.0)
            reasons.setdefault(to_domain, [])
            scores[to_domain] += 5.0
            reasons[to_domain].append(f"matches current bottleneck shift to {shift.get('to') or to_domain}")
        sat = getattr(state, "saturated_directions", {}) or {}
        if isinstance(sat, dict):
            for domain, row in sat.items():
                if not isinstance(row, dict):
                    continue
                d = str(domain or row.get("domain") or "").strip()
                if not d:
                    continue
                scores.setdefault(d, 0.0)
                reasons.setdefault(d, [])
                if bool(row.get("saturated")):
                    scores[d] -= 100.0
                    reasons[d].append(f"saturated at {row.get('within_pct')}% within roofline; deprioritized")
                else:
                    scores[d] += 1.0
                    reasons[d].append("not saturated in latest roofline snapshot")
        log_rows = list(getattr(state, "cycle_strategy_log", []) or [])
        tried = {str(r.get("focus") or "") for r in log_rows if isinstance(r, dict)}
        for row in log_rows:
            if not isinstance(row, dict):
                continue
            domain = str(row.get("focus") or "").strip()
            if not domain:
                continue
            scores.setdefault(domain, 0.0)
            reasons.setdefault(domain, [])
            gd = row.get("gain_delta")
            if isinstance(gd, (int, float)):
                scores[domain] += max(-2.0, min(3.0, float(gd)))
                reasons[domain].append(f"historical cycle gain_delta={float(gd):+.2f}%")
        for domain in domains:
            if domain not in tried:
                scores[domain] += 1.5
                reasons[domain].append("exploration bonus: not yet used as cycle focus")
        negative_counts = self._negative_ledger_domain_counts()
        for domain, count in negative_counts.items():
            scores.setdefault(domain, 0.0)
            reasons.setdefault(domain, [])
            penalty = min(4.0, 0.5 * float(count))
            scores[domain] -= penalty
            reasons[domain].append(f"recent negative ledger count={count} penalty={penalty:.1f}")
        focus = max(scores.items(), key=lambda kv: (kv[1], kv[0]))[0] if scores else "freeform_specialist"
        rationale_bits = reasons.get(focus) or ["fallback focus; no stronger cycle-level evidence"]
        return {
            "cycle": cycle,
            "focus": focus,
            "score": round(float(scores.get(focus, 0.0)), 3),
            "rationale": "; ".join(rationale_bits[:4]),
            "bottleneck_at_start": str(shift.get("to") or self.shared_state.current_top_bottleneck() or ""),
            "saturated_at_start": sorted(
                str(k)
                for k, v in (sat.items() if isinstance(sat, dict) else [])
                if isinstance(v, dict) and bool(v.get("saturated"))
            ),
            "gain_at_start": float(getattr(state, "gain_at_cycle_start", 0.0) or 0.0),
            "gain_delta": None,
        }

    def _record_cycle_strategy_for_current_cycle(self) -> None:
        """Append/update the advisory cycle-strategy row for the current cycle."""
        state = self.shared_state
        planned = self._plan_cycle_focus()
        log_rows = [r for r in (getattr(state, "cycle_strategy_log", []) or []) if isinstance(r, dict)]
        cycle = int(planned.get("cycle", 0) or 0)
        replaced = False
        for idx, row in enumerate(log_rows):
            if int(row.get("cycle", -1) or -1) == cycle:
                merged = dict(row)
                merged.update(planned)
                log_rows[idx] = merged
                replaced = True
                break
        if not replaced:
            log_rows.append(planned)
        state.cycle_strategy_log = log_rows[-50:]

    def _cycle_strategy_seed_block(self) -> str:
        """Render persisted cycle focus facts for orchestration SEED prompts."""
        rows = [r for r in (getattr(self.shared_state, "cycle_strategy_log", []) or []) if isinstance(r, dict)]
        if not rows:
            return ""
        cur_cycle = int(getattr(self.shared_state, "macro_cycle", 0) or 0)
        current = next((r for r in reversed(rows) if int(r.get("cycle", -1) or -1) == cur_cycle), rows[-1])
        lines = [
            f"=== Cycle {cur_cycle} strategy ===",
            f"focus={current.get('focus') or '(none)'} score={current.get('score', 0)}",
        ]
        rationale = str(current.get("rationale") or "").strip()
        if rationale:
            lines.append(f"rationale: {rationale}")
        saturated = current.get("saturated_at_start") or []
        if saturated:
            lines.append(f"saturated_at_start={saturated}")
        prior = [r for r in rows if int(r.get("cycle", -1) or -1) != cur_cycle][-5:]
        if prior:
            lines.append("previous cycles:")
            for row in prior:
                lines.append(
                    f"  - cycle={row.get('cycle')} focus={row.get('focus')} "
                    f"gain_delta={row.get('gain_delta')} saturated={row.get('saturated_at_start') or []}"
                )
        lines.append("Advisory only: use this as a prior, not a dispatch gate.")
        return "\n".join(lines)

    def _cycle_directive_fallback(self) -> str:
        """Render a deterministic cycle focus from ``_plan_cycle_focus``.

        Used when the LLM checkpoint produced no ``next_cycle_directive``; keeps
        every cycle's CYCLE DIRECTIVE section grounded in real telemetry.
        """
        try:
            planned = self._plan_cycle_focus()
        except Exception:  # noqa: BLE001 — fallback must never raise
            return ""
        focus = str(planned.get("focus") or "").strip()
        if not focus:
            return ""
        parts = [f"focus={focus}"]
        rationale = str(planned.get("rationale") or "").strip()
        if rationale:
            parts.append(rationale)
        bottleneck = str(planned.get("bottleneck_at_start") or "").strip()
        if bottleneck:
            parts.append(f"bottleneck={bottleneck}")
        saturated = planned.get("saturated_at_start") or []
        if saturated:
            parts.append(f"deprioritize saturated={list(saturated)}")
        return "; ".join(parts)

    def _reseed_orch_prompt_for_cycle(self) -> bool:
        """Rebuild the orchestration system prompt for the new macro-cycle.

        Injects the freshly-captured ``next_cycle_directive`` (or a deterministic
        fallback) into a rebuilt prompt, mutates ``system_prompt_overrides``,
        snapshots the installed scope, and records the directive in the
        ``cycle_directive_history`` ring. Skips a user-supplied
        ``--orch-prompt``. Best-effort; returns True when reseeded.
        """
        if getattr(self, "_orch_prompt_is_user_supplied", False):
            return False
        rebuild = getattr(self, "_rebuild_orch_prompt", None)
        if rebuild is None:
            return False
        state = self.shared_state
        cycle = int(getattr(state, "macro_cycle", 0) or 0)
        directive = str((dict(getattr(state, "orchestration_memory", {}) or {})).get("next_cycle_directive", "") or "")
        source = "llm"
        if not directive:
            directive = self._cycle_directive_fallback()
            source = "deterministic"
        new_prompt = rebuild(macro_cycle=cycle, cycle_directive=directive, phase=state.phase)
        overrides = getattr(self, "system_prompt_overrides", None)
        if not isinstance(overrides, dict):
            return False
        overrides["orchestration"] = new_prompt
        _write_prompt_snapshot(self.session_dir, "orchestration", new_prompt, phase=state.phase)
        history = list(getattr(state, "cycle_directive_history", []) or [])
        history.append(
            {
                "cycle": cycle,
                "directive": directive,
                "source": source,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
        )
        state.cycle_directive_history = history[-10:]
        return True

    def _apply_macro_cycle_reloop(self, evidence: dict[str, Any]) -> None:
        """Open a new macro-cycle on a SWEEP loopback into FRAMEWORK_AGENT.

        Increments ``macro_cycle``, persists the no-gain streak + per-cycle gain
        anchor, and resets per-cycle counters (including re-opening FRAMEWORK) for
        a fresh budget / plateau evaluation. The explore ledger is preserved.

        Args:
            evidence: The loopback evidence dict from ``compute_next_phase``;
                may carry ``no_gain_cycle_streak_effective`` which is persisted
                onto the new cycle.
        """
        state = self.shared_state
        prior_cycle = int(getattr(state, "macro_cycle", 0) or 0)
        try:
            prev_delta = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0) - float(
                getattr(state, "gain_at_cycle_start", 0.0) or 0.0
            )
            rows = [r for r in (getattr(state, "cycle_strategy_log", []) or []) if isinstance(r, dict)]
            for row in rows:
                if int(row.get("cycle", -1) or -1) == prior_cycle and row.get("gain_delta") is None:
                    row["gain_delta"] = round(prev_delta, 6)
            state.cycle_strategy_log = rows[-50:]
        except Exception:  # noqa: BLE001 — advisory bookkeeping only
            log.exception("Coordinator: cycle_strategy gain_delta backfill failed")
        state.macro_cycle = prior_cycle + 1
        if isinstance(evidence, dict) and "no_gain_cycle_streak_effective" in evidence:
            state.no_gain_cycle_streak = int(evidence.get("no_gain_cycle_streak_effective", 0) or 0)
        # Anchor gain for the cycle we are about to start.
        try:
            state.gain_at_cycle_start = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
        except (TypeError, ValueError):
            state.gain_at_cycle_start = 0.0
        # Reset per-cycle counters.
        try:
            state.reset_per_cycle_plateau_state()
        except Exception:  # noqa: BLE001 — resets are best-effort
            log.exception("Coordinator: per-cycle reset failed on reloop")
        # Mark a macro-cycle boundary in the preserved progress ledger so the
        # consecutive-no-keep plateau gate ignores the prior cycle's trailing
        # no-KEEP streak.
        try:
            progress = getattr(state, "framework_agent_phase_progress", None)
            if not isinstance(progress, list):
                progress = []
                state.framework_agent_phase_progress = progress
            if not (
                progress
                and isinstance(progress[-1], dict)
                and str(progress[-1].get("status") or "") == "cycle_boundary"
            ):
                progress.append(
                    {
                        "candidate_id": "",
                        "status": "cycle_boundary",
                        "kept": False,
                        "gain_pct": 0.0,
                        "cycle": int(getattr(state, "macro_cycle", 0) or 0),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                )
        except Exception:  # noqa: BLE001 — plateau-reset marker is best-effort
            log.exception("Coordinator: cycle_boundary marker append failed")
        try:
            self._record_cycle_strategy_for_current_cycle()
        except Exception:  # noqa: BLE001 — focus is advisory only
            log.exception("Coordinator: cycle strategy planning failed on reloop")
        log.info(
            "Coordinator: macro-cycle reloop %d → %d (no_gain_streak=%d, gain_anchor=%.4f)",
            prior_cycle,
            state.macro_cycle,
            state.no_gain_cycle_streak,
            state.gain_at_cycle_start,
        )

    async def _run_cycle_soft_restart(
        self,
        *,
        prior_cycle: int,
        new_cycle: int,
    ) -> dict[str, Any] | None:
        """Medium-intensity soft restart at a macro-cycle boundary.

        Recycles transient/per-cycle resources (fresh leases, pruned DB, cleared
        caches, conversation reset) without losing accumulated optimization state;
        ``current_best`` / ``optimization_stack`` / ``explore_search`` are
        preserved. Idempotent and best-effort: every step is independently
        guarded so one failure never aborts the run loop.

        Args:
            prior_cycle: The macro-cycle number that just finished.
            new_cycle: The macro-cycle number being entered.

        Returns:
            A summary dict of the restart steps performed, or ``None`` when the
            soft restart is disabled.
        """
        if not getattr(self, "_cycle_soft_restart", False):
            return None
        summary: dict[str, Any] = {
            "prior_cycle": int(prior_cycle),
            "new_cycle": int(new_cycle),
        }
        # 1) Compact the cycle's conversation into durable memory, re-focus the
        # orchestration prompt for the new cycle, then reset.
        try:
            compacted = await self._maybe_checkpoint_orchestration(
                tick=int(getattr(self.shared_state, "tick", 0) or 0),
                phase_changed=True,
                force=True,
            )
            summary["memory_compacted"] = bool(compacted)
            try:
                summary["orch_prompt_reseeded"] = self._reseed_orch_prompt_for_cycle()
            except Exception:  # noqa: BLE001 — reseed is best-effort
                log.exception("cycle soft-restart: orchestration prompt reseed failed")
            # Reset unconditionally so a no-op checkpoint still reseeds next turn.
            self._reset_orchestration_conversation()
            summary["conversation_reset"] = True
        except Exception:  # noqa: BLE001 — soft restart never aborts the run loop
            log.exception("cycle soft-restart: conversation reset failed")
        # 2-3) Reap leases, reclaim orphaned running tasks, prune DB.
        await run_lease_and_db_reclaim(self, summary, reason="cycle_soft_restart")
        # 4) Deep-clean any lingering inference-server processes.
        if getattr(self, "_cycle_restart_servers", False):
            try:
                self._restart_inference_servers()
                summary["servers_restarted"] = True
            except Exception:  # noqa: BLE001
                log.exception("cycle soft-restart: server restart failed")
        log.info(
            "cycle soft-restart %d → %d: %s",
            int(prior_cycle),
            int(new_cycle),
            summary,
        )
        try:
            await self._record_observation(
                "coordinator",
                "observation",
                {"kind": "cycle_soft_restart", **summary},
            )
        except Exception:  # noqa: BLE001
            log.exception("cycle soft-restart: observation write failed")
        return summary

    def _restart_inference_servers(self) -> None:
        """Deep-clean lingering inference-server processes (macro-cycle soft restart).

        Reuses the grid runner's ``_kill_stale_servers`` /proc sweep, which only
        targets vLLM/SGLang/atom server processes outside our own process group
        and is a no-op in multi-node mode.
        """
        from ..actions.executors._grid_runner import _kill_stale_servers

        _kill_stale_servers()

    async def _on_cycle_start_reprofile(self, *, from_phase: str) -> None:
        """Force a fresh analysis at the start of a reopened macro-cycle.

        Reached on every cycle start now. It used to be attached to the config-arm
        entry, and the reloop targeted FRAMEWORK_AGENT whenever the framework
        phase was enabled -- so with the default configuration this never ran,
        and each new cycle re-targeted the bottleneck the *previous* cycle
        measured. One phase means one entry, and the reprofile happens.

        Args:
            from_phase: The phase being left; only a SWEEP origin starts a cycle.
        """
        if (from_phase or "").upper() == _phase_state.PHASE_SWEEP and int(
            getattr(self.shared_state, "macro_cycle", 0) or 0
        ) > 0:
            try:
                task = await self._enqueue_internal_analysis_task(
                    reason="cycle_start",
                )
                self.shared_state.auto_roofline_pending_task_id = task.task_id
                log.info(
                    "cycle %d start: forced reprofile task=%s",
                    int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    task.task_id,
                )
            except Exception:  # noqa: BLE001 — reprofile is best-effort
                log.exception(
                    "cycle start: forced reprofile enqueue failed",
                )

    async def _maybe_force_stalled_domain_specialist(self) -> None:
        """Force-dispatch a domain specialist for a domain untouched for too many
        config-arm rounds that still has an open gap in the gaps[] ledger.

        A real scheduling event (a domain delegate routed through PolicyGate +
        warmup + the GPU specialist pool). Idempotent per
        ``(anchor, round, macro_cycle)`` and self-throttling (zeroes the
        per-anchor counter on dispatch). At most one forced dispatch per tick.

        Note:
            Side-effecting: may dispatch a domain specialist via
            ``_handle_intent`` and mutate per-anchor throttle counters on
            ``shared_state``. Returns nothing.
        """
        state = self.shared_state
        if str(getattr(state, "phase", "") or "").upper() != _phase_state.PHASE_FRAMEWORK_AGENT:
            return None
        if not bool(getattr(state, "force_stalled_specialist_enabled", True)):
            return None
        spec_thr = max(1, int(getattr(state, "force_stalled_specialist_rounds", 0) or FORCE_STALLED_SPECIALIST_ROUNDS))
        keep_thr = max(1, int(getattr(state, "force_stalled_keep_rounds", 0) or FORCE_STALLED_KEEP_ROUNDS))
        try:
            stalled = state.stalled_domains(
                specialist_threshold=spec_thr,
                keep_threshold=keep_thr,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("stalled-domain force: stalled_domains() failed")
            return None
        if not stalled:
            return None

        from ..specialists.domains import domain_for_tag

        round_id = int((state.explore_search or {}).get("cursor") or 0)
        for anchor in stalled:
            gap_cid = state.best_gap_for_anchor(anchor)
            if not gap_cid:
                continue
            dom = domain_for_tag(anchor)
            if dom is None:
                continue
            params: dict[str, Any] = {
                "domain": dom.key,
                "tags": [anchor],
                "gap_canonical_id": gap_cid,
                "scope": "domain",
                "source": "coordinator_internal",
                "reason": f"stalled_domain_force:{anchor}",
            }
            intent = Intent(
                type=IntentType.DELEGATE,
                payload={
                    "action_name": "specialist",
                    "params": params,
                    "idempotency_key": (f"forced-stalled-{anchor}-round{round_id}{self._cycle_idem_suffix()}"),
                },
            )
            # Zero the counter up-front so a slow enqueue can't re-fire next tick.
            try:
                state.note_specialist_dispatched(anchor)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "stalled-domain force: counter reset failed for %s",
                    anchor,
                )
            try:
                await self._handle_intent("orchestration", intent)
            except Exception:  # noqa: BLE001 — defensive, never crash the tick
                log.exception(
                    "stalled-domain force: dispatch failed for anchor=%s domain=%s gap=%s",
                    anchor,
                    dom.key,
                    gap_cid,
                )
                continue
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.exception("stalled-domain force: state save failed")
            log.info(
                "stalled-domain force: dispatched domain=%s anchor=%s gap=%s round=%d (spec_thr=%d keep_thr=%d)",
                dom.key,
                anchor,
                gap_cid,
                round_id,
                spec_thr,
                keep_thr,
            )
            # One forced dispatch per tick.
            return None
        return None

    def _seed_gaps_from_research_hints(self) -> None:
        """Inject research hints as advisory gaps[] seeds (idempotent)."""
        from ..knowledge import research_hints as _research_hints

        hints = _research_hints.load_hints(self.session_dir)
        for hint in hints:
            what = str(hint.get("what") or "").strip()
            source = str(hint.get("source") or "").strip()
            if not what or not source:
                continue
            tags = hint.get("domain_tags") or []
            key = f"{what.lower()}::{source.lower()}"
            cid = f"gap.research_hint.{sha1(key.encode()).hexdigest()[:16]}"
            try:
                self.shared_state.upsert_gap(
                    {
                        "canonical_id": cid,
                        "symptom": what,
                        "layer": "research_hint",
                        "severity": "medium",
                        "domain_hint": str(tags[0]) if tags else "",
                        "source": "research_scout",
                        "provenance": str(hint.get("source") or ""),
                    }
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "research-scout: upsert_gap failed for %s",
                    cid,
                )

    async def _fan_out_specialist_wave(
        self,
        source: str,
        intent: Intent,
        params: dict[str, Any],
    ) -> None:
        """Fan a specialist delegate carrying ``params.tasks=[...]`` into N
        standard free-form specialist dispatches (scope=freeform, lane=cpu,
        mode=research defaults). Each fanned task is re-dispatched through the
        normal ``_handle_delegate`` path. Per-task idempotency keys derive from
        the wave key. Each entry must pass the same structural checks as
        :func:`validate_freeform_wave_task` (the PolicyGate runs these first).

        Args:
            source: The agent issuing the wave delegate.
            intent: The originating specialist DELEGATE intent.
            params: The delegate params carrying the ``tasks`` list to fan out.
        """
        tasks = params.get("tasks") or []
        shared = {k: v for k, v in params.items() if k != "tasks"}
        base_key = str(intent.payload.get("idempotency_key") or "").strip()
        pending: list[Intent] = []
        for idx, task in enumerate(tasks):
            desc = validate_freeform_wave_task(task, index=idx)
            sub_params = dict(shared)
            sub_params["scope"] = "freeform"
            sub_params["task_description"] = desc
            summary = str(task.get("task_summary") or "").strip()
            if summary:
                sub_params["task_summary"] = summary
            for carry in (
                "mode",
                "bench",
                "lane",
                "model",
                "priority",
                "timeout_minutes",
                "max_turns",
            ):
                if isinstance(task, dict) and carry in task:
                    sub_params[carry] = task[carry]
            sub_params.setdefault("mode", "research")
            sub_params.setdefault("lane", "cpu")
            sub_payload = dict(intent.payload)
            sub_payload["params"] = sub_params
            if base_key:
                sub_payload["idempotency_key"] = f"{base_key}-w{idx}"
            else:
                sub_payload.pop("idempotency_key", None)
            sub_intent = Intent(type=intent.type, payload=sub_payload)
            try:
                self.policy.validate_intent(source, sub_intent)
            except PolicyDenied as denied:
                await self._record_policy_denied(source, sub_intent, denied)
                raise
            pending.append(sub_intent)
        for sub_intent in pending:
            await self._handle_delegate(source, sub_intent)

    async def _record_specialist_retry_exhausted(
        self,
        *,
        task: "Task",
        ftype: SpecialistFailureType,
        error: str,
        attempts_used: int,
        cap: int,
        detail: str,
    ) -> None:
        """Broadcast that an infra-failed specialist is being abandoned.

        Args:
            task: The specialist task whose final attempt failed.
            ftype: The classified failure type.
            error: The failure reason carried by the attempt.
            attempts_used: Retry attempts already spent.
            cap: Configured retry ceiling.
            detail: Why no further retry was scheduled.
        """
        params = task.params or {}
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "specialist_auto_retry_exhausted",
                "task_id": task.task_id,
                "domain": str(params.get("domain") or ""),
                "gap_canonical_id": str(params.get("gap_canonical_id") or ""),
                "attempts_used": attempts_used,
                "max_attempts": cap,
                "failure_type": ftype.value,
                "reason": error[:200],
                "detail": detail,
            },
        )
        log.warning(
            "specialist auto-retry exhausted: task=%s failure=%s attempts=%d/%d (%s)",
            task.task_id,
            ftype.value,
            attempts_used,
            cap,
            detail,
        )

    async def _maybe_auto_retry_specialist(
        self,
        task: "Task",
        result: "SubAgentResult",
    ) -> bool:
        """Re-enqueue a fresh specialist task on a transient infra failure.

        Returns ``True`` when a retry was scheduled (the caller must then skip
        this attempt's delegated_result + bookkeeping). Only infra failures
        (timeout / crash / stale-heartbeat, per ``classify_specialist_failure``)
        are retried, capped at :data:`SPECIALIST_AUTO_RETRY_MAX`; the failure
        reason is injected into the retry prompt. Disabled when
        ``INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY`` is set to ``0``.

        Args:
            task: The specialist task whose attempt just failed.
            result: The sub-agent result classified for infra-failure
                eligibility.

        Returns:
            ``True`` when a retry was scheduled (caller must skip this
            attempt's bookkeeping); ``False`` otherwise.
        """
        flag = (
            os.environ.get(
                "INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY",
                "1",
            )
            .strip()
            .lower()
        )
        if flag in ("0", "false", "no", "off"):
            return False
        try:
            cap = int(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY_MAX",
                    str(SPECIALIST_AUTO_RETRY_MAX),
                )
            )
        except (TypeError, ValueError):
            cap = SPECIALIST_AUTO_RETRY_MAX
        if cap <= 0:
            return False
        from ..specialists.runner import classify_specialist_failure

        result_dict = result.result if isinstance(result.result, dict) else {}
        runner_status = str(result_dict.get("runner_status") or "")
        # The specialist executor never raises, so the reason lives in the
        # result envelope rather than on SubAgentResult.
        error = str(result.error or result_dict.get("error") or "")
        ftype, retry_eligible = classify_specialist_failure(runner_status, error)
        if not retry_eligible:
            return False
        params = task.params or {}
        attempt = int(params.get("_auto_retry_attempt", 0) or 0)
        if attempt >= cap:
            await self._record_specialist_retry_exhausted(
                task=task,
                ftype=ftype,
                error=error,
                attempts_used=attempt,
                cap=cap,
                detail="retry cap reached",
            )
            return False
        next_attempt = attempt + 1

        retry_params = dict(params)
        retry_params["_auto_retry_attempt"] = next_attempt
        retry_params["_auto_retry_reason"] = f"{ftype.value}: {error}"[:300]

        # Mirror _handle_delegate lane/ttl resolution so the retry task holds the
        # same pools as the original and cannot run concurrently with serving.
        lanes, ttl = self._registry_lanes_ttl("specialist")
        from ..specialists.profile import resolve_specialist_profile, uses_whole_machine_gpu_lane

        if resolve_specialist_profile(retry_params).reserves_benchmark_lane:
            lanes = list(dict.fromkeys((*lanes, "benchmark_lane")))
        needs_gpu_raw = retry_params.get("needs_gpu", False)
        needs_gpu = (
            needs_gpu_raw.strip().lower() in ("1", "true", "yes", "on")
            if isinstance(needs_gpu_raw, str)
            else bool(needs_gpu_raw)
        )
        if not needs_gpu and uses_whole_machine_gpu_lane(retry_params):
            # bench specialist: ensure needs_gpu is set so gpu_research_lane is acquired.
            needs_gpu = True
        if needs_gpu:
            lanes = list(dict.fromkeys((*lanes, "gpu_research_lane")))
            try:
                ttl = self._gpu_lease_ttl_sec(
                    int(ttl or 0),
                    params=retry_params,
                )
            except Exception:  # noqa: BLE001
                log.exception("specialist auto-retry: gpu_research_lane TTL re-source failed; using registry default")

        # Stable base key across attempts: strip any prior ``-autoretryN`` suffix.
        base_key = str(task.idempotency_key or task.task_id or "")
        if "-autoretry" in base_key:
            head, _, tail = base_key.rpartition("-autoretry")
            if tail.isdigit():
                base_key = head
        retry_key = f"{base_key}-autoretry{next_attempt}"

        new_task, was_existing = await self.tasks.create_or_return_existing(
            kind="specialist",
            params=retry_params,
            idempotency_key=retry_key,
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if was_existing:
            # Retry slot already taken: let normal bookkeeping record this attempt.
            await self._record_specialist_retry_exhausted(
                task=task,
                ftype=ftype,
                error=error,
                attempts_used=attempt,
                cap=cap,
                detail="retry slot already taken",
            )
            return False
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "specialist_auto_retry",
                "task_id": task.task_id,
                "retry_task_id": new_task.task_id,
                "attempt": next_attempt,
                "max_attempts": cap,
                "failure_type": ftype.value,
                "reason": error[:200],
            },
        )
        log.info(
            "specialist auto-retry: task=%s failure=%s attempt=%d/%d re-enqueued as %s",
            task.task_id,
            ftype.value,
            next_attempt,
            cap,
            new_task.task_id,
        )
        return True

    async def _warm_specialist_params(self, params: dict[str, Any]) -> None:
        """Fill specialist task params with KnowledgePlane data before enqueue (mutates in place); all best-effort, missing fields stay empty.

        Args:
            params: The specialist task params dict mutated in place with PR
                feed, warm-start, hardware/workload and gap/roofline context.
        """
        state = self.shared_state
        plane = self.knowledge_plane

        from ..specialists.domains import normalize_dispatch_tags
        from ..specialists.profile import resolve_specialist_profile

        # Bench-capable specialists run a real serving + benchmark loop, so
        # default needs_gpu to route them through the gpu_specialist_pool.
        if resolve_specialist_profile(params).reserves_benchmark_lane:
            params.setdefault("needs_gpu", True)

        domain = str(params.get("domain") or "").strip()
        normalize_dispatch_tags(params)

        if "pr_monitor_available" not in params:
            params["pr_monitor_available"] = bool(plane is not None and getattr(plane, "pr_monitor_enabled", True))

        params.setdefault("kb_subgraph", {})

        # Warm-start recipe + pitfalls + lessons from T0 anchor.
        if state.warm_start_recipe and "warm_start_recipe" not in params:
            params["warm_start_recipe"] = dict(state.warm_start_recipe)
        if state.warm_start_pitfalls and "warm_start_pitfalls" not in params:
            params["warm_start_pitfalls"] = list(state.warm_start_pitfalls)
        if state.warm_start_lessons and "warm_start_lessons" not in params:
            params["warm_start_lessons"] = list(state.warm_start_lessons)
        # runtime framework/version for version-mismatch annotation.
        if "framework" not in params:
            fw = str(getattr(state, "framework", "") or "").strip()
            if fw:
                params["framework"] = fw
        if "framework_version" not in params:
            fp_meta = getattr(state, "stack_fingerprint_meta", None) or {}
            if isinstance(fp_meta, dict):
                fw = str(params.get("framework") or getattr(state, "framework", "") or "").lower()
                if fw in ("sglang", "vllm"):
                    v = str(fp_meta.get(fw) or "").strip()
                    if v and v != "unknown":
                        params["framework_version"] = v

        # Local-source navigation hint.
        if "framework_source_roots" not in params:
            try:
                from ..framework.paths import resolve_source_file_allowlist

                roots = resolve_source_file_allowlist()
                if roots:
                    params["framework_source_roots"] = list(roots)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "specialist warmup: framework_source_roots lookup failed: %r",
                    exc,
                )

        # Hardware + workload hints from SharedState; else dataclass defaults win.
        params.setdefault("gpu_type", state.gpu_type or "")
        # Active server framework name.
        if getattr(state, "framework", "") or "":
            params.setdefault("framework", str(state.framework))
        if int(getattr(state, "tp", 0) or 0) > 0:
            params.setdefault("tp", int(state.tp))
        if getattr(state, "precision", "") or "":
            params.setdefault("precision", str(state.precision))
        if int(getattr(state, "conc", 0) or 0) > 0:
            params.setdefault("conc", int(state.conc))
        if int(getattr(state, "isl", 0) or 0) > 0:
            params.setdefault("isl", int(state.isl))
        if int(getattr(state, "osl", 0) or 0) > 0:
            params.setdefault("osl", int(state.osl))
        if int(getattr(state, "max_model_len", 0) or 0) > 0:
            params.setdefault("max_model_len", int(state.max_model_len))

        # Advisory model_arch profile via arch_notes carrier (prompt-context only).
        if "arch_notes" not in params:
            from ..state.shared_state import render_model_arch_compact

            _arch_notes = render_model_arch_compact(getattr(state, "model_arch", None))
            if _arch_notes:
                params["arch_notes"] = _arch_notes

        # Static-recon specialist extras: structured model_info + checklist-derived
        # source-hint directories for the recon focus block.
        if domain == "static_recon_specialist":
            if "model_info" not in params:
                _minfo = getattr(state, "model_info", None)
                if isinstance(_minfo, dict) and _minfo:
                    params["model_info"] = dict(_minfo)
            if "source_hint_directories" not in params:
                try:
                    from ..knowledge import static_recon_checklist as _src_recon

                    _dirs = _src_recon.source_hint_directories_for(
                        model_class=str(getattr(state, "model_class", "") or ""),
                        gpu_type=str(getattr(state, "gpu_type", "") or ""),
                        precision=str(getattr(state, "precision", "") or ""),
                    )
                    if _dirs:
                        params["source_hint_directories"] = list(_dirs)
                except Exception:  # noqa: BLE001 — advisory; never block dispatch
                    log.exception(
                        "static-recon: source_hint_directories lookup failed",
                    )

        if "target_gap_notes" not in params:
            try:
                _gap_notes = self._target_gap_advisory_block()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: specialist target gap advisory failed")
                _gap_notes = ""
            if _gap_notes:
                params["target_gap_notes"] = _gap_notes

        if "research_hints" not in params:
            try:
                from ..knowledge import research_hints as _research_hints

                _hints_block = _research_hints.summarise_for_prompt(
                    self.session_dir,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: specialist research hints failed")
                _hints_block = ""
            if _hints_block:
                params["research_hints"] = _hints_block

        # Fill gap-specific anchors from the gaps[] ledger.
        gap_cid = str(params.get("gap_canonical_id") or "").strip() or str(params.get("gap") or "").strip()
        if gap_cid:
            gap = state.find_gap(gap_cid)
            if gap is not None:
                if not params.get("gap_symptom"):
                    params["gap_symptom"] = str(gap.get("symptom") or "")
                if not params.get("gap_layer"):
                    params["gap_layer"] = str(gap.get("layer") or "")
                if not params.get("domain"):
                    # LLM omitted domain → gap's domain_hint wins.
                    hint = str(gap.get("domain_hint") or "")
                    if hint:
                        params["domain"] = hint
                evidence = params.get("gap_evidence")
                if not isinstance(evidence, dict) or not evidence:
                    attempts = list(gap.get("attempts") or [])[-5:]
                    if attempts:
                        params["gap_evidence"] = {
                            "recent_attempts": attempts,
                            "severity": str(gap.get("severity") or ""),
                        }

        if "baseline_tput" not in params:
            _bt = float(getattr(state, "baseline_tput", 0.0) or 0.0)
            if _bt > 0:
                params["baseline_tput"] = _bt
        if "current_tput" not in params:
            cb = getattr(state, "current_best", None)
            _ct = float((cb.get("tput") if isinstance(cb, dict) else 0) or 0.0)
            if _ct > 0:
                params["current_tput"] = _ct
        if "cumulative_gain_validated" not in params:
            _cgv = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
            if _cgv != 0:
                params["cumulative_gain_validated"] = _cgv
        if "keep_threshold_pct" not in params:
            params["keep_threshold_pct"] = _phase_state.resolve_keep_threshold(state)
        if "applied_stack" not in params:
            _stack = list(getattr(state, "optimization_stack", None) or [])
            if _stack:
                params["applied_stack"] = [
                    {"variant_name": str(e.get("variant_name") or ""), "gain_pct": float(e.get("gain_pct") or 0.0)}
                    for e in _stack
                    if isinstance(e, dict)
                ]

        # Pack bottleneck signals into roofline_evidence for the specialist.
        # Hot kernels alone are enough: a trace whose quality gate withheld
        # analysis.md still names where device time goes.
        last_ta = getattr(state, "last_trace_analyze", None) or {}
        has_evidence = isinstance(last_ta, dict) and bool(
            last_ta.get("analysis_md_text") or last_ta.get("hot_kernels_top15")
        )
        if has_evidence and "roofline_evidence" not in params:
            from ..kernel.roofline_snapshot import extract_workload_summary

            analysis_path = str(last_ta.get("analysis_md_path") or "")
            executive_summary: dict[str, Any] = {}
            if analysis_path:
                try:
                    executive_summary = extract_workload_summary(analysis_path)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "specialist warmup: extract_workload_summary(%s) failed: %r",
                        analysis_path,
                        exc,
                    )
                    executive_summary = {}
            hot_kernels = list(last_ta.get("hot_kernels_top15") or [])[:8]
            params["roofline_evidence"] = {
                "analysis_md_path": analysis_path,
                "roofline_snapshot_id": last_ta.get("roofline_snapshot_id"),
                "executive_summary": executive_summary,
                "hot_kernels_top15": hot_kernels,
            }

    async def _refresh_gaps(self, *, reason: str) -> None:
        """Refresh :attr:`SharedState.gaps` from observable signals. Additive upsert deduped by canonical_id; best-effort.

        Args:
            reason: Tag describing the refresh trigger, used only in logging.
        """
        state = self.shared_state
        try:
            for entry in self._extract_gaps_from_baseline():
                state.upsert_gap(entry)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("gaps refresh: baseline extraction failed")
        try:
            for entry in self._extract_gaps_from_attempts():
                state.upsert_gap(entry)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("gaps refresh: attempts extraction failed")

        plane = getattr(self, "knowledge_plane", None)
        if plane is not None and hasattr(plane, "recipe_kb_traverse_issues"):
            try:
                traverse = getattr(plane, "recipe_kb_traverse_issues")
                rows = traverse(
                    model_class=getattr(state, "model_class", "") or "",
                    gpu_type=getattr(state, "gpu_type", "") or "",
                )
                if isinstance(rows, list):
                    for entry in rows:
                        if isinstance(entry, dict):
                            entry = dict(entry)
                            entry.setdefault("source", "recipe_kb")
                            state.upsert_gap(entry)
            except Exception:  # noqa: BLE001 — defensive
                log.warning(
                    "gaps refresh: recipe_kb_traverse_issues failed (reason=%s)",
                    reason,
                    exc_info=True,
                )
        log.debug(
            "gaps refresh (reason=%s): %d gaps after merge",
            reason,
            len(state.gaps),
        )

    def _extract_gaps_from_baseline(self) -> list[dict[str, Any]]:
        """Derive initial gap rows from the baseline snapshot (throughput_below_target, baseline_unstable); reuse the workload canonical_id (``_workload_canonical_id``, matching ``recipe_kb_t0.run_t0_anchor``) so traverse rows align.

        Returns:
            A list of gap row dicts derived from the baseline; empty when no
            baseline throughput is recorded.
        """
        state = self.shared_state
        gaps: list[dict[str, Any]] = []
        if state.baseline_tput <= 0:
            return gaps
        anchor = self._workload_canonical_id()
        target_gap = float(getattr(state, "target_gap_pct", 0.0) or 0.0)
        if target_gap > 0.0:
            severity = "high" if target_gap >= 10.0 else "medium" if target_gap >= 3.0 else "low"
            gaps.append(
                {
                    "canonical_id": f"{anchor}#throughput_below_target",
                    "symptom": (f"current_best is {target_gap:.1f}% short of the run objective target"),
                    "layer": "framework",
                    "severity": severity,
                    "domain_hint": self._framework_authoring_domain(),
                    "source": "baseline",
                }
            )
        if state.baseline_failure_streak > 0:
            gaps.append(
                {
                    "canonical_id": f"{anchor}#baseline_unstable",
                    "symptom": (f"baseline crashed {state.baseline_failure_streak} consecutive time(s)"),
                    "layer": "system",
                    "severity": ("high" if state.baseline_failure_streak >= 2 else "medium"),
                    "domain_hint": "system_specialist",
                    "source": "baseline",
                }
            )
        return gaps

    def _extract_gaps_from_attempts(self) -> list[dict[str, Any]]:
        """Derive gaps from rolling failures + winners history (recurring (action, error_class[, variant]) + explore plateau).

        Returns:
            A list of gap row dicts derived from recurring action failures and
            an explore-plateau signal.
        """
        state = self.shared_state
        anchor = self._workload_canonical_id()
        gaps: list[dict[str, Any]] = []

        # Already capped by ``record_action_failure``; read the whole log.
        seen_failures: dict[str, dict[str, Any]] = {}
        for row in state.last_action_failures or []:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").strip() or "unknown"
            err = str(row.get("error_class") or "").strip() or "unknown_error"
            variant = str(row.get("variant_name") or "").strip()
            # Variant discriminator keeps distinct crash causes in distinct gaps.
            key = f"{action}::{err}::{variant}" if variant else f"{action}::{err}"
            layer, domain = self._gap_layer_for_action(action, str(getattr(self.shared_state, "framework", "") or ""))
            excerpt = str(row.get("error_excerpt") or "")
            detail = next((ln.strip() for ln in excerpt.splitlines() if ln.strip()), err)[:200]
            symptom = f"{action}/{variant} fails: {detail}" if variant else f"{action} repeatedly fails with {detail}"
            attempt = {
                "action": action,
                "variant_name": variant,
                "outcome": "REVERT",
                "error_class": err,
                "ts": str(row.get("ts") or datetime.now(timezone.utc).isoformat()),
            }
            if key in seen_failures:
                seen_failures[key]["attempts"].append(attempt)
            else:
                cid_variant = f":{variant}" if variant else ""
                seen_failures[key] = {
                    "canonical_id": f"{anchor}#fail:{action}:{err}{cid_variant}",
                    "symptom": symptom,
                    "layer": layer,
                    "severity": "medium",
                    "domain_hint": domain,
                    "source": "attempts",
                    "attempts": [attempt],
                }
        gaps.extend(seen_failures.values())

        no_promote = int(state.params_no_promote_streak or 0)
        explore_search = state.explore_search or {}
        winners_hist = []
        if isinstance(explore_search, dict):
            winners_hist = list(explore_search.get("winners_history") or [])
        recent_promotions = sum(
            1 for w in winners_hist[-5:] if isinstance(w, dict) and float(w.get("gain_pct") or 0.0) > 0.0
        )
        if no_promote >= 3 and recent_promotions == 0:
            gaps.append(
                {
                    "canonical_id": f"{anchor}#explore_plateau",
                    "symptom": (f"{no_promote} consecutive grid rounds without a new current_best"),
                    "layer": "framework",
                    "severity": "high" if no_promote >= 6 else "medium",
                    "domain_hint": self._framework_authoring_domain(),
                    "source": "attempts",
                }
            )
        return gaps

    def _framework_authoring_domain(self) -> str:
        """Return the authoring domain matching this session's framework kind.

        Returns:
            str: ``"framework_rewrite_specialist"`` for a scriptable framework,
            else ``"serving_specialist"``.
        """
        from ..specialists.domains import authoring_domain_for_framework

        return authoring_domain_for_framework(getattr(self.shared_state, "framework", ""))

    @staticmethod
    def _gap_layer_for_action(action: str, framework: str = "") -> tuple[str, str]:
        """Map an action name → (layer, domain_hint) for gap rows.

        Args:
            action: The action name to classify.
            framework: The session's framework, which decides the authoring
                domain for framework-layer rows. Defaults to the serving domain
                so a caller with no framework in hand keeps the old mapping.

        Returns:
            A ``(layer, domain_hint)`` tuple for the action.
        """
        from ..specialists.domains import authoring_domain_for_framework

        a = str(action or "").strip().lower()
        if a in {
            "kernel_opt",
            "integrate",
            "trace_analyze",
            "run_gemm_tuning",
            "run_optimization",
            "profile",
            "roofline",
        }:
            return ("kernel_agent", "kernel_switch_specialist")
        if a in {"baseline"}:
            return ("system", "system_specialist")
        return ("framework", authoring_domain_for_framework(framework))

    def _record_explore_round_gaps(
        self,
        *,
        task: "Task | None",
        result: dict[str, Any],
    ) -> None:
        """Append per-variant KEEP/REVERT outcomes to the matching gap (or the anchor gap as fallback).

        Args:
            task: The explore task whose params carry the gap canonical id;
                ``None`` is a no-op.
            result: The explore result; its ``per_variant_outcomes`` drive the
                appended gap attempts.
        """
        if task is None:
            return
        per_variant = result.get("per_variant_outcomes")
        if not isinstance(per_variant, list) or not per_variant:
            return
        params = dict(task.params or {})
        canonical = str(params.get("gap_canonical_id") or "").strip() or self._workload_canonical_id()
        state = self.shared_state
        existing = state.find_gap(canonical)
        if existing is None:
            state.upsert_gap(
                {
                    "canonical_id": canonical,
                    "symptom": "explore round outcomes",
                    "layer": "framework",
                    "severity": "medium",
                    "domain_hint": self._framework_authoring_domain(),
                    "source": "attempts",
                }
            )
        for outcome in per_variant:
            if not isinstance(outcome, dict):
                continue
            attempt: dict[str, Any] = {
                "action": "explore",
                "variant_name": str(outcome.get("variant_name") or ""),
                "outcome": str(outcome.get("outcome") or "").upper(),
                "gain_pct": outcome.get("gain_pct"),
                "reason": str(outcome.get("reason") or ""),
                "error_class": str(outcome.get("error_class") or ""),
            }
            for key in _GAP_ATTEMPT_ARTIFACT_KEYS:
                value = outcome.get(key)
                if value:
                    attempt[key] = str(value)
            state.append_gap_attempt(canonical, attempt)

    def _record_explore_variant_failures(
        self,
        *,
        task: "Task | None",
        result: dict[str, Any],
    ) -> None:
        """Record each unmeasured ``per_variant_outcomes`` row as failure evidence + ``last_action_failures``.

        A crashed variant does not fail the round, so the round-level recorder
        never sees it.

        Args:
            task: The completed explore task; ``None`` is a no-op.
            result: The explore result dict carrying ``per_variant_outcomes``.
        """
        if task is None:
            return
        per_variant = result.get("per_variant_outcomes")
        if not isinstance(per_variant, list):
            return
        task_id = str(task.task_id or "")
        round_id = str(result.get("round_id") or "")
        for vo in per_variant:
            if not isinstance(vo, dict):
                continue
            if str(vo.get("outcome") or "").upper() not in UNMEASURED_OUTCOMES:
                continue
            fe = failure_from_variant_outcome(task_id=task_id, round_id=round_id, vo=vo)
            self.shared_state.record_failure_evidence(fe)
            self.shared_state.record_action_failure(
                action="explore",
                task_id=task_id,
                result={
                    "variant_name": str(vo.get("variant_name") or ""),
                    "error_class": str(vo.get("error_class") or ""),
                    "error": str(vo.get("reason") or ""),
                    "workspace": vo.get("workspace"),
                    "stderr_log_path": vo.get("server_log_path"),
                    "failure_id": fe.get("failure_id"),
                },
            )

    @staticmethod
    def _task_id_from_specialist_source(source: str) -> str:
        """Extract the task_id from a ``specialist:<task_id>`` source ("" when prefix is absent).

        Args:
            source: The from-agent string to parse.

        Returns:
            The task id when the specialist prefix is present, else ``""``.
        """
        if not source:
            return ""
        if source.startswith(SPECIALIST_FROM_AGENT_PREFIX):
            return source[len(SPECIALIST_FROM_AGENT_PREFIX) :]
        return ""

    async def _maybe_materialize_mn_explore(
        self,
        *,
        task: Task,
        domain: str,
        proposals: list[Any],
    ) -> None:
        """Multi-node bridge: turn a specialist ``proposal_set`` into a
        benchmarked ``explore`` task automatically.

        Single-node is a no-op (``is_multi_node()`` False): there the
        Orchestration LLM drives ``explore`` directly. In multi-node the GPU
        cluster lives on remote SSH pods, so the only materialisation channel is
        a structured ``explore`` action; this helper enqueues the explore grid
        itself. ``proposal_set`` entries reuse the explore variant schema
        (``name`` / ``extra_args`` / ``extra_envs``) and pass straight through;
        ``canonical_fingerprint`` dedup + the per-variant KEEP/REVERT gain gate
        are the safety net.

        Args:
            task: The completed specialist task whose id seeds the explore
                idempotency key.
            domain: The specialist domain, stamped onto variant provenance.
            proposals: The specialist ``proposal_set`` entries materialised into
                the explore grid (capped at ``_MN_AUTO_EXPLORE_GRID_CAP``).
        """
        # Framework config-generation specialists own their proposal_set; skip.
        if bool((getattr(task, "params", None) or {}).get("framework_config_generation")):
            return
        from ..actions.executors._multi_node_env import is_multi_node
        from ..actions.executors._proposal_identity import controls_of, is_executable, normalize_proposal

        if not is_multi_node() or not proposals:
            return
        grid: list[dict[str, Any]] = []
        for i, p in enumerate(proposals[: self._MN_AUTO_EXPLORE_GRID_CAP]):
            if not isinstance(p, dict):
                continue
            fields = normalize_proposal(p)
            if not is_executable(fields):
                continue
            grid.append(
                {
                    "name": fields["name"] or f"{domain or 'specialist'}-{task.task_id[:8]}-{i}",
                    "extra_args": fields["extra_args"],
                    "extra_envs": fields["extra_envs"],
                    **controls_of(fields),
                    "provenance": f"specialist:{domain}" if domain else "specialist",
                    "note": fields["reason"][:200],
                }
            )
        if not grid:
            return
        state = self.shared_state
        params: dict[str, Any] = {
            "source": "coordinator_internal_mn",
            "reason": f"mn_auto_materialize:{domain or 'specialist'}",
            "grid": grid,
        }
        if state.baseline_config_path:
            params["config_path"] = state.baseline_config_path
        inject_stack_base_params(params, state, anchor=True)
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        try:
            lanes, ttl = self._registry_lanes_ttl("explore")
            etask, was_existing = await self.tasks.create_or_return_existing(
                kind="explore",
                params=params,
                idempotency_key=f"mn-auto-explore-{task.task_id}",
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
            )
            log.info(
                "mn_auto_materialize: enqueued explore task_id=%s "
                "(variants=%d, from specialist=%s domain=%s, existing=%s)",
                etask.task_id,
                len(grid),
                task.task_id,
                domain,
                was_existing,
            )
        except Exception:  # noqa: BLE001 — defensive; never block bookkeeping
            log.exception(
                "mn_auto_materialize: failed to enqueue explore from specialist=%s domain=%s",
                task.task_id,
                domain,
            )

    async def _maybe_autosubmit_specialist_patches(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any],
    ) -> None:
        """Auto-surface a specialist's source patches to the Critic via a synthetic integrate_patch proposal; idempotent per specialist.

        Args:
            task: The completed specialist task whose worktree patches are
                surfaced.
            done_payload: The specialist done payload carrying
                ``patches_written`` and proposal metadata.
        """
        patches = done_payload.get("patches_written") or []
        if not isinstance(patches, list):
            patches = []
        sid = str(task.task_id or "").strip()
        if not sid:
            return
        # Resolve patches_written; submit only when >=1 real file exists.
        from hyperloom.inference_optimizer.session.session_paths import runs_dir as _runs_dir
        from ..loop.coordinator import _resolvable_artifacts_from_done

        resolve_bases: list[Path] = []
        if self.session_dir is not None:
            spec_root = _runs_dir(Path(self.session_dir), "specialist", sid)
            resolve_bases = [spec_root / "worktree", spec_root]
        existing_patches: list[str] = []
        for p in patches:
            raw = Path(str(p))
            cands = [raw] if raw.is_absolute() else []
            for base in resolve_bases:
                cands.append(base / raw)
            if any(c.is_file() for c in cands):
                existing_patches.append(str(p))
        # A non-diff tuned artifact is also a routable deliverable; route it like
        # a patch.
        routable_artifacts = _resolvable_artifacts_from_done(done_payload, resolve_bases)
        if not existing_patches and not routable_artifacts:
            if patches:
                await self._record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "specialist_patch_autosubmit_skipped_no_files",
                        "specialist_task_id": sid,
                        "claimed": [str(x) for x in patches][:8],
                    },
                )
            return
        # Already ruled on by the Critic (e.g. after resume) — nothing to do.
        try:
            if self.shared_state.get_specialist_patch_verdict(sid):
                return
        except Exception:  # noqa: BLE001 — defensive
            pass
        # A synthetic review for this specialist is already in flight.
        for p in self.state.pending_proposals.values():
            try:
                if getattr(p, "action_name", "") != "integrate_patch":
                    continue
                pl = getattr(p, "payload", {}) or {}
                if (pl.get("params") or {}).get("specialist_task_id") == sid:
                    return
            except Exception:  # noqa: BLE001 — defensive
                continue
        proposals = done_payload.get("proposal_set") or []
        patch_name = ""
        if isinstance(proposals, list) and proposals:
            patch_name = str((proposals[0] or {}).get("name") or "")
        spec_params = getattr(task, "params", None) or {}
        integrate_params: dict[str, Any] = {
            "specialist_task_id": sid,
            "provenance": "specialist",
            "patch_name": patch_name,
        }
        _forward_integrate_source(
            spec_params,
            integrate_params,
        )
        # FRAMEWORK authoring provenance passthrough: propagate the PR
        # candidate/batch id onto the synthetic integrate_patch task so the
        # authored-outcome bridge keys the progress row on the real candidate id.
        try:
            if bool(spec_params.get("framework_agent_authoring")):
                integrate_params["framework_agent_authoring"] = True
                fa_cand = str(spec_params.get("framework_agent_candidate_id") or "")
                fa_batch = str(spec_params.get("framework_batch_id") or "")
                if fa_cand:
                    integrate_params["framework_agent_candidate_id"] = fa_cand
                if fa_batch:
                    integrate_params["framework_batch_id"] = fa_batch
            # Propagate the enablement marker (+ optional launch probe) so
            # integrate_patch applies the runnable_decision gate.
            if bool(spec_params.get("enablement")):
                integrate_params["enablement"] = True
                _forward_enablement_carriers(spec_params, integrate_params)
                probe = str(spec_params.get("launch_probe") or "").strip()
                if probe:
                    integrate_params["launch_probe"] = probe
                # Forward the pre-patch failure signature for the runnable gate.
                before_sig = spec_params.get("enablement_before_signature")
                if isinstance(before_sig, dict):
                    integrate_params["enablement_before_signature"] = before_sig
                # Forward the stacked base patches for integrate_patch to re-apply.
                base_patches = spec_params.get("enablement_base_patches")
                if isinstance(base_patches, list) and base_patches:
                    integrate_params["enablement_base_patches"] = [str(p) for p in base_patches]
                # Forward stacked base setup commands to replay before boot.
                base_setup = spec_params.get("enablement_setup_commands")
                if isinstance(base_setup, list) and base_setup:
                    integrate_params["enablement_setup_commands"] = [str(c) for c in base_setup]
        except Exception:  # noqa: BLE001 — provenance passthrough is best-effort
            log.debug(
                "FRAMEWORK: authoring provenance passthrough failed for task=%s",
                sid,
                exc_info=True,
            )
        propose_payload = {
            "action_name": "integrate_patch",
            "provenance": "specialist",
            "predicted_gain_pct": 0.0,
            "params": integrate_params,
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
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "specialist_patch_autosubmitted_for_review",
                "specialist_task_id": sid,
                "proposal_msg_id": msg.msg_id,
                "patch_name": patch_name,
                "patches": [str(x) for x in patches][:8],
                # Artifact-only deliverables: record their install targets.
                "artifacts_written": [
                    str((a or {}).get("target") or "")
                    for a in (done_payload.get("artifacts_written") or [])
                    if isinstance(a, dict)
                ][:8],
            },
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "save after specialist patch autosubmit failed for task=%s",
                sid,
            )

    async def _maybe_autosubmit_framework_config(
        self,
        *,
        task: "Task",
        done_payload: dict[str, Any],
    ) -> None:
        """Route a FRAMEWORK config-lever deliverable through integrate_patch.

        Companion to :meth:`_maybe_autosubmit_specialist_patches`: fires when a
        FRAMEWORK authoring specialist returns NO source patch but a config-lever
        ``proposal_set`` (extra_args / extra_envs). The levers go into
        integrate_patch's ``config_changes`` channel (apply + bench + accuracy
        gate + KEEP/REVERT), which owns the terminal FRAMEWORK row. Idempotent
        per specialist.

        Args:
            task: The completed authoring specialist task.
            done_payload: Its ``specialist_done`` payload.
        """
        spec_params = getattr(task, "params", None) or {}
        if not bool(spec_params.get("framework_agent_authoring")):
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
        # Normally route only when there are config levers to test. For an
        # ENABLEMENT round ALWAYS route (even a setup-only or empty deliverable):
        # integrate_patch owns the enablement stall accounting, so an enablement
        # round must reach it to bump ``enablement_stall_streak`` / clear
        # ``enablement_stall_streak`` and eventually fire ``enablement_stalled``.
        # Non-enablement config deliverables keep the strict "levers required" gate.
        if not config_levers and not is_enablement:
            return
        sid = str(task.task_id or "").strip()
        if not sid:
            return
        if config_levers and self._config_lever_known_bad(config_levers):
            return
        # Already ruled on (e.g. after resume) — nothing to do.
        try:
            if self.shared_state.get_specialist_patch_verdict(sid):
                return
        except Exception:  # noqa: BLE001 — defensive
            pass
        # A synthetic review for this specialist is already in flight.
        for p in self.state.pending_proposals.values():
            try:
                if getattr(p, "action_name", "") != "integrate_patch":
                    continue
                pl = getattr(p, "payload", {}) or {}
                if (pl.get("params") or {}).get("specialist_task_id") == sid:
                    return
            except Exception:  # noqa: BLE001 — defensive
                continue
        proposals = done_payload.get("proposal_set") or []
        patch_name = ""
        if isinstance(proposals, list) and proposals and isinstance(proposals[0], dict):
            patch_name = str(proposals[0].get("name") or "")
        integrate_params: dict[str, Any] = {
            "specialist_task_id": sid,
            "provenance": "specialist",
            "patch_name": patch_name,
            "extra_server_args": str(config_levers.get("extra_server_args") or ""),
            "extra_envs": dict(config_levers.get("extra_envs") or {}),
        }
        _forward_integrate_source(
            spec_params,
            integrate_params,
        )
        # FRAMEWORK authoring provenance passthrough for the authored-outcome bridge.
        fa_cand = str(spec_params.get("framework_agent_candidate_id") or "")
        fa_batch = str(spec_params.get("framework_batch_id") or "")
        integrate_params["framework_agent_authoring"] = True
        if fa_cand:
            integrate_params["framework_agent_candidate_id"] = fa_cand
        if fa_batch:
            integrate_params["framework_batch_id"] = fa_batch
        # Enablement passthrough (mirrors _maybe_autosubmit_specialist_patches): a
        # config-lever-only enablement deliverable MUST still flow the enablement
        # marker + setup_commands into integrate_patch, otherwise the result never
        # carries ``enablement=True``, ``_maybe_rearm_enablement`` no-ops, and
        # the enablement stall streak is only advanced via _maybe_rearm_enablement.
        if bool(spec_params.get("enablement")):
            integrate_params["enablement"] = True
            _forward_enablement_carriers(spec_params, integrate_params)
            probe = str(spec_params.get("launch_probe") or "").strip()
            if probe:
                integrate_params["launch_probe"] = probe
            before_sig = spec_params.get("enablement_before_signature")
            if isinstance(before_sig, dict):
                integrate_params["enablement_before_signature"] = before_sig
            base_patches = spec_params.get("enablement_base_patches")
            if isinstance(base_patches, list) and base_patches:
                integrate_params["enablement_base_patches"] = [str(p) for p in base_patches]
            # Merge the stacked base setup commands with any NEW setup_commands the
            # specialist just proposed in this deliverable (e.g. a stack upgrade),
            # so a config-lever-only enablement round actually replays the install
            # step before booting instead of silently dropping it.
            merged_setup: list[str] = []
            for c in spec_params.get("enablement_setup_commands") or []:
                sc = str(c)
                if sc and sc not in merged_setup:
                    merged_setup.append(sc)
            for c in done_payload.get("setup_commands") or []:
                sc = str(c)
                if sc and sc not in merged_setup:
                    merged_setup.append(sc)
            if merged_setup:
                integrate_params["enablement_setup_commands"] = merged_setup
        propose_payload = {
            "action_name": "integrate_patch",
            "provenance": "specialist",
            "predicted_gain_pct": 0.0,
            "params": integrate_params,
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
        await self._record_observation(
            "coordinator",
            "observation",
            {
                "kind": "framework_config_autosubmitted_for_review",
                "specialist_task_id": sid,
                "proposal_msg_id": msg.msg_id,
                "candidate_id": fa_cand,
                "extra_server_args": integrate_params["extra_server_args"],
                "extra_envs": dict(integrate_params["extra_envs"]),
            },
        )
        log.info(
            "FRAMEWORK: config-lever deliverable routed to integrate_patch candidate=%s args=%s env_keys=%s",
            fa_cand or sid,
            integrate_params["extra_server_args"],
            sorted(integrate_params["extra_envs"]),
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "FRAMEWORK: save after config autosubmit failed for task=%s",
                sid,
            )

    def _config_lever_known_bad(self, config_levers: dict[str, Any]) -> bool:
        """True when this exact config already lost an accuracy gate.

        Different upstream PRs often reduce to the same server args / envs, so
        the ledger is keyed by content fingerprint rather than by PR.
        """
        from ..actions.executors._canonical_fingerprint import canonical_fingerprint

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
        except Exception:  # noqa: BLE001 — advisory gate must never block dispatch
            # Warning, not debug: swallowing this re-dispatches config levers
            # that already lost an accuracy gate, so it must be visible.
            log.warning("FRAMEWORK: config-lever ledger check failed", exc_info=True)
        return False

    def _build_specialist_round_entry(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
        run_error: str = "",
    ) -> dict[str, Any]:
        """Translate a specialist done payload into a SharedState.specialist_rounds[] row; round_id defaults to task_id for idempotent overwrite.

        Args:
            task: The completed specialist task.
            done_payload: The specialist done payload (proposal_set, domain,
                tags, summary, etc.).
            source: The emitting agent string, recorded on the row.
            run_error: Dispatch failure text when no valid payload was produced.

        Returns:
            A specialist-round row dict suitable for
            ``SharedState.record_specialist_round``.
        """
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        task_params = task.params or {}
        round_id = str(task_params.get("round_id") or task.task_id)
        source_phase = (
            str(
                task_params.get("source_phase")
                or done_payload.get("source_phase")
                or getattr(getattr(self, "shared_state", None), "phase", "")
                or ""
            )
            .strip()
            .upper()
        )
        from ..specialists.domains import normalize_dispatch_tags

        # Knowledge-domain tags; reported tags win over dispatch params.
        tags = normalize_dispatch_tags(done_payload)
        if not tags:
            tags = normalize_dispatch_tags(task.params or {})
        entry: dict[str, Any] = {
            "round_id": round_id,
            "task_id": task.task_id,
            "source": source or "coordinator",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "domain": str(done_payload.get("domain") or task_params.get("domain") or ""),
            "tags": list(tags),
            "gap_canonical_id": str(done_payload.get("gap_canonical_id") or task_params.get("gap_canonical_id") or ""),
            "empty": bool(done_payload.get("empty")) or len(proposals) == 0,
            "proposals_total": len(proposals),
            "proposal_set": list(proposals),
            "summary": str(done_payload.get("summary") or "")[:480],
            "reason": str(run_error or done_payload.get("reason") or "")[:480],
            "confidence": done_payload.get("confidence"),
            "new_findings": list(done_payload.get("new_findings") or []),
            "residual_questions": list(done_payload.get("residual_questions") or []),
        }
        for key in (
            "task_kind",
            "scope",
            "proposal_msg_id",
            "framework_agent_candidate_id",
            "framework_batch_id",
            "reauthor_attempt",
            "apply_retry_attempt",
        ):
            value = done_payload.get(key)
            if value in (None, "", [], {}):
                value = task_params.get(key)
            if value not in (None, "", [], {}):
                entry[key] = value
        for key in ("candidate_discovery", "framework_agent_authoring"):
            if bool(done_payload.get(key) or task_params.get(key)):
                entry[key] = True
        if run_error:
            entry["status"] = "failed"
            entry["error"] = str(run_error)[:1000]
            entry["run_error"] = str(run_error)[:1000]
        elif done_payload.get("status") not in (None, ""):
            entry["status"] = str(done_payload.get("status"))
        if source_phase:
            entry["source_phase"] = source_phase
        gpu_ids = done_payload.get("allocated_gpu_ids") or []
        if isinstance(gpu_ids, list) and gpu_ids:
            entry["allocated_gpu_ids"] = [
                int(g) for g in gpu_ids if isinstance(g, (int, str)) and str(g).strip().lstrip("-").isdigit()
            ]
        specialist_notes = done_payload.get("_specialist_notes") or []
        if isinstance(specialist_notes, list) and specialist_notes:
            entry["notes"] = [str(n) for n in specialist_notes]
        return entry
