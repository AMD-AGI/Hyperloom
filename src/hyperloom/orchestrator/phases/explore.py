# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""EXPLORE phase handler: macro-cycle strategy, specialist fan-out/retry, gap
tracking, and autosubmit of specialist patches / framework configs."""

from __future__ import annotations
import logging as _logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from . import machine_state as _phase_state
from ..loop.coordinator_helpers import _parse_iso_unix
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from ..bus.message_bus import Message
from ..policy.gate import (
    SPECIALIST_FROM_AGENT_PREFIX,
)
from ..loop.sub_agent_runner import SubAgentResult
from ..state.task_registry import Task
from ..loop.coordinator import (
    FORCE_STALLED_KEEP_ROUNDS,
    FORCE_STALLED_SPECIALIST_ROUNDS,
    PendingProposal,
    SPECIALIST_AUTO_RETRY_MAX,
    _framework_config_levers_from_done,
)
from .base import PhaseHandler

log = _logging.getLogger(__name__)


class ExplorePhase(PhaseHandler):
    """Extracted phase handler; delegates unknown attrs to its Coordinator."""

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
                    reasons[d].append(
                        f"saturated at {row.get('within_pct')}% within roofline; deprioritized"
                    )
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

    def _apply_macro_cycle_reloop(self, evidence: dict[str, Any]) -> None:
        """Open a new macro-cycle on a SWEEP loopback (to FRAMEWORK or EXPLORE).

        Increments ``macro_cycle``, persists the no-gain streak + the per-cycle
        gain anchor, resets per-cycle counters (including re-opening FRAMEWORK)
        so the new cycle gets a fresh budget / plateau evaluation. The explore
        ledger is preserved; its already-KEEP entries stay blocked while
        sub-threshold ones may unblock as the KEEP bar decays.

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
        # Carry the effective no-gain streak computed by should_reloop.
        if isinstance(evidence, dict) and "no_gain_cycle_streak_effective" in evidence:
            state.no_gain_cycle_streak = int(evidence.get("no_gain_cycle_streak_effective", 0) or 0)
        # Anchor gain for the cycle we are about to start.
        try:
            state.gain_at_cycle_start = float(getattr(state, "cumulative_gain_validated", 0.0) or 0.0)
        except (TypeError, ValueError):
            state.gain_at_cycle_start = 0.0
        # Reset per-cycle counters (fresh plateau/dispatch budget for the cycle).
        try:
            state.reset_specialist_dispatched()
            state.reset_explore_plateau_proxy()
        except Exception:  # noqa: BLE001 — resets are best-effort
            log.exception("Coordinator: per-cycle reset failed on reloop")
        # Re-open FRAMEWORK for the new cycle so the loopback target does not
        # instantly self-skip as "already done". Already-tested PRs are still
        # skipped: framework_agent_batches (and the per-candidate progress rows that
        # dedup within them) are preserved, so a fresh discover only surfaces PRs
        # merged upstream since, and fast-exits via
        # ``discover_returned_no_new_candidates`` when there are none.
        state.framework_agent_phase_done = False
        state.framework_agent_discover_failures = 0
        # Reset the config-exploration guard so each macro-cycle re-runs the
        # framework config lane (default OFF; no-op unless enabled).
        state.framework_config_lane_state = ""
        state.framework_config_lane_round = 0
        state.framework_config_pending_grid = []
        # Mark a macro-cycle boundary in the preserved progress ledger so the
        # consecutive-no-keep plateau gate (``_framework_agent_consecutive_no_keep``)
        # does NOT carry the prior cycle's trailing no-KEEP streak into this
        # cycle. Without this, FRAMEWORK_AGENT plateaus the instant it re-enters
        # (the cycle-0 no-KEEP rows already satisfy the streak threshold), so the
        # new cycle's candidate never gets a fair evaluation. The marker carries
        # no candidate_id, so candidate dedup (_unprocessed_framework_agent_candidates)
        # is unaffected; kept=False keeps KEEP-count reporting honest.
        try:
            progress = getattr(state, "framework_agent_phase_progress", None)
            if not isinstance(progress, list):
                progress = []
                state.framework_agent_phase_progress = progress
            if not (progress and isinstance(progress[-1], dict) and str(progress[-1].get("status") or "") == "cycle_boundary"):
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
            log.exception("Coordinator: framework_agent cycle_boundary marker append failed")
        # Clear the per-cycle SWEEP completion markers: exit_normal_sweep keys off
        # last_sweep / last_conc_sweep status, so a stale "succeeded" from the
        # prior cycle would make the next cycle's SWEEP exit instantly without
        # running a fresh sweep. (current_best / optimization_stack are global
        # and intentionally preserved.)
        state.last_sweep = {}
        state.last_conc_sweep = {}
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

        Brings the single-session run the per-session restart benefits (fresh
        leases, pruned DB, cleared transient caches, compacted-memory
        conversation reset) without losing accumulated optimization state.
        ``current_best`` / ``optimization_stack`` / ``explore_search`` (the
        negative ledger) are GLOBAL and deliberately preserved here — only
        transient / per-cycle resources recycle, and the routine is idempotent
        so a resume mid-restart never double-cleans or replays tasks.

        Best-effort: every step is independently guarded so one failure never
        aborts the run loop. Returns a summary dict when it ran, else ``None``.

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
        # 1) Compact the just-finished cycle's conversation into durable memory
        #    and reset so the new cycle reseeds from the compressed seed instead
        #    of dragging the full transcript across the boundary.
        try:
            compacted = await self._maybe_checkpoint_orchestration(
                tick=int(getattr(self.shared_state, "tick", 0) or 0),
                phase_changed=True,
                force=True,
            )
            summary["memory_compacted"] = bool(compacted)
            # Reset unconditionally so even a no-op checkpoint (e.g. mock backend)
            # still reseeds from the latest compacted memory next turn.
            self._reset_orchestration_conversation()
            summary["conversation_reset"] = True
        except Exception:  # noqa: BLE001 — soft restart never aborts the run loop
            log.exception("cycle soft-restart: conversation reset failed")
        # 2) Reap TTL-expired serving + GPU leases immediately (don't wait for
        #    the maintenance cadence) so the new cycle starts on fresh capacity.
        try:
            reaped = await self.locks.reap_expired()
            summary["leases_reaped"] = len(reaped or [])
        except Exception:  # noqa: BLE001
            log.exception("cycle soft-restart: serving-lease reap failed")
        try:
            summary["gpu_leases_reaped"] = await self.gpu_specialist_pool.reap_expired()
        except Exception:  # noqa: BLE001
            log.exception("cycle soft-restart: gpu-lease reap failed")
        # 2b) Reclaim orphaned running tasks (lease expired) → failed so they
        #     free their lanes and stay retry-eligible. Idempotent.
        try:
            reclaimed = await self.tasks.reclaim_expired_running(
                reason="cycle_soft_restart",
            )
            summary["running_tasks_reclaimed"] = len(reclaimed)
        except Exception:  # noqa: BLE001
            log.exception("cycle soft-restart: running-task reclaim failed")
        # 3) Prune the events/tasks DB (strictly below the resume anchor).
        try:
            from ..bus import db_maintenance as _db_maint

            res = await _db_maint.run_db_retention(self.db, self.cursors)
            summary["events_pruned"] = res.events_deleted
            summary["tasks_pruned"] = res.tasks_deleted
        except Exception:  # noqa: BLE001
            log.exception("cycle soft-restart: DB retention failed")
        # 4) Clear transient knowledge-plane caches so the new cycle pulls a
        #    fresh PR feed instead of reusing the prior cycle's window.
        try:
            if self.knowledge_plane is not None:
                self.knowledge_plane.reset_round_caches()
                summary["caches_cleared"] = True
        except Exception:  # noqa: BLE001
            log.exception("cycle soft-restart: cache clear failed")
        # 5) Deep-clean any lingering inference-server processes so the next
        #    cycle's first benchmark starts a fresh server (no stale zmq /
        #    shared-mem / VRAM held by escaped workers).
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
        """Deep-clean lingering inference-server processes (used by the macro-cycle soft restart).

        Reuses the grid runner's ``_kill_stale_servers`` /proc sweep, which only
        targets vLLM/SGLang/atom server processes outside our own process group
        (never our live children) and is a no-op in multi-node mode. Safe at a
        cycle boundary where no benchmark is in flight.
        """
        from ..actions.executors._grid_runner import _kill_stale_servers

        _kill_stale_servers()

    async def _on_enter_explore(self, *, from_phase: str) -> None:
        """Run EXPLORE-entry housekeeping. Roofline lives in PRELUDE, not here (except the per-cycle forced reprofile below).

        Args:
            from_phase: The phase being left; a SWEEP origin in cyclic mode
                triggers the R3 per-cycle forced reprofile.
        """
        # At the start of each macro-cycle (cyclic loopback SWEEP→EXPLORE),
        # force a fresh roofline/profile so the new cycle re-targets the current
        # bottleneck instead of reusing the prior cycle's stale picture. The
        # cycle-scoped idempotency key guarantees a new task each cycle.
        if (
            _phase_state.is_cyclic_phases_enabled()
            and (from_phase or "").upper() == _phase_state.PHASE_SWEEP
            and int(getattr(self.shared_state, "macro_cycle", 0) or 0) > 0
        ):
            try:
                task = await self._enqueue_internal_analysis_task(
                    reason="cycle_start",
                )
                self.shared_state.auto_roofline_pending_task_id = task.task_id
                log.info(
                    "cycle %d EXPLORE entry: forced reprofile task=%s",
                    int(getattr(self.shared_state, "macro_cycle", 0) or 0),
                    task.task_id,
                )
            except Exception:  # noqa: BLE001 — reprofile is best-effort
                log.exception(
                    "cycle EXPLORE entry: forced reprofile enqueue failed",
                )

    async def _maybe_force_stalled_domain_specialist(self) -> None:
        """Hard-trigger: force-dispatch a domain specialist for a
        domain that has gone untouched for too many EXPLORE rounds *and* still
        has an open gap in the gaps[] ledger.

        This is the L2 supervisor escalation the long-run coverage lower-bound
        relies on — a real scheduling event (a normal domain delegate routed
        through PolicyGate + warmup + the GPU specialist pool), not an advisory
        nudge. Idempotent per ``(anchor, round, macro_cycle)`` so it can't spam
        the bus yet still re-fires in a later cycle (the cycle suffix keeps a new
        macro-cycle from dedup-matching a prior cycle's forced task), and it
        self-throttles by zeroing the per-anchor counter on dispatch. Routes
        through ``_handle_intent`` exactly as an LLM delegate would; at most one
        forced dispatch per tick.

        Note:
            Side-effecting: may dispatch a domain specialist via
            ``_handle_intent`` and mutate per-anchor throttle counters on
            ``shared_state``. Returns nothing.
        """
        state = self.shared_state
        if str(getattr(state, "phase", "") or "").upper() != "EXPLORE":
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
                # No pending work pinned to this domain → nothing to force.
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
            # Zero the counter up-front so a slow enqueue can't re-fire next
            # tick; the eventual specialist completion resets it again.
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
            # One forced dispatch per tick keeps the scheduler calm.
            return None
        return None

    def _seed_gaps_from_research_hints(self) -> None:
        """Inject research hints as advisory gaps[] seeds (idempotent)."""
        from ..knowledge import research_hints as _research_hints

        hints = _research_hints.load_hints(self.session_dir)
        for idx, hint in enumerate(hints):
            what = str(hint.get("what") or "").strip()
            if not what:
                continue
            tags = hint.get("domain_tags") or []
            cid = f"gap.research_hint.{idx}"
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

    async def _scan_stale_specialists(self) -> list[dict[str, Any]]:
        """Return specialist task rows running longer than ``_specialist_stale_sec``; never raises, returns [] on failure.

        Returns:
            A list of stale specialist task row dicts; empty on failure or when
            none are stale.
        """
        try:
            running = await self.tasks.running()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: tasks.running() failed during stale scan")
            return []
        if not running:
            return []
        stale: list[dict[str, Any]] = []
        now_unix = time.time()
        for t in running:
            if (t.kind or "").strip() != "specialist":
                continue
            # updated_at on a running task = when the dispatcher promoted it (start of running window).
            started_unix = _parse_iso_unix(t.updated_at)
            if started_unix <= 0:
                continue
            running_sec = max(0.0, now_unix - started_unix)
            if running_sec >= self._specialist_stale_sec:
                stale.append(
                    {
                        "task_id": t.task_id,
                        "kind": t.kind,
                        "running_seconds": running_sec,
                    }
                )
        return stale

    async def _fan_out_specialist_wave(
        self,
        source: str,
        intent: Intent,
        params: dict[str, Any],
    ) -> None:
        """Fan a specialist delegate carrying ``params.tasks=[...]`` into N
        standard free-form specialist dispatches (scope=freeform, lane=cpu,
        mode=research defaults). Each fanned task is re-dispatched through the
        normal ``_handle_delegate`` path (warm + idempotency + TaskRegistry +
        lease + reap), preserving the low-cost wide-net recon the retired
        dynamic_specialist channel provided. Per-task idempotency keys derive
        from the wave key; non-dict / empty-description entries are skipped.

        Args:
            source: The agent issuing the wave delegate.
            intent: The originating specialist DELEGATE intent.
            params: The delegate params carrying the ``tasks`` list to fan out.
        """
        tasks = params.get("tasks") or []
        shared = {k: v for k, v in params.items() if k != "tasks"}
        base_key = str(intent.payload.get("idempotency_key") or "").strip()
        for idx, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            desc = str(task.get("task_description") or task.get("task_summary") or "").strip()
            if not desc:
                continue
            sub_params = dict(shared)
            sub_params["scope"] = "freeform"
            sub_params["task_description"] = desc
            summary = str(task.get("task_summary") or "").strip()
            if summary:
                sub_params["task_summary"] = summary
            # Per-task dial overrides (a wave task may opt into patch / bench /
            # gpu) take precedence over the shared params; then fall back to the
            # freeform recon defaults (research on the cpu lane).
            for carry in (
                "mode",
                "bench",
                "lane",
                "model",
                "priority",
                "timeout_minutes",
                "max_turns",
            ):
                if carry in task:
                    sub_params[carry] = task[carry]
            sub_params.setdefault("mode", "research")
            sub_params.setdefault("lane", "cpu")
            sub_payload = dict(intent.payload)
            sub_payload["params"] = sub_params
            if base_key:
                sub_payload["idempotency_key"] = f"{base_key}-w{idx}"
            else:
                sub_payload.pop("idempotency_key", None)
            await self._handle_delegate(
                source,
                Intent(type=intent.type, payload=sub_payload),
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
        error = str(result.error or "")
        ftype, retry_eligible = classify_specialist_failure(runner_status, error)
        if not retry_eligible:
            return False
        params = task.params or {}
        attempt = int(params.get("_auto_retry_attempt", 0) or 0)
        if attempt >= cap:
            return False
        next_attempt = attempt + 1

        retry_params = dict(params)
        retry_params["_auto_retry_attempt"] = next_attempt
        retry_params["_auto_retry_reason"] = f"{ftype.value}: {error}"[:300]

        # Mirror _handle_delegate lane/ttl resolution (incl. benchmark_lane for
        # bench-enabled specialists, gpu_research_lane + GPU-TTL for any
        # needs_gpu specialist) so the retry task holds the same pools as the
        # original and cannot run concurrently with serving.
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
            # bench specialist: needs_gpu defaulted at warm time (_warm_specialist_params);
            # ensure it is set here too so gpu_research_lane is acquired.
            needs_gpu = True
        if needs_gpu:
            lanes = list(dict.fromkeys((*lanes, "gpu_research_lane")))
            try:
                ttl = self._gpu_lease_ttl_sec(int(ttl or 0))
            except Exception:  # noqa: BLE001
                log.exception(
                    "specialist auto-retry: gpu_research_lane TTL re-source failed; "
                    "using registry default"
                )

        # Stable base key across attempts: strip any prior ``-autoretryN``
        # suffix (distinct from _handle_delegate's ``-retryN`` collision keys
        # so the two mechanisms never share an idempotency namespace).
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
            # Retry slot already taken (e.g. resume replay): let the normal
            # bookkeeping record this attempt rather than silently dropping it.
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

    # specialist pre-dispatch warmup
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

        # Bench-capable (mode=patch & bench=true) specialists run a real
        # serving + benchmark loop on their own cards, so they must hold a GPU
        # lease: default needs_gpu so the dispatcher routes them through the
        # gpu_specialist_pool quota + TTL throttle (operator/LLM may still
        # override explicitly).
        if resolve_specialist_profile(params).reserves_benchmark_lane:
            params.setdefault("needs_gpu", True)

        domain = str(params.get("domain") or "").strip()
        # Knowledge-domain tags drive multi-anchor prompt assembly; a single ``domain`` is the legacy single-tag alias.
        normalize_dispatch_tags(params)

        if "pr_monitor_available" not in params:
            params["pr_monitor_available"] = bool(plane is not None and getattr(plane, "pr_monitor_enabled", True))

        # kb_subgraph kept defaulted for stable SpecialistPromptInputs.
        params.setdefault("kb_subgraph", {})

        # Warm-start recipe + pitfalls + lessons from T0 anchor.
        if state.warm_start_recipe and "warm_start_recipe" not in params:
            params["warm_start_recipe"] = dict(state.warm_start_recipe)
        if state.warm_start_pitfalls and "warm_start_pitfalls" not in params:
            params["warm_start_pitfalls"] = list(state.warm_start_pitfalls)
        if state.warm_start_lessons and "warm_start_lessons" not in params:
            params["warm_start_lessons"] = list(state.warm_start_lessons)
        # KG graph-recommended knobs (cross-recipe IMPROVES candidates from the
        # T0 warm-start context); advisory positive priors for the specialist.
        if "kg_recommended_knobs" not in params:
            wsc = getattr(state, "warm_start_context", None) or {}
            kg_knobs = wsc.get("recommended_knobs") if isinstance(wsc, dict) else None
            if kg_knobs:
                params["kg_recommended_knobs"] = [k for k in kg_knobs if isinstance(k, dict)]
        # KG graph-guided config knobs (journal KNOB_IMPROVES, runnable args/envs);
        # only present when GBRAIN_KG_GUIDED enabled the T0 enhancement.
        if "kg_guided_knobs" not in params:
            wsc = getattr(state, "warm_start_context", None) or {}
            guided = wsc.get("graph_guided_knobs") if isinstance(wsc, dict) else None
            if guided:
                params["kg_guided_knobs"] = [k for k in guided if isinstance(k, dict)]
        # runtime framework/version so the prompt's _format_version_note annotates version-mismatched lessons.
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

        # Local-source navigation hint — same source the Kernel-agent uses for source_file containment.
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

        # Hardware + workload hints from SharedState; else dataclass defaults win (e.g. tp=1 self-vetoes comm_specialist).
        params.setdefault("gpu_type", state.gpu_type or "")
        # Active server framework name — switches per-domain hint blocks to atom paths when framework == "atom".
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

        # Advisory model_arch profile → specialist via arch_notes carrier; prompt-context only, no gating.
        if "arch_notes" not in params:
            from ..state.shared_state import render_model_arch_compact

            _arch_notes = render_model_arch_compact(getattr(state, "model_arch", None))
            if _arch_notes:
                params["arch_notes"] = _arch_notes

        # Static-recon specialist extras: structured model_info (machine-parseable
        # companion to arch_notes) + checklist-derived source-hint directories so
        # the recon focus block can gate + navigate. Other domains unaffected.
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

        # Fill gap-specific anchors from the gaps[] ledger: stamp symptom/layer/domain_hint/attempts onto the task so the prompt has structured context.
        gap_cid = str(params.get("gap_canonical_id") or "").strip() or str(params.get("gap") or "").strip()
        if gap_cid:
            gap = state.find_gap(gap_cid)
            if gap is not None:
                if not params.get("gap_symptom"):
                    params["gap_symptom"] = str(gap.get("symptom") or "")
                if not params.get("gap_layer"):
                    params["gap_layer"] = str(gap.get("layer") or "")
                if not params.get("domain"):
                    # LLM omitted domain → gap's domain_hint wins (PolicyGate R2 still validates routing).
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

        # ROOFLINE EVIDENCE — pack bottleneck signals into roofline_evidence + analysis_md_path for the specialist.
        last_ta = getattr(state, "last_trace_analyze", None) or {}
        if isinstance(last_ta, dict) and last_ta.get("analysis_md_text") and "roofline_evidence" not in params:
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

        # PR discovery lives in the FRAMEWORK_AGENT phase pump.

        # proposal_set cap into params so SpecialistRunner reads it; setdefault lets a delegate shrink it.
        from hyperloom.orchestrator.policy.gate import (
            DEFAULT_SPECIALIST_MAX_PROPOSALS,
        )

        params.setdefault("max_proposals", DEFAULT_SPECIALIST_MAX_PROPOSALS)

    # gaps[] ledger refresh
    async def _refresh_gaps(self, *, reason: str) -> None:
        """Refresh :attr:`SharedState.gaps` from observable signals (Coordinator is sole writer, Inv-1). Additive upsert deduped by canonical_id; best-effort.

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
        if plane is not None and hasattr(plane, "cortex_traverse_issues"):
            try:
                traverse = getattr(plane, "cortex_traverse_issues")
                rows = traverse(
                    model_class=getattr(state, "model_class", "") or "",
                    gpu_type=getattr(state, "gpu_type", "") or "",
                )
                if isinstance(rows, list):
                    for entry in rows:
                        if isinstance(entry, dict):
                            entry = dict(entry)
                            entry.setdefault("source", "cortex")
                            state.upsert_gap(entry)
            except Exception:  # noqa: BLE001 — defensive
                log.warning(
                    "gaps refresh: cortex_traverse_issues failed (reason=%s)",
                    reason,
                    exc_info=True,
                )
        log.debug(
            "gaps refresh (reason=%s): %d gaps after merge",
            reason,
            len(state.gaps),
        )

    def _extract_gaps_from_baseline(self) -> list[dict[str, Any]]:
        """Derive initial gap rows from the baseline snapshot (throughput_below_target, baseline_unstable); reuse the M1 anchor canonical_id so traverse rows align.

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
                    "symptom": (f"current_best is {target_gap:.1f}% short of --target-gain"),
                    "layer": "framework",
                    "severity": severity,
                    "domain_hint": "serving_specialist",
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
        """Derive gaps from rolling failures + winners history (recurring (action, error_class) + explore plateau).

        Returns:
            A list of gap row dicts derived from recurring action failures and
            an explore-plateau signal.
        """
        state = self.shared_state
        anchor = self._workload_canonical_id()
        gaps: list[dict[str, Any]] = []

        failures = list(state.last_action_failures or [])[-10:]
        seen_failures: dict[str, dict[str, Any]] = {}
        for row in failures:
            if not isinstance(row, dict):
                continue
            action = str(row.get("action") or "").strip() or "unknown"
            err = str(row.get("error_class") or "").strip() or "unknown_error"
            key = f"{action}::{err}"
            layer, domain = self._gap_layer_for_action(action)
            attempt = {
                "action": action,
                "variant_name": str(row.get("variant_name") or ""),
                "outcome": "REVERT",
                "error_class": err,
                "ts": str(row.get("ts") or datetime.now(timezone.utc).isoformat()),
            }
            if key in seen_failures:
                seen_failures[key]["attempts"].append(attempt)
            else:
                seen_failures[key] = {
                    "canonical_id": f"{anchor}#fail:{action}:{err}",
                    "symptom": f"{action} repeatedly fails with {err}",
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
                    "domain_hint": "serving_specialist",
                    "source": "attempts",
                }
            )
        return gaps

    @staticmethod
    def _gap_layer_for_action(action: str) -> tuple[str, str]:
        """Map an action name → (layer, domain_hint) for gap rows (fallback ("framework", "serving_specialist")).

        Args:
            action: The action name to classify.

        Returns:
            A ``(layer, domain_hint)`` tuple for the action.
        """
        a = str(action or "").strip().lower()
        if a in {"kernel_opt", "integrate", "trace_analyze", "run_gemm_tuning", "run_optimization"}:
            return ("kernel_agent", "kernel_switch_specialist")
        if a in {"profile", "roofline"}:
            return ("kernel_agent", "kernel_switch_specialist")
        if a in {"sweep", "explore"}:
            return ("framework", "serving_specialist")
        if a in {"baseline"}:
            return ("system", "system_specialist")
        return ("framework", "serving_specialist")

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
                    "domain_hint": "serving_specialist",
                    "source": "attempts",
                }
            )
        for outcome in per_variant:
            if not isinstance(outcome, dict):
                continue
            state.append_gap_attempt(
                canonical,
                {
                    "action": "explore",
                    "variant_name": str(outcome.get("variant_name") or ""),
                    "outcome": str(outcome.get("outcome") or "").upper(),
                    "gain_pct": outcome.get("gain_pct"),
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
        Orchestration LLM drives ``explore`` directly (local bash +
        explore delegates), so this deterministic materialisation stays
        multi-node-scoped and the single-node path is unchanged
        bit-for-bit.

        Why this exists: in multi-node the GPU cluster lives on remote
        SSH pods, so the LLM cannot bench proposals via local bash — the
        only materialisation channel is a structured ``explore`` action.
        Observation-only surfacing (the default) relies on the LLM
        emitting that delegate, which it does not do reliably in
        multi-node, leaving approved proposals un-benchmarked. This
        helper closes that gap by enqueuing the explore grid itself.

        ``proposal_set`` entries already reuse the explore variant schema
        (``name`` / ``extra_args`` / ``extra_envs``), so they pass
        straight through as the grid. The explore executor's
        ``canonical_fingerprint`` dedup means a later LLM-emitted explore
        on the same content collapses to the same row (no double-bench),
        and its per-variant KEEP/REVERT gain gate is the safety net
        (no critic dependency).

        Args:
            task: The completed specialist task whose id seeds the explore
                idempotency key.
            domain: The specialist domain, stamped onto variant provenance.
            proposals: The specialist ``proposal_set`` entries materialised into
                the explore grid (capped at ``_MN_AUTO_EXPLORE_GRID_CAP``).
        """
        # Framework config-generation specialists own their proposal_set via the
        # config subphase; skip the mn-explore bridge so it is not double-consumed.
        if bool((getattr(task, "params", None) or {}).get("framework_config_generation")):
            return
        from ..actions.executors._multi_node_env import is_multi_node

        if not is_multi_node() or not proposals:
            return
        grid: list[dict[str, Any]] = []
        for i, p in enumerate(proposals[: self._MN_AUTO_EXPLORE_GRID_CAP]):
            if not isinstance(p, dict):
                continue
            args = str(p.get("extra_args") or p.get("extra_server_args") or "").strip()
            envs_raw = p.get("extra_envs")
            envs = {str(k): str(v) for k, v in envs_raw.items()} if isinstance(envs_raw, dict) else {}
            controls: dict[str, Any] = {}
            for key in ("remove_args", "unset_envs"):
                raw = p.get(key)
                if isinstance(raw, str):
                    vals = [raw.strip()] if raw.strip() else []
                elif isinstance(raw, (list, tuple, set)):
                    vals = [str(v).strip() for v in raw if str(v).strip()]
                else:
                    vals = []
                if vals:
                    controls[key] = vals
            mode = str(p.get("args_mode") or "append").strip().lower()
            if mode == "replace":
                controls["args_mode"] = "replace"
            # Drop entries with neither a server-arg nor an env override —
            # nothing for the restart to apply (e.g. research-only items)
            # unless the entry removes inherited args/envs.
            if not args and not envs and not controls:
                continue
            name = str(p.get("name") or "").strip() or (f"{domain or 'specialist'}-{task.task_id[:8]}-{i}")
            grid.append(
                {
                    "name": name,
                    "extra_args": args,
                    "extra_envs": envs,
                    **controls,
                    "provenance": f"specialist:{domain}" if domain else "specialist",
                    "note": str(p.get("reason") or "")[:200],
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
        cb = state.current_best or {}
        if isinstance(cb, dict):
            cb_args = str(cb.get("extra_server_args") or "")
            if cb_args:
                params["base_extra_args"] = cb_args
            _raw_remove = cb.get("remove_args")
            _raw_unset = cb.get("unset_envs")
            cb_remove = [_raw_remove] if isinstance(_raw_remove, str) and _raw_remove.strip() else [
                str(v) for v in (_raw_remove or []) if str(v).strip()
            ]
            cb_unset = [_raw_unset] if isinstance(_raw_unset, str) and _raw_unset.strip() else [
                str(v) for v in (_raw_unset or []) if str(v).strip()
            ]
            if cb_remove:
                params["base_remove_args"] = cb_remove
            if cb_unset:
                params["base_unset_envs"] = cb_unset
            if str(cb.get("args_mode") or "").strip().lower() == "replace":
                params["base_args_mode"] = "replace"
        base_tput = float(getattr(state, "baseline_tput", 0.0) or 0.0)
        if base_tput:
            params["base_tput"] = base_tput
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        try:
            etask, was_existing = await self.tasks.create_or_return_existing(
                kind="explore",
                params=params,
                idempotency_key=f"mn-auto-explore-{task.task_id}",
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
        # Resolve patches_written against worktree + workspace; submit only when >=1 real file exists.
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
        # A non-diff tuned artifact (``artifacts_written`` with a real source
        # file) is also a routable deliverable: integrate_patch installs it
        # (backup + gate + REVERT). Route it exactly like a patch. Shared
        # routable-signal with the empty-outcome bridge (no FRAMEWORK livelock).
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
        integrate_params: dict[str, Any] = {
            "specialist_task_id": sid,
            "provenance": "specialist",
            "patch_name": patch_name,
        }
        # FRAMEWORK authoring provenance passthrough: a candidate dispatched
        # to the authoring specialist carries its originating PR candidate/batch
        # id in the specialist task params. Propagate them onto the synthetic
        # integrate_patch task so ``_record_framework_agent_authored_outcome`` can
        # key the progress row on the real candidate id (a PR URL). Without this
        # the bridge falls back to the integrate_patch task_id, the progress row
        # never matches ``_select_next_framework_agent_candidate``'s candidate id,
        # and the FRAMEWORK pump re-dispatches the same candidate forever
        # (livelock observed in the 84-candidate batch run).
        try:
            spec_params = getattr(task, "params", None) or {}
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
                probe = str(spec_params.get("launch_probe") or "").strip()
                if probe:
                    integrate_params["launch_probe"] = probe
                # Forward the pre-patch failure signature so the runnable gate
                # can detect the same actionable failure re-appearing post-patch.
                before_sig = spec_params.get("enablement_before_signature")
                if isinstance(before_sig, dict):
                    integrate_params["enablement_before_signature"] = before_sig
                # Forward the stacked base patches (prior progressing rounds) so
                # integrate_patch re-applies them before this round's patch.
                base_patches = spec_params.get("enablement_base_patches")
                if isinstance(base_patches, list) and base_patches:
                    integrate_params["enablement_base_patches"] = [str(p) for p in base_patches]
                # Forward stacked base setup commands (prior rounds' installs) so
                # integrate_patch replays them before boot; the current round's
                # own setup_commands are read from specialist_done directly.
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
                # Artifact-only deliverables route with empty ``patches``; record
                # their install targets so the observation is not silently blank.
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

        Companion to :meth:`_maybe_autosubmit_specialist_patches`. That bridge
        fires only on ``patches_written``; this one fires when a FRAMEWORK
        *authoring* specialist returns NO source patch but a config-lever
        ``proposal_set`` (extra_args / extra_envs) — the relaxed FRAMEWORK
        rule that lets a PR's benefit land as serving flags / env vars (e.g. an
        MTP toggle) without writing source. The levers go into integrate_patch's
        existing ``config_changes`` channel (apply + bench + accuracy gate +
        KEEP/REVERT); integrate_patch owns the terminal FRAMEWORK row via
        ``_record_framework_agent_authored_outcome``. Idempotent per specialist.

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
        config_changes = _framework_config_levers_from_done(done_payload)
        if not config_changes:
            return
        sid = str(task.task_id or "").strip()
        if not sid:
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
            "config_changes": dict(config_changes),
        }
        # FRAMEWORK authoring provenance passthrough so the authored-outcome
        # bridge keys the terminal row on the real PR candidate id.
        fa_cand = str(spec_params.get("framework_agent_candidate_id") or "")
        fa_batch = str(spec_params.get("framework_batch_id") or "")
        integrate_params["framework_agent_authoring"] = True
        if fa_cand:
            integrate_params["framework_agent_candidate_id"] = fa_cand
        if fa_batch:
            integrate_params["framework_batch_id"] = fa_batch
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
                "config_changes": dict(config_changes),
            },
        )
        log.info(
            "FRAMEWORK: config-lever deliverable routed to integrate_patch "
            "candidate=%s keys=%s",
            fa_cand or sid,
            sorted(config_changes.keys()),
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "FRAMEWORK: save after config autosubmit failed for task=%s",
                sid,
            )

    def _build_specialist_round_entry(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        """Translate a specialist done payload into a SharedState.specialist_rounds[] row; round_id defaults to task_id for idempotent overwrite.

        Args:
            task: The completed specialist task.
            done_payload: The specialist done payload (proposal_set, domain,
                tags, summary, etc.).
            source: The emitting agent string, recorded on the row.

        Returns:
            A specialist-round row dict suitable for
            ``SharedState.record_specialist_round``.
        """
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        round_id = str((task.params or {}).get("round_id") or task.task_id)
        truncated_from = done_payload.get("proposals_truncated_from")
        from ..specialists.domains import normalize_dispatch_tags

        # Knowledge-domain tags for breakdown attribution; reported tags win over dispatch params.
        tags = normalize_dispatch_tags(done_payload)
        if not tags:
            tags = normalize_dispatch_tags(task.params or {})
        entry: dict[str, Any] = {
            "round_id": round_id,
            "task_id": task.task_id,
            "source": source or "coordinator",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "domain": str(done_payload.get("domain") or ""),
            "tags": list(tags),
            "gap_canonical_id": str(done_payload.get("gap_canonical_id") or ""),
            "empty": bool(done_payload.get("empty")) or len(proposals) == 0,
            "proposals_total": len(proposals),
            "proposal_set": list(proposals),
            "summary": str(done_payload.get("summary") or "")[:480],
            "reason": str(done_payload.get("reason") or "")[:480],
            "confidence": done_payload.get("confidence"),
            "new_findings": list(done_payload.get("new_findings") or []),
            "residual_questions": list(done_payload.get("residual_questions") or []),
        }
        gpu_ids = done_payload.get("allocated_gpu_ids") or []
        if isinstance(gpu_ids, list) and gpu_ids:
            entry["allocated_gpu_ids"] = [
                int(g) for g in gpu_ids if isinstance(g, (int, str)) and str(g).strip().lstrip("-").isdigit()
            ]
        if isinstance(truncated_from, int) and truncated_from > len(proposals):
            entry["proposals_truncated_from"] = truncated_from
        return entry
