"""Coordinator — DESIGN v0.6 §7.5 / §21 main loop.

The Coordinator is the **protocol manager** (not a decision-maker). It owns:

* MessageBus + ResourceLockManager + TaskRegistry + CursorStore
* PolicyGate (intent validation choke-point)
* REQUEST/RESPONSE routing (Plan A: orchestration → kernel only)
* Critic Review gate (§18) — pending proposals wait for verdict
* Robustness scheduling-police execution (§19.3): kill_task / prune /
  force_dispatch / escalate_strategy_change
* Per-agent reactor loops + dispatcher

**P0-3 scope**: a minimal, bounded-tick reactor that:
* spins up one reactor task per agent (calls backend.run() each tick)
* validates emitted intents through PolicyGate
* persists / routes intents (REQUEST/RESPONSE/REVIEW_VERDICT/etc.)
* for delegated tasks: enqueues into TaskRegistry and pumps SubAgentRunner

Everything else (real backends, accuracy gate, scheduler scoring,
checkpoint cadence) lands in P0-5 and beyond.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import shlex
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..paths import db_path_for, make_session_dir
from ..storage.connection import SqliteConnection
from .action_registry import ActionRegistry
from .agent_role import AgentRole, default_role_registry
from .backends.base import Backend, BackendError, BackendTurnResult
from .cursor_store import CursorStore
from .intent_parser import Intent, IntentType, NoIntentEmitted
from .kb_digest import format_kb_digest_for_orchestration
from .kernel_request_handlers import KERNEL_REQUEST_HANDLERS, get_handler
from .message_bus import Message, MessageBus
from .objective import Objective, TimeOnlyObjective
from .policy import (
    KILL_TASK_SOURCE_ALLOWLIST,
    PolicyDenied,
    PolicyGate,
    REQUEST_ROUTING,
    REVIEW_VERDICT_SOURCE_ALLOWLIST,
    ROBUSTNESS_ONLY_SOURCE_ALLOWLIST,
)
from .resource_lock import ResourceLockManager, SqliteLeaseBackend
from .shared_state import SharedState
from .sub_agent_runner import SubAgentRunner
from .task_registry import Task, TaskRegistry
from .action_executors.benchmark_result import is_valid_measurement
from . import scoring as _scoring
from .system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
)


log = logging.getLogger(__name__)


@dataclass
class PendingProposal:
    """A propose_action intent waiting for Critic Review (§18)."""

    proposal_msg_id: str
    from_agent: str
    action_name: str
    predicted_gain_pct: float
    payload: dict[str, Any]
    decided: bool = False
    verdict: str | None = None  # approve / reject / redirect / advise / needs_review


@dataclass
class CoordinatorState:
    """In-memory ephemeral state for the reactor + dispatcher.

    The persistent counterpart lives in :class:`SharedState` (state.json).
    pruned_families and similar long-lived flags now go through SharedState;
    in-flight reactor data (pending_proposals) stays here.
    """

    pending_proposals: dict[str, PendingProposal] = field(default_factory=dict)


class Coordinator:
    """The single Coordinator instance per session.

    Construct, ``await ctor.start()``, optionally ``await ctor.tick(n)``
    for bounded test runs, then ``await ctor.stop()``.
    """

    def __init__(
        self,
        session_dir: Path,
        *,
        backends: dict[str, Backend],
        role_registry: dict[str, AgentRole] | None = None,
        sub_agent_runner: SubAgentRunner | None = None,
        bus_class: type[MessageBus] = MessageBus,
    ):
        self.session_dir = Path(session_dir)
        self.role_registry = role_registry or default_role_registry()

        # Validate every reactor we expect actually has a backend wired.
        for name in self.role_registry:
            if name not in backends:
                raise ValueError(
                    f"missing backend for role {name!r} "
                    f"(provide via Coordinator(backends={{...}}))"
                )
        self.backends = dict(backends)

        # Persistence layer
        db_path = db_path_for(self.session_dir)
        self.db = SqliteConnection(db_path)

        self.bus = bus_class(self.db)
        self.locks = ResourceLockManager(SqliteLeaseBackend(self.db))
        self.tasks = TaskRegistry(self.db)
        self.cursors = CursorStore(self.db)
        # `strict_paths` defers to the env flag (CLI flips this on for
        # production; tests omit the env so the path check stays off and
        # legacy `/tmp/<fixture>` payload values still pass).
        self.policy = PolicyGate(
            role_registry=self.role_registry,
            session_dir=self.session_dir,
        )
        self.sub = sub_agent_runner or SubAgentRunner(
            self.locks, self.tasks, session_dir=self.session_dir,
        )

        # Persistent session state (state.json) — load existing for resume;
        # save() is called whenever the Coordinator mutates a persistent field.
        self.shared_state = SharedState.load_or_init(self.session_dir)
        self.state = CoordinatorState()
        self._stop = asyncio.Event()
        self._tasks_running: list[asyncio.Task] = []

        # Stable tick order derived from the live role_registry. Must NOT
        # use the module-level `roles_for_run()` which is a cached hardcoded
        # tuple containing "kernel" even when --no-kernel stripped it.
        _CANONICAL_ORDER = ("orchestration", "kernel", "critic", "robustness")
        self._tick_roles: tuple[str, ...] = tuple(
            r for r in _CANONICAL_ORDER if r in self.role_registry
        )

        # Action registry + per-action scoring (see orchestrator/scoring.py
        # and the plan ``action-scoring-in-shared-state``). The registry is
        # cheap to load — a handful of small yaml files — so we eagerly load
        # it once and use the in-memory copy to both seed scores and render
        # the per-tick scoreboard. A load failure falls back to ``None``;
        # downstream callers handle a missing registry gracefully.
        try:
            self.action_registry: ActionRegistry | None = ActionRegistry().load()
        except Exception:  # noqa: BLE001 — defensive; missing yaml shouldn't kill the run.
            log.exception("Coordinator: failed to load ActionRegistry; "
                          "scoring will be disabled this session.")
            self.action_registry = None
        # Latest objective wired by ``Coordinator.run()``. Used by
        # ``_compose_prompt`` to refresh ``shared_state.target_gap_pct`` on
        # every Orchestration tick. None outside a run (e.g. bounded tick()
        # tests) and the scoreboard renderer falls back to multiplier=1.0.
        self._current_objective: Objective | None = None

        # Resume detection runs BEFORE we seed action_scores so a fresh
        # session is not misdetected as a resume (seeding writes state.json
        # which the resume probe treats as evidence of an existing session).
        self._resumed_from = self._detect_resume_state()
        # Seed per-action scoring now that resume status is locked in.
        self._ensure_action_scores_seeded()

    # ==================================================================
    # Resume
    # ==================================================================
    def _detect_resume_state(self) -> dict[str, Any]:
        """Synchronously inspect persistence to determine if this is a resume.

        Called from __init__ — must not block on the event loop. Returns a
        small dict with `is_resume` + summary stats. The actual rebuild of
        in-memory CoordinatorState happens lazily on the first call to
        :meth:`tick` (or the dedicated :meth:`replay_for_resume`); this lets
        construction stay synchronous + fast.
        """
        ev_count = self.bus.db.fetchone_sync("SELECT COUNT(*) AS c FROM events")
        events_present = (int(ev_count["c"]) if ev_count else 0) > 0
        state_path = SharedState.state_path(self.session_dir)
        return {
            "is_resume": events_present or state_path.exists(),
            "event_count": int(ev_count["c"]) if ev_count else 0,
            "state_json_present": state_path.exists(),
            "rebuilt": False,  # set by replay_for_resume()
        }

    async def replay_for_resume(self) -> dict[str, Any]:
        """Walk the event log to reconstruct ``CoordinatorState.pending_proposals``.

        Idempotent — re-running rebuilds from scratch. Returns a small dict
        of stats so tests can assert what was restored.

        We treat a proposal as **undecided** when there is no
        ``review_verdict`` event addressed to it AND no ``decision`` event
        materializing it (`kind == "approved_proposal"`).
        """
        # 1. Collect all proposal events.
        proposal_msgs = await self.bus.tail(topic="proposal", n=10_000)
        # 2. Collect verdicts and approved decisions, keyed by proposal_msg_id.
        verdicts = await self.bus.tail(topic="review_verdict", n=10_000)
        decisions = await self.bus.tail(topic="decision", n=10_000)

        decided_ids: set[str] = set()
        verdict_by_target: dict[str, str] = {}
        for v in verdicts:
            target = v.payload.get("target_proposal_msg_id")
            if target:
                verdict_by_target[target] = v.payload.get("verdict", "")
                decided_ids.add(target)
        for d in decisions:
            if d.payload.get("kind") == "approved_proposal":
                # The coordinator stores task_id, not the original proposal_msg_id,
                # in the decision; tasks created via materialization have
                # idempotency_key f"approved-{proposal_msg_id}" — we can
                # back-trace through the tasks table if needed, but for
                # pending-proposal rebuild it's enough that the verdict event
                # already marked the proposal as decided.
                pass

        # 3. Rebuild PendingProposal entries for undecided proposals.
        rebuilt = 0
        self.state.pending_proposals.clear()
        for p in proposal_msgs:
            if p.msg_id in decided_ids:
                # Optional: also remember the verdict so the Coordinator can
                # surface it if asked (e.g. /status command).
                continue
            payload = p.payload or {}
            self.state.pending_proposals[p.msg_id] = PendingProposal(
                proposal_msg_id=p.msg_id,
                from_agent=p.from_agent,
                action_name=str(payload.get("action_name", "")),
                predicted_gain_pct=float(payload.get("predicted_gain_pct", 0.0)),
                payload=dict(payload),
            )
            rebuilt += 1

        self._resumed_from["rebuilt"] = True
        self._resumed_from["pending_restored"] = rebuilt
        return {
            "is_resume": self._resumed_from["is_resume"],
            "event_count": self._resumed_from["event_count"],
            "state_json_present": self._resumed_from["state_json_present"],
            "pending_restored": rebuilt,
            "verdicts_seen": len(verdicts),
        }

    @property
    def resumed_from(self) -> dict[str, Any]:
        """Read-only snapshot of resume detection (set by ``__init__``).

        Returns ``{"is_resume": bool, "event_count": int, "state_json_present":
        bool, "rebuilt": bool, ...}``.
        """
        return dict(self._resumed_from)

    # ==================================================================
    # Lifecycle
    # ==================================================================
    async def stop(self) -> None:
        self._stop.set()
        for t in self._tasks_running:
            if not t.done():
                t.cancel()
        for t in self._tasks_running:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                log.exception("reactor task raised on shutdown")
        self.db.close()

    # ==================================================================
    # Action scoring
    # ==================================================================
    def _score_action_keep(self, action_name: str, *, gain_pct: float) -> None:
        """Apply a KEEP score update for one action and persist."""
        raw = self.shared_state.action_scores.get(action_name)
        if not isinstance(raw, dict):
            return
        a = _scoring.ActionScore.from_dict(raw)
        _scoring.apply_keep(
            a,
            gain_pct=float(gain_pct or 0.0),
            tick=int(self.shared_state.tick or 0),
            action_name=action_name,
        )
        self.shared_state.action_scores[action_name] = a.to_dict()

    def _score_action_discard(self, action_name: str) -> None:
        raw = self.shared_state.action_scores.get(action_name)
        if not isinstance(raw, dict):
            return
        a = _scoring.ActionScore.from_dict(raw)
        _scoring.apply_discard(
            a,
            tick=int(self.shared_state.tick or 0),
            action_name=action_name,
        )
        self.shared_state.action_scores[action_name] = a.to_dict()

    def _score_action_lock(self, action_name: str, reason: str) -> None:
        raw = self.shared_state.action_scores.get(action_name)
        if not isinstance(raw, dict):
            return
        a = _scoring.ActionScore.from_dict(raw)
        if not a.locked_reason:
            _scoring.apply_lock(a, reason)
            self.shared_state.action_scores[action_name] = a.to_dict()

    def _apply_action_score_update(
        self,
        task_kind: str,
        result: dict[str, Any],
        *,
        promoted: bool | None = None,
        gain_vs_cb: float | None = None,
    ) -> None:
        """Single hook called by ``_promote_to_shared_state`` once the
        existing task-kind branch has decided whether to promote / lift.

        ``promoted`` and ``gain_vs_cb`` are only meaningful for the
        ``backends`` / ``params`` / ``sweep`` family; other branches pass
        ``None`` and the helper decides what to do from ``task_kind`` alone.

        The helper is a no-op when ``action_scores`` has not been seeded
        (e.g. early sessions where ActionRegistry failed to load).
        """
        if not self.shared_state.action_scores:
            return
        if not task_kind:
            return
        if task_kind == "baseline":
            # Treated as a gate; do not score it. baseline_failure_streak
            # already tracks repeated failures.
            return
        if task_kind in {"profile", "pmc_roofline"}:
            self._score_action_keep(task_kind, gain_pct=0.0)
            return
        if task_kind == "validate_stack":
            # Bumps runs + cooldown so we don't validate_stack spam.
            self._score_action_keep("validate_stack", gain_pct=0.0)
            return
        if task_kind in {"backends", "params", "sweep"}:
            if promoted:
                self._score_action_keep(
                    task_kind, gain_pct=float(gain_vs_cb or 0.0),
                )
            else:
                self._score_action_discard(task_kind)
            # Lock the grid when it is fully exhausted so the LLM sees
            # the row as unavailable. Only applies to params today
            # (the only family with a deterministic exhaustion check).
            if task_kind == "params" and self._params_grid_exhausted():
                self._score_action_lock("params", "grid_exhausted")
            return
        # Support actions (dream / re_explore / recover / etc.): just
        # register the run so cooldown + aging math evolves.
        self._score_action_keep(task_kind, gain_pct=0.0)

    def _ensure_action_scores_seeded(self) -> None:
        """Populate ``shared_state.action_scores`` once per session.

        Idempotent: if any scores are already present (resume case) we skip.
        Otherwise we seed from the action registry, biased by ``model_class``
        marathon priors (default ``moe_mla`` when classify hasn't run yet).
        Persistence is best-effort: any save failure is logged but does not
        block construction.
        """
        if self.action_registry is None:
            return
        if self.shared_state.action_scores:
            return
        kernel_enabled = "kernel" in self.role_registry
        enabled = FULL_ENABLED_ACTIONS if kernel_enabled else NO_KERNEL_ENABLED_ACTIONS
        model_class = (self.shared_state.model_class or "moe_mla").strip()
        try:
            seeded = _scoring.seed_action_scores(
                self.action_registry,
                model_class=model_class,
                enabled=enabled,
            )
        except Exception:  # noqa: BLE001
            log.exception(
                "Coordinator: scoring.seed_action_scores failed; "
                "action_scores stay empty for this session.",
            )
            return
        if not seeded:
            return
        self.shared_state.action_scores = seeded
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "Coordinator: failed to persist seeded action_scores",
            )

    # ==================================================================
    # Bounded test interface
    # ==================================================================
    async def tick(self, n: int = 1) -> None:
        """Run exactly ``n`` reactor passes for **every** agent.

        Used by P0-3 / P0-5 / P1-4 tests. Each agent's backend.run() is
        called once per pass; intents are validated + routed before the
        next pass starts. Dispatcher pumps queued tasks at the end of
        each pass.

        On the first tick, if this Coordinator was constructed against a
        non-empty session, it lazily reruns ``replay_for_resume()`` so
        in-memory state catches up before any new reactor work runs.
        """
        if self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]:
            await self.replay_for_resume()
        for _ in range(n):
            self.shared_state.increment_tick()
            for name in self._tick_roles:
                await self._reactor_pass(name)
            await self._pump_dispatcher_once()

    # ==================================================================
    # Long-run interface (DESIGN §9 + §21)
    # ==================================================================
    async def run(
        self,
        *,
        objective: Objective | None = None,
        max_minutes: float | None = None,
        tick_interval_sec: float = 0.0,
        max_ticks: int | None = None,
        stop_when: Callable[["Coordinator"], Awaitable[bool] | bool] | None = None,
        install_signal_handlers: bool = False,
        crash_emergency_threshold: int = 25,
    ) -> str:
        """Run reactor + dispatcher in a long-running loop until a stop
        condition fires.

        Stop signals (in priority order, see DESIGN §9.1):

        * ``self._stop`` set (from SIGINT/SIGTERM or ``stop()``)
            → ``stop_reason="signal"``
        * objective.reached(shared_state)  → ``"target_reached"``
        * no remaining automated levers     → ``"no_more_leverage"``
        * wall-clock budget exceeded        → ``"time_exhausted"``
        * crash_count >= ``crash_emergency_threshold`` → ``"emergency"``
        * custom ``stop_when`` callback returns True → ``"custom"``
        * ``max_ticks`` reached (test guard) → ``"max_ticks"``

        On stop, ``shared_state.stop_reason`` is set + saved + the final
        value is returned.
        """
        objective = objective or TimeOnlyObjective()
        # Stash so ``_compose_prompt`` can update target_gap_pct.
        self._current_objective = objective
        deadline = (
            time.monotonic() + max_minutes * 60.0 if max_minutes else None
        )
        max_minutes_value = max_minutes if max_minutes is not None else 0
        # Persist budget so prompts and Resume can see it.
        if max_minutes is not None:
            self.shared_state.max_minutes = int(max_minutes)
            self.shared_state.save(self.session_dir)

        previous_handlers: dict[int, Any] = {}
        if install_signal_handlers:
            try:
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, self._stop.set)
                    previous_handlers[sig] = True
                log.info("Coordinator.run: SIGINT/SIGTERM handlers installed")
            except (NotImplementedError, RuntimeError) as exc:  # noqa: BLE001
                # add_signal_handler is unavailable on Windows or when
                # we're not on the main thread (pytest-asyncio worker).
                log.info("Coordinator.run: signal handlers not installed (%s)", exc)
                previous_handlers = {}

        if self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]:
            await self.replay_for_resume()

        tick_n = 0
        stop_reason = ""
        try:
            while not stop_reason:
                tick_n += 1
                # Bump the persistent tick counter — drives cooldown / aging
                # math in orchestrator/scoring.py. Persisted on the next
                # save() (after _promote_to_shared_state or stop).
                self.shared_state.increment_tick()
                # Run one reactor + dispatcher pass.
                for name in self._tick_roles:
                    if self._stop.is_set():
                        break
                    await self._reactor_pass(name)
                if not self._stop.is_set():
                    await self._pump_dispatcher_once()

                # ---- check stop conditions ----
                if self._stop.is_set():
                    stop_reason = "signal"
                    break
                if self.shared_state.stop_reason:
                    stop_reason = self.shared_state.stop_reason
                    break
                if objective.reached(self.shared_state):
                    stop_reason = "target_reached"
                    break
                if await self._has_no_more_leverage():
                    stop_reason = "no_more_leverage"
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    stop_reason = "time_exhausted"
                    break
                if self.shared_state.crash_count >= crash_emergency_threshold:
                    stop_reason = "emergency"
                    break
                if max_ticks is not None and tick_n >= max_ticks:
                    stop_reason = "max_ticks"
                    break
                if stop_when is not None:
                    triggered = stop_when(self)
                    if asyncio.iscoroutine(triggered):
                        triggered = await triggered
                    if bool(triggered):
                        stop_reason = "custom"
                        break

                # Brief wait between ticks to avoid 100% CPU spin in dev
                # mode while still being responsive to signals. 0.0 keeps
                # tests fast.
                if tick_interval_sec > 0:
                    try:
                        await asyncio.wait_for(
                            self._stop.wait(), timeout=tick_interval_sec
                        )
                        stop_reason = "signal"
                        break
                    except asyncio.TimeoutError:
                        pass
        finally:
            self.shared_state.stop_reason = stop_reason or "unknown"
            self.shared_state.save(self.session_dir)
            log.info(
                "Coordinator.run: stopped tick=%d reason=%s baseline_tput=%.1f "
                "cumulative_gain=%.2f%% max_minutes=%.0f",
                tick_n, stop_reason or "unknown",
                self.shared_state.baseline_tput,
                self.shared_state.cumulative_gain,
                max_minutes_value,
            )
            # Best-effort cleanup of installed handlers; the asyncio loop
            # cleans up automatically on shutdown but explicit removal is
            # tidy and matches the install-step.
            if previous_handlers:
                try:
                    loop = asyncio.get_running_loop()
                    for sig in previous_handlers:
                        loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass
        return self.shared_state.stop_reason

    async def _has_no_more_leverage(self) -> bool:
        """Return True when automated params/backend/kernel levers are exhausted."""
        if self.shared_state.baseline_tput <= 0:
            return False
        if not self.shared_state.current_best:
            return False
        if self.state.pending_proposals:
            return False
        if await self.tasks.queued() or await self.tasks.running():
            return False

        if self.shared_state.params_no_promote_streak < 5:
            return False
        if not self._params_grid_exhausted():
            return False
        return self._all_reusable_kernels_rejected()

    def _params_grid_exhausted(self) -> bool:
        search = self.shared_state.params_search or {}
        if not isinstance(search, dict) or not search:
            return False
        try:
            from .action_executors.params import DEFAULT_PARAMS_GRID
            grid_size = len(DEFAULT_PARAMS_GRID)
        except Exception:  # noqa: BLE001
            grid_size = 0
        tested = search.get("tested") or {}
        tested_count = len(tested) if isinstance(tested, dict) else 0
        cursor = int(search.get("cursor") or 0)
        rejected_count = len(search.get("rejected") or [])
        if grid_size <= 0:
            return bool(search.get("params_search_exhausted"))
        return (
            tested_count >= grid_size
            or cursor >= grid_size
            or rejected_count >= grid_size
        )

    def _all_reusable_kernels_rejected(self) -> bool:
        select = self.shared_state.last_select_kernels or {}
        reusable = {
            str(k) for k in (select.get("reusable_native_kernel_ids") or [])
            if k
        }
        if not reusable:
            return bool(self.shared_state.last_profile_trace)

        rejected = {
            str(k) for k in (self.shared_state.rejected_kernel_ids or [])
            if k
        }
        for entry in self.shared_state.rejected_kernel_patches or []:
            if isinstance(entry, dict) and entry.get("kernel_id"):
                rejected.add(str(entry["kernel_id"]))
        last = self.shared_state.last_kernel_opt or {}
        if last.get("kernel_id") and last.get("decision") == "REVERT":
            rejected.add(str(last["kernel_id"]))
        return reusable <= rejected

    # ==================================================================
    # Reactor
    # ==================================================================
    async def _reactor_pass(self, agent_name: str) -> None:
        backend = self.backends[agent_name]
        prompt = await self._compose_prompt(agent_name)
        sys_prompt = await self._load_system_prompt(agent_name)
        tools = self.policy.allowed_tools_for_agent(agent_name)
        # max_turns=0 → backend uses its own default. ClaudeBackend needs
        # ≥ 2 to accommodate the tool_use → tool_result → final-text turn
        # sequence; mock backends ignore max_turns.
        try:
            result: BackendTurnResult = await backend.run(
                prompt=prompt, system_prompt=sys_prompt, tools=tools, max_turns=0,
            )
        except BackendError as exc:
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "backend_error", "agent": agent_name, "error": repr(exc)},
            )
            return
        except NoIntentEmitted as exc:
            # Reactor turn produced no parseable intents (LLM hiccup,
            # malformed envelope, missing required payload field, ...).
            # Surface as a structured observation so the next tick sees
            # the failure and self-corrects, instead of killing the run.
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "no_intent_emitted", "agent": agent_name,
                 "error": str(exc)[:500]},
            )
            return
        except Exception as exc:  # noqa: BLE001
            # Catch-all so one agent's bad turn never stops the long-run
            # loop. Logged + recorded; the agent gets another shot next
            # tick. (Repeated crashes still drive crash_count → emergency
            # stop in `run()`.)
            log.exception("reactor pass for %s raised", agent_name)
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "reactor_exception", "agent": agent_name,
                 "error": f"{type(exc).__name__}: {str(exc)[:500]}"},
            )
            self.shared_state.crash_count += 1
            self.shared_state.save(self.session_dir)
            return
        for intent in result.intents:
            await self._handle_intent(agent_name, intent)

    async def _compose_prompt(self, agent_name: str) -> str:
        """v0.6 §8.3 prompt: SharedState summary + inbox tail.

        Layout::

            === Shared session state ===
            session_id=...   model=...   baseline_tput=...   ...
            === Inbox for <agent> (newest last) ===
            seq=... msg_id=... from=... topic=... payload=...

        We include the canonical ``msg_id`` for each inbox row so reactive
        mock backends (and future real LLMs) can address replies via
        ``in_reply_to`` / ``target_proposal_msg_id`` without extra lookups.
        """
        sections: list[str] = []

        # 0. SESSION_DIR contract — tell every persistent agent the literal
        # path so they reference it instead of fabricating one. This pairs
        # with PolicyGate's path-containment guard.
        sections.append(f"SESSION_DIR={self.session_dir}")

        # 0a. Mission progress (Orchestration only). Phase 2 — this is the
        # *outcome-shaped* projection of SharedState (raw vs validated
        # gain, time spent vs budget, validate_stack staleness) that the
        # decision framework in the system prompt expects to see at the
        # very top of every tick. Crucially, it surfaces the
        # ``validate_stack required`` signal *before* the verbose
        # SharedState dump so the LLM doesn't miss it.
        if agent_name == "orchestration":
            sections.append("=== Mission progress ===")
            sections.append(self.shared_state.to_mission_summary())

        # 1. Shared session state — gives the agent goal + progress context
        # even on tick 1 when the inbox is empty.
        sections.append("=== Shared session state ===")
        sections.append(self.shared_state.to_prompt_summary())
        if agent_name == "orchestration":
            # Refresh target_gap_pct from the live objective so the rendered
            # scoreboard reflects the *current* remaining gap (objective is
            # already cached on the Coordinator after Coordinator.run set
            # the budget). Non-``gain_pct`` objectives leave the value at 0
            # which maps to a 1.0 multiplier inside scoring.target_gap_multiplier.
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
            if (
                self.action_registry is not None
                and self.shared_state.action_scores
            ):
                sections.append(
                    self.shared_state.to_action_scores_summary(
                        registry=self.action_registry, top_k=12,
                    )
                )
            required_step = self._required_next_step()
            if required_step:
                sections.append("=== Execution checklist (Coordinator-enforced) ===")
                sections.append(required_step)

        # 1b. Marathon KB retrieval — curated lessons (validated stacks).
        if agent_name == "orchestration":
            # Framework is resolved at CLI start time and re-exported as
            # $FRAMEWORK so the KB digest reads the right partition. Fall
            # back to sglang for parity with the CLI default.
            kb_text = format_kb_digest_for_orchestration(
                model_name=getattr(self.shared_state, "model_name", "") or "",
                framework=os.environ.get("FRAMEWORK", "sglang").strip().lower() or "sglang",
            )
            if kb_text.strip():
                sections.append("=== Knowledge base hints ===")
                sections.append(kb_text)

        # 2. Inbox tail since this agent's last cursor.
        cursor = await self.cursors.load(agent_name)
        msgs = await self.bus.replay_for(agent_name, after_seq=cursor.last_processed_seq)
        if msgs:
            sections.append(f"=== Inbox for {agent_name} (newest last) ===")
            for m in msgs[-20:]:
                sections.append(
                    f"  seq={m.seq} msg_id={m.msg_id} from={m.from_agent} "
                    f"topic={m.topic} payload={m.payload}"
                )
        else:
            sections.append(f"=== Inbox for {agent_name} ===")
            sections.append("(no new messages)")

        return "\n".join(sections)

    async def _load_system_prompt(self, agent_name: str) -> str:
        # Demo / test override: callers may pre-stuff a system prompt by
        # setting ``self.system_prompt_overrides[agent_name]``. Useful to
        # short-circuit prereq exploration during smoke runs.
        override = getattr(self, "system_prompt_overrides", {}).get(agent_name)
        if override is not None:
            return override
        role = self.role_registry[agent_name]
        try:
            return role.load_system_prompt()
        except FileNotFoundError:
            return f"(no system prompt for {agent_name})"

    # ==================================================================
    # Execution order guard
    # ==================================================================
    def _required_next_step(self) -> str:
        """Return the coordinator-enforced next step, or empty if flexible.

        The Orchestration prompt says baseline → profile → select_kernels,
        but the LLM can still skip to backends/params. This guard makes that
        sequence deterministic and visible in the prompt every tick.

        Phase 2 addition — once at least one KEEP'd entry has been added to
        ``optimization_stack`` since the last successful ``validate_stack``,
        the validate_stack TODO takes precedence over every other
        non-prep step. Skipping it would let the LLM keep stacking
        per-round gains that don't compose linearly and therefore over-
        report ``cumulative_gain`` in the final report.
        """
        if self.shared_state.stop_reason:
            return ""
        if self.shared_state.baseline_tput <= 0:
            return (
                "TODO 1/4: baseline is required now. Propose/delegate only "
                "`baseline` until baseline_tput > 0."
            )
        # Profile/select_kernels guards only apply when kernel agent is
        # alive in the role registry — no-kernel runs don't have a way to
        # service the request and the mandate would be meaningless.
        if "kernel" in self.role_registry:
            if not self.shared_state.last_profile_trace:
                return (
                    "TODO 2/4: profile is required now. Baseline exists but "
                    "last_profile_trace is empty; propose/delegate only `profile`. "
                    "Do not run backends/params/sweep yet."
                )
            select = self.shared_state.last_select_kernels or {}
            if select.get("trace_input") != self.shared_state.last_profile_trace:
                return (
                    "TODO 3/4: select_kernels is required now. Emit "
                    "request{target_agent='kernel', kind='select_kernels', "
                    "params={trace_input: last_profile_trace, top_k: 10}} before "
                    "backends/params/sweep."
                )
        if self.shared_state.optimization_stack_has_unvalidated_keeps():
            return (
                "TODO 4/4: validate_stack required. New KEEP'd entries have "
                "landed on optimization_stack since the last validate_stack run "
                f"(stack_len={len(self.shared_state.optimization_stack)}, "
                f"validated_at_len="
                f"{self.shared_state.cumulative_gain_validated_stack_len}). "
                "Propose/delegate only `validate_stack` until "
                "cumulative_gain_validated reflects the current stack. "
                "Per-round gains do NOT compose linearly — the final report "
                "quotes the validated number, so this is the only honest gain."
            )
        return ""

    def _sequence_denial_for_action(self, action_name: str) -> PolicyDenied | None:
        """Reject orchestration action/delegate attempts that skip required steps.

        Phase 2 addition: once optimization_stack has unvalidated KEEPs,
        the only allowed actions are ``validate_stack`` itself, ``recover``,
        and ``report`` (the last only when stop_reason is set, which we
        already short-circuit above). Everything else is denied with a
        ``validate_stack_required`` rule so Orchestration sees the
        ``policy_denied`` and self-corrects on the next tick.
        """
        action = str(action_name or "").strip()
        sequence_actions = {
            "baseline", "profile", "backends", "params", "sweep", "report",
            "validate_stack",
        }
        if action not in sequence_actions:
            return None
        if self.shared_state.stop_reason:
            return None
        if self.shared_state.baseline_tput <= 0 and action != "baseline":
            return PolicyDenied(
                f"action={action!r} denied: baseline must run first",
                rule="execution_order",
                hint="propose/delegate `baseline` until baseline_tput > 0",
            )
        # Profile/select_kernels guards only apply when kernel agent is in
        # the role registry — no-kernel mode skips them.
        if "kernel" in self.role_registry:
            if (
                self.shared_state.baseline_tput > 0
                and not self.shared_state.last_profile_trace
                and action not in {"profile", "validate_stack"}
            ):
                return PolicyDenied(
                    f"action={action!r} denied: profile must run before {action!r}",
                    rule="execution_order",
                    hint="propose/delegate `profile`; last_profile_trace is empty",
                )
            select = self.shared_state.last_select_kernels or {}
            needs_select = (
                bool(self.shared_state.last_profile_trace)
                and select.get("trace_input") != self.shared_state.last_profile_trace
            )
            if needs_select and action in {"backends", "params", "sweep", "report"}:
                return PolicyDenied(
                    f"action={action!r} denied: select_kernels must run first",
                    rule="execution_order",
                    hint=(
                        "emit request{target_agent='kernel', kind='select_kernels', "
                        "params={trace_input: last_profile_trace, top_k: 10}}"
                    ),
                )
        # validate_stack precedence — once new KEEPs are stacked we must
        # rebench before any further explore / report. We allow:
        #   - validate_stack itself
        #   - baseline (ad-hoc re-baseline is still okay; rare)
        # and deny the rest.
        if (
            self.shared_state.optimization_stack_has_unvalidated_keeps()
            and action not in {"validate_stack", "baseline"}
        ):
            return PolicyDenied(
                f"action={action!r} denied: validate_stack required first",
                rule="validate_stack_required",
                hint=(
                    "optimization_stack has KEEPs that have not been "
                    "validated end-to-end; propose/delegate `validate_stack` "
                    "before any further explore or report"
                ),
            )
        return None

    def _sequence_denial_for_request(
        self, target_agent: str, kind: str,
    ) -> PolicyDenied | None:
        """Reject kernel requests that skip baseline/profile prerequisites."""
        target = str(target_agent or "").strip()
        req_kind = str(kind or "").strip()
        if target != "kernel" or self.shared_state.stop_reason:
            return None
        # select_kernels is the prerequisite request itself. It is also used
        # directly by tests/tools that pass an explicit trace_input, so allow it
        # through; later backends/params/sweep are guarded until the result is
        # cached in SharedState.
        if req_kind == "select_kernels":
            return None
        if get_handler(req_kind) is None:
            return None
        if self.shared_state.baseline_tput <= 0:
            return PolicyDenied(
                f"request kind={req_kind!r} denied: baseline must run first",
                rule="execution_order",
                hint="propose/delegate `baseline` before kernel requests",
            )
        if not self.shared_state.last_profile_trace:
            return PolicyDenied(
                f"request kind={req_kind!r} denied: profile must run first",
                rule="execution_order",
                hint="propose/delegate `profile` before select_kernels/run_optimization",
            )
        select = self.shared_state.last_select_kernels or {}
        needs_select = select.get("trace_input") != self.shared_state.last_profile_trace
        if needs_select and req_kind != "select_kernels":
            return PolicyDenied(
                f"request kind={req_kind!r} denied: select_kernels must run first",
                rule="execution_order",
                hint="emit request kind='select_kernels' for last_profile_trace",
            )
        return None

    @staticmethod
    def _pmc_roofline_enabled() -> bool:
        return os.environ.get("HYPERLOOM_ENABLE_PMC_ROOFLINE", "").strip().lower() in {
            "1", "true", "yes", "on",
        }

    @staticmethod
    def _pmc_roofline_force() -> bool:
        return os.environ.get("HYPERLOOM_PMC_ROOFLINE_FORCE", "").strip().lower() in {
            "1", "true", "yes", "on",
        }

    def _build_pmc_roofline_params(self) -> dict[str, Any] | None:
        """Build pmc_roofline task params from the materialized Magpie YAML."""
        config_path = self.shared_state.baseline_config_path
        if not config_path:
            log.info("PMC roofline enabled but baseline_config_path is empty")
            return None
        path = Path(config_path)
        if not path.exists():
            log.warning("PMC roofline config missing: %s", path)
            return None
        try:
            import yaml  # type: ignore[import-untyped]
            cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read PMC roofline workload config %s: %s", path, exc)
            return None

        bench = cfg.get("benchmark") if isinstance(cfg.get("benchmark"), dict) else {}
        envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
        framework = str(bench.get("framework") or self.shared_state.framework or "vllm").lower()
        model = str(bench.get("model") or self.shared_state.model_path or "").strip()
        if not model:
            log.warning("PMC roofline config has no model path")
            return None

        port = int(os.environ.get("HYPERLOOM_PMC_ROOFLINE_PORT", "30001"))
        tp = str(envs.get("TP") or os.environ.get("TP") or "1")
        precision = str(bench.get("precision") or os.environ.get("PRECISION") or "bf16").lower()
        max_model_len = str(envs.get("MAX_MODEL_LEN") or os.environ.get("MAX_MODEL_LEN") or "8192")
        extra_key = "EXTRA_VLLM_ARGS" if framework == "vllm" else "EXTRA_SGLANG_ARGS"
        extra_args = shlex.split(str(envs.get(extra_key) or ""))
        gpu_type = (
            os.environ.get("HYPERLOOM_PMC_ROOFLINE_GPU_TYPE")
            or self.shared_state.gpu_type
            or os.environ.get("GPU_TYPE", "")
        ).strip().lower()

        if framework == "vllm":
            dtype = "bfloat16" if precision in {"bf16", "bfloat16"} else precision
            server_cmd = [
                "vllm", "serve", model,
                "--host", "0.0.0.0",
                "--port", str(port),
                "--tensor-parallel-size", tp,
                "--trust-remote-code",
                "--dtype", dtype,
                "--max-model-len", max_model_len,
            ] + extra_args
            backend = "vllm"
        else:
            server_cmd = [
                "python", "-m", "sglang.launch_server",
                "--model-path", model,
                "--host", "0.0.0.0",
                "--port", str(port),
                "--tensor-parallel-size", tp,
                "--trust-remote-code",
                "--max-model-len", max_model_len,
            ] + extra_args
            backend = "sglang"

        inferencex = os.environ.get("INFERENCEX_PATH", "/hyperloom/InferenceX").rstrip("/")
        bench_script = f"{inferencex}/utils/bench_serving/benchmark_serving.py"
        isl = str(envs.get("ISL") or os.environ.get("ISL") or "256")
        osl = str(envs.get("OSL") or os.environ.get("OSL") or "256")
        conc = int(envs.get("CONC") or os.environ.get("CONC") or 1)
        num_prompts = str(envs.get("NUM_PROMPTS") or max(conc, 1))
        benchmark_cmd = [
            "python", bench_script,
            "--backend", backend,
            "--base-url", f"http://127.0.0.1:{port}",
            "--model", model,
            "--dataset-name", "random",
            "--random-input-len", isl,
            "--random-output-len", osl,
            "--num-prompts", num_prompts,
            "--max-concurrency", str(conc),
            "--request-rate", "inf",
            "--ignore-eos",
        ]

        return {
            "profile_mode": os.environ.get("HYPERLOOM_PMC_ROOFLINE_MODE", "launch"),
            "server_cmd": server_cmd,
            "health_url": f"http://127.0.0.1:{port}/health",
            "benchmark_cmd": benchmark_cmd,
            "output_dir": str(Path(self.session_dir) / "runs" / "pmc_roofline" / "auto"),
            "duration_ms": int(os.environ.get("HYPERLOOM_PMC_ROOFLINE_DURATION_MS", "15000")),
            "precision": precision,
            "gpu_type": gpu_type,
            "startup_timeout_s": int(os.environ.get("HYPERLOOM_PMC_ROOFLINE_STARTUP_TIMEOUT_S", "600")),
        }

    async def _maybe_enqueue_pmc_roofline(self) -> None:
        if not self._pmc_roofline_enabled():
            return
        if self.shared_state.last_profile_roofline and not self._pmc_roofline_force():
            return
        params = self._build_pmc_roofline_params()
        if not params:
            return
        key_src = "|".join([
            self.shared_state.last_profile_trace or "",
            self.shared_state.baseline_config_path or "",
            str(params.get("profile_mode") or ""),
        ])
        key = hashlib.sha256(key_src.encode("utf-8", errors="ignore")).hexdigest()[:16]
        task = await self.tasks.create(
            kind="pmc_roofline",
            params=params,
            idempotency_key=f"auto-pmc-roofline-{key}",
            requires_lanes=["profile_lane"],
            lease_ttl_sec=1800,
        )
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "event",
            {
                "kind": "task_queued",
                "task_id": task.task_id,
                "source": "coordinator_auto_pmc_roofline",
                "action": "pmc_roofline",
            },
        ))

    # ==================================================================
    # Intent handling
    # ==================================================================
    async def _handle_intent(self, source: str, intent: Intent) -> None:
        try:
            self.policy.validate_intent(source, intent)
        except PolicyDenied as denied:
            await self._record_policy_denied(source, intent, denied)
            return

        it = intent.type
        if it == IntentType.PROPOSE_ACTION:
            await self._handle_propose_action(source, intent)
        elif it == IntentType.REVIEW_VERDICT:
            await self._handle_review_verdict(source, intent)
        elif it == IntentType.DELEGATE:
            await self._handle_delegate(source, intent)
        elif it == IntentType.REQUEST:
            await self._handle_request(source, intent)
        elif it == IntentType.RESPONSE:
            await self._handle_response(source, intent)
        elif it == IntentType.KILL_TASK:
            await self._handle_kill_task(source, intent)
        elif it == IntentType.PRUNE_BRANCH:
            await self._handle_prune_branch(source, intent)
        elif it == IntentType.FORCE_DISPATCH:
            await self._handle_force_dispatch(source, intent)
        elif it == IntentType.ESCALATE_STRATEGY_CHANGE:
            await self._handle_escalate_strategy_change(source, intent)
        elif it == IntentType.SEND_MESSAGE:
            await self._handle_send_message(source, intent)
        elif it == IntentType.ALERT:
            await self._handle_alert(source, intent)
        elif it == IntentType.UPDATE_STATE:
            await self._handle_update_state(source, intent)
        else:
            # ASK_QUESTION / ANSWER / UPDATE_PERSONA — record for replay
            await self._record_observation(
                source, "observation",
                {"intent": it.value, "payload": intent.payload},
            )
        await self._cursor_advance_to_latest(source)

    # ------------------------------------------------------------------
    # PROPOSE_ACTION + REVIEW_VERDICT
    # ------------------------------------------------------------------
    async def _handle_propose_action(self, source: str, intent: Intent) -> None:
        action_name = intent.payload["action_name"]
        # Pruned-family check reads from persistent SharedState so resume
        # after crash still respects the prune.
        if self.shared_state.is_pruned(action_name):
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "proposal_pruned", "from": source, "action": action_name},
            )
            return
        denied = self._sequence_denial_for_action(action_name)
        if denied is not None:
            await self._record_policy_denied(source, intent, denied)
            return
        msg = Message.new(
            source, "*", "proposal",
            {**intent.payload, "needs_review": True},
            priority=1,
        )
        await self.bus.append_and_seq(msg)
        self.state.pending_proposals[msg.msg_id] = PendingProposal(
            proposal_msg_id=msg.msg_id,
            from_agent=source,
            action_name=action_name,
            predicted_gain_pct=float(intent.payload.get("predicted_gain_pct", 0.0)),
            payload=dict(intent.payload),
        )

    async def _handle_review_verdict(self, source: str, intent: Intent) -> None:
        target = intent.payload["target_proposal_msg_id"]
        verdict = intent.payload["verdict"]
        pending = self.state.pending_proposals.get(target)
        if pending is None:
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "verdict_for_unknown_proposal", "target": target, "verdict": verdict},
            )
            return
        pending.decided = True
        pending.verdict = verdict
        # Mirror the verdict onto the bus so the original proposer's reactor
        # picks it up next tick.
        await self.bus.append_and_seq(Message.new(
            source, pending.from_agent, "review_verdict",
            {"target_proposal_msg_id": target, "verdict": verdict,
             "reasoning": intent.payload.get("reasoning", "")},
            priority=0 if verdict == "reject" else 1,
            in_reply_to=target,
        ))
        if verdict == "approve":
            await self._materialize_approved_proposal(pending)

    async def _materialize_approved_proposal(self, pending: PendingProposal) -> None:
        """Promote an approved proposal into a TaskRegistry entry.

        For grid-style executors (backends / params / sweep) we inject the
        current best throughput as ``base_tput`` so they can compute
        gain%; otherwise the runner's default of 0.0 makes
        best_gain_pct uninformative (DESIGN §16 baseline_tput parameter).
        """
        params = dict(pending.payload.get("params") or {})
        cb = self.shared_state.current_best or {}
        cb_args = (
            str(cb.get("extra_sglang_args") or "")
            if isinstance(cb, dict) else ""
        )
        if pending.action_name == "profile":
            # Profile itself does not yet consume base_extra_args, but we
            # stamp it onto the task so the post-task promotion records the
            # server config that produced this trace.
            params.setdefault("base_extra_args", cb_args)
        if pending.action_name in ("backends", "params", "sweep"):
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            base = cb_tput if isinstance(cb_tput, (int, float)) and cb_tput > 0 \
                else self.shared_state.baseline_tput
            params.setdefault("base_tput", float(base or 0.0))
            params.setdefault("base_extra_args", cb_args)
            # Plumb baseline's materialized YAML so variant runs honor the
            # same workload contract (CONC/ISL/OSL/TP/etc.) baseline ran.
            # Without this each variant would re-render from the shipped
            # YAML's smoke defaults and produce ~10x lower throughput.
            # `setdefault` lets the proposer override (e.g. profile re-uses
            # this path too, or a deliberate cross-workload sweep).
            if self.shared_state.baseline_config_path:
                params.setdefault(
                    "config_path", self.shared_state.baseline_config_path
                )
            # T2 — pass the cumulative synergy_attempted set so backends
            # phase-2 doesn't re-test the same combo across rounds, and
            # cap each round at 5 phase-1 variants (parity with `params`).
            # We also feed prior-round phase-1 variant names so each new
            # round deepens the search instead of replaying the first 5.
            if pending.action_name == "backends":
                params.setdefault(
                    "synergy_attempted",
                    list(self.shared_state.synergy_attempted),
                )
                params.setdefault("max_candidates_per_round", 5)
                params.setdefault("max_synergy_combos", 4)
                # Flatten every variant name observed in past
                # backend_winners_history rounds; the executor uses this
                # to skip already-tested entries when cap > 0.
                tested_names: list[str] = []
                for round_entry in self.shared_state.backend_winners_history:
                    if not isinstance(round_entry, dict):
                        continue
                    if round_entry.get("action") != "backends":
                        continue
                    for w in round_entry.get("winners") or []:
                        if isinstance(w, dict) and w.get("name"):
                            tested_names.append(str(w["name"]))
                if tested_names:
                    params.setdefault(
                        "tested_variant_names", tested_names,
                    )
            if pending.action_name == "params":
                params.setdefault("params_search", self.shared_state.params_search)
                # Long runs should advance the search incrementally while
                # still covering the params grid quickly enough to compete
                # with kernel work. Direct runner calls/tests can still pass
                # 0 to run the full grid.
                params.setdefault("max_candidates_per_round", 5)
                if isinstance(cb, dict) and cb.get("variant_name"):
                    params.setdefault("base_variant_name", str(cb["variant_name"]))
        task = await self.tasks.create(
            kind=pending.action_name,
            params=params,
            idempotency_key=f"approved-{pending.proposal_msg_id}",
        )
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "decision",
            {"kind": "approved_proposal", "task_id": task.task_id,
             "action_name": pending.action_name, "from_agent": pending.from_agent},
        ))

    # ------------------------------------------------------------------
    # DELEGATE
    # ------------------------------------------------------------------
    async def _handle_delegate(self, source: str, intent: Intent) -> None:
        action_name = intent.payload["action_name"]
        denied = self._sequence_denial_for_action(action_name)
        if denied is not None:
            await self._record_policy_denied(source, intent, denied)
            return
        params = dict(intent.payload.get("params") or {})
        # Plumb baseline's materialized YAML into grid-style delegated tasks
        # so they inherit the workload contract (CONC/ISL/OSL/TP/...) baseline
        # ran. See `_materialize_approved_proposal` for the same logic on the
        # proposal/review path. `setdefault` lets the delegator override.
        if (
            action_name in ("backends", "params", "sweep")
            and self.shared_state.baseline_config_path
        ):
            params.setdefault(
                "config_path", self.shared_state.baseline_config_path
            )
        # T2 — same synergy-dedup + per-round cap plumbing as the
        # proposal-review path.
        if action_name == "backends":
            params.setdefault(
                "synergy_attempted",
                list(self.shared_state.synergy_attempted),
            )
            params.setdefault("max_candidates_per_round", 5)
            params.setdefault("max_synergy_combos", 4)
            tested_names: list[str] = []
            for round_entry in self.shared_state.backend_winners_history:
                if not isinstance(round_entry, dict):
                    continue
                if round_entry.get("action") != "backends":
                    continue
                for w in round_entry.get("winners") or []:
                    if isinstance(w, dict) and w.get("name"):
                        tested_names.append(str(w["name"]))
            if tested_names:
                params.setdefault(
                    "tested_variant_names", tested_names,
                )
        idempotency_key = intent.payload.get("idempotency_key") or f"{source}:{action_name}:{len(self.state.pending_proposals)}"
        task = await self.tasks.create(
            kind=action_name,
            params=params,
            idempotency_key=idempotency_key,
        )
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "event",
            {"kind": "task_queued", "task_id": task.task_id, "source": source, "action": action_name},
        ))

    # ------------------------------------------------------------------
    # REQUEST / RESPONSE (Plan A)
    # ------------------------------------------------------------------
    async def _handle_request(self, source: str, intent: Intent) -> None:
        target_agent = intent.payload["target_agent"]
        kind = intent.payload["kind"]
        denied = self._sequence_denial_for_request(target_agent, kind)
        if denied is not None:
            await self._record_policy_denied(source, intent, denied)
            return
        # Always record the request on the bus so the kernel reactor
        # (and tests / replay) can see it.
        request_msg = Message.new(
            source, target_agent, "request", dict(intent.payload), priority=1,
        )
        await self.bus.append_and_seq(request_msg)

        # Safety net: if target agent was removed from the role registry
        # (e.g. --no-kernel), auto-reject the request so Orchestration
        # does not hang waiting for a response from a non-existent agent.
        if target_agent not in self.role_registry:
            await self.bus.append_and_seq(Message.new(
                target_agent, source, "response",
                {
                    "in_reply_to": request_msg.msg_id,
                    "kind": f"{kind}_done",
                    "status": "failed",
                    "result": {
                        "status": "failed",
                        "error_class": "agent_disabled",
                        "error": f"{target_agent} agent is disabled for this session",
                    },
                    "source": "coordinator_auto_reject",
                },
                in_reply_to=request_msg.msg_id, priority=1,
            ))
            return

        # Programmatic shortcut: if the kernel agent has a registered
        # handler for this `kind`, run it inline and emit RESPONSE on its
        # behalf so we don't burn an extra LLM turn for a deterministic
        # shell-tool invocation. See kernel_request_handlers.py for the
        # rationale.
        if target_agent == "kernel":
            handler = get_handler(kind)
            if handler is not None:
                params = intent.payload.get("params") or {}
                merged_payload = {**intent.payload, **params}
                if (
                    kind == "select_kernels"
                    and self.shared_state.last_profile_roofline
                    and not merged_payload.get("roofline_json")
                ):
                    merged_payload["roofline_json"] = self.shared_state.last_profile_roofline
                cache_hit_source = None
                cached_result = self._cached_kernel_request(kind, merged_payload)
                if cached_result is not None:
                    result = cached_result
                    cache_hit_source = "shared_state_cache"
                else:
                    rejected = (
                        self.shared_state.find_rejected_kernel_patch(merged_payload)
                        if kind == "integrate"
                        else None
                    )
                    if rejected is not None:
                        result = {
                            "status": "skipped",
                            "decision": "REVERT",
                            "error_class": "kernel_patch_rejected",
                            "error": "same kernel patch already exhausted E2E attempts",
                            "kernel_id": rejected.get("kernel_id"),
                            "patch_path": rejected.get("patch_path"),
                            "target_file": rejected.get("target_file"),
                            "extra_sglang_args": rejected.get("extra_sglang_args", ""),
                            "attempt_count": rejected.get("attempt_count"),
                            "best_gain_pct": rejected.get("best_gain_pct"),
                            "reason": rejected.get("reason"),
                        }
                        cache_hit_source = "shared_state_kernel_rejection"
                    else:
                        try:
                            result = await handler(
                                merged_payload,
                                session_dir=self.session_dir,
                            )
                        except Exception as exc:  # noqa: BLE001
                            log.exception(
                                "kernel_request_handler[%s] crashed for source=%s",
                                kind, source,
                            )
                            result = {
                                "status": "failed",
                                "error_class": "handler_exception",
                                "error": repr(exc),
                            }
                await self.bus.append_and_seq(Message.new(
                    "kernel", source, "response",
                    {
                        "in_reply_to": request_msg.msg_id,
                        "kind": f"{kind}_done",
                        "status": result.get("status", "ok"),
                        "result": result,
                        "source": cache_hit_source or "programmatic_handler",
                    },
                    in_reply_to=request_msg.msg_id, priority=1,
                ))
                # Cache select_kernels output so subsequent identical
                # requests are short-circuited next tick. Only cache real
                # successful runs, not failures, to avoid sticky errors.
                if (
                    kind == "select_kernels"
                    and cache_hit_source is None
                    and result.get("status") in ("ok", "succeeded")
                ):
                    self.shared_state.record_select_kernels(merged_payload, result)
                    await self._maybe_enqueue_pmc_roofline()
                    self.shared_state.save(self.session_dir)
                # Mirror kernel-opt outcomes into SharedState so Orch
                # sees decision/speedup in its prompt next tick and
                # doesn't re-dispatch the same kernel_id forever.
                if kind == "run_optimization":
                    self.shared_state.record_kernel_opt(result)
                    # Wire run_optimization decision (KEEP / REVERT / PARTIAL)
                    # into the per-action scoring for kernel_opt. KEEP uses
                    # the micro_speedup if it surfaces a percentage-shaped
                    # number; otherwise we record a zero-gain KEEP (still
                    # bumps runs + sets cooldown).
                    verification = (
                        result.get("verification")
                        if isinstance(result, dict)
                        else None
                    ) or {}
                    proposal = (
                        result.get("proposal")
                        if isinstance(result, dict)
                        else None
                    ) or {}
                    decision = str(proposal.get("decision", "")).upper()
                    if decision == "KEEP":
                        speedup = verification.get("micro_speedup")
                        gain = 0.0
                        if isinstance(speedup, (int, float)) and speedup > 1.0:
                            gain = float(speedup - 1.0) * 100.0
                        self._score_action_keep("kernel_opt", gain_pct=gain)
                    elif decision in {"REVERT", "PARTIAL"}:
                        self._score_action_discard("kernel_opt")
                    else:
                        # Unknown decision — treat as a measurement, bump
                        # runs + cooldown without rewarding the action.
                        self._score_action_keep("kernel_opt", gain_pct=0.0)
                    self.shared_state.save(self.session_dir)
                if kind == "integrate":
                    if result.get("status") != "skipped":
                        self.shared_state.record_kernel_integrate_result(result)
                    decision = str(result.get("decision", "")).upper()
                    if decision == "KEEP":
                        self._record_integrate_keep(result)
                        gain = result.get("gain_pct")
                        gain_f = (
                            float(gain) if isinstance(gain, (int, float)) else 0.0
                        )
                        self._score_action_keep("integrate", gain_pct=max(0.0, gain_f))
                    elif decision in {"REVERT", "NEEDS_REVIEW"}:
                        self._score_action_discard("integrate")
                    elif result.get("status") != "skipped":
                        self._score_action_keep("integrate", gain_pct=0.0)
                    self.shared_state.save(self.session_dir)
                # Bug B fix: the request was just answered programmatically,
                # so the LLM-backed kernel agent should NOT see the request
                # in its inbox next tick (otherwise it duplicates the
                # response with hallucinated content). Advance the kernel
                # cursor past this request seq. cursor.advance is monotonic,
                # so this is safe even if kernel had already processed
                # earlier seqs in the same tick.
                await self.cursors.advance(
                    target_agent,
                    seq=request_msg.seq,
                    msg_id=request_msg.msg_id,
                )

    def _cached_kernel_request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return a cached programmatic_handler result if applicable."""
        if kind != "select_kernels":
            return None
        cached = self.shared_state.last_select_kernels or {}
        if not isinstance(cached, dict) or not cached:
            return None
        trace_input = payload.get("trace_input") or payload.get("trace_dir")
        if not trace_input or trace_input != cached.get("trace_input"):
            return None
        candidates_path = cached.get("candidates_path")
        if not candidates_path or not Path(candidates_path).exists():
            return None
        return {
            "status": "ok",
            "candidates_path": candidates_path,
            "hot_kernels_top15": cached.get("hot_kernels_top15", []),
            "reusable_native_kernel_ids": cached.get(
                "reusable_native_kernel_ids", []
            ),
            "cached_at": cached.get("ts"),
            "note": "served from shared_state.last_select_kernels cache",
        }

    async def _handle_response(self, source: str, intent: Intent) -> None:
        in_reply_to = intent.payload["in_reply_to"]
        # Locate the original requester so we can address the response.
        original = await self.bus.lookup_by_id(in_reply_to)
        target = original.from_agent if original else "*"
        await self.bus.append_and_seq(Message.new(
            source, target, "response",
            dict(intent.payload), in_reply_to=in_reply_to, priority=1,
        ))

    # ------------------------------------------------------------------
    # Robustness scheduling-police
    # ------------------------------------------------------------------
    async def _handle_kill_task(self, source: str, intent: Intent) -> None:
        task_id = intent.payload["task_id"]
        try:
            task = await self.tasks.get(task_id)
        except Exception:  # noqa: BLE001 — TaskNotFound
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "kill_task_unknown", "task_id": task_id, "source": source},
            )
            return
        if task.state in ("queued", "running"):
            await self.tasks.transition(
                task_id, "cancelled",
                evidence={"reason": intent.payload.get("reason"), "by": source},
            )
        await self.bus.append_and_seq(Message.new(
            source, "*", "kill",
            {"task_id": task_id, "reason": intent.payload.get("reason")},
        ))

    async def _handle_prune_branch(self, source: str, intent: Intent) -> None:
        family = intent.payload["family"]
        if self.shared_state.add_pruned_family(family):
            self.shared_state.save(self.session_dir)
        cancelled = await self.tasks.cancel_family([family])
        await self.bus.append_and_seq(Message.new(
            source, "*", "event",
            {"kind": "prune_branch", "family": family,
             "cancelled_task_ids": cancelled,
             "reason": intent.payload.get("reason")},
        ))

    async def _handle_force_dispatch(self, source: str, intent: Intent) -> None:
        # P0-3 stub: just emit an event; real dispatcher reordering lands
        # in P0-5 with the priority queue.
        await self.bus.append_and_seq(Message.new(
            source, "*", "event",
            {"kind": "force_dispatch", "task_id": intent.payload["task_id"],
             "reason": intent.payload.get("reason")},
        ))

    async def _handle_escalate_strategy_change(self, source: str, intent: Intent) -> None:
        # Priority-0 broadcast — non-destructive (DESIGN §19.3.4).
        await self.bus.append_and_seq(Message.new(
            source, "*", "strategy_change",
            dict(intent.payload), priority=0,
        ))

    # ------------------------------------------------------------------
    # SEND_MESSAGE / ALERT / UPDATE_STATE — minimal persistence
    # ------------------------------------------------------------------
    async def _handle_send_message(self, source: str, intent: Intent) -> None:
        topic = intent.payload.get("topic", "observation")
        if topic not in __import__("inference_optimizer.orchestrator.message_bus",
                                    fromlist=["TOPIC_ALLOWLIST"]).TOPIC_ALLOWLIST:
            # Soft-degrade unknown topic per DESIGN §13.2.
            topic = "observation"
        to_agent = intent.payload.get("to") or "*"
        await self.bus.append_and_seq(Message.new(
            source, to_agent, topic, {k: v for k, v in intent.payload.items() if k != "to"},
        ))

    async def _handle_alert(self, source: str, intent: Intent) -> None:
        prio = 0 if intent.payload.get("severity") == "high" else 1
        await self.bus.append_and_seq(Message.new(
            source, "*", "alert", dict(intent.payload), priority=prio,
        ))

    async def _handle_update_state(self, source: str, intent: Intent) -> None:
        # Apply to persistent SharedState (PolicyGate already enforced that
        # the source role can't write CORE_STATE_FIELDS unless allowed).
        applied = self.shared_state.apply_changes(
            intent.payload["changes"], allow_core=False,
        )
        if applied:
            self.shared_state.save(self.session_dir)
        await self.bus.append_and_seq(Message.new(
            source, "*", "observation",
            {"kind": "update_state", "changes": applied,
             "rejected": sorted(set(intent.payload["changes"]) - set(applied))},
        ))

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    async def _record_policy_denied(
        self, source: str, intent: Intent, denied: PolicyDenied
    ) -> None:
        await self.bus.append_and_seq(Message.new(
            "coordinator", source, "observation",
            {
                "kind": "policy_denied",
                "intent_type": intent.type.value,
                "rule": denied.rule,
                "hint": denied.hint,
                "reason": str(denied),
            },
            priority=0,
        ))

    async def _record_observation(self, source: str, topic: str, payload: dict) -> None:
        await self.bus.append_and_seq(Message.new(source, "*", topic, payload))

    async def _cursor_advance_to_latest(self, agent_name: str) -> None:
        latest = await self.bus.tail(n=1, to_agent=agent_name)
        if latest:
            top = latest[0]
            await self.cursors.advance(agent_name, seq=top.seq, msg_id=top.msg_id)

    def _record_integrate_keep(self, result: dict[str, Any]) -> None:
        new_tput = result.get("new_tput")
        if not isinstance(new_tput, (int, float)) or new_tput <= 0:
            return
        if not self.shared_state.optimization_stack:
            self.shared_state.seed_stack_from_current_best()

        cb = self.shared_state.current_best or {}
        extra_args = str(
            result.get("extra_sglang_args")
            or (cb.get("extra_sglang_args") if isinstance(cb, dict) else "")
            or ""
        ).strip()
        apply_result = result.get("apply_result") or {}
        entry = {
            "action": "integrate",
            "kernel_id": result.get("kernel_id"),
            "patch_path": result.get("patch_path"),
            "target_file": result.get("target_file"),
            "backup_manifest": (
                apply_result.get("manifest_path")
                if isinstance(apply_result, dict) else None
            ),
            "gain_pct": result.get("gain_pct"),
            "tput": float(new_tput),
            "workspace": result.get("workspace"),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        key = (entry["kernel_id"], entry["patch_path"], entry["target_file"])
        existing = {
            (item.get("kernel_id"), item.get("patch_path"), item.get("target_file"))
            for item in self.shared_state.optimization_stack
            if isinstance(item, dict) and item.get("action") == "integrate"
        }
        if key not in existing:
            self.shared_state.optimization_stack.append(entry)

        self.shared_state.current_best = {
            "action": "integrate",
            "tput": float(new_tput),
            "kernel_id": result.get("kernel_id"),
            "extra_sglang_args": extra_args,
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": result.get("ttft_mean_ms"),
            "e2el_mean_ms": result.get("e2el_mean_ms"),
            "workspace": result.get("workspace"),
        }
        if self.shared_state.baseline_tput > 0:
            self.shared_state.cumulative_gain = (
                (float(new_tput) - self.shared_state.baseline_tput)
                / self.shared_state.baseline_tput * 100.0
            )

    # ==================================================================
    # Dispatcher (pulls queued tasks → SubAgentRunner)
    # ==================================================================
    def _is_promotable_result(self, task_kind: str, result: dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        if task_kind in ("baseline", "profile", "validate_stack"):
            return is_valid_measurement(result)
        if task_kind in ("backends", "params", "sweep"):
            return result.get("status") == "succeeded"
        return result.get("status") != "failed"

    async def _handle_unpromotable_result(
        self, task: Task, result: dict[str, Any] | None,
    ) -> None:
        if task.kind != "baseline" or self.shared_state.baseline_tput > 0:
            return
        self.shared_state.baseline_failure_streak += 1
        if self.shared_state.baseline_failure_streak >= 3:
            self.shared_state.stop_reason = "baseline_failed"
        self.shared_state.save(self.session_dir)
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "event",
            {
                "kind": "baseline_not_promoted",
                "task_id": task.task_id,
                "failure_streak": self.shared_state.baseline_failure_streak,
                "stop_reason": self.shared_state.stop_reason,
                "result_status": (result or {}).get("status"),
                "error_class": (result or {}).get("error_class"),
            },
        ))

    async def _pump_dispatcher_once(self) -> None:
        queued = await self.tasks.queued()
        for task in queued:
            try:
                result = await self.sub.run_task(task)
            except Exception as exc:  # noqa: BLE001
                log.exception("dispatcher: failed to run task %s", task.task_id)
                continue
            await self.bus.append_and_seq(Message.new(
                "coordinator", "*", "delegated_result",
                {"task_id": task.task_id, "kind": task.kind,
                 "state": result.state, "result": result.result,
                 "error": result.error},
            ))
            # Auto-promote certain succeeded results into SharedState core
            # fields (Coordinator is the only writer of CORE_STATE_FIELDS;
            # see DESIGN §14.5 / §17.2).
            # Guard: SubAgentResult.state == "succeeded" only means the
            # executor didn't throw. Promotion is tied to task-specific
            # invariants: baseline/profile require a real measurement, while
            # grid actions require at least one successful variant.
            if (
                result.state == "succeeded"
                and self._is_promotable_result(task.kind, result.result or {})
            ):
                await self._promote_to_shared_state(
                    task.kind, result.result, task=task,
                )
            else:
                await self._handle_unpromotable_result(task, result.result)

    def _lift_to_current_best(
        self, task_kind: str, best_tput: float, bv: dict[str, Any],
    ) -> None:
        """Update SharedState.current_best + recompute cumulative_gain.

        Helper for both the 1-shot KEEP threshold path and the
        cross-round consistent-winner path in _promote_to_shared_state.
        """
        previous = self.shared_state.current_best or {}
        if not self.shared_state.optimization_stack:
            self.shared_state.seed_stack_from_current_best()

        base_args = ""
        if isinstance(previous, dict):
            base_args = str(previous.get("extra_sglang_args") or "").strip()
        candidate_args = ""
        if isinstance(bv, dict):
            candidate_args = str(
                bv.get("candidate_extra_sglang_args")
                or bv.get("extra_sglang_args")
                or ""
            ).strip()
        full_args = ""
        if isinstance(bv, dict):
            full_args = str(bv.get("extra_sglang_args") or "").strip()
        if base_args and candidate_args and full_args == candidate_args:
            full_args = " ".join((base_args, candidate_args))
        elif not full_args:
            full_args = " ".join(
                part for part in (base_args, candidate_args) if part
            )

        variant_name = bv.get("name") if isinstance(bv, dict) else None
        if candidate_args or variant_name:
            existing = {
                (str(e.get("action")), str(e.get("variant_name")))
                for e in self.shared_state.optimization_stack
                if isinstance(e, dict)
            }
            key = (task_kind, str(variant_name or ""))
            if key not in existing:
                self.shared_state.optimization_stack.append({
                    "action": task_kind,
                    "variant_name": variant_name,
                    "candidate_extra_sglang_args": candidate_args,
                    "extra_envs": (
                        dict(bv.get("extra_envs") or {})
                        if isinstance(bv, dict) else {}
                    ),
                    "tput": float(best_tput),
                    "workspace": (
                        bv.get("workspace") if isinstance(bv, dict) else None
                    ),
                    "ts": datetime.now(timezone.utc).isoformat(),
                })

        self.shared_state.current_best = {
            "action": task_kind,
            "tput": float(best_tput),
            "variant_name": variant_name,
            "extra_sglang_args": full_args,
            "extra_envs": (
                dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}
            ),
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": bv.get("ttft_mean_ms") if isinstance(bv, dict) else None,
            "e2el_mean_ms": bv.get("e2el_mean_ms") if isinstance(bv, dict) else None,
            "workspace": bv.get("workspace") if isinstance(bv, dict) else None,
        }
        if self.shared_state.baseline_tput > 0:
            self.shared_state.cumulative_gain = (
                (float(best_tput) - self.shared_state.baseline_tput)
                / self.shared_state.baseline_tput * 100.0
            )

    async def _promote_to_shared_state(
        self,
        task_kind: str,
        result: dict,
        *,
        task: "Task | None" = None,
    ) -> None:
        """Lift specific action-result fields into the persistent SharedState.

        Currently handled:

        * ``baseline``  → baseline_tput (output_throughput) + baseline_accuracy
                          (when present) + current_best snapshot
        """
        if not isinstance(result, dict):
            return
        changed = False
        if task_kind == "baseline":
            tput = result.get("output_throughput")
            if isinstance(tput, (int, float)) and tput > 0:
                self.shared_state.baseline_tput = float(tput)
                self.shared_state.baseline_failure_streak = 0
                changed = True
            acc = result.get("accuracy")
            if isinstance(acc, (int, float)):
                self.shared_state.baseline_accuracy = float(acc)
                changed = True
            # Persist the materialized YAML so downstream params/backends/
            # sweep tasks can reuse the exact workload contract baseline ran
            # (see _materialize_approved_proposal / _handle_delegate where we
            # plumb this in as ``task.params["config_path"]``).
            materialized = result.get("materialized_config")
            if isinstance(materialized, str) and materialized:
                self.shared_state.baseline_config_path = materialized
                changed = True
            self.shared_state.current_best = {
                "action": "baseline",
                "tput": float(tput) if isinstance(tput, (int, float)) else None,
                "ttft_mean_ms": result.get("ttft_mean_ms"),
                "e2el_mean_ms": result.get("e2el_mean_ms"),
                "workspace": result.get("workspace"),
            }
            changed = True
        elif task_kind == "profile":
            # Bug C fix: surface the trace path produced by ProfileExecutor
            # to SharedState so Orch can pass a real path to the kernel
            # `select_kernels` REQUEST instead of fabricating one.
            trace_path = (
                result.get("main_trace_path")
                or (result.get("trace_files") or [None])[0]
                or result.get("trace_dir")
            )
            if trace_path:
                self.shared_state.last_profile_trace = str(trace_path)
                # Record the server config in effect for this trace so
                # Orchestration can decide whether to re-profile.
                profile_args = ""
                if task is not None:
                    profile_args = str(
                        (task.params or {}).get("base_extra_args") or ""
                    )
                self.shared_state.last_profile_args = profile_args
                # Stale select_kernels cache no longer matches this trace.
                self.shared_state.last_select_kernels = {}
                changed = True
            # profile result may also include a tput; promote into
            # current_best on the same +1% rule the grid path uses below.
            tput = result.get("output_throughput")
            cb = self.shared_state.current_best or {}
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            cur_best = float(cb_tput) if isinstance(cb_tput, (int, float)) and cb_tput > 0 \
                else float(self.shared_state.baseline_tput or 0.0)
            if (
                isinstance(tput, (int, float)) and tput > 0 and cur_best > 0
                and (tput - cur_best) / cur_best * 100.0 >= 1.0
            ):
                self.shared_state.current_best = {
                    "action": "profile",
                    "tput": float(tput),
                    "ttft_mean_ms": result.get("ttft_mean_ms"),
                    "e2el_mean_ms": result.get("e2el_mean_ms"),
                    "workspace": result.get("workspace"),
                }
                if self.shared_state.baseline_tput > 0:
                    self.shared_state.cumulative_gain = (
                        (float(tput) - self.shared_state.baseline_tput)
                        / self.shared_state.baseline_tput * 100.0
                    )
                changed = True
        elif task_kind == "pmc_roofline":
            if result.get("pmc_summary_path"):
                self.shared_state.last_profile_pmc_summary = str(result["pmc_summary_path"])
                changed = True
            if result.get("roofline_path"):
                self.shared_state.last_profile_roofline = str(result["roofline_path"])
                self.shared_state.last_select_kernels = {}
                changed = True
            if result.get("kernel_breakdown_path"):
                self.shared_state.last_profile_kernel_breakdown = str(result["kernel_breakdown_path"])
                changed = True
        elif task_kind == "validate_stack":
            # Phase 3 — apply the rebenched throughput from the
            # ValidateStackExecutor as the *only* source of truth for
            # ``cumulative_gain_validated`` (CORE_STATE_FIELDS member).
            # We deliberately DO NOT touch ``current_best`` /
            # ``cumulative_gain`` / ``optimization_stack`` here because
            # validate_stack is a measurement of an already-applied
            # configuration, not a new modification.
            tput = result.get("output_throughput")
            stack_len_at_run = result.get("validated_stack_len")
            if isinstance(tput, (int, float)) and tput > 0 and self.shared_state.baseline_tput > 0:
                gain = (
                    (float(tput) - self.shared_state.baseline_tput)
                    / self.shared_state.baseline_tput * 100.0
                )
                self.shared_state.cumulative_gain_validated = float(gain)
                self.shared_state.cumulative_gain_validated_ts = (
                    datetime.now(timezone.utc).isoformat()
                )
                # Pin the validation to the stack length the executor
                # actually re-bench'd against. Falling back to the
                # current stack length is dangerous: if a new KEEP
                # sneaks in between the executor reading state.json and
                # the Coordinator processing the result, the fallback
                # would silently mark the new KEEP as validated. The
                # executor surfaces validated_stack_len explicitly to
                # avoid this race; if missing (defensive fallback for
                # tests), we use the current length.
                if isinstance(stack_len_at_run, int) and stack_len_at_run >= 0:
                    self.shared_state.cumulative_gain_validated_stack_len = (
                        stack_len_at_run
                    )
                else:
                    self.shared_state.cumulative_gain_validated_stack_len = (
                        len(self.shared_state.optimization_stack)
                    )
                changed = True
                log.info(
                    "validate_stack promoted: validated_gain=%.2f%% "
                    "(tput=%.2f vs baseline=%.2f) at stack_len=%d",
                    gain, float(tput), self.shared_state.baseline_tput,
                    self.shared_state.cumulative_gain_validated_stack_len,
                )
        elif task_kind in ("backends", "params", "sweep"):
            # T1/T2 — persist discovered_flags + synergy_attempted +
            # winners_history so the next Orchestration tick (and IR-26
            # idea-generation prompt section) sees the full search-space
            # context. These are independent of the promotion path below
            # so they always run, even when no winner crossed the gate.
            disc_update = result.get("discovered_flags_update")
            if isinstance(disc_update, dict):
                self.shared_state.record_discovered_flags(
                    framework=str(disc_update.get("framework") or ""),
                    backend_flags=disc_update.get("backend_flags"),
                    param_flags=disc_update.get("param_flags"),
                    source_path=str(disc_update.get("source_path") or ""),
                )
                changed = True
            new_attempts = result.get("synergy_attempted_new") or []
            for combo in new_attempts:
                if isinstance(combo, list):
                    self.shared_state.mark_synergy_attempted(
                        [str(x) for x in combo],
                    )
                    changed = True
            winners_for_history = result.get("winners") or []
            if winners_for_history and task_kind != "sweep":
                self.shared_state.push_backend_winners_round(
                    action=task_kind,
                    base_tput=float(result.get("base_tput") or 0.0),
                    base_extra_args=str(
                        (task.params or {}).get("base_extra_args", "")
                        if task is not None else ""
                    ),
                    winners=winners_for_history,
                    best=result.get("best_variant"),
                )
                changed = True
            if task_kind == "sweep":
                self.shared_state.record_sweep(result)
                changed = True
                # Sweep maps the current stack across workloads; it is not
                # itself a new serving config, so don't overwrite current_best.
                self.shared_state.params_no_promote_streak += 1
                # Treat sweep as a DISCARD-shaped score update: it records the
                # run + sets a cooldown but does not register a KEEP because
                # sweep itself never promotes a new current_best.
                self._apply_action_score_update(
                    "sweep", result, promoted=False, gain_vs_cb=0.0,
                )
                self.shared_state.save(self.session_dir)
                return
            # Promote a grid-runner winner if it actually beat the
            # current best by a meaningful margin. We use 0.5% as the
            # 1-shot KEEP threshold (relaxed from marathon's original
            # 1.0% per the resume5 9h finding: 35/38 winners landed in
            # the 0.3–0.84% band but never promoted because each
            # individual run sat under 1.0%) AND, as a separate path,
            # promote ANY consistent winner that wins ≥ 2 of last 3
            # rounds with average gain ≥ 0.3% — that's the cross-round
            # signal-vs-noise check.
            PROMOTE_THRESHOLD_PCT = 0.5
            CROSS_ROUND_LOOKBACK = 3
            CROSS_ROUND_MIN_APPEARANCES = 2
            CROSS_ROUND_MIN_AVG_GAIN_PCT = 0.3
            best_tput = result.get("output_throughput")
            bv = result.get("best_variant") or {}
            best_gain = result.get("best_gain_pct")
            if task_kind == "params" and isinstance(result.get("params_search_update"), dict):
                self.shared_state.apply_params_search_update(
                    result["params_search_update"],
                )
                changed = True
            cb = self.shared_state.current_best or {}
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            cur_best = float(cb_tput) if isinstance(cb_tput, (int, float)) and cb_tput > 0 \
                else float(self.shared_state.baseline_tput or 0.0)
            # Compute gain vs current_best (different from result.best_gain_pct
            # which is gain vs base_tput injected at materialize time).
            gain_vs_cb = (
                (best_tput - cur_best) / cur_best * 100.0
                if isinstance(best_tput, (int, float)) and best_tput > 0 and cur_best > 0
                else None
            )
            # Always record this round to the rolling history regardless
            # of whether it promotes — `consistent_winner` consults it.
            if isinstance(bv, dict) and bv.get("name") and gain_vs_cb is not None:
                self.shared_state.push_params_winner(
                    action=task_kind,
                    variant_name=bv.get("name"),
                    tput=best_tput,
                    gain_pct=gain_vs_cb,
                )
            promoted = False
            if gain_vs_cb is not None and gain_vs_cb >= PROMOTE_THRESHOLD_PCT:
                # Accuracy gate: if the winner touches precision-affecting
                # flags, verify its GSM8K accuracy didn't drop > 5%.
                # The eval ran during the benchmark (RUN_EVAL=true) so we
                # check the result that came back.
                from .action_executors._accuracy_gate import (
                    accuracy_passed,
                    is_high_accuracy_risk,
                )
                winner_args = str(bv.get("extra_sglang_args") or "")
                winner_envs = dict(bv.get("extra_envs") or {})
                accuracy_ok = True
                if is_high_accuracy_risk(winner_args, winner_envs):
                    new_acc = result.get("accuracy")
                    base_acc = self.shared_state.baseline_accuracy
                    if isinstance(new_acc, (int, float)) and base_acc > 0:
                        accuracy_ok = accuracy_passed(base_acc, new_acc)
                        if not accuracy_ok:
                            log.warning(
                                "accuracy gate FAILED for %s variant=%s: "
                                "baseline=%.4f new=%.4f (drop=%.4f > 0.05)",
                                task_kind, bv.get("name"),
                                base_acc, new_acc, base_acc - new_acc,
                            )
                    elif base_acc <= 0:
                        log.info(
                            "accuracy gate skipped (no baseline_accuracy yet) "
                            "for high-risk variant=%s", bv.get("name"),
                        )
                if accuracy_ok:
                    self._lift_to_current_best(task_kind, best_tput, bv)
                    promoted = True
                else:
                    log.info("accuracy gate blocked promotion of %s/%s",
                             task_kind, bv.get("name"))
            else:
                # Cross-round signal: same variant winning consistently
                # at sub-threshold but real gains.
                consistent = self.shared_state.consistent_winner(
                    lookback=CROSS_ROUND_LOOKBACK,
                    min_appearances=CROSS_ROUND_MIN_APPEARANCES,
                    min_avg_gain_pct=CROSS_ROUND_MIN_AVG_GAIN_PCT,
                )
                if consistent and consistent.get("tput", 0) > cur_best:
                    # Lift the consistent winner — synthesise a best_variant
                    # from the history record (we don't have its full
                    # extra_sglang_args here, so leave that blank; Orch
                    # consults `params_winner_history` if it needs to know
                    # which variant_name is the consistent one).
                    self._lift_to_current_best(
                        consistent["action"], consistent["tput"],
                        {"name": consistent["variant_name"]},
                    )
                    log.info(
                        "promoted consistent winner: variant=%s avg_gain=%.2f%% (%d rounds)",
                        consistent["variant_name"],
                        consistent["gain_pct"],
                        CROSS_ROUND_LOOKBACK,
                    )
                    promoted = True
            if promoted:
                self.shared_state.params_no_promote_streak = 0
                changed = True
            else:
                # Plateau detection: count consecutive grid runs that
                # didn't move current_best. Prompt summary surfaces this
                # so Orch knows when to switch to kernel-opt.
                self.shared_state.params_no_promote_streak += 1
                changed = True  # streak counter changed → save state.json
            # Mirror the promoted decision into per-action scoring so
            # cooldown + diminishing-returns decay (or DISCARD dampening)
            # surface in the prompt scoreboard. Always runs for backends /
            # params so an LLM stuck in the explore loop sees the row
            # decay deterministically.
            self._apply_action_score_update(
                task_kind, result,
                promoted=bool(promoted),
                gain_vs_cb=(
                    float(gain_vs_cb) if isinstance(gain_vs_cb, (int, float)) else 0.0
                ),
            )
            changed = True
        # Out-of-band score updates for the task_kinds that don't have a
        # promoted-vs-discard notion (profile / pmc_roofline / validate_stack
        # bump runs + cooldown; baseline is treated as a gate and skipped
        # inside the helper). Sweep + backends/params/kernel branches above
        # already called the helper themselves, so we filter to the
        # measurement-style kinds here.
        if task_kind in {"profile", "pmc_roofline", "validate_stack"}:
            self._apply_action_score_update(task_kind, result)
            changed = True
        if changed:
            self.shared_state.save(self.session_dir)


__all__ = ["Coordinator", "CoordinatorState", "PendingProposal", "SharedState"]
