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
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..paths import db_path_for, make_session_dir
from ..storage.connection import SqliteConnection
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
        self.policy = PolicyGate(role_registry=self.role_registry)
        self.sub = sub_agent_runner or SubAgentRunner(self.locks, self.tasks)

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

        # Resume: rebuild CoordinatorState.pending_proposals from the SQLite
        # event log so a Coordinator restart picks up undecided proposals
        # without losing the Critic Review queue (DESIGN §17.5).
        self._resumed_from = self._detect_resume_state()

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

        # 1. Shared session state — gives the agent goal + progress context
        # even on tick 1 when the inbox is empty.
        sections.append("=== Shared session state ===")
        sections.append(self.shared_state.to_prompt_summary())
        if agent_name == "orchestration":
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
        """
        if self.shared_state.stop_reason:
            return ""
        if self.shared_state.baseline_tput <= 0:
            return (
                "TODO 1/3: baseline is required now. Propose/delegate only "
                "`baseline` until baseline_tput > 0."
            )
        if not self.shared_state.last_profile_trace:
            return (
                "TODO 2/3: profile is required now. Baseline exists but "
                "last_profile_trace is empty; propose/delegate only `profile`. "
                "Do not run backends/params/sweep yet."
            )
        select = self.shared_state.last_select_kernels or {}
        if select.get("trace_input") != self.shared_state.last_profile_trace:
            return (
                "TODO 3/3: select_kernels is required now. Emit "
                "request{target_agent='kernel', kind='select_kernels', "
                "params={trace_input: last_profile_trace, top_k: 10}} before "
                "backends/params/sweep."
            )
        return ""

    def _sequence_denial_for_action(self, action_name: str) -> PolicyDenied | None:
        """Reject orchestration action/delegate attempts that skip required steps."""
        action = str(action_name or "").strip()
        sequence_actions = {"baseline", "profile", "backends", "params", "sweep", "report"}
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
        if self.shared_state.baseline_tput > 0 and not self.shared_state.last_profile_trace:
            if action != "profile":
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
                    self.shared_state.save(self.session_dir)
                # Mirror kernel-opt outcomes into SharedState so Orch
                # sees decision/speedup in its prompt next tick and
                # doesn't re-dispatch the same kernel_id forever.
                if kind == "run_optimization":
                    self.shared_state.record_kernel_opt(result)
                    self.shared_state.save(self.session_dir)
                if kind == "integrate":
                    if result.get("status") != "skipped":
                        self.shared_state.record_kernel_integrate_result(result)
                    if result.get("decision") == "KEEP":
                        self._record_integrate_keep(result)
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
            # executor didn't throw. The executor itself may report
            # status="failed" (e.g. Magpie benchmark report success=false
            # or 0 completed requests). Only promote truly successful runs.
            executor_status = (result.result or {}).get("status", "")
            if result.state == "succeeded" and executor_status != "failed":
                await self._promote_to_shared_state(
                    task.kind, result.result, task=task,
                )

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
        elif task_kind in ("backends", "params", "sweep"):
            if task_kind == "sweep":
                self.shared_state.record_sweep(result)
                changed = True
                # Sweep maps the current stack across workloads; it is not
                # itself a new serving config, so don't overwrite current_best.
                self.shared_state.params_no_promote_streak += 1
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
        if changed:
            self.shared_state.save(self.session_dir)


__all__ = ["Coordinator", "CoordinatorState", "PendingProposal", "SharedState"]
