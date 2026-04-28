"""Conductor — DESIGN §15.

Single owner of the run. Wires every subsystem (storage, locks, bus,
cursors, kb, scheduler, policy, sub_agent_runner, ...) and runs the asyncio
gather of reactors + clock + stopping_watcher.

STATUS (v0.6 — PolicyGate + multi-reactor):
    - ``_bootstrap`` wires storage / bus / cursors / tasks / locks + objective +
      shared state + role-registry + PolicyGate. SubAgentRunner / ActionRegistry
      / Scheduler are still ``None`` in dry-run.
    - One reactor task is spawned per role returned by ``roles_for_mode``:
        quick     -> [executor]
        guided    -> [executor, critic]
        marathon  -> [executor, critic, watchdog, sage]
    - Every parsed intent runs through :meth:`PolicyGate.validate_intent`
      before ``_handle_intent``. Denied intents are logged on the bus as a
      ``policy_denied`` observation; the reactor keeps going.
    - The clock fires elapsed-time updates and reflection ticks. Once
      ``elapsed_minutes >= max_minutes`` it triggers a graceful
      ``time_exhausted`` stop.

What is intentionally **NOT** here yet (see IMPLEMENTATION-CHECKLIST):
    - Action proposal + sub-agent dispatch (Phase 7.10 / F3)
    - Accuracy gate / scheduler updates (Phase 5 / 9)
    - Marathon cadences: persona distill, strategic review, KB synthesis
    - Resume from checkpoint
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..storage.connection import SqliteConnection
from .agent_role import AgentRole, default_role_registry, roles_for_mode
from .backends import Backend, MockBackend
from .checkpoint import (
    CheckpointHandle,
    ResumeState,
    TriggerReason,
    Checkpoint,
    resume_from_session_dir,
)
from .cursor_store import CursorStore
from .execution_mode import ExecutionMode, choose_execution_mode
from .feature_flags import FeatureFlags, build_feature_flags
from .intent_parser import Intent, IntentType
from .iron_rules import render_for_prompt as render_iron_rules
from .message_bus import Message, MessageBus, TOPIC_ALLOWLIST
from .objective import Objective, build_objective
from .policy import PolicyDenied, PolicyGate
from .resource_lock import ResourceLockManager, SqliteLeaseBackend
from .shared_state import SharedState
from .sub_agent_runner import SubAgentRunner, dispatch_pending_delegates
from .task_registry import TaskRegistry

if TYPE_CHECKING:  # pragma: no cover - type-only
    from .action_registry import ActionRegistry
    from .kb import KnowledgeBase
    from .sage_query_service import SageQueryService
    from .scheduler import BudgetAwareScheduler


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-topic prompt-render table (consumed by ``Conductor._render_msg_body``).
#
# Keys mirror the dispatcher publish sites in ``_handle_*`` below — when a
# new ``_handle_<intent>`` lands, add the matching entry here so the inbox
# view shows the meaningful field instead of an empty body. The fallback
# branch in ``_render_msg_body`` keeps unknown shapes from rendering as
# ``""`` (which is what triggered Critic / Watchdog "empty body" alerts in
# v0.7).
# ---------------------------------------------------------------------------
def _render_proposal(p: dict) -> str:
    if p.get("kind") == "delegate_queued":
        return (
            f"delegate_queued action={p.get('action_name', '?')} "
            f"task={str(p.get('task_id', '?'))[:8]} "
            f"params={p.get('params', {})}"
        )
    return (
        f"action={p.get('action_name', '?')} "
        f"predicted_gain={p.get('predicted_gain_pct', '?')}% "
        f"params={p.get('params', {})}"
    )


def _render_decision(p: dict) -> str:
    head = p.get("kind") or "decision"
    rationale = p.get("rationale", "")
    s = f"{head} changes={p.get('changes', {})}"
    if rationale:
        s += f" rationale={rationale}"
    return s


def _render_alert(p: dict) -> str:
    sev = p.get("severity", "?")
    summary = p.get("summary", "")
    detail = p.get("detail", "")
    out = f"[{sev}] {summary}"
    if detail:
        out += f" — {detail}"
    return out


def _render_question(p: dict) -> str:
    q_topic = p.get("topic", "")
    q = p.get("question", "")
    return f"[topic={q_topic}] {q}" if q_topic else q


def _render_answer(p: dict) -> str:
    q_topic = p.get("topic", "")
    a = p.get("answer", "")
    return f"[topic={q_topic}] {a}" if q_topic else a


def _render_objection(p: dict) -> str:
    target = str(p.get("target_msg_id", ""))[:8]
    text = p.get("objection") or p.get("body_md") or ""
    return f"target={target} {text}" if target else text


def _render_vote(p: dict) -> str:
    target = str(p.get("target_msg_id", ""))[:8]
    return f"target={target} vote={p.get('vote', '?')}"


def _render_reflection_tick(p: dict) -> str:
    elapsed = p.get("elapsed_minutes")
    left = p.get("time_left_minutes")
    if isinstance(elapsed, (int, float)) and isinstance(left, (int, float)):
        return f"elapsed={elapsed:.2f}m time_left={left:.2f}m"
    return ""


_TOPIC_RENDERERS = {
    "proposal": _render_proposal,
    "decision": _render_decision,
    "alert": _render_alert,
    "question": _render_question,
    "answer": _render_answer,
    "objection": _render_objection,
    "vote": _render_vote,
    "reflection_tick": _render_reflection_tick,
}


# ---------------------------------------------------------------------------
class StopReason:
    TARGET_REACHED = "target_reached"
    TIME_EXHAUSTED = "time_exhausted"
    NO_MORE_LEVERAGE = "no_more_leverage"
    BRIER_PLATEAU = "brier_plateau"
    EMERGENCY = "emergency"


@dataclass
class ConductorContext:
    """Bundle of components passed to reactors and helpers."""

    state: SharedState
    objective: Objective
    flags: FeatureFlags
    bus: MessageBus
    cursors: CursorStore
    locks: ResourceLockManager
    tasks: TaskRegistry
    db: SqliteConnection
    role_registry: dict[str, AgentRole] = field(default_factory=dict)
    roles: list[AgentRole] = field(default_factory=list)
    policy: PolicyGate | None = None
    actions: "ActionRegistry | None" = None
    scheduler: "BudgetAwareScheduler | None" = None
    sub_agent_runner: SubAgentRunner | None = None
    kb: "KnowledgeBase | None" = None
    sage: "SageQueryService | None" = None
    persona_index: dict[str, str] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
class Conductor:
    """Single source of truth for the run.

    Construction:
        ``Conductor(session_dir, backend=..., env=..., db=...)``

    The ``backend`` injection is what makes dry-runs (MockBackend) and real
    runs (ClaudeBackend / CodexBackend) interchangeable.
    """

    DEFAULT_REACTOR_TICK_S = 2.0
    DEFAULT_CLOCK_TICK_S = 5.0

    def __init__(
        self,
        session_dir: Path,
        *,
        backend: Backend | None = None,
        env: dict[str, str] | None = None,
        db: SqliteConnection | None = None,
        role_registry: dict[str, AgentRole] | None = None,
        action_registry: "ActionRegistry | None" = None,
        kb: "KnowledgeBase | None" = None,
        sage: "SageQueryService | None" = None,
        reactor_tick_s: float | None = None,
        clock_tick_s: float | None = None,
        enable_dispatcher: bool = True,
        enable_checkpointing: bool = False,
    ) -> None:
        self.session_dir = Path(session_dir)
        self.env = dict(env) if env is not None else dict(os.environ)
        self.backend: Backend = backend or MockBackend()
        self._stop_event = asyncio.Event()
        self.ctx: ConductorContext | None = None
        self._owns_db = db is None
        self._db = db
        self._role_registry_override = role_registry
        self._action_registry = action_registry
        self._kb_override = kb
        self._sage_override = sage
        self._reactor_tick_s = reactor_tick_s or self.DEFAULT_REACTOR_TICK_S
        self._clock_tick_s = clock_tick_s or self.DEFAULT_CLOCK_TICK_S
        self._enable_dispatcher = enable_dispatcher
        self._enable_checkpointing = enable_checkpointing
        self._dispatcher_stop = asyncio.Event()
        self._last_checkpoint_ts: float = 0.0

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------
    async def _bootstrap(self) -> ConductorContext:
        mode = choose_execution_mode(self.env)
        flags = build_feature_flags(mode)
        objective = build_objective(self.env)

        if self._db is None:
            from ..paths import db_path_for
            self._db = SqliteConnection(db_path_for(self.session_dir))

        bus = MessageBus(self._db)
        cursors = CursorStore(self._db)
        tasks = TaskRegistry(self._db)
        locks = ResourceLockManager(SqliteLeaseBackend(self._db))

        # Roles + PolicyGate (DESIGN §5.1 / §10.5.7).
        role_registry = (
            self._role_registry_override
            if self._role_registry_override is not None
            else default_role_registry()
        )
        roles = roles_for_mode(mode, registry=role_registry)
        policy = PolicyGate(
            flags=flags,
            mode=mode,
            role_registry=role_registry,
            action_registry=self._action_registry,
        )

        max_minutes = float(self.env["MAX_HOURS"]) * 60.0
        model_path = self.env.get("MODEL_PATH", "")

        state = SharedState(
            session_id=self.session_dir.name,
            model_path=model_path,
            model_name=Path(model_path).name if model_path else "",
            model_class="unknown",  # real classify_model lands with §9.2
            cwd=str(self.session_dir),
            start_ts=time.time(),
            max_minutes=max_minutes,
            execution_mode=mode,
        )

        # Optional SubAgentRunner — only meaningful when an ActionRegistry
        # is wired in. Without an action registry there are no metadata
        # entries to look up so the runner cannot dispatch anything.
        sub_agent_runner: SubAgentRunner | None = None
        if self._action_registry is not None:
            sub_agent_runner = SubAgentRunner(
                backend=self.backend,
                policy=policy,
                locks=locks,
                action_registry=self._action_registry,
                tasks=tasks,
                workspace=self.session_dir,
                agent_name="executor",
                env=self.env,
                # Route executor-emitted intents through the same
                # PolicyGate + handle_intent pipeline as LLM-emitted
                # intents. This is what makes ``baseline`` /
                # ``bench_runner`` updates land on the events bus +
                # SharedState exactly like a Claude reactor would have
                # done if it had perfect tool use.
                intent_sink=self._executor_intent_sink,
            )

        # Persona index — populate from existing files; reactor prompt
        # composition reads this map.
        persona_index = self._load_persona_index()

        ctx = ConductorContext(
            state=state,
            objective=objective,
            flags=flags,
            bus=bus,
            cursors=cursors,
            locks=locks,
            tasks=tasks,
            db=self._db,
            role_registry=role_registry,
            roles=roles,
            policy=policy,
            actions=self._action_registry,
            sub_agent_runner=sub_agent_runner,
            kb=self._kb_override,
            sage=self._sage_override,
            persona_index=persona_index,
        )
        self.ctx = ctx

        await bus.append_and_seq(
            Message.new(
                from_agent="conductor",
                to_agent="*",
                topic="event",
                payload={
                    "kind": "run_started",
                    "session_id": state.session_id,
                    "mode": mode.value,
                    "objective": objective.describe(),
                    "max_minutes": max_minutes,
                    "model_path": model_path,
                    "roles": [r.name for r in roles],
                },
                priority=0,
            )
        )
        state.write_snapshot(self.session_dir)
        log.info(
            "conductor: bootstrapped session=%s mode=%s roles=%s max_minutes=%.1f",
            state.session_id, mode.value, [r.name for r in roles], max_minutes,
        )
        return ctx

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    async def run(self) -> ConductorContext:
        """Main entry. Returns the populated context after a graceful stop.

        Reactors spawned: one per role from :func:`roles_for_mode` plus the
        clock, the stopping-watcher and (if action_registry was provided) a
        long-running ``dispatch_pending_delegates`` loop that drains the
        queued ``delegate`` tasks via :class:`SubAgentRunner`.
        """
        ctx = await self._bootstrap()
        try:
            reactor_tasks = [
                asyncio.create_task(
                    self._reactor(role.name), name=f"reactor-{role.name}"
                )
                for role in ctx.roles
            ]
            tasks: list[asyncio.Task] = [
                *reactor_tasks,
                asyncio.create_task(self._clock(), name="clock"),
                asyncio.create_task(
                    self._stopping_watcher(), name="stopping-watcher"
                ),
            ]
            if (
                self._enable_dispatcher
                and ctx.sub_agent_runner is not None
            ):
                tasks.append(
                    asyncio.create_task(
                        self._dispatcher_loop(),
                        name="delegate-dispatcher",
                    )
                )
            await self._stop_event.wait()
            self._dispatcher_stop.set()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._graceful_stop(ctx.state.stop_reason or StopReason.EMERGENCY)
        finally:
            if self._owns_db and self._db is not None:
                self._db.close()
        return ctx

    # ------------------------------------------------------------------
    # Per-agent reactor
    # ------------------------------------------------------------------
    async def _reactor(self, agent_name: str) -> None:
        assert self.ctx is not None
        ctx = self.ctx
        role = ctx.role_registry.get(agent_name)
        allowed_tools = (
            tuple(ctx.policy.allowed_tools_for_agent(agent_name))
            if ctx.policy is not None
            else ("emit_intent",)
        )
        try:
            while not ctx.state.should_stop():
                cursor = await ctx.cursors.load(agent_name)
                msgs = await ctx.bus.replay_for(
                    agent_name, after_seq=cursor.last_processed_seq
                )
                # Drop self-emitted messages (the bus broadcasts ``to=*`` so a
                # reactor would otherwise consume its own output).
                to_process = [m for m in msgs if m.from_agent != agent_name]
                if not msgs:
                    await asyncio.sleep(self._reactor_tick_s)
                    continue

                if to_process:
                    prompt = self._compose_prompt(agent_name, to_process)
                    try:
                        intents = await self.backend.run(
                            prompt,
                            agent_name=agent_name,
                            allowed_tools=allowed_tools,
                            extra={"role": role.name if role else agent_name},
                        )
                    except Exception as exc:  # noqa: BLE001 — log + continue
                        log.exception(
                            "reactor %s: backend failed: %s", agent_name, exc
                        )
                        await self._record_observation(
                            from_agent=agent_name,
                            kind="backend_error",
                            detail={"error": repr(exc)},
                        )
                        intents = []

                    for intent in intents:
                        if not await self._gate_intent(agent_name, intent):
                            continue
                        await self._handle_intent(agent_name, intent)

                last_msg = msgs[-1]
                assert last_msg.seq is not None
                await ctx.cursors.advance(
                    agent_name,
                    seq=last_msg.seq,
                    msg_id=last_msg.msg_id,
                )
        except asyncio.CancelledError:
            log.debug("reactor %s cancelled", agent_name)
            raise

    async def _gate_intent(
        self, from_agent: str, intent: Intent
    ) -> bool:
        """Run :class:`PolicyGate` on an intent. On denial, log a structured
        observation and return ``False`` so the caller skips dispatch."""
        assert self.ctx is not None
        ctx = self.ctx
        if ctx.policy is None:
            return True
        try:
            ctx.policy.validate_intent(from_agent, intent, ctx.state)
        except PolicyDenied as exc:
            await self._record_observation(
                from_agent="conductor",
                kind="policy_denied",
                detail={
                    "agent": from_agent,
                    "intent_type": intent.type.value,
                    "rule": exc.rule or "unknown",
                    "reason": str(exc),
                },
            )
            log.info(
                "policy denied: agent=%s intent=%s rule=%s reason=%s",
                from_agent, intent.type.value, exc.rule, exc,
            )
            return False
        return True

    async def _handle_intent(self, from_agent: str, intent: Intent) -> None:
        """Dispatch on ``intent.type`` and apply the side-effect.

        Dispatch table (DESIGN §10.5.6 + §15):

            send_message    -> bus.append(topic from payload)
            alert           -> bus.append(topic="alert") + findings/alerts.jsonl
            propose_action  -> tasks.create(kind="proposal") + topic="proposal"
            delegate        -> tasks.create(kind="delegate")  + topic="proposal"
                               (queued; SubAgentRunner picks it up — Phase F3)
            update_state    -> state.apply_validated_transition + topic="decision"
            update_persona  -> personas/<agent>.md append + topic="event"
            ask_question    -> bus.append(topic="question")
            answer          -> bus.append(topic="answer")
            objection       -> bus.append(topic="objection")
            vote            -> bus.append(topic="vote")

        Every branch is idempotent at the SQLite layer (events are append-only,
        tasks use ``idempotency_key``).
        """
        assert self.ctx is not None
        ctx = self.ctx
        payload = dict(intent.payload or {})

        if intent.type == IntentType.SEND_MESSAGE:
            await self._handle_send_message(from_agent, payload)
        elif intent.type == IntentType.ALERT:
            await self._handle_alert(from_agent, payload)
        elif intent.type == IntentType.PROPOSE_ACTION:
            await self._handle_propose_action(from_agent, payload)
        elif intent.type == IntentType.DELEGATE:
            await self._handle_delegate(from_agent, payload)
        elif intent.type == IntentType.UPDATE_STATE:
            await self._handle_update_state(from_agent, payload)
        elif intent.type == IntentType.UPDATE_PERSONA:
            await self._handle_update_persona(from_agent, payload)
        elif intent.type == IntentType.ASK_QUESTION:
            await self._handle_simple_topic(
                from_agent, payload, topic="question",
            )
        elif intent.type == IntentType.ANSWER:
            await self._handle_simple_topic(
                from_agent, payload, topic="answer",
            )
        elif intent.type == IntentType.OBJECTION:
            await self._handle_simple_topic(
                from_agent, payload, topic="objection",
            )
        elif intent.type == IntentType.VOTE:
            await self._handle_simple_topic(
                from_agent, payload, topic="vote",
            )
        else:  # pragma: no cover — defensive
            await self._record_observation(
                from_agent=from_agent,
                kind="intent_unhandled",
                detail={"type": intent.type.value, "payload": payload},
            )

    # ------------------------------------------------------------------
    # _handle_intent helpers
    # ------------------------------------------------------------------
    async def _handle_send_message(
        self, from_agent: str, payload: dict[str, Any]
    ) -> None:
        assert self.ctx is not None
        ctx = self.ctx
        topic = str(payload.get("topic", "heartbeat"))
        if topic not in TOPIC_ALLOWLIST:
            topic = "observation"
        to_agent = str(payload.get("to", "*"))
        body = payload.get("body_md", "")
        priority = int(payload.get("priority", 1))
        extras = {
            k: v for k, v in payload.items()
            if k not in {"to", "topic", "body_md", "priority"}
        }
        await ctx.bus.append_and_seq(
            Message.new(
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                payload={"body_md": body, **extras},
                priority=priority,
            )
        )

    async def _handle_alert(
        self, from_agent: str, payload: dict[str, Any]
    ) -> None:
        """Alert -> bus event + findings/alerts.jsonl mirror."""
        assert self.ctx is not None
        ctx = self.ctx
        severity = str(payload.get("severity", "medium"))
        summary = str(payload.get("summary", ""))
        detail = payload.get("detail", "")
        priority_map = {"critical": 0, "high": 0, "medium": 1, "low": 2}
        priority = priority_map.get(severity, 1)

        await ctx.bus.append_and_seq(
            Message.new(
                from_agent=from_agent,
                to_agent="*",
                topic="alert",
                payload={
                    "severity": severity,
                    "summary": summary,
                    "detail": detail,
                },
                priority=priority,
            )
        )
        # Mirror to disk so the post-mortem report can grep them.
        try:
            self._append_finding(
                "alerts.jsonl",
                {
                    "from": from_agent,
                    "severity": severity,
                    "summary": summary,
                    "detail": detail,
                    "ts": time.time(),
                    "session_id": ctx.state.session_id,
                },
            )
        except OSError:  # pragma: no cover — best-effort
            log.exception("failed to mirror alert to findings/alerts.jsonl")

    async def _handle_propose_action(
        self, from_agent: str, payload: dict[str, Any]
    ) -> None:
        """propose_action -> queued task with kind='proposal' + bus event."""
        assert self.ctx is not None
        ctx = self.ctx
        action_name = str(payload.get("action_name", ""))
        predicted_gain_pct = payload.get("predicted_gain_pct")
        params = dict(payload.get("params", {}) or {})

        idem = self._task_idempotency_key(
            kind="proposal",
            from_agent=from_agent,
            action_name=action_name,
            params=params,
        )
        task = await ctx.tasks.create(
            kind="proposal",
            params={
                "action_name": action_name,
                "predicted_gain_pct": predicted_gain_pct,
                "params": params,
                "requested_by": from_agent,
                **{
                    k: v for k, v in payload.items()
                    if k not in {"action_name", "predicted_gain_pct", "params"}
                },
            },
            idempotency_key=idem,
        )
        await ctx.bus.append_and_seq(
            Message.new(
                from_agent=from_agent,
                to_agent="*",
                topic="proposal",
                payload={
                    "task_id": task.task_id,
                    "action_name": action_name,
                    "predicted_gain_pct": predicted_gain_pct,
                    "params": params,
                },
                priority=1,
            )
        )

    async def _handle_delegate(
        self, from_agent: str, payload: dict[str, Any]
    ) -> None:
        """delegate -> queued task with kind='delegate' + bus event.

        SubAgentRunner (Phase F3) picks up queued delegate tasks and spawns
        an OOB Claude sub-agent. In v0.6 the task simply sits in the queue.
        """
        assert self.ctx is not None
        ctx = self.ctx
        action_name = str(payload.get("action_name", ""))
        params = dict(payload.get("params", {}) or {})
        idem = self._task_idempotency_key(
            kind="delegate",
            from_agent=from_agent,
            action_name=action_name,
            params=params,
        )
        # Pull declared side-effects / tools from action_registry if available.
        requires_lanes: list[str] = []
        allowed_tools: list[str] = []
        side_effects: list[str] = []
        lease_ttl_sec: int = 0
        if ctx.actions is not None:
            action = ctx.actions.get(action_name)
            if action is not None:
                requires_lanes = list(getattr(action, "requires_lanes", []) or [])
                allowed_tools = list(getattr(action, "allowed_tools", []) or [])
                side_effects = list(getattr(action, "side_effects", []) or [])
                lease_ttl_sec = int(getattr(action, "lease_ttl_sec", 0) or 0)

        task = await ctx.tasks.create(
            kind="delegate",
            params={
                "action_name": action_name,
                "params": params,
                "requested_by": from_agent,
            },
            idempotency_key=idem,
            requires_lanes=requires_lanes,
            allowed_tools=allowed_tools,
            side_effects=side_effects,
            lease_ttl_sec=lease_ttl_sec,
        )
        await ctx.bus.append_and_seq(
            Message.new(
                from_agent=from_agent,
                to_agent="conductor",
                topic="proposal",
                payload={
                    "kind": "delegate_queued",
                    "task_id": task.task_id,
                    "action_name": action_name,
                    "params": params,
                },
                priority=1,
            )
        )

    async def _handle_update_state(
        self, from_agent: str, payload: dict[str, Any]
    ) -> None:
        """update_state -> apply transition + emit decision event.

        PolicyGate has already verified the agent can write these fields, so
        we apply the changes directly. SharedState.apply_validated_transition
        silently ignores non-allowlisted keys (defence-in-depth).

        After the transition lands, we re-derive ``cumulative_gain`` from
        the resulting (baseline_tput, current_tput) pair so the objective
        + early-stop signals see the up-to-date number on the next tick
        (DESIGN §6.3 / §8 / §7.1 #1).
        """
        assert self.ctx is not None
        ctx = self.ctx
        changes = dict(payload.get("changes", {}) or {})
        ctx.state.apply_validated_transition(from_agent, changes)
        # Auto-derive cumulative_gain so executors / LLM only need to
        # report tput numbers; the Conductor owns the gain math (which
        # also keeps it on the CORE_STATE_FIELDS PolicyGate allowlist).
        self._maybe_recompute_gain(from_agent)
        await ctx.bus.append_and_seq(
            Message.new(
                from_agent=from_agent,
                to_agent="*",
                topic="decision",
                payload={
                    "kind": "state_updated",
                    "changes": changes,
                    "rationale": payload.get("rationale", ""),
                    "derived": {
                        "cumulative_gain": ctx.state.cumulative_gain,
                    },
                },
                priority=1,
            )
        )

    async def _handle_update_persona(
        self, from_agent: str, payload: dict[str, Any]
    ) -> None:
        """update_persona -> append-only L3 persona file + bus event."""
        assert self.ctx is not None
        ctx = self.ctx
        body_md = str(payload.get("body_md", ""))
        if not body_md.strip():
            return
        persona_dir = self.session_dir / "personas"
        persona_dir.mkdir(parents=True, exist_ok=True)
        persona_path = persona_dir / f"{from_agent}.md"
        try:
            with persona_path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n\n<!-- ts={time.time()} -->\n{body_md}\n")
        except OSError:  # pragma: no cover — best-effort
            log.exception("failed to append persona for %s", from_agent)
            return
        # Refresh in-memory persona index so the next prompt sees it.
        self._refresh_persona_index(from_agent)
        await ctx.bus.append_and_seq(
            Message.new(
                from_agent=from_agent,
                to_agent="conductor",
                topic="event",
                payload={
                    "kind": "persona_update",
                    "agent": from_agent,
                    "chars": len(body_md),
                },
                priority=2,
            )
        )

    def _maybe_recompute_gain(self, from_agent: str) -> None:
        """Derive ``cumulative_gain`` from the current
        (baseline_tput, current_tput) pair.

        Convention: gain is the percentage *throughput improvement* over
        the baseline measured on the same model + framework. We accept
        any value on the real line (executors can write a regression
        and the early-stop / objective layers still work). Skips when
        baseline is missing or zero (would divide-by-zero).
        """
        if self.ctx is None:
            return
        state = self.ctx.state
        try:
            base = float(state.baseline_tput or 0)
            cur = float(state.current_tput or 0)
        except (TypeError, ValueError):
            return
        if base <= 0:
            # Initial baseline call also writes ``current_tput=baseline``;
            # gain is exactly 0 in that case (handled below).
            if cur > 0 and base == 0:
                state.cumulative_gain = 0.0
            return
        new_gain = (cur - base) / base * 100.0
        # Apply via the same transition path so a derived field never
        # sneaks past the allowed-list (cumulative_gain *is* on it).
        state.apply_validated_transition(
            "conductor", {"cumulative_gain": new_gain},
        )

    async def _executor_intent_sink(
        self, from_agent: str, intent: Intent
    ) -> None:
        """Receive an intent emitted by an :class:`ActionExecutor` and
        push it through the same handler pipeline as LLM intents — but
        **bypassing PolicyGate**.

        Rationale: Python ``ActionExecutor`` instances are trusted code
        (their inputs come from `subprocess` invocations of the bundled
        scripts, not from a free-text LLM). Their intents represent
        measurement facts (``baseline_tput``, ``current_tput``,
        ``cumulative_gain``) that PolicyGate's CORE_STATE_FIELDS guard
        was designed to block *agent reactors* (LLMs) from setting
        directly. Re-using the same channel for trusted code requires
        the bypass — otherwise legitimate baseline measurements get
        denied.

        We re-attribute the from_agent to ``"conductor"`` on the bus so
        downstream consumers can distinguish trusted measurement
        events from LLM proposals. The handler still records the full
        ``rationale`` so audit logs trace back to the originating
        executor.
        """
        await self._handle_intent("conductor", intent)

    async def _handle_simple_topic(
        self,
        from_agent: str,
        payload: dict[str, Any],
        *,
        topic: str,
    ) -> None:
        """Generic dispatcher for ASK_QUESTION / ANSWER / OBJECTION / VOTE.

        These intents are pure messages — they have no side-effects beyond
        appending one event to the bus. The payload is forwarded as-is.
        """
        assert self.ctx is not None
        ctx = self.ctx
        if topic not in TOPIC_ALLOWLIST:  # pragma: no cover — defensive
            topic = "observation"
        await ctx.bus.append_and_seq(
            Message.new(
                from_agent=from_agent,
                to_agent=str(payload.get("to", "*")),
                topic=topic,
                payload=payload,
                priority=int(payload.get("priority", 1)),
                in_reply_to=payload.get("in_reply_to"),
            )
        )

    # ------------------------------------------------------------------
    # Prompt rendering helpers
    # ------------------------------------------------------------------
    # Per-message body cap when the body is *synthesised* from structured
    # payload fields (proposal/decision/observation/...). Free-text
    # ``body_md`` from heartbeats / send_message is left intact because it
    # is the agent's own composed content.
    _MSG_BODY_RENDER_CAP = 360

    @staticmethod
    def _render_msg_body(topic: str, payload: Any) -> str:
        """Render a single bus event into one line for the inbox view.

        DESIGN §10.5.6 — every published payload shape varies per topic
        (``send_message`` carries ``body_md``; ``propose_action`` carries
        ``action_name``/``predicted_gain_pct``/``params``; ``observation``
        carries ``kind``/``reason``; ``reflection_tick`` carries
        ``elapsed_minutes``; etc.). The pre-fix renderer only looked at
        ``body_md`` and ``kind`` keys, so most non-text events showed up
        as empty bodies in the prompt — and Critic/Watchdog correctly
        flagged this as "I cannot evaluate" alerts (see findings/alerts).

        Contract:

        - When ``payload['body_md']`` is non-empty text, return it verbatim
          (no truncation — agent-authored content is intentional).
        - Otherwise, dispatch on ``topic`` to a known shape renderer.
        - Fall back to a compact ``kind=... key=val ...`` summary, then a
          ``repr(payload)`` truncated to :attr:`_MSG_BODY_RENDER_CAP`.

        ``payload`` is duck-typed; non-dict shapes are coerced via ``str``
        (defensive — should never happen since the bus enforces dicts).
        """
        if not isinstance(payload, dict):
            return str(payload or "")

        body_md = payload.get("body_md")
        if isinstance(body_md, str) and body_md.strip():
            return body_md

        renderer = _TOPIC_RENDERERS.get(topic)
        if renderer is not None:
            rendered = renderer(payload)
            if rendered:
                return Conductor._truncate(rendered, Conductor._MSG_BODY_RENDER_CAP)

        # Generic fallback: prefer ``kind`` if present, then a compact
        # ``key=val`` summary of the remaining keys.
        kind = payload.get("kind")
        rest = {k: v for k, v in payload.items() if k != "kind"}
        if isinstance(kind, str) and kind:
            summary = (
                f"{kind} {Conductor._kvs(rest)}"
                if rest
                else kind
            )
        else:
            summary = Conductor._kvs(payload) or repr(payload)
        return Conductor._truncate(summary, Conductor._MSG_BODY_RENDER_CAP)

    @staticmethod
    def _kvs(d: dict[str, Any]) -> str:
        """Compact ``k=v`` formatter — preserves field meaning without
        spending characters on JSON quoting."""
        parts = []
        for k, v in d.items():
            if isinstance(v, str):
                parts.append(f"{k}={v}")
            elif isinstance(v, (int, float, bool)) or v is None:
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={v!r}")
        return " ".join(parts)

    @staticmethod
    def _truncate(s: str, cap: int) -> str:
        return s if len(s) <= cap else s[: cap - 3] + "..."

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _task_idempotency_key(
        *,
        kind: str,
        from_agent: str,
        action_name: str,
        params: dict[str, Any],
    ) -> str:
        """Stable hash so re-emitting an identical proposal de-duplicates."""
        import hashlib
        import json as _json
        body = _json.dumps(
            {
                "kind": kind,
                "from": from_agent,
                "action_name": action_name,
                "params": params,
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return f"{kind}-{hashlib.sha256(body).hexdigest()[:32]}"

    def _append_finding(
        self, filename: str, record: dict[str, Any]
    ) -> None:
        import json as _json
        findings = self.session_dir / "findings"
        findings.mkdir(parents=True, exist_ok=True)
        path = findings / filename
        with path.open("a", encoding="utf-8") as fh:
            fh.write(_json.dumps(record, default=str) + "\n")

    async def _record_observation(
        self, *, from_agent: str, kind: str, detail: dict[str, Any]
    ) -> None:
        assert self.ctx is not None
        await self.ctx.bus.append_and_seq(
            Message.new(
                from_agent=from_agent,
                to_agent="conductor",
                topic="observation",
                payload={"kind": kind, **detail},
            )
        )

    # ------------------------------------------------------------------
    # Clock + stopping watcher
    # ------------------------------------------------------------------
    async def _clock(self) -> None:
        assert self.ctx is not None
        ctx = self.ctx
        try:
            while not ctx.state.should_stop():
                ctx.state.refresh_elapsed()
                if ctx.state.elapsed_minutes >= ctx.state.max_minutes > 0:
                    ctx.state.set_stopping(StopReason.TIME_EXHAUSTED)
                    break
                await ctx.bus.append_and_seq(
                    Message.new(
                        from_agent="clock",
                        to_agent="executor",
                        topic="reflection_tick",
                        payload={
                            "elapsed_minutes": ctx.state.elapsed_minutes,
                            "time_left_minutes": ctx.state.time_left_minutes,
                        },
                        priority=2,
                    )
                )
                ctx.state.write_snapshot(self.session_dir)
                # 30-min checkpoint cadence (DESIGN §13.7) — only fires when
                # the conductor is configured for checkpointing AND we've
                # actually exceeded the cadence interval.
                await self._maybe_checkpoint()
                await asyncio.sleep(self._clock_tick_s)
        except asyncio.CancelledError:
            log.debug("clock cancelled")
            raise

    async def _maybe_checkpoint(self) -> CheckpointHandle | None:
        if not self._enable_checkpointing or self.ctx is None:
            return None
        # Cadence: 30 min wall-clock OR after a KEEP. We only handle the
        # cadence path here; KEEP-triggered checkpoints come from
        # `_handle_update_state` once we wire it up.
        elapsed_real = time.time() - self._last_checkpoint_ts
        if elapsed_real < 1800.0 and self._last_checkpoint_ts > 0:
            return None
        try:
            handle = await Checkpoint.create(
                self.session_dir, self.ctx.db, self.ctx.state,
                trigger=TriggerReason.PERIODIC,
            )
            self._last_checkpoint_ts = time.time()
            log.info("checkpoint written: %s", handle.path)
            return handle
        except Exception:  # noqa: BLE001
            log.exception("checkpoint cadence failed")
            return None

    async def _stopping_watcher(self) -> None:
        assert self.ctx is not None
        ctx = self.ctx
        try:
            while not self._stop_event.is_set():
                if ctx.state.should_stop():
                    self._stop_event.set()
                    return
                await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            log.debug("stopping_watcher cancelled")
            raise

    async def _graceful_stop(self, reason: str) -> None:
        assert self.ctx is not None
        ctx = self.ctx
        ctx.state.refresh_elapsed()
        ctx.state.set_stopping(reason)
        try:
            await ctx.bus.append_and_seq(
                Message.new(
                    from_agent="conductor",
                    to_agent="*",
                    topic="graceful_stop",
                    payload={
                        "reason": reason,
                        "elapsed_minutes": ctx.state.elapsed_minutes,
                        "max_minutes": ctx.state.max_minutes,
                        "cumulative_gain": ctx.state.cumulative_gain,
                    },
                    priority=0,
                )
            )
        except Exception:  # noqa: BLE001 — best-effort
            log.exception("graceful_stop: failed to append final event")
        ctx.state.write_snapshot(self.session_dir)
        log.info(
            "conductor: stopped reason=%s elapsed=%.2fm",
            reason, ctx.state.elapsed_minutes,
        )

    # ------------------------------------------------------------------
    # Prompt composition
    # ------------------------------------------------------------------
    def _render_action_catalogue(self, mode: ExecutionMode) -> str:
        """Render the per-mode action catalogue as a markdown table.

        Injected into every reactor prompt so the LLM knows the exact
        action names + cost / risk envelopes it can ``propose_action``
        / ``delegate``. Without this the LLM tends to invent
        plausible-but-wrong action names that PolicyGate then rejects.
        """
        if self.ctx is None or self.ctx.actions is None:
            return ""
        actions = self.ctx.actions.allowed_for_mode(mode)
        if not actions:
            return ""
        lines = [
            "## Available actions for this mode",
            "",
            "| name | family | lanes | cost_p75 | acc_risk |",
            "|---|---|---|---:|---:|",
        ]
        for a in sorted(actions, key=lambda x: (x.family, x.name)):
            lanes = ", ".join(a.requires_lanes) or "—"
            lines.append(
                f"| `{a.name}` | {a.family} | {lanes} | "
                f"{a.cost_minutes_p75:.0f}m | {a.accuracy_risk:.2f} |"
            )
        lines.append("")
        lines.append(
            "Use these exact names in `propose_action` / `delegate` "
            "intents. Unknown names are rejected by PolicyGate."
        )
        return "\n".join(lines) + "\n\n"

    def _render_first_action_hint(self, agent_name: str) -> str:
        """Tiny first-action nudge so the executor reactor doesn't sit on
        heartbeats from cold start. Only fired when:

        * the agent is the executor (other roles don't propose actions),
        * baseline_tput is still 0 (no measurement has landed yet), and
        * no proposal/delegate event already exists in the events log.
        """
        if self.ctx is None or agent_name != "executor":
            return ""
        state = self.ctx.state
        try:
            base = float(state.baseline_tput or 0)
        except (TypeError, ValueError):
            base = 0.0
        if base > 0:
            return ""
        return (
            "## First action hint (live)\n"
            "No baseline measurement exists yet (baseline_tput=0). "
            "The natural first step is to delegate the `baseline` action "
            "(no params required — it reads MODEL/TP/CONC/ISL/OSL/"
            "INFERENCEX_PATH from the run env). "
            "Emit `delegate` with `action_name: baseline` and a brief "
            "`reason`. After the executor returns, follow up with "
            "`bench_runner` to get `cumulative_gain`.\n\n"
        )

    def _compose_prompt(
        self,
        agent_name: str,
        msgs: list[Message],
        *,
        sage_hint: str | None = None,
    ) -> str:
        assert self.ctx is not None
        ctx = self.ctx
        role = ctx.role_registry.get(agent_name)
        msg_lines = []
        for m in msgs[-10:]:
            body = self._render_msg_body(m.topic, m.payload)
            msg_lines.append(
                f"  - seq={m.seq} from={m.from_agent} topic={m.topic} :: {body}"
            )
        msgs_block = "\n".join(msg_lines) or "  (no recent messages)"
        # Per-role system prompt. Backends that support a separate system
        # message (Claude/Codex SDK) consume this via the ``extra`` channel;
        # MockBackend just stores the full string.
        role_header = (
            f"# Role: {agent_name}\n## Persona (static)\n"
            f"{role.system_prompt()}\n\n"
            if role is not None
            else f"# Role: {agent_name}\n\n"
        )
        # L3 long-term persona file (from `personas/<agent>.md`). Empty
        # string when the agent has not written anything yet.
        long_persona = ctx.persona_index.get(agent_name, "")
        long_persona_block = (
            f"## Persona (running notes)\n{long_persona}\n\n"
            if long_persona.strip()
            else ""
        )
        # IronRules for this mode.
        iron = render_iron_rules(ctx.state.execution_mode)
        # Optional sage hint.
        sage_block = (
            f"## Sage hint (KB recall)\n{sage_hint}\n\n"
            if sage_hint and sage_hint.strip()
            else ""
        )
        # Action catalogue + first-action hint (executor only).
        catalogue_block = self._render_action_catalogue(ctx.state.execution_mode)
        first_hint_block = self._render_first_action_hint(agent_name)
        return (
            f"{role_header}"
            f"{long_persona_block}"
            f"{iron}\n"
            f"{ctx.state.summary()}\n"
            f"## Objective\n{ctx.objective.describe()}\n\n"
            f"{catalogue_block}"
            f"{first_hint_block}"
            f"{sage_block}"
            f"## Recent inbox (latest first)\n{msgs_block}\n\n"
            f"## Mode flags\n{ctx.flags!r}\n"
        )

    # ------------------------------------------------------------------
    # Dispatcher / persona / resume helpers
    # ------------------------------------------------------------------
    async def _dispatcher_loop(self) -> None:
        """Spawned in :meth:`run`; drains queued delegate tasks via
        :class:`SubAgentRunner` until ``_dispatcher_stop`` is set.
        """
        assert self.ctx is not None
        ctx = self.ctx
        if ctx.sub_agent_runner is None:
            return
        try:
            await dispatch_pending_delegates(
                ctx.sub_agent_runner,
                db=ctx.db,
                poll_interval_s=max(0.1, self._reactor_tick_s),
                stop=self._dispatcher_stop,
            )
        except asyncio.CancelledError:
            log.debug("dispatcher cancelled")
            raise

    def _load_persona_index(self) -> dict[str, str]:
        """Read ``personas/<agent>.md`` files into an in-memory map."""
        out: dict[str, str] = {}
        personas_dir = self.session_dir / "personas"
        if not personas_dir.is_dir():
            return out
        for p in personas_dir.glob("*.md"):
            try:
                out[p.stem] = p.read_text(encoding="utf-8")
            except OSError:
                continue
        return out

    def _refresh_persona_index(self, agent_name: str) -> None:
        """Re-read a single agent's persona after an ``update_persona`` write."""
        if self.ctx is None:
            return
        path = self.session_dir / "personas" / f"{agent_name}.md"
        if not path.is_file():
            return
        try:
            self.ctx.persona_index[agent_name] = path.read_text(encoding="utf-8")
        except OSError:
            pass

    async def _init_session_resume(self) -> ResumeState | None:
        """Optional resume path — loads cursors / personas / inflight tasks
        from a previous run via :func:`resume_from_session_dir`. Quietly
        returns ``None`` when nothing was previously persisted.
        """
        if self.ctx is None:
            return None
        ctx = self.ctx
        try:
            return await resume_from_session_dir(
                self.session_dir, ctx.db, ctx.locks, tasks=ctx.tasks,
            )
        except Exception:  # noqa: BLE001 — resume is best-effort
            log.exception("resume_from_session_dir failed")
            return None

    # ------------------------------------------------------------------
    # Emergency / parliament
    # ------------------------------------------------------------------
    async def ephemeral_rca_via_critic(self) -> dict[str, Any] | None:
        """One-shot Critic invocation that produces an RCA finding.

        Used by the guided emergency branch (DESIGN §5.1.3 / §7.2). Returns
        the parsed RCA finding dict on success, or ``None`` on failure.
        """
        assert self.ctx is not None
        ctx = self.ctx
        prompt = (
            "# RCA Critic — emergency post-mortem\n"
            f"{ctx.state.summary()}\n\n"
            "## Recent events\n"
            + "\n".join(
                f"  - {e.seq} {e.from_agent}->{e.to_agent} {e.topic}"
                for e in await ctx.bus.tail(n=200)
            )
            + "\n\n## Task: produce ONE rca_finding intent.\n"
        )
        try:
            intents = await self.backend.run(
                prompt,
                agent_name="critic",
                allowed_tools=("emit_intent",),
                extra={"rca_mode": True},
            )
        except Exception:  # noqa: BLE001
            log.exception("ephemeral_rca_via_critic: backend.run failed")
            return None
        for intent in intents or []:
            if intent.type != IntentType.SEND_MESSAGE:
                continue
            payload = dict(intent.payload or {})
            if str(payload.get("topic")) == "rca_finding":
                return payload
        return None

    async def _open_parliament(self, proposal: dict[str, Any]) -> str:
        """Marathon-only — broadcast a proposal and collect votes.

        Returns ``"approved"`` / ``"rejected"`` / ``"abstained"`` based on
        the simple majority rule. The actual vote tally is recorded on the
        ``vote`` topic; consumers can replay them for audit.
        """
        assert self.ctx is not None
        ctx = self.ctx
        if ctx.state.execution_mode is not ExecutionMode.MARATHON_MULTI_AGENT:
            return "abstained"
        await ctx.bus.append_and_seq(
            Message.new(
                from_agent="conductor",
                to_agent="*",
                topic="proposal",
                payload={"kind": "parliament_open", **proposal},
                priority=0,
            )
        )
        # Collect every ``vote`` event posted in the next half-tick window.
        # In production we wait for explicit ``end-of-vote`` markers; v0.7
        # uses a fixed 2 s window which is fine for unit tests using mock
        # backends that respond instantly.
        await asyncio.sleep(min(2.0, self._reactor_tick_s * 4))
        events = await ctx.bus.tail(n=200)
        votes = [
            e.payload for e in events
            if e.topic == "vote" and isinstance(e.payload, dict)
        ]
        approve = sum(1 for v in votes if str(v.get("verdict")) == "approve")
        reject = sum(1 for v in votes if str(v.get("verdict")) == "reject")
        if approve > reject:
            return "approved"
        if reject > approve:
            return "rejected"
        return "abstained"

    async def _record_proposal_for_self_review(
        self, proposal: dict[str, Any]
    ) -> None:
        """Quick / guided modes — log proposal for the Executor self-review."""
        assert self.ctx is not None
        ctx = self.ctx
        await ctx.bus.append_and_seq(
            Message.new(
                from_agent="conductor",
                to_agent="executor",
                topic="proposal",
                payload={"kind": "self_review", **proposal},
                priority=2,
            )
        )


# ---------------------------------------------------------------------------
class TokenBudgetMeter:
    """Live token-spend tracker; warns when approaching mode budget.

    The Conductor records prompt + completion tokens via :meth:`record`.
    :meth:`should_throttle` flips ``True`` once the cumulative spend
    crosses ``warn_at`` (80% of the mode budget) — the Conductor uses
    that signal to ask the Critic for a 20% sampling drop (DESIGN §5.2).
    """

    _BUDGET_TOKENS = {
        ExecutionMode.QUICK_PARAM_SWEEP: 500_000,
        ExecutionMode.GUIDED_KERNEL_OPT: 3_000_000,
        ExecutionMode.MARATHON_MULTI_AGENT: 11_500_000,
    }

    def __init__(self, mode: ExecutionMode) -> None:
        self.mode = mode
        self.tokens_used = 0
        self.budget = self._budget_for_mode(mode)
        self.warn_at = int(self.budget * 0.8)
        self._tripped = False

    @classmethod
    def _budget_for_mode(cls, mode: ExecutionMode) -> int:
        return cls._BUDGET_TOKENS[mode]

    # ------------------------------------------------------------------
    def record(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> int:
        """Add tokens; returns the new running total."""
        self.tokens_used += max(0, int(prompt_tokens))
        self.tokens_used += max(0, int(completion_tokens))
        if self.tokens_used >= self.warn_at:
            self._tripped = True
        return self.tokens_used

    def should_throttle(self) -> bool:
        """``True`` once spend ≥ 80% of the mode budget."""
        return self._tripped

    def remaining(self) -> int:
        return max(0, self.budget - self.tokens_used)

    def reset(self) -> None:
        self.tokens_used = 0
        self._tripped = False
