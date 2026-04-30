"""Conductor — DESIGN v0.6 §7.5 / §21 main loop.

The Conductor is the **protocol manager** (not a decision-maker). It owns:

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..paths import db_path_for, make_session_dir
from ..storage.connection import SqliteConnection
from .agent_role import AgentRole, default_role_registry, roles_for_run
from .backends.base import Backend, BackendError, BackendTurnResult
from .cursor_store import CursorStore
from .intent_parser import Intent, IntentType
from .message_bus import Message, MessageBus
from .policy import (
    KILL_TASK_SOURCE_ALLOWLIST,
    PolicyDenied,
    PolicyGate,
    REQUEST_ROUTING,
    REVIEW_VERDICT_SOURCE_ALLOWLIST,
    ROBUSTNESS_ONLY_SOURCE_ALLOWLIST,
)
from .resource_lock import ResourceLockManager, SqliteLeaseBackend
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
class ConductorState:
    """In-memory mirror of session state used by reactor + dispatcher."""

    pending_proposals: dict[str, PendingProposal] = field(default_factory=dict)
    pruned_families: set[str] = field(default_factory=set)


class Conductor:
    """The single Conductor instance per session.

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
                    f"(provide via Conductor(backends={{...}}))"
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

        self.state = ConductorState()
        self._stop = asyncio.Event()
        self._tasks_running: list[asyncio.Task] = []

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

        Used by P0-3 / P0-5 tests. Each agent's backend.run() is called
        once per pass; intents are validated + routed before the next
        pass starts. Dispatcher pumps queued tasks at the end of each pass.
        """
        for _ in range(n):
            for name in roles_for_run():
                await self._reactor_pass(name)
            await self._pump_dispatcher_once()

    # ==================================================================
    # Reactor
    # ==================================================================
    async def _reactor_pass(self, agent_name: str) -> None:
        backend = self.backends[agent_name]
        prompt = await self._compose_prompt(agent_name)
        sys_prompt = await self._load_system_prompt(agent_name)
        tools = self.policy.allowed_tools_for_agent(agent_name)
        try:
            result: BackendTurnResult = await backend.run(
                prompt=prompt, system_prompt=sys_prompt, tools=tools, max_turns=1,
            )
        except BackendError as exc:
            await self._record_observation(
                "conductor", "observation",
                {"kind": "backend_error", "agent": agent_name, "error": repr(exc)},
            )
            return
        for intent in result.intents:
            await self._handle_intent(agent_name, intent)

    async def _compose_prompt(self, agent_name: str) -> str:
        """Minimal v0.6 prompt: pending messages tail + state summary.

        v0.6 §8.3 specifies a richer composition (KB hint / Critic verdict
        / Robustness alert / persona). P0-3 ships the skeleton; the rich
        composition lands in P0-5 before the e2e demo.

        We include the canonical ``msg_id`` for each inbox row so reactive
        mock backends (and future real LLMs) can address replies via
        ``in_reply_to`` / ``target_proposal_msg_id`` without extra lookups.
        """
        cursor = await self.cursors.load(agent_name)
        msgs = await self.bus.replay_for(agent_name, after_seq=cursor.last_processed_seq)
        if not msgs:
            return f"(no new messages for {agent_name})"
        lines = [f"Inbox for {agent_name} (newest last):"]
        for m in msgs[-20:]:
            lines.append(
                f"  seq={m.seq} msg_id={m.msg_id} from={m.from_agent} "
                f"topic={m.topic} payload={m.payload}"
            )
        return "\n".join(lines)

    async def _load_system_prompt(self, agent_name: str) -> str:
        role = self.role_registry[agent_name]
        try:
            return role.load_system_prompt()
        except FileNotFoundError:
            return f"(no system prompt for {agent_name})"

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
        if action_name in self.state.pruned_families:
            # Soft-reject pruned families (Robustness already pruned this)
            await self._record_observation(
                "conductor", "observation",
                {"kind": "proposal_pruned", "from": source, "action": action_name},
            )
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
                "conductor", "observation",
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
        """Promote an approved proposal into a TaskRegistry entry."""
        task = await self.tasks.create(
            kind=pending.action_name,
            params=pending.payload.get("params") or {},
            idempotency_key=f"approved-{pending.proposal_msg_id}",
        )
        await self.bus.append_and_seq(Message.new(
            "conductor", "*", "decision",
            {"kind": "approved_proposal", "task_id": task.task_id,
             "action_name": pending.action_name, "from_agent": pending.from_agent},
        ))

    # ------------------------------------------------------------------
    # DELEGATE
    # ------------------------------------------------------------------
    async def _handle_delegate(self, source: str, intent: Intent) -> None:
        action_name = intent.payload["action_name"]
        params = intent.payload.get("params") or {}
        idempotency_key = intent.payload.get("idempotency_key") or f"{source}:{action_name}:{len(self.state.pending_proposals)}"
        task = await self.tasks.create(
            kind=action_name,
            params=params,
            idempotency_key=idempotency_key,
        )
        await self.bus.append_and_seq(Message.new(
            "conductor", "*", "event",
            {"kind": "task_queued", "task_id": task.task_id, "source": source, "action": action_name},
        ))

    # ------------------------------------------------------------------
    # REQUEST / RESPONSE (Plan A)
    # ------------------------------------------------------------------
    async def _handle_request(self, source: str, intent: Intent) -> None:
        target_agent = intent.payload["target_agent"]
        kind = intent.payload["kind"]
        msg = Message.new(source, target_agent, "request",
                          dict(intent.payload), priority=1)
        await self.bus.append_and_seq(msg)

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
                "conductor", "observation",
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
        self.state.pruned_families.add(family)
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
        # Persist as observation; full SharedState lands in P0-5.
        await self.bus.append_and_seq(Message.new(
            source, "*", "observation",
            {"kind": "update_state", "changes": intent.payload["changes"]},
        ))

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------
    async def _record_policy_denied(
        self, source: str, intent: Intent, denied: PolicyDenied
    ) -> None:
        await self.bus.append_and_seq(Message.new(
            "conductor", source, "observation",
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
                "conductor", "*", "delegated_result",
                {"task_id": task.task_id, "kind": task.kind,
                 "state": result.state, "result": result.result,
                 "error": result.error},
            ))


__all__ = ["Conductor", "ConductorState", "PendingProposal"]
