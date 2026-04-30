"""Conductor — DESIGN §15 / standalone_agent_design §13 (v0.4 MVP).

Single owner of the run. Wires every subsystem (storage, locks, bus,
cursors, scheduler, policy, sub_agent_runner, ...) and runs the asyncio
gather of reactors + clock + stopping_watcher.

v0.4 MVP roster (per ``roles_for_mode``):
    quick     -> [executor, triage]
    guided    -> [executor, critic, kernel, triage]
    marathon  -> [executor, critic, kernel, triage]

Triage is always-on (active in every mode) with its own slow tick
(``triage_tick_s = 60.0``) so it can scan sibling agents' outbox/inbox
files for crash/stall signals. It is the only role allowed to emit
:attr:`IntentType.KILL_TASK` (PolicyGate enforces).

Removed in v0.4 (vs v0.3):
    * sage role + ``SageQueryService`` (no KB in MVP)
    * watchdog role (renamed to triage with broader powers)
    * parliament mode + OBJECTION/VOTE intents + ``_open_parliament``
    * ``ephemeral_rca_via_critic`` (triage covers RCA always-on)
    * legacy session resume — ``resume_from_session_dir`` rejects any
      session_dir whose state references the old roster

Every parsed intent runs through :meth:`PolicyGate.validate_intent`
before ``_handle_intent``. Denied intents are logged on the bus as a
``policy_denied`` observation; the reactor keeps going.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

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
from .multi_cli import (
    AgentCard,
    MultiCLILauncher,
    MultiCLIRouter,
    StagedAgent,
    discover_agent_cards,
)
from .objective import Objective, build_objective
from .policy import PolicyDenied, PolicyGate
from .resource_lock import ResourceLockManager, SqliteLeaseBackend
from .shared_state import SharedState
from .sub_agent_runner import SubAgentRunner, dispatch_pending_delegates
from .task_registry import TaskRegistry

if TYPE_CHECKING:  # pragma: no cover - type-only
    from .action_registry import ActionRegistry
    from .kb import KnowledgeBase
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


def _render_kill(p: dict) -> str:
    task = str(p.get("task_id", ""))[:8]
    reason = p.get("reason", "")
    return f"task={task} reason={reason}"


def _render_reflection_tick(p: dict) -> str:
    elapsed = p.get("elapsed_minutes")
    left = p.get("time_left_minutes")
    if isinstance(elapsed, (int, float)) and isinstance(left, (int, float)):
        return f"elapsed={elapsed:.2f}m time_left={left:.2f}m"
    return ""


def _render_request(p: dict) -> str:
    target = p.get("target_agent", "?")
    kind = p.get("kind", "?")
    src_msg = str(p.get("source_msg_id", ""))[:8]
    extras = ""
    params = p.get("params") or {}
    if isinstance(params, dict) and params:
        # Compact preview of the first 2 param keys; full payload still on disk.
        preview = ", ".join(f"{k}={v!r}" for k, v in list(params.items())[:2])
        if len(params) > 2:
            preview += f", +{len(params) - 2} more"
        extras = f" params={{{preview}}}"
    suffix = f" src={src_msg}" if src_msg else ""
    return f"target={target} kind={kind}{suffix}{extras}"


def _render_response(p: dict) -> str:
    in_reply = str(p.get("in_reply_to", ""))[:8]
    kind = p.get("kind", "?")
    status = p.get("status", "?")
    return f"in_reply_to={in_reply} kind={kind} status={status}"


_TOPIC_RENDERERS = {
    "proposal": _render_proposal,
    "decision": _render_decision,
    "alert": _render_alert,
    "question": _render_question,
    "answer": _render_answer,
    "kill": _render_kill,
    "reflection_tick": _render_reflection_tick,
    "request": _render_request,
    "response": _render_response,
}


# ---------------------------------------------------------------------------
class StopReason:
    TARGET_REACHED = "target_reached"
    TIME_EXHAUSTED = "time_exhausted"
    NO_MORE_LEVERAGE = "no_more_leverage"
    BRIER_PLATEAU = "brier_plateau"
    EMERGENCY = "emergency"


# ---------------------------------------------------------------------------
# Transport / process model for agents
# ---------------------------------------------------------------------------
class TransportMode:
    """How agent reactors are physically realised.

    SINGLE_PROC
        Every active role gets an in-process asyncio task driving
        ``backend.run`` per tick. The legacy v0.x model.

    MULTI_CLI
        Every active role runs as an independent CLI subprocess (claude
        --print --continue or codex with explicit conversation log),
        spawned by :class:`MultiCLILauncher`. The Conductor keeps only
        the Router + dispatcher in-process. See plan
        ``.cursor/plans/multi-cli-agents-a2a_*.plan.md``.

    HYBRID
        Selected roles run as CLIs (named via ``cli_agents``); the rest
        keep the in-process reactor. Useful as the Phase 2 migration
        rung where executor is the only CLI.
    """

    SINGLE_PROC = "single-proc"
    MULTI_CLI = "multi-cli"
    HYBRID = "hybrid"


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
    persona_index: dict[str, str] = field(default_factory=dict)
    multi_cli_router: MultiCLIRouter | None = None
    cli_agents: tuple[str, ...] = ()
    in_proc_roles: list[AgentRole] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
class Conductor:
    """Single source of truth for the run.

    Construction:
        ``Conductor(session_dir, backend=..., env=..., db=...)``

    The ``backend`` injection is what makes dry-runs (MockBackend) and real
    runs (ClaudeBackend / CodexBackend) interchangeable.

    Transport modes (``transport_mode`` constructor arg):

    * ``single-proc`` — every active role runs as an in-process asyncio
      reactor task. **Constructor default** for backward compatibility
      with library/test callers that instantiate :class:`Conductor`
      directly. Cheap to start, easy to debug, but bounded by the shared
      Python process's lifetime/context.
    * ``multi-cli`` — every active role becomes its own
      ``claude --print --continue`` (or ``codex`` with explicit
      conversation log) restart-loop CLI under tmux. **CLI default**
      since v0.9 (see ``cli.py``). Required for marathon >6h runs to
      avoid context-window exhaustion + give per-agent fault isolation.
    * ``hybrid`` — only the role names listed in ``cli_agents`` go to
      CLIs; the rest stay as in-process reactors. Useful as a
      transitional rung when migrating one role at a time.
    """

    DEFAULT_REACTOR_TICK_S = 2.0
    DEFAULT_CLOCK_TICK_S = 5.0
    # v0.4 MVP — triage runs slower than other reactors. Per
    # standalone_agent_design §13.9.3: 60s tick is enough for crash/stall
    # detection without burning Claude tokens.
    DEFAULT_TRIAGE_TICK_S = 60.0

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
        reactor_tick_s: float | None = None,
        clock_tick_s: float | None = None,
        triage_tick_s: float | None = None,
        enable_dispatcher: bool = True,
        enable_checkpointing: bool = False,
        transport_mode: str = TransportMode.SINGLE_PROC,
        cli_agents: Iterable[str] = (),
        agents_root: Path | None = None,
        router_tick_s: float | None = None,
        launch_cli_agents: str = "off",
        cli_shutdown_grace_s: float = 30.0,
        launcher_env: dict[str, str] | None = None,
        launcher_extra_dirs: Iterable[Path] = (),
        launcher_overrides: dict[str, AgentCard] | None = None,
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
        self._reactor_tick_s = reactor_tick_s or self.DEFAULT_REACTOR_TICK_S
        self._clock_tick_s = clock_tick_s or self.DEFAULT_CLOCK_TICK_S
        self._triage_tick_s = triage_tick_s or self.DEFAULT_TRIAGE_TICK_S
        self._enable_dispatcher = enable_dispatcher
        self._enable_checkpointing = enable_checkpointing
        self._dispatcher_stop = asyncio.Event()
        self._last_checkpoint_ts: float = 0.0
        # Multi-CLI transport (plan: multi-cli-agents-a2a)
        self._transport_mode = transport_mode
        self._cli_agents: tuple[str, ...] = tuple(cli_agents)
        self._agents_root = agents_root
        self._router_tick_s = router_tick_s
        # Auto-launch of agent CLI subprocesses (Phase A — patch plan).
        # ``off``       -> Conductor only runs Router; operator spawns
        #                  CLIs externally. Default; backward-compatible.
        # ``subprocess`` -> Conductor calls launcher.launch_subprocess()
        #                  during _bootstrap and waits for children on
        #                  graceful shutdown. No tmux dep.
        # ``tmux``      -> Conductor calls launcher.launch() (tmux-based)
        #                  during _bootstrap. Requires tmux on PATH.
        if launch_cli_agents not in ("off", "subprocess", "tmux"):
            raise ValueError(
                f"launch_cli_agents must be 'off' / 'subprocess' / 'tmux', "
                f"got {launch_cli_agents!r}"
            )
        self._launch_cli_agents: str = launch_cli_agents
        self._cli_shutdown_grace_s: float = float(cli_shutdown_grace_s)
        self._launcher_env: dict[str, str] = dict(launcher_env or {})
        self._launcher_extra_dirs: tuple[Path, ...] = tuple(launcher_extra_dirs)
        # Test seam: tests inject custom AgentCard overrides (e.g. backend=mock-cli)
        # so the e2e harness can drive real subprocesses without claude/codex.
        self._launcher_overrides: dict[str, AgentCard] = dict(launcher_overrides or {})
        self._launcher: MultiCLILauncher | None = None
        self._staged_cli_agents: dict[str, StagedAgent] = {}

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

        # ----- Multi-CLI plumbing -----
        # Decide which roles run as out-of-process CLIs (= MultiCLIRouter
        # owns their inbox/outbox) vs in-process reactors (= legacy
        # asyncio reactor task). The split is driven by transport_mode:
        #
        #   single-proc  -> all reactors in-process (legacy)
        #   multi-cli    -> every active role gets a CLI; in-proc list empty
        #   hybrid       -> roles named in self._cli_agents get a CLI;
        #                   the rest stay in-process.
        cli_agent_names: tuple[str, ...] = self._resolve_cli_agent_names(roles)
        in_proc_roles: list[AgentRole] = [
            r for r in roles if r.name not in set(cli_agent_names)
        ]
        multi_cli_router: MultiCLIRouter | None = None
        if cli_agent_names:
            multi_cli_router = self._build_router(
                bus=bus, policy=policy,
                cli_agent_names=cli_agent_names,
            )
            # Auto-launch the CLI subprocesses if the operator opted in.
            # We do this at bootstrap (vs in run()) so the agents are up
            # by the time the very first bus event lands; otherwise the
            # ``run_started`` event would be mirrored into an inbox no
            # one is yet listening to.
            if self._launch_cli_agents != "off":
                self._launcher, self._staged_cli_agents = self._launch_cli_agent_processes(
                    cli_agent_names=cli_agent_names,
                    router=multi_cli_router,
                )

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
            persona_index=persona_index,
            multi_cli_router=multi_cli_router,
            cli_agents=cli_agent_names,
            in_proc_roles=in_proc_roles,
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
                    "transport": self._transport_mode,
                    "cli_agents": list(cli_agent_names),
                },
                priority=0,
            )
        )
        state.write_snapshot(self.session_dir)
        log.info(
            "conductor: bootstrapped session=%s mode=%s roles=%s max_minutes=%.1f "
            "transport=%s cli_agents=%s in_proc=%s",
            state.session_id, mode.value, [r.name for r in roles], max_minutes,
            self._transport_mode, list(cli_agent_names),
            [r.name for r in in_proc_roles],
        )
        return ctx

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    async def run(self) -> ConductorContext:
        """Main entry. Returns the populated context after a graceful stop.

        Reactor topology depends on ``transport_mode``:

        * ``single-proc``: one in-process asyncio reactor per role
          (legacy behaviour).
        * ``multi-cli`` / ``hybrid``: roles listed in ``cli_agents`` are
          driven by an external CLI — the Router pumps their JSONL
          inbox/outbox files. The remaining roles still get in-process
          reactors.

        The clock, stopping-watcher and (when wired) delegate-dispatcher
        always run in-process because they don't need backend turns.
        """
        ctx = await self._bootstrap()
        try:
            reactor_tasks = [
                asyncio.create_task(
                    self._reactor(role.name), name=f"reactor-{role.name}"
                )
                for role in ctx.in_proc_roles
            ]
            tasks: list[asyncio.Task] = [
                *reactor_tasks,
                asyncio.create_task(self._clock(), name="clock"),
                asyncio.create_task(
                    self._stopping_watcher(), name="stopping-watcher"
                ),
            ]
            if ctx.multi_cli_router is not None:
                tasks.append(
                    asyncio.create_task(
                        ctx.multi_cli_router.run(), name="multi-cli-router",
                    )
                )
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
            if ctx.multi_cli_router is not None:
                ctx.multi_cli_router.request_stop()
            # Tell every spawned CLI agent to stop BEFORE we cancel our
            # asyncio tasks — that way the Router gets one more drain
            # tick to pick up any final intents the CLIs flushed before
            # exiting.
            await self._shutdown_cli_agent_processes()
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._graceful_stop(ctx.state.stop_reason or StopReason.EMERGENCY)
        finally:
            if self._owns_db and self._db is not None:
                self._db.close()
        return ctx

    # ------------------------------------------------------------------
    # Multi-CLI bootstrap helpers
    # ------------------------------------------------------------------
    def _resolve_cli_agent_names(
        self, roles: list[AgentRole]
    ) -> tuple[str, ...]:
        """Return the subset of ``roles`` that should run as CLI processes.

        Resolution order:

        1. ``transport_mode == single-proc`` -> ``()`` (no CLIs).
        2. ``transport_mode == multi-cli`` -> every active role.
        3. ``transport_mode == hybrid`` -> intersect ``self._cli_agents``
           with the active role names.

        Unknown role names in ``self._cli_agents`` are silently dropped
        and logged at INFO level — this keeps a stale ``--cli-agents``
        flag from breaking a single-proc fallback.
        """
        if self._transport_mode == TransportMode.SINGLE_PROC:
            return ()
        active = tuple(r.name for r in roles)
        if self._transport_mode == TransportMode.MULTI_CLI:
            return active
        # hybrid
        requested = set(self._cli_agents)
        keep = tuple(name for name in active if name in requested)
        dropped = sorted(requested - set(active))
        if dropped:
            log.info(
                "transport=hybrid: ignoring cli_agents=%s "
                "(not in active roster %s)", dropped, list(active),
            )
        return keep

    def _build_router(
        self,
        *,
        bus: MessageBus,
        policy: PolicyGate,
        cli_agent_names: tuple[str, ...],
    ) -> MultiCLIRouter:
        """Construct + register :class:`MultiCLIRouter` for the named agents.

        Cards are discovered under ``self._agents_root`` (default:
        ``src/inference_optimizer/agents/``). Names not represented by a
        card produce a stub :class:`AgentCard` so the Router still has a
        path namespace; this lets early phases run before every role has
        an authored card.
        """
        # Late-bind the agents root so tests can inject a tmp tree via
        # the AGENTS_ROOT env var. Default = the package path.
        if self._agents_root is not None:
            root = Path(self._agents_root)
        else:
            from ..agents import agents_root as _default_root
            root = _default_root()
        cards_by_name = discover_agent_cards(root) if root.is_dir() else {}
        agents: list[AgentCard] = []
        for name in cli_agent_names:
            card = cards_by_name.get(name)
            if card is None:
                log.warning(
                    "multi-cli: no agent_card for %r under %s; using stub",
                    name, root,
                )
                card = self._stub_agent_card(name)
            agents.append(card)

        router = MultiCLIRouter(
            session_dir=self.session_dir,
            bus=bus,
            policy=None,  # PolicyGate is enforced inside _gate_intent
            agents=agents,
            intent_handler=self._router_intent_handler,
            deny_recorder=self._router_deny_recorder,
            tick_s=self._router_tick_s,
        )
        return router

    @staticmethod
    def _stub_agent_card(name: str) -> AgentCard:
        """Return a placeholder :class:`AgentCard` when no YAML is on disk.

        Used so the Router still has correct inbox/outbox paths even
        when an operator stands up a brand-new role before authoring
        its card. The launcher refuses to spawn stubs.
        """
        # Local import to avoid pulling agents/ module at module import.
        from .multi_cli.agent_card import RestartPolicy
        return AgentCard(
            name=name,
            role="executor",  # benign default; PolicyGate uses role_registry
            backend="mock",
            card_path=Path("/dev/null"),
            card_dir=Path("/dev/null"),
            enabled=True,
            restart_policy=RestartPolicy(),
        )

    def _launch_cli_agent_processes(
        self,
        *,
        cli_agent_names: tuple[str, ...],
        router: MultiCLIRouter,
    ) -> tuple[MultiCLILauncher, dict[str, StagedAgent]]:
        """Spawn one CLI process per name in ``cli_agent_names``.

        Card resolution order:
            1. ``self._launcher_overrides[name]`` (test seam — e.g.
               wires a ``backend=mock-cli`` card that drives the in-tree
               mock_agent module).
            2. The card the Router picked up from agents_root (the
               authored ``agent_card.yaml``).
            3. Skip with a warning if neither is available — without a
               real card we can't pick a backend template.
        """
        cards_for_launch: dict[str, AgentCard] = {}
        for name in cli_agent_names:
            card = self._launcher_overrides.get(name) or router.agents.get(name)
            if card is None:
                log.warning(
                    "auto-launch: no card for %r; skipping process spawn",
                    name,
                )
                continue
            if not card.enabled:
                log.info("auto-launch: card for %r disabled; skipping spawn", name)
                continue
            cards_for_launch[name] = card

        if not cards_for_launch:
            log.warning("auto-launch: no cards to spawn; running router-only")
            return MultiCLILauncher(
                session_dir=self.session_dir, cards={}
            ), {}

        # Pull through every env var the agent CLIs are likely to need
        # for action executors (MODEL_PATH, MAX_HOURS, INFERENCEX_PATH...).
        merged_env: dict[str, str] = dict(self.env)
        merged_env.update(self._launcher_env)

        launcher = MultiCLILauncher(
            session_dir=self.session_dir,
            cards=cards_for_launch,
            env=merged_env,
            extra_dirs=tuple(self._launcher_extra_dirs),
            agent_root=self._agents_root,
        )
        if self._launch_cli_agents == "tmux":
            staged = launcher.launch()
        else:
            staged = launcher.launch_subprocess()
        log.info(
            "auto-launch: started %d CLI agent(s) via %s: %s",
            len(staged), self._launch_cli_agents, sorted(staged.keys()),
        )
        return launcher, staged

    async def _shutdown_cli_agent_processes(self) -> None:
        """Drop STOP sentinels + wait for every spawned subprocess to exit.

        Called by :meth:`run` after the asyncio tasks are cancelled.
        Tmux panes own their own process tree so we only signal them
        via the STOP file; the subprocess path uses :meth:`wait_for_exit`
        with a SIGKILL fallback to keep the Conductor process from
        deadlocking on a wedged child.
        """
        if self._launcher is None or not self._staged_cli_agents:
            return
        try:
            self._launcher.request_stop_all(self._staged_cli_agents)
        except Exception:  # noqa: BLE001
            log.exception("auto-launch: failed to request stop on CLI agents")
        if self._launch_cli_agents == "subprocess":
            # ``wait_for_exit`` is blocking; offload to a thread so the
            # asyncio event loop stays responsive (unit-test gracefully).
            try:
                rcs = await asyncio.to_thread(
                    self._launcher.wait_for_exit,
                    self._staged_cli_agents,
                    timeout_s=self._cli_shutdown_grace_s,
                )
                log.info("auto-launch: CLI agents exited with rcs=%s", rcs)
            except Exception:  # noqa: BLE001
                log.exception("auto-launch: error while waiting for CLI exit")

    async def _router_intent_handler(
        self, agent_name: str, intent: Intent
    ) -> None:
        """Bridge for outbox intents → Conductor's normal intent pipeline.

        Reuses :meth:`_gate_intent` so every cross-process intent goes
        through the *same* PolicyGate as in-process reactors. Denied
        intents are short-circuited inside ``_gate_intent``; accepted
        intents flow through ``_handle_intent`` exactly like reactor
        emissions.
        """
        if not await self._gate_intent(agent_name, intent):
            return
        await self._handle_intent(agent_name, intent)

    async def _router_deny_recorder(
        self, agent_name: str, intent: Intent, rule: str, reason: str
    ) -> None:
        """Persist a malformed-envelope deny as a ``policy_denied`` observation.

        ``_gate_intent`` already records its own deny observation, so
        this helper only fires for upstream Envelope-level errors
        (unknown intent_type, missing fields). It mirrors the same
        observation shape so the monitor view stays uniform.
        """
        await self._record_observation(
            from_agent="conductor",
            kind="policy_denied",
            detail={
                "agent": agent_name,
                "intent_type": intent.type.value,
                "rule": rule,
                "reason": reason,
                "via": "multi_cli_router",
            },
        )

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
        # v0.4 — triage runs at a slower tick than other reactors. For the
        # triage reactor, ``tick_s`` is the **minimum interval between
        # turns** (enforced even when the bus has new events), so a 60s
        # tick truly throttles it. For other reactors we keep the legacy
        # behaviour: tick_s only governs the idle wait. See
        # standalone_agent_design §13.9.3.
        is_triage = agent_name == "triage"
        tick_s = self._triage_tick_s if is_triage else self._reactor_tick_s
        try:
            while not ctx.state.should_stop():
                if is_triage:
                    # Throttle triage between turns regardless of inbox
                    # depth so its slow tick is honoured even under load.
                    await asyncio.sleep(tick_s)
                    if ctx.state.should_stop():
                        break
                cursor = await ctx.cursors.load(agent_name)
                msgs = await ctx.bus.replay_for(
                    agent_name, after_seq=cursor.last_processed_seq
                )
                # Drop self-emitted messages (the bus broadcasts ``to=*`` so a
                # reactor would otherwise consume its own output).
                to_process = [m for m in msgs if m.from_agent != agent_name]
                if not msgs:
                    await asyncio.sleep(tick_s)
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

        Dispatch table (DESIGN §10.5.6 + §15 + standalone §13.3):

            send_message    -> bus.append(topic from payload)
            alert           -> bus.append(topic="alert") + findings/alerts.jsonl
            propose_action  -> tasks.create(kind="proposal") + topic="proposal"
            delegate        -> tasks.create(kind="delegate")  + topic="proposal"
                               (queued; SubAgentRunner picks it up — Phase F3)
            update_state    -> state.apply_validated_transition + topic="decision"
            update_persona  -> personas/<agent>.md append + topic="event"
            ask_question    -> bus.append(topic="question")
            answer          -> bus.append(topic="answer")
            request         -> mirror to target agent's inbox (RPC)
            response        -> reverse-route to original requester (RPC)
            kill_task       -> tasks.transition(cancelled) + bus(topic="kill")
                               + findings/kills.jsonl (triage only — v0.4)

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
        elif intent.type == IntentType.REQUEST:
            await self._handle_request(from_agent, payload)
        elif intent.type == IntentType.RESPONSE:
            await self._handle_response(from_agent, payload)
        elif intent.type == IntentType.KILL_TASK:
            await self._handle_kill_task(from_agent, payload)
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

    async def _handle_kill_task(
        self, from_agent: str, payload: dict[str, Any]
    ) -> None:
        """kill_task -> tasks.transition(cancelled) + bus(topic="kill") +
        findings/kills.jsonl mirror.

        v0.4 MVP — triage-only (PolicyGate enforces the source allowlist).
        Cancel is "simplest cooperative": the tasks row is marked cancelled
        so the dispatcher won't schedule new attempts; any in-flight
        ActionExecutor finishes its current run naturally. We do NOT
        forcibly cancel asyncio tasks or kill subprocesses — see
        standalone_agent_design §13.9.2.
        """
        from .task_registry import IllegalTransition, TaskNotFound

        assert self.ctx is not None
        ctx = self.ctx
        task_id = str(payload.get("task_id", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        force = bool(payload.get("force", False))
        scope = str(payload.get("scope") or "task").strip()

        cancel_status = "ok"
        cancel_detail = ""
        try:
            task_before = await ctx.tasks.get(task_id)
            if task_before.state in {"succeeded", "cancelled", "needs_manual_review"}:
                cancel_status = "noop_terminal"
                cancel_detail = f"task already in {task_before.state}"
            else:
                await ctx.tasks.transition(
                    task_id,
                    "cancelled",
                    evidence={
                        "kill_by": from_agent,
                        "reason": reason,
                        "force": force,
                        "scope": scope,
                    },
                )
        except TaskNotFound:
            cancel_status = "not_found"
            cancel_detail = f"no task with task_id={task_id!r}"
            log.warning(
                "kill_task: task_id=%s not found (from=%s)", task_id, from_agent,
            )
        except IllegalTransition as exc:
            cancel_status = "illegal_transition"
            cancel_detail = str(exc)
            log.warning("kill_task: illegal transition for %s: %s", task_id, exc)

        await ctx.bus.append_and_seq(
            Message.new(
                from_agent=from_agent,
                to_agent="*",
                topic="kill",
                payload={
                    "task_id": task_id,
                    "reason": reason,
                    "force": force,
                    "scope": scope,
                    "status": cancel_status,
                    "detail": cancel_detail,
                },
                priority=0,
            )
        )
        try:
            self._append_finding(
                "kills.jsonl",
                {
                    "from": from_agent,
                    "task_id": task_id,
                    "reason": reason,
                    "force": force,
                    "scope": scope,
                    "status": cancel_status,
                    "detail": cancel_detail,
                    "ts": time.time(),
                    "session_id": ctx.state.session_id,
                },
            )
        except OSError:  # pragma: no cover — best-effort
            log.exception("failed to mirror kill to findings/kills.jsonl")

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
        # Detect idempotency-hit: if the returned task is already in a
        # terminal state (succeeded / failed / safely_failed), the LLM is
        # re-delegating something that won't get re-run by the dispatcher.
        # Emit an extra ``event`` so the executor reactor sees the dead
        # end and pivots to a different action_name on the next turn,
        # instead of looping (the v0.8 marathon-mode failure where
        # ``profile`` was re-delegated 10+ times after a single failure).
        terminal_states = {"succeeded", "failed", "safely_failed",
                           "needs_manual_review"}
        if task.state in terminal_states:
            await ctx.bus.append_and_seq(
                Message.new(
                    from_agent="conductor",
                    to_agent=from_agent,
                    topic="event",
                    payload={
                        "kind": "delegate_dedup_to_terminal",
                        "task_id": task.task_id,
                        "action_name": action_name,
                        "task_state": task.state,
                        "hint": (
                            f"action {action_name!r} already ran and ended "
                            f"in state={task.state!r}; the dispatcher will "
                            f"NOT re-run it. Pick a different action_name "
                            f"from the catalogue (e.g. bench_runner, "
                            f"param_sweep_run, kernel_opt) or change "
                            f"params to break idempotency."
                        ),
                    },
                    priority=0,
                )
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
                    "task_state": task.state,
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
    # REQUEST / RESPONSE — agent-to-agent RPC (Plan A: kernel agent)
    # ------------------------------------------------------------------
    async def _handle_request(
        self, from_agent: str, payload: dict[str, Any]
    ) -> None:
        """REQUEST → ``topic="request"`` event addressed to ``target_agent``.

        Routing relies on the existing bus + Router/MultiCLI mirror path:
        when ``to_agent`` matches an active agent's name, the message is
        replayed into that agent's inbox via ``bus.replay_for`` (single-
        proc reactor) or ``MultiCLIRouter.mirror_bus_tick`` (multi-cli).

        We do NOT validate ``target_agent`` here — :class:`PolicyGate` did
        that already. Priority is bumped to 2 so the target agent's
        reactor processes the request promptly without waiting for the
        next batch flush.
        """
        assert self.ctx is not None
        ctx = self.ctx
        target = str(payload.get("target_agent", "")).strip()
        if not target:  # pragma: no cover — PolicyGate would reject first
            await self._record_observation(
                from_agent="conductor",
                kind="request_missing_target",
                detail={"from": from_agent, "payload": payload},
            )
            return
        kind = str(payload.get("kind", "")).strip() or "?"
        # Allocate the request its own envelope on the bus; the msg_id we
        # get back is what the target uses as ``in_reply_to`` later.
        msg = Message.new(
            from_agent=from_agent,
            to_agent=target,
            topic="request",
            payload={
                "kind": kind,
                "target_agent": target,
                "params": payload.get("params") or {},
                "reason": payload.get("reason", ""),
                # Echo any extra free-form keys verbatim so the target
                # doesn't lose context (e.g. correlation_id).
                **{
                    k: v for k, v in payload.items()
                    if k not in {
                        "kind", "target_agent", "params", "reason",
                        "to", "topic", "priority", "in_reply_to",
                    }
                },
            },
            priority=int(payload.get("priority", 2)),
        )
        await ctx.bus.append_and_seq(msg)

    async def _handle_response(
        self, from_agent: str, payload: dict[str, Any]
    ) -> None:
        """RESPONSE → ``topic="response"`` event reverse-routed to the
        request's original sender.

        We use ``payload['in_reply_to']`` to look up the original request
        envelope on the bus (best-effort; falls back to broadcast
        ``to_agent="*"`` if the request can't be found, e.g. very old
        request that aged out of the events window).
        """
        assert self.ctx is not None
        ctx = self.ctx
        in_reply_to = str(payload.get("in_reply_to", "")).strip()
        if not in_reply_to:  # pragma: no cover — PolicyGate would reject first
            await self._record_observation(
                from_agent="conductor",
                kind="response_missing_in_reply_to",
                detail={"from": from_agent, "payload": payload},
            )
            return

        # Look up the original request to find who to send back to.
        original_sender = await self._lookup_original_request_sender(in_reply_to)
        if original_sender is None:
            log.warning(
                "response: could not find original request msg_id=%s; "
                "broadcasting to=*", in_reply_to,
            )
            original_sender = "*"
        kind = str(payload.get("kind", "")).strip() or "?"
        msg = Message.new(
            from_agent=from_agent,
            to_agent=original_sender,
            topic="response",
            payload={
                "kind": kind,
                "status": payload.get("status", "succeeded"),
                "result": payload.get("result"),
                **{
                    k: v for k, v in payload.items()
                    if k not in {
                        "kind", "status", "result",
                        "to", "topic", "priority", "in_reply_to",
                    }
                },
            },
            priority=int(payload.get("priority", 2)),
            in_reply_to=in_reply_to,
        )
        await ctx.bus.append_and_seq(msg)

    async def _lookup_original_request_sender(
        self, request_msg_id: str
    ) -> str | None:
        """Reverse-lookup: find the ``from_agent`` of the request envelope
        carrying ``msg_id == request_msg_id``.

        Uses :meth:`MessageBus.lookup_by_id` (indexed by msg_id) so this
        stays O(1) regardless of bus size. Returns ``None`` if not found
        — caller falls back to broadcast.
        """
        assert self.ctx is not None
        try:
            msg = await self.ctx.bus.lookup_by_id(request_msg_id)
        except Exception:  # noqa: BLE001 — best-effort
            log.exception("response: bus.lookup_by_id failed for reverse-route")
            return None
        if msg is None or msg.topic != "request":
            return None
        return msg.from_agent

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
    # Self-review (replaces parliament since v0.4 — see §13.1)
    # ------------------------------------------------------------------
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
