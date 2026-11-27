# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, AbstractSet, Any, Awaitable, Callable

from hyperloom_common.env import env_float, env_int
from hyperloom_common.timeutil import now_iso
from hyperloom.orchestrator.knowledge.config import KnowledgeConfig, KnowledgeStoreMode
from hyperloom.orchestrator.knowledge.recipe_kb import RecipeKB

# Periodic in-process maintenance/reaper cadence (lease reaping + DB retention), in wall-clock seconds.
MAINTENANCE_INTERVAL_SEC: int = 1800

# Default per-macro-cycle wall-clock window (hours) in cyclic mode.
DEFAULT_CYCLE_HOURS: float = 24.0
# Trailing window for the crash-rate emergency stop, in seconds.
_CRASH_EMERGENCY_WINDOW_SEC: float = 24.0 * 3600.0
# Coordinator exceptions within the trailing window that end the run as ``emergency``.
_CRASH_EMERGENCY_THRESHOLD: int = 25
# Retry delay while the LLM backend keeps failing, doubling from the base to the ceiling, in seconds. Ints, so the
# doubling cannot overflow a float however long a streak runs.
_BACKEND_RETRY_BASE_SEC: int = 2
_BACKEND_RETRY_MAX_SEC: int = 300
# How long the backend may fail without a successful turn before the session stops, in seconds.
_BACKEND_UNHEALTHY_STOP_SEC: float = 3600.0
# How many turns in a row may be answered without an intent before the session stops.
_NO_INTENT_STOP_TURNS: int = 50
from ..phases import machine_state as _phase_state
from hyperloom.inference_optimizer.session.session_binding import bind_session
from ..roles.base import Backend, BackendError, BackendTurnResult, LLMCallFailed
from ..bus.cursor_store import CursorStore
from ..bus.storage.connection import SqliteConnection
from hyperloom.inference_optimizer.protocol.intent import NoIntentEmitted
from ..bus.message_bus import MessageBus
from ..state.objective import Objective, TimeOnlyObjective
from ..policy.gate import PolicyGate
from ..policy.projection import serving_tp_for_policy
from ..state.round_store import RoundStore
from ..bus.gpu_pool import SpecialistGpuPool, resolve_gpu_specialist_devices, resolve_whole_machine_devices
from ..bus.resource_lock import SqliteLeaseBackend
from ..state.shared_state import SharedState, effective_closing_grace_sec
from ..state.state_file import database_target
from .signals import SignalDrain
from .intent_router import IntentRouter
from .reactor_turn import (
    close_backends,
    resolve_reactor_turn_timeout_sec,
    trace_mcp_setup,
    trace_reactor_llm_call,
)
from hyperloom.inference_optimizer.model_config_utils import infer_model_class
from .sub_agent_runner import SubAgentRunner
from .turn_pump import PumpBesideTurn
from ..state.task_registry import TaskRegistry, task_dispatch_origin
from ..collaborator import ORCHESTRATION_AGENT, OrchestrationPrompt
from ..mcp_tool_names import ORCHESTRATION_TOOL_GRANT
from hyperloom.inference_optimizer.trace.llm_trace import (
    LLMCallRecord,
    LLMCallSite,
    LLMTraceRowError,
    append_llm_call,
    new_call_id,
)
from hyperloom.inference_optimizer.trace.context_events import PromptSnapshotTracker, record_prompt_snapshot
from hyperloom.inference_optimizer.trace.trajectory_trace import (
    EVENT_LLM_CALL,
    EVENT_SESSION,
    llm_call_summary,
    trajectory_scope,
    trajectory_span,
)
from hyperloom.common.deadline import Deadline, seconds_until
from hyperloom_common.timeutil import format_exc_brief


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from hyperloom.inference_optimizer.experience_kb import ExperienceKBIntegration


def reap_orphaned_servers_best_effort(session_dir: Path, *, phase: str) -> None:
    """Stop the single-node servers this session's ``server.pid`` files still record.

    Runs at boot, at shutdown and on resume. Every other teardown is in-band, so none of it runs when the owner dies
    without getting to; this is the backstop, and with no cgroup or pid namespace the only one: the pidfile outlives
    its writer. Scoped to this session's own pidfiles, and to the incarnation and run mark each one records, so a
    co-located session's server and a recycled pid are never touched.
    """
    try:
        from ..actions.executors._multi_node_env import is_multi_node

        if is_multi_node():
            return
        from ..actions.executors.serving._server_lifecycle import reap_orphaned_servers

        reaped = reap_orphaned_servers(session_dir)
        if reaped:
            log.warning(
                "coordinator: reaped %d orphaned serving process(es) at %s: %s",
                len(reaped),
                phase,
                reaped,
            )
    except (OSError, RuntimeError, ValueError):
        log.exception("coordinator: orphan server reaper failed at %s (ignored)", phase)


@dataclass(frozen=True)
class CoordinatorSettings:
    """Boot-time choices the launcher makes for a session.

    Attributes:
        model_class: Model class to record when the session has none yet.
        phase_budget_pct: Per-phase budget percentages; library defaults when absent.
    """

    model_class: str | None = None
    phase_budget_pct: dict[str, float] | None = None


@dataclass(frozen=True)
class AdvisoryServices:
    """Optional advisory collaborators the session consults.

    Attributes:
        knowledge_plane: KnowledgePlane facade; pre-warms PR feed + advisory context.
    """

    knowledge_plane: Any = None


@dataclass(frozen=True)
class RunOptions:
    """How long and how a Coordinator run ticks, and what stops it early.

    Attributes:
        max_minutes: Session wall-clock bound; unbounded when ``None``.
        tick_interval_sec: Pause between ticks.
        max_ticks: Stop after this many ticks, when set.
        install_signal_handlers: Whether to arm the stop-signal drain.
        closing_grace_sec: Bound on the closing phase; the default when ``None``.
    """

    max_minutes: float | None = None
    tick_interval_sec: float = 0.0
    max_ticks: int | None = None
    install_signal_handlers: bool = False
    closing_grace_sec: float | None = None


class Coordinator:
    """The single Coordinator instance per session."""

    def __init__(
        self,
        session_dir: Path,
        *,
        backends: dict[str, Backend],
        settings: CoordinatorSettings = CoordinatorSettings(),
        advisory: AdvisoryServices = AdvisoryServices(),
    ):
        """Construct the per-session Coordinator and wire persistence, policy, and agents."""
        self.session_dir = Path(session_dir)
        # Bind the session for the SBD V6 recorders once, here, so no recorder entry point below has to be handed a
        # path.
        bind_session(self.session_dir)
        # Orchestration prompt overrides, rebuild closure, and snapshot writes. Empty until the launcher installs the
        # system prompt; a turn taken before that raises.
        self.orch_prompt = OrchestrationPrompt(overrides={})
        # KnowledgePlane facade; pre-warms PR feed + advisory context.
        self.knowledge_plane: Any = advisory.knowledge_plane
        self._model_class_override: str = (settings.model_class or "").strip()

        if ORCHESTRATION_AGENT not in backends:
            raise ValueError(
                f"missing backend for role {ORCHESTRATION_AGENT!r} (provide via Coordinator(backends={{...}}))"
            )
        self.backends = dict(backends)
        self.reactor_turn_timeout_sec = resolve_reactor_turn_timeout_sec()

        # Persistent session state (state.json) — load existing for resume; its operator pins decide the database's mode.
        self.shared_state = SharedState.load_or_init(self.session_dir)
        self._open_persistence()
        # Thread live SharedState into the runner so executors get it via ctx.extra.
        self.sub.shared_state = self.shared_state
        self._init_gpu_pools()
        # The durable bring-up mutex. The gate never reads it; ``open`` decides.
        self.rounds = RoundStore(self.db)
        self.policy = PolicyGate(
            session_dir=self.session_dir,
            shared_state=self.shared_state,
            # Seeded at boot: an empty snapshot would advise against a GPU
            # dispatch the pool can in fact satisfy.
        )
        self.sub.policy = self.policy
        # Attach read-only context-pull MCP tools to Orchestration backend.
        self.conversation.attach_orchestration_context_tools()
        # Resume detection must run before any boot-time state.json write.
        self.writeback.detect_resume_state()
        # Reap serving processes orphaned by a prior monitor-process crash, scoped strictly to this session's own
        # pidfiles.
        reap_orphaned_servers_best_effort(self.session_dir, phase="boot")
        # Before any bring-up can dispatch: an attempt classified against a
        # different pin is not comparable with its neighbours.
        self._pin_source_trees()
        # Derive model_class once at boot if not supplied; never overwrite a resume.
        if not (self.shared_state.model_class or "").strip():
            self.shared_state.model_class = self._model_class_override or infer_model_class(
                self.shared_state.model_path or os.environ.get("MODEL_PATH", "")
            )
        self._stop = asyncio.Event()
        self._stop_classification = ""

        # Wall-clock stamp of the last maintenance pass (lease reaping + DB retention).
        self._last_maintenance_ts: float = time.monotonic()

        # Pin a per-macro-cycle budget window so per-phase budget fractions apply per cycle.
        if float(self.shared_state.cycle_minutes or 0) <= 0:
            _cycle_hours = env_float("INFERENCE_OPTIMIZER_CYCLE_HOURS", default=DEFAULT_CYCLE_HOURS)
            self.shared_state.cycle_minutes = max(1.0, _cycle_hours * 60.0)

        self._prompt_snapshots = PromptSnapshotTracker()

        # Consecutive BackendErrors, the back-off exponent; crossing the threshold records one backend_unhealthy, which
        # re-arms after a successful turn.
        self._backend_error_streak: int = 0
        self._backend_error_alarm_armed: bool = True
        # Monotonic time the streak began, and before which the tick loop does not call the backend again.
        self._backend_error_since: float | None = None
        self._backend_retry_at: float = 0.0
        self._backend_error_streak_threshold: int = max(
            1,
            env_int("INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD", default=5),
        )
        # Run of consecutive turns that emitted no intent; a turn with an intent ends it.
        self._no_intent_streak: int = 0

        # Wall-clock budget tracking for per-tick Time-budget prompt injection.
        self._run_deadline: Deadline | None = None
        # Closing-grace bound; used only while ``closing_phase`` is set so CLOSE
        # work is not skipped just because the session deadline has passed.
        self._closing_deadline: Deadline | None = None
        self._signals: SignalDrain | None = None
        # Latest objective wired by run(); None outside a run.
        self._current_objective: Objective | None = None

        # Initialise phase machine (fresh session enters PRELUDE). Idempotent.
        self.phase_machine.ensure_phase_initialised(settings.phase_budget_pct)
        # Recipe KB T0 defensive fallback for direct SDK/test callers; best-effort.
        self.phase_machine.ensure_recipe_kb_t0_anchored()

    def _open_persistence(self) -> None:
        """Open the session database and the bus, lease, task, cursor and sub-agent layers over it."""
        db_path, file_mode = database_target(self.session_dir, self.shared_state)
        # The session directory can sit on a networked filesystem where WAL's
        # shared-memory mapping corrupts the database, so the mode is resolved
        # before the file is opened.
        self.db = SqliteConnection(db_path, file_mode=file_mode)

        self.bus = MessageBus(self.db)
        self.locks = SqliteLeaseBackend(self.db, research_lane_capacity=self.shared_state.research_lane_capacity)
        self.tasks = TaskRegistry(
            self.db,
            dispatch_origin_provider=lambda: task_dispatch_origin(self.shared_state),
        )
        self.cursors = CursorStore(self.db)
        self.sub = SubAgentRunner(
            self.locks,
            self.tasks,
            session_dir=self.session_dir,
        )

    def _init_gpu_pools(self) -> None:
        """Carve the specialist and whole-node GPU pools."""
        # Serving-disjoint invariant: the live serving process holds the first ``serving_tp`` cards, carved off the
        # specialist pool.
        self.gpu_specialist_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_gpu_specialist_devices(
                int(self.shared_state.gpu_specialist_capacity or 0),
                serving_tp=serving_tp_for_policy(self.shared_state),
            ),
        )
        # Whole-node pool for enablement and bench specialists.
        self.framework_gpu_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_whole_machine_devices(),
        )
        # gpu_research_lane stays capacity-1 (strictly serial GPU specialists);
        # the GPU pool partitions physical cards within that one lease.

    @property
    def recipe_kb(self) -> RecipeKB | None:
        """RecipeKB owned by the knowledge plane."""
        plane = self.knowledge_plane
        return plane.recipe_kb if plane is not None else None

    @property
    def experience_kb(self) -> ExperienceKBIntegration | None:
        """The session's one Experience service integration, built on first use; None when reads are off."""
        if "_experience_kb" not in self.__dict__:
            from hyperloom.inference_optimizer.experience_kb import integration_for

            self.__dict__["_experience_kb"] = integration_for(self.shared_state)
        return self.__dict__["_experience_kb"]

    @property
    def run_deadline(self) -> Deadline | None:
        """Wall-clock deadline of the current run leg; None outside a run."""
        return self._run_deadline

    def target_gap_pct(self) -> float:
        """Percent improvement still needed to reach the run objective; 0.0 outside a run."""
        objective = self._current_objective
        return objective.gap_pct(self.shared_state) if objective is not None else 0.0

    @property
    def router(self) -> IntentRouter:
        """Intent routing collaborator (extracted from this class)."""
        return self._collaborator("_router", IntentRouter)

    def _collaborator(self, attr: str, factory):
        """Lazily build + cache a collaborator object; works for ``Coordinator.__new__`` test doubles too (uses ``__dict__``)."""
        obj = self.__dict__.get(attr)
        if obj is None:
            obj = factory(self)
            self.__dict__[attr] = obj
        return obj

    # Phase handlers, in call-chain order.
    @property
    def phase_machine(self):
        from ..phases.machine import MachinePhase

        return self._collaborator("_phase_machine", MachinePhase)

    @property
    def phase_prelude(self):
        from ..phases.prelude import PreludePhase

        return self._collaborator("_phase_prelude", PreludePhase)

    @property
    def phase_cycle_end(self):
        from ..phases.cycle_end import CycleEndPhase

        return self._collaborator("_phase_cycle_end", CycleEndPhase)

    @property
    def phase_close(self):
        from ..phases.close import ClosePhase

        return self._collaborator("_phase_close", ClosePhase)

    @property
    def phase_internal(self):
        from ..phases.internal import InternalTasksPhase

        return self._collaborator("_phase_internal", InternalTasksPhase)

    @property
    def phase_kernel_stack(self):
        from ..phases.kernel_stack import KernelStackPhase

        return self._collaborator("_phase_kernel_stack", KernelStackPhase)

    @property
    def phase_kernel(self):
        from ..phases.kernel import KernelPhase

        return self._collaborator("_phase_kernel", KernelPhase)

    @property
    def phase_macro_cycle(self):
        from ..phases.macro_cycle import MacroCycleCollaborator

        return self._collaborator("_phase_macro_cycle", MacroCycleCollaborator)

    @property
    def specialist_dispatch(self):
        from ..specialists.dispatch import SpecialistDispatchCollaborator

        return self._collaborator("_specialist_dispatch", SpecialistDispatchCollaborator)

    @property
    def gap_refresh(self):
        from .gap_refresh import GapRefreshCollaborator

        return self._collaborator("_gap_refresh", GapRefreshCollaborator)

    @property
    def phase_framework(self):
        from ..phases.framework import FrameworkPhase

        return self._collaborator("_phase_framework", FrameworkPhase)

    @property
    def gpu_lanes(self):
        """GPU-lease params and lane resolution, shared by both dispatchers."""
        from ..gpu_lanes import GpuLanes

        return self._collaborator("_gpu_lanes", GpuLanes)

    @property
    def enablement_params(self):
        """Enablement authoring-specialist request construction."""
        from ..enablement.params import EnablementParams

        return self._collaborator("_enablement_params", EnablementParams)

    @property
    def enablement_lane(self):
        """Enablement round admission / in-flight / re-arm."""
        from ..enablement.lane import EnablementLane

        return self._collaborator("_enablement_lane", EnablementLane)

    @property
    def enablement_build(self):
        """Off-loop compiled-build escalation and outcome routing."""
        from ..enablement.build import EnablementBuild

        return self._collaborator("_enablement_build", EnablementBuild)

    @property
    def enablement_revalidation(self):
        """Genuine-baseline revalidation of a kept enablement round."""
        from ..enablement.revalidation import EnablementRevalidation

        return self._collaborator("_enablement_revalidation", EnablementRevalidation)

    @property
    def reconciler(self):
        """The unconditional repair pass run at the top of every tick.

        Not a collaborator: it takes its dependencies explicitly so the rules
        can be exercised against a bare database.
        """
        r = self.__dict__.get("_reconciler")
        if r is None:
            from ..bringup.reconcile import ReconcileLinks, Reconciler, ReconcileStores

            r = Reconciler(
                ReconcileStores(rounds=self.rounds, tasks=self.tasks, locks=self.locks),
                self.shared_state,
                links=ReconcileLinks(
                    resources=self.policy.resources,
                    session_dir=self.session_dir,
                    # The window a revalidation holds is closed by the writeback when
                    # its result is booked; this pass only closes one nobody will.
                    result_pending=lambda task_id: self.dispatcher.owes_result(task_id),
                ),
            )
            self.__dict__["_reconciler"] = r
        return r

    @property
    def conversation(self):
        from .conversation import ConversationCollaborator

        return self._collaborator("_conversation", ConversationCollaborator)

    @property
    def proposals(self):
        from .proposals import ProposalsCollaborator

        return self._collaborator("_proposals", ProposalsCollaborator)

    @property
    def dispatcher(self):
        from .dispatcher import DispatcherCollaborator

        return self._collaborator(
            "_dispatcher",
            lambda coordinator: DispatcherCollaborator(
                coordinator, refresh_admission_freeze=self._refresh_admission_freeze_unless_closing
            ),
        )

    async def _refresh_admission_freeze_unless_closing(self) -> None:
        # CLOSE's closing report is dispatched by the pump, which a freeze would hold back.
        if not self.shared_state.closing_phase:
            await self.phase_machine.refresh_admission_freeze()

    @property
    def writeback(self):
        from .writeback import WritebackCollaborator

        return self._collaborator("_writeback", WritebackCollaborator)

    @property
    def recipe_journal(self):
        """Optimization journal, Recipe KB facts and the final Recipe."""
        from ..knowledge.recipe_journal import RecipeJournalCollaborator

        return self._collaborator("_recipe_journal", RecipeJournalCollaborator)

    @property
    def maintenance(self):
        from .maintenance import MaintenanceCollaborator

        return self._collaborator("_maintenance", MaintenanceCollaborator)

    @property
    def build_lifecycle(self):
        from .build_lifecycle import BuildLifecycleCollaborator

        return self._collaborator("_build_lifecycle", BuildLifecycleCollaborator)

    def _pin_source_trees(self) -> None:
        """Pin the source trees this session observes and patches, once at boot.

        Every failure digest is keyed on a frame normalised against these roots,
        so the pin must not move once the first attempt has run. A host with no
        framework tree on disk pins an empty set.
        """
        from ..bringup import resolve_trees, write_trees

        write_trees(resolve_trees(), session_dir=self.session_dir)

    # Lifecycle
    async def stop(self) -> None:
        """Signal shutdown, cancel in-flight work, and close the DB."""
        self._stop.set()
        try:
            await self.dispatcher.cancel_inflight_actions(reason="coordinator_stop")
        finally:
            await self.dispatcher.close_db_after_executions()
            if self._signals is not None:  # Closed after teardown: a later SIGTERM must not kill the process mid-stop.
                self._signals.close()

    def _bind_session_deadline(
        self,
        *,
        max_minutes: float | None,
        closing_grace_sec: float | None,
    ) -> tuple[float, Deadline, float]:
        """Open this process's leg and derive the instant its budget runs out.

        The stop instant is the session budget minus what previous legs charged,
        so a resumed run gets what is left rather than a second full allowance.
        An unbounded session gets the container cap.

        Args:
            max_minutes: Operator wall-clock budget, or ``None``/0 for unbounded.
            closing_grace_sec: Operator CLOSE window; ``None`` derives a default.

        Returns:
            ``(grace_sec, deadline, max_minutes_value)``.
        """
        grace_sec = effective_closing_grace_sec(max_minutes, closing_grace_sec)
        self.shared_state.closing_grace_sec = closing_grace_sec
        max_minutes_value = max_minutes if max_minutes is not None else 0
        self.shared_state.begin_leg()
        if max_minutes:
            # Store the budget before deriving the deadline: a leg that starts
            # with a smaller ``--max-hours`` than the session was given must run
            # against the smaller one, which only tightens.
            self.shared_state.max_minutes = int(max_minutes)
            self.shared_state.save(self.session_dir)
            deadline = self.shared_state.session_deadline()
            if deadline is None:
                # ``max_minutes`` persists as an int, so a sub-minute budget
                # truncates to 0 and reads as unbounded. It is spent, not absent.
                deadline = Deadline.after(0.0)
        else:
            if max_minutes is not None:
                self.shared_state.max_minutes = int(max_minutes)
            self.shared_state.save(self.session_dir)
            deadline = Deadline.after(_phase_state.DEFAULT_LONGRUN_MAX_MINUTES * 60.0)
        self._run_deadline = deadline
        return grace_sec, deadline, float(max_minutes_value)

    async def _recipe_kb_t4_hook(self) -> None:
        """Finalize or retry on graceful teardown/Ctrl-C."""
        if bool(getattr(self.knowledge_plane, "kb_disabled", False)):
            return
        config = getattr(self.knowledge_plane, "config", None) or KnowledgeConfig.from_env()
        if config.mode is KnowledgeStoreMode.LOCAL:
            if self.recipe_kb is None:
                return
            sid = (self.shared_state.recipe_kb_session_id or "").strip()
            if not sid:
                return
        self.recipe_journal.ensure_recipe_finalized(source="t4_fallback")
        try:
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception("recipe KB T4 SharedState.save failed")

    async def tick(self, n: int = 1) -> None:
        """Run ``n`` ticks of the run() loop body without its deadline or teardown."""
        await self.writeback.replay_resume_if_needed()
        for _ in range(n):
            await self._tick_once()

    async def _tick_once(self) -> bool:
        """Run one tick of the loop body; returns whether the session is in its closing phase."""
        # Repair before anything is admitted: a round nobody will settle, a task row with no process. Ungated,
        # because a stuck round closes every gate this could sit behind.
        await self.reconciler.run(time.time())
        # Bump the persistent tick counter — drives phase/plateau math.
        self.shared_state.increment_tick()
        # A phase-entry hook may have finished early (for example a GEAK no_gain run exits KERNEL immediately).
        await self.await_within_session_bound(
            self.phase_machine.advance_phase_if_needed,
            stage="advance_phase_pre_reactor",
        )
        in_closing = bool(self.shared_state.closing_phase)
        # One reactor + dispatcher pass; during closing skip LLM passes, and any turn that would only hold dispatch back.
        if not in_closing:
            if (
                not self.stop_requested()
                and time.monotonic() >= self._backend_retry_at
                and await self.conversation.admit_turn(time.time())
            ):
                await self.await_within_session_bound(self.reactor_pass, stage=f"reactor:{ORCHESTRATION_AGENT}")
            await self.phase_machine.refresh_admission_freeze()
        if not self.stop_requested():
            await self.dispatcher.pump_dispatcher_once()
        if not in_closing:
            await self.phase_machine.pump_current_phase()
            # Phase-independent enablement pump: repair a non-runnable combo.
            await self.enablement_lane.pump_enablement_safely()
        # phase machine advance; runs even in_closing so CLOSE is recorded.
        try:
            await self.await_within_session_bound(self.phase_machine.advance_phase_if_needed, stage="advance_phase")
        except Exception as exc:
            log.exception("phase advance failed")
            self.record_exception(stage="advance_phase", exc=exc)
        # Periodic reaper + DB retention; time-gated.
        now = time.monotonic()
        if now - self._last_maintenance_ts >= MAINTENANCE_INTERVAL_SEC:
            await self.maintenance.run(tick=self.shared_state.tick)
            self._last_maintenance_ts = now
        return in_closing

    def record_exception(self, *, stage: str, exc: BaseException, agent: str = "") -> None:
        """Record a Coordinator-side exception without killing the session."""
        self._fault_open_phase_event(stage=stage, exc=exc)
        try:
            self.shared_state.record_tick_exception(
                tick=int(self.shared_state.tick or 0),
                stage=stage,
                agent=agent,
                exc=exc,
            )
            self.shared_state.increment_crash_count()
            self.shared_state.save(self.session_dir)
        except OSError:
            log.exception("failed to persist Coordinator exception metadata")

    def _fault_open_phase_event(self, *, stage: str, exc: BaseException) -> None:
        """Name this exception on the phase event it struck, if one is open.

        Every other timeline event is closed on the exception by the executor
        that raised it. A KERNEL visit and a FRAMEWORK entry have no such frame
        -- each spans the ticks the machine sits in its phase, and is closed
        when that span ends -- so the exceptions swallowed here were the ones
        that reached their event nowhere, leaving it to close clean and report
        an outcome for a phase that had blown up.

        Both recorders are absent unless their phase is the one running, which
        is what keeps a fault on the event that was open for it. The enablement
        lane is deliberately not here: it spans the whole session, so it is
        open for every exception and is the thing that raised for almost none
        of them. Its own pump records what it is responsible for.
        """
        for recorder in (self.phase_kernel.timeline(), self.phase_framework.timeline()):
            if recorder is None:
                continue
            recorder.record_fault(stage=stage, exc=exc)

    def _seconds_until_session_bound(self) -> float | None:
        """Seconds left on the active run or closing bound; ``None`` if unbounded."""
        if bool(self.shared_state.closing_phase):
            bound = self._closing_deadline
        else:
            bound = self._run_deadline
        if bound is None:
            return None
        return bound.remaining()

    def _backend_retry_wait_sec(self) -> float:
        """Seconds until the backend's back-off ends, capped at the session bound."""
        wait = self._backend_retry_at - time.monotonic()
        bound = self._seconds_until_session_bound()
        if bound is not None:
            wait = min(wait, bound)
        return max(0.0, wait)

    def _stage_timeout_sec(self, stage: str) -> float | None:
        """Return the total wall-clock ceiling for an inline reactor turn."""
        role = stage.removeprefix("reactor:")
        if role == stage:
            return None
        return self.reactor_turn_timeout_sec

    def stop_requested(self) -> bool:
        """Whether an operator has asked this run to stop.

        Reads the asyncio event and the drain's threading event, so the end-of-tick check sees a signal that arrived
        during that tick.

        Returns:
            bool: True once a stop has been asked for by either route.
        """
        if self._stop.is_set():
            return True
        drain = self._signals
        return drain is not None and drain.requested.is_set()

    def _signal_stop_reason(self) -> str:
        """Classify the stop from the signal captured by the drain."""
        drain = self._signals
        received = frozenset() if drain is None else frozenset(drain.received)
        return self._classify_stop(received) or "signal"

    def _classify_stop(self, received: AbstractSet[int], *, pending: str = "") -> str:
        """Classify final state from terminal outcome and captured signals."""
        if received:
            return "signal"
        return pending or self.shared_state.stop_reason

    @property
    def stop_classification(self) -> str:
        """Authoritative in-process stop classification."""
        return self._stop_classification

    async def await_within_session_bound(
        self,
        factory: Callable[[], Awaitable[Any]],
        *,
        stage: str,
    ) -> None:
        """Run one tick step, cancelling it when the session/closing bound elapses."""
        remaining = self._seconds_until_session_bound()
        if remaining is not None and remaining <= 0.0:
            log.warning("Coordinator: skipping %s; session bound already elapsed", stage)
            return
        stage_timeout = self._stage_timeout_sec(stage)
        timeout = (
            min(remaining, stage_timeout)
            if remaining is not None and stage_timeout is not None
            else remaining
            if stage_timeout is None
            else stage_timeout
        )
        if timeout is None:
            await factory()
            return
        try:
            await asyncio.wait_for(factory(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            log.warning(
                "Coordinator: %s hit its %.1fs bound; cancelled so the tick can close",
                stage,
                timeout,
            )
            if stage_timeout is not None and timeout == stage_timeout:
                self.record_exception(stage=stage, exc=exc, agent=stage.removeprefix("reactor:"))

    # Long-run interface
    async def run(self, *, objective: Objective | None = None, options: RunOptions = RunOptions()) -> str:
        """Run reactor + dispatcher until a stop condition fires (priority order): signal, a stop_reason the phase machine recorded (a met target closes through CYCLE_END as one), time_exhausted (via closing phase), emergency, max_ticks. Sets + saves + returns shared_state.stop_reason.

        The whole run is one ``session`` trajectory span, and every trajectory event recorded beneath it inherits this
        session dir and the live phase / tick.
        """
        with (
            trajectory_scope(
                session_dir=self.session_dir,
                component="coordinator",
                phase_tick_source=self._trajectory_phase_tick,
            ),
            trajectory_span(EVENT_SESSION, attributes={"name": self.session_dir.name}) as span,
        ):
            stop_reason = await self._run_ticks(objective=objective, options=options)
            span.finish(stop_reason=stop_reason)
            return stop_reason

    def _trajectory_phase_tick(self) -> tuple[str | None, int | None]:
        """Live ``(phase, tick)`` for trajectory events recorded inside :meth:`run`."""
        return (self.shared_state.phase or None), int(self.shared_state.tick or 0)

    def _budget_stop_reason(self, options: RunOptions, *, ticks: int) -> str:
        """The reason the crash-rate and tick-count budgets ask the loop to stop for, or ``""``."""
        crashes = self.shared_state.recent_crash_count(window_sec=_CRASH_EMERGENCY_WINDOW_SEC)
        if crashes >= _CRASH_EMERGENCY_THRESHOLD:
            return "emergency"
        if options.max_ticks is not None and ticks >= options.max_ticks:
            return "max_ticks"
        return ""

    async def _closing_phase_spent(self, *, grace_sec: float) -> bool:
        """True once the closing report has reached a terminal state or its grace window is gone."""
        closing_deadline = self._closing_deadline
        report_terminal = await self.phase_close.closing_report_terminal()
        grace_blown = closing_deadline is not None and closing_deadline.expired()
        if grace_blown and not report_terminal:
            log.warning(
                "Coordinator: closing-grace exhausted (%.0fs) before report task %s finished",
                grace_sec,
                self.shared_state.closing_report_task_id,
            )
        if report_terminal or grace_blown:
            return True
        # Wait on the running work: no reactor turn paces the closing ticks.
        poll_sec = self.dispatcher.poll_sec
        await self.dispatcher.wait_for_running_work(
            timeout=min(poll_sec, seconds_until(closing_deadline, unbounded_cap=poll_sec))
        )
        return False

    async def _close_run(self, stop_reason: str, *, ticks: int, tick_failed: bool, max_minutes: float) -> str:
        """Classify the stop, persist it, and release everything the run still holds."""
        final_signals = frozenset() if self._signals is None else frozenset(self._signals.received)
        stop_reason = self._classify_stop(final_signals, pending=stop_reason)
        self._stop_classification = stop_reason
        self.shared_state.closing_phase = False
        # Resuming a terminal session can break out before stop_reason is set.
        self.shared_state.set_stop_reason(
            stop_reason or self.shared_state.stop_reason or ("coordinator_exception" if tick_failed else "unknown")
        )
        self.shared_state.save(self.session_dir)
        close_error: BaseException | None = None
        try:
            await self.phase_close.ensure_close_sequence(reason=self.shared_state.stop_reason)
        except (asyncio.CancelledError, Exception) as exc:  # re-raised once the run's resources are released
            log.exception("Coordinator: terminal close sequence did not finish")
            close_error = exc
        await self._recipe_kb_t4_hook()
        log.info(
            "Coordinator.run: stopped tick=%d reason=%s baseline_tput=%.1f "
            "cumulative_gain_validated=%.2f%% max_minutes=%.0f",
            ticks,
            stop_reason or "unknown",
            self.shared_state.baseline_tput,
            self.shared_state.cumulative_gain_validated,
            max_minutes,
        )
        await close_backends(self.backends)
        # A server outliving the run holds every GPU it was given, so the
        # session's last act is to reap its own pidfiles.
        await asyncio.to_thread(reap_orphaned_servers_best_effort, self.session_dir, phase="shutdown")
        self.shared_state.save(self.session_dir)
        if close_error is not None:
            raise close_error
        return stop_reason

    async def _run_ticks(self, *, objective: Objective | None, options: RunOptions) -> str:
        """Tick loop and shutdown sequence behind :meth:`run`."""
        objective = objective or TimeOnlyObjective()
        self._current_objective = objective
        # A dedicated thread reading the interpreter's wakeup pipe, not a loop
        # callback: a TERM has to be recorded while the loop is busy.
        if options.install_signal_handlers:
            self._signals = SignalDrain(
                loop=asyncio.get_running_loop(),
                stop_event=self._stop,
            )
            log.info("Coordinator.run: stop-signal drain armed=%s", self._signals.armed)

        await self.writeback.replay_resume_if_needed()
        grace_sec, deadline, max_minutes_value = self._bind_session_deadline(
            max_minutes=options.max_minutes,
            closing_grace_sec=options.closing_grace_sec,
        )

        ticks = 0
        stop_reason = ""
        tick_failed = False
        try:
            while not stop_reason:
                ticks += 1
                in_closing = bool(self.shared_state.closing_phase)
                try:
                    in_closing = await self._tick_once()
                except Exception as exc:
                    tick_failed = True
                    log.exception("Coordinator.run: tick %d body raised", self.shared_state.tick)
                    self.record_exception(stage="tick_body", exc=exc)
                if self.stop_requested():
                    stop_reason = self._signal_stop_reason()
                    break
                if self.shared_state.stop_reason and not in_closing:
                    stop_reason = self.shared_state.stop_reason
                    break
                if objective.reached(self.shared_state) and not self.shared_state.target_reached_at:
                    # The phase machine reads the marker; the transition it makes next persists it.
                    self.shared_state.target_reached_at = now_iso()
                if deadline.expired() and not in_closing:
                    if grace_sec <= 0:
                        stop_reason = "time_exhausted"
                        break
                    self._closing_deadline = await self.phase_close.enter_closing_phase(grace_sec=grace_sec)
                    continue
                if in_closing and await self._closing_phase_spent(grace_sec=grace_sec):
                    stop_reason = "time_exhausted"
                    break
                stop_reason = self._budget_stop_reason(options, ticks=ticks)
                if stop_reason:
                    break
                # A tick dispatches and books tasks whether or not an agent takes a turn in it, so a backend's
                # back-off never holds the next tick off for longer than the dispatcher polls.
                retry_wait_sec = 0.0 if in_closing else min(self._backend_retry_wait_sec(), self.dispatcher.poll_sec)
                wait_sec = max(options.tick_interval_sec, retry_wait_sec)
                if wait_sec > 0:
                    with contextlib.suppress(asyncio.TimeoutError):
                        await asyncio.wait_for(self._stop.wait(), timeout=wait_sec)
                        stop_reason = self._signal_stop_reason()
        finally:
            stop_reason = await self._close_run(
                stop_reason, ticks=ticks, tick_failed=tick_failed, max_minutes=max_minutes_value
            )
        return stop_reason or self.shared_state.stop_reason

    # Reactor
    async def reactor_pass(self, *, request: str = "") -> BackendTurnResult | None:
        """Run one Orchestration turn, route its intents, and return the turn's result.

        ``request`` is appended to the composed prompt; a turn that carries one may legitimately reply without
        intents. The turn is scoped as the agent on the trajectory, and the dispatcher pumps beside its LLM call.
        """
        pump = PumpBesideTurn(self.dispatcher, self.stop_requested)
        with trajectory_scope(component=ORCHESTRATION_AGENT, agent=ORCHESTRATION_AGENT):
            return await self._reactor_turn(request=request, pump=pump)

    async def _reactor_turn(self, *, request: str, pump: PumpBesideTurn) -> BackendTurnResult | None:
        """Body of :meth:`reactor_pass`."""
        backend = self.backends[ORCHESTRATION_AGENT]
        turn = await self.conversation.compose_turn(backend)
        prompt = f"{turn.prompt}\n\n{request}" if request else turn.prompt
        tools = list(ORCHESTRATION_TOOL_GRANT)
        _t0 = time.perf_counter()
        call_id = new_call_id()
        try:
            with trajectory_span(
                EVENT_LLM_CALL,
                call_id=call_id,
                attributes={"name": ORCHESTRATION_AGENT, "model": getattr(backend, "model", None)},
            ) as call_span:
                record_prompt_snapshot(
                    self._prompt_snapshots.observe(
                        ORCHESTRATION_AGENT, prompt=prompt, system_prompt=turn.system_prompt, tools=tools
                    )
                )
                async with pump:
                    result: BackendTurnResult = await backend.run(
                        prompt=prompt, system_prompt=turn.system_prompt, tools=tools
                    )
                call_span.finish(**llm_call_summary(result.metadata))
        except BackendError as exc:
            if isinstance(exc, LLMCallFailed):
                self._trace_reactor_llm_failure(
                    exc,
                    latency_ms=int((time.perf_counter() - _t0) * 1000),
                    call_id=call_id,
                )
            await self.bus.record_observation(
                "coordinator",
                "observation",
                {"kind": "backend_error", "agent": ORCHESTRATION_AGENT, "error": repr(exc)},
            )
            await self._track_backend_error_streak(exc)
            return None
        except NoIntentEmitted as exc:
            # No parseable intents; surface as observation so the next tick self-corrects.
            await self._record_no_intent_turn(str(exc)[:500])
            await self.conversation.settle_turn()
            return None
        except Exception as exc:  # isolation boundary: one reactor turn
            log.exception("reactor pass for %s raised", ORCHESTRATION_AGENT)
            await self._record_reactor_exception(exc)
            return None
        finally:
            trace_mcp_setup(self.session_dir, backend)
        self._reset_backend_error_streak()
        # Record this reactor turn's token spend on the unified ledger.
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        trace_reactor_llm_call(self.session_dir, self.shared_state, result, latency_ms=latency_ms)
        # Full-trace: persist the redacted prompt+response for this turn.
        self.conversation.record_reactor_conversation(result, context=turn.context)
        with trajectory_scope(call_id=call_id, parent_span_id=call_span.span_id):
            for intent in result.intents:
                await self.router.handle_intent(ORCHESTRATION_AGENT, intent)
        await self._settle_routed_turn(result, request=request)
        return result

    async def _record_reactor_exception(self, exc: Exception) -> None:
        """Record an unexpected reactor-turn failure as an observation and a session exception."""
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {"kind": "reactor_exception", "agent": ORCHESTRATION_AGENT, "error": format_exc_brief(exc, limit=500)},
        )
        self.record_exception(
            stage="reactor_pass",
            agent=ORCHESTRATION_AGENT,
            exc=exc,
        )

    def _reset_backend_error_streak(self) -> None:
        """Clear the backend-error streak and re-arm its alarm: a successful turn proves the backend is alive again."""
        self._backend_error_streak = 0
        self._backend_error_alarm_armed = True
        self._backend_error_since = None
        self._backend_retry_at = 0.0

    async def _settle_routed_turn(self, result: BackendTurnResult, *, request: str) -> None:
        """Book a turn whose intents were routed: its no-intent streak and rendered cursor."""
        if result.intents:
            self._no_intent_streak = 0
        elif not request:
            await self._record_no_intent_turn("the turn emitted no intents")
        await self.conversation.settle_turn()

    def _trace_reactor_llm_failure(
        self,
        error: LLMCallFailed,
        *,
        latency_ms: int | None = None,
        call_id: str | None = None,
    ) -> None:
        """Append one ``status=\"error\"`` ``llm_calls.jsonl`` row for a failed turn."""
        try:
            site = LLMCallSite(
                session_id=self.session_dir.name,
                component=ORCHESTRATION_AGENT,
                role=ORCHESTRATION_AGENT,
                tick=int(self.shared_state.tick or 0),
                phase=(self.shared_state.phase or "") or None,
            )
            model = getattr(self.backends[ORCHESTRATION_AGENT], "model", None)
            record = LLMCallRecord.for_failure(site, error, model=model, call_id=call_id, latency_ms=latency_ms)
            append_llm_call(session_dir=self.session_dir, record=record)
        except LLMTraceRowError:
            log.debug("full-trace: reactor llm_call failure append failed", exc_info=True)

    async def _track_backend_error_streak(self, exc: BackendError) -> None:
        """Extend the ``BackendError`` streak and back the next turn off.

        A streak that has lasted ``_BACKEND_UNHEALTHY_STOP_SEC`` stops the session unless something else already has.
        Crossing the threshold emits one backend_unhealthy event, which re-arms only after a successful turn.
        """
        from hyperloom.inference_optimizer.breakdown.stop_reasons import BACKEND_UNHEALTHY_STOP_REASON

        now = time.monotonic()
        self._backend_error_streak += 1
        streak = self._backend_error_streak
        log.warning("Coordinator: %s backend turn failed (%d in a row): %s", ORCHESTRATION_AGENT, streak, exc)
        if self._backend_error_since is None:
            self._backend_error_since = now
        failing_sec = now - self._backend_error_since
        self._backend_retry_at = now + min(_BACKEND_RETRY_MAX_SEC, _BACKEND_RETRY_BASE_SEC * 2 ** (streak - 1))
        if failing_sec >= _BACKEND_UNHEALTHY_STOP_SEC and not self.shared_state.stop_reason:
            log.error(
                "Coordinator: the %s backend has failed for %.0fs without a successful turn",
                ORCHESTRATION_AGENT,
                failing_sec,
            )
            self.shared_state.set_stop_reason(BACKEND_UNHEALTHY_STOP_REASON)
        threshold = self._backend_error_streak_threshold
        if streak >= threshold and self._backend_error_alarm_armed:
            self._backend_error_alarm_armed = False
            await self.bus.record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "backend_unhealthy",
                    "agent": ORCHESTRATION_AGENT,
                    "consecutive_errors": streak,
                    "threshold": threshold,
                    "latest_error": repr(exc)[:500],
                    "severity": "high",
                    "hint": "the backend has failed >= threshold times consecutively; repair its transport",
                },
            )

    async def _record_no_intent_turn(self, error: str) -> None:
        """Record a turn that emitted no intent and extend the no-intent streak.

        The ``_NO_INTENT_STOP_TURNS``-th such turn in a row records one high-severity no_intent_streak observation and
        stops the session unless something else already has.
        """
        from hyperloom.inference_optimizer.breakdown.stop_reasons import BACKEND_UNHEALTHY_STOP_REASON

        await self.bus.record_observation(
            "coordinator",
            "observation",
            {"kind": "no_intent_emitted", "agent": ORCHESTRATION_AGENT, "error": error},
        )
        self._no_intent_streak += 1
        streak = self._no_intent_streak
        if streak != _NO_INTENT_STOP_TURNS:
            return
        log.error("Coordinator: the %s agent answered %d turns in a row without an intent", ORCHESTRATION_AGENT, streak)
        await self.bus.record_observation(
            "coordinator",
            "observation",
            {
                "kind": "no_intent_streak",
                "agent": ORCHESTRATION_AGENT,
                "consecutive_turns": streak,
                "severity": "high",
                "hint": (
                    "the agent keeps answering without calling emit_intent; for the orchestration agent, "
                    "agents/orchestration/mcp_setup.json records whether Claude Code offered the tool"
                ),
            },
        )
        if not self.shared_state.stop_reason:
            self.shared_state.set_stop_reason(BACKEND_UNHEALTHY_STOP_REASON)


__all__ = ["Coordinator"]
