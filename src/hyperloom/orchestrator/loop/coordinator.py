# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import AbstractSet, Any, Awaitable, Callable

from hyperloom.common.env import env_bool, env_flag
from hyperloom.common.timeutil import now_iso
from hyperloom.orchestrator.knowledge.config import KnowledgeConfig, KnowledgeStoreMode
from hyperloom.orchestrator.knowledge.recipe_kb import RecipeKB


# Periodic in-process maintenance/reaper cadence (lease reaping + DB retention), in wall-clock seconds.
MAINTENANCE_INTERVAL_SEC: int = 1800

# Default per-macro-cycle wall-clock window (hours) in cyclic mode.
DEFAULT_CYCLE_HOURS: float = 24.0
# Trailing window for the crash-rate emergency stop, in seconds.
_CRASH_EMERGENCY_WINDOW_SEC: float = 24.0 * 3600.0
from ..phases import machine_state as _phase_state
from ..state.optimization_journal import Journal
from hyperloom.inference_optimizer.session.paths import db_path_for
from hyperloom.inference_optimizer.session.session_binding import bind_session
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE, ActionMetadata
from ..roles.agent_role import AgentRole, default_role_registry
from ..roles.base import Backend, BackendError, BackendTurnResult, LLMCallFailed
from ..bus.cursor_store import CursorStore
from ..bus.storage.connection import SqliteConnection, resolve_journal_mode
from hyperloom.inference_optimizer.protocol.intent import NoIntentEmitted
from ..bus.message_bus import MessageBus
from ..state.objective import Objective, TimeOnlyObjective
from ..policy.gate import (
    PolicyGate,
    SPECIALIST_FROM_AGENT_PREFIX,
)
from ..state.round_store import RoundStore
from ..bus.gpu_pool import (
    SpecialistGpuPool,
    resolve_gpu_specialist_devices,
    resolve_whole_machine_devices,
)
from ..bus.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from ..state.shared_state import SharedState, effective_closing_grace_sec, timed_teardown_step
from .signals import SignalDrain
from .intent_router import IntentRouter
from ..phases.machine import MachinePhase
from ..phases.prelude import PreludePhase
from ..phases.sweep import SweepPhase
from ..phases.close import ClosePhase
from ..phases.internal import InternalTasksPhase
from ..phases.kernel_stack import KernelStackPhase
from ..phases.kernel import KernelPhase
from ..phases.macro_cycle import MacroCycleCollaborator
from .cycle_memory import CycleMemoryCollaborator
from ..specialists.dispatch import SpecialistDispatchCollaborator
from ..state.gaps import GapRefreshCollaborator
from ..phases.framework import FrameworkPhase
from ..gpu_lanes import GpuLanes
from ..enablement.params import EnablementParams
from ..enablement.lane import EnablementLane
from ..enablement.build import EnablementBuild
from ..enablement.revalidation import EnablementRevalidation
from .maintenance import MaintenanceCollaborator
from .build_lifecycle import BuildLifecycleCollaborator
from .writeback import WritebackCollaborator
from .dispatcher import DispatcherCollaborator
from .proposals import ProposalsCollaborator
from .conversation import ConversationCollaborator
from .sub_agent_runner import SubAgentRunner
from ..state.task_registry import TaskRegistry
from ..trace.llm_trace import LLMCallRecord, append_llm_call
from hyperloom.common.deadline import Deadline
from ..trace.orchestration_trace import (
    write_mcp_setup_once,
)
from .coordinator_shared import _AUTHORED_LANE_MAX_ATTEMPTS, PendingProposal
from .coordinator_helpers import (
    _infer_model_class_from_config,
    format_exc_brief,
    resolve_reactor_turn_timeout_sec,
)


log = logging.getLogger(__name__)


def _resolvable_artifacts_from_done(
    done_payload: dict[str, Any] | None,
    resolve_bases: list[Path],
) -> list[dict[str, Any]]:
    """Return ``artifacts_written`` entries whose ``source`` file exists on disk."""
    if not isinstance(done_payload, dict):
        return []
    arts = done_payload.get("artifacts_written")
    if not isinstance(arts, list):
        return []
    out: list[dict[str, Any]] = []
    # Sandbox bases are invariant across entries — resolve once.
    bases_resolved = [base.resolve() for base in resolve_bases]
    for entry in arts:
        if not isinstance(entry, dict):
            continue
        src = str(entry.get("source") or "").strip()
        tgt = str(entry.get("target") or "").strip()
        if not src or not tgt:
            continue
        raw = Path(src)
        # An absolute ``source`` is checked as-is; a relative one is resolved under each base.
        cands = [raw] if raw.is_absolute() else [base / raw for base in resolve_bases]
        for cand in cands:
            resolved = cand.resolve()
            if not resolved.is_file():
                continue
            contained = False
            for base in bases_resolved:
                try:
                    resolved.relative_to(base)
                except ValueError:
                    continue
                contained = True
                break
            if contained:
                out.append(entry)
                break
    return out


# Path-like keys surfaced from a kernel handler payload/result so operators can see where a step's artifacts went.
_LIFECYCLE_PATH_KEYS: tuple[str, ...] = (
    "trace_input",
    "trace_dir",
    "candidates_path",
    "analysis_md_path",
    "kernel_candidates",
    "best_artifact_path",
    "patch_path",
    "target_file",
    "workspace",
    "workspace_path",
    "out_dir",
    "output_dir",
    "run_dir",
    "report_path",
    "json_path",
    "md_path",
    "tracelens_agent_report",
    # TraceLens analysis outputs surfaced by trace_analyze_handler.
    "trace_report_path",
    "analysis_report_path",
    "tracelens_summary_path",
    "kernel_roofline_path",
    "cli_log_path",
)


def _lifecycle_paths(payload: Any) -> dict[str, str]:
    """Extract present, non-empty path-like fields from a kernel handler payload or result dict."""
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for key in _LIFECYCLE_PATH_KEYS:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val
    return out


@dataclass
class CoordinatorState:
    """In-memory ephemeral state for the reactor + dispatcher."""

    pending_proposals: dict[str, PendingProposal] = field(default_factory=dict)


class Coordinator(
    MachinePhase,
    PreludePhase,
    SweepPhase,
    ClosePhase,
    InternalTasksPhase,
    KernelStackPhase,
    KernelPhase,
    MacroCycleCollaborator,
    CycleMemoryCollaborator,
    SpecialistDispatchCollaborator,
    GapRefreshCollaborator,
    FrameworkPhase,
    GpuLanes,
    EnablementParams,
    EnablementLane,
    EnablementBuild,
    EnablementRevalidation,
    IntentRouter,
    MaintenanceCollaborator,
    BuildLifecycleCollaborator,
    WritebackCollaborator,
    DispatcherCollaborator,
    ProposalsCollaborator,
    ConversationCollaborator,
):
    """The single Coordinator instance per session."""

    def __init__(
        self,
        session_dir: Path,
        *,
        backends: dict[str, Backend],
        role_registry: dict[str, AgentRole] | None = None,
        sub_agent_runner: SubAgentRunner | None = None,
        bus_class: type[MessageBus] = MessageBus,
        model_class: str | None = None,
        recipe_kb: RecipeKB | None = None,
        phase_budget_pct: dict[str, float] | None = None,
        knowledge_plane: Any = None,
        proposal_scorer: Any = None,
        warm_replay_enabled: bool = True,
        warm_replay_min_confidence: float = 0.7,
        warm_replay_min_reproduce_pct: float = 0.8,
    ):
        """Construct the per-session Coordinator and wire persistence, policy, and agents."""
        self.session_dir = Path(session_dir)
        self._init_dispatch_state()
        # Bind the session for the SBD V6 recorders once, here, so no recorder entry point below has to be handed a
        # path.
        bind_session(self.session_dir)
        self.role_registry = role_registry or default_role_registry()
        # KnowledgePlane owns RecipeKB.
        plane_recipe_kb = getattr(knowledge_plane, "recipe_kb", None)
        self.recipe_kb: RecipeKB | None = plane_recipe_kb if plane_recipe_kb is not None else recipe_kb
        # Per-session optimization journal; lazy-instantiated on first use.
        self._journal: Journal | None = None
        # Warm-recipe replay controls (PRELUDE auto-apply of KB best_config).
        self._warm_replay_enabled: bool = bool(warm_replay_enabled)
        self._warm_replay_min_confidence: float = float(warm_replay_min_confidence)
        self._warm_replay_min_reproduce_pct: float = float(warm_replay_min_reproduce_pct)
        # KnowledgePlane facade; pre-warms PR feed + advisory context.
        self.knowledge_plane: Any = knowledge_plane
        # ProposalScorer facade (advisory only).
        self._proposal_scorer: Any = proposal_scorer
        # Phase budget percentages, normalised once at construction.
        self._phase_budget_pct: dict[str, float] = _phase_state.normalize_budget_pct(phase_budget_pct)
        self._model_class_override: str = (model_class or "").strip()

        # Validate every reactor has a backend wired.
        for name in self.role_registry:
            if name not in backends:
                raise ValueError(f"missing backend for role {name!r} (provide via Coordinator(backends={{...}}))")
        self.backends = dict(backends)
        self.reactor_turn_timeout_sec = resolve_reactor_turn_timeout_sec()

        # Persistence layer
        db_path = db_path_for(self.session_dir)
        # The session directory can sit on a networked filesystem where WAL's
        # shared-memory mapping corrupts the database, so the mode is resolved
        # before the file is opened.
        self.db = SqliteConnection(db_path, journal_mode=resolve_journal_mode())

        self.bus = bus_class(self.db)
        self.locks = ResourceLockManager(SqliteLeaseBackend(self.db))
        self.tasks = TaskRegistry(self.db)
        self.cursors = CursorStore(self.db)
        self.sub = sub_agent_runner or SubAgentRunner(
            self.locks,
            self.tasks,
            session_dir=self.session_dir,
        )

        # Persistent session state (state.json) — load existing for resume.
        self.shared_state = SharedState.load_or_init(self.session_dir)
        # Lifecycle save debounce: terminal events flush immediately; bursty non-terminal markers coalesce within a
        # short window.
        self._lifecycle_last_save: float = 0.0
        self._lifecycle_save_min_interval_s: float = 2.0
        # Thread live SharedState into the runner so executors get it via ctx.extra.
        self.sub.shared_state = self.shared_state
        # Serving-disjoint invariant: the live serving process holds the first ``serving_tp`` cards, carved off the
        # specialist pool.
        self.gpu_specialist_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_gpu_specialist_devices(
                int(getattr(self.shared_state, "gpu_specialist_capacity", 0) or 0),
                serving_tp=self._resolve_serving_tp(),
            ),
        )
        # Framework-authoring pool over the whole node.
        self.framework_gpu_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_whole_machine_devices(),
        )
        # Dispatcher re-scan poll cadence: re-scan the queue while awaiting in-flight tasks so a queued GPU task
        # starts the moment its lane frees.
        self._dispatcher_poll_sec = 10.0
        # Sync research_lane capacity into lane_capacity so acquire_many honours the cap.
        try:
            from ..bus.storage.schema import set_lane_capacity as _set_lane_capacity

            cap = int(self.shared_state.research_lane_capacity or 0)
            if cap >= 0:
                _set_lane_capacity(self.db.raw, "research_lane", cap)
        except Exception:
            log.exception("failed to sync research_lane_capacity to leases DB")
        # gpu_research_lane stays capacity-1 (strictly serial GPU specialists);
        # the GPU pool partitions physical cards within that one lease.
        # `strict_paths` defers to the env flag.
        # The durable bring-up mutex. The gate never reads it; ``open`` decides.
        self.rounds = RoundStore(self.db)
        self.policy = PolicyGate(
            role_registry=self.role_registry,
            session_dir=self.session_dir,
            shared_state=self.shared_state,
            # Seeded at boot: an empty snapshot would advise against a GPU
            # dispatch the pool can in fact satisfy.
        )
        self.sub.policy = self.policy
        # Attach read-only context-pull MCP tools to Orchestration backend.
        self._attach_orchestration_context_tools()
        # Resume detection must run before any boot-time state.json write.
        self._resumed_from = self._detect_resume_state()
        # Reap serving processes orphaned by a prior monitor-process crash (e.g. a raylet death that took the
        # optimizer down mid-benchmark), scoped strictly to this session's own pidfiles.
        self._reap_orphaned_servers_best_effort(phase="boot")
        # Before any bring-up can dispatch: an attempt classified against a
        # different pin is not comparable with its neighbours.
        self._pin_source_trees()
        # Derive model_class once at boot if not supplied; never overwrite a resume.
        if not (self.shared_state.model_class or "").strip():
            self.shared_state.model_class = self._model_class_override or _infer_model_class_from_config(
                self.shared_state.model_path or os.environ.get("MODEL_PATH", "")
            )
        self.state = CoordinatorState()
        self._stop = asyncio.Event()
        self._stop_classification = ""
        self._tasks_running: list[asyncio.Task] = []

        # Wall-clock stamp of the last maintenance pass (lease reaping + DB retention).
        self._last_maintenance_ts: float = time.monotonic()

        # Pin a per-macro-cycle budget window so per-phase budget fractions apply per cycle.
        if float(getattr(self.shared_state, "cycle_minutes", 0) or 0) <= 0:
            try:
                _cycle_hours = float(
                    os.environ.get(
                        "INFERENCE_OPTIMIZER_CYCLE_HOURS",
                        str(DEFAULT_CYCLE_HOURS),
                    )
                )
            except ValueError:
                _cycle_hours = DEFAULT_CYCLE_HOURS
            self.shared_state.cycle_minutes = max(1.0, _cycle_hours * 60.0)

        # Medium-intensity soft restart at each macro-cycle boundary.
        self._cycle_soft_restart: bool = not env_bool("INFERENCE_OPTIMIZER_DISABLE_CYCLE_SOFT_RESTART")
        # The soft restart's inference-server deep-clean kills lingering server processes; separately gated, defaults
        # ON within the soft restart.
        self._cycle_restart_servers: bool = not env_bool("INFERENCE_OPTIMIZER_DISABLE_CYCLE_SERVER_RESTART")

        # Per-agent (seq, msg_id) of the last message its prompt rendered.
        self._rendered_cursor: dict[str, tuple[int, str]] = {}

        # Per-agent BackendError streak; crossing threshold records one backend_unhealthy, then re-arms.
        self._backend_error_streak: dict[str, int] = {name: 0 for name in self.role_registry}
        self._backend_error_alarm_armed: dict[str, bool] = {name: True for name in self.role_registry}
        try:
            self._backend_error_streak_threshold: int = max(
                1,
                int(
                    os.environ.get(
                        "INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD",
                        "5",
                    )
                ),
            )
        except ValueError:
            self._backend_error_streak_threshold = 5

        # Stable tick order from the live role_registry.
        _CANONICAL_ORDER = ("orchestration", "critic")
        self._tick_roles: tuple[str, ...] = tuple(r for r in _CANONICAL_ORDER if r in self.role_registry)

        # Inline fast-action execution: run cheap lane-light action in-turn. Default ON.
        self._inline_fast_actions_enabled: bool = env_flag("INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS", default=True)
        self._coordinator_loop: asyncio.AbstractEventLoop | None = None
        # Wall-clock budget tracking for per-tick Time-budget prompt injection.
        self._run_deadline: Deadline | None = None
        self._run_started_monotonic: float | None = None
        # Closing-grace bound; used only while ``closing_phase`` is set so CLOSE
        # work is not skipped just because the session deadline has passed.
        self._closing_deadline: Deadline | None = None
        self._signals: SignalDrain | None = None
        # Latest objective wired by run(); refreshes target_gap_pct each tick. None outside a run.
        self._current_objective: Objective | None = None

        # Initialise phase machine (fresh session enters PRELUDE). Idempotent.
        self._ensure_phase_initialised()
        # Recipe KB T0 defensive fallback for direct SDK/test callers; best-effort.
        self._ensure_recipe_kb_t0_anchored()

    @property
    def reconciler(self):
        """The unconditional repair pass run at the top of every tick.

        Not a collaborator: it takes its dependencies explicitly so the rules
        can be exercised against a bare database.
        """
        r = self.__dict__.get("_reconciler")
        if r is None:
            from ..bringup.reconcile import Reconciler

            r = Reconciler(
                rounds=self.rounds,
                tasks=self.tasks,
                locks=self.locks,
                shared_state=self.shared_state,
                resources=self.policy.resources,
                # A callable, not a captured mapping: the resume replay rebuilds
                # its contents, so a snapshot taken here would be pre-replay.
                proposals=lambda: self.state.pending_proposals,
                session_dir=self.session_dir,
            )
            self.__dict__["_reconciler"] = r
        return r

    def _kb_hardware_slug(self) -> str:
        """Topology-aware hardware dimension for the recipe ``canonical_id``."""
        from hyperloom.orchestrator.actions.executors._multi_node_env import resolve_kb_topology
        from hyperloom.inference_optimizer.recipe_snapshot_constants import kb_hardware_slug

        ss = self.shared_state
        return kb_hardware_slug(ss.gpu_type or "unknown_gpu", **resolve_kb_topology())

    def _reap_orphaned_servers_best_effort(self, *, phase: str) -> None:
        """Reap leftover single-node serving processes via this session's pidfiles.

        Runs at boot and again at shutdown. Every other teardown here is in-band -- a benchmark wrapper's own signal
        trap, or a ``killpg`` on a handle we still hold -- so none of it runs when the owner dies without getting to
        execute. This is the backstop for that case, and with no cgroup or pid-namespace available it is the only one:
        the pidfile outlives whatever wrote it. Scoped to this session's own pidfiles and gated on a cmdline match, so
        a co-located session's server and a recycled pid are never touched.
        """
        try:
            from ..actions.executors._multi_node_env import is_multi_node

            if is_multi_node():
                return
            from ..actions.executors._server_lifecycle import reap_orphaned_servers

            reaped = reap_orphaned_servers(self.session_dir)
            if reaped:
                log.warning(
                    "coordinator: reaped %d orphaned serving process(es) at %s: %s",
                    len(reaped),
                    phase,
                    reaped,
                )
        except Exception:
            log.exception("coordinator: orphan server reaper failed at %s (ignored)", phase)

    def _pin_source_trees(self) -> None:
        """Pin the source trees this session observes and patches, once at boot.

        Every failure digest is keyed on a frame normalised against these roots,
        so the pin must not move once the first attempt has run. A host with no
        framework tree on disk pins an empty set.
        """
        from ..bringup import resolve_trees, write_trees

        write_trees(resolve_trees(), session_dir=self.session_dir)

    # Advisory disk guard: when the session partition runs low, LRU-trim the
    # bulkiest churn (per-task runs/ workspaces); durable state is never touched.
    _DISK_FREE_MIN_GB: float = 20.0
    _DISK_USED_MAX_FRAC: float = 0.85
    _DISK_RUNS_KEEP_PER_ACTION: int = 50
    _STATE_JSON_WARN_BYTES: int = 50 * 1024 * 1024

    # Action catalogue mapping action_name -> metadata.
    action_registry: Mapping[str, ActionMetadata] = ACTION_CATALOGUE

    # Inline fast-action execution; deny report/session_breakdown (CLOSE artifacts).
    _INLINE_ACTION_DENY: frozenset[str] = frozenset(
        {
            "report",
            "session_breakdown",
        }
    )

    # Lifecycle
    async def stop(self) -> None:
        """Signal shutdown, cancel in-flight work, and close the DB."""
        self._stop.set()
        try:
            await self.cancel_inflight_actions(reason="coordinator_stop")
        except Exception:
            log.exception("Coordinator.stop: cancelling in-flight actions raised")
        for t in self._tasks_running:
            if not t.done():
                t.cancel()
        for t in self._tasks_running:
            try:
                await t
            except asyncio.CancelledError:
                # Expected: we just cancelled these tasks.
                pass
            except Exception:
                log.exception("reactor task raised on shutdown")
        await self.close_db_after_executions()

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
        self._run_started_monotonic = time.monotonic()
        self._run_deadline = deadline
        return grace_sec, deadline, float(max_minutes_value)

    async def _recipe_kb_t4_hook(self) -> None:
        """Finalize or retry on graceful teardown/Ctrl-C."""
        if bool(getattr(getattr(self, "knowledge_plane", None), "kb_disabled", False)):
            return
        finalize_status = str(getattr(self.shared_state, "recipe_finalize_status", "") or "")
        if getattr(self.shared_state, "close_sequence_done", False) and finalize_status in {
            "written",
            "skipped",
            "disabled",
        }:
            return
        config = getattr(getattr(self, "knowledge_plane", None), "config", None) or KnowledgeConfig.from_env()
        if config.mode is KnowledgeStoreMode.LOCAL:
            if self.recipe_kb is None:
                return
            sid = (self.shared_state.recipe_kb_session_id or "").strip()
            if not sid:
                return
        self.ensure_recipe_finalized(source="t4_fallback")
        try:
            self.shared_state.save(self.session_dir)
        except Exception:
            log.exception("recipe KB T4 SharedState.save failed")

    # Statuses that mean the candidate was ADOPTED; everything else is a negative signal for the ranker.
    _FRAMEWORK_KEEP_STATUSES: frozenset[str] = frozenset({"kept"})

    # Max tried-candidate rows fed into the ranker/discovery working memory.
    _FRAMEWORK_TRIED_MEMORY_CAP: int = 12

    _CRITIC_PRIORS_OUTCOME_TAIL: int = 5

    # Relative-change floor for the pre-GEAK reprofile: any change above this re-runs profile+TraceLens (effectively
    # "any change", absorbing float noise).
    _REPROFILE_CHANGE_TOL: float = 1e-5

    # Max re-author rounds per candidate on a needs_review verdict.
    _MAX_REAUTHOR_ATTEMPTS: int = _AUTHORED_LANE_MAX_ATTEMPTS

    # Backstop: max Critic-review submissions for a single candidate before the pump force-stamps
    # ``repeated_review_abort`` and stops re-selecting it.
    _MAX_REPEATED_REVIEW_SUBMISSIONS: int = 3

    # CLOSE step 0 post-opt roofline hard cap; on timeout the optimized snapshot is skipped so report/breakdown always
    # run.
    CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC: float = 600.0

    # Floor on how long CLOSE waits for its full-stack revalidation. The bound scales to two baseline runtimes (a cold
    # boot plus the warm decision round); explore's own session-deadline check keeps it inside the run's budget.
    CLOSE_STACK_REVALIDATION_TIMEOUT_SEC: float = 600.0

    # optimization_stack actions warranting a post-opt roofline; pure param-search (explore) is excluded.
    _POST_OPT_ROOFLINE_ACTIONS = frozenset({"integrate", "integrate_patch", "gemm_tuning", "geak_e2e"})

    async def tick(self, n: int = 1) -> None:
        """Run exactly ``n`` reactor passes for every agent; dispatcher pumps at pass end, lazy resume replay on tick 1."""
        await self._replay_resume_if_needed()
        for _ in range(n):
            # The tick's first act; see
            # :mod:`hyperloom.orchestrator.bringup.reconcile`.
            await self.reconciler.run(time.time())
            self.shared_state.increment_tick()
            # A phase-entry hook may have finished by setting a pending phase hint (for example current GEAK returning
            # no_gain -> skip_to_sweep).
            await self._await_within_session_bound(
                self._advance_phase_if_needed,
                stage="advance_phase_pre_reactor",
            )
            if str(getattr(self.shared_state, "pending_escalate_hint", "") or "").strip():
                await self._await_within_session_bound(
                    self._advance_phase_if_needed,
                    stage="advance_phase_hint",
                )
            for name in self._tick_roles:
                await self._await_within_session_bound(
                    lambda n=name: self._reactor_pass(n),
                    stage=f"reactor:{name}",
                )
            await self._pump_dispatcher_once()
            # FRAMEWORK_AGENT phase pump: enqueue next candidate / fetch next batch.
            await self._pump_framework_agent_phase_safely(caller="tick")
            # Phase-independent enablement pump: repair a non-runnable combo.
            await self._pump_enablement_safely(caller="tick")
            # phase machine advance at tick boundary.
            await self._await_within_session_bound(
                self._advance_phase_if_needed,
                stage="advance_phase",
            )

    def _record_coordinator_exception(
        self,
        *,
        stage: str,
        exc: BaseException,
        tick: int | None = None,
        agent: str = "",
    ) -> None:
        """Record a Coordinator-side exception without killing the session."""
        self._fault_open_phase_event(stage=stage, exc=exc)
        try:
            self.shared_state.record_tick_exception(
                tick=int(tick if tick is not None else self.shared_state.tick or 0),
                stage=stage,
                agent=agent,
                exc_type=type(exc).__name__,
                message=str(exc),
                traceback_text="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            )
            self.shared_state.increment_crash_count()
            self.shared_state.save(self.session_dir)
        except Exception:
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
        from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import active_kernel_recorder

        for recorder in (active_kernel_recorder(), self._framework_timeline()):
            if recorder is None:
                continue
            recorder.record_fault(stage=stage, exc=exc)

    def _seconds_until_session_bound(self) -> float | None:
        """Seconds left on the active run or closing bound; ``None`` if unbounded."""
        if bool(getattr(self.shared_state, "closing_phase", False)):
            bound = self._closing_deadline
        else:
            bound = self._run_deadline
        if bound is None:
            return None
        return bound.remaining()

    def _stage_timeout_sec(self, stage: str) -> float | None:
        """Return the total wall-clock ceiling for an inline reactor turn."""
        role = stage.removeprefix("reactor:")
        if role == stage:
            return None
        return self.reactor_turn_timeout_sec

    def _stop_requested(self) -> bool:
        """Whether an operator has asked this run to stop.

        Reads both the asyncio event and the drain's threading event, so the
        end-of-tick check sees a signal that arrived during that tick.

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

    async def _await_within_session_bound(
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
                self._record_coordinator_exception(stage=stage, exc=exc, agent=stage.removeprefix("reactor:"))

    # Long-run interface
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
        closing_grace_sec: float | None = None,
    ) -> str:
        """Run reactor + dispatcher until a stop condition fires (priority order): signal, a stop_reason the phase machine recorded (a met target closes through SWEEP as one), time_exhausted (via closing phase), emergency, custom, max_ticks. Sets + saves + returns shared_state.stop_reason."""
        objective = objective or TimeOnlyObjective()
        # Stash so _compose_prompt can update target_gap_pct.
        self._current_objective = objective
        # Capture the live loop for the inline fast-action context tool.
        try:
            self._coordinator_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._coordinator_loop = None

        # A dedicated thread reading the interpreter's wakeup pipe, not a loop
        # callback: a TERM has to be recorded while the loop is busy.
        if install_signal_handlers:
            self._signals = SignalDrain(
                loop=asyncio.get_running_loop(),
                stop_event=self._stop,
            )
            log.info("Coordinator.run: stop-signal drain armed=%s", self._signals.armed)

        await self._replay_resume_if_needed()
        grace_sec, deadline, max_minutes_value = self._bind_session_deadline(
            max_minutes=max_minutes,
            closing_grace_sec=closing_grace_sec,
        )

        tick_n = 0
        stop_reason = ""
        last_tick_exc: BaseException | None = None
        closing_deadline: Deadline | None = None
        try:
            while not stop_reason:
                tick_n += 1
                in_closing = bool(self.shared_state.closing_phase)
                try:
                    # Repair before anything is admitted: a round nobody will
                    # settle, a task row with no process, a review nobody
                    # answered. Ungated, because a stuck round closes every gate
                    # this could sit behind.
                    await self.reconciler.run(time.time())
                    # Bump the persistent tick counter — drives phase/plateau math.
                    self.shared_state.increment_tick()
                    try:
                        await self._await_within_session_bound(
                            self._advance_phase_if_needed,
                            stage="advance_phase_pre_reactor",
                        )
                        if str(getattr(self.shared_state, "pending_escalate_hint", "") or "").strip():
                            await self._await_within_session_bound(
                                self._advance_phase_if_needed,
                                stage="advance_phase_hint",
                            )
                    except Exception as exc:
                        log.exception("phase advance before reactors (run) failed")
                        self._record_coordinator_exception(
                            stage="advance_phase_pre_reactor",
                            exc=exc,
                            tick=tick_n,
                        )
                    in_closing = self.shared_state.closing_phase
                    # One reactor + dispatcher pass; during closing skip LLM passes.
                    if not in_closing:
                        for name in self._tick_roles:
                            if self._stop_requested():
                                break
                            await self._await_within_session_bound(
                                lambda n=name: self._reactor_pass(n),
                                stage=f"reactor:{name}",
                            )
                    if not self._stop_requested():
                        await self._pump_dispatcher_once()
                    # FRAMEWORK_AGENT phase pump: see ``tick()`` for rationale.
                    if not in_closing:
                        await self._pump_framework_agent_phase_safely(caller="run")
                        # Phase-independent enablement pump.
                        await self._pump_enablement_safely(caller="run")
                    # phase machine advance; runs even in_closing so CLOSE is recorded.
                    try:
                        await self._await_within_session_bound(
                            self._advance_phase_if_needed,
                            stage="advance_phase",
                        )
                    except Exception as exc:
                        log.exception("phase advance (run) failed")
                        self._record_coordinator_exception(
                            stage="advance_phase",
                            exc=exc,
                            tick=tick_n,
                        )
                    # Periodic reaper + DB retention; time-gated.
                    now = time.monotonic()
                    if now - self._last_maintenance_ts >= MAINTENANCE_INTERVAL_SEC:
                        await self._run_maintenance(tick=tick_n)
                        self._last_maintenance_ts = now
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as exc:
                    last_tick_exc = exc
                    log.exception("Coordinator.run: tick %d body raised", tick_n)
                    self._record_coordinator_exception(
                        stage="tick_body",
                        exc=exc,
                        tick=tick_n,
                    )

                # check stop conditions
                if self._stop_requested():
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
                    closing_deadline = await self._enter_closing_phase(
                        grace_sec=grace_sec,
                    )
                    self._closing_deadline = closing_deadline
                    continue
                if in_closing:
                    report_terminal = await self._closing_report_terminal()
                    grace_blown = closing_deadline is not None and closing_deadline.expired()
                    if report_terminal or grace_blown:
                        if grace_blown and not report_terminal:
                            log.warning(
                                "Coordinator: closing-grace exhausted (%.0fs) before report task %s finished",
                                grace_sec,
                                self.shared_state.closing_report_task_id,
                            )
                        stop_reason = "time_exhausted"
                        break
                if (
                    self.shared_state.recent_crash_count(
                        window_sec=_CRASH_EMERGENCY_WINDOW_SEC,
                    )
                    >= crash_emergency_threshold
                ):
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

                # Brief wait between ticks to avoid CPU spin while staying signal-responsive; 0.0 keeps tests fast.
                if tick_interval_sec > 0:
                    try:
                        await asyncio.wait_for(self._stop.wait(), timeout=tick_interval_sec)
                        stop_reason = self._signal_stop_reason()
                        break
                    except asyncio.TimeoutError:
                        # Normal path: no stop signal within the tick interval.
                        pass
        finally:
            try:
                await self._await_kernel_entry_task()
            except (asyncio.CancelledError, Exception):
                log.exception("Coordinator: KERNEL entry hook did not settle before shutdown")
            final_signals: AbstractSet[int] = frozenset()
            if self._signals is not None:
                final_signals = self._signals.close()
                self._signals = None
            stop_reason = self._classify_stop(final_signals, pending=stop_reason)
            self._stop_classification = stop_reason
            if self.shared_state.closing_phase:
                self.shared_state.closing_phase = False
            # Resuming a terminal session can break out before stop_reason is set.
            self.shared_state.set_stop_reason(
                stop_reason
                or self.shared_state.stop_reason
                or ("coordinator_exception" if last_tick_exc is not None else "unknown")
            )
            self.shared_state.save(self.session_dir)
            try:
                await self.ensure_close_sequence(reason=self.shared_state.stop_reason)
            except (asyncio.CancelledError, Exception):
                log.exception("Coordinator: terminal close sequence did not finish")
            await self._recipe_kb_t4_hook()
            log.info(
                "Coordinator.run: stopped tick=%d reason=%s baseline_tput=%.1f "
                "cumulative_gain_validated=%.2f%% max_minutes=%.0f",
                tick_n,
                stop_reason or "unknown",
                self.shared_state.baseline_tput,
                self.shared_state.cumulative_gain_validated,
                max_minutes_value,
            )
            with timed_teardown_step(self.shared_state, "close_backends"):
                await self._close_backends()
            # A server outliving the run holds every GPU it was given, so the
            # session's last act is to reap its own pidfiles.
            with timed_teardown_step(self.shared_state, "reap_orphaned_servers"):
                await asyncio.to_thread(self._reap_orphaned_servers_best_effort, phase="shutdown")
            self.shared_state.save(self.session_dir)
        return stop_reason or self.shared_state.stop_reason

    async def _close_backends(self) -> None:
        """Release every backend holding a live agent session."""
        for name, backend in list(self.backends.items()):
            closer = getattr(backend, "aclose", None)
            if not callable(closer):
                continue
            try:
                await closer()
            except Exception:
                log.exception("Coordinator: closing the %s backend failed", name)

    # Reactor
    async def _reactor_pass(self, agent_name: str) -> None:
        """Run one reactor turn for ``agent_name`` and route its intents."""
        backend = self.backends[agent_name]
        sys_prompt = await self._load_system_prompt(agent_name)
        prompt = await self._compose_prompt(agent_name)
        tools = self.policy.allowed_tools_for_agent(agent_name)
        # Stamp timeline keys onto backends that self-write their trace row.
        _set_trace_ctx = getattr(backend, "set_trace_context", None)
        backend_self_traces = callable(_set_trace_ctx)
        if backend_self_traces:
            _set_trace_ctx(
                tick=int(self.shared_state.tick or 0),
                phase=(self.shared_state.phase or "") or None,
                macro_cycle=int(self.shared_state.macro_cycle or 0),
            )
        # max_turns=0 → backend default.
        _t0 = time.perf_counter()
        try:
            result: BackendTurnResult = await backend.run(
                prompt=prompt,
                system_prompt=sys_prompt,
                tools=tools,
                max_turns=0,
            )
        except BackendError as exc:
            if isinstance(exc, LLMCallFailed) and not backend_self_traces:
                self._trace_reactor_llm_failure(
                    agent_name,
                    exc,
                    latency_ms=int((time.perf_counter() - _t0) * 1000),
                )
            await self._record_observation(
                "coordinator",
                "observation",
                {"kind": "backend_error", "agent": agent_name, "error": repr(exc)},
            )
            await self._track_backend_error_streak(agent_name, exc)
            return
        except NoIntentEmitted as exc:
            # No parseable intents; surface as observation so the next tick self-corrects.
            await self._record_observation(
                "coordinator",
                "observation",
                {"kind": "no_intent_emitted", "agent": agent_name, "error": str(exc)[:500]},
            )
            await self._advance_rendered_cursor(agent_name)
            return
        except Exception as exc:
            # Catch-all so one agent's bad turn never stops the loop.
            log.exception("reactor pass for %s raised", agent_name)
            await self._record_observation(
                "coordinator",
                "observation",
                {"kind": "reactor_exception", "agent": agent_name, "error": format_exc_brief(exc, limit=500)},
            )
            self._record_coordinator_exception(
                stage="reactor_pass",
                agent=agent_name,
                exc=exc,
            )
            return
        finally:
            self._trace_mcp_setup(agent_name=agent_name, backend=backend)
        # Reset the streak — a successful turn proves the backend is alive again.
        if self._backend_error_streak.get(agent_name):
            self._backend_error_streak[agent_name] = 0
            self._backend_error_alarm_armed[agent_name] = True
        # Record this reactor turn's token spend on the unified ledger.
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        self._trace_reactor_llm_call(agent_name, result, latency_ms=latency_ms)
        # Full-trace: persist the redacted prompt+response for this turn.
        self._record_reactor_conversation(agent_name, result)
        for intent in result.intents:
            await self._handle_intent(agent_name, intent)
        await self._advance_rendered_cursor(agent_name)
        self.shared_state.agent_last_active[agent_name] = time.time()

    def _trace_mcp_setup(self, *, agent_name: str, backend: Backend) -> None:
        """Persist orchestration MCP setup once per session."""
        if agent_name != "orchestration":
            return
        try:
            setup_getter = getattr(backend, "get_mcp_setup_diagnostic", None)
            if callable(setup_getter):
                setup = setup_getter()
                if isinstance(setup, dict):
                    write_mcp_setup_once(session_dir=self.session_dir, setup=setup)
        except Exception:
            log.debug("orchestration mcp setup trace failed", exc_info=True)

    def _trace_reactor_llm_call(
        self,
        agent_name: str,
        result: BackendTurnResult,
        *,
        latency_ms: int | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a reactor turn."""
        try:
            metadata = result.metadata or {}
            has_tokens = any(
                metadata.get(k) is not None
                for k in (
                    "input_tokens",
                    "output_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
            )
            if not has_tokens:
                return
            record = LLMCallRecord.from_metadata(
                session_id=self.session_dir.name,
                component=agent_name,
                role=agent_name,
                metadata=metadata,
                tick=int(self.shared_state.tick or 0),
                phase=(self.shared_state.phase or "") or None,
                latency_ms=latency_ms,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:
            log.debug(
                "full-trace: reactor llm_call append failed for %s",
                agent_name,
                exc_info=True,
            )

    def _trace_reactor_llm_failure(
        self,
        agent_name: str,
        error: LLMCallFailed,
        *,
        latency_ms: int | None = None,
    ) -> None:
        """Append one ``status=\"error\"`` ``llm_calls.jsonl`` row for a failed turn."""
        try:
            record = LLMCallRecord.for_failure(
                session_id=self.session_dir.name,
                component=agent_name,
                role=agent_name,
                error=error,
                tick=int(self.shared_state.tick or 0),
                phase=(self.shared_state.phase or "") or None,
                latency_ms=latency_ms,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:
            log.debug(
                "full-trace: reactor llm_call failure append failed for %s",
                agent_name,
                exc_info=True,
            )

    async def _track_backend_error_streak(
        self,
        agent_name: str,
        exc: BackendError,
    ) -> None:
        """Increment the per-agent ``BackendError`` streak; emit one backend_unhealthy event on crossing the threshold (re-arms only after a successful turn)."""
        new_value = self._backend_error_streak.get(agent_name, 0) + 1
        self._backend_error_streak[agent_name] = new_value
        threshold = self._backend_error_streak_threshold
        if new_value >= threshold and self._backend_error_alarm_armed.get(agent_name, True):
            self._backend_error_alarm_armed[agent_name] = False
            await self._record_observation(
                "coordinator",
                "observation",
                {
                    "kind": "backend_unhealthy",
                    "agent": agent_name,
                    "consecutive_errors": new_value,
                    "threshold": threshold,
                    "latest_error": repr(exc)[:500],
                    "severity": "high",
                    "hint": (
                        "subprocess backend has failed >= threshold times "
                        "consecutively; consider switching to a mock "
                        "backend (e.g. --critic-mock) "
                        "while the underlying transport is repaired"
                    ),
                },
            )

    # Multi-node only: cap on specialist proposal_set entries auto-materialised into a single explore grid per round.
    _MN_AUTO_EXPLORE_GRID_CAP = 6

    # Phases whose long, serially-drained GPU grids must not starve the per-phase cyclic budget exit.
    _BUDGET_GATED_DISPATCH_PHASES: frozenset[str] = frozenset({"FRAMEWORK_AGENT", "KERNEL_AGENT"})

    # Fact-write surface — journal + direct KB lesson/pitfall/recipe writes.
    PITFALL_REGRESS_THRESHOLD_PCT: float = -5.0  # gain_pct ≤ this → pitfall


__all__ = [
    "Coordinator",
    "CoordinatorState",
    "PendingProposal",
    "SharedState",
    # Re-exported from coordinator_helpers / state.shared_state for callers/tests.
    "_infer_model_class_from_config",
    "effective_closing_grace_sec",
    # Re-exported from policy.gate; referenced via ``coordinator.<name>`` in tests.
    "SPECIALIST_FROM_AGENT_PREFIX",
]
