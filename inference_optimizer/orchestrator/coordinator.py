# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import signal
import time
import traceback
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..compat.payload_aliases import read_extra_server_args
from ..recipe_kb import RecipeKB, recipe_canonical_id
from ..recipe_snapshot_constants import detect_framework_version

# Recipe snapshot severity tags (schema has no fixed enum).
_SEVERITY_CRASH:   str = "crash"
_SEVERITY_REGRESS: str = "regress"

# Bounded transient-failure auto-retry for specialist dispatches: a subprocess
# timeout / crash / stale-heartbeat re-dispatches up to this many times before
# the failure is recorded normally. Infra-only (see classify_specialist_failure);
# semantic empties are left for the orchestrator. Env override / disable via
# INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY (set "0" to disable).
SPECIALIST_AUTO_RETRY_MAX: int = 2
from . import phase_state as _phase_state
from .optimization_journal import (
    Journal,
    JournalEntry,
    OUTCOME_KEEP,
    OUTCOME_NO_PROMOTE,
    OUTCOME_REVERT,
    classify_change_kind,
    summarize_change,
)
from ..paths import db_path_for
from ..storage.connection import SqliteConnection
from .action_registry import ActionRegistry
from .agent_role import AgentRole, default_role_registry
from .backends.base import Backend, BackendError, BackendTurnResult
from .cursor_store import CursorStore
from ..protocol.intent import Intent, IntentType, NoIntentEmitted
from .kernel_request_handlers import get_handler
from .message_bus import Message, MessageBus
from .objective import Objective, TimeOnlyObjective
from .policy import (
    PolicyDenied,
    PolicyGate,
    RESEARCH_LANE_NAME,
    SPECIALIST_FROM_AGENT_PREFIX,
)
from .gpu_pool import (
    SpecialistGpuPool,
    resolve_gpu_specialist_devices,
)
from .resource_lock import (
    KNOWN_LANES,
    ResourceLockManager,
    SqliteLeaseBackend,
    _expand_lanes,
)
from .shared_state import SharedState
from .sub_agent_runner import SubAgentResult, SubAgentRunner
from .task_registry import Task, TaskRegistry
from .trace.conversation_trace import ConversationRecord, append_conversation
from .trace.llm_trace import LLMCallRecord, append_llm_call
from .action_executors.benchmark_result import is_valid_measurement
from .coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _BASELINE_SELF_LOOP_THRESHOLD,
    _DEFAULT_ROOFLINE_WATERMARK_RATIO,
    _MULTI_VALUE_SGLANG_FLAGS,
    _ROOFLINE_WATERMARK_RATIO_ENV,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_sglang_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _resolve_roofline_watermark_ratio,
    _summarize_failed_variants,
    effective_closing_grace_sec,
)


log = logging.getLogger(__name__)


# Audit-trail kinds (must match shared_state._AUDIT_ACTIONS); kernel-owned actions excluded.
_AUDIT_ACTIONS: frozenset[str] = frozenset({
    "baseline", "profile", "sweep", "explore",
    # Composite roofline runs profile + trace_analyze atomically.
    "roofline",
})

# Default per-repo candidate cap for ``fa phase-discover`` (FRAMEWORK_PR).
DEFAULT_FRAMEWORK_PR_MAX_CANDIDATES: int = 8


# Result keys surfaced in delegated_result inbox line; first match wins per group.
_OUTCOME_GAIN_KEYS: tuple[str, ...] = (
    "validated_gain_pct", "gain_pct", "predicted_gain_pct", "delta_pct",
)
_OUTCOME_TPUT_KEYS: tuple[str, ...] = (
    "tokens_per_s", "tput", "throughput", "tput_tok_s",
)
_OUTCOME_STATUS_KEYS: tuple[str, ...] = ("status", "verdict", "outcome")


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    """Return ``d[k]`` for the first ``k`` in ``keys`` present + non-None."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _format_inbox_event(m: "Message") -> str:
    """Render one inbox ``Message`` as a compact, high-signal line (Path A/A1)."""
    topic = (m.topic or "").strip()
    payload = m.payload if isinstance(m.payload, dict) else {}
    # DESIGN §13.1: canonical inbox header ordering downstream parsers anchor on.
    if getattr(m, "msg_id", None):
        head = (
            f"seq={m.seq} msg_id={m.msg_id} from={m.from_agent} topic={topic}"
        )
    else:
        head = f"seq={m.seq} from={m.from_agent} topic={topic}"

    if topic == "delegated_result":
        kind = payload.get("kind")
        state = payload.get("state")
        error = payload.get("error")
        result = payload.get("result")
        parts = [head, f"kind={kind!r}", f"state={state!r}"]
        if isinstance(result, dict):
            status = _first_present(result, _OUTCOME_STATUS_KEYS)
            gain = _first_present(result, _OUTCOME_GAIN_KEYS)
            tput = _first_present(result, _OUTCOME_TPUT_KEYS)
            kept = result.get("kept")
            if status is not None:
                parts.append(f"status={status!r}")
            if kept is not None:
                parts.append(f"kept={kept!r}")
            if gain is not None:
                parts.append(f"gain={gain}")
            if tput is not None:
                parts.append(f"tput={tput}")
        if error:
            parts.append(f"error={str(error)[:200]!r}")
        return " ".join(parts)

    if topic in ("policy_denial", "denial") or (
        topic == "observation" and payload.get("kind") == "policy_denial"
    ):
        return (
            f"{head} action={payload.get('action_name')!r} "
            f"rule={payload.get('rule')!r} "
            f"hint={str(payload.get('hint') or '')[:140]!r}"
        )

    if topic == "review_verdict":
        return (
            f"{head} target={payload.get('target_proposal_msg_id')!r} "
            f"verdict={payload.get('verdict')!r} "
            f"reasoning={str(payload.get('reasoning') or '')[:140]!r}"
        )

    if topic == "observation":
        kind = payload.get("kind")
        if kind is not None:
            return f"{head} kind={kind!r} payload={payload}"

    return f"{head} payload={payload}"


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
    # Kept for review_verdict envelope schema + resume replay; no live writer populates it.
    verdict_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Always-empty, kept for forward compat + stable defer/restore shape.
    kb_edge_ids: dict[str, str] = field(default_factory=dict)


# #266 lifecycle: path-like keys worth surfacing from a kernel handler
# payload (inputs) or result (outputs) so operators can see where a step's
# artifacts went without enumerating every per-handler return shape.
_LIFECYCLE_PATH_KEYS: tuple[str, ...] = (
    "trace_input", "trace_dir", "candidates_path", "analysis_md_path",
    "kernel_candidates", "best_artifact_path", "patch_path", "target_file",
    "workspace", "workspace_path", "out_dir", "output_dir", "run_dir",
    "report_path", "json_path", "md_path", "tracelens_agent_transcript",
    "tracelens_agent_report",
    # TraceLens analysis outputs surfaced by trace_analyze_handler — the
    # analysis.md report, its alias, the per-run audit summary, the roofline
    # sidecar and the CLI log — so operators can reach them from lifecycle END.
    "trace_report_path", "analysis_report_path", "tracelens_summary_path",
    "kernel_roofline_path", "cli_log_path",
)


def _lifecycle_paths(payload: Any) -> dict[str, str]:
    """Extract present, non-empty path-like fields from a kernel handler
    payload or result dict (#266). Best-effort: a non-dict argument yields
    an empty mapping so callers never have to guard the type."""
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


class Coordinator:
    """The single Coordinator instance per session."""

    def __init__(
        self,
        session_dir: Path,
        *,
        backends: dict[str, Backend],
        role_registry: dict[str, AgentRole] | None = None,
        sub_agent_runner: SubAgentRunner | None = None,
        bus_class: type[MessageBus] = MessageBus,
        compare_against_gpu: str | None = None,
        model_class: str | None = None,
        cortex_kb: RecipeKB | None = None,
        phase_budget_pct: dict[str, float] | None = None,
        knowledge_plane: Any = None,
        proposal_scorer: Any = None,
        warm_replay_enabled: bool = True,
        warm_replay_min_confidence: float = 0.7,
        warm_replay_min_reproduce_pct: float = 0.8,
    ):
        """Construct the single per-session Coordinator and wire its plane.

        Builds the persistence layer (SQLite connection, MessageBus,
        ResourceLockManager, TaskRegistry, CursorStore), the PolicyGate,
        the phase machine, and the per-agent reactor bookkeeping, then
        detects whether this session is a resume and anchors the Cortex KB.

        Args:
            session_dir (Path): Directory holding this session's state.json,
                SQLite DB and artifacts.
            backends (dict[str, Backend]): Map of agent-role name to its
                Backend; every role in ``role_registry`` must be present.
            role_registry (dict[str, AgentRole] | None): Agent-role registry;
                ``None`` uses :func:`default_role_registry`.
            sub_agent_runner (SubAgentRunner | None): Runner for delegated
                sub-agent tasks; ``None`` constructs a default one.
            bus_class (type[MessageBus]): MessageBus class to instantiate.
            compare_against_gpu (str | None): Reference GPU id for target
                analysis priors; ``None``/blank disables external comparison.
            model_class (str | None): Model-class override seeded into
                SharedState when none is already persisted.
            cortex_kb (RecipeKB | None): Recipe-snapshot KB dispatcher; ``None``
                makes the fact-write hooks no-ops.
            phase_budget_pct (dict[str, float] | None): Per-phase wall-clock
                budget percentages; ``None`` uses library defaults.
            knowledge_plane (Any): Optional KnowledgePlane facade used to
                pre-warm specialist knowledge before enqueue.
            warm_replay_enabled (bool): Whether warm-recipe replay may
                auto-apply the KB best_config.
            warm_replay_min_confidence (float): Minimum KB confidence required
                to fire a warm replay.
            warm_replay_min_reproduce_pct (float): Minimum reproduce fraction
                required to fire a warm replay.

        Raises:
            ValueError: If a role in ``role_registry`` has no matching backend.

        Attributes:
            session_dir (Path): Session working directory.
            backends (dict[str, Backend]): Wired per-role backends.
            db (SqliteConnection): Session persistence connection.
            bus (MessageBus): Message routing bus.
            locks (ResourceLockManager): Lane/resource lease manager.
            tasks (TaskRegistry): Delegated-task registry.
            cursors (CursorStore): Per-agent message cursors.
            sub (SubAgentRunner): Sub-agent task runner.
            shared_state (SharedState): Persistent session state (state.json).
            policy (PolicyGate): Intent-validation choke-point.
            state (CoordinatorState): In-memory reactor/dispatcher state.
        """
        self.session_dir = Path(session_dir)
        self.role_registry = role_registry or default_role_registry()
        # Recipe-snapshot KB dispatcher; ``None`` makes fact-write hooks no-ops.
        self.cortex_kb: RecipeKB | None = cortex_kb
        # Per-session optimization journal; lazy-instantiated on first use.
        self._journal: Journal | None = None
        # Warm-recipe replay controls (PRELUDE auto-apply of KB best_config).
        self._warm_replay_enabled: bool = bool(warm_replay_enabled)
        self._warm_replay_min_confidence: float = float(warm_replay_min_confidence)
        self._warm_replay_min_reproduce_pct: float = float(warm_replay_min_reproduce_pct)
        # KnowledgePlane facade; when non-None pre-warms PR feed + advisory context.
        self.knowledge_plane: Any = knowledge_plane
        # ProposalScorer facade (advisory only, never gates).
        self._proposal_scorer: Any = proposal_scorer
        # Phase budget percentages, normalised once at construction.
        self._phase_budget_pct: dict[str, float] = _phase_state.normalize_budget_pct(
            phase_budget_pct
        )
        # Specialist stale scan threshold (seconds).
        try:
            self._specialist_stale_sec: float = max(
                0.0,
                float(os.environ.get(
                    "INFERENCE_OPTIMIZER_SPECIALIST_STALE_SEC", "600",
                )),
            )
        except ValueError:
            self._specialist_stale_sec = 600.0
        # External launcher config; decides whether target_analysis fetches real rows.
        self._compare_against_gpu: str = (compare_against_gpu or "").strip()
        self._model_class_override: str = (model_class or "").strip()

        # Validate every reactor has a backend wired.
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
        self.sub = sub_agent_runner or SubAgentRunner(
            self.locks, self.tasks, session_dir=self.session_dir,
        )

        # Persistent session state (state.json) — load existing for resume.
        self.shared_state = SharedState.load_or_init(self.session_dir)
        # #266 lifecycle save debounce: terminal events (END/ERROR) flush
        # immediately so operators see produced artifacts promptly; bursty
        # non-terminal markers (START/ENTER) coalesce within a short window to
        # avoid amplifying state.json writes on long / multi-kernel sessions
        # over NFS. ``_lifecycle_last_save`` is a monotonic timestamp.
        self._lifecycle_last_save: float = 0.0
        self._lifecycle_save_min_interval_s: float = 2.0
        # Thread live SharedState into the runner (constructed earlier) so
        # executors get it via ctx.extra; durable backstop for per-dispatch
        # ``base_tput`` injection.
        self.sub.shared_state = self.shared_state
        self.gpu_specialist_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_gpu_specialist_devices(
                int(getattr(self.shared_state, "gpu_specialist_capacity", 0) or 0)
            ),
        )
        # Sync research_lane capacity into lane_capacity so acquire_many honours the cap.
        try:
            from ..storage.schema import set_lane_capacity as _set_lane_capacity
            cap = int(self.shared_state.research_lane_capacity or 0)
            if cap >= 0:
                _set_lane_capacity(self.db.raw, "research_lane", cap)
        except Exception:  # noqa: BLE001 — non-fatal; default seed wins
            log.exception("failed to sync research_lane_capacity to leases DB")
        # `strict_paths` defers to the env flag (on in production, off in tests).
        self.policy = PolicyGate(
            role_registry=self.role_registry,
            session_dir=self.session_dir,
            shared_state=self.shared_state,
        )
        # Attach read-only context-pull MCP tools to Orchestration backend (plan Step 2).
        self._attach_orchestration_context_tools()
        # Resume detection must run before any boot-time state.json write.
        self._resumed_from = self._detect_resume_state()
        # Derive model_class once at boot if not supplied; never overwrite a resume.
        if not (self.shared_state.model_class or "").strip():
            self.shared_state.model_class = (
                self._model_class_override
                or _infer_model_class_from_config(
                    self.shared_state.model_path or os.environ.get("MODEL_PATH", "")
                )
            )
        self.state = CoordinatorState()
        self._stop = asyncio.Event()
        self._tasks_running: list[asyncio.Task] = []
        # Orchestration prompt mode (plan Step 3): first turn full SEED, later turns DELTA.
        self._orchestration_seeded: bool = False
        # Orchestration working-memory checkpoint policy + tracker (plan Step 4).
        from . import orchestration_memory as _orch_mem
        self._checkpoint_policy = _orch_mem.CheckpointPolicy()
        self._checkpoint_tracker = _orch_mem.CheckpointTracker(
            last_phase=str(getattr(self.shared_state, "phase", "") or ""),
        )
        # Disable checkpointing entirely via env.
        self._checkpoint_enabled: bool = (
            os.environ.get(
                "INFERENCE_OPTIMIZER_DISABLE_ORCH_CHECKPOINT", "",
            ).strip().lower() not in {"1", "true", "yes", "on"}
        )
        # Seed memory rendered into the next full SEED push (resume recovery source).
        self._orchestration_seed_memory: str = _orch_mem.render_memory_for_seed(
            dict(getattr(self.shared_state, "orchestration_memory", {}) or {})
        )
        # No-progress circuit-breaker telemetry (plan Step 6); threshold = high-severity cutoff.
        self._progress_marker: dict[str, Any] = {}
        try:
            self._no_progress_threshold: int = max(
                1,
                int(os.environ.get(
                    "INFERENCE_OPTIMIZER_NO_PROGRESS_TICKS", "15",
                )),
            )
        except ValueError:
            self._no_progress_threshold = 15

        # Per-agent BackendError streak; crossing threshold records one backend_unhealthy, then re-arms.
        self._backend_error_streak: dict[str, int] = {
            name: 0 for name in self.role_registry
        }
        self._backend_error_alarm_armed: dict[str, bool] = {
            name: True for name in self.role_registry
        }
        try:
            self._backend_error_streak_threshold: int = max(
                1,
                int(os.environ.get(
                    "INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD",
                    "5",
                )),
            )
        except ValueError:
            self._backend_error_streak_threshold = 5

        # Stable tick order from the live role_registry (NOT the cached
        # module-level roles_for_run, which keeps "kernel" under --no-kernel).
        _CANONICAL_ORDER = ("orchestration", "kernel", "critic", "robustness")
        self._tick_roles: tuple[str, ...] = tuple(
            r for r in _CANONICAL_ORDER if r in self.role_registry
        )

        # Action registry — yaml catalogue mapping action_name → metadata;
        # load failure falls back to ``None`` (handled gracefully).
        try:
            self.action_registry: ActionRegistry | None = ActionRegistry().load()
        except Exception:  # noqa: BLE001 — defensive; missing yaml shouldn't kill the run.
            log.exception("Coordinator: failed to load ActionRegistry.")
            self.action_registry = None
        # Inline fast-action execution (Path A / A3): run cheap lane-light action in-turn. Default ON.
        _inline_raw = os.environ.get(
            "INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS", "",
        ).strip().lower()
        self._inline_fast_actions_enabled: bool = _inline_raw not in {
            "0", "false", "no", "off",
        }
        self._coordinator_loop: asyncio.AbstractEventLoop | None = None
        # Wall-clock budget tracking for per-tick Time-budget prompt injection.
        self._run_deadline: float | None = None
        self._run_started_monotonic: float | None = None
        # Latest objective wired by run(); refreshes target_gap_pct each tick. None outside a run.
        self._current_objective: Objective | None = None

        # Initialise phase machine (fresh session enters PRELUDE). Idempotent.
        self._ensure_phase_initialised()
        # Cortex T0 defensive fallback for direct SDK/test callers; best-effort.
        self._ensure_cortex_t0_anchored()

    # Context-pull tools (plan Step 2)
    def _orchestration_conversational(self) -> bool:
        """True when the orchestration backend runs in persistent-conversation mode (plan Step 1)."""
        backend = self.backends.get("orchestration")
        return bool(getattr(backend, "conversational", False))

    def _reset_orchestration_conversation(self) -> None:
        """Force the next orchestration turn to re-seed a fresh conversation (plan Step 4)."""
        backend = self.backends.get("orchestration")
        reset = getattr(backend, "reset_conversation", None)
        if callable(reset):
            try:
                reset()
            except Exception:  # noqa: BLE001
                log.exception("Coordinator: orchestration reset_conversation failed")
        self._orchestration_seeded = False

    def _conversation_progress_signal(self) -> dict[str, Any]:
        """Compute the no-progress circuit-breaker signal (plan Step 6)."""
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
            self._progress_marker = {
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

    async def _maybe_checkpoint_orchestration(
        self, *, tick: int, phase_changed: bool = False,
    ) -> bool:
        """Compact the orchestration conversation into durable memory (plan Step 4).

        Returns True when a checkpoint was taken. Best-effort.
        """
        if not self._checkpoint_enabled:
            return False
        if not self._orchestration_conversational():
            return False
        backend = self.backends.get("orchestration")
        if backend is None or not getattr(backend, "conversational", False):
            return False
        # Nothing to compact before the first real turn seeded the session.
        if not self._orchestration_seeded:
            return False

        from . import orchestration_memory as _orch_mem

        now_min = 0.0
        if self._run_started_monotonic is not None:
            now_min = (time.monotonic() - self._run_started_monotonic) / 60.0
        tracker = self._checkpoint_tracker
        ticks_since = max(0, tick - tracker.last_tick)
        minutes_since = max(0.0, now_min - tracker.last_minute_mark)
        # Approximate conversation growth from recorded prompt char counts since last reset.
        if not self._checkpoint_policy.should_checkpoint(
            ticks_since_last=ticks_since,
            minutes_since_last=minutes_since,
            chars_since_last=tracker.chars_since_last,
            phase_changed=phase_changed,
        ):
            return False

        try:
            sys_prompt = await self._load_system_prompt("orchestration")
            result = await backend.run(
                prompt=_orch_mem.CHECKPOINT_REQUEST_PROMPT,
                system_prompt=sys_prompt,
                tools=[],
                max_turns=0,
                # Checkpoint summary is plain-text, not emit_intent; relax no-intent guard.
                allow_no_intent=True,
            )
            raw_text = getattr(result, "raw_text", "") or ""
            parsed = _orch_mem.parse_checkpoint_reply(raw_text)
            seq = 0
            try:
                row = self.bus.db.fetchone_sync(
                    "SELECT COALESCE(MAX(seq), 0) AS s FROM events"
                )
                seq = int(row["s"]) if row else 0
            except Exception:  # noqa: BLE001
                seq = 0
            record = _orch_mem.build_memory_record(
                parsed,
                seq=seq,
                tick=tick,
                previous=dict(
                    getattr(self.shared_state, "orchestration_memory", {}) or {}
                ),
            )
            self.shared_state.orchestration_memory = record
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.exception("Coordinator: failed to persist orchestration_memory")
            # Reset so the next turn re-seeds from the compacted memory.
            self._orchestration_seed_memory = _orch_mem.render_memory_for_seed(
                record
            )
            self._reset_orchestration_conversation()
            tracker.reset(
                tick=tick,
                minute_mark=now_min,
                phase=str(getattr(self.shared_state, "phase", "") or ""),
            )
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind": "orchestration_checkpoint",
                    "tick": tick,
                    "seq": seq,
                    "checkpoint_count": record.get("checkpoint_count", 0),
                    "phase_changed": bool(phase_changed),
                },
            )
            return True
        except Exception:  # noqa: BLE001 — never let a checkpoint kill the loop
            log.exception("Coordinator: orchestration checkpoint failed")
            return False

    def _attach_orchestration_context_tools(self) -> None:
        """Bind a read-only ContextProvider to the orchestration backend (no-op without setter)."""
        backend = self.backends.get("orchestration")
        setter = getattr(backend, "set_context_provider", None)
        if setter is None:
            return
        try:
            from .backends.mcp_context_tools import ContextProvider
            provider = ContextProvider(
                shared_state=self.shared_state,
                inbox_reader=self._context_inbox_reader,
                analysis_reader=self._context_analysis_reader,
                recent_outcomes_reader=self._context_recent_outcomes_reader,
                action_runner=self._run_action_now_sync,
            )
            setter(provider)
        except Exception:  # noqa: BLE001 — context pull is best-effort
            log.exception("Coordinator: failed to attach orchestration context tools")

    def _context_inbox_reader(self, since_seq: int = 0) -> str:
        """Synchronous projection of the orchestration inbox tail (sync SQLite path)."""
        try:
            rows = self.bus.db.fetchall_sync(
                "SELECT * FROM events WHERE seq > ? AND "
                "(to_agent = ? OR to_agent = '*') ORDER BY seq ASC",
                (int(since_seq or 0), "orchestration"),
            )
        except Exception as exc:  # noqa: BLE001
            return f"(inbox unavailable: {exc!r})"
        if not rows:
            return "(no inbox events)"
        from .message_bus import Message
        msgs = [Message.from_row(r) for r in rows]
        lines = [_format_inbox_event(m) for m in msgs[-40:]]
        return "\n".join(lines)

    def _context_recent_outcomes_reader(self, top_k: int = 8) -> str:
        """Synchronous projection of recent action outcomes (Path A / A2)."""
        try:
            k = max(1, min(int(top_k or 8), 50))
        except (TypeError, ValueError):
            k = 8
        try:
            rows = self.bus.db.fetchall_sync(
                "SELECT * FROM events WHERE topic IN "
                "('delegated_result', 'review_verdict') "
                "ORDER BY seq DESC LIMIT ?",
                (k,),
            )
        except Exception as exc:  # noqa: BLE001
            return f"(recent outcomes unavailable: {exc!r})"
        if not rows:
            return "(no recent outcomes)"
        from .message_bus import Message
        # Flip newest-first query to newest-last for chronological reading.
        msgs = [Message.from_row(r) for r in rows][::-1]
        lines = ["=== Recent action outcomes (newest last) ==="]
        lines.extend(_format_inbox_event(m) for m in msgs)
        return "\n".join(lines)

    # Inline fast-action execution (Path A / A3); deny report/session_breakdown (CLOSE artifacts).
    _INLINE_ACTION_DENY: frozenset[str] = frozenset({
        "report", "session_breakdown",
    })

    def _inline_action_whitelist(self) -> frozenset[str]:
        """Derive the set of actions safe to run inline (A3): lane-light, registered executor, not in _INLINE_ACTION_DENY. PolicyGate remains the real security boundary."""
        reg = getattr(self, "action_registry", None)
        if reg is None:
            return frozenset()
        executors = getattr(self.sub, "executor_registry", {}) or {}
        names_fn = getattr(reg, "names", None)
        try:
            names = list(names_fn()) if callable(names_fn) else []
        except Exception:  # noqa: BLE001 — defensive
            names = []
        allowed: set[str] = set()
        for name in names:
            if name in self._INLINE_ACTION_DENY:
                continue
            if name not in executors:
                continue
            lanes, _ttl = self._registry_lanes_ttl(name)
            if lanes:
                continue
            allowed.add(name)
        return frozenset(allowed)

    def _run_action_now_sync(
        self, action_name: str, params: dict[str, Any] | None = None,
    ) -> str:
        """Bridge callable for the ``run_action_now`` context tool (A3): marshals the executor coroutine onto the Coordinator loop and blocks with a timeout."""
        if not self._inline_fast_actions_enabled:
            return (
                "(run_action_now disabled: set "
                "INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS to a non-off value "
                "to enable; use emit_intent delegate for async execution)"
            )
        name = (action_name or "").strip()
        if not name:
            return "(run_action_now: action_name required)"
        whitelist = self._inline_action_whitelist()
        if name not in whitelist:
            return (
                f"(run_action_now: {name!r} is not inline-eligible — only "
                f"cheap, lane-light actions may run inline: "
                f"{sorted(whitelist)}. Use emit_intent delegate to run it "
                f"asynchronously.)"
            )
        loop = self._coordinator_loop
        if loop is None or loop.is_closed():
            return "(run_action_now unavailable: coordinator loop not running)"
        coro = self._run_action_now(name, dict(params or {}))
        # Cap inline wait under backend timeout so a slow action can't wedge the turn.
        try:
            timeout_s = float(
                os.environ.get(
                    "INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S", "120",
                ) or 120
            )
        except (TypeError, ValueError):
            timeout_s = 120.0
        try:
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
        except RuntimeError as exc:
            return (
                f"(run_action_now: could not schedule on coordinator loop: "
                f"{exc!r})"
            )
        try:
            return fut.result(timeout=timeout_s)
        except FuturesTimeoutError:
            return (
                f"(run_action_now: {name!r} still running after "
                f"{timeout_s:.0f}s; it keeps running asynchronously — check "
                "get_recent_outcomes or the next-tick inbox for its "
                "delegated_result)"
            )
        except Exception as exc:  # noqa: BLE001 — never crash the turn
            log.exception("run_action_now: inline run of %r failed", name)
            return f"(run_action_now: {name!r} errored: {exc!r})"

    async def _run_action_now(
        self, action_name: str, params: dict[str, Any],
    ) -> str:
        """Coordinator-loop coroutine that runs a whitelisted action inline through PolicyGate + SubAgentRunner, publishing a delegated_result for audit/inbox parity."""
        from .message_bus import Message
        # PolicyGate parity (R1): validate synthetic delegate intent so phase/role/paths/red-line gates apply.
        intent = Intent(
            type=IntentType.DELEGATE,
            payload={"action_name": action_name, "params": dict(params or {})},
        )
        try:
            self.policy.validate_intent("orchestration", intent)
        except PolicyDenied as denied:
            await self._record_policy_denied("orchestration", intent, denied)
            return (
                f"(run_action_now: {action_name!r} denied by policy: "
                f"{getattr(denied, 'rule', '')!s} — "
                f"{str(getattr(denied, 'hint', denied))[:200]})"
            )
        seq_denied = self._sequence_denial_for_action(action_name, params)
        if seq_denied is not None:
            await self._record_policy_denied(
                "orchestration", intent, seq_denied, action_name=action_name,
            )
            return (
                f"(run_action_now: {action_name!r} denied: "
                f"{str(getattr(seq_denied, 'hint', seq_denied))[:200]})"
            )
        lanes, ttl = self._registry_lanes_ttl(action_name)
        content_fp = hashlib.sha1(
            json.dumps(params or {}, sort_keys=True, default=str).encode()
        ).hexdigest()[:10]
        key = (
            f"inline:orchestration:{action_name}:"
            f"t{int(self.shared_state.tick or 0)}:{content_fp}"
        )
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=action_name,
            params=dict(params or {}),
            idempotency_key=key,
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if was_existing and task.state not in (
            "queued", "succeeded", "failed", "cancelled", "needs_manual_review",
        ):
            return (
                f"(run_action_now: an identical {action_name!r} task is "
                f"already {task.state!r}; wait for its delegated_result)"
            )
        result = await self.sub.run_task(task)
        result_payload = {
            "task_id": task.task_id, "kind": task.kind,
            "state": result.state, "result": result.result,
            "error": result.error,
        }
        try:
            await self.bus.append_and_seq(Message.new(
                "coordinator", "*", "delegated_result",
                {**result_payload, "inline": True},
            ))
        except Exception:  # noqa: BLE001 — audit best-effort
            log.exception(
                "run_action_now: failed to append delegated_result for %s",
                task.task_id,
            )
        rendered = _format_inbox_event(Message.new(
            "coordinator", "orchestration", "delegated_result", result_payload,
        ))
        return f"inline run complete: {rendered}"

    def _context_analysis_reader(self) -> str:
        """Return the latest TraceLens analysis.md snapshot text."""
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

    # Resume
    def _detect_resume_state(self) -> dict[str, Any]:
        """Synchronously inspect persistence to determine if this is a resume (non-blocking)."""
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
        """Walk the event log to reconstruct ``CoordinatorState.pending_proposals``. Idempotent; a proposal is undecided when no review_verdict targets it."""
        proposal_msgs = await self.bus.tail(topic="proposal", n=10_000)
        verdicts = await self.bus.tail(topic="review_verdict", n=10_000)

        decided_ids: set[str] = set()
        verdict_by_target: dict[str, str] = {}
        for v in verdicts:
            target = v.payload.get("target_proposal_msg_id")
            if not target:
                continue
            # Backward-compat: synthesise needs_review for historical verdict_map events lacking a summary.
            summary = v.payload.get("verdict") or ""
            if not summary and isinstance(v.payload.get("verdict_map"), dict):
                summary = "needs_review"
            verdict_by_target[target] = summary
            decided_ids.add(target)

        rebuilt = 0
        self.state.pending_proposals.clear()
        for p in proposal_msgs:
            if p.msg_id in decided_ids:
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
        """Read-only snapshot of resume detection (set by ``__init__``)."""
        return dict(self._resumed_from)

    # Lifecycle
    async def stop(self) -> None:
        """Signal shutdown, cancel reactor tasks, finalize, and close the DB.

        Sets the stop event, cancels and awaits every running reactor task,
        runs the Cortex T4 safety-net finalize hook (in case the CLOSE phase
        sequencer never ran), then closes the SQLite connection. Exceptions
        raised by reactor tasks during teardown are logged, not propagated.
        """
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
        # Safety net: recipe/journal finalize when CLOSE sequencer didn't run.
        await self._cortex_t4_hook()
        self.db.close()

    async def _cortex_t4_hook(self) -> None:
        """T4 — finalize recipe at session end. Safety net for crash/Ctrl-C where CLOSE sequencer didn't run; no-op when close_sequence_done."""
        if self.cortex_kb is None:
            return
        if getattr(self.shared_state, "close_sequence_done", False):
            return
        sid = (self.shared_state.cortex_session_id or "").strip()
        if not sid:
            return
        try:
            self.cortex_finalize_recipe_and_journal()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("cortex T4 fact_finalize fallback failed")
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception("cortex T4 SharedState.save failed")

    # phase state machine
    def _ensure_phase_initialised(self) -> None:
        """Set ``phase`` + persist ``phase_budget_pct`` once per session (idempotent)."""
        state = self.shared_state
        # Phase budget normalised + persisted so CLI flags land in state.json for resume parity.
        if not state.phase_budget_pct:
            state.phase_budget_pct = dict(self._phase_budget_pct)
        current = (state.phase or "").strip().upper()
        if current in _phase_state.PHASE_NAMES:
            # Already initialised; keep CLI-side budget override authoritative.
            state.phase_budget_pct = dict(self._phase_budget_pct)
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: save after phase budget refresh failed")
            return
        # Fresh start; pre-phase-machine resume state is treated as fresh (cross-version unsupported).
        state.record_phase_transition(
            to_phase=_phase_state.PHASE_PRELUDE,
            reason="phase_entered",
            evidence={"trigger": "fresh_session"},
        )
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: save after phase init failed")

    def _ensure_cortex_t0_anchored(self) -> None:
        """v0.8 KB_gaps/Gap-12 — defensive T0 anchor for SDK callers constructed without cli plumbing. Skips when cortex_kb is None or cortex_session_id set."""
        client = self.cortex_kb
        if client is None or not getattr(client, "enabled", True):
            return
        state = self.shared_state
        if (state.cortex_session_id or "").strip():
            # cli already T0'd or resume picked up the sid; gate up here to skip the import.
            return
        # Derive workload / hw from SharedState.
        workload = (
            getattr(state, "model_name", "") or "unknown_model"
        )
        hw = getattr(state, "gpu_type", "") or "unknown_gpu"
        # marathon_dispatch_id mirrors the cli path: the hyperloom-internal manifest session id.
        extra_attrs = {
            "marathon_dispatch_id": getattr(state, "session_id", "") or "",
            "framework":   getattr(state, "framework", "") or "",
            "model_class": getattr(state, "model_class", "") or "",
            "claw_session_id":  getattr(state, "claw_session_id", "") or "",
            "sandbox_user_id":  getattr(state, "sandbox_user_id", "") or "",
            # boot_origin is a dev-debug label, NOT written to KB; distinguishes SDK-fallback from cli path.
            "boot_origin": "coordinator_fallback",
        }
        try:
            # Reuse the held dispatcher so T0 anchors the SAME local store KEEP/REVERT/CLOSE target.
            from .cortex_t0 import run_t0_anchor

            run_t0_anchor(
                client,
                state,
                workload=workload,
                hw=hw,
                extra_attrs=extra_attrs,
                fail_fast=False,
                session_dir=self.session_dir,
                save_state=True,
            )
        except Exception:  # noqa: BLE001 — defensive; helper is itself best-effort
            log.exception(
                "Coordinator T0 fallback: run_t0_anchor raised "
                "(workload=%s, hw=%s); warm_start stays empty",
                workload, hw,
            )

    def _kernel_enabled(self) -> bool:
        """Whether the kernel role is registered and enabled.

        Returns:
            ``True`` if the kernel role exists and the persisted
            ``kernel_enabled`` flag is set.
        """
        # Mirror persisted kernel_enabled flag; --no-kernel removes the kernel role.
        return "kernel" in self.role_registry and bool(
            getattr(self.shared_state, "kernel_enabled", True)
        )

    def _explore_enabled(self) -> bool:
        """Whether the EXPLORE phase is enabled for this run.

        Returns:
            ``True`` unless ``--no-explore`` disabled it (collapsing to
            KERNEL/SWEEP).
        """
        # Mirror persisted explore_enabled flag; --no-explore collapses to KERNEL/SWEEP. EXPLORE is a phase, not a role.
        return bool(getattr(self.shared_state, "explore_enabled", True))

    async def _advance_phase_if_needed(self) -> None:
        """Scan exit conditions and transition phase at most once per tick.

        Priority order (Inv-8.2): abort > exit_terminal > exit_normal, per phase_state.compute_next_phase.
        """
        state = self.shared_state
        max_hours_arg: float | None = None
        mm = float(getattr(state, "max_minutes", 0) or 0.0)
        if mm > 0:
            max_hours_arg = mm / 60.0
        next_phase = _phase_state.compute_next_phase(
            state,
            kernel_enabled=self._kernel_enabled(),
            budget_pct=self._phase_budget_pct,
            # Default True to match SharedState.framework_phase_enabled + cli resume fallback (cli.py:3231).
            framework_phase_enabled=bool(
                getattr(state, "framework_phase_enabled", True)
            ),
            explore_enabled=self._explore_enabled(),
            max_hours=max_hours_arg,
        )
        if str(state.phase or "").upper() == "EXPLORE":
            await self._maybe_enqueue_explore_research_scout()
        if next_phase is None:
            return
        target, reason, evidence = next_phase
        if target == (state.phase or "").upper():
            return  # already there
        prior = state.phase
        # Consume escalate hint after a hint-driven transition so the next tick re-evaluates fresh.
        if isinstance(evidence, dict) and (
            evidence.get("evidence") == "llm_escalation"
            or "hint" in evidence
        ):
            state.consume_pending_escalate_hint()
        # Terminal transition (target=CLOSE): mirror vocab stop_reason onto state via ENUM-validated writer.
        if (
            target == _phase_state.PHASE_CLOSE
            and isinstance(evidence, dict)
            and evidence.get("terminal")
            and reason
            and _phase_state.is_valid_stop_reason(reason)
            and not state.stop_reason
        ):
            state.set_stop_reason(reason)
        state.record_phase_transition(
            to_phase=target,
            reason=reason,
            evidence=evidence,
        )
        # #266: mirror the phase boundary into the operator-facing
        # lifecycle log so a launcher poll surfaces "entered <phase>" in
        # chat (with the human-friendly label) alongside the step-level
        # events. Uses the ENTER status (not START): a phase boundary is a
        # point-in-time marker, not a paired START/END interval, so it must
        # not read as "still running" forever. Best-effort; must never roll
        # back the transition.
        try:
            state.record_lifecycle_event(
                step=target,
                status=_phase_state.LIFECYCLE_STATUS_ENTER,
                phase=target,
                detail=f"reason={reason}" if reason else "",
            )
        except Exception:  # noqa: BLE001 — defensive
            log.debug("Coordinator: lifecycle phase emit failed", exc_info=True)
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: save after phase transition failed")
        log.info(
            "Coordinator.phase: %s → %s (reason=%s)",
            prior or "<unset>", target, reason,
        )
        try:
            await self.bus.append_and_seq(Message.new(
                "coordinator", "*", "event",
                {
                    "kind":       "phase_transition",
                    "from_phase": prior or "",
                    "to_phase":   target,
                    "reason":     reason,
                    "evidence":   evidence,
                },
            ))
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: phase_transition event bus write failed")
        # Phase-entry side effects are additive; hook failures are logged but never roll back the transition.
        try:
            await self._on_phase_entered(from_phase=prior or "", to_phase=target)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: _on_phase_entered hook failed")

    async def _on_phase_entered(self, *, from_phase: str, to_phase: str) -> None:
        """Fire per-phase entry side effects (pure dispatcher; hooks catch + log internally). CLOSE runs the 5-step sequencer (KB_design §3.2 §5.5 + KB_gaps/Gap-06; sets close_sequence_done)."""
        # Orchestration checkpoint at the phase seam (plan Step 4); runs before per-phase side effects.
        try:
            await self._maybe_checkpoint_orchestration(
                tick=int(getattr(self.shared_state, "tick", 0) or 0),
                phase_changed=True,
            )
        except Exception:  # noqa: BLE001
            log.exception("Coordinator: phase-boundary checkpoint failed")

        target = (to_phase or "").upper()
        if target == _phase_state.PHASE_FRAMEWORK_PR:
            await self._on_enter_framework_pr(from_phase=from_phase)
        elif target == _phase_state.PHASE_EXPLORE:
            await self._on_enter_explore(from_phase=from_phase)
        elif target == _phase_state.PHASE_KERNEL:
            await self._on_enter_kernel(from_phase=from_phase)
        elif target == _phase_state.PHASE_SWEEP:
            await self._on_enter_sweep(from_phase=from_phase)
        elif target == _phase_state.PHASE_CLOSE:
            await self._on_enter_close(from_phase=from_phase)

    async def _on_enter_explore(self, *, from_phase: str) -> None:
        """Warm ``KnowledgePlane.pr_feed`` across specialist domains (best-effort) on EXPLORE entry. Roofline lives in PRELUDE, not here."""
        plane = self.knowledge_plane
        if plane is None:
            return
        try:
            results = plane.pr_feed_warm_all_domains()
            total_prs = sum(len(prs) for prs, _w in results.values())
            log.info(
                "EXPLORE entry (from=%s): warmed pr_feed across %d "
                "domains (total PRs cached=%d)",
                from_phase or "<unknown>", len(results), total_prs,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "EXPLORE entry: pr_feed_warm_all_domains failed: %r", exc,
            )

    async def _on_enter_framework_pr(self, *, from_phase: str) -> None:
        """FRAMEWORK_PR entry hook: trigger the per-batch pump once on entry (best-effort; later batches driven from the main tick)."""
        log.info(
            "FRAMEWORK_PR entry (from=%s): pumping initial batch",
            from_phase or "<unknown>",
        )
        try:
            await self._pump_framework_pr_phase()
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning("FRAMEWORK_PR entry pump failed: %r", exc)

    async def _pump_framework_pr_phase(self) -> None:
        """Drive the FRAMEWORK_PR phase: enqueue the next candidate. Idempotent; a discover failure flips framework_pr_phase_done so the phase advances rather than wedging."""
        state = self.shared_state
        if (state.phase or "").strip().upper() != _phase_state.PHASE_FRAMEWORK_PR:
            return
        if bool(getattr(state, "framework_pr_phase_done", False)):
            return
        # Skip if a framework_pr task is already queued or running.
        try:
            queued = await self.tasks.queued()
            running = await self.tasks.running()
        except Exception:  # noqa: BLE001 — defensive
            queued, running = [], []
        for t in (*queued, *running):
            if getattr(t, "kind", "") == "framework_pr":
                return
        # Find the next un-dispatched candidate, or request a new batch if exhausted.
        next_candidate = self._select_next_framework_pr_candidate()
        if next_candidate is None:
            # Hold the phase open while authored patches are still benched/critic-reviewed (gains must land before plateau judge); gated by authoring flag.
            if (
                getattr(self.shared_state, "framework_pr_authoring_enabled", False)
                and await self._framework_pr_authoring_inflight()
            ):
                return
            # Discover a fresh batch; only DISCOVER_FAILURE_RETRY_LIMIT consecutive failures or an empty-but-valid payload mark the phase done.
            from . import framework_agent_client as _fa_client
            ok = await self._discover_next_framework_pr_batch()
            if not ok:
                failures = int(
                    getattr(state, "framework_pr_discover_failures", 0) or 0
                )
                if failures >= _fa_client.DISCOVER_FAILURE_RETRY_LIMIT or failures == 0:
                    # Retries exhausted or clean empty payload — both real exits; stamp a summary row.
                    self._record_framework_pr_phase_done(
                        reason=(
                            "discover_retries_exhausted"
                            if failures >= _fa_client.DISCOVER_FAILURE_RETRY_LIMIT
                            else "discover_empty_payload"
                        ),
                        failure_count=failures,
                    )
                    state.framework_pr_phase_done = True
                    state.save(self.session_dir)
                return
            next_candidate = self._select_next_framework_pr_candidate()
            if next_candidate is None:
                self._record_framework_pr_phase_done(
                    reason="discover_returned_no_new_candidates",
                    failure_count=int(
                        getattr(state, "framework_pr_discover_failures", 0) or 0,
                    ),
                )
                state.framework_pr_phase_done = True
                state.save(self.session_dir)
                return
        # Critic gate before apply: reject short-circuits with a critic_denied row; approve/abstain enqueues.
        verdict = await self._critic_review_framework_pr_candidate(next_candidate)
        if verdict.get("verdict") == "reject":
            cand_id = str(
                next_candidate.get("candidate_id")
                or next_candidate.get("pr_url")
                or "",
            )
            progress = getattr(state, "framework_pr_phase_progress", None)
            if not isinstance(progress, list):
                progress = []
                state.framework_pr_phase_progress = progress
            progress.append({
                "candidate_id": cand_id,
                "batch_id":     next_candidate.get("batch_id") or "",
                "task_id":      None,
                "status":       "critic_denied",
                "rationale":    str(verdict.get("rationale") or ""),
                "ts":           datetime.now(timezone.utc).isoformat(),
            })
            state.save(self.session_dir)
            log.info(
                "FRAMEWORK_PR: critic rejected candidate=%s batch=%s "
                "rationale=%r",
                cand_id, next_candidate.get("batch_id") or "",
                str(verdict.get("rationale") or "")[:200],
            )
            return
        await self._enqueue_framework_pr_task(next_candidate)
        # Authoring track (PR-G): also hand the PR to a write-capable specialist to author its own patch.
        if getattr(state, "framework_pr_authoring_enabled", False):
            try:
                await self._enqueue_framework_pr_authoring_specialist(
                    next_candidate,
                )
            except Exception as exc:  # noqa: BLE001 — never wedge the pump
                log.warning(
                    "FRAMEWORK_PR: authoring specialist dispatch failed: %r",
                    exc,
                )

    async def _framework_pr_authoring_inflight(self) -> bool:
        """True while a FRAMEWORK_PR-authored patch is still in flight (specialist/integrate_patch task or pending integrate_patch proposal); pump waits before advancing."""
        try:
            queued = await self.tasks.queued()
            running = await self.tasks.running()
        except Exception:  # noqa: BLE001 — defensive
            queued, running = [], []
        for t in (*queued, *running):
            if getattr(t, "kind", "") in ("specialist", "integrate_patch"):
                return True
        # An authored patch may sit in the Critic queue before the integrate_patch task.
        try:
            for p in self.state.pending_proposals.values():
                if getattr(p, "action_name", "") == "integrate_patch":
                    return True
        except Exception:  # noqa: BLE001 — defensive
            pass
        return False

    async def _enqueue_framework_pr_authoring_specialist(
        self, candidate: dict[str, Any],
    ) -> None:
        """Dispatch a write-capable specialist seeded with ``candidate`` (Inv-5.1: flows through autosubmit → Critic → integrate_patch → bench → KEEP/REVERT)."""
        state = self.shared_state
        cand_id = str(
            candidate.get("candidate_id")
            or candidate.get("pr_url")
            or candidate.get("ref")
            or ""
        )
        batch_id = str(candidate.get("batch_id") or "")
        gap_cid = (
            str(candidate.get("gap_canonical_id") or "").strip()
            or f"gap.framework_pr.{cand_id}"
        )
        title = str(candidate.get("title") or "").strip()
        pr_url = str(candidate.get("pr_url") or "").strip()
        diff_url = str(candidate.get("diff_url") or "").strip()
        notes = "\n".join([
            "FRAMEWORK_PR AUTHORING TASK.",
            "",
            "A candidate upstream PR was discovered as a lead for this gap.",
            "Study it as INSPIRATION, then author your OWN source patch into",
            "your worktree. You are NOT limited to copying the PR's diff — go",
            "beyond it where the live source + profile evidence justify a",
            "stronger or more targeted change. If, after reading the source,",
            "the upstream change is already optimal, you may reproduce its",
            "essential edit, but prefer a patch tailored to this model /",
            "hardware / workload.",
            "",
            f"- PR title: {title or '(none)'}",
            f"- PR url: {pr_url or '(none)'}",
            f"- Unified diff: {diff_url or '(none)'}"
            " (fetch with WebFetch to read the upstream change)",
            "",
            "Deliverable: a unified-diff patch file in your worktree, listed in",
            "``patches_written``. The Coordinator applies + benches it and",
            "decides KEEP/REVERT; you do not benchmark.",
        ])
        params: dict[str, Any] = {
            "domain": "serving_specialist",
            "gap_canonical_id": gap_cid,
            "gap_symptom": (
                title
                or f"Author a framework source patch inspired by "
                f"{pr_url or cand_id}"
            ),
            "gap_layer": "framework",
            "framework": str(
                candidate.get("framework")
                or getattr(state, "framework", "") or ""
            ).strip().lower(),
            # Provenance markers so the dispatcher-side bridge recognises an authored FRAMEWORK_PR patch.
            "framework_pr_authoring": True,
            "framework_pr_candidate_id": cand_id,
            "framework_pr_batch_id": batch_id,
            "source": "coordinator_internal",
            "readonly": False,
            "notes": notes,
        }
        try:
            await self._warm_specialist_params(params)
        except Exception:  # noqa: BLE001 — best-effort warmup
            log.debug(
                "FRAMEWORK_PR authoring: warm specialist params failed",
                exc_info=True,
            )
        idem = f"framework_pr_authoring:{batch_id}:{cand_id}"
        await self.tasks.create_or_return_existing(
            kind="specialist",
            params=params,
            idempotency_key=idem,
            requires_lanes=["research_lane"],
            allowed_tools=[
                "Read", "Grep", "Glob", "Write", "Edit", "Bash",
                "WebSearch", "WebFetch",
            ],
            side_effects=["writes_results", "writes_patches"],
            lease_ttl_sec=3600,
        )
        log.info(
            "FRAMEWORK_PR: dispatched authoring specialist candidate=%s "
            "batch=%s gap=%s",
            cand_id, batch_id, gap_cid,
        )

    def _select_next_framework_pr_candidate(self) -> dict[str, Any] | None:
        """Return the next unprocessed candidate in the latest batch (processed = has progress entry)."""
        state = self.shared_state
        batches = getattr(state, "framework_pr_batches", None) or []
        if not batches:
            return None
        latest = batches[-1]
        if not isinstance(latest, dict):
            return None
        candidates = latest.get("candidates") or []
        if not isinstance(candidates, list):
            return None
        processed = {
            str(p.get("candidate_id") or "")
            for p in (getattr(state, "framework_pr_phase_progress", None) or [])
            if isinstance(p, dict)
        }
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            cand_id = str(
                cand.get("candidate_id")
                or cand.get("pr_url")
                or cand.get("ref")
                or ""
            )
            if cand_id and cand_id not in processed:
                return cand
        return None

    def _framework_pr_known_candidate_ids(self) -> set[str]:
        """All candidate ids already discovered into any prior batch (dedup for new batches)."""
        state = self.shared_state
        ids: set[str] = set()
        batches = getattr(state, "framework_pr_batches", None) or []
        if not isinstance(batches, list):
            return ids
        for batch in batches:
            if not isinstance(batch, dict):
                continue
            for cand in batch.get("candidates") or []:
                if not isinstance(cand, dict):
                    continue
                cid = str(
                    cand.get("candidate_id")
                    or cand.get("pr_url")
                    or cand.get("ref")
                    or f"{cand.get('repo','')}-{cand.get('pr_number','')}"
                )
                if cid:
                    ids.add(cid)
        # Fold in PR ids the research scout already mined so the two mechanisms never re-process a PR.
        for pid in getattr(state, "research_scout_seen_pr_ids", None) or []:
            pid = str(pid or "").strip()
            if pid:
                ids.add(pid)
        return ids

    def _framework_pr_tried_refs(self) -> list[str]:
        """Refs already discovered this phase (fed to compose_gap to bias away from prior PR categories)."""
        refs: list[str] = []
        for cid in self._framework_pr_known_candidate_ids():
            if cid:
                refs.append(cid)
        return refs

    def _framework_pr_discover_repo_urls(self, framework: str) -> list[str]:
        """Repo URLs to query for the FRAMEWORK_PR batch: framework's own repo + pr_intel_specialist cross-repo set, dedup preserving order."""
        from . import framework_agent_client as _fa_client

        urls: list[str] = []

        def _add(u: str) -> None:
            """Append a trimmed URL to ``urls`` if non-empty and not already present.

            Args:
                u (str): A candidate repo URL.
            """
            u = (u or "").strip()
            if u and u not in urls:
                urls.append(u)

        # Primary: the framework's own repo.
        _add(_fa_client.repo_url_for_framework(framework))

        # Cross-repo: the pr_intel_specialist repo set (owner/name -> URL).
        try:
            from .specialist_domains import get_domain
            domain = get_domain("pr_intel_specialist")
            for repo in getattr(domain, "pr_repos", ()) or ():
                repo = str(repo or "").strip()
                if not repo:
                    continue
                if repo.startswith("http"):
                    _add(repo)
                elif "/" in repo:
                    _add(f"https://github.com/{repo}.git")
        except Exception:  # noqa: BLE001 — defensive
            pass

        if not urls:
            # Last-ditch: let phase_discover resolve from framework itself.
            _add(_fa_client.repo_url_for_framework(framework or "sglang"))
        return urls

    def _record_framework_pr_phase_done(
        self, *, reason: str, failure_count: int,
    ) -> None:
        """Append a framework_pr_phase_done row to phase_history describing why the pump gave up."""
        state = self.shared_state
        try:
            history = getattr(state, "phase_history", None)
            if not isinstance(history, list):
                return
            from . import framework_agent_client as _fa_client
            history.append({
                "event":              "framework_pr_phase_done",
                "reason":             reason,
                "failure_count":      int(failure_count),
                "retry_limit":        int(_fa_client.DISCOVER_FAILURE_RETRY_LIMIT),
                "batches_discovered": len(getattr(state, "framework_pr_batches", None) or []),
                "ts":                 datetime.now(timezone.utc).isoformat(),
            })
        except Exception:  # noqa: BLE001 — defensive
            pass

    async def _discover_next_framework_pr_batch(self) -> bool:
        """Call ``fa phase-discover`` and append a batch to SharedState. Returns True iff a non-empty batch was appended; transient failures return False (see DISCOVER_FAILURE_RETRY_LIMIT)."""
        from . import framework_agent_client as _fa_client

        state = self.shared_state
        # Directed gap composition: seed search from latest bottleneck + workload taxonomy via compose_gap,
        # then merge structured state.gaps so each batch retargets the current bottleneck.
        framework = ""
        try:
            framework = str(getattr(state, "framework", "") or "").strip().lower()
        except Exception:  # noqa: BLE001
            framework = ""
        directed_keywords: list[str] = []
        directed_gap = ""
        try:
            from .action_executors._framework_gap_composer import compose_gap
            directed_gap, directed_keywords = compose_gap(
                framework=framework,
                gpu_type=str(getattr(state, "gpu_type", "") or ""),
                model_class=str(getattr(state, "model_class", "") or ""),
                precision=str(getattr(state, "precision", "") or ""),
                profile_kernel_breakdown_path=getattr(
                    state, "last_profile_kernel_breakdown", None,
                ),
                tried_refs=self._framework_pr_tried_refs(),
            )
        except Exception:  # noqa: BLE001 — defensive
            directed_gap, directed_keywords = "", []
        gaps: list[dict[str, str]] = []
        try:
            gap_list = getattr(state, "gaps", None) or []
            for g in gap_list:
                if not isinstance(g, dict):
                    continue
                gaps.append({
                    "gap_canonical_id": str(g.get("canonical_id") or ""),
                    "gap_description":  str(
                        g.get("symptom") or g.get("description") or ""
                    ),
                })
        except Exception:  # noqa: BLE001 — defensive
            gaps = []
        if directed_gap:
            # Prepend the directed gap so fa's search leads with bottleneck-aware phrasing; de-dup.
            existing = {str(g.get("gap_description") or "") for g in gaps}
            if directed_gap not in existing:
                gaps.insert(0, {
                    "gap_canonical_id": "directed",
                    "gap_description":  directed_gap,
                })
        if not gaps:
            gaps = [{"gap_canonical_id": "", "gap_description": ""}]
        timeout_sec = float(
            getattr(self, "framework_pr_discover_timeout_sec", 0.0)
            or _fa_client.DEFAULT_FA_PHASE_TIMEOUT_SEC
        )
        max_candidates = int(
            getattr(state, "framework_pr_max_candidates", 0) or 0
        ) or DEFAULT_FRAMEWORK_PR_MAX_CANDIDATES
        # Cross-repo: query every pr_intel_specialist repo so discovery isn't confined to one framework repo.
        repo_urls = self._framework_pr_discover_repo_urls(framework)
        payload: dict[str, Any] | None = None
        merged_candidates: list[dict[str, Any]] = []
        batch_id = ""
        any_call_ok = False
        last_exc: Exception | None = None
        # Spread the phase timeout across repos so one slow repo can't blow the whole budget.
        per_repo_timeout = (
            timeout_sec / float(len(repo_urls)) if repo_urls else timeout_sec
        )
        per_repo_timeout = max(per_repo_timeout, 30.0)
        for repo_url in repo_urls:
            try:
                repo_payload = await _fa_client.phase_discover(
                    model=str(getattr(state, "model", "") or ""),
                    framework=framework or "sglang",
                    gpu_type=str(getattr(state, "gpu_type", "") or ""),
                    gaps=gaps,
                    session_dir=self.session_dir,
                    repo_url=repo_url,
                    keywords=directed_keywords,
                    max_candidates=max_candidates,
                    timeout_sec=per_repo_timeout,
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                last_exc = exc
                log.warning(
                    "fa phase-discover failed for repo_url=%r: %r",
                    repo_url, exc,
                )
                continue
            any_call_ok = True
            if payload is None:
                payload = repo_payload
            if not batch_id:
                batch_id = str((repo_payload or {}).get("batch_id") or "")
            repo_cands = (repo_payload or {}).get("candidates") or []
            if isinstance(repo_cands, list):
                merged_candidates.extend(
                    c for c in repo_cands if isinstance(c, dict)
                )
        if not any_call_ok:
            failures = int(getattr(state, "framework_pr_discover_failures", 0) or 0) + 1
            state.framework_pr_discover_failures = failures
            log.warning(
                "fa phase-discover failed across all %d repo(s) "
                "(attempt %d/%d): %r",
                len(repo_urls), failures,
                _fa_client.DISCOVER_FAILURE_RETRY_LIMIT, last_exc,
            )
            try:
                history = getattr(state, "phase_history", None)
                if isinstance(history, list):
                    history.append({
                        "event":   "framework_pr_discover_failed",
                        "attempt": failures,
                        "limit":   _fa_client.DISCOVER_FAILURE_RETRY_LIMIT,
                        "error":   repr(last_exc),
                        "ts":      datetime.now(timezone.utc).isoformat(),
                    })
            except Exception:  # noqa: BLE001 — defensive
                pass
            state.save(self.session_dir)
            return False
        # Successful call — reset failure counter regardless of whether
        # the payload contained candidates.
        if int(getattr(state, "framework_pr_discover_failures", 0) or 0) != 0:
            state.framework_pr_discover_failures = 0
        if not merged_candidates:
            return False
        batch_id = str((payload or {}).get("batch_id") or "")
        # Cross-batch + cross-repo de-dup so the new batch only carries genuinely new PRs.
        seen_ids = self._framework_pr_known_candidate_ids()
        primary_repo_url = repo_urls[0] if repo_urls else ""
        # Normalise each candidate for consistent executor fields + a stable progress-ledger id.
        norm: list[dict[str, Any]] = []
        for c in merged_candidates:
            if not isinstance(c, dict):
                continue
            cand_id = str(
                c.get("pr_url") or c.get("ref")
                or f"{c.get('repo','')}-{c.get('pr_number','')}"
            )
            if cand_id and cand_id in seen_ids:
                continue
            seen_ids.add(cand_id)
            # Stamp the candidate's repo URL so the executor knows same-repo (fetchable) vs foreign (diff_url).
            discovered_repo_url = str(
                c.get("repo_url") or c.get("discovered_repo_url")
                or primary_repo_url
            )
            norm.append({
                **c,
                "candidate_id": cand_id,
                "batch_id": batch_id,
                "discovered_repo_url": discovered_repo_url,
            })
        if not norm:
            return False
        batch_entry = {
            "batch_id": batch_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "candidate_count": len(norm),
            "candidates": norm,
            "max_gain_pct_observed_in_batch": 0.0,
        }
        if not isinstance(state.framework_pr_batches, list):
            state.framework_pr_batches = []
        state.framework_pr_batches.append(batch_entry)
        state.save(self.session_dir)
        log.info(
            "FRAMEWORK_PR: discovered batch=%s with %d candidates",
            batch_id or "<unset>", len(norm),
        )
        return True

    async def _enqueue_framework_pr_task(self, candidate: dict[str, Any]) -> None:
        """Enqueue a single ``framework_pr`` task for ``candidate``.

        Builds the task params (candidate, batch id, baseline throughput,
        framework) and creates an idempotent ``framework_pr`` task holding the
        server / workspace / benchmark lanes. On enqueue failure, records an
        ``enqueue_failed`` progress row so the pump skips the candidate next
        tick instead of spinning.

        Args:
            candidate (dict[str, Any]): The discovered PR candidate to apply
                and benchmark.
        """
        state = self.shared_state
        params = {
            "candidate": candidate,
            "batch_id": candidate.get("batch_id") or "",
            "base_tput": float(getattr(state, "baseline_tput", 0.0) or 0.0),
            "framework": str(
                candidate.get("framework")
                or getattr(state, "framework", "") or ""
            ).strip().lower(),
        }
        cand_id = str(candidate.get("candidate_id") or candidate.get("pr_url") or "")
        idem = f"framework_pr:{candidate.get('batch_id','')}:{cand_id}"
        try:
            await self.tasks.create_or_return_existing(
                kind="framework_pr",
                params=params,
                idempotency_key=idem,
                requires_lanes=[
                    "server_lifecycle",
                    "workspace_mutation",
                    "benchmark_lane",
                ],
            )
            log.info(
                "FRAMEWORK_PR: enqueued candidate=%s batch=%s",
                cand_id, candidate.get("batch_id") or "",
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "FRAMEWORK_PR: failed to enqueue candidate=%s: %r",
                cand_id, exc,
            )
            # Record enqueue_failed progress row so the candidate is skipped next tick (else the loop spins).
            progress = getattr(state, "framework_pr_phase_progress", None)
            if not isinstance(progress, list):
                progress = []
                state.framework_pr_phase_progress = progress
            progress.append({
                "candidate_id": cand_id,
                "batch_id":     candidate.get("batch_id") or "",
                "task_id":      None,
                "status":       "enqueue_failed",
                "error":        repr(exc),
                "ts":           datetime.now(timezone.utc).isoformat(),
            })
            state.save(self.session_dir)

    _CRITIC_PRIORS_DECISION_TAIL: int = 5
    _CRITIC_PRIORS_OUTCOME_TAIL: int = 5

    def _collect_framework_pr_priors(self) -> dict[str, Any]:
        """Return compact session-local priors for the Critic gate (recent_decisions + recent_outcomes); best-effort."""
        state = self.shared_state
        decisions: list[dict[str, Any]] = []
        try:
            raw_decisions = getattr(state, "framework_pr_critic_decisions", None) or []
            for row in raw_decisions[-self._CRITIC_PRIORS_DECISION_TAIL:]:
                if not isinstance(row, dict):
                    continue
                decisions.append({
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "verdict":      str(row.get("verdict") or ""),
                    "rationale":    str(row.get("rationale") or "")[:200],
                })
        except Exception:  # noqa: BLE001
            decisions = []
        outcomes: list[dict[str, Any]] = []
        try:
            raw_progress = getattr(state, "framework_pr_phase_progress", None) or []
            terminal = {"kept", "reverted", "no_patch", "enqueue_failed", "critic_denied"}
            tail = [r for r in raw_progress if isinstance(r, dict) and str(r.get("status") or "") in terminal]
            for row in tail[-self._CRITIC_PRIORS_OUTCOME_TAIL:]:
                outcomes.append({
                    "candidate_id": str(row.get("candidate_id") or ""),
                    "status":       str(row.get("status") or ""),
                    "gain_pct":     row.get("gain_pct"),
                })
        except Exception:  # noqa: BLE001
            outcomes = []
        return {
            "recent_decisions": decisions,
            "recent_outcomes":  outcomes,
        }

    async def _critic_review_framework_pr_candidate(
        self, candidate: dict[str, Any],
    ) -> dict[str, str]:
        """Ask the Critic backend whether to apply ``candidate``.

        Returns {"verdict": "approve"|"reject"|"abstain", "rationale": str}. abstain is the safe degraded
        path (treated as approve by caller); decisions cached in framework_pr_critic_decisions for resume.
        """
        state = self.shared_state
        cand_id = str(
            candidate.get("candidate_id")
            or candidate.get("pr_url")
            or "",
        )
        # Resume-safe cache lookup.
        cached = getattr(state, "framework_pr_critic_decisions", None)
        if isinstance(cached, list):
            for row in cached:
                if not isinstance(row, dict):
                    continue
                if str(row.get("candidate_id") or "") == cand_id and cand_id:
                    return {
                        "verdict":   str(row.get("verdict") or "abstain"),
                        "rationale": str(row.get("rationale") or ""),
                    }
        critic_backend = self.backends.get("critic")
        if critic_backend is None:
            return {"verdict": "abstain", "rationale": "no critic backend"}
        # Proposal-formatted prompt for both Mock + Agent critic backends; deterministic all-hex msg_id
        # from candidate id so MockCriticBackend's [a-f0-9]+ regex + dedupe set stay consistent.
        msg_id = hashlib.md5(
            f"framework_pr:{cand_id}".encode(),
        ).hexdigest()
        payload = {
            "action":     "framework_pr",
            "candidate":  {
                "candidate_id":     cand_id,
                "pr_url":           str(candidate.get("pr_url") or ""),
                "diff_url":         str(candidate.get("diff_url") or ""),
                "repo":             str(candidate.get("repo") or ""),
                "ref":              str(candidate.get("ref") or ""),
                "title":            str(candidate.get("title") or ""),
                "framework":        str(candidate.get("framework") or ""),
                "gap_canonical_id": str(candidate.get("gap_canonical_id") or ""),
                "rationale":        str(candidate.get("rationale") or ""),
            },
            "batch_id":   candidate.get("batch_id") or "",
            # Session-local priors (classified candidates + apply/bench outcomes); bounded to keep prompt compact.
            "priors":     self._collect_framework_pr_priors(),
        }
        prompt = (
            f"seq=1 msg_id={msg_id} from=coordinator topic=proposal "
            f"payload={json.dumps(payload, sort_keys=True)}"
        )
        verdict_row: dict[str, str] = {
            "verdict":   "abstain",
            "rationale": "no verdict emitted",
        }
        try:
            result = await critic_backend.run(
                prompt=prompt,
                system_prompt=None,
                tools=[],
                max_turns=1,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "FRAMEWORK_PR: critic call failed for candidate=%s: %r",
                cand_id, exc,
            )
            verdict_row = {
                "verdict":   "abstain",
                "rationale": f"critic call failed: {exc!r}",
            }
        else:
            for intent in getattr(result, "intents", []) or []:
                itype = getattr(intent, "type", None)
                itype_val = getattr(itype, "value", itype)
                if str(itype_val) != "review_verdict":
                    continue
                ipayload = getattr(intent, "payload", {}) or {}
                if not isinstance(ipayload, dict):
                    continue
                if str(ipayload.get("target_proposal_msg_id") or "") != msg_id:
                    continue
                v = str(ipayload.get("verdict") or "").strip().lower()
                # Map Critic verdict vocab onto {approve, reject, abstain}; redirect/advise/needs_review -> abstain.
                if v == "approve":
                    mapped = "approve"
                elif v == "reject":
                    mapped = "reject"
                else:
                    mapped = "abstain"
                verdict_row = {
                    "verdict":   mapped,
                    "rationale": str(
                        ipayload.get("reasoning")
                        or ipayload.get("rationale")
                        or "",
                    ),
                }
                break
        decisions = getattr(state, "framework_pr_critic_decisions", None)
        if not isinstance(decisions, list):
            decisions = []
            state.framework_pr_critic_decisions = decisions
        decisions.append({
            "candidate_id": cand_id,
            "batch_id":     candidate.get("batch_id") or "",
            "verdict":      verdict_row["verdict"],
            "rationale":    verdict_row["rationale"],
            "ts":           datetime.now(timezone.utc).isoformat(),
        })
        state.save(self.session_dir)
        return verdict_row

    async def _on_enter_kernel(self, *, from_phase: str) -> None:
        """Run deterministic KERNEL-entry setup before LLM kernel work (FP8 GEMM tuning gate)."""
        if not self._kernel_enabled():
            # Should not happen — --no-kernel routes EXPLORE → SWEEP.
            log.info(
                "KERNEL entry hook fired with kernel_enabled=False "
                "(from=%s)", from_phase or "<unknown>",
            )
            return
        if not self._gemm_tuning_required_before_kernel_opt():
            return

        log.info(
            "KERNEL entry: running FP8 GEMM tuning before source-level kernel_opt",
        )
        self._record_phase_entry_evidence(
            gemm_tuning={"status": "running", "source": "kernel_entry_auto"},
        )
        try:
            from .kernel_request_handlers import run_gemm_tuning_handler

            result = await run_gemm_tuning_handler(
                {
                    "task_id": "kernel_entry_gemm_tuning",
                    "reason": "kernel_entry_auto",
                },
                session_dir=self.session_dir,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("KERNEL entry GEMM tuning failed")
            result = {
                "status": "failed",
                "decision": "REVERT",
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        self.shared_state.record_gemm_tuning(result)
        self._promote_gemm_tuning_keep(result)
        self.shared_state.save(self.session_dir)
        status = str(result.get("status") or "unknown")
        await self.bus.append_and_seq(Message.new(
            "kernel",
            "orchestration",
            "response",
            {
                "in_reply_to": "",
                "kind": "run_gemm_tuning_done",
                "status": status,
                "result": result,
                "source": "kernel_entry_auto",
            },
            priority=1,
        ))
        self._record_phase_entry_evidence(
            gemm_tuning={
                "status": "done" if status in {"ok", "complete", "succeeded"} else status,
                "source": "kernel_entry_auto",
                "best_speedup": result.get("best_speedup"),
                "tuned_file": result.get("tuned_file"),
            },
        )
        if self._should_continue_kernel_after_gemm():
            await self._run_kernel_opt_after_gemm()

    def _promote_gemm_tuning_keep(self, result: dict[str, Any]) -> None:
        """Promote a successful GEMM tuning run into the main gain ledger.

        Only acts on a successful, ``KEEP``-decision result with a speedup
        greater than 1.0 and a known baseline. Appends an entry to the
        optimization stack (deduped on tuned file), updates ``current_best``,
        and stamps ``cumulative_gain`` / ``cumulative_gain_validated`` since
        the GEMM benchmark is itself an end-to-end serving measurement.

        Args:
            result (dict[str, Any]): The GEMM tuning handler result; ignored if
                not a successful KEEP.
        """
        if not isinstance(result, dict):
            return
        status = str(result.get("status") or "").strip().lower()
        decision = str(result.get("decision") or "").strip().upper()
        if status not in {"ok", "complete", "completed", "succeeded", "success"}:
            return
        if decision != "KEEP":
            return
        try:
            speedup = float(result.get("best_speedup") or 0.0)
            baseline = float(self.shared_state.baseline_tput or 0.0)
        except (TypeError, ValueError):
            return
        if speedup <= 1.0 or baseline <= 0:
            return
        tuned_tput = baseline * speedup
        tuned_file = str(result.get("tuned_file") or "")
        final_report = str(result.get("final_report_path") or "")
        existing = {
            str(item.get("tuned_file") or "")
            for item in (self.shared_state.optimization_stack or [])
            if isinstance(item, dict) and item.get("action") == "gemm_tuning"
        }
        ts = datetime.now(timezone.utc).isoformat()
        entry = {
            "action": "gemm_tuning",
            "variant_name": "a8w8_blockscale_tuned_gemm",
            "tuned_file": tuned_file,
            "final_report_path": final_report,
            "gain_pct": (speedup - 1.0) * 100.0,
            "tput": tuned_tput,
            "workspace": result.get("workspace"),
            "extra_envs": (
                {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": tuned_file}
                if tuned_file else {}
            ),
            "source": "kernel_entry_auto",
            "ts": ts,
        }
        if tuned_file not in existing:
            self.shared_state.optimization_stack.append(entry)
            self.shared_state.append_stack_gain_entry(
                action="gemm_tuning",
                variant_name="a8w8_blockscale_tuned_gemm",
                new_tput=tuned_tput,
                ts=ts,
            )
        self.shared_state.current_best = {
            "action": "gemm_tuning",
            "tput": tuned_tput,
            "variant_name": "a8w8_blockscale_tuned_gemm",
            "tuned_file": tuned_file,
            "final_report_path": final_report,
            "workspace": result.get("workspace"),
            "extra_envs": entry["extra_envs"],
        }
        self.shared_state.cumulative_gain = (speedup - 1.0) * 100.0
        # GEMM's tuned benchmark is end-to-end serving, so it's already a validated stack measurement.
        self.shared_state.cumulative_gain_validated = self.shared_state.cumulative_gain
        self.shared_state.cumulative_gain_validated_ts = ts
        self.shared_state.cumulative_gain_validated_stack_len = len(
            self.shared_state.optimization_stack or []
        )

    def _should_continue_kernel_after_gemm(self) -> bool:
        """Decide whether to run source-level kernel_opt right after GEMM tuning.

        Returns:
            bool: ``True`` when the ``continue_kernel_after_gemm`` flag is set
                and there are untried hot reusable kernels remaining.
        """
        if not bool(getattr(self.shared_state, "continue_kernel_after_gemm", True)):
            return False
        return bool(self.shared_state.untried_hot_reusable_kernels())

    async def _run_kernel_opt_after_gemm(self) -> None:
        """Run the source-level kernel optimization batch after GEMM tuning."""
        cached = self.shared_state.last_trace_analyze or {}
        candidates_path = str(cached.get("candidates_path") or "")
        if not candidates_path:
            log.info("KERNEL entry: skip kernel_opt after GEMM; no candidates_path")
            return
        log.info(
            "KERNEL entry: continuing to source-level kernel_opt after GEMM tuning",
        )
        try:
            from .kernel_request_handlers import run_optimization_handler

            result = await run_optimization_handler(
                {
                    "candidates_path": candidates_path,
                    "session_id": self.session_dir.name,
                },
                session_dir=self.session_dir,
                record_partial=self._record_kernel_opt_partial,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("KERNEL entry run_optimization after GEMM failed")
            result = {
                "status": "failed",
                "error_class": exc.__class__.__name__,
                "error": repr(exc),
            }
        await self.bus.append_and_seq(Message.new(
            "kernel",
            "orchestration",
            "response",
            {
                "in_reply_to": "",
                "kind": "run_optimization_done",
                "status": result.get("status", "ok") if isinstance(result, dict) else "failed",
                "result": result,
                "source": "kernel_entry_auto_after_gemm",
            },
            priority=1,
        ))
        if isinstance(result, dict) and not result.get("batch_mode"):
            self.shared_state.record_kernel_opt(result)
        self.shared_state.save(self.session_dir)

    # Auto-roofline — PRELUDE bootstrap + 10% watermark refresh anchored on last_roofline_tput.
    _ROOFLINE_WATERMARK_RATIO: float = 1.10   # 10% step over last roofline

    def _current_tput_from_validated_gain(self) -> float:
        """Project current tput from ``baseline_tput * (1 + cumulative_gain_validated/100)``; 0.0 when baseline unknown (watermark not-yet-armed)."""
        state = self.shared_state
        try:
            base = float(state.baseline_tput or 0.0)
        except (TypeError, ValueError):
            base = 0.0
        if base <= 0:
            return 0.0
        try:
            gain = float(state.cumulative_gain_validated or 0.0)
        except (TypeError, ValueError):
            gain = 0.0
        return base * (1.0 + gain / 100.0)

    def _needs_roofline_for_watermark(self) -> bool:
        """True iff projected tput crossed the 10% watermark over ``last_roofline_tput`` (bootstrap guard: False until PRELUDE roofline ran; re-arm guard: False while auto_roofline_pending_task_id is in-flight)."""
        state = self.shared_state
        try:
            last_rl = float(state.last_roofline_tput or 0.0)
        except (TypeError, ValueError):
            last_rl = 0.0
        if (state.auto_roofline_pending_task_id or "").strip():
            return False
        if last_rl <= 0:
            try:
                failure_streak = int(
                    getattr(state, "roofline_failure_streak", 0) or 0
                )
            except (TypeError, ValueError):
                failure_streak = 0
            if failure_streak <= 0:
                return False
            try:
                last_rl = float(state.baseline_tput or 0.0)
            except (TypeError, ValueError):
                last_rl = 0.0
            if last_rl <= 0:
                return False
        cur = self._current_tput_from_validated_gain()
        if cur <= 0:
            return False
        return cur / last_rl >= _resolve_roofline_watermark_ratio()

    async def _maybe_enqueue_watermark_roofline(
        self, *, reason: str,
    ) -> bool:
        """Enqueue a fresh roofline if the watermark crossed; idempotency-keyed via ``reason``, stamps auto_roofline_pending_task_id. Returns True when enqueued."""
        if not self._needs_roofline_for_watermark():
            return False
        try:
            task = await self._enqueue_internal_analysis_task(reason=reason)
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "watermark-roofline (%s): failed to enqueue: %r", reason, exc,
            )
            return False
        self.shared_state.auto_roofline_pending_task_id = task.task_id
        log.info(
            "watermark-roofline (%s): enqueued task=%s "
            "(cur=%.2f, last_roofline=%.2f, ratio>=%.2f)",
            reason, task.task_id,
            self._current_tput_from_validated_gain(),
            float(self.shared_state.last_roofline_tput or 0.0),
            self._ROOFLINE_WATERMARK_RATIO,
        )
        return True

    def _internal_analysis_kind(self) -> str:
        """Pick the kind for the next Coordinator-internal analysis task: roofline (composite) when enable_roofline else profile. Both absent from PHASE_LLM_PROPOSABLE_ACTIONS (PolicyGate R1 denies LLM proposals)."""
        return "roofline" if bool(
            getattr(self.shared_state, "enable_roofline", True),
        ) else "profile"

    def _registry_lanes_ttl(self, kind: str) -> tuple[list[str], int]:
        """Resolve ``(requires_lanes, lease_ttl_sec)`` from the ActionRegistry; lanes filtered to KNOWN_LANES, returns ([], 0) for unknown actions."""
        reg = getattr(self, "action_registry", None)
        if reg is None:
            return [], 0
        meta = reg.get(kind)
        if meta is None:
            return [], 0
        lanes = [
            lane for lane in (getattr(meta, "requires_lanes", ()) or ())
            if lane in KNOWN_LANES
        ]
        return lanes, int(getattr(meta, "lease_ttl_sec", 0) or 0)

    def _warm_recipe_proven_items(self) -> list[dict[str, str]]:
        """Summarise warm-start ``what_worked`` items the scout can skip ({name, source}); fail-soft."""
        state = self.shared_state
        warm = getattr(state, "warm_start_recipe", None) or {}
        if not isinstance(warm, dict) or not warm:
            return []
        recipe = warm.get("recipe") or {}
        recipe_attrs = (recipe.get("attrs") or recipe) if isinstance(recipe, dict) else {}
        what_worked = recipe_attrs.get("what_worked") or []
        if not isinstance(what_worked, list):
            return []
        out: list[dict[str, str]] = []
        for row in what_worked:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            out.append({"name": name, "source": str(row.get("source") or "").strip()})
        return out

    def _inject_warm_recipe_history_into_ledger(self) -> int:
        """GAP 1 — pre-fill ``explore_search.rejected`` with the warm recipe's ``what_failed`` rows (fingerprinted so the dedup gate denies re-tests). Idempotent via warm_history_injected; returns rows added."""
        state = self.shared_state
        if getattr(state, "warm_history_injected", False):
            return 0
        warm = state.warm_start_recipe or {}
        if not isinstance(warm, dict) or not warm:
            state.warm_history_injected = True
            return 0
        recipe = warm.get("recipe") or {}
        # v2 arbor keeps what_failed top-level; v1 nested under attrs. Fall back to the recipe itself.
        recipe_attrs = (recipe.get("attrs") or recipe) if isinstance(recipe, dict) else {}
        what_failed = recipe_attrs.get("what_failed") or []
        if not isinstance(what_failed, list) or not what_failed:
            state.warm_history_injected = True
            return 0

        from .action_executors._canonical_fingerprint import (
            canonical_fingerprint,
        )

        es_raw = getattr(state, "explore_search", None) or {}
        es = dict(es_raw) if isinstance(es_raw, dict) else {}
        rejected = list(es.get("rejected") or [])
        existing_fps = {
            str(r.get("fingerprint") or "")
            for r in rejected
            if isinstance(r, dict)
        }
        existing_fps.discard("")
        added = 0
        tier = str((warm or {}).get("tier") or "")
        for row in what_failed:
            if not isinstance(row, dict):
                continue
            args = str(
                row.get("extra_sglang_args")
                or row.get("args")
                or "",
            ).strip()
            envs = row.get("extra_envs") or row.get("envs") or {}
            if not isinstance(envs, dict):
                envs = {}
            if not args and not envs:
                continue
            fp = canonical_fingerprint(args, envs)
            if fp in existing_fps:
                continue
            existing_fps.add(fp)
            rejected.append({
                "name":              str(row.get("name") or "")[:120],
                "fingerprint":       fp,
                "reason":            "warm_recipe_what_failed",
                "extra_sglang_args": args,
                "extra_envs":        dict(envs),
                "source":            "warm_start_recipe",
                "source_tier":       tier,
                # Recipe-carried fields preserved for forensics; not used by the dedup gate.
                "gain_pct":          row.get("gain_pct"),
                "error_class":       row.get("error_class") or row.get("reason"),
            })
            added += 1

        if added:
            es["rejected"] = rejected
            state.explore_search = es
            log.info(
                "warm-recipe history: injected %d what_failed rows into "
                "explore_search.rejected (tier=%s)",
                added, tier,
            )
        state.warm_history_injected = True
        return added

    async def _maybe_enqueue_warm_replay(
        self, *, baseline_tput: float,
    ) -> "Task | None":
        """GAP 1 — enqueue a one-shot ``replay_warm_recipe`` task for a high-confidence T0 prior.

        Skips on --no-warm-replay/resume/low-confidence/empty best_config; otherwise mints an internal
        task running the baseline workload contract with the KB config applied. Idempotent via warm-replay-prelude.
        """
        state = self.shared_state
        if not getattr(self, "_warm_replay_enabled", True):
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "disabled_by_flag",
            }
            # Flip the one-shot guard even on disabled-skip so a resume without --no-warm-replay can't
            # retroactively trigger a replay against the operator's original intent.
            state.warm_replay_attempted = True
            return None
        if state.warm_replay_attempted:
            # Resume safety: a previous boot already enqueued/ran the replay.
            return None
        warm = state.warm_start_recipe or {}
        if not isinstance(warm, dict) or not warm:
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "no_warm_start_recipe",
            }
            state.warm_replay_attempted = True
            return None
        # tier/conf stamped at T0 by find_recipe_with_fallback.
        tier = str(warm.get("tier") or "").strip()
        try:
            conf = float(warm.get("confidence") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        min_conf = float(getattr(self, "_warm_replay_min_confidence", 0.7) or 0.7)
        if conf < min_conf:
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": f"confidence_below_threshold ({conf:.2f} < {min_conf:.2f})",
                "warm_recipe_tier": tier,
                "warm_recipe_conf": conf,
            }
            state.warm_replay_attempted = True
            return None
        recipe = warm.get("recipe") or {}
        if not isinstance(recipe, dict):
            recipe = {}
        # v2 RecipeKB keeps best_config/sessions top-level; v1 nested under attrs. Fall back to recipe itself.
        recipe_attrs = recipe.get("attrs") or recipe
        best_config = recipe_attrs.get("best_config") or {}
        if not isinstance(best_config, dict):
            best_config = {}
        # Read canonical extra_server_args FIRST, then legacy extra_sglang_args/args.
        bc_args = str(
            best_config.get("extra_server_args")
            or best_config.get("extra_sglang_args")
            or best_config.get("args")
            or ""
        ).strip()
        bc_envs = best_config.get("extra_envs") or best_config.get("envs") or {}
        if not isinstance(bc_envs, dict):
            bc_envs = {}
        # Prefer the WarmStartContext's ready-to-replay champion when T0
        # built one (status=hit). It is the model-facing projection that
        # already normalized args/envs, so it wins over re-deriving from
        # the raw recipe row; the recipe path stays as the fallback for
        # legacy state.json without a context.
        wsc = getattr(state, "warm_start_context", None) or {}
        if isinstance(wsc, dict) and str(wsc.get("status") or "") == "hit":
            replay = wsc.get("recommended_replay") or {}
            if isinstance(replay, dict):
                rep_args = str(replay.get("extra_server_args") or "").strip()
                rep_envs = replay.get("extra_envs") or {}
                if rep_args or (isinstance(rep_envs, dict) and rep_envs):
                    bc_args = rep_args or bc_args
                    if isinstance(rep_envs, dict) and rep_envs:
                        bc_envs = rep_envs
        if not bc_args and not bc_envs:
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "best_config_empty",
                "warm_recipe_tier": tier,
                "warm_recipe_conf": conf,
            }
            state.warm_replay_attempted = True
            return None
        # Historical gain anchor for _promote_warm_replay: MAX gain across attrs.sessions[]; fallback 0.0 accepts any positive measurement.
        expected_gain = 0.0
        sessions_field = recipe_attrs.get("sessions")
        if isinstance(sessions_field, list):
            session_gains: list[float] = []
            for s in sessions_field:
                if not isinstance(s, dict):
                    continue
                try:
                    g = float(s.get("gain_pct") or 0.0)
                except (TypeError, ValueError):
                    continue
                session_gains.append(g)
            if session_gains:
                expected_gain = max(session_gains)
        # Last-chance fallback for offline-ingested seed rows carrying a
        # flat ``gain_pct`` attr.
        if expected_gain <= 0:
            try:
                fallback = float(recipe_attrs.get("gain_pct") or 0.0)
            except (TypeError, ValueError):
                fallback = 0.0
            if fallback > 0:
                expected_gain = fallback
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": "warm_replay_prelude",
            "extra_sglang_args": bc_args,
            "extra_envs": dict(bc_envs),
            # Reuse the baseline's workload contract; else replay renders from YAML smoke defaults.
            "config_path": str(state.baseline_config_path or ""),
            # Carry the historical-gain anchor forward for the promote path's reproduce ratio.
            "warm_expected_gain_pct": expected_gain,
            "warm_recipe_tier": tier,
            "warm_recipe_conf": conf,
            "baseline_tput_anchor": float(baseline_tput),
        }
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="replay_warm_recipe",
            params=params,
            idempotency_key="warm-replay-prelude",
        )
        if not was_existing:
            log.info(
                "PRELUDE: warm-replay enqueued task=%s (tier=%s conf=%.2f "
                "expected_gain=%.2f baseline_tput=%.2f)",
                task.task_id, tier, conf, expected_gain, baseline_tput,
            )
        state.warm_replay_attempted = True
        state.warm_replay_outcome = {
            "status": "in_flight",
            "warm_recipe_tier": tier,
            "warm_recipe_conf": conf,
            "expected_gain_pct": expected_gain,
            "replay_task_id": task.task_id,
        }
        return task

    def _promote_warm_replay(
        self, result: dict, *, task: "Task | None" = None,
    ) -> None:
        """GAP 1 — interpret a ``replay_warm_recipe`` result: any measured uplift pushes warm config onto optimization_stack + current_best; failures set status and never propagate."""
        state = self.shared_state
        outcome = dict(state.warm_replay_outcome or {})
        expected_gain = float(outcome.get("expected_gain_pct") or 0.0)
        if not isinstance(result, dict):
            outcome["status"] = "failed"
            outcome["reason"] = "non_dict_result"
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            return
        status = str(result.get("status") or "")
        if status != "succeeded":
            outcome["status"] = "failed"
            outcome["error_class"] = str(result.get("error_class") or "")
            outcome["reason"] = str(result.get("error") or result.get("reason") or "")[:240]
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            log.info(
                "warm-replay failed (status=%s, error_class=%s)",
                status, outcome.get("error_class"),
            )
            return
        tput_raw = result.get("output_throughput")
        try:
            tput = float(tput_raw) if tput_raw is not None else 0.0
        except (TypeError, ValueError):
            tput = 0.0
        # Use the baseline_tput captured at enqueue time so a mid-replay baseline rerun can't shift the anchor; fall back to live state.baseline_tput.
        anchor_raw = None
        if task is not None and isinstance(getattr(task, "params", None), dict):
            anchor_raw = task.params.get("baseline_tput_anchor")
        try:
            baseline_tput = float(anchor_raw) if anchor_raw is not None else 0.0
        except (TypeError, ValueError):
            baseline_tput = 0.0
        if baseline_tput <= 0:
            baseline_tput = float(state.baseline_tput or 0.0)
        if tput <= 0 or baseline_tput <= 0:
            outcome["status"] = "failed"
            outcome["reason"] = f"invalid_tput tput={tput} baseline={baseline_tput}"
            state.warm_replay_outcome = outcome
            state.save(self.session_dir)
            return
        measured_gain = (tput / baseline_tput - 1.0) * 100.0
        min_reproduce = float(
            getattr(self, "_warm_replay_min_reproduce_pct", 0.8) or 0.8,
        )
        # Adopt KB best_config whenever replay beats baseline; expected_gain/min_reproduce kept for audit only, not gating.
        reproduced = measured_gain > 0
        outcome["actual_gain_pct"] = round(measured_gain, 3)
        outcome["throughput_after"] = tput
        if expected_gain > 0:
            historical_bar = expected_gain * min_reproduce
            if measured_gain > 0 and measured_gain < historical_bar:
                outcome["below_historical_reproduce_pct"] = True
                outcome["historical_reproduce_bar_pct"] = round(
                    historical_bar, 3,
                )
        if reproduced:
            # R4-4 defense: an empty stack entry corrupts session_breakdown attribution; degrade gracefully when task=None.
            params = (task.params if task is not None else {}) or {}
            warm_args = str(params.get("extra_sglang_args") or "").strip()
            warm_envs = dict(params.get("extra_envs") or {})
            if not warm_args and not warm_envs:
                outcome["status"] = "reproduced_but_no_params"
                outcome["reason"] = "task.params missing extra_sglang_args/extra_envs"
                log.warning(
                    "warm-replay measured +%.2f%% but cannot push stack "
                    "(task=%r has no warm args/envs)",
                    measured_gain, task,
                )
                state.warm_replay_outcome = outcome
                state.save(self.session_dir)
                return
            outcome["status"] = "reproduced"
            # Push warm best_config onto the stack (schema mirrors explore-KEEP); ts/workspace/gain_pct feed session_breakdown attribution.
            stack_entry = {
                "action":            "replay_warm_recipe",
                "name":              "warm_replay",
                "variant_name":      "warm_replay",
                # Canonical key matching EXPLORE-KEEP stack entries so downstream readers key on the same name.
                "extra_server_args": warm_args,
                "extra_envs":        warm_envs,
                "tput":              float(tput),
                "gain_pct":          round(measured_gain, 3),
                "workspace":         str(result.get("workspace") or ""),
                "ts":                datetime.now(timezone.utc).isoformat(),
                # source_tier records the warm-recipe tier (exact/relative) for breakdown attribution.
                "source_tier":       outcome.get("warm_recipe_tier", ""),
                "source_confidence": outcome.get("warm_recipe_conf", 0.0),
            }
            # Resume safety: DO NOT clobber existing stack entries; recompute cumulative gain from baseline → current tput.
            state.optimization_stack = list(state.optimization_stack or [])
            # Idempotency guard: skip push if a prior promote run already pushed the warm_replay entry.
            already_pushed = any(
                isinstance(e, dict) and e.get("action") == "replay_warm_recipe"
                for e in state.optimization_stack
            )
            if already_pushed:
                log.info(
                    "warm-replay promote: stack already carries the entry; "
                    "skipping duplicate push (likely resume mid-promote)",
                )
                state.warm_replay_outcome = outcome
                state.save(self.session_dir)
                return
            state.optimization_stack.append(stack_entry)
            # gain_per_stack_entry runs in lock-step with optimization_stack.
            gp = list(getattr(state, "gain_per_stack_entry", []) or [])
            gp.append(round(measured_gain, 3))
            state.gain_per_stack_entry = gp
            # Cumulative gain from absolute tput/baseline (stack is superposition, not additive deltas).
            total_gain = (tput / baseline_tput - 1.0) * 100.0
            state.cumulative_gain = round(total_gain, 3)
            state.cumulative_gain_validated = round(total_gain, 3)
            state.cumulative_gain_validated_stack_len = len(
                state.optimization_stack
            )
            state.current_best = {
                "action": "warm_replay",
                "name": "warm_replay",
                "tput": tput,
                # Canonical key — matches the current_best shape _lift_to_current_best writes for KEEPs.
                "extra_server_args": warm_args,
                "extra_envs": warm_envs,
            }
            log.info(
                "warm-replay REPRODUCED: measured=+%.2f%% (expected=+%.2f%%, "
                "min_required=+%.2f%%); pushed warm_replay onto stack",
                measured_gain,
                expected_gain,
                expected_gain * min_reproduce if expected_gain > 0 else 0.0,
            )
            # Journal warm-replay as a synthetic KEEP; no KB lesson (verification, not a new fact).
            try:
                journal = self._ensure_journal()
                from .optimization_journal import KIND_OTHER, OUTCOME_KEEP
                journal.append_entry(JournalEntry(
                    phase=str(getattr(state, "phase", "PRELUDE")).upper() or "PRELUDE",
                    iter=int(state.tick or 0),
                    kind=KIND_OTHER,
                    change=f"warm_replay({outcome.get('warm_recipe_tier', '?')}): {warm_args}",
                    outcome=OUTCOME_KEEP,
                    gain_pct=round(measured_gain, 3),
                    throughput_after=tput,
                    task_id=str(task.task_id if task is not None else ""),
                    tick=int(state.tick or 0),
                ))
            except Exception:  # noqa: BLE001 — defensive
                log.exception("warm-replay journal append failed")
        else:
            outcome["status"] = "drift"
            outcome["reason"] = (
                f"measured +{measured_gain:.2f}% below {min_reproduce * 100:.0f}%"
                f" of expected +{expected_gain:.2f}%"
            )
            log.info(
                "warm-replay DRIFT: measured=+%.2f%% < expected=+%.2f%% × %.0f%%",
                measured_gain, expected_gain, min_reproduce * 100,
            )
        state.warm_replay_outcome = outcome
        state.save(self.session_dir)

    async def _maybe_enqueue_prelude_initial_analysis_after_baseline(
        self,
        *,
        baseline_tput: float | None = None,
    ) -> None:
        """Enqueue the PRELUDE-bootstrap roofline/profile task after baseline; skipped while warm-replay is in_flight (GPU/port contention)."""
        state = self.shared_state
        if _phase_state.warm_replay_in_flight(state):
            log.info(
                "PRELUDE: deferring initial %s until warm-replay completes",
                self._internal_analysis_kind(),
            )
            return
        if baseline_tput is None:
            try:
                baseline_tput = float(state.baseline_tput or 0.0)
            except (TypeError, ValueError):
                baseline_tput = 0.0
        if not isinstance(baseline_tput, (int, float)) or baseline_tput <= 0:
            return
        if (state.auto_roofline_pending_task_id or "").strip():
            return
        try:
            rl_task = await self._enqueue_internal_analysis_task(
                reason="prelude_initial",
            )
            state.auto_roofline_pending_task_id = rl_task.task_id
            log.info(
                "PRELUDE: baseline landed (tput=%.2f); auto-enqueued "
                "initial %s task=%s",
                float(baseline_tput), rl_task.kind, rl_task.task_id,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "PRELUDE: failed to enqueue initial analysis task "
                "after baseline: %r", exc,
            )

    async def _enqueue_internal_analysis_task(self, *, reason: str) -> Task:
        """Build + enqueue a Coordinator-internal analysis task (roofline or profile). Kind-agnostic idempotency key internal-analysis-<reason>; omits baseline_config_path so ProfileExecutor enables torch_profiler."""
        state = self.shared_state
        kind = self._internal_analysis_kind()
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
        }
        if reason != "prelude_initial":
            cb = state.current_best or {}
            if isinstance(cb, dict):
                cb_args = str(cb.get("extra_server_args") or "")
                if cb_args:
                    params["base_extra_args"] = cb_args
        else:
            # PRELUDE roofline profiles the baseline arm: inject baseline's own
            # server args (from its materialized yaml), never current_best's,
            # so a later warm-replay can't swap in compile/fp8 flags that
            # destabilize profiling and skew the baseline ceiling.
            try:
                from .roofline_ceiling import read_baseline_server_args
                bl_args = read_baseline_server_args(state).strip()
            except Exception:  # noqa: BLE001 — best-effort; empty falls through
                bl_args = ""
            if bl_args:
                params["base_extra_args"] = bl_args
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        lanes, ttl = self._registry_lanes_ttl(kind)
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=kind,
            params=params,
            idempotency_key=f"internal-analysis-{reason}",
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if was_existing:
            log.info(
                "internal-analysis task already exists (idempotent: "
                "kind=%s task_id=%s, state=%s)",
                kind, task.task_id, task.state,
            )
        return task

    def _record_phase_entry_evidence(self, **kvs: Any) -> None:
        """Merge ``kvs`` into the latest phase_history row's evidence dict (Gap-04; no-op when empty)."""
        history = self.shared_state.phase_history or []
        if not history:
            return
        row = history[-1]
        if not isinstance(row, dict):
            return
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
            row["evidence"] = evidence
        for k, v in kvs.items():
            evidence[k] = v
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "phase entry evidence: SharedState.save failed for kvs=%r",
                kvs,
            )

    # SWEEP phase auto-dispatch
    async def _drain_pending_keep_integrates(self) -> None:
        """Bug #7: drain pending KEEP integrates inherited from KERNEL so sweep measures full current_best. Cap 10; failures → rejected_kernel_ids."""
        from .kernel_request_handlers import integrate_handler
        state = self.shared_state
        drained = 0
        max_drain = 10
        while drained < max_drain:
            kid = state.next_pending_keep_kernel_id()
            if not kid:
                break
            log.info(
                "SWEEP entry: draining pending KEEP integrate for "
                "kernel_id=%s (drained %d so far)", kid, drained,
            )
            try:
                base = float(
                    (state.current_best or {}).get("tput")
                    or state.baseline_tput or 0.0
                )
                result = await integrate_handler(
                    {"kernel_id": kid, "base_tput": base},
                    session_dir=self.session_dir,
                )
                if isinstance(result, dict) and result.get("status") != "skipped":
                    state.record_kernel_integrate_result(result)
                    if str(result.get("decision") or "").upper() == "KEEP":
                        await self._record_integrate_keep(result)
                state.save(self.session_dir)
            except Exception as exc:  # noqa: BLE001 — never block SWEEP entry
                log.exception(
                    "SWEEP entry: integrate(%s) raised %r; marking "
                    "rejected to prevent drain loop deadlock", kid, exc,
                )
                if state.rejected_kernel_ids is None:
                    state.rejected_kernel_ids = []
                if kid not in state.rejected_kernel_ids:
                    state.rejected_kernel_ids.append(kid)
                state.save(self.session_dir)
            drained += 1
        if drained >= max_drain:
            log.warning(
                "SWEEP entry: drain cap (%d) reached; remaining pending "
                "KEEPs will be visible in summary.by_kernel as KEEP_PENDING",
                max_drain,
            )

    def _positive_needs_review_integrates(self) -> list[dict[str, Any]]:
        """Return positive NEEDS_REVIEW integrate entries eligible for stack validation."""
        out: list[dict[str, Any]] = []
        stack_resolved_ids = self._stack_resolved_kernel_ids()
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            kernel_id = str(entry.get("kernel_id") or "").strip()
            if (
                bool(entry.get("stack_resolved"))
                or bool(entry.get("stack_validation_in_progress"))
                or kernel_id in stack_resolved_ids
            ):
                continue
            if str(entry.get("last_decision") or "").upper() != "NEEDS_REVIEW":
                continue
            try:
                best_gain = float(entry.get("best_gain_pct") or 0.0)
            except (TypeError, ValueError):
                best_gain = 0.0
            if best_gain <= 0:
                continue
            patch_path = str(entry.get("patch_path") or "").strip()
            target_file = str(entry.get("target_file") or "").strip()
            if patch_path and target_file and kernel_id:
                out.append(entry)
        out.sort(key=lambda e: float(e.get("best_gain_pct") or 0.0), reverse=True)
        return out

    def _stack_resolved_kernel_ids(self) -> set[str]:
        """Kernel ids already covered by a kept stack validation."""
        resolved: set[str] = set()
        for item in self.shared_state.optimization_stack or []:
            if not isinstance(item, dict):
                continue
            if item.get("action") != "integrate":
                continue
            stack_ids = item.get("stack_kernel_ids")
            if isinstance(stack_ids, list):
                resolved.update(str(kid) for kid in stack_ids if str(kid))
                continue
            kernel_id = str(item.get("kernel_id") or "")
            if "+" in kernel_id:
                resolved.update(kid for kid in kernel_id.split("+") if kid)
        return resolved

    def _mark_stack_validation_entries_resolved(
        self,
        entries: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        """Mark component NEEDS_REVIEW entries as handled by a kept stack."""
        stack_id = str(result.get("kernel_id") or "")
        decision = str(result.get("decision") or "").upper()
        if decision != "KEEP" or not stack_id:
            return
        now = datetime.now(timezone.utc).isoformat()
        wanted = {
            (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            for entry in entries
            if isinstance(entry, dict)
        }
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            identity = (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            if identity not in wanted:
                continue
            entry["stack_resolved"] = True
            entry["stack_validation_kernel_id"] = stack_id
            entry["stack_decision"] = decision
            entry["stack_resolved_at"] = now
            entry.pop("stack_validation_in_progress", None)

    def _stack_component_identities(
        self, entries: list[dict[str, Any]],
    ) -> set[tuple[str, str, str]]:
        """Return (kernel_id, patch_path, target_file) tuples for stack members."""
        return {
            (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            for entry in entries
            if isinstance(entry, dict)
        }

    def _mark_stack_validation_in_progress(
        self,
        entries: list[dict[str, Any]],
        stack_id: str,
    ) -> None:
        """Persist an in-flight stack guard before applying patches."""
        now = datetime.now(timezone.utc).isoformat()
        wanted = self._stack_component_identities(entries)
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            identity = (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            if identity not in wanted:
                continue
            entry["stack_validation_in_progress"] = True
            entry["stack_validation_kernel_id"] = stack_id
            entry["stack_validation_started_at"] = now

    def _clear_stack_validation_in_progress(
        self, entries: list[dict[str, Any]],
    ) -> None:
        """Clear the in-flight stack guard for component integrate entries."""
        wanted = self._stack_component_identities(entries)
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            identity = (
                str(entry.get("kernel_id") or ""),
                str(entry.get("patch_path") or ""),
                str(entry.get("target_file") or ""),
            )
            if identity not in wanted:
                continue
            entry.pop("stack_validation_in_progress", None)

    def _clear_pending_stack_validation_checkpoints(self) -> None:
        """Drop crash-recovery checkpoints once a stack attempt is finished."""
        self.shared_state.pending_stack_validation_result = {}
        self.shared_state.pending_stack_validation_apply_results = []

    async def _recover_interrupted_stack_validation(self) -> bool:
        """Resume or abort a stack validation interrupted by crash."""
        from .kernel_request_handlers import _maybe_revert_kernel_patch

        pending = self.shared_state.pending_stack_validation_result
        if isinstance(pending, dict) and pending:
            stack = self._stack_entries_for_validation(
                pending.get("stack_kernel_ids") or [],
                stack_id=str(pending.get("kernel_id") or ""),
            )
            if len(stack) >= 2:
                await self._finalize_stack_validation_outcome(stack, pending)
                return True

        partial_applies = list(
            self.shared_state.pending_stack_validation_apply_results or [],
        )
        in_progress = [
            entry for entry in (self.shared_state.kernel_integrate_attempts or {}).values()
            if isinstance(entry, dict) and entry.get("stack_validation_in_progress")
        ]
        if not partial_applies and not in_progress:
            return False

        if partial_applies:
            for applied in reversed(partial_applies):
                _maybe_revert_kernel_patch(applied)
        if in_progress:
            self._clear_stack_validation_in_progress(in_progress)
        self._clear_pending_stack_validation_checkpoints()
        self.shared_state.save(self.session_dir)
        log.warning(
            "Recovered interrupted stack validation: reverted partial applies "
            "and cleared in-progress guards",
        )
        return True

    def _stack_entries_for_validation(
        self,
        kernel_ids: list[Any],
        *,
        stack_id: str = "",
    ) -> list[dict[str, Any]]:
        """Rebuild component integrate ledger rows for a stack id."""
        wanted_ids = {str(kid) for kid in kernel_ids if str(kid)}
        if not wanted_ids and stack_id:
            wanted_ids = {kid for kid in stack_id.split("+") if kid}
        out: list[dict[str, Any]] = []
        for entry in (self.shared_state.kernel_integrate_attempts or {}).values():
            if not isinstance(entry, dict):
                continue
            kid = str(entry.get("kernel_id") or "")
            if kid in wanted_ids:
                out.append(entry)
        out.sort(key=lambda e: str(e.get("kernel_id") or ""))
        return out

    async def _finalize_stack_validation_outcome(
        self,
        stack: list[dict[str, Any]],
        result: dict[str, Any],
    ) -> None:
        """Record stack validation, promote KEEP, and clear recovery checkpoints."""
        self.shared_state.record_kernel_integrate_result(result)
        decision = str(result.get("decision") or "").upper()
        if decision == "KEEP":
            self._mark_stack_validation_entries_resolved(stack, result)
            self.shared_state.save(self.session_dir)
            await self._record_integrate_keep(result)
        else:
            self._clear_stack_validation_in_progress(stack)
        self._clear_pending_stack_validation_checkpoints()
        self.shared_state.save(self.session_dir)

    async def _maybe_validate_positive_needs_review_stack(self) -> None:
        """Run one E2E stack validation for multiple small positive kernel patches.

        Single-patch ``NEEDS_REVIEW`` is not retried automatically. When two or
        more pending kernel patches individually show positive but sub-threshold
        E2E gain, validate their combined effect once before moving to SWEEP.
        """
        if await self._recover_interrupted_stack_validation():
            return
        entries = self._positive_needs_review_integrates()
        if len(entries) < 2:
            return
        # Avoid applying two whole-file patches to the same target file.
        seen_targets: set[str] = set()
        stack: list[dict[str, Any]] = []
        for entry in entries:
            target = str(entry.get("target_file") or "")
            if target in seen_targets:
                continue
            seen_targets.add(target)
            stack.append(entry)
        if len(stack) < 2:
            return
        stack_id = "+".join(str(e.get("kernel_id") or "") for e in stack)
        self._mark_stack_validation_in_progress(stack, stack_id)
        self._clear_pending_stack_validation_checkpoints()
        self.shared_state.save(self.session_dir)
        result = await self._run_kernel_stack_validation_e2e(stack)
        if not isinstance(result, dict):
            self._clear_stack_validation_in_progress(stack)
            self._clear_pending_stack_validation_checkpoints()
            self.shared_state.save(self.session_dir)
            return
        self.shared_state.pending_stack_validation_result = result
        self.shared_state.save(self.session_dir)
        await self._finalize_stack_validation_outcome(stack, result)

    async def _run_kernel_stack_validation_e2e(
        self, entries: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply multiple kernel patches, run one E2E benchmark, then keep or revert the stack."""
        from .action_executors.baseline import BaselineExecutor
        from .action_executors.benchmark_result import is_valid_measurement
        from .kernel_request_handlers import (
            KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT,
            _maybe_apply_kernel_patch,
            _maybe_revert_kernel_patch,
        )
        from .sub_agent_runner import RunnerContext
        from .task_registry import Task
        from ..session_paths import runs_dir

        kernel_ids = [str(e.get("kernel_id") or "") for e in entries]
        stack_id = "+".join(kernel_ids)
        apply_results: list[dict[str, Any]] = []
        try:
            for entry in entries:
                payload = {
                    "kernel_id": entry.get("kernel_id"),
                    "patch_path": entry.get("patch_path"),
                    "target_file": entry.get("target_file"),
                    "allow_unknown_target": True,
                }
                applied = _maybe_apply_kernel_patch(
                    payload,
                    session_dir=self.session_dir,
                    kernel_id=str(entry.get("kernel_id") or ""),
                )
                apply_results.append(applied)
                self.shared_state.pending_stack_validation_apply_results = list(
                    apply_results,
                )
                self.shared_state.save(self.session_dir)
                if applied.get("status") != "ok":
                    raise RuntimeError(
                        f"stack patch apply failed for {entry.get('kernel_id')}: {applied}"
                    )

            workspace = runs_dir(self.session_dir, "integrate", f"integrate-stack-{stack_id}")
            workspace.mkdir(parents=True, exist_ok=True)
            fake_task = Task(
                task_id=f"integrate-stack-{stack_id}",
                kind="baseline",
                state="running",
                params={
                    "config_path": self.shared_state.baseline_config_path,
                    "output_dir": str(workspace),
                    "timeout_sec": 20 * 60,
                    "extra_server_args": (
                        (self.shared_state.current_best or {}).get("extra_server_args")
                        or ""
                    ),
                },
                idempotency_key=f"integrate-stack-{stack_id}-rebaseline",
            )
            bench_result = await BaselineExecutor(session_dir=self.session_dir)(
                RunnerContext(task=fake_task, lease=None)
            )
            if not is_valid_measurement(bench_result):
                decision = "REVERT"
                new_tput = 0.0
                gain_pct = -100.0
            else:
                base_tput = float(self.shared_state.baseline_tput or 0.0)
                new_tput = float(bench_result.get("output_throughput") or 0.0)
                gain_pct = (
                    (new_tput - base_tput) / base_tput * 100.0
                    if base_tput > 0 else 0.0
                )
                decision = (
                    "KEEP"
                    if gain_pct > KERNEL_STACK_VALIDATION_KEEP_THRESHOLD_PCT
                    else "REVERT"
                )

            result = {
                "status": "ok",
                "decision": decision,
                "kernel_id": stack_id,
                "patch_path": "+".join(str(e.get("patch_path") or "") for e in entries),
                "target_file": "+".join(str(e.get("target_file") or "") for e in entries),
                "base_tput": float(self.shared_state.baseline_tput or 0.0),
                "new_tput": new_tput,
                "gain_pct": gain_pct,
                "report_path": bench_result.get("report_path") if isinstance(bench_result, dict) else None,
                "workspace": bench_result.get("workspace") if isinstance(bench_result, dict) else str(workspace),
                "apply_result": {"status": "ok", "stack_apply_results": apply_results},
                "stack_kernel_ids": kernel_ids,
                "stack_validation": True,
            }
            for metric in ("ttft_mean_ms", "e2el_mean_ms", "tpot_mean_ms"):
                if isinstance(bench_result, dict) and metric in bench_result:
                    result[metric] = bench_result.get(metric)
            if decision != "KEEP":
                result["revert_result"] = {
                    "status": "ok",
                    "stack_reverts": [
                        _maybe_revert_kernel_patch(applied)
                        for applied in reversed(apply_results)
                    ],
                }
            else:
                result["revert_result"] = {"status": "skipped", "reason": "KEEP decision"}
            return result
        except Exception as exc:  # noqa: BLE001
            reverts = [
                _maybe_revert_kernel_patch(applied)
                for applied in reversed(apply_results)
            ]
            return {
                "status": "failed",
                "decision": "REVERT",
                "kernel_id": stack_id,
                "error": repr(exc),
                "apply_result": {"status": "failed", "stack_apply_results": apply_results},
                "revert_result": {"status": "ok", "stack_reverts": reverts},
                "stack_kernel_ids": kernel_ids,
                "stack_validation": True,
            }

    async def _on_enter_sweep(self, *, from_phase: str) -> None:
        """Auto-enqueue a ``sweep`` task on SWEEP entry (§3.2 §5.4). Idempotent via internal-sweep-phase_entry (Inv-2.1); PolicyGate's sweep_phase_singleton then denies LLM-emitted sweep (OOM race)."""
        state = self.shared_state
        # Bug #7 fix: drain pending KEEP integrates from prior KERNEL so sweep measures full current_best.
        if getattr(state, "has_keep_pending_integrate", False):
            await self._drain_pending_keep_integrates()
        # Always attempt stack validation for positive NEEDS_REVIEW kernels,
        # regardless of whether there were pending KEEPs to drain.
        await self._maybe_validate_positive_needs_review_stack()
        try:
            task = await self._enqueue_internal_sweep_task(
                reason="phase_entry",
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "SWEEP entry hook: failed to enqueue auto-sweep: %r", exc,
            )
            self._record_phase_entry_evidence(auto_sweep_error=repr(exc)[:240])
            return
        # Mirror the chosen grid + source onto evidence without re-running lookup.
        grid_source = str(task.params.get("source") or "")
        isl_osl = task.params.get("isl_osl_configs") or []
        conc_values = task.params.get("conc_values") or []
        # Combos = |conc_values| × |isl_osl_configs| (sweep fans out CONC × (ISL,OSL)).
        combos = int(len(conc_values)) * int(len(isl_osl)) if (
            conc_values and isl_osl
        ) else 0
        log.info(
            "SWEEP entry (from=%s): auto-enqueued sweep task=%s "
            "(grid_source=%s, combos=%d)",
            from_phase or "<unknown>", task.task_id, grid_source, combos,
        )
        self._record_phase_entry_evidence(
            auto_sweep_enqueued=True,
            auto_sweep_task_id=task.task_id,
            auto_sweep_grid_source=grid_source,
            auto_sweep_combos=combos,
        )

    async def _enqueue_internal_conc_sweep_task(
        self, *, reason: str,
    ) -> Task | None:
        """Build + enqueue a Coordinator-internal ``conc_sweep`` task (caller checks conc_sweep_enabled). Idempotency key + PolicyGate singleton ensure ≤1 per SWEEP; returns None on error."""
        state = self.shared_state
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
            "concs": list(state.conc_sweep_concs or []),
            "variant_timeout_sec": int(state.conc_sweep_variant_timeout_sec or 0),
            "total_budget_sec":    int(state.conc_sweep_total_budget_sec or 0),
        }
        try:
            task, was_existing = await self.tasks.create_or_return_existing(
                kind="conc_sweep",
                params=params,
                idempotency_key=f"internal-conc_sweep-{reason}",
                # lease_ttl matches total_budget_sec so a multi-hour conc_sweep doesn't expire mid-flight.
                lease_ttl_sec=int(state.conc_sweep_total_budget_sec or 9000),
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "conc_sweep: failed to enqueue internal task: %r", exc,
            )
            return None
        if was_existing:
            log.info(
                "internal-conc_sweep task already exists (idempotent: "
                "task_id=%s, state=%s)", task.task_id, task.state,
            )
        else:
            log.info(
                "internal-conc_sweep task enqueued (task_id=%s reason=%s "
                "concs=%s total_budget_sec=%s)",
                task.task_id, reason,
                params["concs"], params["total_budget_sec"],
            )
        # Bug #11 fix: stamp evidence so PolicyGate's conc_sweep_phase_singleton denies later LLM conc_sweep.
        self._record_phase_entry_evidence(auto_conc_sweep_task_id=task.task_id)
        return task

    async def _enqueue_internal_sweep_task(
        self, *, reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``sweep`` task. Grid priority: warm_start_recipe.sweep_grid then SKILL.md defaults. Idempotency key internal-sweep-<reason>."""
        state = self.shared_state
        grid_params = self._build_sweep_params_from_recipe(state)
        params: dict[str, Any] = {
            "source": grid_params["source"],
            "reason": str(reason),
            "conc_values":      list(grid_params["conc_values"]),
            "isl_osl_configs":  list(grid_params["isl_osl_configs"]),
            "num_prompts_factor": int(grid_params["num_prompts_factor"]),
        }
        if state.baseline_config_path:
            params["config_path"] = state.baseline_config_path
        cb = state.current_best or {}
        if isinstance(cb, dict):
            cb_args = str(cb.get("extra_server_args") or "")
            if cb_args:
                params["base_extra_args"] = cb_args
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            # Mirror baseline's benchmark_script so re-launch uses the same wrapper (Gap-04).
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="sweep",
            params=params,
            idempotency_key=f"internal-sweep-{reason}",
        )
        if was_existing:
            log.info(
                "internal-sweep task already exists (idempotent: "
                "task_id=%s, state=%s)", task.task_id, task.state,
            )
        return task

    @staticmethod
    def _build_sweep_params_from_recipe(state: SharedState) -> dict[str, Any]:
        """Pick a sweep grid (§3.14 R-13): warm_start_recipe.sweep_grid takes precedence over SKILL.md defaults; per-field fallback. Returns source/conc_values/isl_osl_configs/num_prompts_factor."""
        from .action_executors.sweep import (
            DEFAULT_CONC_VALUES,
            DEFAULT_ISL_OSL,
            DEFAULT_NUM_PROMPTS_FACTOR,
        )

        recipe = getattr(state, "warm_start_recipe", None)
        sweep_grid = None
        if isinstance(recipe, dict):
            sg = recipe.get("sweep_grid")
            if isinstance(sg, dict):
                sweep_grid = sg

        def _coerce_int_list(value: Any) -> list[int] | None:
            """Coerce a recipe value into a non-empty list of ints.

            Args:
                value (Any): The raw recipe field (expected: list of ints).

            Returns:
                list[int] | None: The coerced ints, or ``None`` if ``value`` is
                    not a non-empty all-int list.
            """
            if not isinstance(value, list) or not value:
                return None
            out: list[int] = []
            for v in value:
                try:
                    out.append(int(v))
                except (TypeError, ValueError):
                    return None
            return out if out else None

        def _coerce_isl_osl_list(value: Any) -> list[str] | None:
            """Coerce a recipe value into a list of ``"<ISL>:<OSL>"`` strings.

            Accepts either ``"<ISL>:<OSL>"`` strings or ``[isl, osl]`` pairs.

            Args:
                value (Any): The raw recipe field.

            Returns:
                list[str] | None: Normalized ISL:OSL strings, or ``None`` if the
                    value is not a recognisable non-empty list.
            """
            if not isinstance(value, list) or not value:
                return None
            out: list[str] = []
            for v in value:
                # Accept either "<ISL>:<OSL>" strings or [isl, osl] pairs.
                if isinstance(v, str) and ":" in v:
                    out.append(v)
                    continue
                if (
                    isinstance(v, (list, tuple)) and len(v) == 2
                    and all(isinstance(x, (int, str)) for x in v)
                ):
                    out.append(f"{int(v[0])}:{int(v[1])}")
                    continue
                return None
            return out if out else None

        conc_values: list[int] = list(DEFAULT_CONC_VALUES)
        isl_osl_configs: list[str] = list(DEFAULT_ISL_OSL)
        num_prompts_factor: int = int(DEFAULT_NUM_PROMPTS_FACTOR)
        used_recipe = False

        if sweep_grid is not None:
            cv = _coerce_int_list(sweep_grid.get("conc_values"))
            if cv is not None:
                conc_values = cv
                used_recipe = True
            io = _coerce_isl_osl_list(sweep_grid.get("isl_osl_configs"))
            if io is not None:
                isl_osl_configs = io
                used_recipe = True
            npf_raw = sweep_grid.get("num_prompts_factor")
            try:
                npf = int(npf_raw) if npf_raw is not None else None
            except (TypeError, ValueError):
                npf = None
            if npf is not None and npf > 0:
                num_prompts_factor = npf
                used_recipe = True
            if not used_recipe:
                log.warning(
                    "sweep recipe present but unusable (no recognisable "
                    "fields); falling back to SKILL.md defaults"
                )

        return {
            "source": "cortex_recipe" if used_recipe else "skill_md_default",
            "conc_values":        conc_values,
            "isl_osl_configs":    isl_osl_configs,
            "num_prompts_factor": num_prompts_factor,
        }

    # CLOSE phase sequencer; class-level wait-for-task timeouts (overridable per-instance in tests).
    CLOSE_REPORT_TIMEOUT_SEC: float = 600.0
    CLOSE_SESSION_BREAKDOWN_TIMEOUT_SEC: float = 300.0
    CLOSE_NDJSON_DRAIN_TIMEOUT_SEC: float = 60.0

    def _derive_close_stop_reason(self) -> str:
        """Best-effort ``stop_reason`` for a CLOSE reached blank: recover from the newest CLOSE-bound phase_history row, else time_exhausted."""
        history = self.shared_state.phase_history or []
        for row in reversed(history):
            if not isinstance(row, dict):
                continue
            if (row.get("to_phase") or "").strip().upper() != \
                    _phase_state.PHASE_CLOSE:
                continue
            reason = (row.get("reason") or "").strip()
            if reason and _phase_state.is_valid_stop_reason(reason):
                return reason
            # Newest CLOSE-bound row had no usable reason — stop rather than pick up a stale older transition.
            break
        return "time_exhausted"

    async def _on_enter_close(self, *, from_phase: str) -> None:
        """CLOSE 5-step sequencer (fixed order): report → session_breakdown → fact_finalize → ndjson_drain (no-op) → mark close_sequence_done + stop_reason. Best-effort steps; final done step always runs."""
        log.info("CLOSE entered (from=%s); starting 5-step close sequence",
                 from_phase or "<unknown>")
        await self._record_close_step("sequencer_started", status="running")

        # stop_reason MUST persist BEFORE step 2's breakdown (collector derives it from state.json); fill only when blank, derive rather than hard-code time_exhausted.
        if not self.shared_state.stop_reason:
            derived = self._derive_close_stop_reason()
            self.shared_state.set_stop_reason(derived)
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "CLOSE: early stop_reason persist failed; step 5 will retry"
                )

        # CLOSE-entry auto-roofline (former N31) deleted in favour of EXPLORE/KERNEL-entry hooks.

        # Step 1: report
        try:
            self._emit_lifecycle(
                step="report",
                status="START",
                detail="close_phase_entry",
            )
            report_task = await self._enqueue_internal_report_task(
                reason="close_phase_entry",
            )
            report_result = await self.sub.run_task(report_task)
            terminal_state = report_result.state
            if terminal_state in {"succeeded", None}:
                await self._record_close_step(
                    "report", status="done",
                    task_id=report_task.task_id,
                )
                # #266: surface the final report location in the lifecycle
                # log. report_executor writes final.{json,md} under
                # reports_dir(session_dir); advertise whichever exist.
                from ..session_paths import reports_dir as _reports_dir
                _rd = _reports_dir(self.session_dir)
                _artifacts = {
                    "json_path": str(_rd / "final.json")
                    if (_rd / "final.json").exists() else "",
                    "md_path": str(_rd / "final.md")
                    if (_rd / "final.md").exists() else "",
                }
                self._emit_lifecycle(
                    step="report",
                    status="END",
                    artifacts=_artifacts,
                    detail="close_phase_entry",
                )
            else:
                detail = f"task_state={terminal_state!r}"
                self._emit_lifecycle(
                    step="report",
                    status="ERROR",
                    detail=detail,
                )
                await self._record_close_step(
                    "report", status="failed",
                    task_id=report_task.task_id,
                    detail=detail,
                )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception("CLOSE step 1 (report) failed")
            self._emit_lifecycle(
                step="report",
                status="ERROR",
                detail=repr(exc)[:240],
            )
            await self._record_close_step(
                "report", status="failed", detail=repr(exc)[:240],
            )

        # Step 2: session_breakdown
        try:
            bd_task = await self._enqueue_internal_session_breakdown_task(
                reason="close_phase_entry",
            )
            bd_result = await self.sub.run_task(bd_task)
            terminal_state = bd_result.state
            if terminal_state in {"succeeded", None}:
                await self._record_close_step(
                    "session_breakdown", status="done",
                    task_id=bd_task.task_id,
                )
            else:
                await self._record_close_step(
                    "session_breakdown", status="failed",
                    task_id=bd_task.task_id,
                    detail=f"task_state={terminal_state!r}",
                )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception("CLOSE step 2 (session_breakdown) failed")
            await self._record_close_step(
                "session_breakdown", status="failed",
                detail=repr(exc)[:240],
            )

        # ---------------- Step 2.5: Langfuse flush + receipt splice --------
        # MUST run before the artifact package (step 2.6): flush_session
        # reconciles out-of-process children + flips the receipt to final
        # counts, and patch_breakdown_langfuse splices that post-flush
        # receipt back into session_breakdown.json. If this ran AFTER
        # packaging, the bundled SBD would carry counts_final=false and the
        # final langfuse_receipt.json would be missing from the bundle.
        # No-op unless live push is enabled; idempotent (a later cli.finally
        # flush only re-writes the receipt). Best-effort.
        try:
            from .trace.langfuse_emitter import (
                flush_session,
                record_session_breakdown,
            )
            flush_session(self.session_dir)
            from ..breakdown import patch_breakdown_langfuse
            patch_breakdown_langfuse(self.session_dir)
            # After the breakdown file is in its final (post-flush) form, attach
            # the complete JSON to the trace as a ``session_breakdown``
            # observation. Best-effort; no-op when live push is disabled.
            record_session_breakdown(self.session_dir)
        except Exception as exc:  # noqa: BLE001 — defensive
            log.debug("CLOSE step 2.5 (langfuse flush) failed", exc_info=True)
            await self._record_close_step(
                "langfuse_flush", status="failed", detail=repr(exc)[:240],
            )

        # ---------------- Step 2.6: artifact package -> /workspace -------
        # Bundle the curated result/report/analysis files (incl. the
        # session_breakdown just written in step 2) into a single zip
        # placed under ``/workspace`` so the Claw sandbox sync ships it
        # to object storage even when ``$USER_DATA_PATH`` points at a
        # wekafs path outside ``/workspace`` (the common production case).
        # Best-effort: failures are recorded but never abort the close
        # sequence. The zip carries its own PACKAGE_MANIFEST log of what
        # went in / what was missing.
        try:
            from ..breakdown import package_session_artifacts
            pkg_path = package_session_artifacts(
                self.session_dir,
                session_id=str(getattr(self.shared_state, "session_id", "") or ""),
            )
            if pkg_path is not None:
                await self._record_close_step(
                    "artifact_package", status="done", detail=str(pkg_path),
                )
            else:
                await self._record_close_step(
                    "artifact_package", status="skipped",
                    detail="no artifacts matched or dest unwritable",
                )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception("CLOSE step 2.6 (artifact_package) failed")
            await self._record_close_step(
                "artifact_package", status="failed", detail=repr(exc)[:240],
            )

        # ---------------- Step 4: fact finalize (Cortex commit) ----------
        # The canonical step-4 "Cortex session commit": writes
        # update_recipe + finalises the local journal (final_throughput /
        # total_gain_pct). Recorded as the ``fact_finalize`` close_step.
        # Ordered before the retired NDJSON-drain no-op (step 3) so the
        # recipe write is part of the same flush.
        try:
            self.cortex_finalize_recipe_and_journal()
            await self._record_close_step("fact_finalize", status="done")
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception("CLOSE step 4 (fact_finalize) failed")
            await self._record_close_step(
                "fact_finalize", status="failed", detail=repr(exc)[:240],
            )

        # Step 3: (retired) NDJSON drain — no-op marker for close-step ledger consumers (v2 RecipeKB is local-only).
        await self._record_close_step("ndjson_drain", status="skipped")

        # Step 5: mark done
        self.shared_state.close_sequence_done = True
        # CLOSE must set stop_reason so the main run loop terminates next tick (else it ticks forever).
        # Idempotent backstop to the early persist; re-derive rather than hard-code time_exhausted.
        if not self.shared_state.stop_reason:
            self.shared_state.set_stop_reason(self._derive_close_stop_reason())
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "CLOSE step 5 (close_sequence_done save) failed; cli.finally "
                "will still write a safety-net breakdown"
            )
        await self._record_close_step("done", status="done")
        log.info("CLOSE 5-step sequencer complete")

    async def _enqueue_internal_report_task(
        self, *, reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``report`` task (idempotency_key internal-report-<reason>).

        Reuses closing_report_task_id when set so the wall-clock + CLOSE-sequencer paths don't race.
        """
        existing_id = (self.shared_state.closing_report_task_id or "").strip()
        if existing_id:
            try:
                task = await self.tasks.get(existing_id)
                log.info(
                    "internal-report task already enqueued by wall-clock "
                    "deadline path (task_id=%s, state=%s); sequencer will "
                    "wait for it", task.task_id, task.state,
                )
                return task
            except Exception:  # noqa: BLE001 — TaskNotFound + friends
                # Stale id (resume from a wiped-tasks-table session); fall through to fresh enqueue.
                pass

        params: dict[str, Any] = {
            "source":         "coordinator_internal",
            "reason":         str(reason),
            "session_dir":    str(self.session_dir),
            "max_highlights": 50,
        }
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="report",
            params=params,
            idempotency_key=f"internal-report-{reason}",
            requires_lanes=[],
            allowed_tools=["Read"],
            side_effects=["writes_results"],
            lease_ttl_sec=120,
        )
        # Mirror onto closing_report_task_id (used by wall-clock inspectors + robustness/breakdown).
        if not self.shared_state.closing_report_task_id:
            self.shared_state.closing_report_task_id = task.task_id
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.exception(
                    "internal-report: closing_report_task_id save failed"
                )
        if was_existing:
            log.info(
                "internal-report task reused (idempotent: task_id=%s, "
                "state=%s)", task.task_id, task.state,
            )
        return task

    async def _enqueue_internal_research_scout_task(
        self, *, reason: str, round_id: int,
    ) -> "Task | None":
        """Enqueue a Coordinator-owned read-only research-scout specialist task; idempotency keyed by round, returns None on existing/failure (fail-soft)."""
        if not bool(getattr(self.shared_state, "research_scout_enabled", True)):
            return None
        idempotency_key = f"internal-research-scout-round{int(round_id)}"
        try:
            seen = sorted(self._framework_pr_known_candidate_ids())
        except Exception:  # noqa: BLE001 — defensive
            seen = list(
                getattr(self.shared_state, "research_scout_seen_pr_ids", []) or []
            )
        params: dict[str, Any] = {
            "domain": "research_scout_specialist",
            "gap_canonical_id": f"gap.research_scout.round{int(round_id)}",
            "gap_symptom": (
                "Collect proven priors (reference launch scripts, model "
                "config.json architecture features, cross-framework / "
                "NVIDIA research) into prioritised research hints with "
                "sources; do not benchmark or patch."
            ),
            "gap_layer": "research",
            "max_turns": 10,
            "source": "coordinator_internal",
            "reason": str(reason),
            "seen_pr_ids": seen,
            "readonly": True,
        }
        proven = self._warm_recipe_proven_items()
        if proven:
            params["already_proven"] = proven
        await self._warm_specialist_params(params)
        try:
            task, was_existing = await self.tasks.create_or_return_existing(
                kind="specialist",
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=["research_lane"],
                allowed_tools=[
                    "Read", "Grep", "Glob", "Write",
                    "WebSearch", "WebFetch",
                ],
                side_effects=["writes_results"],
                lease_ttl_sec=1800,
            )
        except Exception:  # noqa: BLE001 — TaskRegistry edge cases
            log.exception(
                "research-scout: enqueue failed (round=%d)", int(round_id),
            )
            return None
        if not was_existing:
            self.shared_state.bump_research_scout_runs()
            self.shared_state.research_scout_last_round = int(round_id)
            self.shared_state.save(self.session_dir)
            log.info(
                "research-scout dispatched: task_id=%s round=%d reason=%s "
                "runs=%d",
                task.task_id, int(round_id), reason,
                self.shared_state.research_scout_runs,
            )
        return task

    async def _maybe_enqueue_prelude_research_scout(self) -> None:
        """Force-dispatch the PRELUDE research scout (not LLM-proposable); writes hints skeleton first."""
        try:
            from . import research_hints as _research_hints
            _research_hints.write_hints_skeleton(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: hints skeleton write failed")
        if not bool(getattr(self.shared_state, "research_scout_enabled", True)):
            return
        try:
            await self._enqueue_internal_research_scout_task(
                reason="prelude_initial", round_id=0,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: PRELUDE dispatch failed")

    async def _maybe_enqueue_explore_research_scout(self) -> None:
        """Re-dispatch the scout every K EXPLORE rounds (append-only)."""
        state = self.shared_state
        if not bool(getattr(state, "research_scout_enabled", True)):
            return
        interval = max(1, int(getattr(state, "research_scout_interval", 3) or 3))
        round_id = int((state.explore_search or {}).get("cursor") or 0)
        if round_id <= 0 or (round_id % interval) != 0:
            return
        if int(getattr(state, "research_scout_last_round", -1)) == round_id:
            return
        try:
            await self._enqueue_internal_research_scout_task(
                reason="explore_periodic", round_id=round_id,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: EXPLORE re-dispatch failed")

    def _harvest_research_scout(self, done_payload: dict[str, Any]) -> None:
        """Persist scout output (hints, competitor target, gap seeds, dedup); all steps fail-soft."""
        from . import research_hints as _research_hints

        block = done_payload.get("research")
        if not isinstance(block, dict):
            block = {}
        hints = block.get("hints") or []
        try:
            added, dropped = _research_hints.append_hints(
                self.session_dir, hints,
            )
            if dropped:
                log.info(
                    "research-scout: dropped %d sourceless hint(s)", dropped,
                )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: append_hints failed")
            added = 0
        try:
            _research_hints.write_competitor_target(
                self.session_dir, block.get("competitor_target"),
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: competitor_target write failed")
        # Share inspected PR ids with the FRAMEWORK_PR dedup set.
        pr_ids: list[Any] = []
        for key in ("prs_fetched", "pr_diffs_read", "nvidia_refs"):
            vals = block.get(key)
            if isinstance(vals, list):
                pr_ids.extend(vals)
        try:
            self.shared_state.register_seen_pr_ids(pr_ids)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: register_seen_pr_ids failed")
        # Seed high-priority hints as gaps[] so EXPLORE tries them early.
        try:
            self._seed_gaps_from_research_hints()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("research-scout: gap seeding failed")
        log.info(
            "research-scout harvested: hints_added=%d seen_pr_ids=%d",
            added, len(self.shared_state.research_scout_seen_pr_ids or []),
        )

    def _seed_gaps_from_research_hints(self) -> None:
        """Inject research hints as advisory gaps[] seeds (idempotent)."""
        from . import research_hints as _research_hints

        hints = _research_hints.load_hints(self.session_dir)
        for idx, hint in enumerate(hints):
            what = str(hint.get("what") or "").strip()
            if not what:
                continue
            tags = hint.get("domain_tags") or []
            cid = f"gap.research_hint.{idx}"
            try:
                self.shared_state.upsert_gap({
                    "canonical_id": cid,
                    "symptom": what,
                    "layer": "research_hint",
                    "severity": "medium",
                    "domain_hint": str(tags[0]) if tags else "",
                    "source": "research_scout",
                    "provenance": str(hint.get("source") or ""),
                })
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "research-scout: upsert_gap failed for %s", cid,
                )

    async def _enqueue_internal_session_breakdown_task(
        self, *, reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``session_breakdown`` task; same idempotency contract as the report helper."""
        params: dict[str, Any] = {
            "source":      "coordinator_internal",
            "reason":      str(reason),
            "session_dir": str(self.session_dir),
        }
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="session_breakdown",
            params=params,
            idempotency_key=f"internal-session_breakdown-{reason}",
            requires_lanes=[],
            allowed_tools=["Read"],
            side_effects=["writes_results"],
            lease_ttl_sec=120,
        )
        if was_existing:
            log.info(
                "internal-session_breakdown task reused (idempotent: "
                "task_id=%s, state=%s)", task.task_id, task.state,
            )
        return task

    async def _wait_for_task_terminal(
        self, task_id: str, *, timeout_sec: float,
    ) -> str | None:
        """Poll the TaskRegistry until ``task_id`` reaches a terminal
        state, with a wall-clock timeout.

        Returns the final ``task.state`` (``"succeeded"`` / ``"failed"``
        / ``"cancelled"`` / ``"needs_manual_review"``) or ``None`` on
        timeout (caller treats None as a soft "let's not block on this
        forever" — the CLOSE sequencer records ``status='timeout'``).

        Polling interval is 100ms — small relative to typical report /
        session_breakdown wall time (5-30s); large enough to not
        thrash sqlite under contention.

        Args:
            task_id (str): The task to wait on.
            timeout_sec (float): Maximum wall-clock seconds to poll.

        Returns:
            str | None: The terminal task state, or ``None`` on timeout or if
                the task is not found.
        """
        from .task_registry import TaskNotFound

        deadline = asyncio.get_event_loop().time() + max(0.0, float(timeout_sec))
        poll_interval = 0.1
        terminal = {"succeeded", "failed", "cancelled", "needs_manual_review"}
        while asyncio.get_event_loop().time() < deadline:
            try:
                task = await self.tasks.get(task_id)
            except TaskNotFound:
                return None
            if task.state in terminal:
                return task.state
            await asyncio.sleep(poll_interval)
        return None

    async def _record_close_step(
        self,
        step: str,
        *,
        status: str,
        task_id: str = "",
        detail: str = "",
    ) -> None:
        """Append one row to ``phase_history[-1].evidence.close_steps`` (best-effort, per-step persist)."""
        history = self.shared_state.phase_history or []
        if not history:
            return
        row = history[-1]
        if not isinstance(row, dict):
            return
        evidence = row.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
            row["evidence"] = evidence
        steps = evidence.get("close_steps")
        if not isinstance(steps, list):
            steps = []
            evidence["close_steps"] = steps
        entry: dict[str, Any] = {
            "step":   step,
            "status": status,
            "ts":     datetime.now(timezone.utc).isoformat(),
        }
        if task_id:
            entry["task_id"] = task_id
        if detail:
            entry["detail"] = detail
        steps.append(entry)
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "close_step save failed for step=%r status=%r", step, status,
            )

    # Bounded test interface
    async def _replay_resume_if_needed(self) -> None:
        """Rebuild in-memory state once for a resumed session (replay log + abandon orphan dispatches)."""
        if not (self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]):
            return
        await self.replay_for_resume()

    async def _pump_framework_pr_phase_safely(self, *, caller: str) -> None:
        """Best-effort FRAMEWORK_PR pump wrapper shared by tick and run."""
        try:
            await self._pump_framework_pr_phase()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("FRAMEWORK_PR pump (%s) failed", caller)

    async def tick(self, n: int = 1) -> None:
        """Run exactly ``n`` reactor passes for every agent (P0-3/P0-5/P1-4 tests); dispatcher pumps at pass end, lazy resume replay on tick 1."""
        await self._replay_resume_if_needed()
        for _ in range(n):
            self.shared_state.increment_tick()
            for name in self._tick_roles:
                await self._reactor_pass(name)
            await self._pump_dispatcher_once()
            # FRAMEWORK_PR phase pump: enqueue next candidate / fetch next batch. Best-effort.
            await self._pump_framework_pr_phase_safely(caller="tick")
            # phase machine advance at tick boundary.
            await self._advance_phase_if_needed()

    def _record_coordinator_exception(
        self,
        *,
        stage: str,
        exc: BaseException,
        tick: int | None = None,
        agent: str = "",
    ) -> None:
        """Record a Coordinator-side exception without killing the session."""
        try:
            self.shared_state.record_tick_exception(
                tick=int(tick if tick is not None else self.shared_state.tick or 0),
                stage=stage,
                agent=agent,
                exc_type=type(exc).__name__,
                message=str(exc),
                traceback_text="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
            )
            self.shared_state.increment_crash_count()
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception("failed to persist Coordinator exception metadata")

    # Long-run interface (DESIGN §9 + §21)
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
        """Run reactor + dispatcher until a stop condition fires (DESIGN §9.1, priority order): signal, target_reached, time_exhausted (via closing phase), emergency, custom, max_ticks. Sets + saves + returns shared_state.stop_reason."""
        objective = objective or TimeOnlyObjective()
        # Stash so _compose_prompt can update target_gap_pct.
        self._current_objective = objective
        grace_sec = effective_closing_grace_sec(max_minutes, closing_grace_sec)
        deadline = (
            time.monotonic() + max_minutes * 60.0 if max_minutes else None
        )
        self._run_started_monotonic = time.monotonic()
        self._run_deadline = deadline
        # Capture the live loop so the inline fast-action context tool (A3) can marshal coroutines back here.
        try:
            self._coordinator_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._coordinator_loop = None
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
                # add_signal_handler unavailable on Windows or off the main thread (pytest-asyncio worker).
                log.info("Coordinator.run: signal handlers not installed (%s)", exc)
                previous_handlers = {}

        await self._replay_resume_if_needed()

        tick_n = 0
        stop_reason = ""
        last_tick_exc: BaseException | None = None
        closing_deadline: float | None = None
        try:
            while not stop_reason:
                tick_n += 1
                in_closing = bool(self.shared_state.closing_phase)
                try:
                    # Bump the persistent tick counter — drives phase/plateau math.
                    self.shared_state.increment_tick()
                    in_closing = self.shared_state.closing_phase
                    # One reactor + dispatcher pass; during closing skip LLM passes, pump deterministic report.
                    if not in_closing:
                        for name in self._tick_roles:
                            if self._stop.is_set():
                                break
                            await self._reactor_pass(name)
                        # Orchestration checkpoint/compaction (plan Step 4); cadence-based, no-op off conversational.
                        if not self._stop.is_set():
                            try:
                                await self._maybe_checkpoint_orchestration(
                                    tick=tick_n,
                                )
                            except Exception:  # noqa: BLE001
                                log.exception(
                                    "Coordinator.run: orchestration checkpoint raised"
                                )
                    if not self._stop.is_set():
                        await self._pump_dispatcher_once()
                    # FRAMEWORK_PR phase pump: see ``tick()`` for rationale.
                    if not in_closing:
                        await self._pump_framework_pr_phase_safely(caller="run")
                    # phase machine advance at tick boundary; runs even in_closing so CLOSE is recorded.
                    try:
                        await self._advance_phase_if_needed()
                    except Exception as exc:  # noqa: BLE001
                        log.exception("phase advance (run) failed")
                        self._record_coordinator_exception(
                            stage="advance_phase",
                            exc=exc,
                            tick=tick_n,
                        )
                except (asyncio.CancelledError, KeyboardInterrupt):
                    raise
                except Exception as exc:  # noqa: BLE001
                    last_tick_exc = exc
                    log.exception("Coordinator.run: tick %d body raised", tick_n)
                    self._record_coordinator_exception(
                        stage="tick_body",
                        exc=exc,
                        tick=tick_n,
                    )

                # check stop conditions
                if self._stop.is_set():
                    stop_reason = "signal"
                    break
                if self.shared_state.stop_reason and not in_closing:
                    stop_reason = self.shared_state.stop_reason
                    break
                if objective.reached(self.shared_state):
                    stop_reason = "target_reached"
                    break
                if (
                    deadline is not None
                    and time.monotonic() >= deadline
                    and not in_closing
                ):
                    if grace_sec <= 0:
                        stop_reason = "time_exhausted"
                        break
                    closing_deadline = await self._enter_closing_phase(
                        grace_sec=grace_sec,
                    )
                    continue
                if in_closing:
                    report_terminal = await self._closing_report_terminal()
                    grace_blown = (
                        closing_deadline is not None
                        and time.monotonic() >= closing_deadline
                    )
                    if report_terminal or grace_blown:
                        if grace_blown and not report_terminal:
                            log.warning(
                                "Coordinator: closing-grace exhausted (%.0fs) "
                                "before report task %s finished",
                                grace_sec,
                                self.shared_state.closing_report_task_id,
                            )
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

                # Brief wait between ticks to avoid CPU spin while staying signal-responsive; 0.0 keeps tests fast.
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
            if self.shared_state.closing_phase:
                self.shared_state.closing_phase = False
            # Resuming a terminal session can break out before stop_reason is set; preserve prior reason.
            self.shared_state.set_stop_reason(
                stop_reason
                or self.shared_state.stop_reason
                or ("coordinator_exception" if last_tick_exc is not None else "unknown")
            )
            self.shared_state.save(self.session_dir)
            log.info(
                "Coordinator.run: stopped tick=%d reason=%s baseline_tput=%.1f "
                "cumulative_gain=%.2f%% max_minutes=%.0f",
                tick_n, stop_reason or "unknown",
                self.shared_state.baseline_tput,
                self.shared_state.cumulative_gain,
                max_minutes_value,
            )
            # Best-effort cleanup of installed signal handlers.
            if previous_handlers:
                try:
                    loop = asyncio.get_running_loop()
                    for sig in previous_handlers:
                        loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    pass
        return self.shared_state.stop_reason

    async def _enter_closing_phase(self, *, grace_sec: float) -> float:
        """Enter report-flush phase after the wall-clock deadline (enqueue deterministic report task)."""
        closing_started = time.time()
        closing_deadline = time.monotonic() + float(grace_sec)
        self.shared_state.closing_phase = True
        self.shared_state.closing_started_unix = closing_started
        self.shared_state.save(self.session_dir)

        log.info(
            "Coordinator: entering closing phase (grace=%.0fs); "
            "enqueueing deterministic report task",
            grace_sec,
        )

        try:
            for q in await self.tasks.queued():
                if q.kind == "report":
                    continue
                await self.tasks.transition(
                    q.task_id,
                    "cancelled",
                    evidence={"reason": "closing_phase"},
                )
        except Exception:  # noqa: BLE001
            log.exception(
                "closing_phase: cancel of queued tasks failed (non-fatal)",
            )

        idempotency_key = (
            f"closing-report-{int(closing_started)}-{uuid.uuid4().hex[:6]}"
        )
        task, _existing = await self.tasks.create_or_return_existing(
            kind="report",
            params={
                "session_dir": str(self.session_dir),
                "max_highlights": 50,
            },
            idempotency_key=idempotency_key,
            requires_lanes=[],
            allowed_tools=["Read"],
            side_effects=["writes_results"],
            lease_ttl_sec=120,
        )
        self.shared_state.closing_report_task_id = task.task_id
        self.shared_state.save(self.session_dir)

        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "event",
            {
                "kind": "closing_phase_entered",
                "task_id": task.task_id,
                "grace_sec": float(grace_sec),
                "closing_started_unix": closing_started,
            },
        ))
        return closing_deadline

    async def _closing_report_terminal(self) -> bool:
        """Report whether the closing-phase report task has finished.

        Returns:
            bool: ``True`` when the report task reached a terminal state (or is
                missing); ``False`` while it is still queued or running.
        """
        task_id = self.shared_state.closing_report_task_id
        if not task_id:
            return False
        from .task_registry import TaskNotFound

        try:
            task = await self.tasks.get(task_id)
        except TaskNotFound:
            return True
        return task.state in {
            "succeeded", "failed", "cancelled", "needs_manual_review",
        }

    # Reactor
    async def _reactor_pass(self, agent_name: str) -> None:
        """Run one reactor turn for ``agent_name`` and route its intents.

        Composes the prompt + system prompt, invokes the agent's backend, and
        dispatches every emitted intent through :meth:`_handle_intent`. Backend
        errors, missing intents, and unexpected exceptions are recorded as
        structured observations so a single bad turn never stops the run;
        repeated crashes still bump ``crash_count`` toward the emergency stop.

        Args:
            agent_name (str): The agent role to run this pass for.
        """
        backend = self.backends[agent_name]
        prompt = await self._compose_prompt(agent_name)
        # Accumulate orchestration prompt size as a proxy for conversation growth (plan Step 4).
        if agent_name == "orchestration" and self._orchestration_conversational():
            try:
                self._checkpoint_tracker.chars_add(len(prompt))
            except Exception:  # noqa: BLE001
                pass
        sys_prompt = await self._load_system_prompt(agent_name)
        tools = self.policy.allowed_tools_for_agent(agent_name)
        # max_turns=0 → backend default; ClaudeBackend needs ≥2 for tool_use→tool_result→final-text.
        try:
            result: BackendTurnResult = await backend.run(
                prompt=prompt, system_prompt=sys_prompt, tools=tools, max_turns=0,
            )
        except BackendError as exc:
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "backend_error", "agent": agent_name, "error": repr(exc)},
            )
            await self._track_backend_error_streak(agent_name, exc)
            return
        except NoIntentEmitted as exc:
            # No parseable intents; surface as observation so the next tick self-corrects instead of killing the run.
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "no_intent_emitted", "agent": agent_name,
                 "error": str(exc)[:500]},
            )
            return
        except Exception as exc:  # noqa: BLE001
            # Catch-all so one agent's bad turn never stops the loop (repeated crashes → emergency stop).
            log.exception("reactor pass for %s raised", agent_name)
            await self._record_observation(
                "coordinator", "observation",
                {"kind": "reactor_exception", "agent": agent_name,
                 "error": f"{type(exc).__name__}: {str(exc)[:500]}"},
            )
            self._record_coordinator_exception(
                stage="reactor_pass",
                agent=agent_name,
                exc=exc,
            )
            return
        # Reset the streak — a successful turn proves the backend is alive again.
        if self._backend_error_streak.get(agent_name):
            self._backend_error_streak[agent_name] = 0
            self._backend_error_alarm_armed[agent_name] = True
        # Full-trace A1: record this reactor turn's token spend on the
        # unified ledger. One call site covers every in-process reactor
        # role (orchestration / kernel) whose backend reports usage on
        # metadata (ClaudeBackend + CodexBackend). Best-effort: a trace
        # failure must never affect intent routing.
        self._trace_reactor_llm_call(agent_name, result)
        # Full-trace (conversations): persist the full, redacted
        # prompt+response for this reactor turn. Separate file from the
        # token ledger; same best-effort posture.
        self._record_reactor_conversation(agent_name, result)
        # Completed orchestration turn means SEED delivered; flip flag so later turns send DELTA (plan Step 3).
        if agent_name == "orchestration":
            self._orchestration_seeded = True
        for intent in result.intents:
            await self._handle_intent(agent_name, intent)

    def _trace_reactor_llm_call(
        self, agent_name: str, result: BackendTurnResult,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a reactor turn.

        The reactor role name (``orchestration`` / ``kernel`` / ``critic`` /
        ``robustness``) doubles as both the trace ``component`` and
        ``role``. Only rows carrying real token counters are written, so
        subprocess-backed reactors (critic / robustness) that don't report
        usage here don't pollute the ledger with empty rows — their token
        spend is captured by the dedicated agent collectors instead.

        Wrapped in a broad ``try`` so any unexpected error in trace
        assembly degrades to a logged warning rather than breaking the
        tick loop.
        """
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
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break the loop
            log.debug(
                "full-trace: reactor llm_call append failed for %s",
                agent_name, exc_info=True,
            )

    def _record_reactor_conversation(
        self, agent_name: str, result: BackendTurnResult,
    ) -> None:
        """Append one ``conversations.jsonl`` row for a reactor turn.

        Persists the full (redacted) prompt + completion the backend put on
        ``metadata`` (``prompt`` / ``response``). Only rows that actually
        carry conversation text are written, so subprocess-backed reactors
        (critic / robustness) that don't surface text here don't emit empty
        rows — their conversation is captured by their own workdir artefacts.

        Best-effort: any failure degrades to a logged warning rather than
        breaking the tick loop.
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
                agent_name, exc_info=True,
            )

    async def _track_backend_error_streak(
        self, agent_name: str, exc: BackendError,
    ) -> None:
        """Increment the per-agent ``BackendError`` streak; emit one backend_unhealthy event on crossing the threshold (re-arms only after a successful turn)."""
        new_value = self._backend_error_streak.get(agent_name, 0) + 1
        self._backend_error_streak[agent_name] = new_value
        threshold = self._backend_error_streak_threshold
        if (
            new_value >= threshold
            and self._backend_error_alarm_armed.get(agent_name, True)
        ):
            self._backend_error_alarm_armed[agent_name] = False
            await self._record_observation(
                "coordinator", "observation",
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
                        "backend (e.g. --robustness-mock / --critic-mock) "
                        "while the underlying transport is repaired"
                    ),
                },
            )

    async def _scan_stale_specialists(self) -> list[dict[str, Any]]:
        """Return specialist task rows running longer than ``_specialist_stale_sec`` (v0.8 §3.3 §4.4); never raises, returns [] on failure."""
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
                stale.append({
                    "task_id":         t.task_id,
                    "kind":            t.kind,
                    "running_seconds": running_sec,
                })
        return stale

    async def _compose_prompt(self, agent_name: str) -> str:
        """v0.6 §8.3 prompt: SharedState summary + inbox tail (with canonical msg_id per inbox row)."""
        sections: list[str] = []

        # 0. SESSION_DIR contract — literal path for every agent (pairs with PolicyGate path containment).
        sections.append(f"SESSION_DIR={self.session_dir}")

        # per-tick phase block for every agent, high in the prompt because R1 rejection is phase-driven.
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

        # 0a. Mission progress (Orchestration only), shown before the verbose dump.
        # Conversational delta gating (plan Step 3): first turn gets full SEED, later turns thin DELTA.
        push_full = True
        if agent_name == "orchestration":
            push_full = (
                not self._orchestration_conversational()
                or not self._orchestration_seeded
            )
            if self._orchestration_conversational():
                log.info(
                    "orchestration prompt mode=%s seeded=%s tick=%s",
                    "SEED" if push_full else "DELTA",
                    self._orchestration_seeded,
                    getattr(self.shared_state, "tick", 0),
                )

        # On a full SEED push, inject recovered working memory (plan Step 4) so the agent re-anchors its plan.
        if (
            agent_name == "orchestration"
            and push_full
            and self._orchestration_conversational()
            and self._orchestration_seed_memory
        ):
            sections.append(self._orchestration_seed_memory)

        if agent_name == "orchestration":
            sections.append("=== Mission progress ===")
            sections.append(self.shared_state.to_mission_summary())
            if self._run_deadline is not None and self._run_started_monotonic is not None:
                remaining_min = max(
                    0.0, (self._run_deadline - time.monotonic()) / 60.0,
                )
                elapsed_min = (
                    time.monotonic() - self._run_started_monotonic
                ) / 60.0
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
                        "`explore` rounds (which inline the stack rebench) "
                        "will likely be cut by the deadline."
                    )

        # Time budget for Robustness — fires deadline_imminent → delegate(report) wind-down.
        if (
            agent_name == "robustness"
            and self._run_deadline is not None
            and self._run_started_monotonic is not None
        ):
            remaining_min = max(
                0.0, (self._run_deadline - time.monotonic()) / 60.0,
            )
            elapsed_min = (
                time.monotonic() - self._run_started_monotonic
            ) / 60.0
            budget_min = self.shared_state.max_minutes or 0
            sections.append("=== Time budget ===")
            sections.append(
                f"elapsed={elapsed_min:.1f}min  remaining={remaining_min:.1f}min  "
                f"budget={budget_min}min  "
                f"closing_phase={self.shared_state.closing_phase}"
            )

        # 1. Shared session state — goal + progress context; omitted on orchestration DELTA turns.
        if push_full:
            sections.append("=== Shared session state ===")
            sections.append(self.shared_state.to_prompt_summary())
        if agent_name == "orchestration":
            # target_gap_pct is a fact (gain still needed for --target-gain); refresh to keep prompt current.
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
            # Advisory/ledger blocks below are part of the full SEED push; omitted on DELTA turns.
            if push_full:
                denial_summary = self.shared_state.to_policy_denial_summary(top_k=6)
                if denial_summary:
                    sections.append(denial_summary)

        # Cortex T0 warm-start snapshot + structured gaps[] ledger (replaces retired kb_digest).
        if agent_name == "orchestration" and push_full:
            try:
                warm_block = self.shared_state.to_warm_start_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: warm_start_summary failed")
                warm_block = ""
            if warm_block:
                sections.append("=== Warm start (Cortex T0) ===")
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
                from . import research_hints as _research_hints
                hints_block = _research_hints.summarise_for_prompt(
                    self.session_dir,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: research_hints summary failed")
                hints_block = ""
            if hints_block:
                sections.append("=== Research hints (advisory) ===")
                sections.append(hints_block)
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

        # Conversational DELTA turn: tell the agent verbose state was not re-pushed + how to pull it.
        if agent_name == "orchestration" and not push_full:
            sections.append("=== Context (pull on demand) ===")
            sections.append(
                "This is a continuation of our ongoing conversation; the "
                "full session state was NOT re-pasted. The Phase, Mission "
                "progress, Time budget, and new inbox events above are the "
                "delta since your last turn. Pull anything else you need "
                "with the read-only context tools: get_shared_state, "
                "get_gaps, get_warm_start, get_proposal_scores, "
                "get_intervention_mix, why_denied, show_analysis_md, "
                "get_inbox. Reason from your own running plan; do not "
                "re-derive it from scratch."
            )

        # Robustness gets phase budget telemetry + specialist health for medium-severity alerts.
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
            try:
                stale = await self._scan_stale_specialists()
                running = await self.tasks.running()
                specialist_running = sum(
                    1 for t in (running or [])
                    if (t.kind or "").strip() == "specialist"
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: specialist health scan failed")
                stale, specialist_running = [], 0
            stale_lines = [
                f"  - task_id={row['task_id']} running_sec={int(row['running_seconds'])}"
                for row in stale
            ]
            sections.append("=== Specialist health ===")
            sections.append(
                f"running={specialist_running} stale={len(stale)} "
                f"stale_threshold_sec={int(self._specialist_stale_sec)}"
            )
            if stale_lines:
                sections.append("stale specialists (consider kill_task):")
                sections.extend(stale_lines)

            # Conversation no-progress circuit-breaker (plan Step 6); Robustness is the external safety net.
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
        if msgs:
            sections.append(f"=== Inbox for {agent_name} (newest last) ===")
            for m in msgs[-20:]:
                # Structured rendering for delegated_result/denial/verdict (Path A/A1); compact dump otherwise.
                sections.append(f"  {_format_inbox_event(m)}")
        else:
            sections.append(f"=== Inbox for {agent_name} ===")
            sections.append("(no new messages)")

        return "\n".join(sections)

    async def _load_system_prompt(self, agent_name: str) -> str:
        """Load the system prompt for an agent, honoring overrides.

        Args:
            agent_name: Name of the agent/role whose prompt to load.

        Returns:
            The override prompt if configured, the role's prompt file
            contents, or a placeholder string when none exists.
        """
        # Demo/test override via self.system_prompt_overrides[agent_name].
        override = getattr(self, "system_prompt_overrides", {}).get(agent_name)
        if override is not None:
            return override
        role = self.role_registry[agent_name]
        try:
            return role.load_system_prompt()
        except FileNotFoundError:
            return f"(no system prompt for {agent_name})"

    # Execution order guard
    def _target_analysis_baseline_exists(self) -> bool:
        """True iff target_analysis produced ``target_baseline.json`` (file existence is a sufficient gate signal)."""
        try:
            from ..session_paths import target_baseline_json
            return target_baseline_json(self.session_dir).exists()
        except Exception:  # noqa: BLE001 — defensive; missing helper -> treat as done.
            return True

    def _kernel_opt_keep_pending(self) -> str:
        """Return the next kernel_id awaiting integrate, or "" if none (delegates to SharedState.next_pending_keep_kernel_id)."""
        return self.shared_state.next_pending_keep_kernel_id()

    def _sequence_denial_for_action(
        self,
        action_name: str,
        proposed_params: dict[str, Any] | None = None,
    ) -> PolicyDenied | None:
        """Reject orchestration action/delegate attempts before baseline. Only invariant: nothing runs until baseline_tput > 0 (a data-dependency). proposed_params kept for signature compat."""
        action = str(action_name or "").strip()
        sequence_actions = {
            "target_analysis",
            "baseline", "profile", "roofline",
            "sweep", "report", "integrate", "explore",
        }
        if action not in sequence_actions:
            return None
        if self.shared_state.stop_reason:
            return None
        if (
            self.shared_state.baseline_tput <= 0
            and action not in {"baseline", "target_analysis"}
        ):
            return PolicyDenied(
                f"action={action!r} denied: baseline must run first",
                rule="execution_order",
                hint="propose/delegate `baseline` until baseline_tput > 0",
            )
        return None

    def _sequence_denial_for_request(
        self, target_agent: str, kind: str,
    ) -> PolicyDenied | None:
        """Reject kernel requests that skip the baseline prerequisite (invariant: nothing kernel-side runs before baseline_tput > 0)."""
        target = str(target_agent or "").strip()
        req_kind = str(kind or "").strip()
        if target != "kernel" or self.shared_state.stop_reason:
            return None
        if req_kind == "trace_analyze":
            return None
        if get_handler(req_kind) is None:
            return None
        if self.shared_state.baseline_tput <= 0:
            return PolicyDenied(
                f"request kind={req_kind!r} denied: baseline must run first",
                rule="execution_order",
                hint="propose/delegate `baseline` before kernel requests",
            )
        return None

    @staticmethod
    def _skip_gemm_tuning() -> bool:
        """Report whether GEMM tuning is disabled via the env escape hatch.

        Returns:
            bool: ``True`` when ``INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING`` is set.
        """
        return os.environ.get(
            "INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", "",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _gemm_tuning_required_before_kernel_opt(self) -> bool:
        """Decide whether FP8 SGLang GEMM tuning must run before kernel_opt.

        Only required for ``precision='fp8'`` + ``framework='sglang'`` sessions
        whose ``last_gemm_tuning`` has not yet reached a terminal status.

        Returns:
            bool: ``True`` when GEMM tuning should run before source-level
                ``kernel_opt``.
        """
        if self._skip_gemm_tuning():
            return False
        ss = self.shared_state
        precision = str(getattr(ss, "precision", "") or "").strip().lower()
        framework = str(getattr(ss, "framework", "") or "").strip().lower()
        if precision != "fp8" or framework != "sglang":
            return False
        last = getattr(ss, "last_gemm_tuning", {}) or {}
        status = str(last.get("status") or "").strip().lower()
        return status not in {
            "ok",
            "succeeded",
            "success",
            "complete",
            "completed",
            "skipped",
            "failed",
        }

    # Intent handling
    async def _handle_intent(self, source: str, intent: Intent) -> None:
        """Validate an emitted intent through PolicyGate, then route it.

        Runs the intent through :meth:`PolicyGate.validate_intent`; a
        :class:`PolicyDenied` is recorded and the intent dropped. Valid intents
        are dispatched to the matching ``_handle_*`` method by type, and the
        agent's message cursor is advanced to the latest sequence afterward.

        Args:
            source (str): The agent that emitted the intent.
            intent (Intent): The parsed intent to validate and route.
        """
        try:
            self.policy.validate_intent(source, intent)
        except PolicyDenied as denied:
            await self._record_policy_denied(source, intent, denied)
            return

        try:
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
            elif it == IntentType.SPECIALIST_DONE:
                # Terminal specialist intent (R3 already validated); handler only bookkeeps. Defense-in-depth.
                await self._handle_specialist_done(source, intent)
            else:
                # ASK_QUESTION / ANSWER / UPDATE_PERSONA — record for replay
                await self._record_observation(
                    source, "observation",
                    {"intent": it.value, "payload": intent.payload},
                )
            await self._cursor_advance_to_latest(source)
        except (asyncio.CancelledError, KeyboardInterrupt):
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("intent handler for %s raised", source)
            self._record_coordinator_exception(
                stage="handle_intent",
                agent=source,
                exc=exc,
            )
            try:
                await self._record_observation(
                    "coordinator",
                    "observation",
                    {
                        "kind": "handle_intent_exception",
                        "agent": source,
                        "intent_type": intent.type.value,
                        "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                    },
                )
            except Exception:  # noqa: BLE001
                log.exception("failed to record handle_intent_exception observation")
            return

    # PROPOSE_ACTION + REVIEW_VERDICT
    async def _handle_propose_action(self, source: str, intent: Intent) -> None:
        """Gate a proposed action and enqueue it for Critic Review.

        Drops proposals for pruned families, applies the pending-roofline and
        execution-order denials, then publishes a ``proposal`` message and
        registers a :class:`PendingProposal` so the Critic gate (§18) can later
        return a verdict.

        Args:
            source (str): The agent proposing the action.
            intent (Intent): The PROPOSE_ACTION intent; ``payload`` carries
                ``action_name`` and optional ``params`` / ``predicted_gain_pct``.
        """
        action_name = intent.payload["action_name"]
        # Pruned families are advisory: proposal still queues, but the inbox carries an advisory note.
        if self.shared_state.is_pruned(action_name):
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind": "proposal_pruned_advisory",
                    "from": source,
                    "action": action_name,
                    "hint": (
                        f"{action_name!r} is in pruned_families; if the "
                        "prune was speculative the LLM may pick this "
                        "action again, otherwise prefer another "
                        "phase-allowed action."
                    ),
                },
            )
        denied = self._sequence_denial_for_action(
            action_name,
            proposed_params=intent.payload.get("params"),
        )
        if denied is not None:
            await self._record_policy_denied(source, intent, denied)
            return
        msg = Message.new(
            source, "*", "proposal",
            {**intent.payload, "needs_review": True},
            priority=1,
        )
        await self.bus.append_and_seq(msg)
        pending = PendingProposal(
            proposal_msg_id=msg.msg_id,
            from_agent=source,
            action_name=action_name,
            predicted_gain_pct=float(intent.payload.get("predicted_gain_pct", 0.0)),
            payload=dict(intent.payload),
        )
        # KB hypothesize/verify retired; proposals enter the queue directly, facts written after task lands.
        self.state.pending_proposals[msg.msg_id] = pending

    def _resolve_issue_canonical(self, pending: PendingProposal) -> str:
        """Find the issue_node canonical_id this proposal addresses. Priority: payload gap_canonical_id → params gap_canonical_id → _gap_anchor_canonical_id (Gap-09)."""
        payload = pending.payload or {}
        explicit_top = str(payload.get("gap_canonical_id") or "").strip()
        if explicit_top:
            return explicit_top
        params = payload.get("params") or {}
        if isinstance(params, dict):
            explicit_params = str(params.get("gap_canonical_id") or "").strip()
            if explicit_params:
                return explicit_params
        return self._gap_anchor_canonical_id()

    def _workload_canonical_id(self) -> str:
        """Canonical 5-tuple recipe id for the current workload. MUST match cortex_t0.run_t0_anchor's derivation so warm-start and KEEP/REVERT/CLOSE writes target the same row."""
        ss = self.shared_state
        workload = ss.model_name or "unknown_model"
        hw = ss.gpu_type or "unknown_gpu"
        framework = str(getattr(ss, "framework", "") or "")
        framework_version = str(getattr(ss, "framework_version", "") or "")
        if not framework_version and framework:
            framework_version = detect_framework_version(framework)
        precision = str(getattr(ss, "precision", "") or "")
        return recipe_canonical_id(
            model=workload,
            hardware=hw,
            framework=framework,
            framework_version=framework_version,
            precision=precision,
        )

    def _gap_anchor_canonical_id(self) -> str:
        """M1 gap anchor: delegates to _workload_canonical_id so anchor and write target never diverge."""
        return self._workload_canonical_id()

    def _kb_amend_recipe(
        self,
        *,
        append_lesson: dict[str, Any] | None = None,
        append_pitfall: dict[str, Any] | None = None,
        recipe_overrides: dict[str, Any] | None = None,
        provenance_details: dict[str, Any] | None = None,
    ) -> None:
        """Read-modify-write helper for the v2 recipe-snapshot KB: load live row, append lesson/pitfall, merge recipe_overrides (unset fields preserved), write back. Best-effort; lesson/pitfall appended without dedup (commit 4d)."""
        if self.cortex_kb is None:
            return
        try:
            cid = self._workload_canonical_id()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("_kb_amend_recipe: cid derivation failed")
            return

        ss = self.shared_state
        framework = str(getattr(ss, "framework", "") or "")
        framework_version = str(getattr(ss, "framework_version", "") or "")
        if not framework_version and framework:
            framework_version = detect_framework_version(framework)
        precision = str(getattr(ss, "precision", "") or "")

        # Read the LOCAL authoritative row (bypass remote-first read; else a central row clobbers this session's lessons/pitfalls).
        try:
            live = self.cortex_kb.local.get_recipe(canonical_id=cid) or {}
        except Exception as exc:  # noqa: BLE001 — best-effort read
            log.info(
                "_kb_amend_recipe: local get_recipe failed (%s); "
                "proceeding with empty live",
                exc,
            )
            live = {}

        lessons = list(live.get("lessons") or [])
        if append_lesson is not None:
            lessons.append(append_lesson)
        pitfalls = list(live.get("pitfalls") or [])
        if append_pitfall is not None:
            pitfalls.append(append_pitfall)

        # Build put_recipe kwargs, preserving live fields the caller didn't override.
        overrides = dict(recipe_overrides or {})
        # Preserve T0-stamped top-level extras across the amend (caller's extras win).
        _reserved = {
            "canonical_id", "version", "created_at", "updated_at",
            "model", "hardware", "framework", "framework_version",
            "precision", "best_config", "best_throughput",
            "what_worked", "what_failed", "remaining_gaps",
            "prs_tested", "pitfalls", "lessons", "last_profiled",
            "stack_fingerprint", "sessions", "authority", "confidence",
            "evidence_refs", "provenance",
        }
        prior_extras = {k: v for k, v in live.items() if k not in _reserved}
        merged_extras = {**prior_extras, **(overrides.get("extras") or {})}
        # Re-stamp config.json architecture-identity tags (architectures/model_type); skipped when unset.
        _arch = getattr(ss, "model_architectures", None) or []
        if isinstance(_arch, list):
            _arch_list = [str(a).strip() for a in _arch if str(a or "").strip()]
            if _arch_list:
                merged_extras["architectures"] = _arch_list
        _mtype = str(getattr(ss, "model_type", "") or "").strip()
        if _mtype:
            merged_extras["model_type"] = _mtype
        put_kwargs: dict[str, Any] = {
            "canonical_id":      cid,
            "model":             ss.model_name or "unknown_model",
            "hardware":          ss.gpu_type   or "unknown_gpu",
            "framework":         framework,
            "framework_version": framework_version,
            "precision":         precision,
            "best_config":       overrides.get("best_config")
                                  if "best_config" in overrides
                                  else dict(live.get("best_config") or {}),
            "best_throughput":   overrides.get("best_throughput")
                                  if "best_throughput" in overrides
                                  else float(live.get("best_throughput") or 0.0),
            "what_worked":       overrides.get("what_worked")
                                  if "what_worked" in overrides
                                  else list(live.get("what_worked") or []),
            "what_failed":       overrides.get("what_failed")
                                  if "what_failed" in overrides
                                  else list(live.get("what_failed") or []),
            "remaining_gaps":    overrides.get("remaining_gaps")
                                  if "remaining_gaps" in overrides
                                  else list(live.get("remaining_gaps") or []),
            "prs_tested":        overrides.get("prs_tested")
                                  if "prs_tested" in overrides
                                  else list(live.get("prs_tested") or []),
            "pitfalls":          pitfalls,
            "lessons":           lessons,
            "last_profiled":     overrides.get("last_profiled")
                                  if "last_profiled" in overrides
                                  else str(live.get("last_profiled") or ""),
            "stack_fingerprint": overrides.get("stack_fingerprint")
                                  if "stack_fingerprint" in overrides
                                  else dict(live.get("stack_fingerprint") or {}),
            "sessions":          overrides.get("sessions")
                                  if "sessions" in overrides
                                  else list(live.get("sessions") or []),
            "extras":            merged_extras,
            # Preserve audit fields across the amend (else put_recipe resets them to defaults).
            "authority":         overrides.get("authority")
                                  if "authority" in overrides
                                  else str(live.get("authority") or "EXPERIENTIAL"),
            "confidence":        overrides.get("confidence")
                                  if "confidence" in overrides
                                  else float(live.get("confidence") or 0.85),
            "evidence_refs":     overrides.get("evidence_refs")
                                  if "evidence_refs" in overrides
                                  else list(live.get("evidence_refs") or []),
            "provenance":        {
                "source":       "hyperloom-inference-optimizer",
                "generator":    "coordinator",
                "generated_at": datetime.now(timezone.utc).isoformat(
                    timespec="microseconds",
                ),
                "details":      dict(provenance_details or {}),
            },
        }
        try:
            self.cortex_kb.put_recipe(**put_kwargs)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "_kb_amend_recipe: put_recipe failed for cid=%s", cid,
            )

    async def _handle_review_verdict(self, source: str, intent: Intent) -> None:
        """Apply a Critic ``review_verdict`` to its target proposal; legacy verdict_map collapsed (approve > reject > needs_review)."""
        target = intent.payload["target_proposal_msg_id"]
        pending = self.state.pending_proposals.get(target)
        verdict_map = intent.payload.get("verdict_map")
        single_verdict = intent.payload.get("verdict")
        if pending is None:
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind":         "verdict_for_unknown_proposal",
                    "target":       target,
                    "verdict":      single_verdict or "",
                    "verdict_map":  bool(verdict_map),
                },
            )
            return
        verdict = str(single_verdict or "")
        if not verdict and isinstance(verdict_map, dict) and verdict_map:
            sub_verdicts = [
                str((entry or {}).get("verdict") or "").strip()
                for entry in verdict_map.values()
            ]
            verdict = (
                "approve" if "approve" in sub_verdicts
                else "reject" if "reject" in sub_verdicts
                else "needs_review"
            )
        await self._handle_single_verdict(
            source=source,
            pending=pending,
            verdict=verdict,
            reasoning=str(intent.payload.get("reasoning") or ""),
        )

    async def _handle_single_verdict(
        self,
        *,
        source: str,
        pending: "PendingProposal",
        verdict: str,
        reasoning: str,
    ) -> None:
        """Legacy v0.6 single-verdict handler (approve materialises proposal as-is); mirrors integrate_patch/specialist verdicts onto specialist_patch_verdicts for PolicyGate."""
        pending.decided = True
        pending.verdict = verdict
        await self.bus.append_and_seq(Message.new(
            source, pending.from_agent, "review_verdict",
            {
                "target_proposal_msg_id": pending.proposal_msg_id,
                "verdict":                verdict,
                "reasoning":              reasoning,
            },
            priority=0 if verdict == "reject" else 1,
            in_reply_to=pending.proposal_msg_id,
        ))
        # Mirror specialist / integrate_patch verdicts onto SharedState so
        # PolicyGate's integrate_patch gate can consult them on the next tick.
        try:
            pa_params = pending.payload.get("params") or {}
        except AttributeError:
            pa_params = {}
        sid_candidate = ""
        if pending.action_name == "integrate_patch":
            sid_candidate = str(
                pa_params.get("specialist_task_id") or ""
            ).strip()
        elif pending.action_name == "specialist":
            # Critic verdict on the specialist proposal counts as the verdict on its patches; task_id is the key.
            sid_candidate = str(pending.task_id or "").strip()
        if sid_candidate and verdict:
            try:
                self.shared_state.record_specialist_patch_verdict(
                    sid_candidate, verdict,
                )
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — best-effort mirror
                log.exception(
                    "failed to mirror critic verdict for specialist task=%s",
                    sid_candidate,
                )
        if verdict == "approve":
            await self._materialize_approved_proposal(pending)

    def _inject_explore_runtime_params(self, params: dict) -> None:
        """Inject explore-task operational knobs from SharedState into ``params`` (single source of truth for both propose/Critic and direct-delegate paths). setdefault preserves LLM overrides. Knobs: baseline_runtime_sec + explore_overtime_kill_ratio (Fix E soft_deadline), variant_timeout_sec, variant_timeout_safety_margin, roofline_saturation_snapshot (advisory)."""
        br = float(getattr(self.shared_state, "baseline_runtime_sec", 0.0) or 0.0)
        if br > 0:
            params.setdefault("baseline_runtime_sec", br)
        kill_ratio = float(getattr(
            self.shared_state, "explore_overtime_kill_ratio", 0.0,
        ) or 0.0)
        if kill_ratio > 0:
            params.setdefault("explore_overtime_kill_ratio", kill_ratio)
        variant_timeout_override = int(getattr(
            self.shared_state, "explore_variant_timeout_sec_override", 0,
        ) or 0)
        if variant_timeout_override > 0:
            params.setdefault("variant_timeout_sec", variant_timeout_override)
        safety_margin_override = float(getattr(
            self.shared_state, "explore_variant_timeout_safety_margin", -1.0,
        ))
        if safety_margin_override >= 0:
            params.setdefault(
                "variant_timeout_safety_margin", safety_margin_override,
            )
        history = list(getattr(
            self.shared_state, "roofline_saturation_history", [],
        ) or [])
        if history and isinstance(history[-1], dict):
            params.setdefault(
                "roofline_saturation_snapshot", dict(history[-1]),
            )
        # Thread the persisted explore_search ledger so ExploreExecutor's canonical_fingerprint dedup has cross-turn memory; setdefault keeps an explicit override.
        es = getattr(self.shared_state, "explore_search", None)
        if isinstance(es, dict) and es.get("tested"):
            params.setdefault("explore_search", es)

    async def _materialize_approved_proposal(
        self,
        pending: PendingProposal,
        *,
        approved_variant_names: set[str] | None = None,
    ) -> None:
        """Promote an approved proposal into a TaskRegistry entry. Grid executors get current best tput as base_tput (DESIGN §16); approved_variant_names filters the explore grid (None keeps full)."""
        params = dict(pending.payload.get("params") or {})
        # Filter the grid to the Critic-approved subset.
        if (
            pending.action_name == "explore"
            and isinstance(params.get("grid"), list)
        ):
            stamped_grid: list[dict[str, Any]] = []
            for variant in params["grid"]:
                if not isinstance(variant, dict):
                    # Non-dict slots can't carry a name: dropped under a filter, pass-through otherwise.
                    if approved_variant_names is None:
                        stamped_grid.append(variant)
                    continue
                vname = str(variant.get("name") or "").strip()
                # drop variants the Critic rejected before they hit the executor.
                if (
                    approved_variant_names is not None
                    and vname not in approved_variant_names
                ):
                    continue
                stamped_grid.append(dict(variant))
            params["grid"] = stamped_grid
            # Audit hint: how many variants the Critic filtered (surfaced as critic_filtered_count).
            if approved_variant_names is not None:
                original_grid_len = len(
                    [
                        v for v in (pending.payload.get("params") or {}).get("grid", [])
                        if isinstance(v, dict)
                    ]
                )
                params["critic_filtered_count"] = max(
                    0, original_grid_len - len(approved_variant_names),
                )
        cb = self.shared_state.current_best or {}
        cb_args = (
            str(cb.get("extra_server_args") or "")
            if isinstance(cb, dict) else ""
        )
        if pending.action_name == "profile":
            # Stamp base_extra_args so post-task promotion records the server config that produced this trace.
            params.setdefault("base_extra_args", cb_args)
        if pending.action_name == "sweep":
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            base = cb_tput if isinstance(cb_tput, (int, float)) and cb_tput > 0 \
                else self.shared_state.baseline_tput
            params.setdefault("base_tput", float(base or 0.0))
            params.setdefault("base_extra_args", cb_args)
            if self.shared_state.baseline_config_path:
                params.setdefault(
                    "config_path", self.shared_state.baseline_config_path
                )
        if pending.action_name == "explore":
            self._inject_explore_runtime_params(params)
            # Inject base_tput/base_extra_args tied to current_best (or baseline_tput); else _gain_pct
            # returns None and every variant lands FAILED. Explicit operator value wins via setdefault.
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            base = cb_tput if isinstance(cb_tput, (int, float)) and cb_tput > 0 \
                else self.shared_state.baseline_tput
            params.setdefault("base_tput", float(base or 0.0))
            params.setdefault("base_extra_args", cb_args)
        lanes, ttl = self._registry_lanes_ttl(pending.action_name)
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=pending.action_name,
            params=params,
            idempotency_key=f"approved-{pending.proposal_msg_id}",
            requires_lanes=lanes,
            lease_ttl_sec=ttl,
        )
        if was_existing:
            # Key is unique per proposal; only a resume/replay collides. Record an observation, not a fresh decision.
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind": "proposal_materialize_skipped",
                    "reason": "duplicate_idempotency_key",
                    "proposal_msg_id": pending.proposal_msg_id,
                    "task_id": task.task_id,
                    "task_state": task.state,
                    "action_name": pending.action_name,
                    "from_agent": pending.from_agent,
                },
            )
            return
        # proposal_msg_id is the resume contract for the deferred queue (see replay_for_resume): pairs a
        # materialize_blocked observation with a later approved_proposal decision as "drained".
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "decision",
            {"kind": "approved_proposal", "task_id": task.task_id,
             "action_name": pending.action_name, "from_agent": pending.from_agent,
             "proposal_msg_id": pending.proposal_msg_id},
        ))

    # DELEGATE
    async def _handle_delegate(self, source: str, intent: Intent) -> None:
        """Validate and enqueue a delegated action as a TaskRegistry task.

        Drops pruned families and execution-order violations, re-routes
        ``explore`` grids through the Critic-review path, and otherwise
        materialises the delegated action (specialist, dynamic action, etc.)
        into a task with the appropriate lanes, tools and warmed params.

        Args:
            source (str): The agent issuing the delegation.
            intent (Intent): The DELEGATE intent; ``payload`` carries
                ``action_name`` and optional ``params``.
        """
        action_name = intent.payload["action_name"]
        if self.shared_state.is_pruned(action_name):
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind": "delegate_pruned_advisory",
                    "from": source,
                    "action": action_name,
                    "hint": (
                        f"{action_name!r} is in pruned_families; if the "
                        "prune was speculative the LLM may pick this "
                        "action again, otherwise prefer another "
                        "phase-allowed action."
                    ),
                },
            )
        denied = self._sequence_denial_for_action(
            action_name,
            proposed_params=intent.payload.get("params"),
        )
        if denied is not None:
            await self._record_policy_denied(
                source, intent, denied, action_name=action_name,
            )
            return
        # delegate explore runs variants directly (config/env grids are not source patches → no Critic pre-review).
        params = dict(intent.payload.get("params") or {})
        # idempotency_key is top-level per schema; treat a nested params value as a compat alias and strip it.
        nested_idempotency_key = params.pop("idempotency_key", None)
        # Plumb baseline's materialized YAML into grid-style tasks for the workload contract; setdefault lets delegator override.
        if (
            action_name in ("sweep", "explore")
            and self.shared_state.baseline_config_path
        ):
            params.setdefault(
                "config_path", self.shared_state.baseline_config_path
            )
        # Parity with _materialize_approved_proposal: direct delegates need the same operational knobs.
        if action_name == "explore":
            self._inject_explore_runtime_params(params)
            # Inject base_tput tied to current_best (or baseline_tput); else every variant lands FAILED.
            # Defensive getattr: lightweight state doubles in tests may omit current_best.
            cb = getattr(self.shared_state, "current_best", None) or {}
            cb_tput = cb.get("tput") if isinstance(cb, dict) else None
            base = cb_tput if isinstance(cb_tput, (int, float)) and cb_tput > 0 \
                else getattr(self.shared_state, "baseline_tput", 0.0)
            params.setdefault("base_tput", float(base or 0.0))
        # Wave sugar: a specialist delegate carrying params.tasks=[...] fans
        # out into N standard freeform specialist tasks (scope=freeform,
        # lane=cpu, mode=research defaults), each dispatched through the
        # normal SpecialistRunner + TaskRegistry + lease + reap path. This
        # preserves the low-cost wide-net recon that the retired
        # dynamic_specialist channel provided.
        if action_name == "specialist" and isinstance(
            params.get("tasks"), list,
        ) and params["tasks"]:
            await self._fan_out_specialist_wave(source, intent, params)
            return
        # Specialist pre-dispatch warmup: warm external-knowledge sections via KnowledgePlane (setdefault fills gaps).
        if action_name == "specialist":
            await self._warm_specialist_params(params)
        # Idempotency-key chain: top-level → nested compat alias → content-fingerprint auto-key.
        # Terminal collisions retry with -retry<N> (up to 5); non-terminal collisions → policy_denied.
        raw_key = intent.payload.get("idempotency_key") or nested_idempotency_key
        if not raw_key:
            content_fp = hashlib.sha1(
                json.dumps(params, sort_keys=True, default=str).encode()
            ).hexdigest()[:10]
            raw_key = (
                f"{source}:{action_name}:t{int(self.shared_state.tick or 0)}:"
                f"{content_fp}"
            )
        idempotency_key = str(raw_key)
        terminal_states = {
            "succeeded", "failed", "cancelled", "needs_manual_review",
        }
        task = None
        was_existing = False
        for attempt in range(6):
            idempotency_key = (
                str(raw_key) if attempt == 0 else f"{raw_key}-retry{attempt}"
            )
            lanes, ttl = self._registry_lanes_ttl(action_name)
            # Bench-enabled specialists serialize against the other GPU
            # benchmark/profile/server work via benchmark_lane (research_lane
            # alone conflicts with nothing).
            if action_name == "specialist":
                from .specialist_profile import resolve_specialist_profile
                if resolve_specialist_profile(params).grants_bench_tool:
                    lanes = tuple(dict.fromkeys((*lanes, "benchmark_lane")))
            task, was_existing = await self.tasks.create_or_return_existing(
                kind=action_name,
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=lanes,
                lease_ttl_sec=ttl,
            )
            if not was_existing:
                break
            if task.state not in terminal_states:
                hint = (
                    f"task {task.task_id} is still {task.state!r}; wait for the "
                    f"delegated_result event instead of re-emitting the same key."
                )
                await self._record_policy_denied(
                    source, intent,
                    PolicyDenied(
                        f"delegate{{action_name={action_name!r}}} duplicate "
                        f"idempotency_key={idempotency_key!r}",
                        rule="duplicate_idempotency_key_running",
                        hint=hint,
                    ),
                    action_name=action_name,
                )
                return
        else:
            hint = (
                f"task {task.task_id if task else '?'} terminated and could not "
                f"allocate a fresh idempotency_key after 5 retries"
            )
            await self._record_policy_denied(
                source, intent,
                PolicyDenied(
                    f"delegate{{action_name={action_name!r}}} duplicate "
                    f"idempotency_key exhausted retries for {raw_key!r}",
                    rule="duplicate_idempotency_key",
                    hint=hint,
                ),
                action_name=action_name,
            )
            return
        self.shared_state.reset_policy_denial_streak(action_name)
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "event",
            {"kind": "task_queued", "task_id": task.task_id, "source": source, "action": action_name},
        ))

    def _record_framework_pr_authored_outcome(
        self, *, task: "Task", result: Any,
    ) -> None:
        """Bridge an authored-patch ``integrate_patch`` outcome into the FRAMEWORK_PR progress ledger (else the gain is invisible). Attributed to the latest batch; only kept/reverted rows."""
        res = getattr(result, "result", None)
        if not isinstance(res, dict):
            return
        status = str(res.get("status") or "")
        if status not in ("kept", "reverted"):
            return
        params = getattr(task, "params", None) or {}
        cand_id = str(
            params.get("framework_pr_candidate_id")
            or params.get("specialist_task_id")
            or getattr(task, "task_id", "")
            or ""
        )
        batch_id = str(params.get("framework_pr_batch_id") or "")
        if not batch_id:
            batches = getattr(self.shared_state, "framework_pr_batches", None) or []
            if isinstance(batches, list) and batches and isinstance(batches[-1], dict):
                batch_id = str(batches[-1].get("batch_id") or "")
        delta_pct = res.get("delta_pct")
        new_tput = res.get("output_throughput")
        gain = float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0
        progress_entry = {
            "candidate_id": cand_id,
            "pr_url":       "",
            "status":       status,
            "provenance":   "authored",
            "pre_tput":     float(
                getattr(self.shared_state, "baseline_tput", 0.0) or 0.0
            ),
            "post_tput":    float(new_tput) if isinstance(new_tput, (int, float)) else 0.0,
            "gain_pct":     gain,
            "kept":         status == "kept",
            "batch_id":     batch_id,
            "ts":           datetime.now(timezone.utc).isoformat(),
        }
        if not isinstance(self.shared_state.framework_pr_phase_progress, list):
            self.shared_state.framework_pr_phase_progress = []
        self.shared_state.framework_pr_phase_progress.append(progress_entry)
        # Roll the batch max-gain stat the plateau judge reads.
        batches = getattr(self.shared_state, "framework_pr_batches", None) or []
        if isinstance(batches, list) and batch_id:
            for entry in reversed(batches):
                if isinstance(entry, dict) and str(entry.get("batch_id") or "") == batch_id:
                    prev = float(entry.get("max_gain_pct_observed_in_batch") or 0.0)
                    if gain > prev:
                        entry["max_gain_pct_observed_in_batch"] = gain
                    break
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "FRAMEWORK_PR authored-outcome: save failed for task=%s",
                getattr(task, "task_id", "?"),
            )
        log.info(
            "FRAMEWORK_PR: authored patch outcome candidate=%s batch=%s "
            "status=%s gain=%.2f%%",
            cand_id, batch_id, status, gain,
        )

    async def _fan_out_specialist_wave(
        self, source: str, intent: Intent, params: dict[str, Any],
    ) -> None:
        """Fan a specialist delegate carrying ``params.tasks=[...]`` into N
        standard free-form specialist dispatches (scope=freeform, lane=cpu,
        mode=research defaults). Each fanned task is re-dispatched through the
        normal ``_handle_delegate`` path (warm + idempotency + TaskRegistry +
        lease + reap), preserving the low-cost wide-net recon the retired
        dynamic_specialist channel provided. Per-task idempotency keys derive
        from the wave key; non-dict / empty-description entries are skipped."""
        tasks = params.get("tasks") or []
        shared = {k: v for k, v in params.items() if k != "tasks"}
        base_key = str(intent.payload.get("idempotency_key") or "").strip()
        for idx, task in enumerate(tasks):
            if not isinstance(task, dict):
                continue
            desc = str(
                task.get("task_description") or task.get("task_summary") or ""
            ).strip()
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
                "mode", "bench", "lane", "model", "priority",
                "timeout_minutes", "max_turns",
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
                source, Intent(type=intent.type, payload=sub_payload),
            )

    async def _maybe_auto_retry_specialist(
        self, task: "Task", result: "SubAgentResult",
    ) -> bool:
        """Re-enqueue a fresh specialist task on a transient infra failure.

        Returns ``True`` when a retry was scheduled (the caller must then skip
        this attempt's delegated_result + bookkeeping). Only infra failures
        (timeout / crash / stale-heartbeat, per ``classify_specialist_failure``)
        are retried, capped at :data:`SPECIALIST_AUTO_RETRY_MAX`; the failure
        reason is injected into the retry prompt. Disabled when
        ``INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY`` is set to ``0``."""
        flag = os.environ.get(
            "INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY", "1",
        ).strip().lower()
        if flag in ("0", "false", "no", "off"):
            return False
        try:
            cap = int(os.environ.get(
                "INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY_MAX",
                str(SPECIALIST_AUTO_RETRY_MAX),
            ))
        except (TypeError, ValueError):
            cap = SPECIALIST_AUTO_RETRY_MAX
        if cap <= 0:
            return False
        from .specialist_runner import classify_specialist_failure
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
        # bench-enabled specialists) so the retry contends for the same pools.
        lanes, ttl = self._registry_lanes_ttl("specialist")
        from .specialist_profile import resolve_specialist_profile
        if resolve_specialist_profile(retry_params).grants_bench_tool:
            lanes = list(dict.fromkeys((*lanes, "benchmark_lane")))

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
            "coordinator", "observation",
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
            "specialist auto-retry: task=%s failure=%s attempt=%d/%d "
            "re-enqueued as %s",
            task.task_id, ftype.value, next_attempt, cap, new_task.task_id,
        )
        return True

    # specialist pre-dispatch warmup
    async def _warm_specialist_params(self, params: dict[str, Any]) -> None:
        """Fill specialist task params with KnowledgePlane data before enqueue (mutates in place); all best-effort, missing fields stay empty."""
        state = self.shared_state
        plane = self.knowledge_plane

        from .specialist_domains import normalize_dispatch_tags
        from .specialist_profile import resolve_specialist_profile

        # Bench-enabled (mode=patch & bench=true) specialists run worktree
        # micro-benchmarks, so they must hold a GPU lease: default needs_gpu so
        # the dispatcher routes them through the gpu_specialist_pool quota +
        # TTL throttle (operator/LLM may still override explicitly).
        if resolve_specialist_profile(params).grants_bench_tool:
            params.setdefault("needs_gpu", True)

        domain = str(params.get("domain") or "").strip()
        # Knowledge-domain tags drive multi-anchor prompt assembly; a single ``domain`` is the legacy single-tag alias.
        tags = normalize_dispatch_tags(params)

        # PR feed (Gap-02 ↔ Gap-01 contract): fetch + merge the warm cache per domain when the plane is wired; failures fall back to empty.
        if plane is not None and "pr_feed" not in params:
            pr_domains = [domain] if domain else list(tags)
            merged_prs: list[dict[str, Any]] = []
            seen_pr_keys: set[str] = set()
            for pr_dom in pr_domains:
                try:
                    prs, _warnings = plane.pr_feed_warm(domain=pr_dom)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "specialist warmup: pr_feed_warm(domain=%r) failed: %r",
                        pr_dom, exc,
                    )
                    continue
                for p in prs:
                    pd = self._pr_summary_to_dict(p)
                    key = str(pd.get("url") or pd.get("title") or id(pd))
                    if key not in seen_pr_keys:
                        seen_pr_keys.add(key)
                        merged_prs.append(pd)
            params["pr_feed"] = merged_prs
        else:
            params.setdefault("pr_feed", [])

        if "pr_monitor_available" not in params:
            params["pr_monitor_available"] = bool(
                plane is not None and getattr(plane, "pr_monitor_enabled", True)
            )

        # Cortex v1 subgraphs removed; keep field defaulted for stable SpecialistPromptInputs.
        params.setdefault("kb_subgraph", {})

        # Warm-start recipe + pitfalls + lessons from T0 anchor.
        if state.warm_start_recipe and "warm_start_recipe" not in params:
            params["warm_start_recipe"] = dict(state.warm_start_recipe)
        if state.warm_start_pitfalls and "warm_start_pitfalls" not in params:
            params["warm_start_pitfalls"] = list(state.warm_start_pitfalls)
        if state.warm_start_lessons and "warm_start_lessons" not in params:
            params["warm_start_lessons"] = list(state.warm_start_lessons)
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

        # Local-source navigation hint — same source the kernel agent uses for source_file containment.
        if "framework_source_roots" not in params:
            try:
                from .framework_paths import resolve_source_file_allowlist
                roots = resolve_source_file_allowlist()
                if roots:
                    params["framework_source_roots"] = list(roots)
            except Exception as exc:  # noqa: BLE001
                log.debug(
                    "specialist warmup: framework_source_roots lookup "
                    "failed: %r", exc,
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
            from .shared_state import render_model_arch_compact

            _arch_notes = render_model_arch_compact(
                getattr(state, "model_arch", None)
            )
            if _arch_notes:
                params["arch_notes"] = _arch_notes

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
                from . import research_hints as _research_hints
                _hints_block = _research_hints.summarise_for_prompt(
                    self.session_dir,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: specialist research hints failed")
                _hints_block = ""
            if _hints_block:
                params["research_hints"] = _hints_block

        # Fill gap-specific anchors from the gaps[] ledger: stamp symptom/layer/domain_hint/attempts onto the task so the prompt has structured context.
        gap_cid = (
            str(params.get("gap_canonical_id") or "").strip()
            or str(params.get("gap") or "").strip()
        )
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
        if (
            isinstance(last_ta, dict)
            and last_ta.get("analysis_md_text")
            and "roofline_evidence" not in params
        ):
            from .roofline_snapshot import extract_workload_summary

            analysis_path = str(last_ta.get("analysis_md_path") or "")
            executive_summary: dict[str, Any] = {}
            if analysis_path:
                try:
                    executive_summary = extract_workload_summary(analysis_path)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "specialist warmup: extract_workload_summary(%s) "
                        "failed: %r", analysis_path, exc,
                    )
                    executive_summary = {}
            hot_kernels = list(last_ta.get("hot_kernels_top15") or [])[:8]
            params["roofline_evidence"] = {
                "analysis_md_path": analysis_path,
                "roofline_snapshot_id": last_ta.get("roofline_snapshot_id"),
                "executive_summary": executive_summary,
                "hot_kernels_top15": hot_kernels,
            }

        # (Legacy framework_pr_scout pre-fetch removed — PR discovery lives in the FRAMEWORK_PR phase pump.)

        # proposal_set cap into params so SpecialistRunner reads it; setdefault lets a delegate shrink it.
        from inference_optimizer.orchestrator.policy import (
            DEFAULT_SPECIALIST_MAX_PROPOSALS,
        )
        params.setdefault("max_proposals", DEFAULT_SPECIALIST_MAX_PROPOSALS)

    @staticmethod
    def _pr_summary_to_dict(pr: Any) -> dict[str, Any]:
        """Flatten a PRSummary into the dict SpecialistPromptBuilder expects."""
        return {
            "repo":   str(getattr(pr, "repo", "")),
            "number": int(getattr(pr, "number", 0) or 0),
            "title":  str(getattr(pr, "title", "")),
            "url":    str(getattr(pr, "url", "")),
            "state":  str(getattr(pr, "state", "")),
            "labels": list(getattr(pr, "labels", ()) or ()),
            "author": str(getattr(pr, "author", "")),
        }

    # gaps[] ledger refresh
    async def _refresh_gaps(self, *, reason: str) -> None:
        """Refresh :attr:`SharedState.gaps` from observable signals (Coordinator is sole writer, Inv-1). Additive upsert deduped by canonical_id; best-effort."""
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
            reason, len(state.gaps),
        )

    def _extract_gaps_from_baseline(self) -> list[dict[str, Any]]:
        """Derive initial gap rows from the baseline snapshot (throughput_below_target, baseline_unstable); reuse the M1 anchor canonical_id so traverse rows align."""
        state = self.shared_state
        gaps: list[dict[str, Any]] = []
        if state.baseline_tput <= 0:
            return gaps
        anchor = self._gap_anchor_canonical_id()
        target_gap = float(getattr(state, "target_gap_pct", 0.0) or 0.0)
        if target_gap > 0.0:
            severity = (
                "high" if target_gap >= 10.0
                else "medium" if target_gap >= 3.0
                else "low"
            )
            gaps.append({
                "canonical_id": f"{anchor}#throughput_below_target",
                "symptom": (
                    f"current_best is {target_gap:.1f}% short of "
                    f"--target-gain"
                ),
                "layer": "framework",
                "severity": severity,
                "domain_hint": "serving_specialist",
                "source": "baseline",
            })
        if state.baseline_failure_streak > 0:
            gaps.append({
                "canonical_id": f"{anchor}#baseline_unstable",
                "symptom": (
                    f"baseline crashed {state.baseline_failure_streak} "
                    f"consecutive time(s)"
                ),
                "layer": "system",
                "severity": (
                    "high" if state.baseline_failure_streak >= 2 else "medium"
                ),
                "domain_hint": "system_specialist",
                "source": "baseline",
            })
        return gaps

    def _extract_gaps_from_attempts(self) -> list[dict[str, Any]]:
        """Derive gaps from rolling failures + winners history (recurring (action, error_class) + explore plateau)."""
        state = self.shared_state
        anchor = self._gap_anchor_canonical_id()
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
            1 for w in winners_hist[-5:]
            if isinstance(w, dict) and float(w.get("gain_pct") or 0.0) > 0.0
        )
        if no_promote >= 3 and recent_promotions == 0:
            gaps.append({
                "canonical_id": f"{anchor}#explore_plateau",
                "symptom": (
                    f"{no_promote} consecutive grid rounds without a new "
                    f"current_best"
                ),
                "layer": "framework",
                "severity": "high" if no_promote >= 6 else "medium",
                "domain_hint": "serving_specialist",
                "source": "attempts",
            })
        return gaps

    @staticmethod
    def _gap_layer_for_action(action: str) -> tuple[str, str]:
        """Map an action name → (layer, domain_hint) for gap rows (fallback ("framework", "serving_specialist"))."""
        a = str(action or "").strip().lower()
        if a in {"kernel_opt", "integrate", "trace_analyze", "run_gemm_tuning", "run_optimization"}:
            return ("kernel", "kernel_switch_specialist")
        if a in {"profile", "roofline"}:
            return ("kernel", "kernel_switch_specialist")
        if a in {"sweep", "explore"}:
            return ("framework", "serving_specialist")
        if a in {"baseline"}:
            return ("system", "system_specialist")
        return ("framework", "serving_specialist")

    def _record_explore_round_gaps(
        self, *, task: "Task | None", result: dict[str, Any],
    ) -> None:
        """Append per-variant KEEP/REVERT outcomes to the matching gap (or the anchor gap as fallback)."""
        if task is None:
            return
        per_variant = result.get("per_variant_outcomes")
        if not isinstance(per_variant, list) or not per_variant:
            return
        params = dict(task.params or {})
        canonical = (
            str(params.get("gap_canonical_id") or "").strip()
            or self._gap_anchor_canonical_id()
        )
        state = self.shared_state
        existing = state.find_gap(canonical)
        if existing is None:
            state.upsert_gap({
                "canonical_id": canonical,
                "symptom": "explore round outcomes",
                "layer": "framework",
                "severity": "medium",
                "domain_hint": "serving_specialist",
                "source": "attempts",
            })
        for outcome in per_variant:
            if not isinstance(outcome, dict):
                continue
            state.append_gap_attempt(canonical, {
                "action": "explore",
                "variant_name": str(outcome.get("variant_name") or ""),
                "outcome": str(outcome.get("outcome") or "").upper(),
                "gain_pct": outcome.get("gain_pct"),
            })

    # specialist_done bookkeeping
    async def _handle_specialist_done(
        self, source: str, intent: Intent,
    ) -> None:
        """Handle a ``specialist_done`` intent (source ``specialist:<task_id>`` per Inv-5.3 / R3); bookkeeping in _record_specialist_result."""
        payload = dict(intent.payload or {})
        task_id = self._task_id_from_specialist_source(source)
        task: Task | None = None
        if task_id:
            try:
                task = await self.tasks.get(task_id)
            except Exception:  # noqa: BLE001 — TaskNotFound and friends
                task = None
        if task is None:
            # PolicyGate R3 should have caught this; log defensively but don't crash.
            log.warning(
                "specialist_done from source=%r references unknown "
                "task_id=%r; skipping bookkeeping (R3 should have "
                "denied; defense in depth)",
                source, task_id,
            )
            return
        await self._record_specialist_result(
            task=task, done_payload=payload, source=source,
        )

    @staticmethod
    def _task_id_from_specialist_source(source: str) -> str:
        """Extract the task_id from a ``specialist:<task_id>`` source ("" when prefix is absent)."""
        if not source:
            return ""
        if source.startswith(SPECIALIST_FROM_AGENT_PREFIX):
            return source[len(SPECIALIST_FROM_AGENT_PREFIX):]
        return ""

    # Multi-node only: cap on how many specialist proposal_set entries
    # are auto-materialised into a single explore grid per specialist
    # round. Keeps the deterministic bridge from flooding the action
    # queue when an LLM specialist returns a large proposal_set.
    _MN_AUTO_EXPLORE_GRID_CAP = 6

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
        """
        from .action_executors._multi_node_env import is_multi_node
        if not is_multi_node() or not proposals:
            return
        grid: list[dict[str, Any]] = []
        for i, p in enumerate(proposals[: self._MN_AUTO_EXPLORE_GRID_CAP]):
            if not isinstance(p, dict):
                continue
            args = str(
                p.get("extra_args") or p.get("extra_server_args") or ""
            ).strip()
            envs_raw = p.get("extra_envs")
            envs = (
                {str(k): str(v) for k, v in envs_raw.items()}
                if isinstance(envs_raw, dict) else {}
            )
            # Drop entries with neither a server-arg nor an env override —
            # nothing for the restart to apply (e.g. research-only items).
            if not args and not envs:
                continue
            name = str(p.get("name") or "").strip() or (
                f"{domain or 'specialist'}-{task.task_id[:8]}-{i}"
            )
            grid.append({
                "name": name,
                "extra_args": args,
                "extra_envs": envs,
                "provenance": f"specialist:{domain}" if domain else "specialist",
                "note": str(p.get("reason") or "")[:200],
            })
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
                etask.task_id, len(grid), task.task_id, domain, was_existing,
            )
        except Exception:  # noqa: BLE001 — defensive; never block bookkeeping
            log.exception(
                "mn_auto_materialize: failed to enqueue explore from "
                "specialist=%s domain=%s", task.task_id, domain,
            )

    async def _record_specialist_result(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
    ) -> None:
        """Common bookkeeping for any specialist task termination (dispatcher loop + intent routing); idempotent on round_id, failures logged not raised."""
        domain = str(done_payload.get("domain") or "").strip()
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        is_empty = bool(done_payload.get("empty")) or len(proposals) == 0

        round_entry = self._build_specialist_round_entry(
            task=task, done_payload=done_payload, source=source,
        )
        # Advisory multi-model scoring of the proposal_set; informational only, gates nothing. Defensive.
        _scorer = getattr(self, "_proposal_scorer", None)
        if _scorer is not None and proposals:
            try:
                scores = await _scorer.score(
                    gap={
                        "domain": domain,
                        "gap_canonical_id": done_payload.get(
                            "gap_canonical_id", ""
                        ),
                        "gap_symptom": (task.params or {}).get("gap_symptom"),
                        "gap_evidence": (task.params or {}).get("gap_evidence"),
                        "summary": done_payload.get("summary", ""),
                    },
                    proposals=proposals,
                )
                if scores and scores.get("models"):
                    round_entry["ensemble_scores"] = scores
            except Exception:  # noqa: BLE001 — advisory; never block
                log.exception(
                    "specialist bookkeeping: proposal scoring failed for "
                    "task=%s (continuing without scores)", task.task_id,
                )
        try:
            self.shared_state.record_specialist_round(round_entry)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: record_specialist_round failed for "
                "task=%s", task.task_id,
            )

        try:
            self.shared_state.bump_specialist_domain_empty_streak(
                domain, empty=is_empty,
            )
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: bump_specialist_domain_empty_streak "
                "failed for task=%s", task.task_id,
            )

        try:
            self.shared_state.update_last_specialist({
                "task_id": task.task_id,
                "domain": domain,
                "gap_canonical_id": str(
                    done_payload.get("gap_canonical_id") or ""
                ),
                "empty": is_empty,
                "proposals_total": len(proposals),
                "confidence": done_payload.get("confidence"),
                "summary": str(done_payload.get("summary") or "")[:480],
                "reason": str(done_payload.get("reason") or "")[:480],
                "ts": datetime.now(timezone.utc).isoformat(),
            })
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: update_last_specialist failed for "
                "task=%s", task.task_id,
            )

        # Persist so a resume picks up the bookkeeping without re-running the specialist.
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "specialist bookkeeping: SharedState.save failed for task=%s",
                task.task_id,
            )

        await self._record_observation(
            source or "coordinator", "observation",
            {
                "kind": "specialist_done_recorded",
                "task_id": task.task_id,
                "domain": domain,
                "gap_canonical_id": done_payload.get("gap_canonical_id", ""),
                "proposals_total": len(proposals),
                "empty": is_empty,
            },
        )

        # Multi-node only: auto-materialise the proposal_set into a
        # benchmarked explore task. No-op single-node (LLM drives explore
        # directly there) and no-op when the proposal_set is empty / has
        # no applicable variants. See :meth:`_maybe_materialize_mn_explore`.
        try:
            await self._maybe_materialize_mn_explore(
                task=task, domain=domain, proposals=proposals,
            )
        except Exception:  # noqa: BLE001 — defensive; never block bookkeeping
            log.exception(
                "mn_auto_materialize: bridge raised for task=%s (continuing)",
                task.task_id,
            )

        # route session_steward_specialist verdicts. Done payload
        # carries extra fields beyond the standard schema; see
        # ``actions/assess_remaining_gaps.md`` and the prompt builder
        # focus template. Coerce out-of-vocab recommendations to
        # ``stop_session`` (defense in depth — the LLM is allowed to
        # write any string but we only honour the closed enum).
        if domain == "session_steward_specialist":
            try:
                await self._route_steward_verdict(
                    task=task, done_payload=done_payload,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "steward routing failed for task=%s; assessment "
                    "left in last_remaining_gaps_assessment but no "
                    "phase-routing change applied",
                    task.task_id,
                )

        # Aggregate any research evidence (PR / diff / NVIDIA refs)
        # reported by this specialist into the exploration-depth tracker.
        # Applies to every domain that self-reports a ``research`` block
        # (pr_intel + research_scout), de-duped across the session.
        try:
            self._aggregate_research_evidence(done_payload)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "depth: research-evidence aggregation failed for task=%s",
                task.task_id,
            )

        # Harvest research-scout output (hints, competitor target, gap seeds, PR dedup). Fail-soft.
        if domain == "research_scout_specialist":
            try:
                self._harvest_research_scout(done_payload)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "research-scout harvest failed for task=%s", task.task_id,
                )

        # Refresh the gaps ledger after a specialist round closes; record the verdict as a gap attempt.
        gap_cid = str(done_payload.get("gap_canonical_id") or "").strip()
        if gap_cid:
            try:
                self.shared_state.append_gap_attempt(gap_cid, {
                    "action": "specialist",
                    "variant_name": domain,
                    "outcome": "EMPTY" if is_empty else "PROPOSALS",
                    "proposals_total": len(proposals),
                })
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "specialist bookkeeping: append_gap_attempt failed for "
                    "gap=%s", gap_cid,
                )
        try:
            await self._refresh_gaps(reason="specialist_done")
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "specialist bookkeeping: _refresh_gaps failed for task=%s",
                task.task_id,
            )
        # B3: push specialist-authored patches to the Critic so integrate_patch can pass.
        try:
            await self._maybe_autosubmit_specialist_patches(
                task=task, done_payload=done_payload,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "B3: specialist patch autosubmit failed for task=%s",
                task.task_id,
            )

    def _plateau_advisory_block(self) -> str:
        """Render the plateau-judgment advisory block (EXPLORE/KERNEL/FRAMEWORK_PR; advisory, never gates). Returns "" when no plateau signal is active."""
        state = self.shared_state
        phase = (getattr(state, "phase", "") or "").strip().upper()
        overrides = getattr(state, "plateau_overrides", None) or {}
        if not isinstance(overrides, dict):
            overrides = {}
        lines: list[str] = []
        if phase == _phase_state.PHASE_EXPLORE:
            triggered, evidence = _phase_state.compute_plateau_explore(
                state,
                lookback=int(overrides.get(
                    "explore_lookback",
                    _phase_state.DEFAULT_PLATEAU_EXPLORE_LOOKBACK,
                )),
                keep_gain_threshold_pct=float(overrides.get(
                    "explore_keep_gain_pct",
                    _phase_state.DEFAULT_PLATEAU_EXPLORE_KEEP_GAIN_PCT,
                )),
                empty_streak_threshold=int(overrides.get(
                    "explore_empty_streak",
                    _phase_state.DEFAULT_PLATEAU_EXPLORE_EMPTY_STREAK,
                )),
            )
            if triggered:
                lines.append(
                    "EXPLORE plateau detected: low recent KEEP gain plus "
                    "specialist empty streak."
                )
                lines.append(
                    "  recent_keep_gain_pct="
                    f"{evidence.get('recent_keep_gain_pct', 0.0)} "
                    f"threshold={evidence.get('keep_gain_threshold_pct', 0.0)} "
                    f"empty_streak={evidence.get('empty_streak', 0)} "
                    f"streak_threshold={evidence.get('empty_streak_threshold', 0)}"
                )
        elif phase == _phase_state.PHASE_KERNEL:
            triggered, evidence = _phase_state.compute_plateau_kernel(
                state,
                lookback=int(overrides.get(
                    "kernel_lookback",
                    _phase_state.DEFAULT_PLATEAU_KERNEL_LOOKBACK,
                )),
                revert_streak_threshold=int(overrides.get(
                    "kernel_revert_streak",
                    _phase_state.DEFAULT_PLATEAU_KERNEL_REVERT_STREAK,
                )),
                keep_gain_threshold_pct=float(overrides.get(
                    "kernel_keep_gain_pct",
                    _phase_state.DEFAULT_PLATEAU_KERNEL_KEEP_GAIN_PCT,
                )),
            )
            if triggered:
                lines.append(
                    "KERNEL plateau detected: REVERT streak or low recent "
                    "KEEP gain."
                )
                lines.append(
                    "  revert_streak="
                    f"{evidence.get('revert_streak', 0)} "
                    f"threshold={evidence.get('revert_streak_threshold', 0)} "
                    f"recent_keep_gain_pct={evidence.get('recent_keep_gain_pct', 0.0)} "
                    f"keep_gain_threshold_pct={evidence.get('keep_gain_threshold_pct', 0.0)}"
                )
        elif phase == _phase_state.PHASE_FRAMEWORK_PR:
            triggered, evidence = _phase_state.compute_plateau_framework_pr(
                state,
                lookback=int(overrides.get(
                    "framework_pr_lookback",
                    _phase_state.DEFAULT_FRAMEWORK_PR_PLATEAU_LOOKBACK,
                )),
                keep_gain_threshold_pct=float(overrides.get(
                    "framework_pr_keep_gain_pct",
                    _phase_state.DEFAULT_FRAMEWORK_PR_PLATEAU_KEEP_GAIN_PCT,
                )),
            )
            if triggered:
                lines.append(
                    "FRAMEWORK_PR plateau detected: recent batches all below "
                    "keep-gain threshold."
                )
                lines.append(
                    "  lookback="
                    f"{evidence.get('lookback', 0)} "
                    f"keep_gain_pct_threshold={evidence.get('keep_gain_pct_threshold', 0.0)} "
                    f"batch_max_gains={evidence.get('batch_max_gains', [])}"
                )
        if not lines:
            return ""
        lines.append(
            "Phase advance is driven only by hard limits (IR-6 force-exit, "
            "phase budget, terminal stop_reason) or explicit "
            "escalate_strategy_change hints; this block is informational."
        )
        return "\n".join(lines)

    def _target_gap_advisory_block(self) -> str:
        """Build the advisory "External target gap" prompt block (current-best vs competitor target; never gates)."""
        state = self.shared_state
        if not bool(getattr(state, "target_advisory_enabled", True)):
            return ""
        from . import research_hints as _research_hints

        target = _research_hints.load_competitor_target(self.session_dir)
        if not target:
            return ""
        best = getattr(state, "current_best", None)
        if not isinstance(best, dict):
            return ""
        tput = best.get("tput")
        tpot = best.get("tpot_mean_ms")
        tp = int(getattr(state, "tp", 0) or 0)
        our_tput_per_gpu = (
            float(tput) / tp
            if isinstance(tput, (int, float)) and tput > 0 and tp > 0
            else None
        )
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
        """Resolve the dominant external gap direction ('latency'/'throughput') from the competitor target, or None when advisory is off / no target. Fail-soft."""
        state = self.shared_state
        if not bool(getattr(state, "target_advisory_enabled", True)):
            return None
        try:
            from . import research_hints as _research_hints

            target = _research_hints.load_competitor_target(self.session_dir)
            if not target:
                return None
            best = getattr(state, "current_best", None)
            if not isinstance(best, dict):
                return None
            tput = best.get("tput")
            tpot = best.get("tpot_mean_ms")
            tp = int(getattr(state, "tp", 0) or 0)
            our_tput_per_gpu = (
                float(tput) / tp
                if isinstance(tput, (int, float)) and tput > 0 and tp > 0
                else None
            )
            our_tpot_ms = (
                float(tpot)
                if isinstance(tpot, (int, float)) and tpot > 0 else None
            )
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
        self, *, max_rounds: int = 2,
    ) -> list[dict[str, Any]]:
        """Collect proposal_set rows from the most recent specialist rounds (deduped by name; fail-soft)."""
        rounds = [
            r for r in (getattr(self.shared_state, "specialist_rounds", []) or [])
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

    def _priors_match_advisory_block(self) -> str:
        """Flag recently proposed variants aligning with proven priors / dominant external gap (advisory ordering, fail-soft)."""
        try:
            from . import research_hints as _research_hints

            variants = self._recent_proposed_variants()
            if not variants:
                return ""
            hints = _research_hints.load_hints(self.session_dir)
            primary_gap = self._current_primary_gap()
            return _research_hints.priors_match_summary(
                variants, hints, primary_gap=primary_gap,
            )
        except Exception:  # noqa: BLE001 — defensive
            return ""

    async def _maybe_autosubmit_specialist_patches(
        self, *, task: "Task", done_payload: dict[str, Any],
    ) -> None:
        """B3: auto-surface a specialist's source patches to the Critic via a synthetic integrate_patch proposal; idempotent per specialist."""
        patches = done_payload.get("patches_written") or []
        if not isinstance(patches, list) or not patches:
            return
        sid = str(task.task_id or "").strip()
        if not sid:
            return
        # B4 guard: resolve patches_written against worktree + workspace; submit only when ≥1 real file exists.
        from ..session_paths import runs_dir as _runs_dir
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
        if not existing_patches:
            await self._record_observation(
                "coordinator", "observation",
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
        propose_payload = {
            "action_name": "integrate_patch",
            "provenance": "specialist",
            "predicted_gain_pct": 0.0,
            "params": {
                "specialist_task_id": sid,
                "provenance": "specialist",
                "patch_name": patch_name,
            },
        }
        msg = Message.new(
            "coordinator", "*", "proposal",
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
            "coordinator", "observation",
            {
                "kind": "specialist_patch_autosubmitted_for_review",
                "specialist_task_id": sid,
                "proposal_msg_id": msg.msg_id,
                "patch_name": patch_name,
                "patches": [str(x) for x in patches][:8],
            },
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "B3: save after specialist patch autosubmit failed for "
                "task=%s", sid,
            )

    def _build_specialist_round_entry(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        """Translate a specialist done payload into a SharedState.specialist_rounds[] row; round_id defaults to task_id for idempotent overwrite (M5)."""
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        round_id = str(
            (task.params or {}).get("round_id")
            or task.task_id
        )
        truncated_from = done_payload.get("proposals_truncated_from")
        from .specialist_domains import normalize_dispatch_tags

        # Knowledge-domain tags for breakdown attribution; reported tags win over dispatch params.
        tags = normalize_dispatch_tags(done_payload)
        if not tags:
            tags = normalize_dispatch_tags(task.params or {})
        entry: dict[str, Any] = {
            "round_id":          round_id,
            "task_id":           task.task_id,
            "source":            source or "coordinator",
            "completed_at":      datetime.now(timezone.utc).isoformat(),
            "domain":            str(done_payload.get("domain") or ""),
            "tags":              list(tags),
            "gap_canonical_id":  str(done_payload.get("gap_canonical_id") or ""),
            "empty":             bool(done_payload.get("empty"))
                                  or len(proposals) == 0,
            "proposals_total":   len(proposals),
            "proposal_set":      list(proposals),
            "summary":           str(done_payload.get("summary") or "")[:480],
            "reason":            str(done_payload.get("reason") or "")[:480],
            "confidence":        done_payload.get("confidence"),
            "new_findings":      list(done_payload.get("new_findings") or []),
            "residual_questions": list(
                done_payload.get("residual_questions") or []
            ),
        }
        gpu_ids = done_payload.get("allocated_gpu_ids") or []
        if isinstance(gpu_ids, list) and gpu_ids:
            entry["allocated_gpu_ids"] = [
                int(g) for g in gpu_ids
                if isinstance(g, (int, str)) and str(g).strip().lstrip("-").isdigit()
            ]
        if isinstance(truncated_from, int) and truncated_from > len(proposals):
            entry["proposals_truncated_from"] = truncated_from
        return entry

    # REQUEST / RESPONSE (Plan A)
    # ------------------------------------------------------------------
    def _emit_lifecycle(
        self,
        *,
        step: str,
        status: str,
        artifacts: dict[str, str] | None = None,
        detail: str = "",
        duration_s: float | None = None,
    ) -> None:
        """Record + persist one operator-facing lifecycle event (#266).

        Best-effort by design: operator-facing logging must never break the
        orchestration loop, so any failure is swallowed at debug level.
        """
        try:
            self.shared_state.record_lifecycle_event(
                step=step,
                status=status,
                artifacts=artifacts,
                detail=detail,
                duration_s=duration_s,
            )
            # Terminal events (END/ERROR) carry the produced artifact paths an
            # operator is waiting on — always flush them. Non-terminal markers
            # (START / phase ENTER) are debounced: skip the write if we flushed
            # within the last ``_lifecycle_save_min_interval_s`` seconds, since
            # the next terminal event (or a later marker past the window) will
            # persist the coalesced tail anyway.
            terminal = status in ("END", "ERROR")
            now = time.monotonic()
            if terminal or (
                now - self._lifecycle_last_save
                >= self._lifecycle_save_min_interval_s
            ):
                self.shared_state.save(self.session_dir)
                self._lifecycle_last_save = now
        except Exception:  # noqa: BLE001 — defensive
            log.debug(
                "Coordinator: lifecycle emit failed (step=%s status=%s)",
                step, status, exc_info=True,
            )

    async def _handle_request(self, source: str, intent: Intent) -> None:
        """Route a REQUEST intent to its target agent (Plan A: → kernel).

        Applies the kernel-request execution-order gate, records the request on
        the bus for the target reactor / replay, and auto-rejects requests whose
        target agent is not in the role registry (e.g. ``--no-kernel``) so the
        requester never hangs.

        Args:
            source (str): The agent issuing the request.
            intent (Intent): The REQUEST intent; ``payload`` carries
                ``target_agent`` and ``kind``.
        """
        target_agent = intent.payload["target_agent"]
        kind = intent.payload["kind"]
        denied = self._sequence_denial_for_request(target_agent, kind)
        if denied is not None:
            await self._record_policy_denied(source, intent, denied)
            return
        # Always record the request on the bus so the kernel reactor (and tests/replay) can see it.
        request_msg = Message.new(
            source, target_agent, "request", dict(intent.payload), priority=1,
        )
        await self.bus.append_and_seq(request_msg)

        # Safety net: auto-reject when the target agent was removed (e.g. --no-kernel) so Orch doesn't hang.
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

        # Programmatic shortcut: run a registered kernel handler inline + emit RESPONSE so a deterministic shell-tool invocation doesn't burn an LLM turn (see kernel_request_handlers.py).
        if target_agent == "kernel":
            handler = get_handler(kind)
            if handler is not None:
                params = intent.payload.get("params") or {}
                merged_payload = {**intent.payload, **params}
                # Force batch dispatch for run_optimization: inject candidates_path from last_trace_analyze (else collapses to single-kernel run). LLM value wins.
                if (
                    kind == "run_optimization"
                    and self.shared_state.last_trace_analyze
                    and not merged_payload.get("candidates_path")
                ):
                    cached_candidates_path = self.shared_state.last_trace_analyze.get(
                        "candidates_path"
                    )
                    if cached_candidates_path:
                        merged_payload["candidates_path"] = cached_candidates_path
                # Commit 1cd9f7d's roofline_json auto-inject is omitted here: Roofline-v2 caches under last_trace_analyze instead.
                cache_hit_source = None
                cached_result = self._cached_kernel_request(kind, merged_payload)
                if cached_result is not None:
                    result = cached_result
                    cache_hit_source = "shared_state_cache"
                    # #266: a cache hit produces a response but never runs the
                    # handler, so emit a single END (no paired START). Without
                    # this the lifecycle log would show no record at all for a
                    # cache-served step, leaving an operator unsure whether it
                    # ran. detail=cache_hit marks it as served-from-cache.
                    self._emit_lifecycle(
                        step=kind,
                        status="END",
                        artifacts=_lifecycle_paths(result),
                        detail="cache_hit",
                    )
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
                            "extra_server_args": rejected.get("extra_server_args", ""),
                            "attempt_count": rejected.get("attempt_count"),
                            "best_gain_pct": rejected.get("best_gain_pct"),
                            "reason": rejected.get("reason"),
                        }
                        cache_hit_source = "shared_state_kernel_rejection"
                        # #266: a short-circuited integrate (patch already
                        # exhausted) also never runs the handler; emit a lone
                        # END so the log records the step was resolved as a
                        # rejection rather than silently missing.
                        self._emit_lifecycle(
                            step=kind,
                            status="END",
                            artifacts=_lifecycle_paths(result),
                            detail="rejected",
                        )
                    else:
                        # Inject base_tput from current_best.tput when an integrate request omits it (else 2nd/3rd multi-KEEP integrate fails base_tput > 0); operator value wins.
                        if (
                            kind == "integrate"
                            and not merged_payload.get("base_tput")
                        ):
                            cb_tput = (
                                self.shared_state.current_best or {}
                            ).get("tput")
                            if isinstance(cb_tput, (int, float)) and cb_tput > 0:
                                merged_payload["base_tput"] = float(cb_tput)

                        # Streaming-record callback for run_optimization batch: each sub-attempt writes immediately (else a slow sibling starves a fast KEEP's integrate).
                        handler_kwargs: dict[str, Any] = {
                            "session_dir": self.session_dir,
                        }
                        if kind == "run_optimization":
                            handler_kwargs["record_partial"] = (
                                self._record_kernel_opt_partial
                            )
                        # #266: bracket the programmatic kernel step with
                        # START / END lifecycle events so operators see the
                        # step ran, how long it took, and where its outputs
                        # landed. ``kind`` is the machine step name
                        # (trace_analyze / run_optimization / integrate /
                        # run_gemm_tuning); the human label is resolved by
                        # SharedState from LIFECYCLE_STEP_LABELS.
                        _lc_t0 = time.monotonic()
                        self._emit_lifecycle(
                            step=kind,
                            status="START",
                            artifacts=_lifecycle_paths(merged_payload),
                        )
                        try:
                            result = await handler(
                                merged_payload,
                                **handler_kwargs,
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
                        _lc_status = (
                            "ERROR"
                            if str(result.get("status", "")).lower()
                            in ("failed", "error")
                            else "END"
                        )
                        _lc_detail = " ".join(
                            str(p) for p in (
                                result.get("decision"),
                                result.get("status"),
                                f"kernel={result.get('kernel_id')}"
                                if result.get("kernel_id") else "",
                            ) if p
                        )
                        self._emit_lifecycle(
                            step=kind,
                            status=_lc_status,
                            artifacts=_lifecycle_paths(result),
                            detail=_lc_detail,
                            duration_s=time.monotonic() - _lc_t0,
                        )
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
                # Cache trace_analyze output (successful runs only) to short-circuit identical next-tick requests.
                if (
                    kind == "trace_analyze"
                    and cache_hit_source is None
                    and result.get("status") in ("ok", "succeeded")
                ):
                    self.shared_state.record_trace_analyze(merged_payload, result)
                    self.shared_state.save(self.session_dir)
                # Mirror kernel-opt outcomes into SharedState so Orch sees decision/speedup next tick.
                if kind == "run_optimization":
                    # Batch mode already streamed each sub-result; re-recording would double-count. Cache hits lack batch_mode.
                    if not bool(
                        isinstance(result, dict) and result.get("batch_mode")
                    ):
                        self.shared_state.record_kernel_opt(result)
                    self.shared_state.save(self.session_dir)
                    # Auto-enqueue integrate for KEEP'd kernels that haven't
                    # been integrated yet (IR-3: integration is mandatory).
                    await self._auto_enqueue_pending_integrations()
                if kind == "run_gemm_tuning":
                    self.shared_state.record_gemm_tuning(result)
                    self.shared_state.save(self.session_dir)
                if kind == "integrate":
                    if result.get("status") != "skipped":
                        self.shared_state.record_kernel_integrate_result(result)
                    decision = str(result.get("decision", "")).upper()
                    if decision == "KEEP":
                        if isinstance(result, dict) and not result.get(
                            "gap_canonical_id"
                        ):
                            payload_gap = str(
                                merged_payload.get("gap_canonical_id") or ""
                            ).strip()
                            if payload_gap:
                                result["gap_canonical_id"] = payload_gap
                        await self._record_integrate_keep(result)
                    self.shared_state.save(self.session_dir)
                # Bug B: advance the kernel cursor past this request seq so the LLM kernel agent doesn't re-answer it next tick.
                await self.cursors.advance(
                    target_agent,
                    seq=request_msg.seq,
                    msg_id=request_msg.msg_id,
                )

    def _cached_kernel_request(self, kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Return a cached programmatic_handler result if applicable (cache key last_trace_analyze)."""
        if kind != "trace_analyze":
            return None
        cached = self.shared_state.last_trace_analyze or {}
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
            "note": "served from shared_state.last_trace_analyze cache",
        }

    async def _handle_response(self, source: str, intent: Intent) -> None:
        """Route a RESPONSE intent back to the original requester.

        Looks up the request message referenced by ``in_reply_to`` to address
        the response, then publishes it on the bus.

        Args:
            source (str): The agent emitting the response.
            intent (Intent): The RESPONSE intent; ``payload`` carries
                ``in_reply_to``.
        """
        in_reply_to = intent.payload["in_reply_to"]
        # Locate the original requester so we can address the response.
        original = await self.bus.lookup_by_id(in_reply_to)
        target = original.from_agent if original else "*"
        await self.bus.append_and_seq(Message.new(
            source, target, "response",
            dict(intent.payload), in_reply_to=in_reply_to, priority=1,
        ))

    # Robustness scheduling-police
    async def _handle_kill_task(self, source: str, intent: Intent) -> None:
        """Cancel a queued/running task in response to a kill_task intent.

        Records an observation for unknown task ids, transitions a
        queued/running task to ``cancelled``, and broadcasts a ``kill`` event.

        Args:
            source (str): The agent (typically robustness) issuing the kill.
            intent (Intent): The KILL_TASK intent; ``payload`` carries
                ``task_id`` and optional ``reason``.
        """
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
        """Prune an action family and cancel its in-flight tasks.

        Adds the family to the persistent pruned set, cancels any tasks in that
        family, and broadcasts a ``prune_branch`` event.

        Args:
            source (str): The agent issuing the prune.
            intent (Intent): The PRUNE_BRANCH intent; ``payload`` carries
                ``family`` and optional ``reason``.
        """
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
        """Handle a ``force_dispatch`` intent by emitting an event.

        Currently a P0-3 stub: it broadcasts a ``force_dispatch`` event;
        real dispatcher reordering arrives in P0-5 with the priority queue.

        Args:
            source: Identifier of the intent's originating agent.
            intent: The ``force_dispatch`` intent carrying ``task_id``.
        """
        # P0-3 stub: emit an event; real dispatcher reordering lands in P0-5 with the priority queue.
        await self.bus.append_and_seq(Message.new(
            source, "*", "event",
            {"kind": "force_dispatch", "task_id": intent.payload["task_id"],
             "reason": intent.payload.get("reason")},
        ))

    async def _handle_escalate_strategy_change(self, source: str, intent: Intent) -> None:
        """Process ``escalate_strategy_change`` (KB_design §3.8 §7.3 + §3.13 M7 §5.3); broadcasts strategy_change, acts on closed-vocab hints, drops unknown (Inv-8.2)."""
        payload = dict(intent.payload or {})
        # Always emit the broadcast first (back-compat with legacy contract tests).
        await self.bus.append_and_seq(Message.new(
            source, "*", "strategy_change",
            payload, priority=0,
        ))
        from .phase_state import (
            ESCALATE_HINT_EXTEND_EXPLORE_BUDGET,
            ESCALATE_HINT_EXTEND_KERNEL_BUDGET,
            ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX,
            PHASE_EXPLORE,
            PHASE_KERNEL,
            apply_escalate_budget_bump,
            is_pause_specialist_hint,
            is_valid_escalate_hint,
        )
        hint = str(payload.get("next_action_hint") or "").strip()
        if not hint or not is_valid_escalate_hint(hint):
            return
        # extend_*_budget mutates phase_budget_pct directly (consulted every tick).
        now_ts = datetime.now(timezone.utc).isoformat()
        if hint == ESCALATE_HINT_EXTEND_EXPLORE_BUDGET:
            self.shared_state.phase_budget_pct = apply_escalate_budget_bump(
                self.shared_state.phase_budget_pct, phase=PHASE_EXPLORE,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        if hint == ESCALATE_HINT_EXTEND_KERNEL_BUDGET:
            self.shared_state.phase_budget_pct = apply_escalate_budget_bump(
                self.shared_state.phase_budget_pct, phase=PHASE_KERNEL,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        # pause_specialist_<domain>: bump the per-domain empty-streak so the next EXPLORE round skips it.
        if is_pause_specialist_hint(hint):
            domain = hint[len(ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX):]
            self.shared_state.bump_specialist_domain_empty_streak(
                domain, empty=True,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        # skip_to_kernel / skip_to_close are deferred; next compute_next_phase picks them up.
        self.shared_state.set_pending_escalate_hint(hint)
        self.shared_state.save(self.session_dir)

    # SEND_MESSAGE / ALERT / UPDATE_STATE — minimal persistence
    async def _handle_send_message(self, source: str, intent: Intent) -> None:
        """Publish a free-form message onto the bus.

        Soft-degrades an unknown topic to ``observation`` per DESIGN §13.2 and
        routes to the requested recipient (defaulting to broadcast).

        Args:
            source (str): The sending agent.
            intent (Intent): The SEND_MESSAGE intent; ``payload`` may carry
                ``topic`` / ``to`` plus arbitrary message fields.
        """
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
        """Broadcast an alert message, prioritized by severity.

        High-severity alerts are published at priority 0; everything else at
        priority 1.

        Args:
            source (str): The alerting agent.
            intent (Intent): The ALERT intent; ``payload`` may carry
                ``severity`` plus alert detail.
        """
        prio = 0 if intent.payload.get("severity") == "high" else 1
        await self.bus.append_and_seq(Message.new(
            source, "*", "alert", dict(intent.payload), priority=prio,
        ))

    async def _handle_update_state(self, source: str, intent: Intent) -> None:
        """Apply agent-requested SharedState changes and report the result.

        Applies the requested changes (core fields disallowed), persists when
        anything changed, and broadcasts an observation listing the applied vs
        rejected keys.

        Args:
            source (str): The agent requesting the state update.
            intent (Intent): The UPDATE_STATE intent; ``payload`` carries a
                ``changes`` dict.
        """
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

    # Bookkeeping
    async def _record_policy_denied(
        self,
        source: str,
        intent: Intent,
        denied: PolicyDenied,
        *,
        action_name: str | None = None,
    ) -> None:
        """Record a PolicyGate denial and apply escalation side effects.

        Publishes a ``policy_denied`` observation, records the denial streak,
        auto-prunes the action family at streak >= 5, and sets the
        ``policy_loop`` stop reason at streak >= 10.

        Args:
            source (str): The agent whose intent was denied.
            intent (Intent): The denied intent.
            denied (PolicyDenied): The denial carrying rule / hint / reason.
            action_name (str | None): Explicit action name override; falls back
                to ``intent.payload['action_name']``.
        """
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
        resolved_action = action_name or str(
            (intent.payload or {}).get("action_name") or ""
        )
        # Streak counter is a fact for LLM self-correction only; system no longer auto-prunes or stops on it (long-run continuity over loop stop-loss).
        self.shared_state.record_policy_denial(
            action_name=resolved_action,
            rule=str(denied.rule or ""),
            hint=str(denied.hint or ""),
            intent_type=intent.type.value,
            tick=int(self.shared_state.tick or 0),
            intent_payload=intent.payload,
        )

    async def _record_observation(self, source: str, topic: str, payload: dict) -> None:
        """Append a broadcast observation message to the bus.

        Args:
            source (str): The agent recording the observation.
            topic (str): The bus topic to publish under.
            payload (dict): The observation payload.
        """
        await self.bus.append_and_seq(Message.new(source, "*", topic, payload))

    async def _cursor_advance_to_latest(self, agent_name: str) -> None:
        """Advance an agent's read cursor to the latest message addressed to it.

        Args:
            agent_name (str): The agent whose inbox cursor to advance.
        """
        latest = await self.bus.tail(n=1, to_agent=agent_name)
        if latest:
            top = latest[0]
            await self.cursors.advance(agent_name, seq=top.seq, msg_id=top.msg_id)

    async def _auto_enqueue_pending_integrations(self) -> None:
        """Auto-dispatch integrate for KEEP'd kernels not yet integrated (IR-3).

        After kernel_opt completes, any kernel with decision=KEEP that has no
        entry in kernel_integrate_attempts is immediately queued as an integrate
        REQUEST on the kernel agent's bus. This prevents the LLM from proposing
        explore/specialist instead of integrate after a successful kernel_opt.
        """
        state = self.shared_state
        opt_attempts = getattr(state, "kernel_opt_attempts", None) or {}
        integ_attempts = getattr(state, "kernel_integrate_attempts", None) or {}
        if not isinstance(opt_attempts, dict):
            return

        integrated_kids: set[str] = set()
        if isinstance(integ_attempts, dict):
            for entry in integ_attempts.values():
                if isinstance(entry, dict):
                    kid = str(entry.get("kernel_id") or "")
                    if kid:
                        integrated_kids.add(kid)

        # Track dispatched auto-integrates across ticks via instance set.
        if not hasattr(self, "_auto_integrate_dispatched"):
            self._auto_integrate_dispatched: set[str] = set()

        pending_kids: list[str] = []
        for kid, data in opt_attempts.items():
            if not isinstance(data, dict):
                continue
            decision = str(data.get("last_decision") or "").upper()
            if (
                decision == "KEEP"
                and kid not in integrated_kids
                and kid not in self._auto_integrate_dispatched
            ):
                pending_kids.append(kid)

        if not pending_kids:
            return

        for kid in pending_kids:
            log.info(
                "auto-integrate: dispatching integrate for KEEP'd kernel %s "
                "(IR-3 mandatory integration)",
                kid,
            )
            await self.bus.append_and_seq(Message.new(
                "orchestration", "kernel", "request",
                {
                    "kind": "integrate",
                    "kernel_id": kid,
                    "source": "auto_integrate_after_kernel_opt",
                },
                priority=2,
            ))
            self._auto_integrate_dispatched.add(kid)

    def _record_kernel_opt_partial(self, result: dict[str, Any]) -> None:
        """Streaming callback for ``_run_optimization_batch`` sub-attempts: write each per-kernel entry to kernel_opt_attempts immediately so the next-tick prompt is accurate mid-batch."""
        try:
            self.shared_state.record_kernel_opt(result)
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            # Never let a per-sub-attempt hiccup poison the gather; the post-gather record_kernel_opt picks it up.
            log.exception(
                "_record_kernel_opt_partial failed for kernel_id=%s",
                (result or {}).get("kernel_id") if isinstance(result, dict) else None,
            )

    async def _record_integrate_keep(self, result: dict[str, Any]) -> None:
        """Promote a kernel integrate KEEP into the optimization stack.

        Appends a deduped ``integrate`` entry to the optimization stack, mirrors
        the gain into the per-entry gain ledger, updates ``current_best`` and
        ``cumulative_gain`` / ``cumulative_gain_validated``, and fires a
        watermark roofline when the gain crosses the threshold. No-op when the
        result lacks a positive ``new_tput``.

        Args:
            result (dict[str, Any]): The integrate-patch executor result.
        """
        new_tput = result.get("new_tput")
        if not isinstance(new_tput, (int, float)) or new_tput <= 0:
            return
        if not self.shared_state.optimization_stack:
            self.shared_state.seed_stack_from_current_best()

        cb = self.shared_state.current_best or {}
        # Read result via the compat helper (handles legacy extra_sglang_args); cb is migrated at load time.
        extra_args = (
            read_extra_server_args(result)
            or (
                str(cb.get("extra_server_args") or "")
                if isinstance(cb, dict) else ""
            )
        ).strip()
        apply_result = result.get("apply_result") or {}
        backup_manifest = (
            apply_result.get("manifest_path")
            if isinstance(apply_result, dict) else None
        )
        if not backup_manifest and isinstance(apply_result, dict):
            stack_applies = apply_result.get("stack_apply_results")
            if isinstance(stack_applies, list):
                for applied in stack_applies:
                    if isinstance(applied, dict) and applied.get("manifest_path"):
                        backup_manifest = applied.get("manifest_path")
                        break
        entry = {
            "action": "integrate",
            "kernel_id": result.get("kernel_id"),
            "patch_path": result.get("patch_path"),
            "target_file": result.get("target_file"),
            "backup_manifest": backup_manifest,
            "gain_pct": result.get("gain_pct"),
            "tput": float(new_tput),
            "workspace": result.get("workspace"),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        stack_kernel_ids = result.get("stack_kernel_ids")
        if isinstance(stack_kernel_ids, list) and stack_kernel_ids:
            entry["stack_kernel_ids"] = [
                str(kid) for kid in stack_kernel_ids if str(kid)
            ]
        integrate_gap_cid = str(result.get("gap_canonical_id") or "").strip()
        if integrate_gap_cid:
            entry["gap_canonical_id"] = integrate_gap_cid
        key = (entry["kernel_id"], entry["patch_path"], entry["target_file"])
        existing = {
            (item.get("kernel_id"), item.get("patch_path"), item.get("target_file"))
            for item in self.shared_state.optimization_stack
            if isinstance(item, dict) and item.get("action") == "integrate"
        }
        if key not in existing:
            self.shared_state.optimization_stack.append(entry)
            # Mirror into gain_per_stack_entry so breakdown attribution works without re-walking the event log.
            self.shared_state.append_stack_gain_entry(
                action="integrate",
                variant_name=entry.get("kernel_id"),
                new_tput=new_tput,
                extra_server_args=extra_args,
                ts=entry["ts"],
            )

        self.shared_state.current_best = {
            "action": "integrate",
            "tput": float(new_tput),
            "kernel_id": result.get("kernel_id"),
            "extra_server_args": extra_args,
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": result.get("ttft_mean_ms"),
            "e2el_mean_ms": result.get("e2el_mean_ms"),
            "tpot_mean_ms": result.get("tpot_mean_ms"),
            "workspace": result.get("workspace"),
        }
        if self.shared_state.baseline_tput > 0:
            self.shared_state.cumulative_gain = (
                (float(new_tput) - self.shared_state.baseline_tput)
                / self.shared_state.baseline_tput * 100.0
            )
            # Integrate KEEP is already rebench-validated: promote into cumulative_gain_validated + watermark.
            validated_gain = (
                (float(new_tput) - self.shared_state.baseline_tput)
                / self.shared_state.baseline_tput * 100.0
            )
            self.shared_state.cumulative_gain_validated = float(validated_gain)
            self.shared_state.cumulative_gain_validated_ts = (
                datetime.now(timezone.utc).isoformat()
            )
            self.shared_state.cumulative_gain_validated_stack_len = len(
                self.shared_state.optimization_stack
            )
            await self._maybe_enqueue_watermark_roofline(
                reason="integrate_keep_watermark",
            )

    # Dispatcher (pulls queued tasks → SubAgentRunner)
    def _is_promotable_result(self, task_kind: str, result: dict[str, Any]) -> bool:
        """Decide whether a settled task result should be promoted.

        Per-kind rules: baseline/profile require a valid measurement, sweep
        requires ``status == "succeeded"``, ``replay_warm_recipe`` always routes
        through promotion (it owns its own failure bookkeeping), and everything
        else is promotable unless ``status == "failed"``.

        Args:
            task_kind (str): The task's kind.
            result (dict[str, Any]): The task result payload.

        Returns:
            bool: ``True`` when the result should go through
                :meth:`_promote_to_shared_state`.
        """
        if not isinstance(result, dict):
            return False
        if task_kind in ("baseline", "profile"):
            return is_valid_measurement(result)
        if task_kind == "sweep":
            return result.get("status") == "succeeded"
        # replay_warm_recipe ALWAYS routes through _promote_warm_replay (owns succeeded/drift/FAILED + clears in_flight); else PRELUDE blocks forever.
        if task_kind == "replay_warm_recipe":
            return True
        return result.get("status") != "failed"

    def _record_intervention_for_task(
        self, task: "Task", result: Any,
    ) -> None:
        """PR-A8: log a completed task's change_type into SharedState.intervention_mix (explore → config; integrate_patch → code_patch_attempt or code_patch when kept). Best-effort."""
        if not isinstance(result, dict):
            return
        kind = (task.kind or "").strip()
        if kind == "explore":
            # Winner surrogate: result.winners present OR best_variant set.
            winners = result.get("winners") or []
            best = result.get("best_variant")
            if not winners and not best:
                # B2: an explore round that KEPT nothing still counts as a config-only attempt.
                self.shared_state.record_intervention(
                    change_type="config_attempt",
                    action="explore",
                    task_id=task.task_id,
                    delta_pct=None,
                )
                return
            delta_pct = None
            if isinstance(best, dict):
                delta_pct = best.get("gain_pct")
            self.shared_state.record_intervention(
                change_type="config",
                action="explore",
                task_id=task.task_id,
                delta_pct=delta_pct if isinstance(delta_pct, (int, float)) else None,
            )
            return
        if kind == "integrate_patch":
            status = str(result.get("status") or "").strip().lower()
            if not status:
                return
            if status != "kept":
                self.shared_state.record_intervention(
                    change_type="code_patch_attempt",
                    action="integrate_patch",
                    task_id=task.task_id,
                    delta_pct=result.get("delta_pct"),
                )
                return
            self.shared_state.record_intervention(
                change_type="code_patch",
                action="integrate_patch",
                task_id=task.task_id,
                delta_pct=result.get("delta_pct"),
            )

    async def _handle_unpromotable_result(
        self, task: Task, result: dict[str, Any] | None,
    ) -> None:
        """Record a failed / unpromotable task result into SharedState: append to last_action_failures (+ a failed attempts row for _AUDIT_ACTIONS); keep baseline failure_streak/stop_reason logic intact."""
        result_payload = result or {}
        any_changed = False
        # Per-action audit (failed attempt) for the 6 in-scope kinds.
        if task.kind in _AUDIT_ACTIONS:
            audit_extras: dict[str, Any] = {}
            # Stamp baseline-params fingerprint so the self-loop denial helper detects "same params failed twice".
            if task.kind == "baseline":
                audit_extras["fingerprint"] = _baseline_params_fingerprint(
                    task.params
                )
            self.shared_state.record_action_attempt(
                action=task.kind,
                task_id=task.task_id,
                status="failed",
                decision="no_promote",
                result=result_payload,
                extras=audit_extras,
            )
            any_changed = True
        # Global rolling failure log (every kind, including kernel-owned).
        self.shared_state.record_action_failure(
            action=task.kind,
            task_id=task.task_id,
            result=result_payload,
        )
        any_changed = True
        # Baseline-specific gates: streak counter + stop_reason + baseline_not_promoted event.
        # #522: fast arg errors (fast_exit_arg_error) get their own streak so
        # they don't burn the slow-baseline retry budget on deterministic
        # failures that the same params will never fix.
        baseline_event_payload: dict[str, Any] | None = None
        if task.kind == "baseline" and self.shared_state.baseline_tput <= 0:
            err_class = result_payload.get("error_class", "")
            if err_class == "fast_exit_arg_error":
                self.shared_state.baseline_arg_error_streak += 1
                if self.shared_state.baseline_arg_error_streak >= 2:
                    self.shared_state.set_stop_reason("baseline_arg_error")
            else:
                self.shared_state.baseline_failure_streak += 1
                self.shared_state.baseline_arg_error_streak = 0
                if self.shared_state.baseline_failure_streak >= 3:
                    self.shared_state.set_stop_reason("baseline_failed")
            baseline_event_payload = {
                "kind": "baseline_not_promoted",
                "task_id": task.task_id,
                "failure_streak": self.shared_state.baseline_failure_streak,
                "arg_error_streak": self.shared_state.baseline_arg_error_streak,
                "stop_reason": self.shared_state.stop_reason,
                "result_status": result_payload.get("status"),
                "error_class": err_class,
            }
            any_changed = True
        # Mirror the promote-path roofline failure handling: bump streak, clear auto-roofline gate, emit operator warning.
        if task.kind == "roofline":
            if hasattr(self.shared_state, "roofline_failure_streak"):
                self.shared_state.roofline_failure_streak += 1
            if (
                self.shared_state.auto_roofline_pending_task_id
                == task.task_id
            ):
                self.shared_state.auto_roofline_pending_task_id = ""
            any_changed = True
            log.warning(
                "Auto-roofline %s failed (reason=%s phase=%s "
                "error_class=%s); continuing in degraded mode "
                "(specialists / explore proceed without a fresh "
                "analysis_md). No retry, no fallback.",
                task.task_id,
                str((task.params or {}).get("reason") or ""),
                result_payload.get("phase"),
                result_payload.get("error_class"),
            )
        if any_changed:
            self.shared_state.save(self.session_dir)
        if baseline_event_payload is not None:
            await self.bus.append_and_seq(Message.new(
                "coordinator", "*", "event", baseline_event_payload,
            ))

    async def _pump_dispatcher_once(self) -> None:
        """Dispatch queued tasks respecting per-lane capacity (serial at capacity=1, fans out when research_lane.capacity > 1). Inv-7.3: lease bound to task_id, runner releases it."""
        queued = await self.tasks.queued()
        if not queued:
            return
        holders = await self.locks.lane_holders()
        capacities = await self.locks.lane_capacities()
        spawned: list[tuple[Task, asyncio.Task[SubAgentResult], Any]] = []
        for task in queued:
            lanes_needed = list(task.requires_lanes or [])
            if lanes_needed:
                try:
                    expanded = _expand_lanes(lanes_needed)
                except ValueError:
                    log.warning(
                        "dispatcher: task %s has unknown lane in %r; "
                        "skipping until resolved",
                        task.task_id, lanes_needed,
                    )
                    continue
                if not self._lanes_fit(expanded, holders, capacities):
                    # Stays queued; next tick re-evaluates after holders release.
                    continue
                lease = await self.locks.try_acquire_many(
                    lanes_needed,
                    holder_id=task.task_id,
                    task_id=task.task_id,
                    action=task.kind,
                    ttl_sec=task.lease_ttl_sec or 60,
                )
                if lease is None:
                    # Race: another holder grabbed the lane; leave queued.
                    continue
                # Reflect the bump in our local view for the next task in this tick.
                for lane in lease.lanes:
                    holders[lane] = int(holders.get(lane, 0)) + 1
            else:
                lease = None
            gpu_lease = None
            extra_context: dict[str, Any] = {}
            if task.kind == "specialist":
                params = task.params or {}
                needs_gpu_raw = params.get("needs_gpu", False)
                needs_gpu = (
                    needs_gpu_raw.strip().lower() in ("1", "true", "yes", "on")
                    if isinstance(needs_gpu_raw, str)
                    else bool(needs_gpu_raw)
                )
                if needs_gpu:
                    try:
                        gpu_count = int(params.get("gpu_count", 1) or 1)
                    except (TypeError, ValueError):
                        gpu_count = 1
                    try:
                        max_turns = int(params.get("max_turns", 8) or 8)
                    except (TypeError, ValueError):
                        max_turns = 8
                    try:
                        per_turn_s = float(
                            params.get("specialist_per_turn_max_seconds")
                            or os.environ.get(
                                "INFERENCE_OPTIMIZER_SPECIALIST_PER_TURN_MAX_SECONDS",
                                "600",
                            )
                            or 600.0
                        )
                    except (TypeError, ValueError):
                        per_turn_s = 600.0
                    gpu_ttl_sec = max(
                        int(task.lease_ttl_sec or 0),
                        int(max(1, max_turns) * max(1.0, per_turn_s)),
                    )
                    gpu_lease = await self.gpu_specialist_pool.try_acquire(
                        count=gpu_count,
                        holder_id=task.task_id,
                        task_id=task.task_id,
                        ttl_sec=gpu_ttl_sec,
                    )
                    if gpu_lease is None:
                        if lease is not None:
                            await self.locks.release(lease)
                            for lane in lease.lanes:
                                holders[lane] = max(
                                    0, int(holders.get(lane, 0)) - 1,
                                )
                        continue
                    extra_context["gpu_ids"] = list(gpu_lease.gpu_ids)
            spawned.append((
                task,
                asyncio.create_task(
                    self.sub.run_task(
                        task,
                        prebound_lease=lease,
                        extra_context=extra_context,
                    ),
                ),
                gpu_lease,
            ))
        if not spawned:
            return
        # Gather all spawned tasks (keeps tick semantics simple); defensively absorb any leaked exception.
        results = await asyncio.gather(
            *(t for _, t, _ in spawned), return_exceptions=True,
        )
        for (task, _, gpu_lease), maybe_result in zip(spawned, results):
            if gpu_lease is not None:
                try:
                    await self.gpu_specialist_pool.release(gpu_lease)
                except Exception:  # noqa: BLE001 — defensive cleanup
                    log.exception(
                        "dispatcher: failed to release GPU specialist lease "
                        "for task=%s", task.task_id,
                    )
            if isinstance(maybe_result, BaseException):
                log.exception(
                    "dispatcher: spawned task %s raised: %r",
                    task.task_id, maybe_result,
                )
                continue
            result: SubAgentResult = maybe_result
            # Bounded transient-failure auto-retry (infra only): on a subprocess
            # timeout / crash / stale-heartbeat, re-enqueue a fresh specialist
            # task and skip THIS attempt's delegated_result + bookkeeping so the
            # flake neither pollutes the gaps ledger nor provokes a manual
            # re-dispatch. Semantic empties fall through and are recorded.
            if task.kind == "specialist":
                try:
                    if await self._maybe_auto_retry_specialist(task, result):
                        continue
                except Exception:  # noqa: BLE001 — never block the dispatch loop
                    log.exception(
                        "specialist auto-retry hook failed for task=%s",
                        task.task_id,
                    )
            try:
                await self.bus.append_and_seq(Message.new(
                    "coordinator", "*", "delegated_result",
                    {"task_id": task.task_id, "kind": task.kind,
                     "state": result.state, "result": result.result,
                     "error": result.error},
                ))
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "dispatcher: failed to append delegated_result for task=%s",
                    task.task_id,
                )
                self._record_coordinator_exception(
                    stage="dispatcher_result",
                    exc=exc,
                )
                continue
            # Specialist bookkeeping (Gap-01): done payload under result.result['specialist_done']; always runs (incl. empty-synthesised) to keep the ledgers coherent.
            if task.kind == "specialist":
                result_dict = result.result if isinstance(result.result, dict) else {}
                done_payload = result_dict.get("specialist_done") or {}
                if isinstance(done_payload, dict):
                    try:
                        await self._record_specialist_result(
                            task=task,
                            done_payload=done_payload,
                            source=(
                                f"{SPECIALIST_FROM_AGENT_PREFIX}{task.task_id}"
                            ),
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "specialist bookkeeping hook failed for task=%s",
                            task.task_id,
                        )
                # Bump the per-EXPLORE specialist dispatch counter (Robustness reads it to detect storms).
                try:
                    self.shared_state.bump_specialist_dispatched()
                except Exception:  # noqa: BLE001
                    log.exception("PR-A8: bump_specialist_dispatched failed")
            # intervention-mix ledger: log change_type for explore/integrate_patch so Robustness sees config streaks.
            if task.kind in ("explore", "integrate_patch"):
                try:
                    self._record_intervention_for_task(task, result.result)
                except Exception:  # noqa: BLE001
                    log.exception(
                        "PR-A8: intervention ledger update failed for task=%s",
                        task.task_id,
                    )
            # integrate_patch completion handling.
            if task.kind == "integrate_patch":
                # FRAMEWORK_PR authoring bridge: record authored-patch KEEP/REVERT into framework_pr_phase_progress.
                if (
                    getattr(
                        self.shared_state,
                        "framework_pr_authoring_enabled",
                        False,
                    )
                    and (self.shared_state.phase or "").strip().upper()
                    == _phase_state.PHASE_FRAMEWORK_PR
                ):
                    try:
                        self._record_framework_pr_authored_outcome(
                            task=task, result=result,
                        )
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception(
                            "FRAMEWORK_PR authored-outcome bridge failed "
                            "for task=%s", task.task_id,
                        )
            # Auto-promote succeeded results into CORE_STATE_FIELDS (Coordinator-only writer; DESIGN §14.5/§17.2); promotion needs task-specific invariants beyond no-throw.
            kept = (
                result.state == "succeeded"
                and self._is_promotable_result(task.kind, result.result or {})
            )
            try:
                if kept:
                    await self._promote_to_shared_state(
                        task.kind, result.result, task=task,
                    )
                else:
                    await self._handle_unpromotable_result(task, result.result)
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "dispatcher: promotion/unpromotable handling failed "
                    "for task=%s",
                    task.task_id,
                )
                self._record_coordinator_exception(
                    stage="dispatcher_promote",
                    exc=exc,
                )
                continue
            # Fact-write hook: always called so KEEP/REVERT lands in the journal + (when enabled) a KB write.
            # replay_warm_recipe is excluded (verification, not a new fact; _promote_warm_replay journals it).
            if task.kind != "replay_warm_recipe":
                try:
                    await self._fact_write_hook(task=task, result=result, kept=kept)
                except Exception as exc:  # noqa: BLE001
                    log.exception(
                        "dispatcher: fact-write hook failed for task=%s",
                        task.task_id,
                    )
                    self._record_coordinator_exception(
                        stage="dispatcher_fact_write",
                        exc=exc,
                    )
                    continue
            # explore-round gap update: append per-variant KEEP/REVERT to the gap, then re-run the global refresh.
            if task.kind == "explore":
                result_dict = (
                    result.result if isinstance(result.result, dict) else {}
                )
                try:
                    self._record_explore_round_gaps(
                        task=task, result=result_dict,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "gaps refresh: explore-round update failed for "
                        "task=%s", task.task_id,
                    )
                try:
                    await self._refresh_gaps(reason="explore_round")
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "gaps refresh: _refresh_gaps after explore failed "
                        "for task=%s", task.task_id,
                    )

    @staticmethod
    def _lanes_fit(
        expanded_lanes: list[str],
        holders: dict[str, int],
        capacities: dict[str, int],
    ) -> bool:
        """Local-view headroom hint for the concurrent dispatcher (authoritative gate is try_acquire_many)."""
        for lane in expanded_lanes:
            cap = int(capacities.get(lane, 1))
            used = int(holders.get(lane, 0))
            if cap <= 0 or used >= cap:
                return False
        return True

    # Fact-write dispatcher (KEEP / REVERT entry point): route terminal results to journal + KB fact-write helpers.
    def _source_session_id(self) -> str:
        """Return the hyperloom-local session id used as source_session_id on KB fact writes.

        NOT a KB-side session id; prefers cortex_session_id, falls back to session_dir.name.
        """
        return (
            str(getattr(self.shared_state, "cortex_session_id", "") or "")
            or self.session_dir.name
        )

    async def _fact_write_hook(
        self,
        *,
        task: "Task",
        result: Any,
        kept: bool,
    ) -> None:
        """Per-task fact-write entry point (per_variant for explore grids, else per-task); best-effort, never raises."""
        result_dict = result.result if hasattr(result, "result") else (result or {})
        if not isinstance(result_dict, dict):
            result_dict = {}
        source_session_id = self._source_session_id()
        per_variant = result_dict.get("per_variant_outcomes")
        if (
            task.kind == "explore"
            and isinstance(per_variant, list)
            and per_variant
        ):
            for vo in per_variant:
                try:
                    self._record_fact_per_variant(
                        task=task,
                        source_session_id=source_session_id,
                        variant_outcome=vo,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "fact-write per-variant failed (task=%s)", task.task_id,
                    )
        else:
            try:
                self._record_fact_per_task(
                    task=task,
                    source_session_id=source_session_id,
                    result_dict=result_dict,
                    kept=kept,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "fact-write per-task failed (task=%s)", task.task_id,
                )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive; never crash on save
            log.exception("fact-write SharedState.save failed")


    # Fact-write surface — journal + direct KB lesson/pitfall/recipe writes (the fact side of KB integration).
    PITFALL_REGRESS_THRESHOLD_PCT: float = -5.0  # gain_pct ≤ this → pitfall
    def _ensure_journal(self) -> Journal:
        """Lazy-instantiate the per-session :class:`Journal` (load_or_create reads an existing file on resume)."""
        existing = getattr(self, "_journal", None)
        if existing is None:
            ss = self.shared_state
            self._journal = Journal.load_or_create(
                self.session_dir,
                session_id=str(getattr(ss, "cortex_session_id", "") or "")
                           or str(getattr(ss, "session_id", "") or "")
                           or self.session_dir.name,
                model=str(getattr(ss, "model_name", "") or ""),
                hardware=str(getattr(ss, "gpu_type", "") or ""),
                framework=str(getattr(ss, "framework", "") or ""),
                baseline_throughput=float(getattr(ss, "baseline_tput", 0.0) or 0.0),
            )
        else:
            # Backfill baseline once the baseline executor finishes.
            existing.update_baseline(
                float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0)
            )
        return self._journal

    def _pitfall_severity_for(
        self, result_dict: dict[str, Any] | None,
    ) -> str | None:
        """Decide whether a failed result warrants a pitfall row (Threshold-B): crash/oom/hang → SEVERITY_CRASH; gain_pct ≤ -5% → SEVERITY_REGRESS; else None."""
        if not isinstance(result_dict, dict):
            return None
        error_class = str(result_dict.get("error_class") or "").lower()
        if error_class in ("crash", "oom", "hang"):
            return _SEVERITY_CRASH
        status = str(result_dict.get("status") or "").lower()
        if status in ("crash", "oom", "hang"):
            return _SEVERITY_CRASH
        gain = result_dict.get("gain_pct")
        try:
            gain_pct = float(gain) if gain is not None else None
        except (TypeError, ValueError):
            gain_pct = None
        if gain_pct is not None and gain_pct <= self.PITFALL_REGRESS_THRESHOLD_PCT:
            return _SEVERITY_REGRESS
        return None

    def _journal_entry_phase(self) -> str:
        """Return the current phase label for journal entries.

        Returns:
            str: The uppercased phase name, or ``"UNKNOWN"`` when unset.
        """
        return str(getattr(self.shared_state, "phase", "") or "").strip().upper() or "UNKNOWN"

    def _record_fact_per_task(
        self,
        *,
        task: "Task",
        source_session_id: str,
        result_dict: dict[str, Any],
        kept: bool,
    ) -> None:
        """Per-task fact write — one journal row + maybe one KB fact (source_session_id is hyperloom-local)."""
        journal = self._ensure_journal()
        gain_raw = result_dict.get("gain_pct")
        try:
            gain_pct = float(gain_raw) if gain_raw is not None else None
        except (TypeError, ValueError):
            gain_pct = None
        tput_raw = result_dict.get("output_throughput")
        try:
            throughput_after = float(tput_raw) if tput_raw is not None else None
        except (TypeError, ValueError):
            throughput_after = None
        kind = classify_change_kind(task.kind, None)
        change = summarize_change(task.kind, None, result_dict)
        if kept:
            outcome = OUTCOME_KEEP
            error_class = None
            reason = None
        else:
            outcome = OUTCOME_REVERT
            error_class = (str(result_dict.get("error_class") or "") or None)
            reason = (str(result_dict.get("reason") or "") or None)
        journal.append_entry(JournalEntry(
            phase=self._journal_entry_phase(),
            iter=int(self.shared_state.tick or 0),
            kind=kind,
            change=change,
            outcome=outcome,
            gain_pct=gain_pct,
            throughput_after=throughput_after,
            error_class=error_class,
            reason=reason,
            task_id=task.task_id,
            tick=int(self.shared_state.tick or 0),
        ))

        if self.cortex_kb is None:
            return

        models = [str(self.shared_state.model_name or "")] if self.shared_state.model_name else []
        hardware = [str(self.shared_state.gpu_type or "")] if self.shared_state.gpu_type else []
        # evidence_refs (log:task-...) gives traceability since source_session_id lands in attrs.
        evidence_refs = [f"log:task-{task.task_id}"]
        # Workload-shape tags for lesson/pitfall attrs so the warm-start reader filters cross-framework noise.
        workload_tags = self._collect_workload_tags()
        extra = workload_tags if workload_tags else None
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if kept and gain_pct is not None and gain_pct > 0:
            statement = self._build_statement(
                change=change, gain_pct=gain_pct, kind="lesson",
            )
            impact = self._build_measured_impact(
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                stack_depth=len(getattr(self.shared_state, "optimization_stack", []) or []),
                measured_at=now_iso,
            )
            # v2: append onto the recipe's lessons[] (no cross-recipe dedup).
            self._kb_amend_recipe(
                append_lesson={
                    "statement":       statement,
                    "measured_impact": impact,
                },
                provenance_details={
                    "source_session_id": source_session_id,
                    "source_task_id":    task.task_id,
                    "evidence":          list(evidence_refs or []),
                    "applicable_models":   list(models or []),
                    "applicable_hardware": list(hardware or []),
                    "extra":             dict(extra or {}),
                    "now":               now_iso,
                },
            )
            return

        severity = self._pitfall_severity_for(result_dict)
        if severity is not None:
            description = self._build_statement(
                change=change, severity=severity, kind="pitfall",
            )
            self._kb_amend_recipe(
                append_pitfall={
                    "description": description,
                    "severity":    severity,
                },
                provenance_details={
                    "source_session_id": source_session_id,
                    "source_task_id":    task.task_id,
                    "evidence":          list(evidence_refs or []),
                    "applicable_models":   list(models or []),
                    "applicable_hardware": list(hardware or []),
                    "extra":             dict(extra or {}),
                    "now":               now_iso,
                },
            )

    def _build_statement(
        self,
        *,
        change: str,
        kind: str,
        gain_pct: float | None = None,  # kept for backward call-signature compat
        severity: str | None = None,
    ) -> str:
        """Build the lesson statement / pitfall description hashed into the KB canonical_id; MUST exclude volatile fields (e.g. gain_pct) so N sessions merge instead of producing N rows. Identity = framework + change + model/hw."""
        framework = str(getattr(self.shared_state, "framework", "") or "").strip()
        fw_tag = f"[{framework or '?'}] "
        model = self.shared_state.model_name or "?"
        hw = self.shared_state.gpu_type or "?"
        if kind == "lesson":
            # gain_pct intentionally NOT included — see docstring.
            return f"{fw_tag}{change} on {model}/{hw}"
        # kind == "pitfall"
        return f"{fw_tag}{change} → {severity or '?'} on {model}/{hw}"

    @staticmethod
    def _build_measured_impact(
        *,
        gain_pct: float | None,
        throughput_after: float | None,
        stack_depth: int,
        measured_at: str,
        throughput_before: float | None = None,
    ) -> dict[str, Any]:
        """GAP 3 — structured ``measured_impact`` payload (dict not legacy string so consumers parse without regex); stack_depth = stack length before this lesson lands."""
        out: dict[str, Any] = {
            "gain_pct": float(gain_pct) if gain_pct is not None else None,
            "stack_depth_at_apply": int(stack_depth),
            "measured_at": measured_at,
        }
        if throughput_after is not None:
            out["throughput_after"] = float(throughput_after)
        if throughput_before is not None:
            out["throughput_before"] = float(throughput_before)
        # Strip None for compactness (prompt section uses .get).
        return {k: v for k, v in out.items() if v is not None}

    def _record_fact_per_variant(
        self,
        *,
        task: "Task",
        source_session_id: str,
        variant_outcome: dict[str, Any],
    ) -> None:
        """Per-variant fact write — mirror of _record_fact_per_task for explore per-variant decisions."""
        journal = self._ensure_journal()
        outcome_raw = str(variant_outcome.get("outcome") or "")
        if outcome_raw == "KEEP":
            outcome = OUTCOME_KEEP
        elif outcome_raw in ("REVERT", "FAILED", "KEEP_UNSTABLE"):
            outcome = OUTCOME_REVERT
        elif outcome_raw == "SKIPPED_DEDUP":
            return  # nothing to journal
        else:
            outcome = OUTCOME_NO_PROMOTE
        variant_name = str(variant_outcome.get("variant_name") or "")
        metrics = variant_outcome.get("metrics") or {}
        gain_raw = metrics.get("gain_pct") if isinstance(metrics, dict) else None
        try:
            gain_pct = float(gain_raw) if gain_raw is not None else None
        except (TypeError, ValueError):
            gain_pct = None
        tput_raw = metrics.get("output_throughput") if isinstance(metrics, dict) else None
        try:
            throughput_after = float(tput_raw) if tput_raw is not None else None
        except (TypeError, ValueError):
            throughput_after = None
        variant_attrs = variant_outcome.get("variant") or {}
        kind = classify_change_kind(
            task.kind, variant_attrs if isinstance(variant_attrs, dict) else None,
        )
        # Ensure the change summary is variant-specific (else every explore variant writes an identical row).
        change_attrs = dict(variant_attrs) if isinstance(variant_attrs, dict) else {}
        if not (
            change_attrs.get("extra_sglang_args")
            or change_attrs.get("extra_envs")
            or change_attrs.get("name")
        ) and variant_name:
            change_attrs["name"] = variant_name
        change = summarize_change(task.kind, change_attrs, None)
        error_class = None
        reason = None
        if outcome == OUTCOME_REVERT:
            error_class = (str(variant_outcome.get("error_class") or "") or None)
            reason = (str(variant_outcome.get("reason") or "") or None)
        journal.append_entry(JournalEntry(
            phase=self._journal_entry_phase(),
            iter=int(self.shared_state.tick or 0),
            kind=kind,
            change=change,
            outcome=outcome,
            gain_pct=gain_pct,
            throughput_after=throughput_after,
            error_class=error_class,
            reason=reason,
            task_id=task.task_id,
            variant_name=variant_name,
            tick=int(self.shared_state.tick or 0),
        ))

        if self.cortex_kb is None:
            return

        models = [str(self.shared_state.model_name or "")] if self.shared_state.model_name else []
        hardware = [str(self.shared_state.gpu_type or "")] if self.shared_state.gpu_type else []
        evidence_refs = [
            f"log:task-{task.task_id}",
            f"variant:{variant_name}",
        ]
        # Workload-shape tags — see _record_fact_per_task.
        workload_tags = self._collect_workload_tags()
        extra = workload_tags if workload_tags else None

        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if outcome == OUTCOME_KEEP and gain_pct is not None and gain_pct > 0:
            statement = self._build_statement(
                change=change, gain_pct=gain_pct, kind="lesson",
            )
            impact = self._build_measured_impact(
                gain_pct=gain_pct,
                throughput_after=throughput_after,
                stack_depth=len(getattr(self.shared_state, "optimization_stack", []) or []),
                measured_at=now_iso,
            )
            # v2: per-variant lesson append onto recipe.lessons[]
            # (no cross-recipe dedup, see _record_fact_per_task).
            self._kb_amend_recipe(
                append_lesson={
                    "statement":       statement,
                    "measured_impact": impact,
                },
                provenance_details={
                    "source_session_id":   source_session_id,
                    "source_task_id":      task.task_id,
                    "source_variant_name": variant_name,
                    "evidence":            list(evidence_refs or []),
                    "applicable_models":   list(models or []),
                    "applicable_hardware": list(hardware or []),
                    "extra":               dict(extra or {}),
                    "now":                 now_iso,
                },
            )
            return

        severity = self._pitfall_severity_for({
            **(metrics if isinstance(metrics, dict) else {}),
            "error_class": variant_outcome.get("error_class"),
            "status":      variant_outcome.get("outcome"),
        })
        if severity is not None:
            description = self._build_statement(
                change=change, severity=severity, kind="pitfall",
            )
            self._kb_amend_recipe(
                append_pitfall={
                    "description": description,
                    "severity":    severity,
                },
                provenance_details={
                    "source_session_id":   source_session_id,
                    "source_task_id":      task.task_id,
                    "source_variant_name": variant_name,
                    "evidence":            list(evidence_refs or []),
                    "applicable_models":   list(models or []),
                    "applicable_hardware": list(hardware or []),
                    "extra":               dict(extra or {}),
                    "now":                 now_iso,
                },
            )

    def _collect_workload_tags(self) -> dict[str, Any]:
        """Return the workload-shape KB tag dict for the current session (GAP 5); shared by recipe attrs + lesson/pitfall writes so the warm-start reader filters symmetrically."""
        ss = self.shared_state
        out: dict[str, Any] = {}
        framework = str(getattr(ss, "framework", "") or "").strip()
        if framework:
            out["framework"] = framework
        model_class = str(getattr(ss, "model_class", "") or "").strip()
        if model_class:
            out["model_class"] = model_class
        # model_family (v1 fallback) no longer stamped: v2 uses the exact 5-tuple canonical_id.
        model_name = str(getattr(ss, "model_name", "") or "").strip()
        if model_name:
            out["model_name"] = model_name
        for src_attr, dst_key in (
            ("precision",     "precision"),
            ("tp",            "tp"),
            ("ep",            "ep"),
            ("conc",          "conc"),
            ("isl",           "isl"),
            ("osl",           "osl"),
            ("max_model_len", "max_model_len"),
        ):
            v = getattr(ss, src_attr, None)
            if v not in (None, "", 0):
                out[dst_key] = v
        # EP env fallback when SharedState.ep is unset (legacy SDK callers).
        if "ep" not in out:
            raw_ep = (os.environ.get("EP") or "").strip()
            try:
                n = int(raw_ep) if raw_ep else 0
            except ValueError:
                n = 0
            if n > 0:
                out["ep"] = n
        # PP — no SharedState field (no CLI surface); env-only.
        raw_pp = (os.environ.get("PP") or "").strip()
        try:
            pp_n = int(raw_pp) if raw_pp else 0
        except ValueError:
            pp_n = 0
        if pp_n > 0:
            out["pp"] = pp_n
        # runtime version tags from stack_fingerprint_meta (cli writes at boot, resume reads verbatim).
        fp_meta = getattr(ss, "stack_fingerprint_meta", None) or {}
        if isinstance(fp_meta, dict):
            # framework_version is whichever of sglang/vllm is active.
            fw_lc = framework.lower()
            if fw_lc in ("sglang", "vllm"):
                v = str(fp_meta.get(fw_lc) or "").strip()
                if v and v != "unknown":
                    out["framework_version"] = v
            for src_key, dst_key in (
                ("rocm",         "rocm_version"),
                ("aiter",        "aiter_version"),
                ("image_digest", "image_digest"),
            ):
                v = str(fp_meta.get(src_key) or "").strip()
                if v and v != "unknown":
                    out[dst_key] = v
        # per-baseline workload extras from materialized YAML; keep bool False (don't drop an "explicitly disabled" signal).
        wl_extra = getattr(ss, "baseline_workload_extra", None) or {}
        if isinstance(wl_extra, dict):
            for k in ("max_running_requests", "max_num_seqs"):
                v = wl_extra.get(k)
                if isinstance(v, int) and v > 0:
                    out[k] = v
            for k in ("chunked_prefill_enabled", "enable_torch_compile"):
                v = wl_extra.get(k)
                if isinstance(v, bool):
                    out[k] = v
            for k in ("quant_scheme", "workload_mode"):
                v = wl_extra.get(k)
                if isinstance(v, str) and v.strip():
                    out[k] = v.strip()
        return out

    def _build_kernel_optimizations_from_state(self) -> list[dict[str, Any]]:
        """Collect KEEP'd kernel optimizations + their E2E verdict by joining kernel_opt_attempts (micro) and kernel_integrate_attempts (E2E) on kernel_id; non-integrated KEEPs surface integrated=False. Returns KernelOptimization-shaped dicts."""
        ss = self.shared_state
        opt_attempts = getattr(ss, "kernel_opt_attempts", {}) or {}
        integ_attempts = getattr(ss, "kernel_integrate_attempts", {}) or {}
        if not isinstance(opt_attempts, dict):
            return []

        # Index integrate results by kernel_id (last write wins; entry carries rolled-up best_gain_pct).
        integ_by_kid: dict[str, dict[str, Any]] = {}
        if isinstance(integ_attempts, dict):
            for entry in integ_attempts.values():
                if not isinstance(entry, dict):
                    continue
                kid = str(entry.get("kernel_id") or "")
                if kid:
                    integ_by_kid[kid] = entry

        out: list[dict[str, Any]] = []
        for kid, e in opt_attempts.items():
            if not isinstance(e, dict):
                continue
            if str(e.get("last_decision", "")).upper() != "KEEP":
                continue
            try:
                micro = float(e.get("last_micro_speedup") or 0.0)
            except (TypeError, ValueError):
                micro = 0.0
            integ = integ_by_kid.get(str(kid))
            e2e_gain = 0.0
            e2e_tput = 0.0
            e2e_decision = ""
            integrated = False
            if isinstance(integ, dict):
                integrated = True
                # Integrate-layer verdict (E2E); lets warm-start skip a micro-win/E2E-loss kernel.
                e2e_decision = str(integ.get("last_decision") or "").upper()
                try:
                    e2e_gain = float(integ.get("best_gain_pct") or 0.0)
                except (TypeError, ValueError):
                    e2e_gain = 0.0
                # Last attempt's E2E re-bench throughput.
                for att in reversed(list(integ.get("attempts") or [])):
                    if isinstance(att, dict) and att.get("new_tput") is not None:
                        try:
                            e2e_tput = float(att.get("new_tput") or 0.0)
                        except (TypeError, ValueError):
                            e2e_tput = 0.0
                        break
            out.append({
                "kernel_id":     str(kid),
                # source persisted under last_source_file; source_file is a legacy fallback.
                "source_file":   str(
                    e.get("last_source_file") or e.get("source_file") or ""
                ),
                "artifact_path": str(e.get("last_artifact_path") or ""),
                "micro_speedup": micro,
                "decision":      "KEEP",
                "e2e_gain_pct":  e2e_gain,
                "e2e_tput":      e2e_tput,
                "e2e_decision":  e2e_decision,
                "integrated":    integrated,
                "ts":            str(e.get("last_ts") or e.get("ts") or ""),
            })
        return out

    def _collect_attempt_provenance(
        self,
    ) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
        """Map proven optimizations to their research-hint origin from the gaps[] attempts ledger; returns (kept_sources by name/kernel, kept_by_gap by canonical_id, reverted_rows). Fail-soft."""
        kept_sources: dict[str, str] = {}
        kept_by_gap: dict[str, str] = {}
        reverted_rows: list[dict[str, Any]] = []
        gaps = getattr(self.shared_state, "gaps", []) or []
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            provenance = str(gap.get("provenance") or "").strip()
            canonical = str(gap.get("canonical_id") or "").strip()
            for attempt in gap.get("attempts") or []:
                if not isinstance(attempt, dict):
                    continue
                variant = str(attempt.get("variant_name") or "").strip()
                kernel = str(attempt.get("kernel_id") or "").strip()
                outcome = str(attempt.get("outcome") or "").strip().upper()
                if outcome == "KEEP" and provenance:
                    if variant:
                        kept_sources.setdefault(variant, provenance)
                    if kernel:
                        kept_sources.setdefault(kernel, provenance)
                    if canonical:
                        kept_by_gap.setdefault(canonical, provenance)
                elif outcome == "REVERT" and (variant or kernel):
                    row: dict[str, Any] = {
                        "name": variant or kernel,
                        "reason": "reverted",
                        "gain_pct": attempt.get("gain_pct"),
                    }
                    if provenance:
                        row["source"] = provenance
                    reverted_rows.append(row)
        return kept_sources, kept_by_gap, reverted_rows

    def _build_recipe_attrs_from_state(self) -> dict[str, Any]:
        """Materialise the recipe-shaped view of :class:`SharedState` (kg-usage-guide §7.4; defensive getattr)."""
        ss = self.shared_state
        current_best = getattr(ss, "current_best", {}) or {}
        opt_stack = getattr(ss, "optimization_stack", []) or []
        gain_per_stack = getattr(ss, "gain_per_stack_entry", []) or []
        last_failures = getattr(ss, "last_action_failures", []) or []
        # Read canonical extra_server_args first, but WRITE the legacy extra_sglang_args key (RecipeKB schema +
        # warm-replay reader still key on it; reading the stale name would break warm-replay reproduction).
        best_config: dict[str, Any] = {}
        if isinstance(current_best, dict):
            cb_args = (
                current_best.get("extra_server_args")
                or current_best.get("extra_sglang_args")
            )
            if cb_args:
                best_config["extra_sglang_args"] = str(cb_args)
            for key in ("extra_envs", "args", "envs", "name", "tput", "accuracy"):
                if key in current_best:
                    best_config[key] = current_best[key]
        # Prefer the last validated stack layer for launch args (current_best may carry a corrupted string).
        if opt_stack:
            last_entry = opt_stack[-1]
            if isinstance(last_entry, dict):
                # Read canonical keys first, legacy *_sglang_args as fallback (#332 best_config fix).
                stack_args = str(
                    last_entry.get("candidate_extra_server_args")
                    or last_entry.get("extra_server_args")
                    or last_entry.get("candidate_extra_sglang_args")
                    or last_entry.get("extra_sglang_args")
                    or "",
                ).strip()
                if stack_args:
                    best_config["extra_sglang_args"] = stack_args
        sediment_on = bool(getattr(ss, "recipe_sediment_enabled", True))
        kept_sources, kept_by_gap, reverted_rows = (
            self._collect_attempt_provenance() if sediment_on else ({}, {}, [])
        )
        what_worked: list[dict[str, Any]] = []
        for idx, entry in enumerate(opt_stack):
            if not isinstance(entry, dict):
                continue
            gain_per: float | None = None
            if idx < len(gain_per_stack):
                gain_per = gain_per_stack[idx]
            name = str(
                entry.get("variant_name")
                or entry.get("name")
                or entry.get("kernel_id")
                or ""
            )
            row: dict[str, Any] = {
                "name":              name,
                "extra_sglang_args": str(
                    entry.get("extra_server_args")
                    or entry.get("extra_sglang_args")
                    or ""
                ),
                "extra_envs":        dict(entry.get("extra_envs") or {}),
                "gain_pct":          gain_per,
            }
            # Prefer the entry's gap-id provenance (naming-independent); fall back to name/kernel_id match.
            entry_gap = str(entry.get("gap_canonical_id") or "").strip()
            src = (
                (kept_by_gap.get(entry_gap) if entry_gap else None)
                or kept_sources.get(name)
                or kept_sources.get(str(entry.get("kernel_id") or ""))
            )
            if src:
                row["source"] = src
            what_worked.append(row)
        what_failed: list[dict[str, Any]] = []
        for failure in last_failures[-10:]:
            if isinstance(failure, dict):
                what_failed.append({
                    "name":  str(failure.get("name") or failure.get("action") or ""),
                    "reason": str(failure.get("reason") or failure.get("error_class") or ""),
                })
        for rev in reverted_rows:
            what_failed.append(rev)
        kernel_optimizations = self._build_kernel_optimizations_from_state()
        cumulative_validated = float(getattr(ss, "cumulative_gain_validated", 0.0) or 0.0)
        cumulative_total = float(getattr(ss, "cumulative_gain", 0.0) or 0.0)
        validated_stack_len = int(
            getattr(ss, "cumulative_gain_validated_stack_len", 0) or 0
        )
        stack_fingerprint = getattr(ss, "stack_fingerprint", "") or ""
        # Workload-shape tags for shape-filtered warm-start queries (shared via _collect_workload_tags).
        workload_tags = self._collect_workload_tags()
        # framework_version left unset here (manifest-derived); the T0 backfill writes it.
        return {
            "best_config":       best_config,
            "best_throughput":   float(current_best.get("tput", 0.0))
                                  if isinstance(current_best, dict) else 0.0,
            "what_worked":       what_worked,
            "what_failed":       what_failed,
            "kernel_optimizations": kernel_optimizations,
            "stack_fingerprint": {"sha": str(stack_fingerprint)} if stack_fingerprint else {},
            "last_profiled":     str(getattr(ss, "cumulative_gain_validated_ts", "") or ""),
            "workload":          workload_tags,
            "sessions":          [{
                "session_id":   str(getattr(ss, "cortex_session_id", "")
                                    or self.session_dir.name),
                "gain_pct":     cumulative_validated or cumulative_total,
                "stack_len":    validated_stack_len or len(opt_stack),
                # arbor-shape provenance so the session row is self-describing (before/after tput + knobs).
                "throughput_before": float(getattr(ss, "baseline_tput", 0.0) or 0.0),
                "throughput_after":  (
                    float(current_best.get("tput", 0.0))
                    if isinstance(current_best, dict) else 0.0
                ),
                "date":          datetime.now(timezone.utc).isoformat(),
                "actions_taken": [
                    nm for nm in (
                        str(
                            e.get("variant_name") or e.get("name")
                            or e.get("action") or ""
                        ).strip()
                        for e in opt_stack if isinstance(e, dict)
                    ) if nm
                ],
            }],
        }

    def cortex_finalize_recipe_and_journal(self) -> None:
        """CLOSE-time fact finalize: final update_recipe + journal finalize (total_gain_pct + final_throughput); idempotent (CLOSE sequencer + _cortex_t4_hook safety net)."""
        try:
            journal = self._ensure_journal()
            ss = self.shared_state
            cb = getattr(ss, "current_best", {}) or {}
            final_tput = float(cb.get("tput", 0.0)) if isinstance(cb, dict) else 0.0
            total_gain = float(
                getattr(ss, "cumulative_gain_validated", 0.0)
                or getattr(ss, "cumulative_gain", 0.0)
                or 0.0,
            )
            journal.finalize(
                final_throughput=final_tput if final_tput > 0 else None,
                total_gain_pct=total_gain,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("optimization_journal.finalize failed")

        if self.cortex_kb is None:
            return
        ss = self.shared_state
        model_name = getattr(ss, "model_name", "") or ""
        gpu_type = getattr(ss, "gpu_type", "") or ""
        if not model_name or not gpu_type:
            log.info(
                "cortex finalize_recipe: missing model/hardware "
                "(model=%r hardware=%r); skipping update_recipe",
                model_name, gpu_type,
            )
            return
        try:
            attrs = self._build_recipe_attrs_from_state()
            # Hoist workload tags flat into top-level recipe attrs (shallow-merged) for warm-start filters.
            workload_tags = attrs.get("workload") or {}

            # sessions[] read-modify-write: read anchor, drop prior entry with our session_id (resume safety), append ours, write back.
            my_sessions = list(attrs["sessions"] or [])
            my_session_ids = {
                str((s or {}).get("session_id") or "")
                for s in my_sessions if isinstance(s, dict)
            }
            # v2: read-modify-write the recipe row; sessions[] merged in-process under the cid flock so concurrent finalises don't tear.
            merged_sessions: list[dict[str, Any]] = list(my_sessions)
            existing_row: dict[str, Any] = {}
            if self.cortex_kb is not None:
                try:
                    cid = self._workload_canonical_id()
                    # Read the LOCAL row (authoritative for writes) so the merge + guard compare against it.
                    existing_row = self.cortex_kb.local.get_recipe(canonical_id=cid) or {}
                    existing_sessions: list[dict[str, Any]] = []
                    for row in (existing_row.get("sessions") or []):
                        if not isinstance(row, dict):
                            continue
                        if str(row.get("session_id") or "") in my_session_ids:
                            # Resume/retry of the same session — our new entry supersedes the prior one.
                            continue
                        existing_sessions.append(dict(row))
                    merged_sessions = existing_sessions + my_sessions
                except Exception as exc:  # noqa: BLE001 — defensive
                    log.info(
                        "recipe read failed (%s); finalize will append "
                        "the current session only; the next finalize "
                        "will catch up.",
                        exc,
                    )

            # KEEP'd kernel optimizations ride the extras channel; merge with prior rows, dedup by kernel_id.
            kopts_new = list(attrs.get("kernel_optimizations") or [])
            new_kids = {
                str((k or {}).get("kernel_id") or "")
                for k in kopts_new if isinstance(k, dict)
            }
            merged_kopts: list[dict[str, Any]] = list(kopts_new)
            for prior in (existing_row.get("kernel_optimizations") or []):
                if not isinstance(prior, dict):
                    continue
                if str(prior.get("kernel_id") or "") in new_kids:
                    continue
                merged_kopts.append(dict(prior))

            extras_payload = dict(workload_tags or {})
            if merged_kopts:
                extras_payload["kernel_optimizations"] = merged_kopts

            overrides: dict[str, Any] = {
                "what_worked":   attrs["what_worked"],
                "what_failed":   attrs["what_failed"],
                "last_profiled": attrs["last_profiled"],
                "sessions":      merged_sessions,
                "extras":        extras_payload,
            }
            # Overwrite best_config/best_throughput only on a real improvement (repro 20260531T144553Z: bare baseline clobbered a validated config): requires has_validated_win AND my_tput > live_tput.
            my_tput = float(attrs.get("best_throughput") or 0.0)
            cb_now = getattr(ss, "current_best", {}) or {}
            cb_args_now = (
                str(cb_now.get("extra_sglang_args") or "").strip()
                if isinstance(cb_now, dict) else ""
            )
            validated_gain = float(
                getattr(ss, "cumulative_gain_validated", 0.0) or 0.0
            )
            has_validated_win = bool(
                (getattr(ss, "optimization_stack", []) or [])
                or validated_gain > 0.0
                or cb_args_now
            )
            try:
                live_tput = float(existing_row.get("best_throughput") or 0.0)
            except (TypeError, ValueError):
                live_tput = 0.0
            if has_validated_win and my_tput > live_tput:
                overrides["best_config"] = attrs["best_config"]
                overrides["best_throughput"] = my_tput
            # Merge stack_fingerprint rather than replace (CLOSE only has the sha; T0 stamps version keys).
            merged_fp = dict(existing_row.get("stack_fingerprint") or {})
            for fp_key, fp_val in (attrs.get("stack_fingerprint") or {}).items():
                if fp_val not in (None, "", {}):
                    merged_fp[fp_key] = fp_val
            if merged_fp:
                overrides["stack_fingerprint"] = merged_fp

            self._kb_amend_recipe(
                recipe_overrides=overrides,
                provenance_details={
                    "phase": "close_finalize",
                    "evidence": [
                        f"log:session-{getattr(ss, 'cortex_session_id', '') or self.session_dir.name}",
                    ],
                },
            )
        # Catch-all keeps CLOSE step 2.5 defensive against programmer bugs.
        except Exception:  # noqa: BLE001 — defensive
            log.exception("update_recipe raised unexpectedly")

    def _lift_to_current_best(
        self, task_kind: str, best_tput: float, bv: dict[str, Any],
        *, gap_canonical_id: str = "",
    ) -> None:
        """Update SharedState.current_best + recompute cumulative_gain; gap_canonical_id (when known) is stamped onto the stack entry so provenance resolves by gap id not name."""
        previous = self.shared_state.current_best or {}
        if not self.shared_state.optimization_stack:
            self.shared_state.seed_stack_from_current_best()

        base_args = ""
        if isinstance(previous, dict):
            base_args = str(previous.get("extra_server_args") or "").strip()
        candidate_args = ""
        if isinstance(bv, dict):
            candidate_args = str(
                bv.get("candidate_extra_server_args")
                or bv.get("extra_server_args")
                or ""
            ).strip()
        full_args = ""
        if isinstance(bv, dict):
            full_args = str(
                bv.get("extra_server_args")
                or bv.get("extra_sglang_args")
                or ""
            ).strip()
        # Build cumulative launch args without double-stacking; helper dedupes repeated --flag pairs (last wins).
        full_args = _merge_cumulative_extra_sglang_args(
            base_args, candidate_args, full_args,
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
                stack_entry: dict[str, Any] = {
                    "action": task_kind,
                    "variant_name": variant_name,
                    "candidate_extra_server_args": candidate_args,
                    "extra_envs": (
                        dict(bv.get("extra_envs") or {})
                        if isinstance(bv, dict) else {}
                    ),
                    "tput": float(best_tput),
                    "workspace": (
                        bv.get("workspace") if isinstance(bv, dict) else None
                    ),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                if gap_canonical_id:
                    stack_entry["gap_canonical_id"] = gap_canonical_id
                # Stamp the variant's stable join key (and source) so breakdown
                # attribution can map this explore gain back to its specialist
                # provenance via explore_search.winners_history. Without it the
                # phase_breakdown.explore.by_domain join always misses and every
                # gain collapses into ``default_grid``.
                fp_val = ""
                prov_val = ""
                if isinstance(bv, dict):
                    fp_val = str(bv.get("fingerprint") or "").strip()
                    if not fp_val:
                        from .action_executors._canonical_fingerprint import (
                            canonical_fingerprint,
                        )
                        fp_val = canonical_fingerprint(
                            candidate_args or full_args,
                            dict(bv.get("extra_envs") or {}),
                        )
                    prov_val = str(bv.get("provenance") or "").strip()
                if fp_val:
                    stack_entry["fingerprint"] = fp_val
                if prov_val:
                    stack_entry["provenance"] = prov_val
                self.shared_state.optimization_stack.append(stack_entry)
                # Mirror append into gain_per_stack_entry so the two parallel lists stay index-aligned.
                self.shared_state.append_stack_gain_entry(
                    action=task_kind,
                    variant_name=variant_name,
                    new_tput=best_tput,
                    extra_server_args=full_args,
                )

        self.shared_state.current_best = {
            "action": task_kind,
            "tput": float(best_tput),
            "variant_name": variant_name,
            "extra_server_args": full_args,
            "extra_envs": (
                dict(bv.get("extra_envs") or {}) if isinstance(bv, dict) else {}
            ),
            "optimization_stack": list(self.shared_state.optimization_stack),
            "ttft_mean_ms": bv.get("ttft_mean_ms") if isinstance(bv, dict) else None,
            "e2el_mean_ms": bv.get("e2el_mean_ms") if isinstance(bv, dict) else None,
            "tpot_mean_ms": bv.get("tpot_mean_ms") if isinstance(bv, dict) else None,
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
        """Lift specific action-result fields into the persistent SharedState (baseline/profile/roofline/grid)."""
        if not isinstance(result, dict):
            return
        changed = False
        # Audit-trail bookkeeping: each branch sets audit_decision/extras; record_action_attempt runs once after.
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
        if task_kind == "baseline":
            tput = result.get("output_throughput")
            if isinstance(tput, (int, float)) and tput > 0:
                # Fair-comparison anchor (measurement-parity fix).
                # The baseline cold-start guard runs a warmup round on a
                # fresh server (discarded for *reporting*) then a measure
                # round that REUSES the now-hot server — the measure
                # number (``output_throughput``) is systematically ~10-15%
                # higher than a single fresh-server round, because an
                # 8-request client warmup does not fully warm vLLM/SGLang
                # (graph capture, scheduler, allocator) the way a full
                # prior benchmark does. Every ``explore`` / ``sweep``
                # variant, by contrast, RESTARTS the server and runs a
                # single round, so judging them against the hot measure
                # number penalizes each variant by that same ~10-15% and
                # genuinely-good params can never clear the KEEP threshold.
                # Use the warmup round's single-fresh-server tput as the
                # comparison ANCHOR (apples-to-apples with variants) when
                # the double-run captured it; keep the hot number for
                # ``current_best`` / reporting below.
                warmup_anchor = result.get("warmup_round_tput")
                if isinstance(warmup_anchor, (int, float)) and warmup_anchor > 0:
                    self.shared_state.baseline_tput = float(warmup_anchor)
                    self.shared_state.baseline_hot_tput = float(tput)
                    log.info(
                        "baseline anchor: using single-round warmup tput "
                        "%.1f as comparison anchor (hot measure %.1f kept "
                        "for reporting) — measurement parity with explore/"
                        "sweep variants",
                        float(warmup_anchor), float(tput),
                    )
                else:
                    self.shared_state.baseline_tput = float(tput)
                self.shared_state.baseline_failure_streak = 0
                self.shared_state.baseline_arg_error_streak = 0
                changed = True
            acc = result.get("accuracy")
            if isinstance(acc, (int, float)):
                self.shared_state.baseline_accuracy = float(acc)
                changed = True
            # Persist the materialized YAML so downstream tasks reuse the exact workload contract baseline ran.
            materialized = result.get("materialized_config")
            if isinstance(materialized, str) and materialized:
                self.shared_state.baseline_config_path = materialized
                changed = True
                # parse workload-shape extras from the YAML for lesson/pitfall attrs. Best-effort.
                try:
                    parsed = _parse_baseline_workload_extra(materialized)
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "baseline workload extra parsing failed for %s",
                        materialized,
                    )
                    parsed = {}
                if parsed:
                    self.shared_state.baseline_workload_extra = parsed
            # Promote baseline wall-clock so ExploreExecutor derives the per-variant overtime kill deadline.
            runtime_sec_raw = result.get("subprocess_runtime_sec")
            if isinstance(runtime_sec_raw, (int, float)) and runtime_sec_raw > 0:
                self.shared_state.baseline_runtime_sec = float(runtime_sec_raw)
                changed = True
            # current_best.tput is the comparison ANCHOR every explore /
            # sweep variant is judged against (the Coordinator injects it
            # as ``params['base_tput']`` in _handle_delegate /
            # _materialize_approved_proposal). It MUST be the fair
            # single-fresh-server anchor (``baseline_tput``, which the
            # block above set to the warmup-round number under the
            # double-run), NOT the hot measure round — otherwise every
            # cold-restarted variant is judged against an unbeatable hot
            # baseline and can never KEEP. Keep the hot number under a
            # separate ``hot_tput`` field for reporting.
            anchor_tput = float(self.shared_state.baseline_tput or 0.0)
            self.shared_state.current_best = {
                "action": "baseline",
                "tput": (
                    anchor_tput if anchor_tput > 0
                    else (float(tput) if isinstance(tput, (int, float)) else None)
                ),
                "hot_tput": (
                    float(tput) if isinstance(tput, (int, float)) else None
                ),
                "ttft_mean_ms": result.get("ttft_mean_ms"),
                "e2el_mean_ms": result.get("e2el_mean_ms"),
                "tpot_mean_ms": result.get("tpot_mean_ms"),
                "workspace": result.get("workspace"),
            }
            changed = True
            audit_decision = (
                "promoted" if isinstance(tput, (int, float)) and tput > 0
                else "discarded"
            )
            audit_extras = {
                "materialized_config": result.get("materialized_config"),
                "accuracy": result.get("accuracy"),
                "baseline_tput": (
                    float(tput) if isinstance(tput, (int, float)) else None
                ),
                # Stamp canonical params fingerprint so the self-loop denial helper compares run-vs-proposed (_baseline_params_fingerprint).
                "fingerprint": _baseline_params_fingerprint(
                    task.params if task is not None else None
                ),
            }
            # seed the gaps[] ledger from baseline (best-effort).
            await self._refresh_gaps(reason="baseline_done")
            # PRELUDE bootstrap (post-baseline), ordering mandatory: (1) inject warm-recipe history, (2) warm-replay, (3) auto-analysis (deferred while replay in_flight, same GPU/port), (4) research scout.
            if (
                isinstance(tput, (int, float)) and tput > 0
                and not (self.shared_state.auto_roofline_pending_task_id or "").strip()
            ):
                # Step 1 — history injection (fires regardless of --no-warm-replay).
                try:
                    self._inject_warm_recipe_history_into_ledger()
                except Exception as exc:  # noqa: BLE001 — defensive
                    log.exception(
                        "PRELUDE: warm-recipe history injection failed: %r",
                        exc,
                    )
                # Step 2 — warm-recipe replay.
                try:
                    await self._maybe_enqueue_warm_replay(
                        baseline_tput=float(tput),
                    )
                except Exception as exc:  # noqa: BLE001 — defensive
                    log.exception(
                        "PRELUDE: failed to enqueue warm-replay task: %r", exc,
                    )
                # Step 3 — auto-analysis (roofline / profile); may defer.
                await self._maybe_enqueue_prelude_initial_analysis_after_baseline(
                    baseline_tput=float(tput),
                )
                # Step 4 — research scout (parallel, read-only, CPU-only).
                await self._maybe_enqueue_prelude_research_scout()
        elif task_kind == "replay_warm_recipe":
            # separate promote path so replay doesn't overwrite baseline_tput/current_best via the baseline branch.
            try:
                self._promote_warm_replay(result, task=task)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("warm-replay promote failed")
            # PRELUDE initial roofline was deferred while replay ran.
            await self._maybe_enqueue_prelude_initial_analysis_after_baseline()
        elif task_kind == "profile":
            # atom profiles natively now, so this skipped arm is defensive; audit as skipped + drop the gate.
            if str(result.get("status") or "") == "skipped":
                audit_decision = "skipped"
                audit_extras = {
                    "error_class": result.get("error_class"),
                    "error": result.get("error"),
                }
                if (
                    task is not None
                    and self.shared_state.auto_roofline_pending_task_id
                    == task.task_id
                ):
                    self.shared_state.auto_roofline_pending_task_id = ""
                    changed = True
            else:
                audit_decision = "promoted"
                audit_extras = {
                    "trace_path": None,
                    "profile_args": None,
                    "output_throughput": result.get("output_throughput"),
                }
            # Bug C fix: surface ProfileExecutor's trace path so Orch passes a real path to trace_analyze.
            trace_path = (
                result.get("main_trace_path")
                or (result.get("trace_files") or [None])[0]
            )
            profile_status = str(result.get("status") or "")
            if profile_status == "failed" or result.get("error_class") == "no_trace_files":
                self.shared_state.last_profile_status = "failed"
                if not trace_path:
                    self.shared_state.last_profile_trace = ""
                changed = True
            elif trace_path:
                self.shared_state.last_profile_trace = str(trace_path)
                self.shared_state.last_profile_status = "succeeded"
                # Record the server config in effect for this trace so Orch can decide whether to re-profile.
                profile_args = ""
                if task is not None:
                    profile_args = str(
                        (task.params or {}).get("base_extra_args") or ""
                    )
                self.shared_state.last_profile_args = profile_args
                # New trace invalidates the stale trace_analyze cache.
                self.shared_state.last_trace_analyze = {}
                changed = True
                audit_extras["trace_path"] = str(trace_path)
                audit_extras["profile_args"] = profile_args
            # profile result may include a tput; promote into current_best on the same +1% rule as the grid path.
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
                    "tpot_mean_ms": result.get("tpot_mean_ms"),
                    "workspace": result.get("workspace"),
                }
                if self.shared_state.baseline_tput > 0:
                    self.shared_state.cumulative_gain = (
                        (float(tput) - self.shared_state.baseline_tput)
                        / self.shared_state.baseline_tput * 100.0
                    )
                changed = True
            # On a successful profile, mirror the roofline-branch watermark handling: re-anchor
            # last_roofline_tput on the projected tput and clear the pending field for THIS task id.
            if profile_status == "succeeded":
                anchor_tput = self._current_tput_from_validated_gain()
                if anchor_tput > 0:
                    self.shared_state.last_roofline_tput = float(anchor_tput)
                    changed = True
            if (
                task is not None
                and self.shared_state.auto_roofline_pending_task_id
                == task.task_id
            ):
                self.shared_state.auto_roofline_pending_task_id = ""
                changed = True
        elif task_kind == "roofline":
            # F1-3 (Roofline-v2): the composite roofline action runs profile + trace_analyze atomically and
            # its executor already writes last_profile_* + last_trace_analyze; here we just record the audit row.
            status = str(result.get("status") or "")
            if status == "skipped":
                # Defensive arm (atom profiles natively now): clean no-op, no streak/watermark touch.
                audit_decision = "skipped"
                audit_extras = {
                    "error_class": result.get("error_class"),
                    "error": result.get("error"),
                }
                # Still clear the pending pointer so the watermark check can re-arm.
                if (
                    task is not None
                    and self.shared_state.auto_roofline_pending_task_id
                    == task.task_id
                ):
                    self.shared_state.auto_roofline_pending_task_id = ""
                    changed = True
            elif status == "succeeded":
                audit_decision = "promoted"
                # prefer the executor's published last_trace_analyze snapshot over the result dict for the audit row.
                _last_ta = self.shared_state.last_trace_analyze or {}
                audit_extras = {
                    "snapshot_id": (
                        _last_ta.get("roofline_snapshot_id")
                        if _last_ta.get("roofline_snapshot_id") is not None
                        else result.get("snapshot_id")
                    ),
                    "last_profile_trace": (
                        self.shared_state.last_profile_trace
                        or result.get("last_profile_trace")
                    ),
                    "analysis_md_path": (
                        _last_ta.get("analysis_md_path")
                        or result.get("analysis_md_path")
                    ),
                    "profile_workspace": result.get("profile_workspace"),
                    "degraded": bool(result.get("degraded", False)),
                }
                # reset the roofline failure streak on a successful snapshot (prompt-visibility only).
                if hasattr(self.shared_state, "roofline_failure_streak"):
                    self.shared_state.roofline_failure_streak = 0
                # Re-anchor the 10% watermark step on the projected current tput.
                anchor_tput = self._current_tput_from_validated_gain()
                if anchor_tput > 0:
                    self.shared_state.last_roofline_tput = float(anchor_tput)
                changed = True
            else:
                audit_decision = "discarded"
                audit_extras = {
                    "phase": result.get("phase"),
                    "error_class": result.get("error_class"),
                    "error": result.get("error"),
                }
                # bump the failure streak (mirrors the audit ledger on SharedState for prompt renderers).
                if hasattr(self.shared_state, "roofline_failure_streak"):
                    self.shared_state.roofline_failure_streak += 1
                changed = True
                log.warning(
                    "Auto-roofline %s failed (reason=%s phase=%s "
                    "error_class=%s); continuing in degraded mode "
                    "(specialists / explore proceed without a fresh "
                    "analysis_md). No retry, no fallback.",
                    task.task_id if task else "?",
                    str((task.params or {}).get("reason") or "")
                    if task is not None else "",
                    result.get("phase"),
                    result.get("error_class"),
                )
            # Clear the pending pointer (matched by task id so an unrelated roofline can't clear another's anchor).
            if (
                task is not None
                and self.shared_state.auto_roofline_pending_task_id
                == task.task_id
            ):
                self.shared_state.auto_roofline_pending_task_id = ""
                changed = True
        elif task_kind == "explore":
            # explore is the merged grid runner; the executor already did per-variant KEEP/REVERT + rebench,
            # so winners are authoritative. Coordinator is single-writer for explore_search.accepted +
            # current_best + optimization_stack and does NOT re-threshold.
            # 1. Apply the executor's ledger increment.
            update = result.get("explore_search_update")
            if isinstance(update, dict):
                self.shared_state.apply_explore_search_update(update)
                changed = True
            # 2. Search-space expansion bookkeeping (honoured defensively even though explore returns None today).
            disc_update = result.get("discovered_flags_update")
            if isinstance(disc_update, dict):
                self.shared_state.record_discovered_flags(
                    framework=str(disc_update.get("framework") or ""),
                    backend_flags=disc_update.get("backend_flags"),
                    param_flags=disc_update.get("param_flags"),
                    source_path=str(disc_update.get("source_path") or ""),
                )
                err = disc_update.get("discovery_error")
                if err:
                    self.shared_state.discovered_flags_error = str(err)
                changed = True
            # 3. Per-winner record_explore_accepted — Coordinator is sole writer of explore_search.accepted.
            winners = result.get("winners") or []
            round_id = str(result.get("round_id") or "")
            best_winner = result.get("best_variant")
            best_tput = result.get("output_throughput")
            promoted = False
            if isinstance(winners, list) and winners:
                for winner in winners:
                    if not isinstance(winner, dict):
                        continue
                    accepted = dict(winner)
                    accepted.setdefault("accepted_at_round", round_id)
                    accepted.setdefault("provenance", winner.get("provenance") or "llm_direct")
                    self.shared_state.record_explore_accepted(accepted)
                    changed = True
                # 4. Lift the best winner into current_best / optimization_stack (best_tput is post-rebench).
                if (
                    isinstance(best_winner, dict)
                    and isinstance(best_tput, (int, float))
                    and best_tput > 0
                ):
                    explore_gap_cid = (
                        str((task.params or {}).get("gap_canonical_id") or "").strip()
                        if task is not None else ""
                    )
                    self._lift_to_current_best(
                        "explore", float(best_tput), best_winner,
                        gap_canonical_id=explore_gap_cid,
                    )
                    promoted = True
                    changed = True
            try:
                self.shared_state.note_explore_outcome(promoted=promoted)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("depth: note_explore_outcome failed")
            if promoted:
                # explore inlines the per-KEEP rebench, so promote it into cumulative_gain_validated +
                # advance validated_stack_len so the TODO 4 stack-rebench guard clears immediately.
                if (
                    self.shared_state.baseline_tput > 0
                    and isinstance(best_tput, (int, float))
                    and best_tput > 0
                ):
                    validated_gain = (
                        (float(best_tput) - self.shared_state.baseline_tput)
                        / self.shared_state.baseline_tput * 100.0
                    )
                    self.shared_state.cumulative_gain_validated = float(validated_gain)
                    self.shared_state.cumulative_gain_validated_ts = (
                        datetime.now(timezone.utc).isoformat()
                    )
                    self.shared_state.cumulative_gain_validated_stack_len = len(
                        self.shared_state.optimization_stack
                    )
                    # Watermark refresh: enqueue a fresh roofline once projected tput crosses +10% over the last.
                    await self._maybe_enqueue_watermark_roofline(
                        reason="explore_keep_watermark",
                    )
            else:
                changed = True
            audit_decision = "promoted" if promoted else "discarded"
            audit_extras = {
                "round_id": round_id,
                "winners_count": (
                    len(winners) if isinstance(winners, list) else 0
                ),
                "losers_count": len(result.get("losers") or []),
                "skipped_dup_count": len(result.get("skipped_dup") or []),
                "best_variant_name": (
                    best_winner.get("name")
                    if isinstance(best_winner, dict) else None
                ),
                "best_gain_pct_vs_base": result.get("best_gain_pct"),
                "output_throughput": best_tput,
                "keep_unstable_count": len(result.get("keep_unstable_in_stack") or []),
                "explore_grid_exhausted": bool(
                    result.get("explore_grid_exhausted")
                ),
            }
        elif task_kind == "framework_pr":
            # FRAMEWORK_PR per-candidate result: append a progress row, update the batch max-gain stat, and on
            # KEEP lift to current_best + optimization_stack + cumulative_gain_validated + watermark roofline.
            status = str(result.get("status") or "")
            candidate = result.get("candidate") or {}
            cand_id = str(
                candidate.get("candidate_id")
                or candidate.get("pr_url")
                or candidate.get("ref")
                or ""
            )
            batch_id = str(result.get("batch_id") or candidate.get("batch_id") or "")
            delta_pct = result.get("delta_pct")
            new_tput = result.get("output_throughput")
            kept_flag = status == "kept"
            progress_entry = {
                "candidate_id": cand_id,
                "pr_url":       str(candidate.get("pr_url") or ""),
                "status":       status,
                "pre_tput":     float(getattr(self.shared_state, "baseline_tput", 0.0) or 0.0),
                "post_tput":    float(new_tput) if isinstance(new_tput, (int, float)) else 0.0,
                "gain_pct":     float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0,
                "kept":         kept_flag,
                "batch_id":     batch_id,
                "ts":           datetime.now(timezone.utc).isoformat(),
            }
            if not isinstance(self.shared_state.framework_pr_phase_progress, list):
                self.shared_state.framework_pr_phase_progress = []
            self.shared_state.framework_pr_phase_progress.append(progress_entry)
            # Update batch max-gain rolling stat (for the plateau judge).
            batches = getattr(self.shared_state, "framework_pr_batches", None) or []
            if isinstance(batches, list) and batches:
                for entry in reversed(batches):
                    if isinstance(entry, dict) and str(entry.get("batch_id") or "") == batch_id:
                        prev = float(entry.get("max_gain_pct_observed_in_batch") or 0.0)
                        gain = (
                            float(delta_pct) if isinstance(delta_pct, (int, float)) else 0.0
                        )
                        if gain > prev:
                            entry["max_gain_pct_observed_in_batch"] = gain
                        break
            changed = True
            if kept_flag and isinstance(new_tput, (int, float)) and new_tput > 0:
                lift = {
                    "name":              f"framework-pr:{cand_id}",
                    "variant_name":      cand_id,
                    "candidate_extra_server_args": "",
                    "extra_envs":        {},
                    "workspace":         result.get("workspace"),
                }
                self._lift_to_current_best("framework_pr", float(new_tput), lift)
                if self.shared_state.baseline_tput > 0:
                    validated_gain = (
                        (float(new_tput) - self.shared_state.baseline_tput)
                        / self.shared_state.baseline_tput * 100.0
                    )
                    self.shared_state.cumulative_gain_validated = float(validated_gain)
                    self.shared_state.cumulative_gain_validated_ts = (
                        datetime.now(timezone.utc).isoformat()
                    )
                    self.shared_state.cumulative_gain_validated_stack_len = len(
                        self.shared_state.optimization_stack
                    )
                    await self._maybe_enqueue_watermark_roofline(
                        reason="framework_pr_keep_watermark",
                    )
            audit_decision = "promoted" if kept_flag else "discarded"
            audit_extras = {
                "candidate_id":   cand_id,
                "batch_id":       batch_id,
                "status":         status,
                "delta_pct":      delta_pct,
                "output_throughput": new_tput,
                "kept":           kept_flag,
            }
        elif task_kind == "sweep":
            pareto = result.get("pareto_front") or []
            self.shared_state.record_action_attempt(
                action="sweep",
                task_id=getattr(task, "task_id", "") if task is not None else "",
                status="succeeded",
                decision="discarded",
                result=result,
                extras={
                    "grid_size": result.get("grid_size"),
                    "best_overall": result.get("best_overall"),
                    "best_for_each_conc": result.get("best_for_each_conc"),
                    "pareto_front_size": (
                        len(pareto) if isinstance(pareto, list) else None
                    ),
                },
            )
            self.shared_state.record_sweep(result)
            # Issue-E: sweep is discovery-only (never promotes) and MUST NOT mutate params_no_promote_streak.
            self.shared_state.save(self.session_dir)
            # SWEEP post-hook: chain conc_sweep after a succeeded sweep when opted in (best-effort, non-blocking).
            if getattr(self.shared_state, "conc_sweep_enabled", False) and \
                    result.get("status") == "succeeded":
                try:
                    await self._enqueue_internal_conc_sweep_task(
                        reason="post_sweep",
                    )
                except Exception:  # noqa: BLE001 — never block SWEEP->CLOSE
                    log.exception(
                        "conc_sweep: post-sweep enqueue raised (non-fatal)"
                    )
            return
        elif task_kind == "conc_sweep":
            self.shared_state.record_action_attempt(
                action="conc_sweep",
                task_id=getattr(task, "task_id", "") if task is not None else "",
                status=str(result.get("status") or "succeeded"),
                decision="discarded",
                result=result,
                extras={
                    "was_skipped":      bool(result.get("was_skipped", False)),
                    "skip_reason":      result.get("skip_reason"),
                    "budget_exhausted": bool(result.get("budget_exhausted", False)),
                    "total_budget_sec": result.get("total_budget_sec"),
                    "elapsed_sec":      result.get("elapsed_sec"),
                    "best_speedup":     (
                        (result.get("summary") or {}).get("best_speedup")
                    ),
                    "best_conc":        (
                        (result.get("summary") or {}).get("best_conc")
                    ),
                    "successful_pairs": (
                        (result.get("summary") or {}).get("successful_pairs")
                    ),
                    "report_path":      result.get("report_json_path"),
                },
            )
            # Bug #12 fix: write last_conc_sweep so exit_normal_sweep can fire conc_sweep_done without budget exhaustion.
            self.shared_state.record_conc_sweep(result)
            self.shared_state.save(self.session_dir)
            return
        # Audit trail (kernel-parity): one succeeded-attempt record with branch-supplied decision/extras.
        if audit_decision is not None and task_kind in _AUDIT_ACTIONS:
            self.shared_state.record_action_attempt(
                action=task_kind,
                task_id=getattr(task, "task_id", "") if task is not None else "",
                status="succeeded",
                decision=audit_decision,
                result=result,
                extras=audit_extras,
            )
            changed = True
        if changed:
            self.shared_state.save(self.session_dir)


__all__ = ["Coordinator", "CoordinatorState", "PendingProposal", "SharedState"]
