"""Coordinator main loop and runtime protocol manager.

The Coordinator owns the durable optimizer loop: phase transitions,
intent validation, task materialization, backend reactors, Critic and
Robustness bridges, kernel/framework handoffs, resume state, and final
artifact production. It should preserve external runtime contracts while
keeping private scheduling and helper details free to change.
"""

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..compat.payload_aliases import read_extra_server_args
from ..recipe_kb import RecipeKB, recipe_canonical_id
from ..recipe_snapshot_constants import detect_framework_version

# Recipe snapshot severity tags. The schema has no fixed enum and
# Coordinator only needs these two values.
_SEVERITY_CRASH:   str = "crash"
_SEVERITY_REGRESS: str = "regress"
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
    DYNAMIC_ACTION_NAME,
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


# Audit-trail kinds (must match shared_state._AUDIT_ACTIONS). Coordinator
# calls SharedState.record_action_attempt for these on both the
# success and failure dispatcher branches so the prompt sees a full
# audit log per non-kernel action. Kernel-owned actions intentionally
# stay outside this set — they have richer bespoke recorders
# (record_kernel_opt / record_kernel_integrate_result).
_AUDIT_ACTIONS: frozenset[str] = frozenset({
    "baseline", "profile", "sweep", "explore",
    # Composite roofline runs profile + trace_analyze atomically; each
    # invocation is visible in ``roofline_attempts``.
    "roofline",
})

# Closed enum of session_steward_specialist recommendations. Any value
# outside this set is coerced to ``stop_session`` in
# :meth:`Coordinator._route_steward_verdict` (defense in depth — the
# LLM can write any string but only the enum drives a phase-routing
# change).
_STEWARD_RECS: frozenset[str] = frozenset({
    "continue_explore", "advance_to_kernel", "stop_session",
})
# Retries for session_steward subprocess / LLM transport failures only.
# Empty or malformed *LLM* verdicts still coerce to stop_session.
_STEWARD_MAX_INFRA_RETRIES: int = 3

# Actions whose output should observe the freshest ``analysis.md`` /
# ``last_profile_trace`` snapshot. Every dispatch path
# (``_handle_delegate`` / ``_handle_propose_action`` / the post-Critic
# ``_materialize_approved_proposal``) consults
# :meth:`Coordinator._auto_roofline_pending_denial` for these and defers
# the dispatch while a Coordinator-internal roofline/profile task is
# still in flight. The field is set by the PRELUDE bootstrap + the
# +10% watermark crossing and cleared in
# :meth:`Coordinator._promote_to_shared_state` once the analysis task
# lands.
_ROOFLINE_GATED_ACTIONS: frozenset[str] = frozenset({
    "specialist",
    "explore",
    "kernel_opt",
    "integrate",
    "deep_kernel_analysis",
    "operator_tuning",
    "vendor_kernel_config",
})


# Default per-repo candidate cap for ``fa phase-discover`` during the
# FRAMEWORK_PR phase. Higher than the historical implicit 5 so each batch
# probes deeper; overridable per session via
# ``SharedState.framework_pr_max_candidates``.
DEFAULT_FRAMEWORK_PR_MAX_CANDIDATES: int = 8


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
    # Per-variant verdict map kept for the ``review_verdict`` envelope
    # schema + resume replay of historical sessions. Proposals are now
    # decided by a single verdict (``_handle_review_verdict`` collapses
    # any map to a summary), so no live writer populates this; it stays
    # empty on the in-memory path.
    verdict_map: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Always-empty dict kept for forward compat: no producer writes it,
    # but callers propagate it across defer/restore (deferred_queue +
    # resume) so the dataclass shape stays stable.
    kb_edge_ids: dict[str, str] = field(default_factory=dict)


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
        self.session_dir = Path(session_dir)
        self.role_registry = role_registry or default_role_registry()
        # Recipe-snapshot KB dispatcher.  When ``None`` (legacy cli
        # path or ``--degraded-kb`` with a missing local store
        # construction) the fact-write hooks become no-ops; the rest
        # of the Coordinator behaves identically.  The dispatcher is
        # thread-safe (LocalRecipeStore uses per-cid flock + thread
        # mutex) so it can be shared across the single-event-loop
        # Coordinator.
        # Field name kept as ``cortex_kb`` for grep stability with the
        # ``cortex_kb=`` kwarg in CLI / SDK callers; type is RecipeKB.
        self.cortex_kb: RecipeKB | None = cortex_kb
        # Per-session optimization journal (optimization_journal.py).
        # Lazy-instantiated on first use so SharedState already has
        # model/hardware/framework. Survives ``--degraded-kb`` (local-only).
        self._journal: Journal | None = None
        # Warm-recipe replay controls (PRELUDE auto-apply of the KB
        # best_config). ``warm_replay_enabled=False`` renders the
        # warm_start_recipe into prompts but never auto-runs it; the
        # threshold / confidence knobs tune the fire / drift rule.
        self._warm_replay_enabled: bool = bool(warm_replay_enabled)
        self._warm_replay_min_confidence: float = float(warm_replay_min_confidence)
        self._warm_replay_min_reproduce_pct: float = float(warm_replay_min_reproduce_pct)
        # KnowledgePlane facade.
        # When non-None, ``_handle_delegate`` pre-warms PR feed plus
        # advisory knowledge context for ``delegate{action='specialist'}``
        # tasks before enqueue. ``None`` means no warmup; specialists
        # still run with empty advisory context.
        self.knowledge_plane: Any = knowledge_plane
        # ProposalScorer facade (advisory). When non-None, a non-empty
        # specialist ``proposal_set`` is scored in
        # ``_record_specialist_result``; scores ride on the round entry
        # under ``ensemble_scores`` as one reference among many. ``None``
        # disables scoring. Never gates anything (advisory only).
        self._proposal_scorer: Any = proposal_scorer
        # Phase budget percentages. ``None`` means library defaults; CLI
        # flags populate this from ``--max-minutes-<phase>-pct``. Normalised
        # once at construction so downstream judges see a complete dict.
        self._phase_budget_pct: dict[str, float] = _phase_state.normalize_budget_pct(
            phase_budget_pct
        )
        # Specialist stale scan threshold (seconds). Robustness uses this
        # to surface domains that stopped producing usable proposals.
        try:
            self._specialist_stale_sec: float = max(
                0.0,
                float(os.environ.get(
                    "INFERENCE_OPTIMIZER_SPECIALIST_STALE_SEC", "600",
                )),
            )
        except ValueError:
            self._specialist_stale_sec = 600.0
        # External launcher configuration. ``target_analysis`` always
        # writes an artifact; ``compare_against_gpu`` decides whether it
        # fetches real InferenceX rows or records a no-target marker.
        self._compare_against_gpu: str = (compare_against_gpu or "").strip()
        self._model_class_override: str = (model_class or "").strip()

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
        self.sub = sub_agent_runner or SubAgentRunner(
            self.locks, self.tasks, session_dir=self.session_dir,
        )

        # Persistent session state (state.json) — load existing for resume;
        # save() is called whenever the Coordinator mutates a persistent field.
        self.shared_state = SharedState.load_or_init(self.session_dir)
        self.gpu_specialist_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_gpu_specialist_devices(
                int(getattr(self.shared_state, "gpu_specialist_capacity", 0) or 0)
            ),
        )
        # sync research_lane capacity from SharedState into
        # the lane_capacity table. CLI/manifest
        # flow already pinned the value onto SharedState at session
        # start; this propagates it into the SQLite source of truth so
        # ``acquire_many`` honours the operator-set cap. Idempotent:
        # the default is already 1 from ``ensure_schema``; we only
        # overwrite when the operator picked something non-default.
        try:
            from ..storage.schema import set_lane_capacity as _set_lane_capacity
            cap = int(self.shared_state.research_lane_capacity or 0)
            if cap >= 0:
                _set_lane_capacity(self.db.raw, "research_lane", cap)
        except Exception:  # noqa: BLE001 — non-fatal; default seed wins
            log.exception("failed to sync research_lane_capacity to leases DB")
        # `strict_paths` defers to the env flag (CLI flips this on for
        # production; tests omit the env so the path check stays off and
        # legacy `/tmp/<fixture>` payload values still pass).
        self.policy = PolicyGate(
            role_registry=self.role_registry,
            session_dir=self.session_dir,
            shared_state=self.shared_state,
        )
        # Resume detection must run before any boot-time state.json write;
        # otherwise a fresh session that only inferred model_class would look
        # like a resume (state_path.exists() → is_resume=True).
        self._resumed_from = self._detect_resume_state()
        # External launchers may provide model_class via flag/env. If not,
        # derive it once at boot; never overwrite a resumed session's value.
        # Persist later via ``_ensure_phase_initialised`` so fresh sessions
        # are not misclassified as resume.
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
        # Critic-approved proposals deferred because an auto-roofline /
        # auto-profile task was still in flight when
        # :meth:`_materialize_approved_proposal` ran. Drained when the
        # analysis task lands (both the ``profile`` and ``roofline``
        # branches in :meth:`_promote_to_shared_state` call
        # :meth:`_drain_proposals_awaiting_roofline`).
        self._proposals_awaiting_roofline: list[
            tuple[PendingProposal, set[str] | None]
        ] = []

        # Per-agent consecutive ``BackendError`` streak. Successful turns
        # reset the counter for that agent; a streak crossing
        # ``_backend_error_streak_threshold`` records a single
        # ``backend_unhealthy`` observation so operators (and the
        # robustness reactor, which tails Coordinator events) notice the
        # subprocess transport is degraded — particularly relevant for
        # the robustness-agent / critic-agent subprocess backends whose
        # in-loop failures otherwise would only show up as scattered
        # ``backend_error`` events. The escalation observation fires once
        # per crossing, then the counter must reset and re-arm before it
        # can fire again, so we never spam the inbox.
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

        # Stable tick order derived from the live role_registry. Must NOT
        # use the module-level `roles_for_run()` which is a cached hardcoded
        # tuple containing "kernel" even when --no-kernel stripped it.
        _CANONICAL_ORDER = ("orchestration", "kernel", "critic", "robustness")
        self._tick_roles: tuple[str, ...] = tuple(
            r for r in _CANONICAL_ORDER if r in self.role_registry
        )

        # Action registry — small yaml catalogue used by PolicyGate /
        # prompt rendering to map action_name → metadata (phase, family,
        # pipeline_phase). A load failure falls back to ``None``; downstream
        # callers handle a missing registry gracefully.
        try:
            self.action_registry: ActionRegistry | None = ActionRegistry().load()
        except Exception:  # noqa: BLE001 — defensive; missing yaml shouldn't kill the run.
            log.exception("Coordinator: failed to load ActionRegistry.")
            self.action_registry = None
        # Wall-clock budget tracking for per-tick Time-budget prompt injection.
        self._run_deadline: float | None = None
        self._run_started_monotonic: float | None = None
        # Latest objective wired by ``Coordinator.run()``. Used by
        # ``_compose_prompt`` to refresh ``shared_state.target_gap_pct`` on
        # every Orchestration tick. None outside a run (e.g. bounded tick()
        # tests).
        self._current_objective: Objective | None = None

        # initialise phase machine. Fresh session enters PRELUDE.
        # Idempotent: second construction on the same session_dir, and
        # same-version resume of an already-initialised state, are no-ops.
        self._ensure_phase_initialised()
        # Cortex T0 defensive fallback. The cli
        # is the canonical T0 entry point (fail-fast banner +
        # sys.exit on Cortex outage); this hook only fires for
        # SDK / integration-test callers that constructed the
        # Coordinator directly. No-op when cortex_kb is None /
        # disabled / sid already set. Best-effort: helper logs a
        # warning + leaves warm_start empty on Cortex failure rather
        # than crashing the long-running reactor.
        self._ensure_cortex_t0_anchored()

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

        We also rebuild :attr:`_proposals_awaiting_roofline` from
        ``proposal_materialize_blocked`` observations whose proposal_msg_id
        never paired with a later ``approved_proposal`` decision — these
        are Critic-approved proposals that were deferred by the
        analysis gate at process shutdown and would otherwise be lost
        (the in-memory deque does not survive a restart, and the
        verdict already marks them as ``decided`` so they get dropped
        from ``pending_proposals``).
        """
        # 1. Collect all proposal events.
        proposal_msgs = await self.bus.tail(topic="proposal", n=10_000)
        # 2. Collect verdicts and approved decisions, keyed by proposal_msg_id.
        verdicts = await self.bus.tail(topic="review_verdict", n=10_000)
        decisions = await self.bus.tail(topic="decision", n=10_000)
        observations = await self.bus.tail(topic="observation", n=10_000)

        decided_ids: set[str] = set()
        verdict_by_target: dict[str, str] = {}
        approved_verdict_ids: set[str] = set()
        for v in verdicts:
            target = v.payload.get("target_proposal_msg_id")
            if not target:
                continue
            # Historical sessions may carry per-variant ``verdict_map``
            # events without a summary ``verdict`` string. Keep this
            # rebuild backward-compatible by synthesising a
            # ``needs_review`` placeholder so the proposal still counts
            # as decided and the prompt's /status surface shows something.
            summary = v.payload.get("verdict") or ""
            if not summary and isinstance(v.payload.get("verdict_map"), dict):
                summary = "needs_review"
            verdict_by_target[target] = summary
            decided_ids.add(target)
            # Track "approved" verdicts so the deferred-queue rebuild
            # below can sanity-check it only acts on Critic-approved
            # proposals (reject verdicts are correctly final).
            sl = str(summary).strip().lower()
            if sl in ("approve", "approved") or sl.startswith("approve") or (
                isinstance(v.payload.get("verdict_map"), dict)
                and any(
                    str(vv).strip().lower().startswith("approve")
                    for vv in (v.payload.get("verdict_map") or {}).values()
                )
            ):
                approved_verdict_ids.add(target)

        drained_proposal_ids: set[str] = set()
        for d in decisions:
            if d.payload.get("kind") == "approved_proposal":
                pid = d.payload.get("proposal_msg_id") or ""
                if pid:
                    drained_proposal_ids.add(pid)

        # Latest-wins map of materialize-blocked observations keyed by
        # proposal_msg_id. ``bus.tail`` returns rows DESC by seq so
        # the FIRST occurrence of a given proposal_msg_id is the
        # newest — use ``setdefault`` to keep that one and ignore
        # older blocked observations (which would otherwise overwrite
        # the freshest ``approved_variant_names`` + ``kb_edge_ids``).
        blocked_by_pid: dict[str, dict[str, Any]] = {}
        for o in observations:
            pl = o.payload or {}
            if pl.get("kind") != "proposal_materialize_blocked":
                continue
            pid = pl.get("proposal_msg_id") or ""
            if pid:
                blocked_by_pid.setdefault(pid, pl)

        # 3. Rebuild PendingProposal entries for undecided proposals.
        rebuilt = 0
        self.state.pending_proposals.clear()
        proposal_by_id: dict[str, Any] = {}
        for p in proposal_msgs:
            proposal_by_id[p.msg_id] = p
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

        # 4. Rebuild the deferred-materialize queue. A proposal belongs
        # here when it was Critic-approved, surfaced a
        # ``proposal_materialize_blocked`` observation, and has no
        # subsequent ``approved_proposal`` decision proving the drain
        # already dispatched it. Without this rebuild the in-memory
        # deque is empty on restart and the original Critic verdict
        # would be permanently silenced (the proposal_msg_id is in
        # ``decided_ids`` so step 3 above also drops it from
        # ``pending_proposals``).
        self._proposals_awaiting_roofline = []
        deferred_restored = 0
        for pid, blocked_payload in blocked_by_pid.items():
            if pid in drained_proposal_ids:
                continue
            if pid not in approved_verdict_ids:
                # Defensive: a blocked-without-approve combination
                # should not occur (materialize only runs after the
                # Critic approves) but if it does, skip rather than
                # re-dispatch a never-approved proposal.
                continue
            src_msg = proposal_by_id.get(pid)
            if src_msg is None:
                continue
            payload = src_msg.payload or {}
            pending = PendingProposal(
                proposal_msg_id=src_msg.msg_id,
                from_agent=src_msg.from_agent,
                action_name=str(payload.get("action_name", "")),
                predicted_gain_pct=float(payload.get("predicted_gain_pct", 0.0)),
                payload=dict(payload),
            )
            kb_edges = blocked_payload.get("kb_edge_ids")
            if isinstance(kb_edges, dict):
                pending.kb_edge_ids = {
                    str(k): str(v) for k, v in kb_edges.items() if v
                }
            approved_names_list = blocked_payload.get("approved_variant_names")
            approved_set: set[str] | None
            if isinstance(approved_names_list, list):
                approved_set = {str(n) for n in approved_names_list}
            else:
                approved_set = None
            self._proposals_awaiting_roofline.append((pending, approved_set))
            deferred_restored += 1

        self._resumed_from["rebuilt"] = True
        self._resumed_from["pending_restored"] = rebuilt
        self._resumed_from["deferred_restored"] = deferred_restored
        return {
            "is_resume": self._resumed_from["is_resume"],
            "event_count": self._resumed_from["event_count"],
            "state_json_present": self._resumed_from["state_json_present"],
            "pending_restored": rebuilt,
            "deferred_restored": deferred_restored,
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
        # safety net (only fires when the CLOSE phase sequencer
        # did NOT get to run — e.g. Ctrl-C / crash mid-EXPLORE).
        # Runs the recipe / journal finalize. The SQLite close still
        # runs afterwards so we don't leak fds.
        await self._cortex_t4_hook()
        self.db.close()

    async def _cortex_t4_hook(self) -> None:
        """T4 — finalize recipe at session end.

        Called from :meth:`stop` once the reactor / dispatcher loops
        have torn down. Acts as a safety net for the crash / Ctrl-C
        path where the CLOSE phase sequencer did not get to run.

        When the CLOSE phase sequencer ran
        (``close_sequence_done=True``), it already called
        ``cortex_finalize_recipe_and_journal`` inline. We
        early-return so the hook is a no-op in that case — the
        journal is already finalised and the recipe row already
        carries this session.

        Under the v2 RecipeKB design writes are local-only and the
        legacy NDJSON pending-queue / drain_pending() path is
        retired. The only T4 action is the safety-net recipe
        finalize.
        """
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

    # ==================================================================
    # phase state machine
    # ==================================================================
    def _ensure_phase_initialised(self) -> None:
        """Set ``phase`` + persist ``phase_budget_pct`` once per session.

        Idempotent: resume / re-construction skips both writes when the
        phase machine has already been initialised. Persists immediately
        so a crash between construction and the first tick still leaves
        a usable phase_history.
        """
        state = self.shared_state
        # Phase budget always normalised so the judge has a complete map.
        # Persisted regardless because the operator-passed CLI flags need
        # to land in state.json for resume parity.
        if not state.phase_budget_pct:
            state.phase_budget_pct = dict(self._phase_budget_pct)
        current = (state.phase or "").strip().upper()
        if current in _phase_state.PHASE_NAMES:
            # Already initialised. Keep CLI-side budget override the
            # latest authoritative value.
            state.phase_budget_pct = dict(self._phase_budget_pct)
            try:
                state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: save after phase budget refresh failed")
            return
        # Fresh start. A resume whose state.json never initialised the
        # phase machine (pre-phase-machine session) is treated as fresh:
        # cross-version resume of such state is not supported.
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
        """v0.8 KB_gaps/Gap-12 — defensive T0 anchor for SDK callers.

        The cli is the canonical T0 entry point (see
        ``cli._bootstrap_cortex_kb``); it runs T0 *before* the
        Coordinator is constructed and threads the resulting
        :class:`CortexKBClient` (with its sid + warm_start already on
        SharedState) into our constructor. This method is a *fallback*
        — it does the same T0 ritual when an SDK / integration-test
        caller constructs a :class:`Coordinator` directly without the
        cli plumbing.

        The hook is intentionally narrow:

        * Skip when ``cortex_kb`` is ``None`` (legacy callers
          who don't even pass a client).
        * ``--degraded-kb`` does NOT skip here: under the v2 design it
          still writes the local store (``RecipeKB.enabled`` is always
          True); only ``cortex_kb=None`` opts out of T0 entirely.
        * Skip when ``shared_state.cortex_session_id`` is already
          non-empty (the cli already ran T0, or a resume picked up
          the prior sid).

        On Cortex failure (``CortexKBError`` / binary missing),
        :func:`run_t0_anchor` with ``fail_fast=False`` logs a warning
        and leaves warm_start empty — the Coordinator never blocks
        its boot on Cortex availability when it's running outside
        cli's fail-fast contract.
        """
        client = self.cortex_kb
        if client is None or not getattr(client, "enabled", True):
            return
        state = self.shared_state
        if (state.cortex_session_id or "").strip():
            # Either cli already T0'd or resume picked up the sid.
            # The helper's `skipped_already` short-circuit would
            # behave the same, but we gate up here so we don't even
            # import the helper for the common cli path.
            return
        # Derive workload / hw from SharedState (cli has the same
        # fallback chain, but SDK callers are expected to seed
        # ``model_name`` + ``gpu_type`` themselves before
        # constructing the Coordinator).
        workload = (
            getattr(state, "model_name", "") or "unknown_model"
        )
        hw = getattr(state, "gpu_type", "") or "unknown_gpu"
        # ``marathon_dispatch_id`` mirrors the cli path: it's the
        # hyperloom-internal manifest session id (state.session_id is
        # the same value when populated from manifest).
        extra_attrs = {
            "marathon_dispatch_id": getattr(state, "session_id", "") or "",
            "framework":   getattr(state, "framework", "") or "",
            "model_class": getattr(state, "model_class", "") or "",
            "claw_session_id":  getattr(state, "claw_session_id", "") or "",
            "sandbox_user_id":  getattr(state, "sandbox_user_id", "") or "",
            # ``boot_origin`` is a dev-debug label, NOT written to KB.
            # It's accepted by run_t0_anchor's whitelist filter and
            # ignored — kept here so log lines can distinguish the
            # SDK-fallback path from the cli-canonical path.
            "boot_origin": "coordinator_fallback",
        }
        try:
            # Reuse the dispatcher the Coordinator already holds so T0
            # anchors the SAME local store that the KEEP/REVERT/CLOSE
            # writes target. (A throwaway local-only dispatcher here
            # used to risk pointing at a different root than
            # ``self.cortex_kb`` — operator ``--cortex-kb-url`` /
            # ``--local-kb-root`` are already baked into ``client``.)
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
        # Mirror the persisted ``kernel_enabled`` flag — CLI's
        # ``--no-kernel`` removes the kernel role; resume picks the
        # value from state.json.
        return "kernel" in self.role_registry and bool(
            getattr(self.shared_state, "kernel_enabled", True)
        )

    def _explore_enabled(self) -> bool:
        # Mirror the persisted ``explore_enabled`` flag — CLI's
        # ``--no-explore`` collapses PRELUDE / FRAMEWORK_PR straight to
        # KERNEL (or SWEEP when --no-kernel is also set). EXPLORE is a
        # phase, not a role, so there is no role-registry gate here.
        return bool(getattr(self.shared_state, "explore_enabled", True))

    async def _advance_phase_if_needed(self) -> None:
        """Scan exit conditions and transition phase at most once per tick.

        Priority order is encoded in :func:`phase_state.compute_next_phase`:
        ``abort > exit_terminal > exit_normal`` (Inv-8.2).  When a
        transition fires we:

        1. Append a phase_history row (built via
           :meth:`SharedState.record_phase_transition`).
        2. Mirror the transition onto the bus as an ``event`` so resume
           replays it and Cortex-side observers (M5+) see it.
        3. Save state.json atomically so a crash between two ticks
           doesn't lose the boundary.
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
            # Default is True to match SharedState.framework_phase_enabled
            # and the CLI resume fallback at cli.py:3231 (which reads
            # ``getattr(state, "framework_phase_enabled", True)``). The
            # old ``False`` fallback here disagreed with both sites, so a
            # resumed session whose state.json predated the field would
            # silently skip FRAMEWORK_PR for one call only — confusing
            # to debug.
            framework_phase_enabled=bool(
                getattr(state, "framework_phase_enabled", True)
            ),
            explore_enabled=self._explore_enabled(),
            max_hours=max_hours_arg,
        )
        if str(state.phase or "").upper() == "EXPLORE":
            await self._maybe_enqueue_explore_research_scout()
        if next_phase is None:
            # On EXPLORE plateau without a steward verdict,
            # compute_next_phase returns None and this schedules the
            # steward; the next tick routes using its verdict.
            await self._maybe_enqueue_steward()
            return
        target, reason, evidence = next_phase
        if target == (state.phase or "").upper():
            return  # already there
        prior = state.phase
        # when the transition was driven by an escalate
        # hint, consume it so the next tick re-evaluates against
        # fresh signals. The phase_state module
        # already wrote the hint into ``evidence["hint"]`` if it
        # mattered, so this is just a cleanup step.
        if isinstance(evidence, dict) and (
            evidence.get("evidence") == "llm_escalation"
            or "hint" in evidence
        ):
            state.consume_pending_escalate_hint()
        # When a terminal transition (target=CLOSE) fires from a
        # vocab stop_reason that isn't already on the state, mirror
        # it onto state.stop_reason via the ENUM-validated writer so
        # the next run() tick winds the loop down (skip_to_close path).
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
        # phase-entry side effects. Side effects are
        # *additive* — failures inside a hook are logged but never
        # roll back the transition. Keeping the dispatch table inside
        # ``_on_phase_entered`` so the per-phase branches stay together
        # and future phase entries (KERNEL auto-profile, SWEEP grid,
        # CLOSE sequencer) only need one new branch each.
        try:
            await self._on_phase_entered(from_phase=prior or "", to_phase=target)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: _on_phase_entered hook failed")

    async def _on_phase_entered(self, *, from_phase: str, to_phase: str) -> None:
        """Fire per-phase entry side effects.

        Pure dispatcher: each branch is a thin wrapper around a
        ``_on_enter_<phase>`` method. Hooks are infallible from the
        caller's perspective (callers should not need to wrap them in
        try/except — the dispatcher itself catches and logs).

        Currently wired:

        * ``FRAMEWORK_PR`` — start the per-candidate pump that calls
          ``fa phase-discover`` for batches, routes each candidate
          through the Critic gate
          (:meth:`_critic_review_framework_pr_candidate`), and
          enqueues a ``framework_pr`` task per approved candidate
          until the exit predicate fires (plateau / force-exit /
          discover retries exhausted).
        * ``EXPLORE`` — pre-warm PR feed across every specialist
          domain. The first specialist dispatch after EXPLORE entry
          then sees a populated cache rather than a cold
          ``pr_monitor`` fetch.
        * ``KERNEL`` — no-op (roofline lives in PRELUDE +
          watermark-driven mid-run refresh).
        * ``SWEEP`` — auto-enqueue ``sweep`` with a recipe-driven
          (or defaults-driven) grid so SWEEP doesn't degrade to
          "LLM 自觉发 sweep".
          Same idempotency contract as KERNEL.
        * ``CLOSE`` — run the 5-step closing sequencer (report →
          session_breakdown → NDJSON drain → Cortex commit → mark
          done; KB_design §3.2 §5.5 + KB_gaps/Gap-06). Sets the
          ``state.close_sequence_done`` flag so ``cli.finally``
          short-circuits its emergency breakdown write.

        All five non-PRELUDE phases with side effects are wired.
        Hook additions for new phases should slot into this
        dispatcher table.
        """
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
        """Warm ``KnowledgePlane.pr_feed`` across all specialist
        domains so subsequent ``_warm_specialist_params`` calls hit
        the cache. Best-effort: any failure is logged + the run
        continues. A downstream specialist dispatch still calls
        :meth:`KnowledgePlane.pr_feed_warm` individually
        (``_handle_delegate`` → ``_warm_specialist_params``), so even
        a hard failure here only loses the upfront cache-priming.

        Roofline lives in PRELUDE (auto-enqueued after baseline) and
        re-fires whenever ``cumulative_gain_validated`` crosses the
        10% watermark over ``last_roofline_tput`` — see
        :meth:`_needs_roofline_for_watermark`. EXPLORE entry no
        longer enqueues roofline.

        Resets the per-round ``dynamic_action`` cap counter so a fresh
        EXPLORE entry restores the ``MAX_DYNAMIC_PER_ROUND`` budget.
        """
        try:
            self.shared_state.reset_dynamic_action_round_count()
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "EXPLORE entry: reset_dynamic_action_round_count failed",
            )
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
        """FRAMEWORK_PR entry hook.

        Triggers the per-batch pump once on entry; subsequent batches
        are driven by :meth:`_pump_framework_pr_phase` invoked from the
        main tick. Best-effort — any failure logs and the run continues
        (``compute_next_phase`` will eventually force-exit via the
        wall-clock guard if the phase never makes progress).
        """
        log.info(
            "FRAMEWORK_PR entry (from=%s): pumping initial batch",
            from_phase or "<unknown>",
        )
        try:
            await self._pump_framework_pr_phase()
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning("FRAMEWORK_PR entry pump failed: %r", exc)

    async def _pump_framework_pr_phase(self) -> None:
        """Drive the FRAMEWORK_PR phase: enqueue the next candidate.

        Invoked from :meth:`_on_enter_framework_pr` and on each phase
        tick while ``state.phase == PHASE_FRAMEWORK_PR``. Idempotent —
        re-entrant calls are no-ops while a ``framework_pr`` task is
        already in flight or the phase has been marked done.

        The pump is best-effort: any failure to call ``fa
        phase-discover`` flips ``framework_pr_phase_done = True`` so the
        next ``compute_next_phase`` advances to EXPLORE rather than
        wedging here.
        """
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
        # Find the next un-dispatched candidate in the current batch, or
        # request a new batch if the previous one is exhausted.
        next_candidate = self._select_next_framework_pr_candidate()
        if next_candidate is None:
            # Try to discover a fresh batch. A transient discover
            # failure (timeout / error) does NOT immediately collapse
            # the phase; only ``DISCOVER_FAILURE_RETRY_LIMIT`` consecutive
            # failures, or an empty-but-valid payload (genuine "no more
            # PRs"), mark the phase done. See _discover_next_framework_pr_batch
            # for the failure-counter semantics.
            from . import framework_agent_client as _fa_client
            ok = await self._discover_next_framework_pr_batch()
            if not ok:
                failures = int(
                    getattr(state, "framework_pr_discover_failures", 0) or 0
                )
                if failures >= _fa_client.DISCOVER_FAILURE_RETRY_LIMIT or failures == 0:
                    # Either we've exhausted retries (failures >= limit),
                    # or the call returned a clean empty payload
                    # (counter was reset to 0). Both are real exits.
                    # Stamp a summary row so the final phase_done flip is
                    # visible alongside per-attempt discover failures.
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
        # Critic gate before apply. The Critic sees PR metadata
        # (diff URL + title + gap target) and returns an
        # ``approve`` / ``reject`` verdict. ``reject`` short-circuits
        # the candidate with a ``critic_denied`` progress row so the
        # apply / bench round is never spent on a candidate the Critic
        # already classified as out-of-scope or unsafe. ``approve`` (and
        # the degraded ``abstain`` returned when no Critic backend is
        # wired) falls through to the enqueue.
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

    def _select_next_framework_pr_candidate(self) -> dict[str, Any] | None:
        """Return the next unprocessed candidate in the latest batch.

        A candidate is "processed" iff
        ``framework_pr_phase_progress`` carries a matching
        ``candidate_id`` entry.
        """
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
        """All candidate ids already discovered into any prior batch.

        Used by :meth:`_discover_next_framework_pr_batch` to drop PRs that
        an earlier batch (or a different repo in the same cross-repo scan)
        already produced, so each new batch only carries genuinely new
        candidates.
        """
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
        # Fold in PR ids the research scout already mined so the two
        # mechanisms never re-process the same PR.
        for pid in getattr(state, "research_scout_seen_pr_ids", None) or []:
            pid = str(pid or "").strip()
            if pid:
                ids.add(pid)
        return ids

    def _framework_pr_tried_refs(self) -> list[str]:
        """Refs already discovered this phase (fed to ``compose_gap`` so it
        can bias the gap away from previously-surfaced PR categories)."""
        refs: list[str] = []
        for cid in self._framework_pr_known_candidate_ids():
            if cid:
                refs.append(cid)
        return refs

    def _framework_pr_discover_repo_urls(self, framework: str) -> list[str]:
        """Repo URLs to query for the FRAMEWORK_PR batch.

        The framework's own repo leads (primary), followed by every other
        repo the ``pr_intel_specialist`` domain tracks (cross-repo top-up).
        This is the same repo set EXPLORE's pr_intel_specialist surveys, so
        the two phases share the discovery surface rather than
        FRAMEWORK_PR owning it privately. Duplicate / empty URLs are
        dropped while preserving order.
        """
        from . import framework_agent_client as _fa_client

        urls: list[str] = []

        def _add(u: str) -> None:
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
        """Append a single ``framework_pr_phase_done`` row to
        ``phase_history`` describing why the pump gave up.

        Per-attempt ``framework_pr_discover_failed`` rows already cover
        each individual error; this summary row makes the final give-up
        decision explicit.
        """
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
        """Call ``fa phase-discover`` and append a batch to SharedState.

        Returns True iff a non-empty batch was appended. Transient
        failures (timeout, non-zero exit, parse error) return False
        without flipping ``framework_pr_phase_done``; the caller
        consults ``framework_pr_discover_failures`` to decide when to
        finally give up — see ``DISCOVER_FAILURE_RETRY_LIMIT``.
        Successful calls (including empty-but-valid responses) reset
        the failure counter so an intermittent error never accumulates.
        """
        from . import framework_agent_client as _fa_client

        state = self.shared_state
        # Directed gap composition: seed the search from the latest
        # profile bottleneck + workload taxonomy (model_class / precision /
        # gpu_type) via compose_gap, then merge any structured
        # ``state.gaps`` rows on top so a fresh batch is re-targeted at the
        # current bottleneck rather than always re-querying the same
        # static gap text.
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
            # Prepend the directed gap so fa's search leads with the
            # bottleneck-aware phrasing; de-dup against any identical
            # structured gap text already present.
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
        # Cross-repo: query every repo the pr_intel_specialist tracks so
        # FRAMEWORK_PR discovery is not confined to the single framework
        # repo. The framework's own repo leads the list (primary), the
        # rest are cross-repo top-ups (aiter / triton / rccl / the other
        # serving framework). This is the same repo set EXPLORE's
        # pr_intel_specialist surveys — neither phase owns it privately.
        repo_urls = self._framework_pr_discover_repo_urls(framework)
        payload: dict[str, Any] | None = None
        merged_candidates: list[dict[str, Any]] = []
        batch_id = ""
        any_call_ok = False
        last_exc: Exception | None = None
        # Spread the phase timeout across the repos so a multi-repo scan
        # cannot blow the whole budget on the first slow repo.
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
        # Cross-batch + cross-repo de-dup: a candidate already discovered
        # in an earlier batch (regardless of which repo surfaced it) is
        # dropped here so the new batch only carries genuinely new PRs.
        # Without this, the cross-repo loop or a re-query could re-list a
        # PR that another repo's serving overlap already produced, padding
        # the batch with duplicates that the plateau judge then counts.
        seen_ids = self._framework_pr_known_candidate_ids()
        primary_repo_url = repo_urls[0] if repo_urls else ""
        # Normalise each candidate so the executor's slug helper has
        # consistent fields and the progress ledger has a stable id.
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
            # Stamp the repo URL the candidate came from so the executor's
            # checkout-head path knows whether the PR lives in the live
            # framework_root's origin (same-repo, fetchable) or a foreign
            # repo (must fall back to diff_url). fa already returns a
            # ``repo_url`` per candidate when known; otherwise fall back to
            # the batch's primary repo URL.
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
        """Enqueue a single ``framework_pr`` task for ``candidate``."""
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
            # Record an ``enqueue_failed`` progress row so the pump's
            # ``_select_next_framework_pr_candidate`` skips this
            # candidate on the next tick rather than retrying it
            # forever. Without this row the candidate_id stays out of
            # the ``processed`` set and the loop spins.
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
        """Return compact session-local priors for the Critic gate.

        Includes:
        - ``recent_decisions``: last N rows from
          ``framework_pr_critic_decisions`` (excluding the candidate
          currently under review, since it has no row yet) so the
          Critic sees its own recent verdicts and can stay consistent.
        - ``recent_outcomes``: last N rows from
          ``framework_pr_phase_progress`` filtered to terminal
          statuses (``kept``, ``reverted``, ``no_patch``,
          ``enqueue_failed``) so the Critic sees what the apply/bench
          pipeline actually did with previously-approved candidates.

        Best-effort: shape mismatches degrade to empty lists; the
        Critic's prompt path must not crash if SharedState evolves.
        """
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

        Returns ``{"verdict": "approve"|"reject"|"abstain", "rationale": str}``.

        - ``abstain`` is the safe degraded path: returned when no
          Critic backend is wired (test harnesses), when the backend
          raises, or when it emits no ``REVIEW_VERDICT`` intent. The
          caller treats ``abstain`` the same as ``approve`` so a
          missing Critic does not silently block the whole phase.
        - Decisions are cached in ``state.framework_pr_critic_decisions``
          keyed by ``candidate_id`` so a resumed session does not
          re-call the Critic for candidates it already classified.
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
        # Build a proposal-formatted prompt that both MockCriticBackend
        # (regex-driven REVIEW_VERDICT emission) and CriticAgentBackend
        # (passes the whole prompt body through to ``prepare-review``)
        # can consume. The ``msg_id`` is deterministic from the
        # candidate id so MockCriticBackend's dedupe set is consistent.
        # All-hex msg_id so MockCriticBackend's ``[a-f0-9]+`` regex
        # captures it cleanly. Prefix would break the parse.
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
            # Session-local priors — every framework_pr candidate
            # already classified plus the apply/bench outcomes from
            # this session. Lets the Critic spot patterns like "the
            # last 3 perf PRs from this repo all crashed at startup"
            # without standing up a separate KB query. Bounded to the
            # tail so the prompt stays compact.
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
                # Map the Critic verdict vocab onto the gate's
                # {approve, reject, abstain}. ``redirect`` / ``advise`` /
                # ``needs_review`` all fall through to abstain so the
                # phase keeps moving instead of stalling.
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
        """Run deterministic KERNEL-entry setup before LLM kernel work.

        Roofline is auto-enqueued at PRELUDE (initial) and on every
        10% watermark crossing of ``last_roofline_tput`` — see
        :meth:`_maybe_enqueue_watermark_roofline`. The KERNEL phase
        no longer needs an entry-time profile anchor: the watermark
        refresh keeps ``analysis.md`` aligned with stack progress,
        and ``trace_analyze`` / ``kernel_opt`` read
        ``last_profile_trace`` written by the same roofline executor.
        """
        if not self._kernel_enabled():
            # Should not happen — compute_next_phase routes
            # --no-kernel runs straight EXPLORE → SWEEP.
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
        """Promote a successful GEMM tuning run into the main gain ledger."""
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
        # The GEMM action's own tuned benchmark is an end-to-end serving run,
        # so it is already a validated stack measurement for this entry.
        self.shared_state.cumulative_gain_validated = self.shared_state.cumulative_gain
        self.shared_state.cumulative_gain_validated_ts = ts
        self.shared_state.cumulative_gain_validated_stack_len = len(
            self.shared_state.optimization_stack or []
        )

    def _should_continue_kernel_after_gemm(self) -> bool:
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

    # ------------------------------------------------------------------
    # Auto-roofline — single-path PRELUDE bootstrap + 10% watermark
    # refresh anchored on ``last_roofline_tput``.
    # ------------------------------------------------------------------
    _ROOFLINE_WATERMARK_RATIO: float = 1.10   # 10% step over last roofline

    def _current_tput_from_validated_gain(self) -> float:
        """Project the current measured tput from
        ``baseline_tput * (1 + cumulative_gain_validated/100)``.

        Returns 0.0 when ``baseline_tput`` is not yet known so callers
        can treat the watermark as not-yet-armed.
        """
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
        """Return True iff the projected current tput has crossed the
        10% watermark over ``last_roofline_tput``.

        Bootstrap guard: returns False when ``last_roofline_tput <= 0``
        and no previous roofline attempt failed, so the PRELUDE initial
        roofline enqueue remains the sole first-attempt entry point.

        Re-arm guard: returns False when an auto-roofline task is
        already in-flight (``auto_roofline_pending_task_id`` non-empty)
        so a single watermark crossing cannot enqueue multiple
        rooflines.
        """
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
        """Enqueue a fresh roofline if the watermark has crossed.

        Idempotency-keyed via ``reason`` so a re-entrant caller (e.g.
        explore-promote followed immediately by kernel-integrate-promote
        in the same tick) collapses to a single task. Sets
        ``auto_roofline_pending_task_id`` so the dispatch gate
        :meth:`_auto_roofline_pending_denial` blocks subsequent
        specialist / explore / kernel actions until the roofline lands.

        Returns True when a task was enqueued (or returned existing),
        False when the watermark check did not fire.
        """
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
        """Pick the kind for the next Coordinator-internal analysis task.

        ``shared_state.enable_roofline`` (CLI flag ``--enable-roofline``
        / ``--no-enable-roofline``, default on) controls the choice:

        * True  → ``roofline`` (composite: profile + trace_analyze +
          analysis.md snapshot).
        * False → ``profile`` (lightweight: trace capture only).

        Both kinds share the same enqueue + watermark-anchor +
        pending-task-gate plumbing; the kind name is the only thing
        that differs. PolicyGate denies either name when the LLM tries
        to propose it (``analysis_action_not_llm_proposable``).
        """
        return "roofline" if bool(
            getattr(self.shared_state, "enable_roofline", True),
        ) else "profile"

    def _registry_lanes_ttl(self, kind: str) -> tuple[list[str], int]:
        """Resolve ``(requires_lanes, lease_ttl_sec)`` for a task kind
        from the ActionRegistry so manually-created tasks inherit the
        same resource isolation + lease budget the dispatcher applies.

        ``requires_lanes`` is filtered to :data:`KNOWN_LANES` — the
        registry list also carries capability tags (e.g. ``emit_intent``)
        that are not dispatcher lanes and would otherwise wedge the task
        in ``_expand_lanes``. Returns ``([], 0)`` for an unknown action
        or when the registry is unavailable.
        """
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
        """Summarise warm-start ``what_worked`` items the scout can skip.

        Returns a list of ``{name, source}`` for proven optimizations from
        the prior session's recipe so the research scout focuses on net-new
        priors instead of re-mining already-validated ones. Empty when no
        warm recipe / no ``what_worked`` rows. Fail-soft.
        """
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
        """GAP 1 supporting helper — pre-fill ``explore_search.rejected``
        with the warm-start recipe's historical ``what_failed`` rows.

        Replays the operator-confirmed intent ("already-tested KEEP /
        REVERT shouldn't be re-tested"): every variant a prior session
        recorded as a failure for this (model, framework, hardware) is
        stamped with a canonical fingerprint and appended to the
        ledger, so PolicyGate's existing dedup gate denies any
        specialist / LLM proposal that lands on the same content.

        Decoupled from :meth:`_maybe_enqueue_warm_replay`:

        * Fires unconditionally after baseline (so ``--no-warm-replay``
          still benefits from negative-history filtering).
        * Idempotent — guarded by ``warm_history_injected`` flag so
          resume doesn't double-inject.
        * Skipped silently when KB is disabled / warm_start_recipe
          empty / no what_failed rows.

        Returns the number of rows newly added to the ledger (for
        logging + breakdown). Best-effort: any unexpected error is
        logged and swallowed so a bad warm-start payload never breaks
        the PRELUDE bootstrap.
        """
        state = self.shared_state
        if getattr(state, "warm_history_injected", False):
            return 0
        warm = state.warm_start_recipe or {}
        if not isinstance(warm, dict) or not warm:
            state.warm_history_injected = True
            return 0
        recipe = warm.get("recipe") or {}
        # v2 arbor shape keeps ``what_failed`` at the top level; v1
        # nested it under ``attrs``. Fall back to the recipe itself.
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
                # Whatever the recipe carried (gain_pct / error_class /
                # reason) is preserved for forensics; not strictly used
                # by the dedup gate.
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
        """GAP 1 — enqueue a one-shot ``replay_warm_recipe`` task when
        the T0 warm-start ladder returned a high-confidence prior.

        Lifecycle:
        1. ``--no-warm-replay`` or ``warm_replay_attempted=True`` (resume
           safety) → skip silently.
        2. No ``warm_start_recipe`` / confidence below threshold /
           best_config missing args+envs → skip with structured outcome.
        3. Otherwise mint a Coordinator-internal task whose params
           carry the warm best_config's ``extra_sglang_args`` /
           ``extra_envs`` so :class:`BaselineExecutor` runs the same
           Magpie subprocess as the baseline, but with the KB config
           applied. The baseline_config_path is forwarded so the
           workload contract (CONC / ISL / OSL / TP / ...) is
           identical to the baseline that just landed.

        Returns the created Task (or ``None`` when skipped). Idempotent
        via the fixed ``warm-replay-prelude`` idempotency key.
        """
        state = self.shared_state
        if not getattr(self, "_warm_replay_enabled", True):
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "disabled_by_flag",
            }
            # Flip the one-shot guard even on disabled-skip so that
            # a robustness resume without ``--no-warm-replay`` (which
            # is the common case — operators don't re-export every
            # flag on resume) cannot retroactively trigger a replay
            # against the operator's original intent. The guard
            # persists in state.json across robustness restarts.
            state.warm_replay_attempted = True
            return None
        if state.warm_replay_attempted:
            # Resume safety: a previous boot already enqueued / ran the
            # replay. The outcome (if any) is preserved in
            # ``warm_replay_outcome``; nothing more to do.
            return None
        warm = state.warm_start_recipe or {}
        if not isinstance(warm, dict) or not warm:
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "no_warm_start_recipe",
            }
            state.warm_replay_attempted = True
            return None
        # tier / conf were stamped at T0 by ``find_recipe_with_fallback``.
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
        # v2 RecipeKB returns the arbor shape with best_config /
        # sessions at the TOP LEVEL (no ``attrs`` wrapper); the retired
        # v1 cortex_kb_client nested them under ``attrs``. Fall back to
        # the recipe dict itself so both shapes round-trip.
        recipe_attrs = recipe.get("attrs") or recipe
        best_config = recipe_attrs.get("best_config") or {}
        if not isinstance(best_config, dict):
            best_config = {}
        # Need at least one of args / envs to be worth replaying. Read the
        # canonical ``extra_server_args`` FIRST (emitted by the gbrain remote
        # round-trip) before the legacy ``extra_sglang_args`` / ``args`` —
        # reading only the legacy names skipped a high-confidence gbrain warm
        # recipe as ``best_config_empty``. Explicit fallback (not the
        # warn-on-legacy compat reader) matches the sibling best_config reads
        # in ``_build_recipe_payload`` and stays quiet for local/cortex rows
        # that still carry the legacy key.
        bc_args = str(
            best_config.get("extra_server_args")
            or best_config.get("extra_sglang_args")
            or best_config.get("args")
            or ""
        ).strip()
        bc_envs = best_config.get("extra_envs") or best_config.get("envs") or {}
        if not isinstance(bc_envs, dict):
            bc_envs = {}
        if not bc_args and not bc_envs:
            state.warm_replay_outcome = {
                "status": "skipped",
                "reason": "best_config_empty",
                "warm_recipe_tier": tier,
                "warm_recipe_conf": conf,
            }
            state.warm_replay_attempted = True
            return None
        # Historical gain anchor — used by ``_promote_warm_replay`` to
        # judge "reproduced" vs "drift". The recipe schema records gain
        # per session in ``attrs.sessions[]`` (each entry is
        # ``{session_id, gain_pct, stack_len}``, written by
        # ``cortex_finalize_recipe_and_journal``). We take the MAX
        # across known sessions to reflect "the best a prior session
        # ever achieved with this recipe" — that's the upper bound a
        # reproduce should still meet.
        #
        # Fallback: 0.0 (recipe imported from a non-hyperloom source
        # without per-session gain — accept any positive measurement
        # in ``_promote_warm_replay``).
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
            # Reuse the baseline's workload contract (CONC/ISL/OSL/TP/...).
            # Without this the replay would render from the YAML's smoke
            # defaults and the gain comparison would be meaningless.
            "config_path": str(state.baseline_config_path or ""),
            # Carry the historical-gain anchor forward so the promote
            # path can compute the reproduce ratio without re-reading
            # warm_start_recipe.
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
        # Outcome stays empty until promote fills it in.
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
        """GAP 1 — interpret the result of a ``replay_warm_recipe`` task.

        Two paths:

        * ``status=="succeeded"`` with a positive tput → compute the
          measured gain vs baseline_tput; if it reproduces
          ≥ ``warm_replay_min_reproduce_pct`` of the recipe's expected
          gain, push the warm config onto :attr:`optimization_stack`
          (so the rest of the session inherits it as a "free" starting
          point) and update ``current_best``. Otherwise tag as
          ``status="drift"`` — no stack push, EXPLORE will start
          from a clean baseline.

        * Anything else (timeout / nonzero / OOM / invalid measurement)
          → tag as ``status="failed"`` with the error_class verbatim.
          No journal lesson / pitfall is written (the replay outcome
          is a compat / environment signal, not a knowledge fact —
          see PR rationale on GAP 1).

        Best-effort: failures are logged but never propagate back into
        the dispatcher (mirroring ``cortex_finalize_recipe_and_journal``).
        """
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
        # Use the baseline_tput captured at enqueue time (carried via
        # task.params) so a hypothetical baseline rerun mid-replay
        # can't shift the comparison anchor. Fall back to live
        # ``state.baseline_tput`` when the anchor wasn't plumbed
        # (defensive: legacy tests that build task params by hand).
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
        # Adopt KB best_config whenever replay beats the session baseline
        # (operator policy: any measured uplift is worth seeding the stack).
        # ``expected_gain`` / ``min_reproduce`` are retained in the outcome
        # for audit only — they no longer gate promotion.
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
            # R4-4 defense: pushing an empty stack entry (with no
            # extra_sglang_args / extra_envs) corrupts session_breakdown
            # attribution and confuses warm-start consumers in the
            # next session. ``task=None`` shouldn't normally happen
            # (``_promote_to_shared_state`` always supplies task), but
            # if it does we degrade gracefully: record the outcome
            # without polluting the stack.
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
            # Push warm best_config onto the stack so the rest of the
            # session inherits it. We synthesize a stack entry matching
            # the shape ``_lift_to_current_best`` writes for normal
            # EXPLORE KEEPs.
            # Stack entry schema mirrors what kernel-integrate /
            # explore-KEEP append (see lines 6520+ / 8030+). Fields are
            # what session_breakdown's attribution + capability_summary
            # consume; missing any of these (``ts`` / ``workspace`` /
            # ``gain_pct``) renders the entry as anonymous "stack
            # increment" in the final report.
            stack_entry = {
                "action":            "replay_warm_recipe",
                "name":              "warm_replay",
                "variant_name":      "warm_replay",
                # Canonical key — matches the stack entries
                # ``_lift_to_current_best`` writes for normal EXPLORE
                # KEEPs so downstream readers (stack-rebench, KB
                # write-back, session_breakdown) all key on the same
                # ``extra_server_args`` name post-rename.
                "extra_server_args": warm_args,
                "extra_envs":        warm_envs,
                "tput":              float(tput),
                "gain_pct":          round(measured_gain, 3),
                "workspace":         str(result.get("workspace") or ""),
                "ts":                datetime.now(timezone.utc).isoformat(),
                # ``source_tier`` here records the warm-recipe tier that
                # produced this entry (``exact`` / ``relative``) — useful
                # for the session_breakdown "where did this gain come
                # from" attribution.
                "source_tier":       outcome.get("warm_recipe_tier", ""),
                "source_confidence": outcome.get("warm_recipe_conf", 0.0),
            }
            # Resume safety: in the canonical PRELUDE → warm_replay
            # flow ``optimization_stack`` is empty when we land here
            # (baseline doesn't push onto stack). But a robustness
            # restart that lost ``warm_replay_attempted=True`` could
            # in theory call us with a non-empty stack — DO NOT clobber
            # the existing entries. Recompute cumulative gain from
            # baseline_tput → current tput instead of treating this
            # entry's measured_gain as the absolute total.
            state.optimization_stack = list(state.optimization_stack or [])
            # Idempotency guard: if a prior promote run already pushed
            # the warm_replay entry (resume mid-promote), DO NOT push
            # again. Detect via action="replay_warm_recipe".
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
            # Cumulative gain — derived from absolute tput / baseline,
            # not from summing per-entry gains (Hyperloom's stack is a
            # superposition rather than additive deltas, so the
            # ``current_tput / baseline_tput - 1`` formula is the
            # authoritative measure).
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
                # Canonical key — matches the ``current_best`` shape
                # ``_lift_to_current_best`` writes for normal KEEPs.
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
            # Journal the warm-replay as a synthetic KEEP entry so the
            # session report shows the inherited gain. We do NOT write
            # a KB lesson — the warm replay is a verification, not a
            # new fact (the recipe already exists in the KB).
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
        """Enqueue the PRELUDE-bootstrap roofline/profile task after baseline.

        Skipped while warm-replay is ``in_flight`` so two Magpie/sglang
        jobs do not contend for the same GPU/port. Called from the
        baseline completion hook and again when warm-replay promotion
        finishes (success, drift, or failure).
        """
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
        """Build + enqueue a Coordinator-internal analysis task.

        The kind (``roofline`` or ``profile``) is selected by
        :meth:`_internal_analysis_kind` from
        :attr:`SharedState.enable_roofline`. PRELUDE bootstrap and
        watermark-crossing paths both go through this single helper.

        Idempotency key: ``internal-analysis-<reason>`` — kind-agnostic
        so a mode flip across resume does not double-enqueue for the
        same reason (the first task for that reason still satisfies
        the gate, regardless of which kind it was). Concurrent callers
        with the same reason collapse to the existing task.

        Config selection: we deliberately do NOT pass
        ``state.baseline_config_path``. Both roofline and profile lean
        on :class:`ProfileExecutor` (RooflineExecutor calls it via
        ``_wrap_profile_ctx``), whose ``_resolve_default_config`` picks
        the ``profile_sglang.yaml`` / ``profile_vllm.yaml`` variant —
        the YAMLs that carry ``profiler.torch_profiler.enabled: true``
        so Magpie writes the ``.trace.json.gz`` files consumed
        downstream. Passing the baseline yaml here silently disables
        the torch profiler and the sub-step ends with
        ``error_class=no_trace_files``. Workload contract (CONC / ISL
        / OSL / TP / PRECISION / MAX_MODEL_LEN) still flows correctly
        because ``materialize_config_with_envs`` re-applies the env
        vars regardless of which YAML the executor starts from.
        """
        state = self.shared_state
        kind = self._internal_analysis_kind()
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
        }
        cb = state.current_best or {}
        if isinstance(cb, dict):
            cb_args = str(cb.get("extra_server_args") or "")
            if cb_args:
                params["base_extra_args"] = cb_args
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        # The ``internal-analysis-<reason>`` key is intentionally
        # kind-agnostic (roofline vs profile) so a mode flip across
        # resume cannot double-enqueue for the same reason. We do NOT
        # accept the old kind-specific spellings
        # (``internal-roofline-*`` / ``internal-profile-*``) — sessions
        # written by commits before this PR enqueue at most one extra
        # analysis task on the first resume tick (no compat shim,
        # acceptable per operator decision; see pr.md Migration notes).
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

    async def _auto_roofline_pending_denial(
        self, *, action_name: str = "",
    ) -> "PolicyDenied | None":
        """Return a ``PolicyDenied`` when an auto-roofline task is
        still in-flight; ``None`` otherwise.

        Gates every action that should observe the freshest
        ``analysis.md`` / ``last_profile_trace`` snapshot before
        running: specialist dispatches, explore-grid dispatches, and
        the KERNEL-owned actions (``kernel_opt`` / ``integrate`` /
        ``deep_kernel_analysis`` / ``operator_tuning`` /
        ``vendor_kernel_config``). The field is set by
        :meth:`_maybe_enqueue_watermark_roofline` (mid-run) or the
        PRELUDE bootstrap (baseline-completion hook) and cleared in
        :meth:`_promote_to_shared_state` once the task lands.
        """
        from .task_registry import TaskNotFound
        pending_id = (
            self.shared_state.auto_roofline_pending_task_id or ""
        ).strip()
        if not pending_id:
            return None
        try:
            pending = await self.tasks.get(pending_id)
        except TaskNotFound:
            # Task got purged (resume edge / corrupt state) — clear
            # the field and let the dispatch through; we never want a
            # corrupted pointer to permanently block dispatches.
            self.shared_state.auto_roofline_pending_task_id = ""
            return None
        terminal_states = {"succeeded", "failed", "cancelled", "needs_manual_review"}
        if pending.state in terminal_states:
            # Race: the promote path hasn't cleared the field yet.
            # Clear it here so subsequent dispatches go through.
            self.shared_state.auto_roofline_pending_task_id = ""
            return None
        label = action_name or "dispatch"
        return PolicyDenied(
            (
                f"delegate{{action_name={label!r}}} waits for the "
                f"auto-enqueued roofline task {pending_id!r} "
                f"(state={pending.state!r}) — downstream actions "
                f"need the fresh analysis.md snapshot before they "
                f"can run."
            ),
            rule="wait_for_auto_roofline",
            hint=(
                "The Coordinator auto-enqueued a `roofline` task "
                "(PRELUDE bootstrap or 10% gain watermark crossing); "
                "re-emit the same delegate next tick (TaskRegistry "
                "dedupes by content fingerprint) once the roofline "
                "result lands."
            ),
        )

    async def _roofline_denial_for_action(
        self, action_name: str,
    ) -> "PolicyDenied | None":
        """Apply the auto-analysis gate only to actions that require it."""
        if action_name not in _ROOFLINE_GATED_ACTIONS:
            return None
        return await self._auto_roofline_pending_denial(action_name=action_name)

    async def _defer_approved_proposal_for_roofline(
        self,
        pending: PendingProposal,
        approved_variant_names: set[str] | None,
    ) -> None:
        """Queue an approved proposal until the pending analysis task lands."""
        self._proposals_awaiting_roofline.append(
            (pending, approved_variant_names),
        )
        # Resume contract: this observation carries everything
        # ``replay_for_resume`` needs to rebuild the deferred queue after a
        # restart. A subsequent ``approved_proposal`` decision carrying the
        # same proposal_msg_id signals that the drain dispatched it.
        await self._record_observation(
            "coordinator", "observation",
            {
                "kind": "proposal_materialize_blocked",
                "reason": "wait_for_auto_roofline",
                "proposal_msg_id": pending.proposal_msg_id,
                "action_name": pending.action_name,
                "from_agent": pending.from_agent,
                "pending_roofline_task_id": (
                    self.shared_state.auto_roofline_pending_task_id
                    or ""
                ),
                "deferred_queue_depth": len(
                    self._proposals_awaiting_roofline,
                ),
                "approved_variant_names": (
                    sorted(approved_variant_names)
                    if approved_variant_names is not None
                    else None
                ),
                "kb_edge_ids": dict(pending.kb_edge_ids or {}),
            },
        )

    async def _drain_proposals_awaiting_roofline(self) -> None:
        """Re-run materialise for proposals deferred by the analysis gate.

        Called from :meth:`_promote_to_shared_state` once the
        Coordinator-internal ``profile`` / ``roofline`` task clears
        ``auto_roofline_pending_task_id``. Drains FIFO so the original
        Critic-approval order is preserved. Each materialise re-checks
        the gate, so if another analysis task slipped in between (rare
        but possible) the proposal is re-queued instead of dispatched
        before the freshest snapshot lands.
        """
        if not self._proposals_awaiting_roofline:
            return
        deferred = self._proposals_awaiting_roofline
        self._proposals_awaiting_roofline = []
        for pending, approved_variant_names in deferred:
            try:
                await self._materialize_approved_proposal(
                    pending, approved_variant_names=approved_variant_names,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "drain_proposals_awaiting_roofline: re-materialise "
                    "failed for proposal=%s action=%s",
                    pending.proposal_msg_id, pending.action_name,
                )

    def _record_phase_entry_evidence(self, **kvs: Any) -> None:
        """Merge ``kvs`` into the latest ``phase_history`` row's
        ``evidence`` dict.

        Used by phase-entry hooks (Gap-04 / future Gap-05/06) to
        record what the hook actually did *after* the transition row
        was already committed by ``record_phase_transition``. Pure
        in-memory mutation + a single state.save; SharedState
        ``phase_history`` is a list of dicts so the reference held
        by ``self.shared_state.phase_history[-1]`` is the live row.

        No-op when ``phase_history`` is empty (defensive).
        """
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

    # ------------------------------------------------------------------
    # SWEEP phase auto-dispatch
    # ------------------------------------------------------------------
    async def _drain_pending_keep_integrates(self) -> None:
        """Bug #7: drain pending KEEP integrates inherited from KERNEL.

        KERNEL→SWEEP transition is a hard budget cap; an orchestration
        tick window between a kernel's final KEEP and ``kernel_phase_
        budget_exhausted`` is often too short to propose+approve+run
        ``integrate``. The KEEP then strands in the ``kernel_opt_attempts``
        ledger with no path to land (SWEEP allowed set is {sweep,
        conc_sweep, recover}; ``integrate`` is denied by phase_incompatible).
        Downstream sweep / conc_sweep then measure the explore-only
        current_best and the kernel's E2E contribution never reaches
        the optimization_stack.

        Mitigation: at SWEEP entry, walk
        ``SharedState.next_pending_keep_kernel_id`` until empty, calling
        ``integrate_handler`` directly. Each integrate runs an actual
        E2E re-baseline (5-20 min) so this can delay SWEEP entry; that
        is acceptable vs hours of GPU idled in singleton-blocked SWEEP
        loops. Safety cap of 10 KEEPs guards against pathological queue
        sizes; failures push the kernel into ``rejected_kernel_ids`` so
        the queue cannot deadlock.
        """
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
                await integrate_handler(
                    {"kernel_id": kid, "base_tput": float(state.baseline_tput or 0.0)},
                    session_dir=self.session_dir,
                )
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

    # ------------------------------------------------------------------
    async def _on_enter_sweep(self, *, from_phase: str) -> None:
        """Auto-enqueue a ``sweep`` task on SWEEP entry.

        SWEEP must "自动构造 sweep grid (来自
        SKILL.md 默认 grid + Cortex ``recipe.sweep_grid`` 字段, 后者
        优先), 自动 enqueue ``sweep`` action". Without this hook the
        phase degrades to "LLM 自觉发 sweep" — and if ``max_minutes``
        runs out before the LLM proposes, ``_enter_closing_phase``
        force-enqueues report and the run finishes with zero sweep
        coverage. Operators lose the cross-workload validation that
        SWEEP exists to provide (§3.2 §5.4 first paragraph).

        Idempotent via ``idempotency_key='internal-sweep-phase_entry'``:
        a re-entry (defended against by Inv-2.1 in production, but
        possible in tests / operator scripts) reuses the existing
        task instead of duplicating.

        After enqueue, stamps ``phase_history[-1].evidence`` with
        ``auto_sweep_enqueued=True`` + ``auto_sweep_task_id`` +
        ``auto_sweep_grid_source`` + ``auto_sweep_combos`` so the
        breakdown collector can verify the hook
        fired and which grid source was used.

        Singleton enforcement: PolicyGate's
        ``sweep_phase_singleton`` rule denies any LLM-emitted
        ``delegate{action_name='sweep'}`` or
        ``propose_action{action_name='sweep'}`` once this hook has
        stamped ``evidence.auto_sweep_task_id`` for the active SWEEP
        phase. Two concurrent sweep tasks would race for the same 8
        GPUs and the same TCP port, crashing both vllm engines on
        init (``HSA_STATUS_ERROR_OUT_OF_RESOURCES``) and producing
        zero workload-curve coverage. The rule self-clears at
        SWEEP→CLOSE because phase_history[-1] turns over. Operator
        debug override: ``params.bypass_sweep_singleton=True`` on
        the LLM intent payload.
        """
        state = self.shared_state
        # Bug #7 fix: drain pending KEEP integrates from prior KERNEL
        # phase so sweep / conc_sweep measure the full current_best
        # (explore + kernel stack), not the explore-only baseline.
        if getattr(state, "has_keep_pending_integrate", False):
            await self._drain_pending_keep_integrates()
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
        # The params dict carries the chosen grid + source so we can
        # mirror them onto the phase_history evidence without
        # re-running the recipe lookup.
        grid_source = str(task.params.get("source") or "")
        isl_osl = task.params.get("isl_osl_configs") or []
        conc_values = task.params.get("conc_values") or []
        # Combos = |conc_values| × |isl_osl_configs| (sweep executor
        # fans out CONC × (ISL,OSL)).
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
        """Build + enqueue a Coordinator-internal ``conc_sweep`` task.

        Mirrors :meth:`_enqueue_internal_sweep_task` but emits the
        SWEEP-phase ``conc_sweep`` action (baseline + current_best
        across a CONC ladder, see
        ``orchestrator/conc_sweep.py``). Off by default — the caller
        (typically the sweep-completion hook) must check
        ``shared_state.conc_sweep_enabled`` first; this method does
        NOT re-check the flag, so callers wanting to dispatch
        unconditionally (operator-driven resume scenarios) can do so.

        Idempotency key: ``internal-conc_sweep-<reason>``. PolicyGate's
        action-singleton rule plus this key together ensure at most
        one conc_sweep task lands per SWEEP phase.

        Returns the freshly-created (or returned-existing) Task; the
        caller logs the task_id. Returns None and logs an error if
        creation raised — conc_sweep is post-sweep gravy, never fatal.
        """
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
                # Conc_sweep can run for hours (default 2.5h, up to
                # ``total_budget_sec``); lease_ttl matches so the lease
                # doesn't expire mid-flight. Coordinator's existing
                # lease-extension heartbeats keep it alive if a
                # variant overshoots.
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
        # Bug #11 fix: stamp evidence so PolicyGate's
        # ``conc_sweep_phase_singleton`` rule can deny subsequent
        # LLM-emitted conc_sweep proposals in the same SWEEP phase.
        # Without this, orchestration loops re-proposing conc_sweep
        # (sweep is singleton-blocked, only conc_sweep is allowed)
        # until SWEEP budget exhausts.
        self._record_phase_entry_evidence(auto_conc_sweep_task_id=task.task_id)
        return task

    async def _enqueue_internal_sweep_task(
        self, *, reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``sweep`` task.

        Mirrors the Gap-04 :meth:`_enqueue_internal_profile_task`
        contract: ``source='coordinator_internal'``, ``reason``
        verbatim into the idempotency_key suffix, ``config_path`` /
        ``base_extra_args`` inherited from SharedState so the sweep
        executor honours the workload contract + current_best
        configuration.

        Grid source priority:

        1. ``state.warm_start_recipe.sweep_grid`` — Cortex-recorded
           grid from a prior session for this (model, gpu) combo.
        2. SKILL.md defaults (``sweep.DEFAULT_CONC_VALUES`` /
           ``DEFAULT_ISL_OSL`` / ``DEFAULT_NUM_PROMPTS_FACTOR``).

        Idempotency key: ``internal-sweep-<reason>``.

        Returns the freshly-created (or returned-existing) Task. The
        caller reads ``task.task_id`` + ``task.params['source']`` for
        logging / evidence stamping.
        """
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
        # Mirror the last-baseline benchmark_script when present so a
        # sweep run that re-launches sglang/vllm uses the same shell
        # wrapper as baseline (matches Gap-04 _enqueue_internal_profile_task).
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
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
        """Pick a sweep grid: Cortex recipe first, defaults fallback.

        "来自 SKILL.md 默认 grid + Cortex
        ``recipe.sweep_grid`` 字段, 后者优先". The recipe lookup is
        defensive — at the time of writing the Cortex T0 response is
        still free-text under ``warm_start_recipe.raw``; a future
        Cortex schema PR (§3.14 R-13) will add a structured
        ``sweep_grid`` field. This helper accepts either:

        * ``warm_start_recipe['sweep_grid'] = {conc_values: [...],
          isl_osl_configs: [...], num_prompts_factor: int}`` — direct
          structured form (preferred when it arrives).
        * Anything else → defaults.

        Returns a dict with keys ``source`` / ``conc_values`` /
        ``isl_osl_configs`` / ``num_prompts_factor``. ``source`` is
        one of ``cortex_recipe`` / ``skill_md_default`` so downstream
        observability (breakdown.sweep.grid_source) can tell which
        path won.

        Defensive against malformed recipe entries:

        * Non-dict ``warm_start_recipe`` → defaults.
        * Non-dict ``sweep_grid`` → defaults.
        * ``conc_values`` not a non-empty list → fall back to the
          SKILL.md default for that single field.
        * Same for ``isl_osl_configs``.
        * ``num_prompts_factor`` not a positive int → default.

        (Per-field fallback rather than all-or-nothing lets a recipe
        that only overrides one dimension still benefit from the
        defaults for the others.)
        """
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

    # ------------------------------------------------------------------
    # CLOSE phase sequencer
    # ------------------------------------------------------------------
    # Class-level timeouts for the CLOSE sequencer's wait-for-task
    # polls. Class attributes (rather than constants in the method)
    # so tests can override per-instance with small values without
    # patching method internals. Production defaults: report ≤ 10 min
    # (matches ``BaselineExecutor`` cap);
    # session_breakdown ≤ 5 min (tiny report, lots of headroom).
    CLOSE_REPORT_TIMEOUT_SEC: float = 600.0
    CLOSE_SESSION_BREAKDOWN_TIMEOUT_SEC: float = 300.0
    CLOSE_NDJSON_DRAIN_TIMEOUT_SEC: float = 60.0

    def _derive_close_stop_reason(self) -> str:
        """Best-effort ``stop_reason`` for a CLOSE reached with a blank one.

        Most CLOSE entries are *not* wall-clock timeouts. The phase
        machine records the transition reason on the latest
        ``phase_history`` row (e.g. ``sweep_done`` / ``conc_sweep_done``
        for a normal SWEEP completion), but those are non-terminal
        transitions so ``_advance_phase_if_needed`` never mirrored them
        onto ``state.stop_reason``. Recover the reason from the most
        recent row that lands in CLOSE when it is a valid vocab term;
        otherwise fall back to ``time_exhausted`` (the wall-clock
        deadline is the only common path that reaches CLOSE without a
        recorded phase-exit reason).
        """
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
            # Newest CLOSE-bound row had no usable reason — stop scanning
            # rather than picking up a stale older transition.
            break
        return "time_exhausted"

    async def _on_enter_close(self, *, from_phase: str) -> None:
        """CLOSE phase sequencer.

        Runs the fixed order:

        1. ``report``            — generate markdown / json report
        2. ``session_breakdown`` — write ``session_breakdown.json``
        2.5 ``fact_finalize``   — write final ``update_recipe`` to KB +
                                  finalize the local optimization journal
                                  (``total_gain_pct`` / ``final_throughput``)
        3. ``ndjson_drain``     — retired no-op (writes are local-only)
        4. mark ``close_sequence_done`` (and ``stop_reason``)

        (The legacy ``cortex_commit`` step was retired — fact writes are
        session-less so there is no remote sid to close.)

        Each step records a row under
        ``phase_history[-1].evidence.close_steps`` so the breakdown
        collector (and operators) can verify the sequence completed.
        Steps are best-effort: a failure in any step stamps
        ``status='failed' / 'timeout'`` evidence but does not abort
        the remaining steps. The final ``done`` step always runs so
        the cli.finally short-circuit is consistent even when earlier
        steps partly failed.

        Idempotence: report / session_breakdown enqueue uses fixed
        idempotency_keys (``internal-report-close_phase_entry`` /
        ``internal-session_breakdown-close_phase_entry``) so a phase
        re-entry (forbidden in production, but resume from a crash
        mid-sequencer counts) reuses existing tasks.

        The sequencer runs INLINE inside the hook — it doesn't wait
        for the reactor / dispatcher tick boundary. Steps 1 and 2
        enqueue tasks then poll ``_wait_for_task_terminal`` until the
        dispatcher (which the same Coordinator.run() loop drives)
        picks them up and finishes. Step 3 (NDJSON drain) is a retired
        no-op; step 4 (Cortex commit) writes the recipe + journal; step
        5 is a single SharedState write.
        """
        log.info("CLOSE entered (from=%s); starting 5-step close sequence",
                 from_phase or "<unknown>")
        await self._record_close_step("sequencer_started", status="running")

        # stop_reason MUST be persisted BEFORE step 2 writes the
        # session_breakdown. The breakdown executor runs as a subprocess
        # that reads state.json from disk, and the collector derives both
        # ``stop_reason`` and ``ended_at_utc`` from it. The old code only
        # set stop_reason in step 5 (below), AFTER the breakdown was
        # already serialized — so any CLOSE reached via a non-wall-clock
        # path (e.g. an LLM ``report`` terminal transition, where the loop
        # had not yet stamped a reason) shipped an empty stop_reason /
        # ended_at_utc downstream. Filling it here (only when still blank;
        # real reasons like baseline_failed / target_reached are already
        # set before CLOSE) closes that race. Step 5 stays as an
        # idempotent backstop.
        #
        # DO NOT unconditionally stamp ``time_exhausted``: most CLOSE
        # entries are NOT wall-clock timeouts. A normal SWEEP / conc-sweep
        # completion transitions to CLOSE with a perfectly good
        # phase-exit reason (``sweep_done`` / ``conc_sweep_done`` — both
        # valid STOP_REASON_VOCAB terms) recorded on the latest
        # ``phase_history`` row by ``_advance_phase_if_needed``, but it is
        # NOT a terminal evidence transition, so the early mirror at line
        # ~1407 left ``stop_reason`` blank. Derive the reason from that
        # row first; only fall back to ``time_exhausted`` when the run
        # genuinely has no usable phase-exit reason.
        if not self.shared_state.stop_reason:
            derived = self._derive_close_stop_reason()
            self.shared_state.set_stop_reason(derived)
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "CLOSE: early stop_reason persist failed; step 5 will retry"
                )

        # CLOSE-entry auto-roofline (former N31) was deleted in favour
        # of the EXPLORE-entry / KERNEL-entry hooks; the gain-only
        # freshness gate (:meth:`_needs_fresh_roofline`) keeps the
        # snapshot aligned with ``cumulative_gain_validated`` without
        # paying the cost a third time on CLOSE.

        # ---------------- Step 1: report ----------------
        try:
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
            else:
                await self._record_close_step(
                    "report", status="failed",
                    task_id=report_task.task_id,
                    detail=f"task_state={terminal_state!r}",
                )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception("CLOSE step 1 (report) failed")
            await self._record_close_step(
                "report", status="failed", detail=repr(exc)[:240],
            )

        # ---------------- Step 2: session_breakdown ----------------
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

        # ---------------- Step 3: (retired) NDJSON drain ----------------
        # The v1 cortex_kb_client used an NDJSON pending queue +
        # drain_pending() to retry failed central writes. Under the
        # v2 RecipeKB design writes are local-only (no remote
        # fan-out queue), so there is nothing to drain. The step is
        # kept as a no-op marker so close-step ledger consumers
        # don't break on a missing entry.
        await self._record_close_step("ndjson_drain", status="skipped")

        # ---------------- Step 5: mark done ----------------
        self.shared_state.close_sequence_done = True
        # phase-machine CLOSE path must set
        # ``stop_reason`` so the main run loop's outer check
        # (Coordinator.run line ~1936) terminates the run on the next
        # tick. Without this the sequencer completes, writes the
        # report + breakdown, and the loop keeps ticking forever
        # (orchestration re-proposing report, robustness re-emitting
        # alerts, Cortex returning 409 on the committed session).
        # The wall-clock deadline path (``_enter_closing_phase``) sets
        # ``time_exhausted`` from the loop body (line ~1971); both
        # paths converge on the same vocab term per
        # ``STOP_REASON_VOCAB``. NOTE: this is now an idempotent
        # backstop — the early persist at the top of the sequencer has
        # normally already filled a blank stop_reason before step 2's
        # breakdown was serialized. We re-derive (rather than hard-code
        # ``time_exhausted``) so this backstop matches the early path and
        # never mislabels a normal SWEEP/conc-sweep completion.
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
        """Build + enqueue a Coordinator-internal ``report`` task.

        Mirrors :meth:`_enqueue_internal_profile_task` /
        :meth:`_enqueue_internal_sweep_task` shape. Idempotency_key
        ``internal-report-<reason>``.

        Reuses ``state.closing_report_task_id`` when set so the
        wall-clock deadline path (``_enter_closing_phase``) and the
        CLOSE phase sequencer don't race to insert two report tasks
        with different ids. When the wall-clock path enqueued first,
        the sequencer simply waits for that task instead.
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
                # Stale id (resume from a wiped-tasks-table session); fall
                # through to fresh enqueue.
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
        # Mirror onto state for back-compat with the existing wall-clock
        # path inspectors (``closing_report_task_id`` is also used by
        # robustness alerting + breakdown summary lines).
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

    @staticmethod
    def _steward_infrastructure_failure(done_payload: dict[str, Any]) -> bool:
        """True when the steward subprocess failed before a real verdict."""
        if not isinstance(done_payload, dict):
            return True
        reason = str(done_payload.get("reason") or "").strip().lower()
        if reason.startswith("subprocess_"):
            return True
        if bool(done_payload.get("empty")) and not str(
            done_payload.get("recommendation") or "",
        ).strip():
            return True
        return False

    async def _enqueue_internal_steward_task(
        self, *, reason: str, retry_attempt: int = 0,
    ) -> "Task | None":
        """Enqueue a Coordinator-owned session_steward_specialist task.

        Mirrors :meth:`_enqueue_internal_report_task` shape, with two
        differences:

        * ``kind='specialist'`` (the standard SpecialistRunner path
          handles the LLM dispatch + worktree-less subprocess);
        * idempotency key includes the current EXPLORE round id so
          repeated calls within the same round collapse to one task,
          while a new plateau in a later round can dispatch fresh.

        Bypasses PolicyGate the same way ``closing_report_task`` does
        (Coordinator-internal callers don't go through the
        propose/delegate validation path). LLM-side proposals of
        ``assess_remaining_gaps`` are throttled separately by
        :meth:`_assess_remaining_gaps_throttle_denial`.

        Returns ``None`` when a task already exists in the registry
        with the same idempotency key (whether queued / running /
        succeeded). The caller (``_maybe_enqueue_steward``) treats
        ``None`` as "nothing to do this tick".
        """
        round_id = int(
            (self.shared_state.explore_search or {}).get("cursor") or 0
        )
        idempotency_key = f"internal-steward-round{round_id}"
        if retry_attempt > 0:
            idempotency_key = (
                f"{idempotency_key}-retry{int(retry_attempt)}"
            )
        # Avoid re-enqueueing if a steward verdict already landed in
        # this round (paranoia — wants_steward_assessment should have
        # returned False, but defense-in-depth).
        last = self.shared_state.last_remaining_gaps_assessment or {}
        if isinstance(last, dict) and last.get(
            "round_at_assessment"
        ) == round_id:
            last_rec = str(last.get("recommendation") or "").strip().lower()
            if last_rec in _STEWARD_RECS:
                return None
        params: dict[str, Any] = {
            "domain": "session_steward_specialist",
            "gap_canonical_id": f"gap.steward.round{round_id}",
            "gap_symptom": (
                "EXPLORE plateau triggered; assess remaining leverage "
                "and recommend continue_explore / advance_to_kernel / "
                "stop_session."
            ),
            "gap_layer": "session_strategy",
            "max_turns": 8,
            "source": "coordinator_internal",
            "reason": str(reason),
        }
        # Warm the specialist params the same way the LLM-dispatch path
        # does so the prompt sees real hardware + workload context.
        await self._warm_specialist_params(params)
        try:
            task, was_existing = await self.tasks.create_or_return_existing(
                kind="specialist",
                params=params,
                idempotency_key=idempotency_key,
                requires_lanes=["research_lane"],
                allowed_tools=[
                    "emit_intent", "Read", "Grep", "Glob", "Bash",
                    "WebSearch", "WebFetch",
                ],
                side_effects=["workspace_write"],
                lease_ttl_sec=1800,
            )
        except Exception:  # noqa: BLE001 — TaskRegistry edge cases
            log.exception(
                "internal-steward: enqueue failed (round=%d)", round_id,
            )
            return None
        if was_existing:
            log.info(
                "internal-steward task reused (task_id=%s, state=%s)",
                task.task_id, task.state,
            )
        else:
            log.info(
                "internal-steward dispatched: task_id=%s round=%d "
                "reason=%s",
                task.task_id, round_id, reason,
            )
        return task

    async def _enqueue_internal_research_scout_task(
        self, *, reason: str, round_id: int,
    ) -> "Task | None":
        """Enqueue a Coordinator-owned research-scout specialist task.

        Read-only collector dispatched without an LLM propose step (same
        internal-dispatch shape as the steward). Idempotency is keyed by
        the qualifying round so repeated ticks within one round collapse
        to a single scout. Returns ``None`` when a task already exists
        for this round or enqueue fails (fail-soft).
        """
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
        """Force-dispatch the PRELUDE research scout (not LLM-proposable).

        Always writes the ``research_hints.md`` skeleton first so the
        artifact exists even if the scout produces nothing or is disabled.
        """
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

    def _aggregate_research_evidence(
        self, done_payload: dict[str, Any],
    ) -> None:
        """De-dup a specialist's reported research evidence into the
        exploration-depth tracker. No-op when no ``research`` block."""
        block = done_payload.get("research")
        if not isinstance(block, dict):
            return

        def _ids(key: str) -> list[Any]:
            vals = block.get(key)
            return list(vals) if isinstance(vals, list) else []

        added = self.shared_state.register_research_evidence(
            prs_fetched=_ids("prs_fetched"),
            pr_diffs_read=_ids("pr_diffs_read"),
            nvidia_refs_compared=_ids("nvidia_refs") or _ids(
                "nvidia_refs_compared"
            ),
        )
        if any(added.values()):
            log.info("depth: research evidence added %s", added)

    def _harvest_research_scout(self, done_payload: dict[str, Any]) -> None:
        """Persist scout output: hints, competitor target, gap seeds, dedup.

        All steps are fail-soft — a malformed research block degrades to
        a no-op rather than aborting the specialist-done handler.
        """
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

    async def _maybe_enqueue_steward(self) -> None:
        """IR-7 — enqueue a steward task when phase_state says we need one.

        Called from ``_advance_phase_if_needed`` after
        ``compute_next_phase`` returns ``None``. The helper is pure
        sugar over :meth:`_enqueue_internal_steward_task` + the
        phase_state predicate.
        """
        try:
            wants = _phase_state.wants_steward_assessment(
                self.shared_state,
                budget_pct=self._phase_budget_pct,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("steward: wants_steward_assessment failed")
            return
        if not wants:
            return
        await self._enqueue_internal_steward_task(
            reason="coordinator_plateau",
        )

    async def _enqueue_internal_session_breakdown_task(
        self, *, reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``session_breakdown``
        task.

        Same idempotency contract as the report helper. The
        ``session_breakdown`` action is already registered in cli's
        ``_REAL_EXECUTORS_FULL`` (action_executors/session_breakdown.py),
        so the standard dispatcher picks it up without special-casing.
        """
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
        """Append one row to ``phase_history[-1].evidence.close_steps``.

        Concrete row shape::

            {"step": "report", "status": "done",
             "task_id": "abc123", "detail": "",
             "ts": "2026-..."}

        Best-effort: a malformed phase_history row (missing dict /
        non-list close_steps) gets a fresh structure installed; a
        SharedState.save failure is logged but doesn't abort the
        sequencer. Mirrors the contract of
        :meth:`_record_phase_entry_evidence` but persists per-step
        rather than per-hook.
        """
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

    # ==================================================================
    # Bounded test interface
    # ==================================================================
    async def _replay_resume_if_needed(self) -> None:
        """Rebuild in-memory state once for a resumed session.

        Shared by ``tick()`` and ``run()``: replay the event log, drain
        proposals that were blocked on a now-complete analysis task, and
        abandon non-terminal dynamic_action dispatches. No-op when the
        session is fresh or already rebuilt.
        """
        if not (self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]):
            return
        await self.replay_for_resume()
        # The analysis task may have completed during shutdown, so the
        # normal drain hook in ``_promote_to_shared_state`` will not fire
        # on restart; kick it explicitly to re-check the roofline gate.
        if self._proposals_awaiting_roofline:
            await self._drain_proposals_awaiting_roofline()
        # Transition orphaned dynamic_action dispatches to ABANDONED and
        # clean up their worktree + git branch.
        self._resume_abandon_dynamic_actions()

    async def _pump_framework_pr_phase_safely(self, *, caller: str) -> None:
        """Best-effort FRAMEWORK_PR pump wrapper shared by tick and run."""
        try:
            await self._pump_framework_pr_phase()
        except Exception:  # noqa: BLE001 — defensive
            log.exception("FRAMEWORK_PR pump (%s) failed", caller)

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
        await self._replay_resume_if_needed()
        for _ in range(n):
            self.shared_state.increment_tick()
            for name in self._tick_roles:
                await self._reactor_pass(name)
            await self._pump_dispatcher_once()
            # FRAMEWORK_PR phase pump: enqueue next candidate / fetch
            # next batch when no framework_pr task is in flight. Best-
            # effort; failures degrade to phase_done so we never wedge.
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
        closing_grace_sec: float | None = None,
    ) -> str:
        """Run reactor + dispatcher in a long-running loop until a stop
        condition fires.

        Stop signals (in priority order, see DESIGN §9.1):

        * ``self._stop`` set (from SIGINT/SIGTERM or ``stop()``)
            → ``stop_reason="signal"``
        * objective.reached(shared_state)  → ``"target_reached"``
        * wall-clock budget exceeded        → enter closing phase, then
            ``"time_exhausted"`` after report flush or grace elapses
        * crash_count >= ``crash_emergency_threshold`` → ``"emergency"``
        * custom ``stop_when`` callback returns True → ``"custom"``
        * ``max_ticks`` reached (test guard) → ``"max_ticks"``

        On stop, ``shared_state.stop_reason`` is set + saved + the final
        value is returned.
        """
        objective = objective or TimeOnlyObjective()
        # Stash so ``_compose_prompt`` can update target_gap_pct.
        self._current_objective = objective
        grace_sec = effective_closing_grace_sec(max_minutes, closing_grace_sec)
        deadline = (
            time.monotonic() + max_minutes * 60.0 if max_minutes else None
        )
        self._run_started_monotonic = time.monotonic()
        self._run_deadline = deadline
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
                    # Bump the persistent tick counter — drives phase / plateau
                    # math. Persisted on the next save() (after
                    # _promote_to_shared_state or stop).
                    self.shared_state.increment_tick()
                    in_closing = self.shared_state.closing_phase
                    # Run one reactor + dispatcher pass. During closing, skip
                    # LLM reactor passes and only pump the deterministic report.
                    if not in_closing:
                        for name in self._tick_roles:
                            if self._stop.is_set():
                                break
                            await self._reactor_pass(name)
                    if not self._stop.is_set():
                        await self._pump_dispatcher_once()
                    # FRAMEWORK_PR phase pump: see ``tick()`` for rationale.
                    if not in_closing:
                        await self._pump_framework_pr_phase_safely(caller="run")
                    # phase machine advance at tick boundary.
                    # Runs even when ``in_closing`` so CLOSE phase still gets
                    # recorded into phase_history when the final breakdown
                    # writer transitions us in.
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

                # ---- check stop conditions ----
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
            if self.shared_state.closing_phase:
                self.shared_state.closing_phase = False
            # Resuming an already-terminal session re-enters the loop past
            # its deadline and can break out before the local stop_reason is
            # reassigned; preserve any prior persisted reason instead of
            # downgrading it to "unknown".
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

    async def _enter_closing_phase(self, *, grace_sec: float) -> float:
        """Enter report-flush phase after the wall-clock deadline.

        Skips Orchestration→Critic; enqueues a deterministic ``report`` task
        directly into the TaskRegistry. Returns the monotonic closing deadline.
        """
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
        """True when the closing-phase report task reached a terminal state."""
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
            await self._track_backend_error_streak(agent_name, exc)
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
            self._record_coordinator_exception(
                stage="reactor_pass",
                agent=agent_name,
                exc=exc,
            )
            return
        # Reset the streak — a successful turn proves the backend is alive
        # again; the next BackendError starts a fresh count.
        if self._backend_error_streak.get(agent_name):
            self._backend_error_streak[agent_name] = 0
            self._backend_error_alarm_armed[agent_name] = True
        for intent in result.intents:
            await self._handle_intent(agent_name, intent)

    async def _track_backend_error_streak(
        self, agent_name: str, exc: BackendError,
    ) -> None:
        """Increment the per-agent ``BackendError`` streak and escalate once.

        Coordinator already records a per-call ``backend_error``
        observation; this helper layers a *streak* observation on top so
        the inbox carries a single, structured ``backend_unhealthy``
        event when an agent's subprocess (typically critic-agent or
        robustness-agent) has been crashing or timing out for
        consecutive ticks. Operators see one high-severity row instead
        of N scattered backend_error rows.

        The alarm re-arms only after the streak resets to zero (a
        successful turn), so we never spam the inbox once the threshold
        is crossed.
        """
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
        """Return TaskRegistry rows for ``kind='specialist'`` tasks whose
        wall-clock running duration exceeds ``self._specialist_stale_sec``.

        Wired through ``_compose_prompt`` so Robustness can decide to
        emit ``kill_task``. v0.8 §3.3 §4.4 contract; M5 lands the actual
        ``specialist`` task kind so the registry stays empty until then —
        the scanner returns ``[]`` and Robustness shows ``count=0``.

        Each row is ``{"task_id": str, "running_seconds": float,
        "kind": "specialist"}`` (a deliberately small projection so the
        prompt block stays narrow). Failures (registry unreachable, etc.)
        return ``[]`` and log; **never** raise — the prompt assembly
        cannot block on an audit query.
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
            # ``updated_at`` is the last state-transition timestamp; for a
            # ``running`` task that's when the dispatcher promoted it
            # (no later transition touches ``updated_at`` until
            # completion). Parse it as the start of the running window.
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

        # per-tick phase block injected for **every** agent.
        # Comes high in the prompt (right after SESSION_DIR) because R1
        # rejection is driven by the current phase; the LLM should see
        # phase context before anything else.
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

        # 0a. Mission progress (Orchestration only). Outcome-shaped
        # projection of SharedState (raw vs validated gain, time spent
        # vs budget, optimization-stack rebench freshness) shown before
        # the verbose SharedState dump so the LLM cannot miss the
        # ``stack rebench required`` signal.
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

        # Time budget for Robustness — it consumes this to fire the
        # ``deadline_imminent`` signal that escalates to a ``delegate(report)``
        # wind-down. Orchestration already got its copy in the Mission-progress
        # block above; Kernel and Critic don't subscribe to it.
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

        # 1. Shared session state — gives the agent goal + progress context
        # even on tick 1 when the inbox is empty.
        sections.append("=== Shared session state ===")
        sections.append(self.shared_state.to_prompt_summary())
        if agent_name == "orchestration":
            # ``target_gap_pct`` is a *fact* (how much gain is still
            # needed for ``--target-gain``). Refresh so the prompt's
            # Mission-progress line stays current.
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
            denial_summary = self.shared_state.to_policy_denial_summary(top_k=6)
            if denial_summary:
                sections.append(denial_summary)
            # Compact projection of the most recent ``dynamic_action``
            # summaries so orchestration sees its own dispatch outcomes
            # without paying the token cost of the full ledger.
            dyn_section = self.shared_state.to_dynamic_actions_prompt_section()
            if dyn_section:
                sections.append(dyn_section)

        # Cortex T0 warm-start snapshot + structured
        # gaps[] ledger injected into the Orchestration
        # prompt. ``kb_digest`` was retired upstream (origin/main commit
        # befbd1381814 — removed the hardcoded marathon path), so this
        # block is the replacement: a structured per-session
        # snapshot the DECISION FRAMEWORK consumes directly.
        if agent_name == "orchestration":
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
            # Advisory multi-model proposal scores (ProposalScorer).
            # One reference among many — parallel to gaps / KB /
            # analysis.md, NOT a ranking directive. Section
            # omitted entirely when no recent round carries scores.
            try:
                scores_block = self.shared_state.to_proposal_scores_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: proposal_scores_summary failed")
                scores_block = ""
            if scores_block:
                sections.append("=== Specialist proposal scores (advisory) ===")
                sections.append(scores_block)
            # Priors-match: which recently proposed variants align with
            # proven research hints / the dominant external gap. Advisory
            # ordering signal only (NOT a score, NOT a gate); omitted
            # when nothing matches.
            try:
                priors_block = self._priors_match_advisory_block()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: priors-match advisory failed")
                priors_block = ""
            if priors_block:
                sections.append("=== Priors-match (advisory ordering) ===")
                sections.append(priors_block)

            # Surface the intervention-mix ledger (config vs code_patch
            # counts) as neutral telemetry for Orchestration.
            try:
                mix_block = self.shared_state.to_intervention_mix_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: intervention_mix_summary failed")
                mix_block = ""
            if mix_block:
                sections.append("=== Intervention mix (telemetry) ===")
                sections.append(mix_block)

        # Robustness gets a phase budget telemetry +
        # specialist health block so it can fire the medium-severity
        # alerts described in the role prompt.
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
    def _target_analysis_baseline_exists(self) -> bool:
        """True iff target_analysis has produced ``target_baseline.json``.

        The TargetAnalysisExecutor's failure policy is "never fail": even if
        InferenceX is unreachable, it still writes a JSON with
        ``status=fetch_error``. So existence of the file is a sufficient
        "this gate has been satisfied" signal — we never expect this gate
        to loop more than once per session.
        """
        try:
            from ..session_paths import target_baseline_json
            return target_baseline_json(self.session_dir).exists()
        except Exception:  # noqa: BLE001 — defensive; missing helper -> treat as done.
            return True

    def _kernel_opt_keep_pending(self) -> str:
        """Return the next kernel_id awaiting integrate, or "" if none.

        Delegates to :meth:`SharedState.next_pending_keep_kernel_id`,
        which scans the per-kernel ``kernel_opt_attempts`` ledger instead
        of the single ``last_kernel_opt`` slot. This is what lets the
        TODO 4/5 integrate gate drain a batch's full KEEP queue (sorted
        strongest-first, same-source-file collapsed) rather than only
        the most recently recorded KEEP.

        Closed when ANY of these hold (all enforced by
        :meth:`SharedState.next_pending_keep_kernel_id`):
          * No ``KEEP`` entries in ``kernel_opt_attempts``.
          * Every pending KEEP has been retired (``rejected_kernel_ids``)
            or already absorbed into ``optimization_stack`` as an
            ``integrate`` entry.
          * Every pending KEEP shares its source_file with a KEEP that
            already landed on the stack (whole-file overwrite conflict).
        """
        return self.shared_state.next_pending_keep_kernel_id()

    def _baseline_self_loop_denial(
        self, proposed_params: dict[str, Any] | None,
    ) -> PolicyDenied | None:
        """Reject a fresh baseline proposal that just replays the last failure.

        The Orchestration prompt's FAILURE RECOVERY section instructs the
        LLM to introduce a new ``benchmark_script`` / ``result_dir`` /
        ``extra_server_args`` override after a baseline failure. This
        method is the PolicyGate stop-loss that fires when the LLM
        ignores that instruction.

        Fires only when ALL of these hold:

        * Two or more consecutive baseline failures have landed on
          ``shared_state.baseline_attempts`` (any decision tail that
          isn't ``status=succeeded`` counts).
        * Those last two failures both carry a
          ``fingerprint`` in their ``extras`` and the fingerprints
          match each other.
        * The current proposal's fingerprint matches the failed-streak
          fingerprint.

        When all three match, return :class:`PolicyDenied` with a
        ``baseline_self_loop`` rule and a hint pointing at the next
        override surface so the prompt sees a deterministic recovery
        path. Returns ``None`` otherwise — the regular execution-order
        rules still apply.
        """
        attempts = list(self.shared_state.baseline_attempts or [])
        # Walk the tail backwards collecting *consecutive* failures.
        tail_failures: list[dict[str, Any]] = []
        for entry in reversed(attempts):
            if not isinstance(entry, dict):
                break
            if entry.get("status") == "succeeded":
                break
            tail_failures.append(entry)
        if len(tail_failures) < _BASELINE_SELF_LOOP_THRESHOLD:
            return None
        recent = tail_failures[: _BASELINE_SELF_LOOP_THRESHOLD]
        prints: list[Any] = []
        for entry in recent:
            extras = entry.get("extras") or {}
            if not isinstance(extras, dict):
                return None
            fp = extras.get("fingerprint")
            if fp is None:
                return None
            prints.append(fp)
        first = prints[0]
        if any(p != first for p in prints[1:]):
            return None
        proposed_fp = _baseline_params_fingerprint(proposed_params)
        if proposed_fp != first:
            return None
        error_class = recent[0].get("error_class") or "unknown"
        hint = (
            "the last "
            f"{_BASELINE_SELF_LOOP_THRESHOLD} `baseline` attempts failed "
            f"with the SAME params fingerprint (error_class={error_class!r}). "
            "Re-proposing the same params will fail the same way. Change at "
            "least one of: params.benchmark_script (sanitized *.sh name, "
            "e.g. \"sglang_mi300x.sh\" to bypass dsr1_fp8_mi300x.sh's "
            "hardcoded --result-dir), params.result_dir (sanitized path; "
            "Coordinator already defaults RESULT_DIR=<workspace>), or "
            "extra_server_args / extra_envs."
        )
        return PolicyDenied(
            "action='baseline' denied: same-fingerprint failure streak",
            rule="baseline_self_loop",
            hint=hint,
        )

    def _assess_remaining_gaps_throttle_denial(
        self,
    ) -> PolicyDenied | None:
        """IR-7 — throttle LLM-initiated ``assess_remaining_gaps``.

        Coordinator-internal dispatches bypass this rule (they go via
        :meth:`_enqueue_internal_steward_task` directly into
        TaskRegistry). The LLM may also propose ``assess_remaining_gaps``
        when it believes plateau is near, but back-to-back proposals
        within ``INFERENCE_OPTIMIZER_ASSESSMENT_MIN_INTERVAL_SEC``
        (default 1800) are denied to keep the dispatch budget bounded.

        Issue-A guard (KB cold-start regression): the action's
        ``actions/assess_remaining_gaps.md`` md preconditions
        (``phase == 'EXPLORE'`` AND ``len(optimization_stack) >= 3``)
        were previously documented but not enforced — letting the LLM
        propose steward as the very first action after baseline (when
        KB priors are missing) and route the session into
        ``no_more_leverage`` before any real exploration. We now deny
        the proposal until at least three stack entries have landed,
        so the steward can only fire on a real plateau judgment (which
        Coordinator-internal dispatch already gates on signals).
        """
        phase = (
            getattr(self.shared_state, "phase", "") or ""
        ).strip().upper()
        if phase != "EXPLORE":
            return PolicyDenied(
                f"action='assess_remaining_gaps' denied: phase={phase!r} "
                f"(steward is EXPLORE-only; see actions/"
                f"assess_remaining_gaps.md preconditions)",
                rule="assess_remaining_gaps_phase",
                hint=(
                    "Steward dispatch is gated to EXPLORE. Continue "
                    "the current phase or wait for the Coordinator's "
                    "internal plateau judge to fire on EXPLORE entry."
                ),
            )
        stack_len = len(getattr(self.shared_state, "optimization_stack", []) or [])
        if stack_len < 3:
            return PolicyDenied(
                f"action='assess_remaining_gaps' denied: "
                f"len(optimization_stack)={stack_len} < 3 "
                f"(steward needs material to assess; see actions/"
                f"assess_remaining_gaps.md preconditions)",
                rule="assess_remaining_gaps_min_stack",
                hint=(
                    "Run more EXPLORE rounds (specialist / params / "
                    "backends) until at least 3 stack entries have "
                    "promoted; the Coordinator's internal plateau "
                    "judge will enqueue the steward on its own when "
                    "v0.8 signals indicate exhaustion."
                ),
            )
        last = self.shared_state.last_remaining_gaps_assessment or {}
        if not isinstance(last, dict):
            return None
        last_ts = str(last.get("ts") or "").strip()
        if not last_ts:
            return None
        try:
            from datetime import datetime, timezone
            ts_dt = datetime.fromisoformat(last_ts)
            if ts_dt.tzinfo is None:
                ts_dt = ts_dt.replace(tzinfo=timezone.utc)
            now_dt = datetime.now(timezone.utc)
            elapsed = (now_dt - ts_dt).total_seconds()
        except (ValueError, TypeError):
            return None
        try:
            min_interval = float(os.environ.get(
                "INFERENCE_OPTIMIZER_ASSESSMENT_MIN_INTERVAL_SEC",
                "1800",
            ))
        except (TypeError, ValueError):
            min_interval = 1800.0
        if elapsed < min_interval:
            return PolicyDenied(
                f"action='assess_remaining_gaps' denied: throttled "
                f"(last assessment {elapsed:.0f}s ago, min interval "
                f"{int(min_interval)}s)",
                rule="assess_remaining_gaps_throttle",
                hint=(
                    "Last steward assessment was recent. Wait for the "
                    "Coordinator to enqueue the next internal steward "
                    "on plateau, OR continue running variants until "
                    "the throttle window elapses."
                ),
            )
        return None

    def _sequence_denial_for_action(
        self,
        action_name: str,
        proposed_params: dict[str, Any] | None = None,
    ) -> PolicyDenied | None:
        """Reject orchestration action/delegate attempts that skip required steps.

        Once optimization_stack has unvalidated KEEPs, the next
        ``explore`` round must carry the inlined stack rebench
        (PolicyGate rule ``stack_rebench_required``). This function
        only enforces the cross-action ordering (target_analysis
        before baseline, baseline before everything else, baseline
        self-loop guard, profile/integrate kernel-agent guards).

        ``proposed_params`` is the ``intent.payload["params"]`` dict
        (propose_action / delegate path). Currently only consumed by
        the baseline self-loop guard above, but the kwarg signature is
        kept open so other per-action stop-losses can plug in without
        further call-site churn.
        """
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
        # target_analysis hard gate: target_baseline.json must be on disk
        # before any other sequence action runs. The executor never raises
        # (failures land as `status=fetch_error`, missing GPU lands as
        # `reason=no_target_gpu_configured`) so this gate always opens
        # after a single attempt regardless of whether
        # --compare-against-gpu was supplied.
        if (
            not self._target_analysis_baseline_exists()
            and action != "target_analysis"
        ):
            if self._compare_against_gpu:
                hint = (
                    "propose/delegate `target_analysis` so InferenceX "
                    f"reference for --compare-against-gpu="
                    f"{self._compare_against_gpu!r} is fetched into "
                    "$SESSION_DIR/target_analysis/target_baseline.json"
                )
            else:
                hint = (
                    "propose/delegate `target_analysis` so a "
                    "reason='no_target_gpu_configured' marker JSON is "
                    "written to $SESSION_DIR/target_analysis/"
                    "target_baseline.json (no --compare-against-gpu was "
                    "supplied; the marker is what unblocks the rest of "
                    "the pipeline)"
                )
            return PolicyDenied(
                f"action={action!r} denied: target_analysis must run first",
                rule="execution_order",
                hint=hint,
            )
        if (
            self.shared_state.baseline_tput <= 0
            and action not in {"baseline", "target_analysis"}
        ):
            return PolicyDenied(
                f"action={action!r} denied: baseline must run first",
                rule="execution_order",
                hint="propose/delegate `baseline` until baseline_tput > 0",
            )
        if action == "baseline":
            self_loop = self._baseline_self_loop_denial(proposed_params)
            if self_loop is not None:
                return self_loop
        # Profile / integrate guards only apply when kernel agent is in
        # the role registry — no-kernel mode skips them.
        # ``trace_analyze`` is enforced at the REQUEST layer
        # (``_sequence_denial_for_request``) for ``run_optimization``
        # only. ``params`` / ``backends`` / ``sweep`` / ``report`` are
        # never gated on a fresh ``last_trace_analyze`` cache — those
        # actions don't need kernel candidates to make progress.
        if "kernel" in self.role_registry:
            if (
                self.shared_state.baseline_tput > 0
                and not self.shared_state.last_profile_trace
                and action != "profile"
            ):
                return PolicyDenied(
                    f"action={action!r} denied: profile must run before {action!r}",
                    rule="execution_order",
                    hint=(
                        "wait for the Coordinator-internal analysis task "
                        "(auto-enqueued at PRELUDE / on every +10% watermark "
                        "crossing) to populate ``last_profile_trace``; "
                        "PolicyGate denies LLM-proposed `profile` / "
                        "`roofline` with "
                        "``rule='analysis_action_not_llm_proposable'``. If "
                        "``auto_roofline_pending_task_id`` is stuck on a "
                        "failed task, emit `recover`."
                    ),
                )
            # integrate gate: kernel_opt KEEP awaiting integrate. Allow
            # integrate / report through; recover is not in
            # ``sequence_actions`` and therefore already bypasses this
            # function (no-op early return).
            pending_kid = self._kernel_opt_keep_pending()
            if pending_kid and action not in {"integrate", "report"}:
                return PolicyDenied(
                    f"action={action!r} denied: integrate must run first",
                    rule="execution_order",
                    hint=(
                        f"kernel_opt returned KEEP for kernel_id="
                        f"{pending_kid!r}; emit request{{target_agent="
                        "'kernel', kind='integrate', params={kernel_id: "
                        f"{pending_kid!r}}}}} before any further explore"
                    ),
                )
        # Hot-kernel report gate. Block ``report`` when any reusable hot
        # kernel with gpu_pct >= 3% has not yet been tried, rejected, or
        # integrated.
        # Allowed through the gate:
        #   - kernel_opt request itself (handled at request layer)
        #   - integrate (still needs to drain prior KEEPs; the
        #     integrate-pending gate above already handles ordering)
        #   - recover (not in sequence_actions, bypasses entirely)
        # Blocked:
        #   - report -- the LLM cannot declare the session done
        #     while a meaningful kernel lever exists.
        #
        # Fire only when ``run_optimization`` is dispatchable to avoid
        # trapping the LLM between opposing gates.
        if action == "report" and self._kernel_opt_unlocked():
            untried = self.shared_state.untried_hot_reusable_kernels()
            if untried:
                untried_str = ", ".join(untried)
                return PolicyDenied(
                    f"action='report' denied: untried hot reusable "
                    f"kernels still present ({untried_str})",
                    rule="hot_kernel_unfinished",
                    hint=(
                        "Every reusable hot kernel with gpu_pct >= 3% "
                        "must get at least one kernel_opt attempt (or "
                        "be retired via REVERT / max_failures) before "
                        f"the session may end. Pending: {untried_str}. "
                        "Emit request{target_agent='kernel', "
                        "kind='run_optimization', "
                        "params={candidates_path=<from "
                        "last_trace_analyze>}} so the batch fans out "
                        "across the queue. Threshold overrides: "
                        "HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT / "
                        "HYPERLOOM_KERNEL_OPT_GATE_TOP_N."
                    ),
                )
        # stack-rebench precedence. Once a
        # KEEP lands on optimization_stack we want the next action
        # that touches the stack to also revalidate it. The merged
        # ``explore`` action carries an *inlined* per-KEEP stack
        # rebench, so we allow it through
        # alongside ``baseline`` (ad-hoc re-baseline) and ``report``
        # (the wind-down).
        if (
            self.shared_state.optimization_stack_has_unvalidated_keeps()
            and action not in {"explore", "baseline", "report"}
        ):
            return PolicyDenied(
                f"action={action!r} denied: stack rebench required first",
                rule="stack_rebench_required",
                hint=(
                    "optimization_stack has KEEPs that have not been "
                    "re-validated end-to-end; propose/delegate "
                    "`explore` (its per-KEEP stack rebench is "
                    "inlined) or `report` before any further action"
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
        # ``trace_analyze`` IS the prerequisite request: it produces the
        # ``last_trace_analyze`` cache the rest of the chain consults.
        # It is also used directly by tests / tools passing an explicit
        # ``trace_input``, so allow it through; later explore/sweep
        # actions are guarded until the result is cached in SharedState.
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
        if not self.shared_state.last_profile_trace:
            return PolicyDenied(
                f"request kind={req_kind!r} denied: profile must run first",
                rule="execution_order",
                hint=(
                    "wait for the Coordinator-internal analysis task to "
                    "populate ``last_profile_trace`` before requesting "
                    "trace_analyze / run_optimization; analysis is "
                    "auto-enqueued at PRELUDE and on every +10% watermark "
                    "crossing. The analysis lane is Coordinator-owned and not "
                    "LLM-proposable (PolicyGate denies with "
                    "``rule='analysis_action_not_llm_proposable'``). If "
                    "``auto_roofline_pending_task_id`` is stuck, emit "
                    "`recover` as the escape hatch."
                ),
            )
        select = self.shared_state.last_trace_analyze or {}
        needs_select = select.get("trace_input") != self.shared_state.last_profile_trace
        if needs_select and req_kind not in {"trace_analyze", "run_gemm_tuning"}:
            return PolicyDenied(
                f"request kind={req_kind!r} denied: trace_analyze must run first",
                rule="execution_order",
                hint="emit request kind='trace_analyze' for last_profile_trace",
            )
        if req_kind == "run_optimization" and self._gemm_tuning_required_before_kernel_opt():
            return PolicyDenied(
                "request kind='run_optimization' denied: FP8 GEMM tuning must run first",
                rule="execution_order",
                hint="emit request kind='run_gemm_tuning' before run_optimization",
            )
        return None

    @staticmethod
    def _allow_early_kernel_opt() -> bool:
        """Escape hatch — ``INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT=1``
        opens the kernel_opt request gate unconditionally (skips the
        roofline-snapshot check). Used by v0-baseline-comparison flows
        and unit tests."""
        return os.environ.get(
            "INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", "",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _kernel_opt_unlocked(self) -> bool:
        """Return True when ``run_optimization`` is actually dispatchable
        right now — i.e. when the hot-kernel report-gate (PR-C) should
        fire and the LLM should be pushed toward kernel_opt.

        The gate is open when ANY of the following hold:
          * escape hatch env set;
          * a roofline snapshot exists (``roofline_snapshot_id`` >= 1).
        """
        if self._allow_early_kernel_opt():
            return True
        ss = self.shared_state
        ta = ss.last_trace_analyze or {}
        snapshot_id = ta.get("roofline_snapshot_id", 0)
        if not isinstance(snapshot_id, int) or snapshot_id < 1:
            return False
        return True

    @staticmethod
    def _skip_gemm_tuning() -> bool:
        return os.environ.get(
            "INFERENCE_OPTIMIZER_SKIP_GEMM_TUNING", "",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _gemm_tuning_required_before_kernel_opt(self) -> bool:
        """Return True when FP8 SGLang GEMM tuning should run first."""
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

    # ==================================================================
    # Intent handling
    # ==================================================================
    async def _handle_intent(self, source: str, intent: Intent) -> None:
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
                # Terminal intent of a specialist task. PolicyGate R3
                # has already validated the
                # ``from_agent='specialist:<task_id>'`` prefix + payload
                # schema + gap/domain match; the handler only does
                # bookkeeping.  The current SpecialistRunner architecture
                # captures the done intent inside the runner instead of
                # routing it via the bus, so this branch is mostly
                # defense-in-depth + future-proofing (an out-of-band path
                # — e.g. an operator script replaying a transcript — still
                # converges on the same handler).
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
        # Gate proposals on a pending auto-roofline / auto-profile task
        # *before* paying for the Critic round-trip. This is the cheap
        # path; ``_materialize_approved_proposal`` carries a symmetric
        # check for the race where the watermark trips while a proposal
        # is already in front of the Critic.
        roofline_denied = await self._roofline_denial_for_action(action_name)
        if roofline_denied is not None:
            await self._record_policy_denied(
                source, intent, roofline_denied, action_name=action_name,
            )
            return
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
        # KB hypothesize/verify protocol retired — proposals enter the
        # pending queue directly; KEEP/REVERT facts are written by
        # ``_fact_write_hook`` after the task lands (see below).
        self.state.pending_proposals[msg.msg_id] = pending

    def _resolve_issue_canonical(self, pending: PendingProposal) -> str:
        """Find the issue_node canonical_id this proposal addresses.

        Priority:

        1. ``pending.payload['gap_canonical_id']`` — explicit
           top-level gap reference (set by Orchestration when it
           routes a specialist proposal to a known gap).
        2. ``pending.payload['params']['gap_canonical_id']`` — same
           thing under ``params`` (where most other proposal fields
           live).
        3. Fallback: :meth:`_gap_anchor_canonical_id` — the M1
           ``workload.<model>.<gpu>`` anchor. Gap-09 will plug
           ``SharedState.gaps[i].canonical_id`` (matched by domain
           / phase) into this fallback once the field lands.
        """
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
        """Canonical 5-tuple recipe id for the current workload.

        KEEP / REVERT / CLOSE all amend the recipe row keyed by this
        id (via :meth:`_kb_amend_recipe`). It MUST stay consistent
        with ``cortex_t0.run_t0_anchor``'s derivation so the row that
        warm-start anchored is exactly the row the writes land on.
        ``run_t0_anchor`` stamps the resolved ``framework`` /
        ``framework_version`` back onto SharedState, so reading them
        here yields the same values T0 used (no write/read cid split
        when the operator didn't pass ``--framework-version``).
        """
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
        """M1 placeholder for the gap anchor.

        Per "M1 simplification — use ``workload_node.canonical_id`` as
        the from side of every hypothesize edge". M5 specialist
        framework will introduce real ``issue_node`` anchors keyed by
        gap descriptors. Delegates to :meth:`_workload_canonical_id`
        so the gap anchor and the KEEP/REVERT/CLOSE write target never
        diverge (framework is plumbed so sglang and vLLM gaps anchor
        on different recipe rows).
        """
        return self._workload_canonical_id()

    def _kb_amend_recipe(
        self,
        *,
        append_lesson: dict[str, Any] | None = None,
        append_pitfall: dict[str, Any] | None = None,
        recipe_overrides: dict[str, Any] | None = None,
        provenance_details: dict[str, Any] | None = None,
    ) -> None:
        """Read-modify-write helper for the v2 recipe-snapshot KB.

        Single replacement for the legacy v1 ``propose_lesson`` /
        ``propose_pitfall`` / ``update_recipe`` calls. Loads the
        live recipe row for the current 5-tuple, appends the
        supplied lesson / pitfall (each is a single dict — the
        caller picks the schema), merges any explicit
        ``recipe_overrides``, and writes the row back. Best-effort:
        any failure is logged + swallowed (the optimization journal
        is a separate audit trail).

        ``recipe_overrides`` follows the LocalRecipeStore.put_recipe
        kwarg shape (``best_config`` / ``best_throughput`` /
        ``what_worked`` / ``sessions`` / ``stack_fingerprint`` /
        ``last_profiled`` / ``extras`` / ...). Anything not provided
        is preserved from the live row.

        Per the user's design choice (see commit 4d): lesson /
        pitfall are appended to the recipe's array WITHOUT cross-
        recipe deduplication. The same statement may appear in
        multiple 5-tuple rows; that's intentional.
        """
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

        # Read the LOCAL authoritative row for the read-modify-write.
        # We deliberately bypass the remote-first dispatcher read here:
        # writes are local-only, so the row we must NOT clobber is the
        # one this session already wrote locally. A remote-first read
        # could return an older/wider central row and silently drop the
        # lessons / pitfalls this session just appended.
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

        # Build the put_recipe kwargs — preserve everything from live
        # that the caller didn't override.
        overrides = dict(recipe_overrides or {})
        # Preserve T0-stamped top-level extras (model_class /
        # image_digest / tp / ep / ...) across the amend: ``live`` has
        # them splatted at the top level, so pull the non-reserved keys
        # back out and merge the caller's extras on top (caller wins).
        # Keys mirror ``recipe_kb.schema.Recipe`` well-known fields.
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
        # Re-stamp the config.json architecture-identity tags
        # (``architectures`` / ``model_type``) on every amend so the row
        # always carries the current model's identity. Carried on
        # SharedState by ``cli._load_model_config_tags``; skipped when
        # unset so a prior T0-stamped value is preserved.
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
            # Preserve audit fields across the amend instead of letting
            # put_recipe reset them to its defaults each time.
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
        """Apply a Critic ``review_verdict`` to its target proposal.

        Every proposal (kernel_opt / integrate / report / specialist
        dispatch / directly-proposed explore) is decided by a single
        verdict. The envelope schema still accepts the legacy
        per-variant ``verdict_map`` shape; when one arrives we collapse
        it to a summary verdict (``approve`` if any variant is approved,
        else ``reject`` if any is rejected, else ``needs_review``) and
        take the same single-verdict path.
        """
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
        """Legacy v0.6 single-verdict handler.

        Used for one-proposal reviews (kernel_opt / integrate /
        report / specialist dispatch / any non-grid action). The
        ``approve`` branch materialises the whole proposal as-is.

        The integrate_patch Critic gate mirrors verdicts: when the
        pending proposal is an ``integrate_patch`` for
        a specialist whose patches the Critic just reviewed, the
        verdict is mirrored onto ``SharedState.specialist_patch_verdicts``
        so PolicyGate's ``integrate_patch_requires_critic_verdict``
        rule can read it on the next delegate. Same mirror runs for a
        verdict targeting the specialist itself, so an orchestrator
        can review a specialist's patches by proposing the underlying
        ``specialist`` action and reading the resulting verdict.
        """
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
            # A critic verdict on the specialist proposal itself is
            # treated as the verdict on the patches that specialist
            # will (or did) produce — the specialist's task_id is the
            # natural key.
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
        # When the verdict targets a dyn_id (routed via the synthesised
        # ``dyn-<id>`` specialist_task_id), persist the
        # ``critic_verdict.json`` envelope and flip the dyn_id lifecycle
        # status so reject/revise short-circuits the integrate dispatch.
        try:
            self._mirror_critic_verdict_to_dynamic_action(
                pending=pending, verdict=verdict, reasoning=reasoning,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action: critic-verdict mirror failed for "
                "proposal_msg_id=%s", pending.proposal_msg_id,
            )
        if verdict == "approve":
            await self._materialize_approved_proposal(pending)

    def _inject_explore_runtime_params(self, params: dict) -> None:
        """Inject explore-task operational knobs from SharedState into ``params``.

        Called from both ``_materialize_approved_proposal`` (the
        propose/Critic single-verdict path) and ``_handle_delegate``
        (the direct-delegate path explore grids take) so a single source
        of truth controls what gets forwarded to the ExploreExecutor.
        ``setdefault`` preserves any LLM-supplied override for one-off
        rebench / debug variants.

        Knobs:
          * ``baseline_runtime_sec`` + ``explore_overtime_kill_ratio`` —
            Fix E (Q1/Q5): the executor derives ``soft_deadline_sec =
            baseline_runtime_sec * explore_overtime_kill_ratio`` for the
            single-variant phase (stack rebench bypasses this gate per Q4).
          * ``variant_timeout_sec`` — operator-pinned hard timeout
            (``0`` = auto-derive in ExploreExecutor).
          * ``variant_timeout_safety_margin`` — auto-derive headroom (no
            effect when ``variant_timeout_sec`` is pinned above).
          * ``roofline_hard_gate`` (+ snapshot) — opt-in saturation filter;
            soft advisory is independent and untouched here.
        """
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
        if bool(getattr(
            self.shared_state, "explore_roofline_hard_gate", False,
        )):
            params.setdefault("roofline_hard_gate", True)
            history = list(getattr(
                self.shared_state, "roofline_saturation_history", [],
            ) or [])
            if history and isinstance(history[-1], dict):
                params.setdefault(
                    "roofline_saturation_snapshot", dict(history[-1]),
                )
        # Thread the persisted explore_search ledger so the ExploreExecutor's
        # canonical_fingerprint dedup has cross-turn memory. Without it,
        # params["explore_search"] is unset every round: the executor restarts
        # dedup from an empty ledger (re-benching an already-tested (args, envs)
        # under a renamed variant that escapes dedup), and the single-round
        # tested/rejected it returns overwrite the persisted ledger via
        # apply_explore_search_update — starving the specialist prompt's
        # exhausted-knob context. setdefault keeps an explicit override.
        es = getattr(self.shared_state, "explore_search", None)
        if isinstance(es, dict) and es.get("tested"):
            params.setdefault("explore_search", es)

    async def _materialize_approved_proposal(
        self,
        pending: PendingProposal,
        *,
        approved_variant_names: set[str] | None = None,
    ) -> None:
        """Promote an approved proposal into a TaskRegistry entry.

        For grid-style executors (backends / params / sweep) we inject the
        current best throughput as ``base_tput`` so they can compute
        gain%; otherwise the runner's default of 0.0 makes
        best_gain_pct uninformative (DESIGN §16 baseline_tput parameter).

        ``approved_variant_names``: when set, filter the ``explore``
        grid down to the named subset before materialising. Only the
        roofline defer/resume path carries a non-``None`` value (it
        preserves a previously-filtered subset across a deferred
        dispatch); the live single-verdict path passes ``None`` and
        keeps the full grid.
        """
        # Safety net for the race where the watermark fires while a
        # proposal is already in front of the Critic.
        # ``_handle_propose_action`` carries the same check (the cheaper
        # path); this one catches proposals the Critic approved between
        # the watermark crossing and the dispatch tick. Defer rather
        # than drop — the Critic already approved, so re-running the
        # round-trip would be wasted budget.
        roofline_denied = await self._roofline_denial_for_action(
            pending.action_name,
        )
        if roofline_denied is not None:
            await self._defer_approved_proposal_for_roofline(
                pending, approved_variant_names,
            )
            return
        params = dict(pending.payload.get("params") or {})
        # Filter the grid down to the Critic-approved subset. Variant
        # traceability is carried by the local journal + KB fact-write
        # ``source_session_id`` / ``source_task_id`` attrs.
        if (
            pending.action_name == "explore"
            and isinstance(params.get("grid"), list)
        ):
            stamped_grid: list[dict[str, Any]] = []
            for variant in params["grid"]:
                if not isinstance(variant, dict):
                    # Non-dict slots can't carry a name, so they are
                    # *always* dropped when a
                    # variant filter is in effect (no way to match
                    # them); pass-through otherwise so legacy
                    # callers keep working.
                    if approved_variant_names is None:
                        stamped_grid.append(variant)
                    continue
                vname = str(variant.get("name") or "").strip()
                # drop variants the Critic
                # rejected before they ever hit the executor.
                if (
                    approved_variant_names is not None
                    and vname not in approved_variant_names
                ):
                    continue
                stamped_grid.append(dict(variant))
            params["grid"] = stamped_grid
            # Audit hint for the executor: how many variants the
            # Critic filtered. Useful for the breakdown to surface
            # "critic_filtered_count" alongside the explore round
            # row without re-walking the bus.
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
            # Profile itself does not yet consume base_extra_args, but we
            # stamp it onto the task so the post-task promotion records the
            # server config that produced this trace.
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
            # Mirror the sweep/integrate branches: inject ``base_tput`` (and
            # ``base_extra_args``) tied to current_best (or baseline_tput as
            # fallback) whenever Orchestration omits them. Without this the
            # ExploreExecutor sees ``base_tput=0``, ``_gain_pct`` returns
            # ``None`` for every variant, and the KEEP/REVERT ladder is
            # skipped — every variant lands in ``FAILED`` regardless of how
            # far it beat baseline. Explicit operator value still wins via
            # setdefault.
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
            # Defensive path — `approved-{proposal_msg_id}` is unique per
            # proposal so a normal flow never collides. The only realistic
            # trigger is a resume / bus replay surfacing an old proposal
            # whose task is already on disk. Skip emitting a fresh
            # `approved_proposal` decision (which would mislead Orch into
            # thinking new work was queued) and record an observation so
            # the audit log still shows what happened.
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
        # ``proposal_msg_id`` is the resume contract for the deferred
        # queue (see ``replay_for_resume``): a ``proposal_materialize_blocked``
        # observation paired with a later ``approved_proposal`` decision
        # for the same proposal_msg_id is interpreted as "drained
        # successfully" and skipped on restart. Without this field
        # the link was task_id-only, which replay cannot reverse.
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "decision",
            {"kind": "approved_proposal", "task_id": task.task_id,
             "action_name": pending.action_name, "from_agent": pending.from_agent,
             "proposal_msg_id": pending.proposal_msg_id},
        ))

    # ------------------------------------------------------------------
    # DELEGATE
    # ------------------------------------------------------------------
    async def _handle_delegate(self, source: str, intent: Intent) -> None:
        action_name = intent.payload["action_name"]
        if self.shared_state.is_pruned(action_name):
            await self._record_policy_denied(
                source, intent,
                PolicyDenied(
                    f"delegate{{action_name={action_name!r}}} pruned",
                    rule="family_pruned",
                    hint=(
                        f"{action_name!r} is in pruned_families; pick another "
                        f"phase-allowed action (see the `=== Phase-allowed "
                        f"actions ===` block)"
                    ),
                ),
                action_name=action_name,
            )
            return
        denied = self._sequence_denial_for_action(
            action_name,
            proposed_params=intent.payload.get("params"),
        )
        if denied is not None:
            await self._record_policy_denied(
                source, intent, denied, action_name=action_name,
            )
            return
        # ``delegate{action_name='explore', params={grid: [...]}}`` runs the
        # variants directly: each variant is benchmarked and judged by the
        # explore KEEP threshold + canonical_fingerprint dedup. Config/env
        # grids are not source patches, so no per-variant Critic pre-review
        # stands between the delegate and the executor.
        params = dict(intent.payload.get("params") or {})
        # The schema says delegate idempotency_key is top-level, but LLMs
        # sometimes place it inside params (especially when following older
        # examples like params={grid: ..., idempotency_key: ...}). Treat the
        # nested value as a compatibility alias and remove it from executor
        # params so downstream action runners never see this control field.
        nested_idempotency_key = params.pop("idempotency_key", None)
        # Plumb baseline's materialized YAML into grid-style delegated tasks
        # so they inherit the workload contract (CONC/ISL/OSL/TP/...) baseline
        # ran. See `_materialize_approved_proposal` for the same logic on the
        # proposal/review path. `setdefault` lets the delegator override.
        if (
            action_name in ("sweep", "explore")
            and self.shared_state.baseline_config_path
        ):
            params.setdefault(
                "config_path", self.shared_state.baseline_config_path
            )
        # Parity with _materialize_approved_proposal: direct delegates
        # bypass the verdict map (legacy resume, tests that delegate
        # explore directly) but still need the same operational knobs.
        if action_name == "explore":
            self._inject_explore_runtime_params(params)
        # ``assess_remaining_gaps`` is a thin wrapper: rewrite
        # the kind to ``specialist`` and force the
        # ``session_steward_specialist`` domain (LLM cannot pick any
        # other domain via this action). Throttle is checked separately
        # via _assess_remaining_gaps_throttle_denial above.
        if action_name == "assess_remaining_gaps":
            throttle = self._assess_remaining_gaps_throttle_denial()
            if throttle is not None:
                await self._record_policy_denied(
                    source, intent, throttle, action_name=action_name,
                )
                return
            # Force the steward domain + a deterministic gap id so
            # PolicyGate R2 passes. Preserve params.reason verbatim
            # so the audit trail captures why the LLM proposed.
            llm_reason = str(params.get("reason") or "").strip()
            round_id = int(
                (self.shared_state.explore_search or {}).get("cursor") or 0
            )
            params["domain"] = "session_steward_specialist"
            params["gap_canonical_id"] = (
                params.get("gap_canonical_id")
                or f"gap.steward.round{round_id}.llm"
            )
            params["gap_symptom"] = (
                "LLM-initiated steward assessment (reason="
                f"{llm_reason or 'unspecified'!r})"
            )
            params.setdefault("gap_layer", "session_strategy")
            params.setdefault("max_turns", 8)
            params.setdefault("source", "orchestration")
            params["reason"] = llm_reason or "llm_uncertainty"
            # Rewire to the specialist task kind so the standard
            # SpecialistRunner picks it up.
            action_name = "specialist"

        # Auto-roofline gate. When the Coordinator has an in-flight
        # roofline task (PRELUDE bootstrap or 10% watermark crossing),
        # Refuse any dispatch whose output should observe the fresh
        # ``analysis.md`` / ``last_profile_trace`` snapshot before it
        # runs. The gated action set is the module-level
        # ``_ROOFLINE_GATED_ACTIONS`` so propose_action / materialize
        # paths gate against the exact same names.
        denied = await self._roofline_denial_for_action(action_name)
        if denied is not None:
            await self._record_policy_denied(
                source, intent, denied, action_name=action_name,
            )
            return

        # Specialist pre-dispatch warmup.
        # When the Orchestration role delegates a specialist task, the
        # Coordinator is the only place with the KnowledgePlane facade
        # in scope. Warm the prompt's external-knowledge sections here
        # so SpecialistRunner's prompt assembly sees them via task
        # params. ``setdefault`` lets the caller (Orchestration) pre-
        # supply values; we only fill the gaps.
        if action_name == "specialist":
            await self._warm_specialist_params(params)
        # For dynamic_action: generate the dyn_id, mkdir the artefact
        # dir, and write spec.json + seed_kit.json before the task is
        # enqueued. PolicyGate has already validated payload + cap;
        # on seed-kit assembly failure the dispatch is rolled back
        # (no task, no round-cap bump). The shared-state record is
        # written only after the task is successfully queued.
        dynamic_action_dispatch_meta: dict[str, Any] | None = None
        if action_name == DYNAMIC_ACTION_NAME:
            try:
                dynamic_action_dispatch_meta = (
                    self._prepare_dynamic_action_dispatch(params)
                )
            except PolicyDenied as denied:
                await self._record_policy_denied(
                    source, intent, denied, action_name=action_name,
                )
                return
        # Idempotency-key resolution chain (most-explicit first):
        #   1. ``intent.payload.idempotency_key`` — schema-correct top-level
        #      placement.
        #   2. ``nested_idempotency_key`` — compatibility for LLM outputs that
        #      accidentally nest the field under ``params={...}`` (HEAD
        #      commit 56840aa; LLMs follow stale prompt examples).
        #   3. Content-fingerprint auto-generated key
        #      ``<source>:<action>:t<tick>:<sha1[:10]>`` — only when neither
        #      explicit form is supplied. Hashing ``params`` keeps the same
        #      operation+inputs collapsing to one task across re-emissions.
        # When the resolved key collides with an existing task in a
        # *terminal* state, the retry loop below re-tries up to 5 times with
        # ``-retry<N>`` suffixes so the operator can resubmit identical
        # params without manual bumping. Collisions with non-terminal tasks
        # are surfaced as policy_denied so the LLM waits for the existing
        # delegated_result instead of spinning.
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
        if was_existing:
            # Should not happen after the retry loop unless create failed.
            await self._record_policy_denied(
                source, intent,
                PolicyDenied(
                    f"delegate{{action_name={action_name!r}}} duplicate "
                    f"idempotency_key={idempotency_key!r}",
                    rule="duplicate_idempotency_key",
                    hint="unexpected duplicate after retry loop",
                ),
                action_name=action_name,
            )
            return
        self.shared_state.reset_policy_denial_streak(action_name)
        if (
            action_name == DYNAMIC_ACTION_NAME
            and dynamic_action_dispatch_meta is not None
        ):
            try:
                self._finalize_dynamic_action_dispatch(
                    task_id=task.task_id,
                    meta=dynamic_action_dispatch_meta,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: finalize hook failed for task=%s",
                    task.task_id,
                )
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "event",
            {"kind": "task_queued", "task_id": task.task_id, "source": source, "action": action_name},
        ))

    # ------------------------------------------------------------------
    # dynamic_action dispatch hooks
    # ------------------------------------------------------------------
    def _prepare_dynamic_action_dispatch(
        self, params: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the dispatch-time artefact bundle for one dispatch.

        Steps:

        1. Generate ``dyn-<round>-<seq>`` and fail-fast on collision.
        2. Mkdir the artefact dir.
        3. Assemble the seed kit; a :class:`SeedKitAssemblyError` is
           re-raised as :class:`PolicyDenied` so the round cap is not
           consumed.
        4. Write ``spec.json`` + ``seed_kit.json``.
        5. Inject ``dyn_id`` / ``artifact_path`` / ``spec_path`` /
           ``seed_kit_path`` into ``params`` so the executor can
           locate the artefacts without traversal.

        Returns the meta dict consumed by
        :meth:`_finalize_dynamic_action_dispatch` once the task lands.
        Any failure removes the partial artefact dir and raises a
        structured :class:`PolicyDenied`.
        """
        from .dynamic_action_seed_kit import (
            SeedKitAssemblyError,
            assemble_seed_kit,
        )
        from ..session_paths import (
            dynamic_action_artifact_dir,
            dynamic_action_seed_kit_path,
            dynamic_action_spec_path,
        )

        state = self.shared_state
        round_id = self._dynamic_action_round_id()
        seq = int(
            getattr(state, "dynamic_action_round_count", 0) or 0
        ) + 1
        dyn_id = f"dyn-{round_id}-{seq}"
        if dyn_id in (getattr(state, "dynamic_actions", None) or {}):
            raise PolicyDenied(
                f"dynamic_action: dyn_id={dyn_id!r} already exists in "
                f"SharedState.dynamic_actions (fail-fast on collision; "
                f"likely a stale round-cap reset or resume race).",
                rule="dynamic_dyn_id_collision",
                hint=(
                    "Coordinator should advance ``explore_search.cursor`` "
                    "before re-dispatching; if this fires post-resume, the "
                    "ABANDONED sweep in P8 should run first."
                ),
            )
        dispatched_at = datetime.now(timezone.utc).isoformat(
            timespec="microseconds",
        )
        payload_snapshot = {
            "motivation_gap_text": str(
                params.get("motivation_gap_text") or "",
            ),
            "scope_domains": list(params.get("scope_domains") or ()),
            "side_effects_declared": list(
                params.get("side_effects_declared") or (),
            ),
            "budget_hint": str(params.get("budget_hint") or "medium"),
        }
        policy_gate_decision = {
            "rules_evaluated": [
                "dynamic_phase_violation",
                "dynamic_source_violation",
                "dynamic_payload_schema",
                "dynamic_scope_too_narrow",
                "dynamic_scope_unknown_domain",
                "dynamic_side_effects_red_line",
                "dynamic_kernel_only_disallowed",
                "dynamic_round_cap_exhausted",
            ],
            "verdict": "approve",
        }
        artifact_dir = (
            dynamic_action_artifact_dir(self.session_dir, dyn_id)
            if self.session_dir is not None else None
        )
        seed_kit_result = None
        try:
            seed_kit_result = assemble_seed_kit(state, payload_snapshot)
            if seed_kit_result.degraded:
                log.info(
                    "dynamic_action: seed kit DEGRADED for dyn_id=%s "
                    "(missing one or more of roofline / profile / "
                    "kept_patches / kb_pitfalls — see "
                    "dynamic_action_seed_kit.assemble_seed_kit for the "
                    "fallback rules). Sub-agent will still run.",
                    dyn_id,
                )
        except SeedKitAssemblyError as exc:
            raise PolicyDenied(
                f"dynamic_action: seed kit assembly failed for "
                f"dyn_id={dyn_id!r}: {exc}",
                rule="dynamic_seed_kit_assembly_failed",
                hint=(
                    "Seed kit invariants (closed schema + ≤8K token cap) "
                    "were violated; the dispatch is rolled back. Reduce "
                    "the scope or wait for state to drain before retrying."
                ),
            ) from exc

        spec = {
            "dyn_id": dyn_id,
            "dispatched_at": dispatched_at,
            "round_index": round_id,
            "payload": payload_snapshot,
            "policy_gate_decision": policy_gate_decision,
            "resource_lane": RESEARCH_LANE_NAME,
            "degraded_dispatch": bool(seed_kit_result.degraded),
            "seed_kit_tokens": int(seed_kit_result.total_tokens),
        }
        if artifact_dir is not None:
            try:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                dynamic_action_spec_path(self.session_dir, dyn_id).write_text(
                    json.dumps(spec, sort_keys=True, indent=2),
                    encoding="utf-8",
                )
                dynamic_action_seed_kit_path(
                    self.session_dir, dyn_id,
                ).write_text(
                    json.dumps(
                        seed_kit_result.payload, sort_keys=True, indent=2,
                    ),
                    encoding="utf-8",
                )
            except OSError as exc:
                self._cleanup_dynamic_action_artifact_dir(dyn_id)
                raise PolicyDenied(
                    f"dynamic_action: failed to persist artefacts for "
                    f"dyn_id={dyn_id!r}: {exc!r}",
                    rule="dynamic_artifact_write_failed",
                    hint=(
                        "Coordinator could not write spec.json / "
                        "seed_kit.json to the session dir; check disk "
                        "/ permissions and retry."
                    ),
                ) from exc

        params["dyn_id"] = dyn_id
        if artifact_dir is not None:
            params["artifact_path"] = str(artifact_dir)
            params["spec_path"] = str(
                dynamic_action_spec_path(self.session_dir, dyn_id),
            )
            params["seed_kit_path"] = str(
                dynamic_action_seed_kit_path(self.session_dir, dyn_id),
            )

        # Append the ``DISPATCHED`` row to dispatch_history.jsonl.
        from .dynamic_action_history import (
            DispatchHistoryEvent,
            append_dispatch_history_row,
        )
        if self.session_dir is not None:
            try:
                append_dispatch_history_row(
                    session_dir=self.session_dir,
                    dyn_id=dyn_id,
                    event=DispatchHistoryEvent.DISPATCHED,
                    payload={
                        "round_index": round_id,
                        "scope_domains": list(payload_snapshot["scope_domains"]),
                        "side_effects_declared": list(
                            payload_snapshot["side_effects_declared"],
                        ),
                        "budget_hint": payload_snapshot["budget_hint"],
                        "degraded_dispatch": bool(spec["degraded_dispatch"]),
                        "seed_kit_tokens": int(spec["seed_kit_tokens"]),
                    },
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: dispatch_history DISPATCHED write "
                    "failed for dyn_id=%s", dyn_id,
                )

        # Populate the closed prompt-projection fields at dispatch
        # time; anything beyond ``SUMMARY_PROMPT_FIELDS`` is private
        # audit metadata that lives on disk only.
        from .dynamic_action_proposal import (
            DynamicActionStatus,
            LAST_OUTCOME_BY_STATUS,
            MOTIVATION_GAP_SHORT_MAX_CHARS,
        )
        motivation_short = payload_snapshot["motivation_gap_text"].strip()
        if len(motivation_short) > MOTIVATION_GAP_SHORT_MAX_CHARS:
            motivation_short = (
                motivation_short[: MOTIVATION_GAP_SHORT_MAX_CHARS - 3].rstrip()
                + "..."
            )
        summary = {
            "dyn_id": dyn_id,
            "round_index": round_id,
            "status": DynamicActionStatus.DISPATCHED.value,
            "last_outcome": LAST_OUTCOME_BY_STATUS[DynamicActionStatus.DISPATCHED],
            "dispatched_at": dispatched_at,
            "scope_domains": payload_snapshot["scope_domains"],
            "motivation_gap_short": motivation_short,
            "verdict": None,
            "cumulative_gain": None,
            "artifact_path": str(artifact_dir) if artifact_dir else "",
            "side_effects_declared": payload_snapshot["side_effects_declared"],
            "budget_hint": payload_snapshot["budget_hint"],
            "degraded_dispatch": spec["degraded_dispatch"],
            "seed_kit_tokens": spec["seed_kit_tokens"],
        }
        return {"dyn_id": dyn_id, "summary": summary}

    def _cleanup_dynamic_action_artifact_dir(self, dyn_id: str) -> None:
        """Remove the partial ``agents/orchestration/dynamic_actions/<dyn_id>/``
        on a dispatch-time failure so the round cap stays clean and the
        next attempt starts from a fresh filesystem state."""
        if self.session_dir is None:
            return
        from ..session_paths import dynamic_action_artifact_dir
        target = dynamic_action_artifact_dir(self.session_dir, dyn_id)
        try:
            if target.exists():
                import shutil
                shutil.rmtree(target, ignore_errors=True)
        except OSError:
            log.exception(
                "dynamic_action: artefact dir cleanup failed for dyn_id=%s",
                dyn_id,
            )

    def _finalize_dynamic_action_dispatch(
        self, *, task_id: str, meta: dict[str, Any],
    ) -> None:
        """Atomically register the dispatch on SharedState (round cap
        +1) + log the task_id into the summary."""
        summary = dict(meta.get("summary") or {})
        summary["task_id"] = task_id
        dyn_id = str(meta.get("dyn_id") or "")
        self.shared_state.record_dynamic_action_dispatch(dyn_id, summary)

    def _dynamic_action_round_id(self) -> int:
        """Stable round id used in the ``dyn-<round>-<seq>`` template.

        Mirrors the round derivation used by the IR-7 assess_remaining_gaps
        wrapper: ``explore_search.cursor`` is the canonical EXPLORE-round
        cursor, falling back to 0 when the ledger is empty.
        """
        cursor = (self.shared_state.explore_search or {}).get("cursor")
        try:
            return int(cursor or 0)
        except (TypeError, ValueError):
            return 0

    # ------------------------------------------------------------------
    # Resume-time abandoned sweep wrapper
    # ------------------------------------------------------------------
    def _resume_abandon_dynamic_actions(self) -> None:
        """Consolidate every non-terminal dyn_id into ABANDONED at
        resume time. Side-effects (worktree cleanup, dispatch_history
        append, summary writes) live in
        :mod:`dynamic_action_resume`; this wrapper only threads the
        coordinator-side context (session_dir, shared_state,
        framework_source_roots) and forwards a single boot-log line."""
        from .dynamic_action_resume import resume_abandon_dynamic_actions
        from .framework_paths import resolve_source_file_allowlist

        if self.session_dir is None:
            return
        try:
            result = resume_abandon_dynamic_actions(
                session_dir=self.session_dir,
                shared_state=self.shared_state,
                coordinator_session_id=str(
                    getattr(self.shared_state, "session_id", "") or "",
                ),
                framework_source_roots=tuple(
                    resolve_source_file_allowlist(),
                ),
            )
        except Exception:  # noqa: BLE001 — defensive: never block boot
            log.exception(
                "dynamic_action resume sweep failed; resume continues",
            )
            return
        if (
            result.abandoned or result.artifact_missing
            or result.summary_missing
        ):
            log.info(result.to_log_line())
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action resume sweep: save after sweep failed",
                )

    # ------------------------------------------------------------------
    # dynamic_action runner / critic / integrate lifecycle hooks
    # ------------------------------------------------------------------
    def _ensure_dynamic_action_dispatched_row(
        self,
        *,
        dyn_id: str,
        params: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Ensure ``dynamic_actions[dyn_id]`` carries a ``DISPATCHED``
        row before downstream lifecycle hooks run.

        The dispatch hook normally creates this row; this helper is a
        defensive fallback for paths where it is missing (runner
        crash before dispatch finalise, resume with no summary, test
        setups). When ``params`` / ``extra`` carry them, the seeded
        row is also enriched with the prompt-projection fields
        (motivation_gap_short / scope_domains / artifact_path).
        """
        from .dynamic_action_proposal import (
            DynamicActionStatus,
            MOTIVATION_GAP_SHORT_MAX_CHARS,
        )
        if dyn_id in (self.shared_state.dynamic_actions or {}):
            return
        params = params or {}
        extra = extra or {}
        scope_from_spec: list[str] = []
        motivation = ""
        artifact_path = str(params.get("artifact_path") or "")
        spec_path = params.get("spec_path")
        if spec_path:
            try:
                spec_dict = json.loads(
                    Path(str(spec_path)).read_text(encoding="utf-8"),
                )
                payload_dict = spec_dict.get("payload") or {}
                scope_from_spec = list(payload_dict.get("scope_domains") or ())
                motivation = str(payload_dict.get("motivation_gap_text") or "")
            except (OSError, json.JSONDecodeError):
                pass
        if len(motivation) > MOTIVATION_GAP_SHORT_MAX_CHARS:
            motivation = (
                motivation[: MOTIVATION_GAP_SHORT_MAX_CHARS - 3].rstrip()
                + "..."
            )
        seed_extra: dict[str, Any] = {
            "dyn_id": dyn_id,
            "scope_domains": scope_from_spec,
            "motivation_gap_short": motivation,
            "artifact_path": artifact_path,
            "verdict": None,
            "cumulative_gain": None,
            "synthesised_row": True,
        }
        seed_extra.update(extra)
        self.shared_state.record_dynamic_action_outcome(
            dyn_id,
            status=DynamicActionStatus.DISPATCHED.value,
            extra=seed_extra,
        )

    def _walk_dynamic_action_to_state(
        self,
        dyn_id: str,
        *,
        target_status: "DynamicActionStatus",
    ) -> None:
        """Step the dyn_id summary along the canonical happy path
        until ``status == target_status``.

        Each step uses the strict writer so illegal transitions still
        log warnings; the helper is a *recovery* primitive that
        catches hook skips without bypassing the state machine.
        Terminal source states stop the walk (a terminal can't be
        advanced).
        """
        from .dynamic_action_proposal import (
            DynamicActionStatus,
            TERMINAL_LIFECYCLE_STATUSES,
        )

        canonical_path = (
            DynamicActionStatus.DISPATCHED,
            DynamicActionStatus.SUB_AGENT_RUNNING,
            DynamicActionStatus.SUB_AGENT_DONE,
            DynamicActionStatus.AWAITING_CRITIC,
            DynamicActionStatus.INTEGRATING,
        )
        if target_status not in canonical_path:
            return
        self._ensure_dynamic_action_dispatched_row(dyn_id=dyn_id)
        target_index = canonical_path.index(target_status)
        for _ in range(len(canonical_path)):
            current_row = (
                self.shared_state.dynamic_actions.get(dyn_id) or {}
            )
            current_raw = str(current_row.get("status") or "")
            try:
                current = DynamicActionStatus(current_raw)
            except ValueError:
                return
            if current == target_status:
                return
            if current in TERMINAL_LIFECYCLE_STATUSES:
                return
            try:
                current_index = canonical_path.index(current)
            except ValueError:
                return
            if current_index >= target_index:
                return
            next_status = canonical_path[current_index + 1]
            self.shared_state.record_dynamic_action_outcome(
                dyn_id, status=next_status.value,
            )

    async def _handle_dynamic_action_runner_result(
        self,
        *,
        task: Task,
        result: SubAgentResult,
    ) -> None:
        """Advance the dyn_id lifecycle after the runner returns.

        Three branches:

        * Non-COMPLETED terminal state → write the status (TIMED_OUT
          / FAILED / COMPLETED_EMPTY / ABANDONED) onto the summary,
          done. No critic, no integrate, no proposal_set materialisation.
        * COMPLETED but mechanical floor (P4) says reject/revise →
          write the critic verdict envelope to
          ``critic_verdict.json``, mark status CRITIC_REJECTED, done.
        * COMPLETED + mechanical floor passes → materialise the
          patch into a specialist-shaped workspace, synthesise an
          integrate_patch proposal, push it through the standard
          PendingProposal pipeline so the Critic + integrate_patch
          executor handle it without any main-chain change.
        """
        from .dynamic_action_pipeline import (
            DYNAMIC_SPECIALIST_TASK_ID_PREFIX,
            build_integrate_patch_proposal_payload,
            compose_critic_verdict_envelope,
            materialize_dynamic_patch_workspace,
            read_runner_proposal_set,
            runner_status_to_lifecycle,
        )
        from .dynamic_action_critic import write_critic_verdict
        from .dynamic_action_history import (
            DispatchHistoryEvent,
            append_dispatch_history_row,
            write_dynamic_action_telemetry,
        )
        from .dynamic_action_proposal import (
            DynamicActionStatus,
            DynamicRunnerTerminalState,
        )
        from ..session_paths import dynamic_action_spec_path

        params = task.params or {}
        dyn_id = str(params.get("dyn_id") or "").strip()
        if not dyn_id:
            log.warning(
                "dynamic_action runner result without dyn_id; task=%s",
                task.task_id,
            )
            return
        result_dict = result.result if isinstance(result.result, dict) else {}
        terminal_state_raw = str(
            result_dict.get("terminal_state") or "",
        ).strip()
        try:
            terminal_state = DynamicRunnerTerminalState(terminal_state_raw)
        except ValueError:
            terminal_state = DynamicRunnerTerminalState.FAILED
        reason = str(result_dict.get("reason") or "")
        turns_used = int(result_dict.get("turns_used") or 0)
        lifecycle = runner_status_to_lifecycle(terminal_state)

        extra = {
            "runner_terminal_state": terminal_state.value,
            "runner_reason": reason,
            "turns_used": turns_used,
            "journal_path": str(result_dict.get("journal_path") or ""),
        }
        # Walk the state machine deliberately so the audit trail
        # captures DISPATCHED → SUB_AGENT_RUNNING transitions even when
        # the runner finished in one tick. Seed the DISPATCHED row
        # first if missing so the writer's transition validator passes.
        self._ensure_dynamic_action_dispatched_row(
            dyn_id=dyn_id,
            params=params,
            extra=extra,
        )
        self.shared_state.record_dynamic_action_outcome(
            dyn_id, status=DynamicActionStatus.SUB_AGENT_RUNNING.value,
        )
        # Non-COMPLETED terminal states emit ``SUB_AGENT_TERMINATED``;
        # ``SUB_AGENT_DONE`` lands later after the proposal_set count
        # is known.
        if terminal_state != DynamicRunnerTerminalState.COMPLETED:
            try:
                append_dispatch_history_row(
                    session_dir=self.session_dir,
                    dyn_id=dyn_id,
                    event=DispatchHistoryEvent.SUB_AGENT_TERMINATED,
                    payload={
                        "terminal_state": terminal_state.value,
                        "reason": reason,
                        "turns_used": turns_used,
                        "journal_path": str(
                            result_dict.get("journal_path") or "",
                        ),
                        "proposal_count": 0,
                    },
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: history SUB_AGENT_TERMINATED write "
                    "failed for dyn_id=%s", dyn_id,
                )
            self.shared_state.record_dynamic_action_outcome(
                dyn_id,
                status=lifecycle.value,
                last_outcome=reason or None,
                extra=extra,
            )
            try:
                write_dynamic_action_telemetry(
                    session_dir=self.session_dir,
                    dyn_id=dyn_id,
                    lifecycle=lifecycle,
                    round_index=self._dynamic_action_round_id(),
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: telemetry write failed for dyn_id=%s "
                    "(non-completed terminal)", dyn_id,
                )
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: save after non-completed runner "
                    "result failed for dyn_id=%s", dyn_id,
                )
            return

        # COMPLETED — load the proposal_set the runner wrote.
        runner_payload = read_runner_proposal_set(self.session_dir, dyn_id)
        proposal_set: list[dict[str, Any]] = []
        if isinstance(runner_payload, dict):
            raw = runner_payload.get("proposal_set") or []
            if isinstance(raw, list):
                proposal_set = [p for p in raw if isinstance(p, dict)]
        # The runner already breaks on the first emit_proposal; this
        # is a defensive truncate in case a schema drift produces
        # extras (downstream materialise only reads [0]).
        from .dynamic_action_proposal import MAX_PROPOSAL_SET_LEN
        if len(proposal_set) > MAX_PROPOSAL_SET_LEN:
            log.warning(
                "dynamic_action: proposal_set len=%d > cap=%d for "
                "dyn_id=%s; truncating to first %d entries",
                len(proposal_set), MAX_PROPOSAL_SET_LEN, dyn_id,
                MAX_PROPOSAL_SET_LEN,
            )
            proposal_set = proposal_set[:MAX_PROPOSAL_SET_LEN]
        # ``SUB_AGENT_DONE`` row carries the post-load proposal_count
        # (0 collapses to COMPLETED_EMPTY downstream; 1 proceeds to
        # the critic).
        try:
            append_dispatch_history_row(
                session_dir=self.session_dir,
                dyn_id=dyn_id,
                event=DispatchHistoryEvent.SUB_AGENT_DONE,
                payload={
                    "terminal_state": terminal_state.value,
                    "reason": reason,
                    "turns_used": turns_used,
                    "journal_path": str(
                        result_dict.get("journal_path") or "",
                    ),
                    "proposal_count": len(proposal_set),
                },
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action: history SUB_AGENT_DONE write failed "
                "for dyn_id=%s", dyn_id,
            )
        if not proposal_set:
            # COMPLETED but empty proposal_set (runner contract
            # mismatch); collapse to COMPLETED_EMPTY for the
            # lifecycle so downstream consumers see the canonical
            # empty signal. Walk through SUB_AGENT_RUNNING already
            # done above; the terminal write is legal.
            self.shared_state.record_dynamic_action_outcome(
                dyn_id,
                status=DynamicActionStatus.COMPLETED_EMPTY.value,
                last_outcome="emit_empty",
                extra=extra,
            )
            try:
                write_dynamic_action_telemetry(
                    session_dir=self.session_dir,
                    dyn_id=dyn_id,
                    lifecycle=DynamicActionStatus.COMPLETED_EMPTY,
                    round_index=self._dynamic_action_round_id(),
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: telemetry write failed for dyn_id=%s "
                    "(COMPLETED_EMPTY)", dyn_id,
                )
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: save after empty proposal_set "
                    "failed for dyn_id=%s", dyn_id,
                )
            return
        proposal = proposal_set[0]
        # Step through SUB_AGENT_RUNNING → SUB_AGENT_DONE before
        # AWAITING_CRITIC so the state-machine audit trail captures
        # the runner-done event distinctly from the critic dispatch.
        self.shared_state.record_dynamic_action_outcome(
            dyn_id, status=DynamicActionStatus.SUB_AGENT_DONE.value,
        )

        # Load spec.json for scope_domains (mechanical-check truth set).
        spec_path = dynamic_action_spec_path(self.session_dir, dyn_id)
        spec_payload: dict[str, Any] = {}
        try:
            spec_dict = json.loads(spec_path.read_text(encoding="utf-8"))
            spec_payload = spec_dict.get("payload") or {}
        except (OSError, json.JSONDecodeError):
            log.warning(
                "dynamic_action: could not read spec.json for "
                "dyn_id=%s; mechanical scope check falls back to "
                "the proposal's own scope_domains",
                dyn_id,
            )
            spec_payload = {
                "scope_domains": list(proposal.get("scope_domains") or ()),
            }

        # Mechanical-floor verdict: if it blocks, skip the LLM critic
        # — write the envelope and flip the status here.
        envelope, lifecycle_after_mech = compose_critic_verdict_envelope(
            dyn_id=dyn_id,
            proposal=proposal,
            spec_scope_domains=list(spec_payload.get("scope_domains") or ()),
            llm_verdict=None,
        )
        if envelope["verdict"] != "approve":
            write_critic_verdict(self.session_dir, dyn_id, envelope)
            try:
                append_dispatch_history_row(
                    session_dir=self.session_dir,
                    dyn_id=dyn_id,
                    event=DispatchHistoryEvent.CRITIC_VERDICT,
                    payload={
                        "verdict": envelope["verdict"],
                        "reason_codes": list(envelope["reason_codes"]),
                        "applied_rules": list(envelope["applied_rules"]),
                        "cross_domain_flag": bool(
                            envelope["cross_domain_flag"],
                        ),
                        "mechanical_floor_blocked": True,
                    },
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: history CRITIC_VERDICT (mechanical) "
                    "write failed for dyn_id=%s", dyn_id,
                )
            self.shared_state.record_dynamic_action_outcome(
                dyn_id,
                status=DynamicActionStatus.CRITIC_REJECTED.value,
                last_outcome=envelope["reason_codes"][0]
                    if envelope["reason_codes"] else "critic_rejected",
                extra={
                    **extra,
                    "critic_verdict": envelope["verdict"],
                    "critic_reason_codes": list(envelope["reason_codes"]),
                    "critic_path": str(
                        self.session_dir
                        / "agents/orchestration/dynamic_actions"
                        / dyn_id / "critic_verdict.json",
                    ),
                },
            )
            try:
                write_dynamic_action_telemetry(
                    session_dir=self.session_dir,
                    dyn_id=dyn_id,
                    lifecycle=DynamicActionStatus.CRITIC_REJECTED,
                    round_index=self._dynamic_action_round_id(),
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: telemetry write failed for dyn_id=%s "
                    "(mechanical CRITIC_REJECTED)", dyn_id,
                )
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: save after mechanical critic "
                    "block failed for dyn_id=%s", dyn_id,
                )
            return

        # Mechanical floor passed — materialise the specialist-shaped
        # workspace + push the proposal onto the bus for the Critic
        # to score. The mirror on _handle_single_verdict will
        # write the verdict into specialist_patch_verdicts so the
        # PolicyGate gate on the eventual integrate_patch delegate
        # passes.
        specialist_task_id, _patches = materialize_dynamic_patch_workspace(
            session_dir=self.session_dir,
            dyn_id=dyn_id,
            proposal=proposal,
        )
        propose_payload = build_integrate_patch_proposal_payload(
            dyn_id=dyn_id,
            specialist_task_id=specialist_task_id,
            proposal=proposal,
            spec_payload=spec_payload,
        )
        # Push a synthetic ``proposal`` event so the Critic agent's
        # reactor pulls it and emits a review_verdict on the next
        # tick. Bypassing PolicyGate is safe here — the message is
        # coordinator-internal and never re-enters the validate_intent
        # path.
        msg = Message.new(
            "coordinator", "*", "proposal",
            {**propose_payload, "needs_review": True},
            priority=1,
        )
        await self.bus.append_and_seq(msg)
        pending = PendingProposal(
            proposal_msg_id=msg.msg_id,
            from_agent="coordinator",
            action_name="integrate_patch",
            predicted_gain_pct=0.0,
            payload=dict(propose_payload),
        )
        self.state.pending_proposals[msg.msg_id] = pending
        self.shared_state.record_dynamic_action_outcome(
            dyn_id,
            status=DynamicActionStatus.AWAITING_CRITIC.value,
            extra={
                **extra,
                "specialist_task_id": specialist_task_id,
                "proposal_msg_id": msg.msg_id,
                "specialist_task_id_prefix": (
                    DYNAMIC_SPECIALIST_TASK_ID_PREFIX
                ),
            },
        )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action: save after critic dispatch failed "
                "for dyn_id=%s", dyn_id,
            )

    def _mirror_critic_verdict_to_dynamic_action(
        self,
        *,
        pending: "PendingProposal",
        verdict: str,
        reasoning: str,
    ) -> None:
        """Compose + persist the critic_verdict.json envelope when a
        single-verdict review targets a dyn_id-shaped proposal.

        Detection key: ``payload.params.specialist_task_id`` starts
        with the synthesised ``dyn-`` prefix. When that's not the
        case (i.e. legacy specialist patches), the helper is a no-op
        and the existing PR-A7 mirror path runs unchanged.

        For approve verdicts the dyn_id status stays DISPATCHED — the
        integrate_patch completion hook flips it to KEPT / REVERTED
        / INTEGRATE_FAILED. For reject / revise the helper flips the
        status to CRITIC_REJECTED so the dispatch never reaches the
        integrate executor.
        """
        from .dynamic_action_pipeline import (
            compose_critic_verdict_envelope,
            is_dynamic_specialist_task_id,
        )
        from .dynamic_action_critic import write_critic_verdict
        from .dynamic_action_history import (
            DispatchHistoryEvent,
            append_dispatch_history_row,
            write_dynamic_action_telemetry,
        )
        from .dynamic_action_proposal import (
            DynamicActionStatus,
            TERMINAL_LIFECYCLE_STATUSES,
        )
        from ..session_paths import dynamic_action_spec_path

        params = pending.payload.get("params") or {}
        sid = str(params.get("specialist_task_id") or "").strip()
        if not is_dynamic_specialist_task_id(sid):
            return
        dyn_id = str(params.get("dyn_id") or sid).strip()
        if not dyn_id:
            return
        # Re-load the on-disk proposal_set + spec so the envelope is
        # composed against the same data the dispatcher hook used.
        from .dynamic_action_pipeline import read_runner_proposal_set
        runner_payload = read_runner_proposal_set(self.session_dir, dyn_id)
        proposal: dict[str, Any] = {}
        if isinstance(runner_payload, dict):
            raw = runner_payload.get("proposal_set") or []
            if isinstance(raw, list) and raw and isinstance(raw[0], dict):
                proposal = raw[0]
        spec_payload: dict[str, Any] = {}
        try:
            spec_dict = json.loads(
                dynamic_action_spec_path(self.session_dir, dyn_id).read_text(
                    encoding="utf-8",
                ),
            )
            spec_payload = spec_dict.get("payload") or {}
        except (OSError, json.JSONDecodeError):
            spec_payload = {
                "scope_domains": list(proposal.get("scope_domains") or ()),
            }
        envelope, lifecycle = compose_critic_verdict_envelope(
            dyn_id=dyn_id,
            proposal=proposal,
            spec_scope_domains=list(spec_payload.get("scope_domains") or ()),
            llm_verdict=verdict,
            llm_reason=reasoning,
        )
        write_critic_verdict(self.session_dir, dyn_id, envelope)
        try:
            append_dispatch_history_row(
                session_dir=self.session_dir,
                dyn_id=dyn_id,
                event=DispatchHistoryEvent.CRITIC_VERDICT,
                payload={
                    "verdict": envelope["verdict"],
                    "reason_codes": list(envelope["reason_codes"]),
                    "applied_rules": list(envelope["applied_rules"]),
                    "cross_domain_flag": bool(envelope["cross_domain_flag"]),
                    "mechanical_floor_blocked": False,
                },
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action: history CRITIC_VERDICT (llm) write "
                "failed for dyn_id=%s", dyn_id,
            )
        new_status = lifecycle.value
        last_outcome = (
            envelope["reason_codes"][0]
            if envelope["reason_codes"]
            else (verdict or "").lower()
        )
        extra = {
            "critic_verdict": envelope["verdict"],
            "critic_reason_codes": list(envelope["reason_codes"]),
            "critic_proposal_msg_id": pending.proposal_msg_id,
            "verdict": envelope["verdict"],
        }
        # Ensure the state machine reaches AWAITING_CRITIC before the
        # verdict transition (covers replays where the runner-done
        # hook never advanced the state).
        self._walk_dynamic_action_to_state(
            dyn_id, target_status=DynamicActionStatus.AWAITING_CRITIC,
        )
        self.shared_state.record_dynamic_action_outcome(
            dyn_id, status=new_status,
            last_outcome=last_outcome, extra=extra,
        )
        try:
            new_status_enum = DynamicActionStatus(new_status)
            if new_status_enum in TERMINAL_LIFECYCLE_STATUSES:
                write_dynamic_action_telemetry(
                    session_dir=self.session_dir,
                    dyn_id=dyn_id,
                    lifecycle=new_status_enum,
                    round_index=self._dynamic_action_round_id(),
                )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action: telemetry write failed for dyn_id=%s "
                "(critic verdict mirror)", dyn_id,
            )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action: save after critic verdict mirror "
                "failed for dyn_id=%s", dyn_id,
            )

    def _maybe_update_dynamic_action_after_integrate(
        self,
        *,
        task: Task,
        result: SubAgentResult,
    ) -> None:
        """Translate an ``integrate_patch`` result to the dyn_id
        summary when the task targets a synthesised dynamic workspace.

        Routing key is ``params.specialist_task_id`` — when it starts
        with ``dyn-`` (set by
        :func:`materialize_dynamic_patch_workspace`) we own the
        translation; otherwise the legacy PR-A7 path keeps running.
        """
        from .dynamic_action_pipeline import (
            integrate_status_to_lifecycle,
            is_dynamic_specialist_task_id,
        )
        from .dynamic_action_history import (
            DispatchHistoryEvent,
            append_dispatch_history_row,
            write_dynamic_action_telemetry,
        )
        from .dynamic_action_proposal import DynamicActionStatus

        params = task.params or {}
        sid = str(params.get("specialist_task_id") or "").strip()
        if not is_dynamic_specialist_task_id(sid):
            return
        dyn_id = str(params.get("dyn_id") or sid).strip()
        result_dict = result.result if isinstance(result.result, dict) else {}
        integrate_status = str(result_dict.get("status") or "").strip()
        lifecycle = integrate_status_to_lifecycle(integrate_status)
        delta_pct = result_dict.get("delta_pct")
        extra = {
            "integrate_task_id": task.task_id,
            "integrate_status": integrate_status,
            "patches_applied": list(result_dict.get("patches_applied") or ()),
            "patches_reverted": list(result_dict.get("patches_reverted") or ()),
            "output_throughput": result_dict.get("output_throughput"),
            "accuracy_pass": result_dict.get("accuracy_pass"),
        }
        gain_value: float | None = None
        if isinstance(delta_pct, (int, float)):
            gain_value = float(delta_pct)
        # The integrate completion is the INTEGRATING →
        # {KEPT, REVERTED, INTEGRATE_FAILED} step. Walk the canonical
        # path so the audit trail captures every intermediate
        # transition even when upstream hooks skipped a state.
        self._walk_dynamic_action_to_state(
            dyn_id,
            target_status=DynamicActionStatus.INTEGRATING,
        )
        self.shared_state.record_dynamic_action_outcome(
            dyn_id,
            status=lifecycle.value,
            last_outcome=integrate_status or None,
            cumulative_gain=gain_value,
            extra=extra,
        )
        try:
            append_dispatch_history_row(
                session_dir=self.session_dir,
                dyn_id=dyn_id,
                event=DispatchHistoryEvent.INTEGRATE_RESULT,
                payload={
                    "integrate_status": integrate_status,
                    "lifecycle": lifecycle.value,
                    "delta_pct": gain_value,
                    "task_id": str(task.task_id or ""),
                    "patches_applied": list(
                        result_dict.get("patches_applied") or (),
                    ),
                    "patches_reverted": list(
                        result_dict.get("patches_reverted") or (),
                    ),
                },
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action: history INTEGRATE_RESULT write failed "
                "for dyn_id=%s", dyn_id,
            )
        try:
            write_dynamic_action_telemetry(
                session_dir=self.session_dir,
                dyn_id=dyn_id,
                lifecycle=lifecycle,
                gain_pct=gain_value,
                round_index=self._dynamic_action_round_id(),
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action: telemetry write failed for dyn_id=%s "
                "(integrate completion)", dyn_id,
            )
        # Tag the intervention ledger with a dynamic-specific action
        # name so analytics can split KEEP rate by source without
        # scraping logs.
        if lifecycle == DynamicActionStatus.KEPT:
            try:
                self.shared_state.record_intervention(
                    change_type="code_patch",
                    action="dynamic_action_integrate",
                    task_id=task.task_id,
                    delta_pct=gain_value,
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action: record_intervention failed for "
                    "dyn_id=%s", dyn_id,
                )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "dynamic_action: save after integrate completion "
                "failed for dyn_id=%s", dyn_id,
            )

    # ------------------------------------------------------------------
    # specialist pre-dispatch warmup
    # ------------------------------------------------------------------
    async def _warm_specialist_params(self, params: dict[str, Any]) -> None:
        """Fill the specialist task params with KnowledgePlane data
        before the task is enqueued.

        Mutates ``params`` in place. The caller (``_handle_delegate``)
        always passes a fresh dict copy, so we don't risk mutating the
        original intent.payload.

        Sources, all best-effort (specialist still runs even if every
        warmup call fails — SpecialistRunner sees empty fields and
        SpecialistPromptBuilder renders ``(none)``):

        * ``pr_feed`` — :meth:`KnowledgePlane.pr_feed_warm` for the
          delegate's ``params.domain``. PRSummary objects are flattened
          to dicts so the prompt builder can render them uniformly.
        * ``pr_monitor_available`` — boolean mirror of
          ``plane.pr_monitor_enabled``.
        * ``warm_start_recipe`` / ``warm_start_pitfalls`` /
          ``warm_start_lessons`` — mirror of ``SharedState.warm_start_*``
          (already T0'd by cli).
        * ``framework_source_roots`` — picked up from
          :func:`resolve_source_file_allowlist` so the LLM has a stable
          local-source navigation hint without needing
          ``$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`` propagated by
          hand.
        * GPU hardware hints (``gpu_type`` / ``tp``) from SharedState.

        Missing fields stay empty; SpecialistPromptBuilder degrades to
        domain defaults, warm-start facts, PR feed, and research hints.
        """
        state = self.shared_state
        plane = self.knowledge_plane

        from .specialist_domains import normalize_dispatch_tags

        domain = str(params.get("domain") or "").strip()
        # Knowledge-domain tags drive multi-anchor prompt assembly. A
        # single ``domain`` is honoured as the legacy single-tag alias.
        tags = normalize_dispatch_tags(params)

        # PR feed (Gap-02 ↔ Gap-01 contract): if the plane is wired and
        # PR Monitor is enabled, fetch the warm cache for each dispatch
        # domain and merge. Any exception falls back to an empty list +
        # non-fatal warning.
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

        # Old Cortex v1 graph subgraphs were removed from KnowledgePlane.
        # Keep the field defaulted so SpecialistPromptInputs remains stable;
        # RecipeKB priors arrive through warm_start_* and PR feed instead.
        params.setdefault("kb_subgraph", {})

        # Warm-start recipe + pitfalls + lessons from T0 anchor.
        if state.warm_start_recipe and "warm_start_recipe" not in params:
            params["warm_start_recipe"] = dict(state.warm_start_recipe)
        if state.warm_start_pitfalls and "warm_start_pitfalls" not in params:
            params["warm_start_pitfalls"] = list(state.warm_start_pitfalls)
        if state.warm_start_lessons and "warm_start_lessons" not in params:
            params["warm_start_lessons"] = list(state.warm_start_lessons)
        # runtime framework / version so the prompt's
        # ``_format_version_note`` can annotate version-mismatched
        # lessons / pitfalls (e.g. "from sglang@0.4, you're on 0.5").
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

        # session_steward gets a panoramic state digest inlined into
        # its prompt (the steward needs stack depth, gain trajectory,
        # plateau signals, gaps, denial history to decide
        # continue_explore / advance_to_kernel / stop_session).
        # Other specialists don't see this — keeps their prompt
        # focused on the task at hand.
        if (
            str(params.get("domain") or "") == "session_steward_specialist"
            and "session_snapshot" not in params
        ):
            params["session_snapshot"] = self._build_session_snapshot()

        # Local-source navigation hint — same source the kernel agent
        # uses for ``source_file`` containment.
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

        # Hardware + workload hints (cheap; pulled from SharedState which
        # ``cli._init_fresh_session`` populates from CLI flags / env at
        # session start, and which ``cli`` re-exports back into env on
        # ``--resume``). Without these the SpecialistPromptInputs
        # dataclass defaults silently win — most notably ``tp=0`` /
        # ``tp=1``, which makes comm_specialist self-veto on TP>1
        # sessions with "cross-GPU collectives non-actionable". The same
        # workload hints surface to the specialist's prompt section 2,
        # so serving / system / kernel-switch specialists reason against the
        # real benchmark workload rather than the dataclass defaults.
        params.setdefault("gpu_type", state.gpu_type or "")
        # Active server framework name — used to switch the per-domain
        # "what to read first" hint blocks to atom paths when
        # SharedState.framework == "atom". sglang / vllm sessions fall
        # through to the canonical hint blocks.
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

        # Advisory ``model_arch`` profile -> specialist ``## 2. HARDWARE
        # CONTEXT`` via the existing ``arch_notes`` carrier. Reuses the
        # single-source renderer from shared_state so the orchestration
        # summary and specialist prompts stay in lockstep. Skipped entirely
        # when no profile was loaded, so non-arch sessions render exactly as
        # before. Prompt-context only; no deterministic gating reads it.
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

        # fill gap-specific anchors from the
        # gaps[] ledger. Orchestration carries a ``gap_canonical_id``
        # via ``delegate.params`` (and also as the ``gap`` field);
        # we look up the matching gap row and stamp its symptom /
        # layer / domain_hint / recent attempts onto the task so the
        # SpecialistPromptBuilder section that renders ``gap_symptom``
        # / ``gap_layer`` / ``gap_evidence`` has structured context
        # instead of falling back to ``(none)``.
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
                    # When the LLM omitted ``domain`` we let the gap's
                    # ``domain_hint`` win — PolicyGate R2 still validates
                    # the routing against SPECIALIST_DOMAIN_KEYS.
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

        # ROOFLINE EVIDENCE — specialists need ground-truth bottleneck
        # signals (Executive Summary + Top Operations) to ground their
        # proposal_set instead of guessing from the gap_evidence prose
        # alone. We pack the canonical fields into ``roofline_evidence``
        # plus ``analysis_md_path`` so the specialist can call
        # ``Read(analysis_md_path)`` for the full TraceLens report on
        # demand. No injection when the cache is empty (specialist
        # prompt builder degrades gracefully).
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

        # (Legacy ``framework_pr_scout`` pre-fetch removed — PR discovery
        # now lives in the standalone FRAMEWORK_PR phase pump rather than
        # piggy-backing on serving_specialist dispatches.)

        # proposal_set cap — flow the single-source-of-truth value into
        # ``params`` so SpecialistRunner can read it without re-importing
        # the constant. ``setdefault`` lets a delegate intent shrink the
        # cap explicitly; values larger than the constant get re-clamped
        # by the runner.
        from inference_optimizer.orchestrator.policy import (
            DEFAULT_SPECIALIST_MAX_PROPOSALS,
        )
        params.setdefault("max_proposals", DEFAULT_SPECIALIST_MAX_PROPOSALS)

    @staticmethod
    def _pr_summary_to_dict(pr: Any) -> dict[str, Any]:
        """Flatten a :class:`PRSummary` (or any object with the same
        attribute names) into the dict shape SpecialistPromptBuilder
        expects (``{title, url, labels, repo, number, state, author}``).
        Defensive against any future field additions in
        :mod:`pr_monitor`.
        """
        return {
            "repo":   str(getattr(pr, "repo", "")),
            "number": int(getattr(pr, "number", 0) or 0),
            "title":  str(getattr(pr, "title", "")),
            "url":    str(getattr(pr, "url", "")),
            "state":  str(getattr(pr, "state", "")),
            "labels": list(getattr(pr, "labels", ()) or ()),
            "author": str(getattr(pr, "author", "")),
        }

    # ------------------------------------------------------------------
    # gaps[] ledger refresh
    # ------------------------------------------------------------------
    async def _refresh_gaps(self, *, reason: str) -> None:
        """Refresh :attr:`SharedState.gaps` from observable signals.

        Coordinator is the sole writer (Inv-1 single-writer +
        PolicyGate ``CORE_STATE_FIELDS`` lock). Called at:

        1. baseline completion          (``reason='baseline_done'``)
        2. EXPLORE round KEEP / REVERT  (``reason='explore_round'``)
        3. specialist_done bookkeeping  (``reason='specialist_done'``)
        4. periodic Cortex refresh      (``reason='cortex_refresh'``)

        The refresh is *additive*: each call upserts entries derived
        from current observable signals (baseline, attempts log,
        winners history, Cortex traverse) without nuking the existing
        list. Dedup is keyed by ``canonical_id``; per-gap
        ``attempts`` is capped at the most recent
        :data:`_GAPS_ATTEMPTS_HISTORY` rows.

        Best-effort: any exception is logged and absorbed so the
        refresh never blocks the calling path (a stale gap list is
        always preferable to a dead reactor).
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
            reason, len(state.gaps),
        )

    def _extract_gaps_from_baseline(self) -> list[dict[str, Any]]:
        """Derive initial gap rows from the baseline snapshot.

        Two synthesised gaps when applicable:

        * ``throughput_below_target`` — fires when ``target_gap_pct``
          is positive (operator asked for a specific gain%, and we
          haven't hit it yet). Severity scales with the gap.
        * ``baseline_unstable`` — fires when ``baseline_failure_streak
          > 0``. Tied to the kernel/system layer because that's where
          most baseline crashes originate.

        These are *anchor* gaps (the M1 fallback in
        :meth:`_resolve_issue_canonical` already minted a workload
        canonical_id; we reuse the same id so traverse rows align).
        """
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
        """Derive gaps from the rolling failures + winners history.

        Walks ``last_action_failures`` to surface recurring (action,
        error_class) failure patterns and ``explore_search.winners_history``
        to surface a "stalled explore plateau" gap when several recent
        rounds failed to produce a new ``current_best``.

        Layer assignment is best-effort and uses the action family as
        a proxy (kernel_opt → kernel; backends/params/explore →
        framework; sweep → framework). ``domain_hint`` follows the
        same mapping.
        """
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
        """Map an action name → (layer, domain_hint) for gap rows.

        Centralised so the four call sites (baseline / attempts /
        explore / specialist_done) agree on the routing. Falls back
        to ``("framework", "serving_specialist")`` for unknown
        action names — that's the broadest catch-all in the M5
        specialist catalogue.
        """
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
        """Append per-variant KEEP/REVERT outcomes to the matching gap.

        Called from :meth:`_promote_to_shared_state` after an explore
        task settles. When the explore proposal carried a
        ``gap_canonical_id`` (Orchestration routed the proposal to a
        specific gap), we append every variant outcome as an attempt
        on that gap; otherwise the attempts are folded under the
        anchor gap so the velocity counter still moves.

        Best-effort: a missing gap row is upserted with a minimal
        symptom rather than dropped — the alternative would silently
        lose the audit trail on cold-start sessions.
        """
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

    # ------------------------------------------------------------------
    # specialist_done bookkeeping
    # ------------------------------------------------------------------
    async def _handle_specialist_done(
        self, source: str, intent: Intent,
    ) -> None:
        """Handle a ``specialist_done`` intent.

        Source format is ``specialist:<task_id>`` per Inv-5.3 / R3
        (PolicyGate already validated this when called via
        :meth:`_handle_intent`; defense in depth re-asserts the
        prefix here so direct callers — e.g. the dispatcher exit
        hook — get the same task lookup logic for free).

        The bookkeeping itself is in :meth:`_record_specialist_result`
        so the runner-internal path (Gap-01 adapter) and the
        intent-routing path (this method) converge on the same writer.

        Note: the current SpecialistRunner captures the done intent
        internally and surfaces it via :class:`SubAgentResult`. The
        dispatcher result hook is therefore the primary entry point
        for production. The intent-routing branch (above) handles
        the future case where an operator replay / out-of-band
        emitter routes a done through the bus.
        """
        payload = dict(intent.payload or {})
        task_id = self._task_id_from_specialist_source(source)
        task: Task | None = None
        if task_id:
            try:
                task = await self.tasks.get(task_id)
            except Exception:  # noqa: BLE001 — TaskNotFound and friends
                task = None
        if task is None:
            # PolicyGate R3 should have caught this. Log defensively
            # but don't crash — the audit trail (transcript on disk +
            # the runner-internal path's delegated_result) is still
            # intact.
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
        """Extract the task_id from a ``specialist:<task_id>`` source.

        Returns an empty string when the source doesn't carry the
        expected prefix (the caller treats that as an unknown-task
        miss and bails out).
        """
        if not source:
            return ""
        if source.startswith(SPECIALIST_FROM_AGENT_PREFIX):
            return source[len(SPECIALIST_FROM_AGENT_PREFIX):]
        return ""

    async def _record_specialist_result(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
    ) -> None:
        """Common bookkeeping for any specialist task termination.

        Called from two paths:

        * The dispatcher's result loop (production), once the
          SpecialistRunner returns a :class:`SubAgentResult` with
          ``result.specialist_done`` populated (Gap-01 adapter).
        * The intent routing table (defense in depth), when an
          out-of-band emitter sends a ``SPECIALIST_DONE`` intent
          through the bus.

        Both paths converge on the same SharedState writes so the
        ledger / streak / ``last_specialist`` mirror stay coherent
        regardless of where the trigger came from. Idempotent on
        ``round_id`` (delegated to :meth:`SharedState.record_specialist_round`).

        Failures inside the bookkeeping are logged but not re-raised:
        the runner has already produced an on-disk transcript +
        ``specialist_done.json`` artifact, so the audit trail
        survives even if SharedState persistence hiccups.

        The legacy T2 per-variant hypothesize used to fire here was
        removed when the hypothesize/verify protocol was retired;
        specialist proposals now write facts only after the executor
        produces a KEEP/REVERT outcome.
        """
        domain = str(done_payload.get("domain") or "").strip()
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        is_empty = bool(done_payload.get("empty")) or len(proposals) == 0

        round_entry = self._build_specialist_round_entry(
            task=task, done_payload=done_payload, source=source,
        )
        # Advisory multi-model scoring of the proposal_set. Purely
        # informational — the scores ride on the round entry and surface
        # to Orchestration as one reference among many; they gate
        # nothing. Defensive: any scorer failure leaves the entry
        # score-free and the run continues unchanged.
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

        # Persist + audit observation so a resume picks up the
        # bookkeeping without re-running the specialist.
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

        # Harvest research-scout output: persist hints + competitor
        # target, seed advisory gaps, and dedup PR ids against
        # FRAMEWORK_PR. Fail-soft; never blocks the round.
        if domain == "research_scout_specialist":
            try:
                self._harvest_research_scout(done_payload)
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "research-scout harvest failed for task=%s", task.task_id,
                )

        # refresh the gaps ledger after a
        # specialist round closes. The dedupe-by-canonical-id keeps the
        # list bounded; the per-gap attempt log captures the
        # specialist's verdict (empty vs non-empty proposal_set) as a
        # fresh attempt row on the gap that triggered the dispatch.
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
        # B3 (auto-apply unlock): if this specialist authored source
        # patches, push them to the Critic now so the integrate_patch
        # gate can pass and the patch actually reaches the serving GPU.
        try:
            await self._maybe_autosubmit_specialist_patches(
                task=task, done_payload=done_payload,
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "B3: specialist patch autosubmit failed for task=%s",
                task.task_id,
            )

    def _target_gap_advisory_block(self) -> str:
        """Build the advisory "External target gap" prompt block.

        Compares our current-best per-GPU throughput / TPOT against the
        LLM-authored competitor target and renders a direction hint when
        the TPOT ratio dominates. Returns an empty string when the
        feature is disabled, no sourced competitor target exists, or our
        side has no comparable numbers. Advisory only — never gates.
        """
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
        """Resolve the dominant external gap direction ('latency' /
        'throughput') from the LLM-authored competitor target, or
        ``None`` when advisory is off / no comparable target exists.

        Fail-soft: any read / parse problem yields ``None``.
        """
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
        """Collect proposal_set rows from the most recent specialist rounds.

        Mirrors the window :meth:`SharedState.to_proposal_scores_summary`
        renders so the priors-match annotation lines up with the scores
        the prompt already shows. Deduped by variant name; fail-soft.
        """
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
        """Flag recently proposed variants that align with proven priors
        or the dominant external gap so Orchestration tries them earlier.

        Advisory only — affects prompt ordering, never Objective /
        scoring / gates. Empty string when nothing matches. Fail-soft.
        """
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

    def _apply_depth_gate_to_verdict(
        self, *, raw_rec: str, next_gap: str, task: "Task",
    ) -> tuple[str, str]:
        """Rewrite a stop / advance verdict to ``continue_explore`` when
        exploration depth is insufficient.

        Returns the (possibly rewritten) ``(recommendation, next_gap)``.
        IR-6 budget exhaustion bypasses the gate entirely; a disabled
        gate or a satisfied depth check leaves the verdict untouched.
        """
        state = self.shared_state
        try:
            if not state.depth_gate_enabled():
                return raw_rec, next_gap
        except Exception:  # noqa: BLE001 — defensive
            return raw_rec, next_gap

        # The HARD force-exit is the only thing the depth gate must never override.
        try:
            force_exit, _ = _phase_state.should_force_exit_explore(
                state, budget_pct=self._phase_budget_pct,
            )
        except Exception:  # noqa: BLE001 — defensive
            force_exit = False
        if force_exit:
            log.info(
                "depth-gate: IR-6 budget force-exit active; honouring "
                "steward '%s' (task=%s)", raw_rec, task.task_id,
            )
            return raw_rec, next_gap

        try:
            satisfied, blockers, next_action = _phase_state.depth_gate(
                state, **self._depth_gate_thresholds(),
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception("depth-gate: evaluation failed; honouring steward")
            return raw_rec, next_gap
        if satisfied:
            return raw_rec, next_gap

        # Rewrite to continue and seed a concrete deepening gap.
        round_id = int((state.explore_search or {}).get("cursor") or 0)
        gap_cid = f"gap.depth.round{round_id}"
        symptom = next_action or (
            "exploration too shallow: " + ", ".join(blockers)
        )
        try:
            state.upsert_gap({
                "canonical_id": gap_cid,
                "symptom": symptom,
                "layer": "exploration_depth",
                "severity": "high",
                "source": "depth_gate",
            })
        except Exception:  # noqa: BLE001 — defensive
            log.exception("depth-gate: upsert_gap failed")
        log.info(
            "depth-gate: rewrote steward '%s' -> 'continue_explore' "
            "(blockers=%s, next_action=%r, task=%s)",
            raw_rec, blockers, next_action, task.task_id,
        )
        return "continue_explore", gap_cid

    def _depth_gate_thresholds(self) -> dict[str, int]:
        """Resolve depth-gate thresholds from SharedState overrides,
        falling back to the phase_state defaults."""
        overrides = getattr(self.shared_state, "depth_gate_thresholds", None)
        out: dict[str, int] = {}
        if isinstance(overrides, dict):
            for key in (
                "scout_runs_min", "prs_fetched_min", "pr_diffs_read_min",
                "nvidia_refs_min", "code_patches_min", "reverts_to_evaluate",
            ):
                if overrides.get(key) is not None:
                    try:
                        out[key] = int(overrides[key])
                    except (TypeError, ValueError):
                        pass
        return out

    async def _maybe_autosubmit_specialist_patches(
        self, *, task: "Task", done_payload: dict[str, Any],
    ) -> None:
        """B3 (Arbor-into-Hyperloom follow-up): auto-surface a specialist's
        source patches to the Critic.

        When a specialist writes ``patches_written`` into its worktree, push
        a synthetic ``integrate_patch`` proposal onto the bus. The Critic
        reviews it and — on approve — ``_handle_single_verdict`` both mirrors
        the verdict onto ``SharedState.specialist_patch_verdicts`` (so
        PolicyGate's ``integrate_patch_requires_critic_verdict`` gate passes)
        and calls ``_materialize_approved_proposal`` to queue the
        integrate_patch task. Mirrors the dynamic_action critic-dispatch
        pattern (see ``_dispatch_dynamic_patch_to_critic``).

        Without this, source patches sit unreviewed in the worktree forever:
        the Critic only ever reviews ``explore`` variants, so the gate keeps
        denying every integrate_patch delegate and config tuning spins with
        no path to a code patch.

        Idempotent: skips when a verdict for this specialist is already on
        record (resume / re-entry) or a pending integrate_patch proposal for
        the same ``specialist_task_id`` is already awaiting the Critic.
        """
        patches = done_payload.get("patches_written") or []
        if not isinstance(patches, list) or not patches:
            return
        sid = str(task.task_id or "").strip()
        if not sid:
            return
        # B4 guard: never submit a patch set that does not exist on disk.
        # ``patches_written`` may carry a dangling claim (worktree never
        # materialised / agent reported a path it never wrote). Resolve each
        # against the specialist worktree + workspace; only submit when at
        # least one real file is found, otherwise integrate_patch would apply
        # nothing. The runner's _finalize already validates for the dispatch
        # path; this re-checks the intent-routing (defense-in-depth) path too.
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

    async def _route_steward_verdict(
        self, *, task: "Task", done_payload: dict[str, Any],
    ) -> None:
        """IR-7 — process a session_steward_specialist verdict.

        Side effects depending on ``recommendation``:

        * ``stop_session``    → ``state.set_pending_escalate_hint('skip_to_sweep')``
          (non-terminal: winds EXPLORE down to SWEEP → CLOSE rather than
          aborting the session).
        * ``advance_to_kernel`` → ``state.set_pending_escalate_hint('skip_to_kernel')``.
        * ``continue_explore`` → append ``next_gap_canonical_id`` to
          ``state.gaps[]``, reset ``params_no_promote_streak`` and
          per-domain empty streak counters, set
          ``steward_continuation_used=True`` (audit marker only).

        A ``stop_session`` / ``advance_to_kernel`` verdict is run through
        the exploration-depth gate: if the session has not explored
        deeply enough it is rewritten to ``continue_explore`` with a
        concrete deepening instruction. This can repeat; the only
        backstop is the IR-6 budget gate (checked first), which lets the
        original stop / advance through once budget is exhausted.

        Infrastructure failures (stale heartbeat, subprocess timeout, empty
        synthesis) schedule up to :data:`_STEWARD_MAX_INFRA_RETRIES` fresh
        steward tasks with a new idempotency key. Only genuine LLM
        out-of-vocab strings coerce to ``stop_session``.
        """
        raw_rec = str(done_payload.get("recommendation") or "").strip().lower()
        if raw_rec not in _STEWARD_RECS:
            if Coordinator._steward_infrastructure_failure(done_payload):
                round_at = int(
                    (self.shared_state.explore_search or {}).get("cursor") or 0
                )
                failures = dict(
                    getattr(
                        self.shared_state,
                        "steward_infra_failures_by_round",
                        None,
                    ) or {},
                )
                round_key = str(round_at)
                attempt = int(failures.get(round_key, 0) or 0) + 1
                failures[round_key] = attempt
                self.shared_state.steward_infra_failures_by_round = failures
                fail_reason = str(done_payload.get("reason") or "")[:240]
                self.shared_state.record_steward_assessment(
                    recommendation="",
                    next_gap_canonical_id="",
                    remaining_potential_pct_estimate=0.0,
                    rationale=(
                        f"steward infrastructure failure ({fail_reason}); "
                        f"attempt {attempt}/{_STEWARD_MAX_INFRA_RETRIES}"
                    ),
                    task_id=task.task_id,
                    round_at_assessment=round_at,
                    source_payload=done_payload,
                )
                try:
                    self.shared_state.save(self.session_dir)
                except Exception:  # noqa: BLE001
                    log.exception("steward: save after infra failure failed")
                if attempt < _STEWARD_MAX_INFRA_RETRIES:
                    log.warning(
                        "steward: infrastructure failure (%s) for task=%s; "
                        "scheduling retry %d/%d",
                        fail_reason or "unknown",
                        task.task_id,
                        attempt,
                        _STEWARD_MAX_INFRA_RETRIES,
                    )
                    await self._enqueue_internal_steward_task(
                        reason="infra_retry",
                        retry_attempt=attempt,
                    )
                    return
                log.warning(
                    "steward: infrastructure retries exhausted for "
                    "round=%s; defaulting to advance_to_kernel "
                    "(not stop_session)",
                    round_at,
                )
                raw_rec = "advance_to_kernel"
            else:
                log.warning(
                    "steward: out-of-vocab recommendation=%r for task=%s; "
                    "coercing to 'stop_session'",
                    raw_rec, task.task_id,
                )
                raw_rec = "stop_session"
        next_gap = str(
            done_payload.get("next_gap_canonical_id") or ""
        ).strip()
        if raw_rec == "continue_explore" and not next_gap:
            log.warning(
                "steward: continue_explore missing next_gap_canonical_id "
                "for task=%s; coercing to 'advance_to_kernel'",
                task.task_id,
            )
            raw_rec = "advance_to_kernel"

        # Exploration-depth gate (deterministic stop guard). When the
        # steward wants to stop / advance but the session has not
        # explored deeply enough, rewrite the verdict to
        # ``continue_explore`` and inject a concrete deepening
        # instruction. This may fire any number of times — the only
        # backstop is the HARD budget gate, which is checked first:
        # once the budget is (about to be) exhausted the depth gate is
        # bypassed and the original stop / advance stands.
        if raw_rec in ("stop_session", "advance_to_kernel"):
            raw_rec, next_gap = self._apply_depth_gate_to_verdict(
                raw_rec=raw_rec, next_gap=next_gap, task=task,
            )

        # Record the assessment unconditionally so the audit trail
        # captures it even when the recommendation was coerced.
        try:
            potential_raw = done_payload.get(
                "remaining_potential_pct_estimate"
            )
            potential = (
                float(potential_raw)
                if isinstance(potential_raw, (int, float)) else 0.0
            )
        except (TypeError, ValueError):
            potential = 0.0
        round_at = int(
            (self.shared_state.explore_search or {}).get("cursor") or 0
        )
        self.shared_state.record_steward_assessment(
            recommendation=raw_rec,
            next_gap_canonical_id=next_gap,
            remaining_potential_pct_estimate=potential,
            rationale=str(done_payload.get("rationale") or ""),
            task_id=task.task_id,
            round_at_assessment=round_at,
            source_payload=done_payload,
        )
        # Route — the heavy work is mostly already in helper writers
        # on SharedState.
        if raw_rec == "stop_session":
            from .phase_state import ESCALATE_HINT_SKIP_TO_SWEEP
            self.shared_state.set_pending_escalate_hint(
                ESCALATE_HINT_SKIP_TO_SWEEP,
            )
            log.info(
                "steward: recommendation='stop_session' for task=%s "
                "-> pending_escalate_hint='skip_to_sweep' (non-terminal; "
                "EXPLORE -> SWEEP -> CLOSE)",
                task.task_id,
            )
        elif raw_rec == "advance_to_kernel":
            from .phase_state import ESCALATE_HINT_SKIP_TO_KERNEL
            self.shared_state.set_pending_escalate_hint(
                ESCALATE_HINT_SKIP_TO_KERNEL,
            )
            log.info(
                "steward: recommendation='advance_to_kernel' for task=%s "
                "-> pending_escalate_hint='skip_to_kernel'",
                task.task_id,
            )
        elif raw_rec == "continue_explore":
            # Inject the next gap and reset plateau counters so the next
            # tick does not immediately re-trigger plateau judgment on
            # the same evidence.
            try:
                self.shared_state.append_gap_attempt(next_gap, {
                    "action": "steward",
                    "variant_name": "session_steward_specialist",
                    "outcome": "CONTINUATION_GRANTED",
                    "rationale": str(
                        done_payload.get("rationale") or ""
                    )[:480],
                })
            except Exception:  # noqa: BLE001
                log.exception(
                    "steward: append_gap_attempt failed for gap=%s",
                    next_gap,
                )
            self.shared_state.reset_explore_plateau_proxy()
            # Per-domain empty streak reset is a courtesy — Orchestration
            # gets a clean slate to re-dispatch domains.
            self.shared_state.specialist_domain_empty_streak = {}
            self.shared_state.steward_continuation_used = True
            log.info(
                "steward: recommendation='continue_explore' for task=%s "
                "-> next_gap=%r, plateau counters reset, "
                "continuation marker set",
                task.task_id, next_gap,
            )
        # Persist immediately so a crash between this routing and the
        # broader _record_specialist_result.save still leaves the
        # routing decision durable.
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception(
                "steward: SharedState.save after routing failed for task=%s",
                task.task_id,
            )

    def _build_specialist_round_entry(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        """Translate a specialist done payload into a row for
        ``SharedState.specialist_rounds[]``.

        ``round_id`` defaults to the task_id so re-recording the same
        task overwrites idempotently (M5 single-specialist case).
        M6 batch dispatch will override this with the orchestration-
        emitted round_id (e.g. ``round-12``).
        """
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        round_id = str(
            (task.params or {}).get("round_id")
            or task.task_id
        )
        truncated_from = done_payload.get("proposals_truncated_from")
        from .specialist_domains import normalize_dispatch_tags

        # Knowledge-domain tags for breakdown attribution. The reported
        # ``tags`` (if any) take precedence over the originating dispatch
        # params; both fall back to the single-domain alias.
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
                # force batch dispatch for run_optimization.
                # ``kernel_request_handlers.run_optimization_handler`` upgrades
                # the request to ``_run_optimization_batch`` only when the
                # payload carries ``candidates_path`` (so it can fan out to
                # every reusable kernel concurrently, bounded by
                # ``_DEFAULT_KERNEL_BATCH_PARALLEL`` / Ray's per-task
                # ``num_gpus`` reservation). A missing / malformed value
                # would silently collapse the dispatch back to a
                # single-kernel run — wasting 7 idle GPUs on a typical
                # MI300X node and serializing the rest of the candidates
                # over many LLM turns. Inject it from the
                # ``last_trace_analyze`` snapshot so batch mode is
                # deterministic regardless of LLM compliance.
                # LLM-supplied value still wins.
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
                # Note: main commit 1cd9f7d also auto-injects
                # ``roofline_json`` from ``last_profile_roofline`` for
                # ``trace_analyze`` requests; that field does not exist on
                # this branch (Roofline-v2 caches the trace under
                # ``last_trace_analyze`` instead), so the inject is omitted.
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
                            "extra_server_args": rejected.get("extra_server_args", ""),
                            "attempt_count": rejected.get("attempt_count"),
                            "best_gain_pct": rejected.get("best_gain_pct"),
                            "reason": rejected.get("reason"),
                        }
                        cache_hit_source = "shared_state_kernel_rejection"
                    else:
                        # Inject base_tput tied to ``current_best.tput`` whenever
                        # Orchestration omits it on an ``integrate`` request -- the
                        # multi-KEEP integrate queue routinely drains 2-3 patches per
                        # session, and a missing base_tput would otherwise fail the
                        # second/third request with ``integrate_handler requires
                        # base_tput > 0`` (the LLM only consistently remembers the
                        # field for the first integrate). Explicit operator value
                        # still wins.
                        if (
                            kind == "integrate"
                            and not merged_payload.get("base_tput")
                        ):
                            cb_tput = (
                                self.shared_state.current_best or {}
                            ).get("tput")
                            if isinstance(cb_tput, (int, float)) and cb_tput > 0:
                                merged_payload["base_tput"] = float(cb_tput)

                        # Streaming-record callback for ``run_optimization`` batch.
                        # Without this, each batch's KEEP/REVERT sub-result is
                        # only seen by SharedState *after* asyncio.gather()
                        # wait-all returns -- so one 60-min timeout sibling
                        # starves a 5-min KEEP's integrate path for the rest
                        # of the session. With it, each sub-attempt completion
                        # writes immediately; the dispatch await still blocks
                        # until gather finishes, but the moment it unblocks the
                        # Orchestration LLM sees all KEEPs queued up via
                        # ``next_pending_keep_kernel_id``.
                        handler_kwargs: dict[str, Any] = {
                            "session_dir": self.session_dir,
                        }
                        if kind == "run_optimization":
                            handler_kwargs["record_partial"] = (
                                self._record_kernel_opt_partial
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
                # Cache trace_analyze output so subsequent identical
                # requests are short-circuited next tick. Only cache
                # real successful runs, not failures, to avoid sticky
                # errors.
                if (
                    kind == "trace_analyze"
                    and cache_hit_source is None
                    and result.get("status") in ("ok", "succeeded")
                ):
                    self.shared_state.record_trace_analyze(merged_payload, result)
                    self.shared_state.save(self.session_dir)
                # Mirror kernel-opt outcomes into SharedState so Orch
                # sees decision/speedup in its prompt next tick and
                # doesn't re-dispatch the same kernel_id forever.
                if kind == "run_optimization":
                    # In batch mode every sub-result was already streamed
                    # via ``_record_kernel_opt_partial`` while the batch
                    # was in flight. Re-recording the best sub-result
                    # here would double-count attempts (+1 per kernel)
                    # and could prematurely trip the PARTIAL retire gate.
                    # Cache-hit results never carry ``batch_mode`` so they
                    # still flow through ``record_kernel_opt`` normally.
                    if not bool(
                        isinstance(result, dict) and result.get("batch_mode")
                    ):
                        self.shared_state.record_kernel_opt(result)
                    # The per-action scoreboard (KEEP / no-promote
                    # accounting) was retired, so the post-record
                    # bookkeeping is omitted.
                    self.shared_state.save(self.session_dir)
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
        """Return a cached programmatic_handler result if applicable.

        Canonical cache key is ``last_trace_analyze``.
        """
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
        """Process ``escalate_strategy_change`` (KB_design §3.8 §7.3 +
        §3.13 M7 §5.3).

        Two responsibilities:

        1. **Always** broadcast a priority-0 ``strategy_change`` event
           so other reactors / breakdown see the LLM's intent (kept
           verbatim from the legacy contract for back-compat).
        2. When ``next_action_hint`` is in the closed ESCALATE_HINT
           vocab, stash the hint onto
           :attr:`SharedState.pending_escalate_hint` so the next phase
           compute call can act on it. ``extend_*_budget`` hints
           directly bump :attr:`SharedState.phase_budget_pct` (no
           pending state needed — the budget map is consulted on
           every tick). ``pause_specialist_<domain>`` hints bump the
           per-domain empty-streak so the next EXPLORE round skips
           that domain.

        Unknown hints are silently dropped (only logged at debug) —
        Inv-8.2 says phase decisions only act on a closed vocab.
        """
        payload = dict(intent.payload or {})
        # Always emit the broadcast first (back-compat with the legacy contract tests
        # that just count strategy_change events).
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
        # ``extend_*_budget`` mutates phase_budget_pct directly (no
        # pending hint needed — the budget map is consulted every
        # tick by phase_state).
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
        # ``pause_specialist_<domain>``: bump the per-domain
        # empty-streak counter so the next EXPLORE round won't pick
        # the domain. Unknown domain suffixes
        # are tolerated (the counter is just incremented; if the
        # domain doesn't match a real specialist nothing happens).
        if is_pause_specialist_hint(hint):
            domain = hint[len(ESCALATE_HINT_PAUSE_SPECIALIST_PREFIX):]
            self.shared_state.bump_specialist_domain_empty_streak(
                domain, empty=True,
            )
            self.shared_state.last_consumed_escalate_hint = hint
            self.shared_state.last_consumed_escalate_hint_ts = now_ts
            self.shared_state.save(self.session_dir)
            return
        # ``skip_to_kernel`` / ``skip_to_close`` are deferred — stash
        # them; the next ``compute_next_phase`` will pick them up.
        self.shared_state.set_pending_escalate_hint(hint)
        self.shared_state.save(self.session_dir)

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
        self,
        source: str,
        intent: Intent,
        denied: PolicyDenied,
        *,
        action_name: str | None = None,
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
        resolved_action = action_name or str(
            (intent.payload or {}).get("action_name") or ""
        )
        # the streak counter is still a fact (LLM sees it via
        # policy_denial_history for self-correction), but the system no
        # longer reacts to it: neither auto-pruning the action family
        # (streak >= 5) nor terminating the run with a ``policy_loop``
        # stop_reason (streak >= 10). Long-run continuity is prioritised
        # over loop-detection stop-loss — the LLM may keep retrying a
        # denied action and the run continues until the wall-clock
        # deadline (or another stop_reason) fires.
        self.shared_state.record_policy_denial(
            action_name=resolved_action,
            rule=str(denied.rule or ""),
            hint=str(denied.hint or ""),
            intent_type=intent.type.value,
            tick=int(self.shared_state.tick or 0),
            intent_payload=intent.payload,
        )

    async def _record_observation(self, source: str, topic: str, payload: dict) -> None:
        await self.bus.append_and_seq(Message.new(source, "*", topic, payload))

    async def _cursor_advance_to_latest(self, agent_name: str) -> None:
        latest = await self.bus.tail(n=1, to_agent=agent_name)
        if latest:
            top = latest[0]
            await self.cursors.advance(agent_name, seq=top.seq, msg_id=top.msg_id)

    def _record_kernel_opt_partial(self, result: dict[str, Any]) -> None:
        """Streaming callback for ``_run_optimization_batch`` sub-attempts.

        Every batch sub-result calls this the instant
        :meth:`_run_kernel_backend_sequence` returns -- well before the
        gather wait-all unblocks the parent ``run_optimization``
        handler. Each call writes the per-kernel entry to
        ``kernel_opt_attempts`` and (when warranted by the KEEP-wins
        overwrite policy) updates ``last_kernel_opt``. The state.json
        write is atomic via :meth:`SharedState.save`.

        Why this exists: the Qwen3-30B-A3B-Base session
        (20260522T093903Z) lost a k009 KEEP @4.13x because the
        gather() was still blocked on k001's GEAK 63min timeout when
        Orch tried to surface KEEPs. Streaming the record makes the
        next-tick prompt accurate even mid-batch (and makes recovery
        possible after a Coordinator crash).
        """
        try:
            self.shared_state.record_kernel_opt(result)
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            # Never let a per-sub-attempt bookkeeping hiccup propagate
            # back into asyncio.gather and poison the entire batch --
            # the worst case is we miss this one sub-result and the
            # final ``record_kernel_opt(result)`` call after gather
            # picks it up later.
            log.exception(
                "_record_kernel_opt_partial failed for kernel_id=%s",
                (result or {}).get("kernel_id") if isinstance(result, dict) else None,
            )

    async def _record_integrate_keep(self, result: dict[str, Any]) -> None:
        new_tput = result.get("new_tput")
        if not isinstance(new_tput, (int, float)) or new_tput <= 0:
            return
        if not self.shared_state.optimization_stack:
            self.shared_state.seed_stack_from_current_best()

        cb = self.shared_state.current_best or {}
        # ``result`` is a sub-agent envelope (kernel_opt → integrate_patch
        # executor result); route the read through the compat helper so a
        # legacy ``extra_sglang_args`` envelope still resolves while
        # logging a single deprecation warning. ``cb`` is Coordinator-
        # internal state and is migrated at load time by
        # ``_migrate_legacy_extra_sglang_args_keys``, so a direct .get
        # is safe.
        extra_args = (
            read_extra_server_args(result)
            or (
                str(cb.get("extra_server_args") or "")
                if isinstance(cb, dict) else ""
            )
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
            # Mirror ``optimization_stack`` into the V1-schema
            # ``gain_per_stack_entry`` ledger so session_breakdown's
            # attribution + capability_summary can attribute "how much of
            # the validated cumulative gain came from this entry" without
            # re-walking the event log. Helper computes cum_gain_after
            # (vs baseline) + delta_pct (vs previous entry) internally.
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
            # Mirror explore's single-writer pattern: an integrate KEEP
            # has already been validated by the kernel-agent rebench,
            # so promote it into ``cumulative_gain_validated`` and run
            # the same 10% watermark check the explore branch uses.
            # This collapses kernel + explore gain bookkeeping into one
            # mechanism so a kernel-driven gain can fire a fresh
            # roofline just like an explore-driven one.
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

    # ==================================================================
    # Dispatcher (pulls queued tasks → SubAgentRunner)
    # ==================================================================
    def _is_promotable_result(self, task_kind: str, result: dict[str, Any]) -> bool:
        if not isinstance(result, dict):
            return False
        if task_kind in ("baseline", "profile"):
            return is_valid_measurement(result)
        if task_kind == "sweep":
            return result.get("status") == "succeeded"
        # ``replay_warm_recipe`` ALWAYS routes through _promote_to_shared_state
        # -> _promote_warm_replay, which owns the full succeeded / drift /
        # FAILED bookkeeping (it clears warm_replay_outcome.status='in_flight'
        # on every terminal outcome). If a failed replay were sent to
        # _handle_unpromotable_result instead, the in_flight flag would never
        # clear and warm_replay_in_flight() would block PRELUDE forever
        # (env-drift OOM/timeout -- exactly what warm-replay exists to detect
        # -- would otherwise burn the whole wall-clock budget in PRELUDE).
        if task_kind == "replay_warm_recipe":
            return True
        return result.get("status") != "failed"

    def _record_intervention_for_task(
        self, task: "Task", result: Any,
    ) -> None:
        """PR-A8 (Arbor-into-Hyperloom): log the change_type of a
        completed task into ``SharedState.intervention_mix``.

        Mapping:
        * ``task.kind == "explore"``       → ``change_type = "config"``
          (every explore KEEP is a config tweak per v0.8 M3).
        * ``task.kind == "integrate_patch"`` records a
          ``code_patch_attempt`` whenever the executor ran, satisfying the
          depth gate's "try a code patch" condition. ``status == "kept"``
          records ``code_patch`` instead, which additionally resets the
          config-only escalation counter.

        Best-effort: the caller wraps in try/except so any field
        access failure is non-fatal.
        """
        if not isinstance(result, dict):
            return
        kind = (task.kind or "").strip()
        if kind == "explore":
            # explore advances ``optimization_stack`` only when at
            # least one variant survived the KEEP gate AND the inlined
            # stack rebench. Use the ledger surrogate: a winner is
            # present in ``result.winners`` OR ``best_variant`` is set.
            winners = result.get("winners") or []
            best = result.get("best_variant")
            if not winners and not best:
                # B2: a config explore round that produced measurements but
                # KEPT nothing (all REVERT / KEEP_UNSTABLE) still counts as a
                # config-only *attempt*. Record it so the intervention-mix
                # telemetry reflects repeated fruitless config rounds even
                # when the noise floor prevents any config KEEP.
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
        """Record a failed / unpromotable task result into SharedState.

        Previously this was a baseline-only branch. The audit-trail plan
        broadens it to:

        * Always (every task kind, including kernel-owned) append to
          :attr:`SharedState.last_action_failures` via
          :meth:`SharedState.record_action_failure` — that's the rich
          rolling log Orchestration consults after the inbox rotates.
        * For the kinds in :data:`_AUDIT_ACTIONS` (baseline / profile /
          sweep / explore) also append a ``status="failed"`` entry to
          ``<kind>_attempts`` via
          :meth:`SharedState.record_action_attempt`, mirroring the
          kernel-equivalent per-action history.
        * Keep the existing baseline-specific
          ``baseline_failure_streak`` / ``stop_reason`` /
          ``baseline_not_promoted`` logic untouched (gated to
          ``task.kind == "baseline"`` while ``baseline_tput == 0``).
        """
        result_payload = result or {}
        any_changed = False
        # Per-action audit (failed attempt) for the 6 in-scope kinds.
        if task.kind in _AUDIT_ACTIONS:
            audit_extras: dict[str, Any] = {}
            # Stamp the baseline-params fingerprint so the prompt's
            # FAILURE RECOVERY block + the self-loop denial helper can
            # detect "same params failed twice; refuse a third attempt".
            # Only baseline is fingerprinted today; other actions would
            # need their own per-action fingerprint key set before this
            # stamp is meaningful.
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
        # Legacy baseline-specific gates (streak counter +
        # ``baseline_failed`` stop reason + ``baseline_not_promoted``
        # event). Kept intact so existing run_optimization heuristics
        # (e.g. abort-after-3-baselines) still apply.
        baseline_event_payload: dict[str, Any] | None = None
        if task.kind == "baseline" and self.shared_state.baseline_tput <= 0:
            self.shared_state.baseline_failure_streak += 1
            if self.shared_state.baseline_failure_streak >= 3:
                self.shared_state.set_stop_reason("baseline_failed")
            baseline_event_payload = {
                "kind": "baseline_not_promoted",
                "task_id": task.task_id,
                "failure_streak": self.shared_state.baseline_failure_streak,
                "stop_reason": self.shared_state.stop_reason,
                "result_status": result_payload.get("status"),
                "error_class": result_payload.get("error_class"),
            }
            any_changed = True
        # Mirror the roofline failure-handling that lives in the promote
        # path (see `task_kind == "roofline"` else branch): bump the
        # outer streak counter, clear the auto-roofline gate, and emit
        # the canonical operator warning so a silent watermark-refresh
        # failure (profile sub-step returned no .trace.json.gz, etc.)
        # is no longer invisible to operators / Orchestration prompt.
        if task.kind == "roofline":
            if hasattr(self.shared_state, "roofline_failure_streak"):
                self.shared_state.roofline_failure_streak += 1
            if (
                self.shared_state.auto_roofline_pending_task_id
                == task.task_id
            ):
                self.shared_state.auto_roofline_pending_task_id = ""
                await self._drain_proposals_awaiting_roofline()
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
        """Dispatch queued tasks, respecting per-lane capacity.

        concurrent dispatch path:

        1. Read the current lane-holder snapshot + per-lane capacities
           up front so the same tick can decide locally whether each
           queued task fits without re-querying SQLite per task.
        2. For each queued task: if the *expanded* lane set (incl.
           cross-lane conflicts) still has headroom in our local view,
           pre-acquire the lease via :meth:`try_acquire_many` and
           spawn an asyncio task that runs the work + collects the
           result. Update the local holder view so subsequent tasks
           in the same tick see the bump.
        3. ``asyncio.gather`` all spawned tasks before returning so
           the tick semantics stay compatible with the original
           serial dispatcher (one ``_pump_dispatcher_once`` call →
           every dispatchable queued task has finished or stayed
           queued).

        With the default capacity=1 everywhere this is behaviourally
        equivalent to serial dispatch (one task per conflict
        group runs at a time). When ``research_lane.capacity > 1``
        (M6 default 6), N specialist tasks fan out in parallel.

        Inv-7.3 (atomic acquire / release per task) holds because the
        pre-acquired lease is bound to the task's ``task_id`` and the
        runner releases it in its own ``finally`` block.
        """
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
                    # Stays queued; next tick re-evaluates after
                    # other holders release.
                    continue
                lease = await self.locks.try_acquire_many(
                    lanes_needed,
                    holder_id=task.task_id,
                    task_id=task.task_id,
                    action=task.kind,
                    ttl_sec=task.lease_ttl_sec or 60,
                )
                if lease is None:
                    # Race: another holder grabbed the lane between
                    # our local read and the acquire. Leave queued.
                    continue
                # Reflect the bump in our local view so the next task
                # in this tick sees the holder.
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
        # Gather; we want to surface per-task results in the order the
        # tasks finished but keep tick semantics simple by awaiting
        # all of them. Exceptions are folded into SubAgentResult
        # (run_task catches inside its body) but defensively absorb
        # anything that leaks here too.
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
            # Specialist bookkeeping.
            # SpecialistRunner returns the done payload under
            # ``result.result['specialist_done']`` (Gap-01 adapter
            # contract). We always run the bookkeeping pass for a
            # specialist task — including the empty-synthesised
            # path — so the per-domain streak / last_specialist
            # mirror / specialist_rounds ledger stay coherent
            # whether the LLM emitted a real done or the runner
            # synthesised one.
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
                # bump the per-EXPLORE
                # specialist dispatch counter. Robustness reads this
                # in its prompt context to detect storms (many
                # specialists dispatched with no winning proposal).
                try:
                    self.shared_state.bump_specialist_dispatched()
                except Exception:  # noqa: BLE001
                    log.exception("PR-A8: bump_specialist_dispatched failed")
            # When the dynamic_action runner returns: write the runner
            # status onto the dyn_id summary and on COMPLETED either
            # reject via the mechanical floor or push a critic-bound
            # proposal through the existing single-verdict path.
            if task.kind == DYNAMIC_ACTION_NAME:
                try:
                    await self._handle_dynamic_action_runner_result(
                        task=task, result=result,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "dynamic_action lifecycle hook failed for task=%s",
                        task.task_id,
                    )
            # intervention-mix ledger: when an explore or
            # integrate_patch task succeeds with a kept variant, log
            # the change_type so Robustness can see config-only
            # streaks. ``explore`` carries config-shaped KEEPs;
            # ``integrate_patch`` carries code_patch KEEPs.
            if task.kind in ("explore", "integrate_patch"):
                try:
                    self._record_intervention_for_task(task, result.result)
                except Exception:  # noqa: BLE001
                    log.exception(
                        "PR-A8: intervention ledger update failed for task=%s",
                        task.task_id,
                    )
            # When an integrate_patch task completes with a dyn-<id>
            # specialist_task_id, update the dyn_id summary with the
            # final KEPT / REVERTED / INTEGRATE_FAILED status.
            if task.kind == "integrate_patch":
                try:
                    self._maybe_update_dynamic_action_after_integrate(
                        task=task, result=result,
                    )
                except Exception:  # noqa: BLE001 — defensive
                    log.exception(
                        "dynamic_action integrate-completion hook "
                        "failed for task=%s",
                        task.task_id,
                    )
            # Auto-promote certain succeeded results into SharedState core
            # fields (Coordinator is the only writer of CORE_STATE_FIELDS;
            # see DESIGN §14.5 / §17.2).
            # Guard: SubAgentResult.state == "succeeded" only means the
            # executor didn't throw. Promotion is tied to task-specific
            # invariants: baseline/profile require a real measurement, while
            # grid actions require at least one successful variant.
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
            # A successful mid-run ``report`` task does not terminate the
            # run loop: the LLM may emit a report snapshot and keep
            # exploring until the wall-clock deadline (or another
            # stop_reason). The closing-phase report path still owns its
            # own terminal transition via ``in_closing``.
            # Fact-write hook. Always called so KEEP / REVERT lands
            # in the local optimization_journal + (when enabled and the
            # threshold matches) a KB lesson / pitfall write.
            #
            # ``replay_warm_recipe`` is excluded: it's a verification of
            # an existing KB recipe, not a new fact. The dedicated
            # ``_promote_warm_replay`` already writes its own journal
            # entry; routing it through the fact-write hook would
            # double-journal it AND write a misleading "warm config
            # gave +N%" lesson that would overwrite the original
            # recipe's measured_impact on KB shallow-merge.
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
            # explore-round gap update.
            # For explore tasks with per-variant outcomes, append each
            # variant's KEEP/REVERT to the matching gap's attempts log
            # (the canonical_id is plumbed via task.params, falling back
            # to the workload anchor). Then re-run the global refresh
            # so the failures/winners-history derived signals catch up.
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
        """Local-view headroom check used by the concurrent dispatcher.

        Returns True iff every lane in ``expanded_lanes`` has at least
        one open slot in our point-in-time snapshot. This is a hint —
        the authoritative gate is :meth:`ResourceLockManager.try_acquire_many`
        which re-checks under ``BEGIN IMMEDIATE``.
        """
        for lane in expanded_lanes:
            cap = int(capacities.get(lane, 1))
            used = int(holders.get(lane, 0))
            if cap <= 0 or used >= cap:
                return False
        return True

    # ------------------------------------------------------------------
    # Fact-write dispatcher (KEEP / REVERT entry point)
    # ------------------------------------------------------------------
    # Single responsibility: route every terminal task result to the
    # journal + KB fact-write helpers (``_record_fact_per_task`` /
    # ``_record_fact_per_variant``).
    # ------------------------------------------------------------------
    def _source_session_id(self) -> str:
        """Return the hyperloom-local session identifier used as the
        ``source_session_id`` field on KB fact writes.

        This is NOT a KB-side session id (the KB session protocol was
        retired). It is whatever uniquely identifies *this* optimizer
        run for cross-session traceability: prefer the cortex T0
        session id when present (until that field is removed too),
        otherwise fall back to ``session_dir.name`` which is the
        per-launch UTC timestamp directory minted by ``paths.make_session_dir``.
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
        """Per-task fact-write entry point.

        Dispatches to ``_record_fact_per_task`` (legacy single-result
        path) or ``_record_fact_per_variant`` (explore-grid path with
        ``per_variant_outcomes``). Best-effort: never raises back into
        the dispatcher; SharedState save failures are logged but not
        re-raised.
        """
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


    # ------------------------------------------------------------------
    # Fact-write surface — journal + direct KB lesson/pitfall/recipe writes
    # ------------------------------------------------------------------
    # The methods below own the *fact* side of the KB integration —
    # everything we know to be true at KEEP / REVERT / CLOSE time.
    # ------------------------------------------------------------------
    PITFALL_REGRESS_THRESHOLD_PCT: float = -5.0  # gain_pct ≤ this → pitfall
    def _ensure_journal(self) -> Journal:
        """Lazy-instantiate the per-session :class:`Journal`.

        Called from every fact-write site so the journal stays valid
        across resume (``load_or_create`` reads the existing file when
        present). Header fields are read from SharedState which by T3
        time has model / hardware / framework / baseline_tput
        populated.

        Uses :func:`getattr` rather than direct attribute access so
        unit-test stubs that bypass ``__init__`` (``Coordinator.__new__``
        pattern in the close-phase sequencer tests) still work.
        """
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
        """Decide whether a failed result warrants a pitfall row.

        Threshold-B contract (confirmed with operator):

        * ``crash`` / ``oom`` / ``hang`` → ``SEVERITY_CRASH``
        * ``gain_pct ≤ -5%``             → ``SEVERITY_REGRESS``
        * otherwise (silent revert, ties, no-op-ish negative) → ``None``

        Filtering at write time avoids polluting the shared KB with
        the long tail of marginal regressions that every marathon
        produces.
        """
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
        return str(getattr(self.shared_state, "phase", "") or "").strip().upper() or "UNKNOWN"

    def _record_fact_per_task(
        self,
        *,
        task: "Task",
        source_session_id: str,
        result_dict: dict[str, Any],
        kept: bool,
    ) -> None:
        """Per-task fact write — one journal row + maybe one KB fact.

        ``source_session_id`` is the hyperloom-local session identifier
        carried into KB attrs for traceability; it's NOT a KB-side
        session id (the KB hypothesize/verify session protocol was
        retired alongside this hook).
        """
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
        ))

        if self.cortex_kb is None:
            return

        models = [str(self.shared_state.model_name or "")] if self.shared_state.model_name else []
        hardware = [str(self.shared_state.gpu_type or "")] if self.shared_state.gpu_type else []
        # No KB upstream point to cite; evidence_refs (log:task-...)
        # still gives full traceability
        # because ``source_session_id`` lands in attrs.
        evidence_refs = [f"log:task-{task.task_id}"]
        # Workload-shape tags written into lesson/pitfall attrs so the
        # warm-start reader (``client.lessons(framework=..., ...)``)
        # can filter cross-framework noise out. Shared with the recipe
        # write path via :meth:`_collect_workload_tags`.
        workload_tags = self._collect_workload_tags()
        extra = workload_tags if workload_tags else None
        # NOTE: propose_lesson / propose_pitfall do NOT raise
        # CortexKBError on transport / business failures — the client
        # swallows them and enqueues NDJSON instead. The caller of
        # this method (``_fact_write_hook``) wraps the whole thing in
        # ``except Exception`` so an unexpected error (OSError writing
        # the pending file, programmer bug, …) still doesn't crash the
        # dispatcher. Wrapping each call site in ``except CortexKBError``
        # here would be dead code.
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
            # v2: append the lesson onto the recipe's lessons[] —
            # no cross-recipe dedup / merge of source_session_ids
            # (per the user-chosen design simplification).
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
        """Build the lesson statement / pitfall description used to
        derive the KB canonical_id.

        Critical contract: this string is hashed into the canonical_id,
        so anything that varies session-to-session for the **same
        experience** must be excluded. Otherwise N sessions writing
        the "same" lesson produce N KB rows instead of merging.

        Identity dimensions (included → become part of canonical_id):

          * ``framework`` — sglang vs vLLM lessons stay separate
            (extra_*_args blobs are incompatible).
          * ``change`` — the actual variant string (args + envs).
          * ``model`` / ``hw`` — different hardware → different
            lessons (best practices vary).

        Volatile signals (excluded → updated via measured_impact /
        validated_count instead):

          * ``gain_pct`` — measure floats drift session-to-session
            (+12.30% / +11.50% / +12.70%). Including it would explode
            "1 lesson, 5 sessions confirming" into "5 lessons,
            each validated_count=1". The actual gain lives in
            ``attrs.measured_impact`` (shallow new-wins → reflects
            most recent measurement) and the list of source sessions
            in ``attrs.source_session_ids``.

        Pitfall description uses ``severity`` (3-value enum:
        ``crash`` / ``regress`` / ``noop``) instead of gain_pct so the
        same drift-stability concern doesn't apply.

        Output shape:

        * ``[<framework>] <change> on <model>/<hw>``               for lessons
        * ``[<framework>] <change> → <severity> on <model>/<hw>``  for pitfalls

        Framework is rendered as ``[?]`` when SharedState doesn't know
        it (legacy / mock callers) so the canonical_id stays stable
        and distinguishable from a real-framework row.
        """
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
        """GAP 3 — structured ``measured_impact`` payload.

        Returns a dict instead of the legacy ``f"gain_pct=... ..."``
        string so downstream consumers (specialist prompt renderer,
        dashboard scripts, future analytical jobs) can parse the
        fields without regex. The prompt builder keeps a back-compat
        renderer for old string-form ``measured_impact`` values.

        ``stack_depth`` is the length of the current optimization
        stack BEFORE this lesson lands — useful for downstream tools
        to discount lessons stacked on top of many other knobs (a
        +10% at depth 0 is more valuable than a +10% at depth 5).
        """
        out: dict[str, Any] = {
            "gain_pct": float(gain_pct) if gain_pct is not None else None,
            "stack_depth_at_apply": int(stack_depth),
            "measured_at": measured_at,
        }
        if throughput_after is not None:
            out["throughput_after"] = float(throughput_after)
        if throughput_before is not None:
            out["throughput_before"] = float(throughput_before)
        # Strip None for compactness (prompt section relies on
        # ``.get`` so missing keys are tolerated naturally).
        return {k: v for k, v in out.items() if v is not None}

    def _record_fact_per_variant(
        self,
        *,
        task: "Task",
        source_session_id: str,
        variant_outcome: dict[str, Any],
    ) -> None:
        """Per-variant fact write — mirror of :meth:`_record_fact_per_task`
        for the explore action's per-variant decisions.

        ``source_session_id`` is the hyperloom-local session identifier
        carried into KB attrs for traceability (no KB-side session).
        """
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
        # Ensure the change summary is variant-specific. When the variant
        # dict carries no args/envs/name, summarize_change drops to the bare
        # task kind ("explore"), so every regressed explore variant would
        # write an indistinguishable KB pitfall/lesson description. Fall back
        # to the variant_name (always present) so the row identifies which
        # variant it was about.
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
        ))

        if self.cortex_kb is None:
            return

        models = [str(self.shared_state.model_name or "")] if self.shared_state.model_name else []
        hardware = [str(self.shared_state.gpu_type or "")] if self.shared_state.gpu_type else []
        evidence_refs = [
            f"log:task-{task.task_id}",
            f"variant:{variant_name}",
        ]
        # Workload-shape tags — see _record_fact_per_task; same
        # rationale (warm-start reader symmetry).
        workload_tags = self._collect_workload_tags()
        extra = workload_tags if workload_tags else None

        # See note in ``_record_fact_per_task``: the client swallows
        # CortexKBError internally and falls back to NDJSON, so no
        # per-call except-block is needed here. ``_fact_write_hook``
        # wraps the entire helper in ``except Exception``.
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

    def _build_session_snapshot(self) -> dict[str, Any]:
        """Build a session-wide state digest for ``session_steward_specialist``.

        The steward specialist needs a panoramic view of where the
        session stands (stack depth, gain trajectory, plateau signals,
        outstanding gaps, recent policy denials) to recommend
        ``continue_explore`` / ``advance_to_kernel`` / ``stop_session``.

        Previously its prompt told the LLM "everything is in
        $SESSION_DIR/state.json which the Coordinator pre-warms below"
        but no inline rendering existed — the steward had to use Bash
        to read state.json off disk, costing a turn and contradicting
        the prompt. This helper produces a compact dict the prompt
        builder renders inline so the steward sees the data in its
        first turn.

        Returned shape (all keys present, empty / 0 when source is
        empty so the prompt template never branches):

        * ``phase`` / ``tick`` — current phase + monotonic tick.
        * ``optimization_stack_len`` — count of KEEP'd entries.
        * ``cumulative_gain_pct`` / ``cumulative_gain_validated_pct``
          — total session uplift (validated_pct is the post-rebench
          number; cumulative is the raw running sum).
        * ``gain_per_stack_entry_tail`` — last 5 entries of the
          per-stack gain ledger, exposing diminishing-returns trends.
        * ``rejected_counts`` — REVERT reasons grouped by the
          explore_search ``reason`` field (``stack_unstable`` /
          ``gain_below_threshold`` / ...). A long tail of one kind is
          a plateau signal.
        * ``specialist_empty_streak`` — per-domain empty-round
          counter (``specialist_domain_empty_streak``). Three
          consecutive ``empty=True`` rounds is a hard plateau.
        * ``gaps_count`` / ``gaps_top5_canonical_ids`` — count + the
          first 5 open gap canonical_ids. Steward references one of
          these in ``next_gap_canonical_id`` when recommending
          ``continue_explore``.
        * ``policy_denial_history_tail`` — last 10 PolicyGate
          denials (rule + reason). Recurrence on the same rule means
          the LLM is thrashing — strong signal to stop.
        * ``steward_continuation_used`` — IR-7 antiloop flag; the
          steward must not recommend ``continue_explore`` twice.
        """
        ss = self.shared_state
        explore_search = getattr(ss, "explore_search", {}) or {}
        rejected_rows = explore_search.get("rejected") or []
        rejected_counts: dict[str, int] = {}
        for row in rejected_rows:
            if not isinstance(row, dict):
                continue
            reason = str(row.get("reason") or "unknown").strip() or "unknown"
            rejected_counts[reason] = rejected_counts.get(reason, 0) + 1
        gain_tail_raw = list(getattr(ss, "gain_per_stack_entry", []) or [])[-5:]
        # Coerce None → 0.0 so the prompt renderer can json.dump
        # without leaking ``null`` tokens (LLM-friendly).
        gain_tail: list[float] = []
        for v in gain_tail_raw:
            try:
                gain_tail.append(float(v) if v is not None else 0.0)
            except (TypeError, ValueError):
                gain_tail.append(0.0)
        gaps = list(getattr(ss, "gaps", []) or [])
        gaps_top5_ids = [
            str((g or {}).get("canonical_id") or "")
            for g in gaps[:5]
            if isinstance(g, dict) and (g or {}).get("canonical_id")
        ]
        denial_tail_raw = list(getattr(ss, "policy_denial_history", []) or [])[-10:]
        denial_tail: list[dict[str, str]] = []
        for d in denial_tail_raw:
            if not isinstance(d, dict):
                continue
            denial_tail.append({
                "rule":   str(d.get("rule") or ""),
                "reason": str(d.get("reason") or "")[:120],
            })
        # surface warm-replay outcome to the steward so its
        # "continue / stop / advance" decision can distinguish gain
        # that came from inheriting a KB recipe vs. gain that came
        # from EXPLORE work this session. Without this, +25%
        # cumulative_gain at tick=0 looks identical to +25% earned
        # over 30 EXPLORE rounds — the steward might wrongly conclude
        # we already have a long stack worth keeping.
        # R4-9 defense: SharedState fields can be tampered by
        # resume / migration / tests. Guard against non-dict values
        # before dereferencing ``.get`` so the snapshot never throws
        # AttributeError mid-render.
        wro_raw = getattr(ss, "warm_replay_outcome", None)
        warm_replay = dict(wro_raw) if isinstance(wro_raw, dict) else {}
        warm_replay_status = str(warm_replay.get("status") or "")
        try:
            warm_replay_actual_gain = float(warm_replay.get("actual_gain_pct") or 0.0)
        except (TypeError, ValueError):
            warm_replay_actual_gain = 0.0
        return {
            "phase":                            str(getattr(ss, "phase", "") or ""),
            "tick":                             int(getattr(ss, "tick", 0) or 0),
            "optimization_stack_len":           len(
                getattr(ss, "optimization_stack", []) or []
            ),
            "cumulative_gain_pct":              float(
                getattr(ss, "cumulative_gain", 0.0) or 0.0
            ),
            "cumulative_gain_validated_pct":    float(
                getattr(ss, "cumulative_gain_validated", 0.0) or 0.0
            ),
            "gain_per_stack_entry_tail":        gain_tail,
            "rejected_counts":                  rejected_counts,
            "specialist_empty_streak":          dict(
                getattr(ss, "specialist_domain_empty_streak", {}) or {}
            ),
            "gaps_count":                       len(gaps),
            "gaps_top5_canonical_ids":          gaps_top5_ids,
            "policy_denial_history_tail":       denial_tail,
            "steward_continuation_used":        bool(
                getattr(ss, "steward_continuation_used", False)
            ),
            "warm_replay_status":               warm_replay_status,
            "warm_replay_actual_gain_pct":      warm_replay_actual_gain,
        }

    def _collect_workload_tags(self) -> dict[str, Any]:
        """Return the workload-shape KB tag dict for the current session.

        Shared by:

        * :meth:`_build_recipe_attrs_from_state` — writes into
          ``recipe.workload`` and re-hoisted into top-level recipe
          attrs by :meth:`cortex_finalize_recipe_and_journal`.
        * :meth:`_record_fact_per_task` /
          :meth:`_record_fact_per_variant` — passed as ``extra_attrs``
          to ``propose_lesson`` / ``propose_pitfall`` so the warm-start
          reader (``client.lessons(framework=..., ...)``) and any
          future shape-filtered queries can match these rows.

        EP / PP fall back to env when ``SharedState`` is unset (legacy
        SDK callers that bypassed ``cli._seed_shared_state`` — and
        ``pp`` has no SharedState field at all because there is no CLI
        surface for it).

        GAP 5 (May 2026) — added ``model_family`` / ``framework_version``
        / ``rocm_version`` / ``aiter_version`` / ``image_digest`` plus
        the per-baseline extras (``max_running_requests`` /
        ``max_num_seqs`` / ``chunked_prefill_enabled`` /
        ``enable_torch_compile`` / ``quant_scheme`` / ``workload_mode``)
        so lesson / pitfall writes carry the same shape filters the
        recipe row gets. Without this symmetry the warm-start reader
        ``client.lessons(framework=..., ...)`` would silently drop
        rows on framework-version drift.

        Defensive ``getattr`` reads keep this helper safe against the
        ``Coordinator.__new__`` stubs the close-phase tests use.
        """
        ss = self.shared_state
        out: dict[str, Any] = {}
        framework = str(getattr(ss, "framework", "") or "").strip()
        if framework:
            out["framework"] = framework
        model_class = str(getattr(ss, "model_class", "") or "").strip()
        if model_class:
            out["model_class"] = model_class
        # Model family — derived from model_name so future warm-start
        # queries (``find_recipe_with_fallback``) can match on
        # family without re-running the slug logic at read time.
        # ``model_family`` was used by the v1 fallback ladder
        # (find_recipe_with_fallback). Under the v2 design we use
        # the exact 5-tuple canonical_id without family fallback,
        # so the family tag is no longer stamped here. (It would
        # still be useful for reporting / dashboards; left as a
        # follow-up if operators ask for it.)
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
        # EP env fallback when SharedState.ep is unset (legacy SDK
        # callers that bypassed cli._seed_shared_state).
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
        # runtime version tags. cli writes these into
        # ``stack_fingerprint_meta`` from manifest / install fingerprint
        # at boot; resume reads them back from state.json verbatim.
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
        # per-baseline workload extras (parsed from the
        # materialized YAML in BaselineExecutor). Empty dict before
        # the first baseline; downstream readers tolerate missing keys.
        #
        # Skip rule per field:
        #   * ``max_*`` integer fields → skip 0 / None (placeholder)
        #   * ``*_enabled`` / ``enable_*`` bools → skip None ONLY
        #     (``False`` is a meaningful "operator explicitly disabled"
        #     signal); without this guard Python's ``False == 0`` would
        #     erase the field from KB attrs and a future warm-start
        #     query couldn't differentiate "enabled=False" from
        #     "field never set".
        #   * string fields → skip "" / None
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
        """Collect KEEP'd kernel optimizations + their E2E verdict.

        Joins two SharedState ledgers by ``kernel_id``:

        * ``kernel_opt_attempts[kid]`` — the micro layer
          (``last_decision`` / ``last_micro_speedup`` / ``last_artifact_path``
          / ``source_file``), written by ``record_kernel_opt``.
        * ``kernel_integrate_attempts[key]`` — the E2E integrate
          verification (``best_gain_pct`` + the last attempt's ``new_tput``),
          written by ``record_kernel_integrate_result``.

        Only KEEP'd kernels are emitted. A KEEP'd kernel with no integrate
        entry surfaces with ``integrated=False`` (micro-only, E2E unknown);
        one that was integrated carries its real ``e2e_gain_pct`` / ``e2e_tput``
        even when that gain is ~0 — that "tried, no E2E payoff" conclusion is
        exactly the warm-start signal we previously dropped. Returns a list of
        ``schema.KernelOptimization``-shaped dicts (the put_recipe extras
        channel + ``Recipe.from_dict`` persist them as first-class rows).
        """
        ss = self.shared_state
        opt_attempts = getattr(ss, "kernel_opt_attempts", {}) or {}
        integ_attempts = getattr(ss, "kernel_integrate_attempts", {}) or {}
        if not isinstance(opt_attempts, dict):
            return []

        # Index integrate results by kernel_id (last write wins; the entry
        # already carries the rolled-up ``best_gain_pct`` across attempts).
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
                # Integrate-layer verdict (KEEP / REVERT / NEEDS_REVIEW).
                # ``decision`` below stays the micro-layer KEEP; this carries
                # the E2E outcome so warm-start can skip a kernel whose patch
                # was reverted at integrate (micro win, E2E loss).
                e2e_decision = str(integ.get("last_decision") or "").upper()
                try:
                    e2e_gain = float(integ.get("best_gain_pct") or 0.0)
                except (TypeError, ValueError):
                    e2e_gain = 0.0
                # Last attempt's measured throughput (the E2E re-bench number).
                for att in reversed(list(integ.get("attempts") or [])):
                    if isinstance(att, dict) and att.get("new_tput") is not None:
                        try:
                            e2e_tput = float(att.get("new_tput") or 0.0)
                        except (TypeError, ValueError):
                            e2e_tput = 0.0
                        break
            out.append({
                "kernel_id":     str(kid),
                # record_kernel_opt persists the source under
                # ``last_source_file``; ``source_file`` is a legacy
                # fallback for older ledger snapshots.
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
        """Map proven optimizations back to their research-hint origin.

        Walks the gaps[] attempts ledger and returns:

        * ``kept_sources`` — ``{variant_name: provenance}`` for KEEP
          attempts whose gap carries an external ``provenance`` (e.g. a
          research-hint PR/blog URL). Also keyed by ``kernel_id`` when
          the attempt carries one.
        * ``kept_by_gap`` — ``{gap_canonical_id: provenance}`` for every
          gap that both carries a ``provenance`` and has at least one
          KEEP attempt. This is the primary, naming-independent lookup:
          a stack entry stamped with its ``gap_canonical_id`` resolves
          its source even when its ``name`` (explore ``variant_name`` vs.
          kernel ``kernel_id``) never matches the attempt's key.
        * ``reverted_rows`` — ``what_failed``-shaped rows for REVERT
          attempts, carrying the variant name, a reason, the measured
          gain, and the originating source when known.

        Fail-soft: any malformed gap/attempt row is skipped.
        """
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
        """Materialise the recipe-shaped view of :class:`SharedState`.

        Returned dict is shallow-merged into the recipe anchor by
        :meth:`CortexKBClient.update_recipe`; see kg-usage-guide §7.4
        for the canonical field shape.

        Defensive ``getattr`` reads keep the helper safe against the
        ``Coordinator.__new__`` stubs the close-phase tests use.
        """
        ss = self.shared_state
        current_best = getattr(ss, "current_best", {}) or {}
        opt_stack = getattr(ss, "optimization_stack", []) or []
        gain_per_stack = getattr(ss, "gain_per_stack_entry", []) or []
        last_failures = getattr(ss, "last_action_failures", []) or []
        # Internal state (current_best / optimization_stack) carries the
        # renamed canonical ``extra_server_args`` after SharedState.load
        # migrates legacy keys. Read canonical first (legacy fallback for
        # pre-migration dicts), but WRITE the KB-legacy ``extra_sglang_args``
        # key into best_config / what_worked because the RecipeKB
        # best_config schema + the warm-replay reader (allowlisted) still
        # key on the legacy name. Reading the stale name here would emit an
        # empty args field and silently break warm-replay reproduction.
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
        # Prefer the last validated stack layer for launch args — current_best
        # can carry a corrupted cumulative string when promote dedupe regressed.
        if opt_stack:
            last_entry = opt_stack[-1]
            if isinstance(last_entry, dict):
                # Read the post-rename canonical keys first
                # (``_lift_to_current_best`` + explore winners write
                # ``candidate_extra_server_args`` / ``extra_server_args``);
                # keep the legacy ``*_sglang_args`` keys as a fallback for
                # pre-migration stack entries. Reading only the legacy keys
                # here yielded an empty string for every real (renamed)
                # entry, silently re-breaking the #332 best_config fix.
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
            # Prefer the gap-id provenance stamped on the entry at KEEP
            # time (naming-independent across explore / kernel stages);
            # fall back to matching the entry's name / kernel_id.
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
        # Workload-shape tags (CLI / env supplied at session start).
        # Plumbing these into recipe attrs lets future warm-start queries
        # filter by precision / parallelism / shape and pick a *closer*
        # historical recipe than ``same-family`` alone.
        # Shared with _record_fact_per_task / _record_fact_per_variant
        # via :meth:`_collect_workload_tags` so KB lesson / pitfall rows
        # get the same shape filter at write time (the warm-start
        # ``client.lessons(framework=...)`` reader pairs symmetrically).
        # Re-writing identical values to KB is harmless — shallow merge
        # on canonical_id.
        workload_tags = self._collect_workload_tags()
        # Framework version: lifted from stack_fingerprint when available.
        # ``stack_fingerprint`` on SharedState is a SHA string (Coordinator
        # writes ``state.stack_fingerprint = sha``); the per-component
        # versions live on manifest.json. Coordinator does not read the
        # manifest, so we leave framework_version unset here — the T0
        # backfill (which has access to manifest via cortex_t0) writes it.
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
                # arbor-shape provenance so the session row is self-describing
                # (before/after tput + when + which knobs). Previously omitted,
                # leaving every recipe.json session at 0.0 / "" / [].
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
        """CLOSE-time fact finalize.

        Writes the final ``update_recipe`` (best_config / what_worked /
        what_failed / stack_fingerprint / sessions) and finalises the
        local journal (total_gain_pct + final_throughput). Idempotent:
        called once from the CLOSE sequencer and again as a safety net
        from :meth:`_cortex_t4_hook`; KB merge is shallow new-wins so
        the second call is a no-op when the state hasn't changed.
        """
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
            # Hoist workload tags (precision / tp / conc / isl / osl / ...)
            # into top-level recipe attrs so warm-start queries can
            # filter on them. ``update_recipe.extra_attrs`` is shallow-
            # merged into the propose_point body so the keys land flat
            # on the recipe attrs, not nested under ``workload``.
            workload_tags = attrs.get("workload") or {}

            # ----- sessions[] read-modify-write -----
            # KB does shallow new-wins merge on attrs, so a naive
            # ``update_recipe(sessions=[my_entry])`` would obliterate
            # every prior session entry on the anchor — we'd lose the
            # "this recipe was tested by N sessions" history that
            # specialist prompts + claw-stats-service consume.
            # Read the current anchor, drop any prior entry sharing
            # our own session_id (resume / retry safety), append our
            # new entry, then write the merged list back.
            #
            # Known race: two sessions finalising the same recipe
            # concurrently both read the prior state and one of their
            # writes overwrites the other. Tolerated for now — single-
            # session finalize is by far the common case; the next
            # session that finalises will pick up the previously-lost
            # entry's traces from the audit log if forensic recovery
            # is ever needed.
            my_sessions = list(attrs["sessions"] or [])
            my_session_ids = {
                str((s or {}).get("session_id") or "")
                for s in my_sessions if isinstance(s, dict)
            }
            # v2: read-modify-write the recipe row through the
            # dispatcher. Sessions[] is merged in-process under the
            # cid lock (LocalRecipeStore.put_recipe holds flock for
            # the whole archival sequence) so concurrent finalise
            # writes don't tear each other.
            merged_sessions: list[dict[str, Any]] = list(my_sessions)
            existing_row: dict[str, Any] = {}
            if self.cortex_kb is not None:
                try:
                    cid = self._workload_canonical_id()
                    # Read the LOCAL row (authoritative for writes) so the
                    # session merge + better-throughput guard compare
                    # against what this session already persisted, not a
                    # possibly-stale central row.
                    existing_row = self.cortex_kb.local.get_recipe(canonical_id=cid) or {}
                    existing_sessions: list[dict[str, Any]] = []
                    for row in (existing_row.get("sessions") or []):
                        if not isinstance(row, dict):
                            continue
                        if str(row.get("session_id") or "") in my_session_ids:
                            # Resume / retry of the same session —
                            # our new entry supersedes the prior one.
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

            # KEEP'd kernel optimizations (incl. E2E-verified-but-no-gain)
            # ride the extras channel: put_recipe splats extras into the
            # payload and Recipe.from_dict parses ``kernel_optimizations``
            # as a first-class field. Merge with any prior rows on the live
            # recipe, de-duped by kernel_id (this session's entry wins).
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
            # Only overwrite best_config / best_throughput when THIS
            # session is an actual improvement (or the row has none).
            # A CLOSE that ended without a validated win must NOT clobber a
            # better historical config. Omitting the keys makes
            # ``_kb_amend_recipe`` preserve the live values.
            #
            # Two-part guard:
            #   1. ``has_validated_win`` — this session produced a real
            #      optimization (non-empty validated stack, positive validated
            #      gain, OR a current_best that carries launch flags). A bare
            #      baseline (no stack, gain<=0, no extra args) is NOT a win:
            #      its tput is cross-workload-incomparable and it carries no
            #      flags, so overwriting both drops the recipe's launch args
            #      and clobbers a validated config (repro session
            #      20260531T144553Z: 2813@isl256 clobbered 2532@isl1024).
            #   2. ``my_tput > live_tput`` — even a real win must beat the
            #      stored best before it replaces it.
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
            # Merge stack_fingerprint rather than replace: T0 stamps
            # vllm_version / aiter_commit / rocm_version; CLOSE only has
            # the ``{"sha": ...}`` digest. Replacing would drop the
            # version keys, so keep live keys and overlay non-empty new
            # ones.
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
        # All KB I/O above is best-effort; this catch-all surfaces
        # programmer bugs (attr lookups blowing up etc.) so CLOSE
        # step 2.5 stays defensive.
        except Exception:  # noqa: BLE001 — defensive
            log.exception("update_recipe raised unexpectedly")

    def _lift_to_current_best(
        self, task_kind: str, best_tput: float, bv: dict[str, Any],
        *, gap_canonical_id: str = "",
    ) -> None:
        """Update SharedState.current_best + recompute cumulative_gain.

        Helper for both the 1-shot KEEP threshold path and the
        cross-round consistent-winner path in _promote_to_shared_state.

        ``gap_canonical_id`` (when known) is stamped onto the appended
        stack entry so recipe sedimentation can resolve the entry's
        research-hint provenance by gap id rather than by name.
        """
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
        # Build the cumulative launch args without double-stacking a
        # candidate that already carries the full cumulative string. The
        # helper also dedupes repeated ``--flag value`` pairs so the final
        # extra_server_args reflects what argparse honors (last value wins),
        # which the reproduce-launch dashboards + session_breakdown rely on.
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
                self.shared_state.optimization_stack.append(stack_entry)
                # Mirror append into ``gain_per_stack_entry`` so indexes
                # stay aligned across the two parallel lists. See the
                # SharedState docstring for the V1 StackGainEntry contract.
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
        """Lift specific action-result fields into the persistent SharedState.

        Currently handled:

        * ``baseline``  → baseline_tput (output_throughput) + baseline_accuracy
                          (when present) + current_best snapshot
        """
        if not isinstance(result, dict):
            return
        changed = False
        # Audit-trail bookkeeping for the 6 non-kernel actions. Each
        # in-scope branch sets ``audit_decision`` (and optionally
        # ``audit_extras``); after all branches we call
        # ``record_action_attempt`` once so adding a new branch is a
        # local change. ``audit_decision`` remaining ``None`` means
        # either the kind is out of scope (kernel-owned) or the branch
        # had nothing to record.
        audit_decision: str | None = None
        audit_extras: dict[str, Any] = {}
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
                # parse workload-shape extras from the YAML so
                # subsequent lesson / pitfall writes can stamp them
                # onto attrs. Best-effort: parse errors fall back to
                # an empty dict so the rest of the promote path is
                # unaffected.
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
            # Promote the baseline Magpie wall-clock so the
            # ExploreExecutor can derive a per-variant overtime kill
            # deadline (``baseline_runtime_sec * explore_overtime_kill_ratio``).
            # Only the success path carries this field; failure paths
            # (timeout / nonzero / no_workspace / no_report) deliberately
            # omit it so a botched baseline cannot seed a tiny / huge
            # deadline downstream.
            runtime_sec_raw = result.get("subprocess_runtime_sec")
            if isinstance(runtime_sec_raw, (int, float)) and runtime_sec_raw > 0:
                self.shared_state.baseline_runtime_sec = float(runtime_sec_raw)
                changed = True
            self.shared_state.current_best = {
                "action": "baseline",
                "tput": float(tput) if isinstance(tput, (int, float)) else None,
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
                # Stamp the canonical params fingerprint so the prompt's
                # FAILURE RECOVERY block (and the self-loop denial helper)
                # can compare what was actually run against what
                # Orchestration is about to propose. See
                # ``_baseline_params_fingerprint``.
                "fingerprint": _baseline_params_fingerprint(
                    task.params if task is not None else None
                ),
            }
            # seed the gaps[] ledger from baseline.
            # Best-effort; failure is logged + absorbed inside the helper.
            await self._refresh_gaps(reason="baseline_done")
            # PRELUDE bootstrap (post-baseline):
            #
            #   Step 1. Inject warm-recipe history into dedup ledger
            #           (so already-tested REVERTs from past sessions
            #            can't be re-proposed). Cheap + KB-disabled-safe.
            #   Step 2. Enqueue warm-replay (Magpie + sglang on GPU).
            #   Step 3. Enqueue auto-analysis (roofline/profile) via
            #            :meth:`_maybe_enqueue_prelude_initial_analysis_after_baseline`.
            #            Deferred while warm-replay is ``in_flight`` and
            #            retried when replay promotion finishes.
            #
            # Ordering rationale: history-inject before warm-replay is
            # mandatory (the replay task must inherit the up-to-date
            # ledger). Warm-replay must finish before the initial
            # roofline — both launch Magpie on the same GPU/port even
            # though their lane keys differ. Stack seed + cumulative_gain
            # from a reproduced replay must land before FRAMEWORK_PR /
            # specialist dispatch (prompts read state.json at dispatch).
            #
            # Skip conditions:
            # * baseline tput missing or invalid;
            # * an analysis task is already in-flight (gate field set).
            if (
                isinstance(tput, (int, float)) and tput > 0
                and not (self.shared_state.auto_roofline_pending_task_id or "").strip()
            ):
                # Step 1 — history injection (fires regardless of
                # ``--no-warm-replay``).
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
            # separate promote path so the replay result does
            # NOT overwrite ``baseline_tput`` / ``current_best`` via
            # the regular baseline branch. The dedicated helper does
            # its own KEEP / REVERT bookkeeping.
            try:
                self._promote_warm_replay(result, task=task)
            except Exception:  # noqa: BLE001 — defensive
                log.exception("warm-replay promote failed")
            # PRELUDE initial roofline was deferred while replay ran.
            await self._maybe_enqueue_prelude_initial_analysis_after_baseline()
        elif task_kind == "profile":
            # atom now profiles natively (Magpie's atom_mi*x.sh bridges
            # PROFILE=1 to --torch-profiler-dir), so this ``skipped``
            # arm is now a *defensive* path: it still runs
            # cleanly if an out-of-date Magpie clone is in play (the
            # only realistic skipped producer left), or if a future
            # executor returns skipped for a different reason. Audit
            # the no-op as "skipped" so the action ledger doesn't claim
            # a fake promotion, and drop the pending gate for THIS task
            # id so downstream proposals are not stuck.
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
                    await self._drain_proposals_awaiting_roofline()
            else:
                audit_decision = "promoted"
                audit_extras = {
                    "trace_path": None,
                    "profile_args": None,
                    "output_throughput": result.get("output_throughput"),
                }
            # Bug C fix: surface the trace path produced by ProfileExecutor
            # to SharedState so Orch can pass a real path to the kernel
            # `trace_analyze` REQUEST instead of fabricating one.
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
                # Record the server config in effect for this trace so
                # Orchestration can decide whether to re-profile.
                profile_args = ""
                if task is not None:
                    profile_args = str(
                        (task.params or {}).get("base_extra_args") or ""
                    )
                self.shared_state.last_profile_args = profile_args
                # Stale trace_analyze cache no longer matches this
                # trace. Clear the canonical cache key.
                self.shared_state.last_trace_analyze = {}
                changed = True
                audit_extras["trace_path"] = str(trace_path)
                audit_extras["profile_args"] = profile_args
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
                    "tpot_mean_ms": result.get("tpot_mean_ms"),
                    "workspace": result.get("workspace"),
                }
                if self.shared_state.baseline_tput > 0:
                    self.shared_state.cumulative_gain = (
                        (float(tput) - self.shared_state.baseline_tput)
                        / self.shared_state.baseline_tput * 100.0
                    )
                changed = True
            # When this profile task came in as the Coordinator-internal
            # analysis action (``--no-enable-roofline`` mode), mirror
            # the roofline-branch watermark + gate handling so PRELUDE
            # bootstrap and watermark-crossing flows behave identically
            # under either kind:
            #   * refresh ``last_roofline_tput`` from the projected
            #     current tput so the next +10% step is anchored on
            #     this measurement (matches the roofline branch math);
            #   * clear ``auto_roofline_pending_task_id`` so downstream
            #     dispatches stop being held by
            #     ``_auto_roofline_pending_denial``.
            # The conditions are kind-agnostic — we always refresh the
            # watermark on a successful profile (so any operator-
            # enqueued profile also re-anchors the watermark) and
            # always clear the pending field for THIS task id so an
            # unrelated profile result cannot clear a gate set by a
            # different task.
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
                await self._drain_proposals_awaiting_roofline()
        elif task_kind == "roofline":
            # F1-3 (Roofline-v2 / ): the
            # composite ``roofline`` action runs profile +
            # trace_analyze atomically. Its executor already writes
            # ``last_profile_trace`` / ``last_profile_status`` /
            # ``last_profile_args`` and calls ``record_trace_analyze``
            # to populate ``last_trace_analyze`` (plus the top-level
            # ``roofline_snapshot_id`` mirror). Coordinator's job here
            # is therefore narrow: surface audit fields so the action
            # attempt ledger records a ``promoted`` / ``discarded``
            # decision, and bump ``changed`` so the post-promote
            # save() path persists the executor's mutations to disk.
            status = str(result.get("status") or "")
            if status == "skipped":
                # atom now profiles natively (Magpie's atom_mi*x.sh
                # bridges PROFILE=1 to --torch-profiler-dir), so this
                # arm is now the
                # *defensive* path: an out-of-date Magpie clone (or a
                # future skipped-emitting executor) still gets a clean
                # no-op rather than a spurious "discarded". Do NOT bump
                # ``roofline_failure_streak`` and do NOT touch the
                # watermark / trace_analyze snapshot.
                audit_decision = "skipped"
                audit_extras = {
                    "error_class": result.get("error_class"),
                    "error": result.get("error"),
                }
                # Still clear any pending gate keyed to this task id so
                # downstream proposals are not held by
                # ``_auto_roofline_pending_denial``.
                if (
                    task is not None
                    and self.shared_state.auto_roofline_pending_task_id
                    == task.task_id
                ):
                    self.shared_state.auto_roofline_pending_task_id = ""
                    changed = True
                    await self._drain_proposals_awaiting_roofline()
            elif status == "succeeded":
                audit_decision = "promoted"
                # prefer the executor's already-published
                # ``last_trace_analyze`` snapshot fields over the
                # result dict so the audit row stays consistent with
                # the SharedState view the LLM prompt renders. The
                # result-dict fields still win for keys the snapshot
                # does not carry (``profile_workspace`` / ``degraded``).
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
                # reset the outer roofline failure streak on a
                # successful snapshot. The streak is exposed for
                # prompt-side visibility only. ``hasattr`` lets test
                # stubs that omit the field still pass.
                if hasattr(self.shared_state, "roofline_failure_streak"):
                    self.shared_state.roofline_failure_streak = 0
                # Refresh the watermark: anchor the 10% step on the
                # projected current tput (matches the watermark check
                # so consecutive small KEEPs don't accidentally re-arm
                # before the stack has actually advanced 10% from this
                # measurement). For PRELUDE initial,
                # ``cumulative_gain_validated`` is 0 so the anchor
                # equals baseline_tput.
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
                # bump the outer failure streak. The action_failure
                # audit ledger already records the structured ``error_class``
                # / ``error`` fields; this counter mirrors that signal on
                # SharedState so prompt renderers
                # (``_format_analysis_md_full``) can surface the cumulative
                # count without having to grep ``action_attempts``.
                # Guard for test stubs that omit the field.
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
            # Clear the auto-roofline gate: regardless of success/failure,
            # this task is no longer in-flight, so subsequent dispatches
            # stop being held by ``_auto_roofline_pending_denial``. We
            # compare task ids so an unrelated roofline result (e.g.
            # operator-enqueued via internal tooling) does not
            # accidentally clear a gate set by a different task.
            if (
                task is not None
                and self.shared_state.auto_roofline_pending_task_id
                == task.task_id
            ):
                self.shared_state.auto_roofline_pending_task_id = ""
                changed = True
                await self._drain_proposals_awaiting_roofline()
        elif task_kind == "explore":
            # ``explore`` is the merged grid runner.
            # The executor already does per-variant KEEP/REVERT gating
            # *and* the inlined per-KEEP stack-rebench, so by the time
            # we land here the result.``winners`` list contains only
            # variants that beat the keep-threshold AND survived the
            # rebench (KEEP_UNSTABLE variants are surfaced separately
            # under ``keep_unstable_in_stack``).
            #
            # Coordinator responsibilities here mirror the
            # backends/params branch (single-writer for
            # ``explore_search.accepted`` + ``current_best`` +
            # ``optimization_stack`` lift), but do **not** re-threshold
            # the winners — the executor's keep_threshold_pct is
            # authoritative.
            #
            # 1. Apply the executor's ledger increment (tested /
            #    rejected / winners_history / cursor / last_round).
            update = result.get("explore_search_update")
            if isinstance(update, dict):
                self.shared_state.apply_explore_search_update(update)
                changed = True
            # 2. Search-space expansion bookkeeping (parity with the
            #    backends/params branch). Explore historically returns
            #    ``discovered_flags_update=None`` but we honour it
            #    defensively so a future enrichment path lands cleanly.
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
            # 3. Per-winner ``record_explore_accepted`` — Coordinator
            #    is the sole writer of ``explore_search.accepted``
            #    per the shared_state docstring; the executor only
            #    populates ``tested`` / ``rejected``.
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
                # 4. Lift the best winner into current_best /
                #    optimization_stack (executor already validated the
                #    rebench, so this is unconditional). ``best_tput``
                #    is the post-rebench running_base_tput per
                #    explore.py:803-810.
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
                # explore inlines the per-KEEP stack rebench, so the
                # post-rebench
                # ``running_base_tput`` measures the *current*
                # optimization_stack end-to-end. Promote it into
                # ``cumulative_gain_validated`` + advance the
                # validated_stack_len bookkeeping so the TODO 4
                # stack-rebench guard clears immediately after the
                # KEEP is recorded.
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
                    # Watermark refresh: if the new validated gain has
                    # pushed projected current tput >= 10% over the
                    # last roofline measurement, enqueue a fresh
                    # composite roofline and block subsequent
                    # specialist / explore / kernel dispatches until
                    # it lands.
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
            # FRAMEWORK_PR per-candidate result. Executor returns
            # ``status ∈ {kept, reverted, apply_failed, no_patch,
            # applied_no_bench, fetch_failed, failed}`` plus
            # ``delta_pct`` / ``output_throughput`` / ``candidate``.
            # Coordinator's job:
            #   1. Append a row to ``framework_pr_phase_progress`` so the
            #      pump knows this candidate has been processed.
            #   2. Update the current batch's
            #      ``max_gain_pct_observed_in_batch`` for plateau math.
            #   3. On KEEP: lift to current_best + optimization_stack +
            #      cumulative_gain_validated and fire a watermark
            #      roofline (single-writer path).
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
            # Update batch max-gain rolling stat (for plateau judge).
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
            # Issue-E (Saturday May 2026): sweep is a discovery-only
            # action that NEVER promotes (the branch above hard-codes
            # ``decision='discarded'``). Previously this branch also
            # bumped ``params_no_promote_streak``, which fed the
            # plateau-judgment legacy proxy and caused a single SWEEP
            # round to push the next EXPLORE phase one step closer to
            # ``plateau_explore`` for no good reason. The streak field
            # is owned by the ``params`` action lifecycle; sweep MUST
            # NOT mutate it.
            self.shared_state.save(self.session_dir)
            # SWEEP-phase post-hook: chain ``conc_sweep`` after a
            # succeeded sweep when the operator opted in via
            # ``--enable-conc-sweep``. Wrapped in best-effort because
            # the post-sweep concurrency comparison is non-critical
            # (the session has already produced a validated
            # current_best); failure to enqueue must not block the
            # SWEEP→CLOSE transition.
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
            # Bug #12 fix: write last_conc_sweep so
            # ``phase_state.exit_normal_sweep`` can fire ``conc_sweep_done``
            # and SWEEP→CLOSE transitions without waiting for budget
            # exhaustion.
            self.shared_state.record_conc_sweep(result)
            self.shared_state.save(self.session_dir)
            return
        # Audit trail (kernel-parity) for the 6 non-kernel actions: one
        # record per attempt with status="succeeded" + branch-supplied
        # decision/extras. The sweep branch records its own attempt
        # before the early ``return``; everything else lands here.
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
