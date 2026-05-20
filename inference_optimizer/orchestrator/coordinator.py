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
import json
import logging
import os
import shlex
import signal
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from ..cortex_kb_client import (
    CortexKBClient,
    CortexKBError,
    attempt_canonical_id,
    optimization_canonical_id,
)
from . import phase_state as _phase_state
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
from .pmc_workload_params import derive_pmc_roofline_params_from_config
from .policy import (
    KILL_TASK_SOURCE_ALLOWLIST,
    PolicyDenied,
    PolicyGate,
    REQUEST_ROUTING,
    REVIEW_VERDICT_SOURCE_ALLOWLIST,
    ROBUSTNESS_ONLY_SOURCE_ALLOWLIST,
    SPECIALIST_FROM_AGENT_PREFIX,
)
from .resource_lock import (
    LaneBusy,
    LaneFull,
    Lease,
    ResourceLockManager,
    SqliteLeaseBackend,
    _expand_lanes,
)
from .shared_state import SharedState
from .sub_agent_runner import SubAgentResult, SubAgentRunner
from .task_registry import Task, TaskRegistry
from .action_executors.benchmark_result import is_valid_measurement
# v0.8 §3.9 — orchestrator.scoring is retired; the legacy
# ``_scoring`` alias is gone. Action-scoring methods on Coordinator
# are now no-op stubs (KB_design §3.9 Inv-9.1).
from .system_prompts.prompt_builder import (
    FULL_ENABLED_ACTIONS,
    NO_KERNEL_ENABLED_ACTIONS,
)


log = logging.getLogger(__name__)


# Audit-trail kinds (must match shared_state._AUDIT_ACTIONS). Coordinator
# calls SharedState.record_action_attempt for these on both the
# success and failure dispatcher branches so the prompt sees a full
# audit log per non-kernel action. Kernel-owned actions intentionally
# stay outside this set — they have richer bespoke recorders
# (record_kernel_opt / record_kernel_integrate_result).
_AUDIT_ACTIONS: frozenset[str] = frozenset({
    "baseline", "profile", "backends", "params", "sweep", "validate_stack",
})

# ---------------------------------------------------------------------------
# Baseline self-loop guard (failure-recovery surface).
#
# Orchestration can recover from a baseline failure by proposing a fresh
# baseline with overrides (``params.benchmark_script`` /
# ``params.result_dir`` / different ``extra_sglang_args`` / etc. — see
# SKILL.md "Magpie leak-path salvage"). What it MUST NOT do is propose
# the *same* params after the same failure mode has fired N times in a
# row — the PolicyGate stop-loss below promotes that into a
# ``policy_denied`` observation with a ``baseline_self_loop`` rule tag
# so the prompt's FAILURE RECOVERY section sees the hint and proposes a
# different override.
#
# The fingerprint covers the eight task.params fields that actually
# change Magpie's behavior end-to-end (script choice / leak path /
# config / model / GPU / accuracy gate). Two failed attempts with
# identical fingerprints + the next proposal carrying the same
# fingerprint → denial. Bumping the threshold via env override is
# intentionally a single source of truth (tests rely on overriding
# this constant rather than monkeypatching the helper).
_BASELINE_FINGERPRINT_KEYS: tuple[str, ...] = (
    "benchmark_script",
    "result_dir",
    "extra_sglang_args",
    "extra_envs",
    "model_path",
    "gpu_type",
    "config_path",
    "disable_run_eval",
)
_BASELINE_SELF_LOOP_THRESHOLD: int = 2


def effective_closing_grace_sec(
    max_minutes: float | None,
    closing_grace_sec: float | None,
) -> float:
    """Resolve the closing-phase grace window after the wall-clock deadline.

    When ``closing_grace_sec`` is explicitly set (including ``0`` to disable
    closing), that value wins. Otherwise default to
    ``min(120, max_minutes * 60 * 0.02)`` so short smoke runs do not burn
  2 minutes on report flush.
    """
    if closing_grace_sec is not None:
        return float(closing_grace_sec)
    return min(120.0, (max_minutes or 0.0) * 60.0 * 0.02)


def _parse_iso_unix(ts: str) -> float:
    """Parse an ISO 8601 UTC timestamp into unix seconds.

    Returns ``0.0`` on any parse failure so callers can treat a missing
    timestamp as "no information"; never raises. Used by the v0.8 §3.3
    stale-specialist scanner to compute task-running duration without
    plumbing a separate ``started_unix`` column through TaskRegistry.
    """
    s = (ts or "").strip()
    if not s:
        return 0.0
    try:
        # ``fromisoformat`` accepts microsecond / timezone-aware strings.
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    if dt.tzinfo is None:
        # Treat naive timestamps as UTC for consistency with _now_iso().
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _baseline_params_fingerprint(params: dict[str, Any] | None) -> dict[str, Any]:
    """Project ``params`` to the keys that determine baseline behavior.

    Used by :meth:`Coordinator._baseline_self_loop_denial` to compare
    "is this proposal the same as the last two failed attempts?" and by
    :meth:`Coordinator._promote_to_shared_state` /
    :meth:`Coordinator._handle_unpromotable_result` to stamp every
    audit-trail entry with a stable identifier the prompt can reason
    about. Missing keys are recorded as ``None`` so absent vs explicit-
    null are indistinguishable (matches what the prompt sees).

    ``extra_envs`` is normalized into a sorted list of ``[key, value]``
    pairs so dict ordering doesn't affect equality. All values are
    stringified for the same reason.
    """
    params = params or {}
    out: dict[str, Any] = {}
    for key in _BASELINE_FINGERPRINT_KEYS:
        if key == "extra_envs":
            envs = params.get(key) or {}
            if isinstance(envs, dict):
                out[key] = sorted(
                    [str(k), str(v)] for k, v in envs.items()
                )
            else:
                out[key] = None
            continue
        value = params.get(key)
        out[key] = None if value is None else str(value)
    return out


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
    # v0.8 M1 — tentative_edge_id returned by Cortex T2
    # ``session hypothesize`` (KB_design §3.6.6 / §3.13 M1 §5.2).
    # Empty when T2 failed sync and went to NDJSON; T3 then falls back
    # to ``propose-edge + late_verified`` rather than ``verify``.
    #
    # Back-compat surface: when the proposal is a multi-variant
    # ``explore`` grid (KB_gaps/Gap-07), this field carries the
    # *representative* edge_id (first variant) so the legacy T3 hook
    # — which still verifies one edge per proposal — keeps working.
    # The full per-variant map lives in :attr:`kb_edge_ids` below;
    # Gap-08 will extend T3 to iterate that map.
    kb_edge_id: str = ""
    # The opt_canonical id minted at T2 (``opt.session-{sid}.proposal-{msg_id}``).
    # Stored on the proposal so the T3 verify path can still emit a
    # ``propose-edge`` even when the sync hypothesize failed.
    kb_opt_canonical: str = ""
    # v0.8 M5 §5 step 6 / KB_gaps/Gap-07 — per-variant edge_ids minted
    # by ``_cortex_t2_hook`` when the proposal is an ``explore``
    # action with a non-empty ``params.grid``. Keyed by variant
    # name; empty dict for non-grid proposals (kernel_opt / integrate
    # / etc.). The explore executor reads each variant's
    # ``kb_edge_id`` via :meth:`_materialize_approved_proposal`
    # stamping, so cross-session KB queries can locate the exact
    # variant that confirmed / refuted a hypothesis (instead of the
    # M3 per-proposal aggregate).
    kb_edge_ids: dict[str, str] = field(default_factory=dict)
    # v0.8 M5 §5 step 6 / KB_gaps/Gap-07 — per-variant opt_canonical
    # ids parallel to :attr:`kb_edge_ids`. Stored separately so the
    # T3 hook can rebuild ``propose-edge`` fallbacks per variant
    # (Gap-08) even when the synchronous T2 hypothesize failed.
    kb_opt_canonicals: dict[str, str] = field(default_factory=dict)


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
        cortex_kb: CortexKBClient | None = None,
        phase_budget_pct: dict[str, float] | None = None,
        knowledge_plane: Any = None,
    ):
        self.session_dir = Path(session_dir)
        self.role_registry = role_registry or default_role_registry()
        # v0.8 M1 — Cortex KB client (KB_design §3.6 + §3.13 M1).  When
        # ``None`` (legacy v0.6 cli path or ``--no-cortex``) all T2/T3/T4
        # hooks become no-ops; the rest of the Coordinator behaves
        # identically to v0.6.  The client itself is stateless apart
        # from the per-session NDJSON queue, so it can be shared across
        # threads (the Coordinator is single-event-loop anyway).
        self.cortex_kb: CortexKBClient | None = cortex_kb
        # v0.8 M4 + KB_gaps/Gap-01/Gap-02 — KnowledgePlane facade.
        # When non-None, ``_handle_delegate`` pre-warms ``pr_feed`` +
        # ``kb_subgraph`` for ``delegate{action='specialist'}`` tasks
        # before enqueue, so the SpecialistRunner prompt assembly sees
        # the latest knowledge.  ``None`` keeps the legacy code path
        # (no warmup; specialist still runs but sees empty knowledge
        # surface).
        self.knowledge_plane: Any = knowledge_plane
        # v0.8 M2 — phase budget percentages (KB_design §3.8 §5.3 +
        # §3.13 M2 §7). ``None`` means library defaults; CLI flags
        # populate this dict from ``--max-minutes-<phase>-pct``. We
        # normalise once at construction so downstream judges can
        # rely on a complete dict.
        self._phase_budget_pct: dict[str, float] = _phase_state.normalize_budget_pct(
            phase_budget_pct
        )
        # v0.8 §3.3 §4.4 — specialist stale scan threshold (seconds).
        # M5 wires real specialist sub-agents; M2 only ships the
        # scanner so the Robustness prompt block lights up the moment
        # M5 lands. Env override mirrors the rest of the v0.8 knobs.
        try:
            self._specialist_stale_sec: float = max(
                0.0,
                float(os.environ.get(
                    "INFERENCE_OPTIMIZER_SPECIALIST_STALE_SEC", "600",
                )),
            )
        except ValueError:
            self._specialist_stale_sec = 600.0
        # External-SKILL-driven configuration. Both replace the deleted
        # `setup` / `classify` orchestration actions: the SKILL caller is
        # expected to supply --model-class (or MODEL_CLASS env) and
        # --compare-against-gpu so the coordinator can seed marathon
        # priors against the right model_class. ``target_analysis`` is
        # *always* hard-gated as TODO 0 (independent of this field) so a
        # marker JSON is written even when no external reference GPU was
        # requested; the field is still threaded through to the executor
        # so it knows whether to fetch real InferenceX rows or write a
        # ``reason='no_target_gpu_configured'`` marker. Both default to
        # "" so legacy callers keep working.
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
        # v0.8 M6 — sync research_lane capacity from SharedState into
        # the lane_capacity table (KB_design §3.7 §4.4). CLI/manifest
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
        # External SKILL fills `model_class` via --model-class / MODEL_CLASS
        # (the deleted `classify` action used to do this from inside the loop).
        # Only overwrite a blank value so a resumed session keeps whatever was
        # previously persisted; explicit overrides require a fresh session.
        if self._model_class_override and not (self.shared_state.model_class or "").strip():
            self.shared_state.model_class = self._model_class_override
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001 — best effort; not worth aborting boot.
                log.exception(
                    "Coordinator: failed to persist model_class=%r at boot",
                    self._model_class_override,
                )
        self.state = CoordinatorState()
        self._stop = asyncio.Event()
        self._tasks_running: list[asyncio.Task] = []

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
        # Wall-clock budget tracking for per-tick Time-budget prompt injection.
        self._run_deadline: float | None = None
        self._run_started_monotonic: float | None = None
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
        # v0.8 M2 — initialise phase machine. Fresh session enters
        # PRELUDE; resume from v0.6 (no phase field) infers a phase via
        # :func:`phase_state.infer_phase_from_state`.  Always idempotent:
        # second construction on the same session_dir is a no-op.
        self._ensure_phase_initialised()

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
        # v0.8 M1 — T4 anchor (KB_design §3.13 M1 §5.4). Drains the
        # NDJSON queue (timeout 60s) then commits the Cortex session.
        # Failures are recorded on SharedState.stop_reason so the
        # operator sees ``cortex_drain_failed`` / ``cortex_commit_failed``
        # in the final summary; the SQLite close still runs so we don't
        # leak fds.
        await self._cortex_t4_hook()
        self.db.close()

    async def _cortex_t4_hook(self) -> None:
        """T4 — drain NDJSON pending + ``session commit``.

        Called from :meth:`stop` once the reactor / dispatcher loops
        have torn down. Idempotent: a second invocation with an empty
        queue + already-committed session is a no-op (cortex-kb commit
        is itself idempotent for a given sid).

        v0.8 §3.2 §5.5 / KB_gaps/Gap-06 — when the CLOSE phase
        sequencer ran (``close_sequence_done=True``), steps 3 + 4
        already drained + committed. We early-return so the hook is
        a no-op in that case (and ``cortex_session_summary`` already
        on SharedState is the authoritative one).
        """
        if self.cortex_kb is None or not self.cortex_kb.enabled:
            return
        if getattr(self.shared_state, "close_sequence_done", False):
            # CLOSE phase sequencer already ran steps 3 + 4 inline; nothing
            # left for stop() to do here.
            return
        sid = (self.shared_state.cortex_session_id or "").strip()
        if not sid:
            return
        # 1. Drain async queue. NDJSON drains *can* take meaningful time
        #    when Cortex was unreachable mid-run; 60s is the documented
        #    upper bound (KB_design §3.13 M1 §5.4).
        try:
            drain_report = self.cortex_kb.drain_pending(timeout_sec=60.0)
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception("cortex T4 drain_pending failed")
            if not self.shared_state.stop_reason:
                self.shared_state.set_stop_reason("cortex_drain_failed")
            try:
                self.shared_state.save(self.session_dir)
            except Exception:  # noqa: BLE001
                log.exception("cortex T4 SharedState.save after drain failed")
            return
        if drain_report.get("remaining", 0) > 0:
            log.warning(
                "cortex T4 drain incomplete: %s remaining; commit will reflect "
                "queued state. flusher daemon should drain on next pickup.",
                drain_report,
            )
            if not self.shared_state.stop_reason:
                self.shared_state.set_stop_reason("cortex_drain_failed")
        # 2. Commit (sync). On failure we record commit_failed so resume
        #    can re-attempt; we do NOT delete cortex_session_id so the
        #    next coordinator wake-up picks up the same sid.
        try:
            summary = self.cortex_kb.session_commit(sid)
        except CortexKBError as exc:
            log.warning("cortex T4 session_commit failed: %s", exc)
            if not self.shared_state.stop_reason:
                self.shared_state.set_stop_reason("cortex_commit_failed")
            summary = {"status": "commit_failed", "error": str(exc)[:512]}
        self.shared_state.cortex_session_summary = dict(summary)
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception("cortex T4 SharedState.save after commit failed")

    # ==================================================================
    # v0.8 §3.9 — scoreboard retired (KB_design §3.9 Inv-9.1)
    # ==================================================================
    # The v0.6 ``_score_action_*`` / ``_apply_action_score_update``
    # surface was deleted. The Coordinator no longer maintains a
    # per-action numeric priority; the LLM decides by reading facts
    # (phase / gaps / KB sub-graphs / specialist proposal_set / recent
    # winners). The stubs below keep the *callable surface* so the
    # existing in-tree call sites (and tests) compile without a
    # rewrite, but every method is a no-op. New code should not call
    # them at all.
    def _score_action_keep(self, action_name: str, *, gain_pct: float = 0.0) -> None:
        return None

    def _score_action_discard(self, action_name: str) -> None:
        return None

    def _score_action_failure(self, action_name: str) -> None:
        return None

    def _score_action_no_promote(self, action_name: str) -> None:
        return None

    def _score_action_lock(self, action_name: str, reason: str) -> None:
        return None

    def _apply_action_score_update(
        self,
        task_kind: str,
        result: dict[str, Any],
        *,
        promoted: bool | None = None,
        gain_vs_cb: float | None = None,
    ) -> None:
        """No-op stub retained for back-compat with v0.6 call sites.

        Real ``params_grid_exhausted`` / ``backends_search_exhausted``
        signals still flow through ``explore_search`` / breakdown —
        the LLM consumes those facts directly without a scoreboard
        intermediary (KB_design §3.9 §6).
        """
        return None

    # ==================================================================
    # v0.8 M2 — phase state machine
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
        # Fresh OR v0.6 resume.
        if self._resumed_from.get("is_resume"):
            inferred, evidence = _phase_state.infer_phase_from_state(state)
            reason = "resumed_from_v06_inferred"
            evidence = {**evidence, "resume_path": True}
        else:
            inferred = _phase_state.PHASE_PRELUDE
            reason = "phase_entered"
            evidence = {"trigger": "fresh_session"}
        state.record_phase_transition(
            to_phase=inferred,
            reason=reason,
            evidence=evidence,
        )
        try:
            state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive
            log.exception("Coordinator: save after phase init failed")

    def _kernel_enabled(self) -> bool:
        # Mirror the persisted ``kernel_enabled`` flag — CLI's
        # ``--no-kernel`` removes the kernel role; resume picks the
        # value from state.json.
        return "kernel" in self.role_registry and bool(
            getattr(self.shared_state, "kernel_enabled", True)
        )

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
        next_phase = _phase_state.compute_next_phase(
            state,
            kernel_enabled=self._kernel_enabled(),
            budget_pct=self._phase_budget_pct,
        )
        if next_phase is None:
            return
        target, reason, evidence = next_phase
        if target == (state.phase or "").upper():
            return  # already there
        prior = state.phase
        # v0.8 M7 — when the transition was driven by an escalate
        # hint, consume it so the next tick re-evaluates against
        # fresh signals (KB_design §3.8 §7.3). The phase_state module
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
        # the next run() tick winds the loop down (KB_design §3.8
        # §6 + §3.13 M7 §5.3 skip_to_close path).
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
        # v0.8 §3.2 — phase-entry side effects (KB_gaps/Gap-02 PR 5.4 +
        # follow-up Gap-04 / Gap-05 / Gap-06). Side effects are
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

        * ``EXPLORE`` — pre-warm PR feed across every specialist domain
          (KB_design §3.6 + KB_gaps/Gap-02). The first specialist
          dispatch after EXPLORE entry then sees a populated cache
          rather than a cold ``pr_monitor`` fetch.
        * ``KERNEL`` — auto-enqueue ``profile`` so KERNEL phase always
          has a fresh ``last_profile_trace`` for ``select_kernels``
          (KB_design §3.2 §5.3 step 1 + KB_gaps/Gap-04). Idempotent
          via a fixed-suffix idempotency_key; skipped on resume when
          a trace already exists or when ``--no-kernel``.
        * ``SWEEP`` — auto-enqueue ``sweep`` with a recipe-driven
          (or defaults-driven) grid so SWEEP doesn't degrade to
          "LLM 自觉发 sweep" (KB_design §3.2 §5.4 + KB_gaps/Gap-05).
          Same idempotency contract as KERNEL.
        * ``CLOSE`` — run the 5-step closing sequencer (report →
          session_breakdown → NDJSON drain → Cortex commit → mark
          done; KB_design §3.2 §5.5 + KB_gaps/Gap-06). Sets the
          ``state.close_sequence_done`` flag so ``cli.finally``
          short-circuits its emergency breakdown write.

        All four phases with side effects are wired. Hook additions
        for new phases should slot into this dispatcher table.
        """
        target = (to_phase or "").upper()
        if target == _phase_state.PHASE_EXPLORE:
            await self._on_enter_explore(from_phase=from_phase)
        elif target == _phase_state.PHASE_KERNEL:
            await self._on_enter_kernel(from_phase=from_phase)
        elif target == _phase_state.PHASE_SWEEP:
            await self._on_enter_sweep(from_phase=from_phase)
        elif target == _phase_state.PHASE_CLOSE:
            await self._on_enter_close(from_phase=from_phase)

    async def _on_enter_explore(self, *, from_phase: str) -> None:
        """KB_gaps/Gap-02 PR 5.4 — warm the PR feed across every
        specialist domain on EXPLORE entry.

        Best-effort: any failure is logged + the run continues. A
        downstream specialist dispatch still calls
        :meth:`KnowledgePlane.pr_feed_warm` individually
        (``_handle_delegate`` → ``_warm_specialist_params``), so even
        a hard failure here only loses the upfront cache-priming.

        The aggregated warnings end up on
        :attr:`KnowledgePlane.last_warnings`; the breakdown collector
        (KB_design §3.12) can surface them next to the session-level
        ``pr_monitor`` status.
        """
        plane = self.knowledge_plane
        if plane is None:
            return
        try:
            results = plane.pr_feed_warm_all_domains()
        except Exception as exc:  # noqa: BLE001 — defensive
            log.warning(
                "EXPLORE entry: pr_feed_warm_all_domains failed: %r", exc,
            )
            return
        total_prs = sum(len(prs) for prs, _w in results.values())
        log.info(
            "EXPLORE entry (from=%s): warmed pr_feed across %d domains "
            "(total PRs cached=%d)",
            from_phase or "<unknown>", len(results), total_prs,
        )

    async def _on_enter_kernel(self, *, from_phase: str) -> None:
        """KB_gaps/Gap-04 — auto-enqueue ``profile`` on KERNEL entry.

        KB_design §3.2 §5.3 step 1: "进入 KERNEL 即跑一次 ``profile``
        (固定动作, 不需要 LLM propose). 该次 profile 写入
        ``last_profile_trace``, 锚定 KERNEL 阶段所有 ``select_kernels``
        的 trace_input." Without this, the LLM has to remember to
        propose profile itself; missing it would silently block every
        subsequent ``select_kernels`` / ``kernel_opt`` (sequence gate
        denies them when ``last_profile_trace == ""``).

        Skipped when:

        * ``--no-kernel`` (defense in depth: ``compute_next_phase``
          should already route to SWEEP, but a corrupt resume might
          still land us here).
        * Resume scenario with a fresh ``last_profile_trace`` already
          on disk. Re-profiling would waste 5–10 min of bench time
          on a trace we already have.

        Idempotent via a phase-scoped ``idempotency_key`` so a phase
        flap (which Inv-2.1 monotonic guard forbids in practice, but
        we still defend against) doesn't double-enqueue.

        After enqueue, mutates ``phase_history[-1].evidence`` to
        record ``auto_profile_enqueued=True`` so the breakdown
        collector (KB_design §3.12 §4 / KB_gaps/Gap-04 acceptance
        criterion 3) can verify the hook fired.
        """
        state = self.shared_state
        if not self._kernel_enabled():
            # Should not happen — compute_next_phase routes
            # --no-kernel runs straight EXPLORE → SWEEP. Defensive log:
            log.info(
                "KERNEL entry hook fired with kernel_enabled=False "
                "(from=%s); skipping auto-profile",
                from_phase or "<unknown>",
            )
            return
        if (state.last_profile_trace or "").strip():
            log.info(
                "KERNEL entry hook (from=%s): last_profile_trace already "
                "set; skipping auto-profile (resume path)",
                from_phase or "<unknown>",
            )
            self._record_phase_entry_evidence(auto_profile_skipped="trace_exists")
            return
        try:
            task = await self._enqueue_internal_profile_task(
                reason="kernel_phase_entry",
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            log.exception(
                "KERNEL entry hook: failed to enqueue auto-profile: %r", exc,
            )
            self._record_phase_entry_evidence(auto_profile_error=repr(exc)[:240])
            return
        log.info(
            "KERNEL entry (from=%s): auto-enqueued profile task=%s",
            from_phase or "<unknown>", task.task_id,
        )
        self._record_phase_entry_evidence(
            auto_profile_enqueued=True,
            auto_profile_task_id=task.task_id,
        )

    async def _enqueue_internal_profile_task(
        self, *, reason: str,
    ) -> Task:
        """Build + enqueue a Coordinator-internal ``profile`` task.

        Used by :meth:`_on_enter_kernel` (KB_gaps/Gap-04) and any
        future phase entry that needs a deterministic profile anchor.
        Routes through :meth:`TaskRegistry.create_or_return_existing`
        so the standard dispatcher / audit / lease machinery applies
        unchanged — the only thing this method is responsible for is
        building a coherent params dict + a stable idempotency_key.

        Params seeded:

        * ``source='coordinator_internal'`` so downstream consumers
          (breakdown collector, robustness alerting) can distinguish
          auto-enqueued from LLM-proposed profiles.
        * ``reason`` mirrored verbatim into the task params and the
          idempotency_key suffix so a re-entry to KERNEL (defended
          against by Inv-2.1 but possible in tests) deduplicates.
        * ``config_path`` from :attr:`SharedState.baseline_config_path`
          when set, so profile inherits the workload contract
          (CONC / ISL / OSL / TP / ...) baseline already pinned.
          Without this, the profile executor would render the
          shipped default smoke YAML and produce a useless trace.
        * ``base_extra_args`` from current_best (consistent with the
          existing ``_materialize_approved_proposal`` profile branch).

        Idempotency key: ``internal-profile-<reason>`` (phase-scoped;
        ``reason`` is currently only ``"kernel_phase_entry"`` but the
        suffix slot is reserved for future phase entries that might
        also need a profile).

        Returns the freshly-created (or returned-existing) Task. The
        caller can read ``task.task_id`` for logging or the
        ``phase_history`` evidence stamp.
        """
        state = self.shared_state
        params: dict[str, Any] = {
            "source": "coordinator_internal",
            "reason": str(reason),
        }
        if state.baseline_config_path:
            params["config_path"] = state.baseline_config_path
        cb = state.current_best or {}
        if isinstance(cb, dict):
            cb_args = str(cb.get("extra_sglang_args") or "")
            if cb_args:
                params["base_extra_args"] = cb_args
        # Last-baseline benchmark_script feeds the executor's preflight
        # path when set; falling back to the default is fine when not.
        last_bl = state.last_baseline or {}
        if isinstance(last_bl, dict):
            bs = str(last_bl.get("benchmark_script") or "").strip()
            if bs:
                params["benchmark_script"] = bs
        task, was_existing = await self.tasks.create_or_return_existing(
            kind="profile",
            params=params,
            idempotency_key=f"internal-profile-{reason}",
        )
        if was_existing:
            log.info(
                "internal-profile task already exists (idempotent: "
                "task_id=%s, state=%s)",
                task.task_id, task.state,
            )
        return task

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
    # v0.8 §3.2 §5.4 + KB_gaps/Gap-05 — SWEEP phase auto-dispatch
    # ------------------------------------------------------------------
    async def _on_enter_sweep(self, *, from_phase: str) -> None:
        """Auto-enqueue a ``sweep`` task on SWEEP entry.

        KB_design §3.2 §5.4: SWEEP must "自动构造 sweep grid (来自
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
        breakdown collector (KB_design §3.12) can verify the hook
        fired and which grid source was used.

        LLM remains free to ``propose_action='sweep'`` after this hook
        runs — duplicate proposals will collide with their own
        ``idempotency_key`` derived from the proposal_msg_id (a
        different key from the internal one), so the LLM-emitted
        sweep would create a second task. PolicyGate's phase
        allowlist still permits sweep in SWEEP phase; the auto-hook
        only guarantees a sweep runs at all.
        """
        state = self.shared_state
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

        Grid source priority (KB_design §3.2 §5.4 step 1):

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
            cb_args = str(cb.get("extra_sglang_args") or "")
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

        KB_design §3.2 §5.4: "来自 SKILL.md 默认 grid + Cortex
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
    # v0.8 §3.2 §5.5 + KB_gaps/Gap-06 — CLOSE phase 5-step sequencer
    # ------------------------------------------------------------------
    # Class-level timeouts for the CLOSE sequencer's wait-for-task
    # polls. Class attributes (rather than constants in the method)
    # so tests can override per-instance with small values without
    # patching method internals. Production defaults match KB_design
    # §3.2 §5.5: report ≤ 10 min (matches ``BaselineExecutor`` cap);
    # session_breakdown ≤ 5 min (tiny report, lots of headroom).
    CLOSE_REPORT_TIMEOUT_SEC: float = 600.0
    CLOSE_SESSION_BREAKDOWN_TIMEOUT_SEC: float = 300.0
    CLOSE_NDJSON_DRAIN_TIMEOUT_SEC: float = 60.0

    async def _on_enter_close(self, *, from_phase: str) -> None:
        """KB_design §3.2 §5.5 — CLOSE phase 5-step sequencer.

        Runs the fixed order the design contract requires:

        1. ``report``                — generate markdown / json report
        2. ``session_breakdown``     — write ``session_breakdown.json``
        3. NDJSON drain              — flush async Cortex queue
        4. Cortex ``session commit`` — promote hypothesize edges
        5. mark ``close_sequence_done``

        Each step records a row under
        ``phase_history[-1].evidence.close_steps`` so the breakdown
        collector (and operators) can verify the sequence completed.
        Steps are best-effort: a failure in any step stamps
        ``status='failed' / 'timeout'`` evidence but does not abort
        the remaining steps. Step 5 always runs (so the cli.finally
        short-circuit is consistent even when steps 1-4 partly
        failed).

        Idempotence: report / session_breakdown enqueue uses fixed
        idempotency_keys (``internal-report-close_phase_entry`` /
        ``internal-session_breakdown-close_phase_entry``) so a phase
        re-entry (Inv-2.1 forbids in production, but resume from a
        crash mid-sequencer counts) reuses existing tasks. NDJSON
        drain + Cortex commit are themselves idempotent for a given
        sid (KB_design §3.13 M1 §5.4).

        The sequencer runs INLINE inside the hook — it doesn't wait
        for the reactor / dispatcher tick boundary. Steps 1 and 2
        enqueue tasks then poll ``_wait_for_task_terminal`` until the
        dispatcher (which the same Coordinator.run() loop drives)
        picks them up and finishes. Step 3 + 4 call the cortex_kb
        client synchronously. Step 5 is a single SharedState write.
        """
        log.info("CLOSE entered (from=%s); starting 5-step close sequence",
                 from_phase or "<unknown>")
        await self._record_close_step("sequencer_started", status="running")

        # ---------------- Step 1: report ----------------
        try:
            report_task = await self._enqueue_internal_report_task(
                reason="close_phase_entry",
            )
            terminal_state = await self._wait_for_task_terminal(
                report_task.task_id,
                timeout_sec=self.CLOSE_REPORT_TIMEOUT_SEC,
            )
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
            terminal_state = await self._wait_for_task_terminal(
                bd_task.task_id,
                timeout_sec=self.CLOSE_SESSION_BREAKDOWN_TIMEOUT_SEC,
            )
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

        # ---------------- Step 3: NDJSON drain ----------------
        if self.cortex_kb is not None and self.cortex_kb.enabled:
            try:
                drain_report = self.cortex_kb.drain_pending(
                    timeout_sec=self.CLOSE_NDJSON_DRAIN_TIMEOUT_SEC,
                )
                remaining = int(drain_report.get("remaining", 0))
                if remaining > 0:
                    await self._record_close_step(
                        "ndjson_drain",
                        status="incomplete",
                        detail=f"remaining={remaining}",
                    )
                else:
                    await self._record_close_step(
                        "ndjson_drain", status="done",
                    )
            except Exception as exc:  # noqa: BLE001 — defensive
                log.exception("CLOSE step 3 (NDJSON drain) failed")
                await self._record_close_step(
                    "ndjson_drain", status="failed",
                    detail=repr(exc)[:240],
                )
        else:
            await self._record_close_step("ndjson_drain", status="skipped")

        # ---------------- Step 4: Cortex session commit ----------------
        sid = (self.shared_state.cortex_session_id or "").strip()
        if (
            self.cortex_kb is not None
            and self.cortex_kb.enabled
            and sid
            and not self.shared_state.cortex_session_summary
        ):
            try:
                summary = self.cortex_kb.session_commit(sid)
                self.shared_state.cortex_session_summary = dict(summary)
                await self._record_close_step("cortex_commit", status="done")
            except Exception as exc:  # noqa: BLE001 — defensive
                log.exception("CLOSE step 4 (Cortex commit) failed")
                # Match _cortex_t4_hook's failure semantics so
                # observability is consistent across the two paths.
                self.shared_state.cortex_session_summary = {
                    "status": "commit_failed",
                    "error":  str(exc)[:512],
                }
                if not self.shared_state.stop_reason:
                    self.shared_state.set_stop_reason("cortex_commit_failed")
                await self._record_close_step(
                    "cortex_commit", status="failed",
                    detail=repr(exc)[:240],
                )
        else:
            # No Cortex wired / already committed / no sid → skip without
            # error. ``cortex_t4_hook`` (Coordinator.stop) also skips.
            await self._record_close_step("cortex_commit", status="skipped")

        # ---------------- Step 5: mark done ----------------
        self.shared_state.close_sequence_done = True
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

    def _ensure_action_scores_seeded(self) -> None:
        """v0.8 §3.9 — no-op stub.

        The Coordinator no longer maintains a scoreboard. Kept as a
        callable so the boot path (which used to invoke it) doesn't
        need a separate guard.
        """
        return None

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
            # v0.8 M2 — phase machine advance at tick boundary.
            await self._advance_phase_if_needed()

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
        * no remaining automated levers     → ``"no_more_leverage"``
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

        if self._resumed_from["is_resume"] and not self._resumed_from["rebuilt"]:
            await self.replay_for_resume()

        tick_n = 0
        stop_reason = ""
        closing_deadline: float | None = None
        try:
            while not stop_reason:
                tick_n += 1
                # Bump the persistent tick counter — drives cooldown / aging
                # math in orchestrator/scoring.py. Persisted on the next
                # save() (after _promote_to_shared_state or stop).
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
                # v0.8 M2 — phase machine advance at tick boundary.
                # Runs even when ``in_closing`` so CLOSE phase still gets
                # recorded into phase_history when the final breakdown
                # writer transitions us in.
                await self._advance_phase_if_needed()

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
                if await self._has_no_more_leverage():
                    stop_reason = "no_more_leverage"
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
            self.shared_state.set_stop_reason(stop_reason or "unknown")
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

        if self.shared_state.params_no_promote_streak < 5 and not self._params_grid_exhausted():
            return False
        if not self._params_grid_exhausted():
            return False
        # Phase 4 of the dedup-by-fingerprint plan: backends has parity
        # with params here. Don't switch off explore until BOTH ledgers
        # have run out of fresh fingerprints — otherwise a brief params
        # plateau prematurely shuts down the (still productive)
        # backends search.
        if not self._backends_grid_exhausted():
            return False
        if (
            self.action_registry is not None
            and self.shared_state.all_top_actions_policy_locked(
                self.action_registry,
            )
        ):
            return True
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

    def _backends_grid_exhausted(self) -> bool:
        """Backends-side parity of :meth:`_params_grid_exhausted`.

        Returns True when ``backends_search.tested`` has covered the
        default seed grid (either by count or by the executor's own
        ``backends_search_exhausted`` flag from the last round). An
        unseeded ledger returns False — Orch hasn't run backends yet,
        so it isn't "exhausted", it's simply unexplored.
        """
        search = self.shared_state.backends_search or {}
        if not isinstance(search, dict) or not search:
            return False
        # The executor stamps this on its last round when no fresh
        # fingerprint survived dedup — trust it when present.
        if search.get("backends_search_exhausted"):
            return True
        try:
            from .action_executors.backends import DEFAULT_BACKENDS_GRID
            grid_size = len(DEFAULT_BACKENDS_GRID)
        except Exception:  # noqa: BLE001
            grid_size = 0
        tested = search.get("tested") or {}
        tested_count = len(tested) if isinstance(tested, dict) else 0
        cursor = int(search.get("cursor") or 0)
        rejected_count = len(search.get("rejected") or [])
        if grid_size <= 0:
            return False
        return (
            tested_count >= grid_size
            or cursor >= grid_size
            or rejected_count >= grid_size
        )

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
            self.shared_state.crash_count += 1
            self.shared_state.save(self.session_dir)
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

        # v0.8 §3.3 — per-tick phase block injected for **every** agent.
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

        # Time budget — emitted for **every** agent that needs it. Robustness
        # consumes it to fire the ``deadline_imminent`` signal that escalates
        # to a ``delegate(report)`` wind-down; Orchestration uses it as a
        # heads-up so it stops proposing fresh explore rounds when the
        # deadline is close. Kernel and Critic ignore the section; their
        # prompt parsers don't subscribe to it.
        if (
            agent_name in ("orchestration", "robustness")
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
            if (
                agent_name == "orchestration"
                and remaining_min <= 5.0
                and not self.shared_state.closing_phase
            ):
                sections.append(
                    "WARNING: < 5 min remaining. Prefer `report` next; new "
                    "explore rounds or validate_stack will likely be cut "
                    "by the deadline."
                )

        # 1. Shared session state — gives the agent goal + progress context
        # even on tick 1 when the inbox is empty.
        sections.append("=== Shared session state ===")
        sections.append(self.shared_state.to_prompt_summary())
        if agent_name == "orchestration":
            # v0.8 §3.9 — ``target_gap_pct`` is a *fact* (how much
            # gain is still needed for ``--target-gain``), not a
            # scoring multiplier. Keep refreshing it so the prompt's
            # Mission-progress line stays current. The Action-scores
            # block has been retired (Inv-9.1); the LLM consumes
            # phase / gaps / KB / specialist_rounds instead.
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
            # v0.8 §3.3 §4.1 — Cortex T0 warm-start snapshot. Empty when
            # `--no-cortex` was set or it's the first session for this
            # (workload, hw) pair. The full JSON snapshot remains on disk
            # at runtime/cortex/.kb_warm.json for Read-tool access.
            try:
                warm_block = self.shared_state.to_warm_start_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: warm_start_summary failed")
                warm_block = ""
            if warm_block:
                sections.append("=== Warm start (Cortex T0) ===")
                sections.append(warm_block)
            # v0.8 KB_gaps/Gap-09 — surface the structured gaps[] ledger
            # right after the warm-start block so the DECISION FRAMEWORK
            # step "Current gaps" finds the canonical entries instead
            # of falling back to ``last_action_failures`` +
            # ``explore_search.winners_history`` proxies.
            try:
                gaps_block = self.shared_state.to_gaps_summary()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("Coordinator: gaps_summary failed")
                gaps_block = ""
            if gaps_block:
                sections.append("=== Current gaps ===")
                sections.append(gaps_block)

        # v0.8 §3.3 §4.4 — Robustness gets a phase budget telemetry +
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
        """Return the kernel_id awaiting integrate, or "" if none.

        Detects the "kernel_opt produced a KEEP whose patch has not yet
        been integrated into the optimization_stack" state. Returns the
        kernel_id so the gate text can name it; empty string means the
        gate is closed.

        Closed when ANY of these hold:
          * ``last_kernel_opt`` is empty (no recent kernel_opt call).
          * Last decision is not ``KEEP``.
          * The kernel_id is already retired (``rejected_kernel_ids``).
          * An ``integrate`` entry with the same kernel_id is already on
            ``optimization_stack`` (i.e. integrate already ran for this
            patch and stuck).
        """
        last = self.shared_state.last_kernel_opt or {}
        decision = str(last.get("decision") or "").upper()
        if decision != "KEEP":
            return ""
        kernel_id = str(last.get("kernel_id") or "").strip()
        if not kernel_id:
            return ""
        if kernel_id in (self.shared_state.rejected_kernel_ids or []):
            return ""
        for entry in self.shared_state.optimization_stack or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("action") != "integrate":
                continue
            if str(entry.get("kernel_id") or "") == kernel_id:
                return ""
        return kernel_id

    def _required_next_step(self) -> str:
        """Return the coordinator-enforced next step, or empty if flexible.

        The Orchestration prompt says baseline -> profile -> select_kernels,
        but the LLM can still skip to backends/params. This guard makes that
        sequence deterministic and visible in the prompt every tick.

        Pipeline (after the deletion of the in-loop ``setup`` / ``classify``
        actions, the deletion of the PMC hard-gate, and the deletion of
        the action-layer ``select_kernels`` hard-gate — ``select_kernels``
        is now only enforced as a prerequisite for ``run_optimization``
        REQUESTs at the request layer, never for explore actions like
        ``params`` / ``backends`` / ``sweep`` / ``report``):

            TODO 0  target_analysis  (only when --compare-against-gpu set)
            TODO 1  baseline
            TODO 2  profile          (kernel mode only)
            TODO 3  integrate        (kernel mode only, after kernel_opt KEEP)
            TODO 4  validate_stack   (when unvalidated KEEPs landed)

        ``validate_stack`` precedence is critical: once at least one KEEP
        has been added to ``optimization_stack`` since the last successful
        ``validate_stack``, skipping it would let the LLM keep stacking
        per-round gains that don't compose linearly and therefore over-
        report ``cumulative_gain`` in the final report.
        """
        if self.shared_state.stop_reason:
            return ""
        if not self._target_analysis_baseline_exists():
            if self._compare_against_gpu:
                return (
                    f"TODO 0/4: target_analysis is required now. "
                    f"--compare-against-gpu="
                    f"{self._compare_against_gpu!r} was set but "
                    "$SESSION_DIR/target_analysis/target_baseline.json is "
                    "missing; propose/delegate only `target_analysis` until "
                    "the external InferenceX reference has been fetched."
                )
            return (
                "TODO 0/4: target_analysis is required now. "
                "$SESSION_DIR/target_analysis/target_baseline.json is "
                "missing; propose/delegate only `target_analysis` so a "
                "reason='no_target_gpu_configured' marker JSON is written "
                "(no --compare-against-gpu was supplied, so this writes a "
                "skipped marker rather than fetching InferenceX data)."
            )
        if self.shared_state.baseline_tput <= 0:
            return (
                "TODO 1/4: baseline is required now. Propose/delegate only "
                "`baseline` until baseline_tput > 0."
            )
        # Profile / integrate guards only apply when the kernel agent is
        # alive — no-kernel runs have no way to service the request and
        # the mandate would be meaningless. Two related gates are NOT
        # surfaced here:
        # * ``pmc_roofline`` is opt-in advisory enrichment for
        #   ``kernel_opt`` via ``HYPERLOOM_ENABLE_PMC_ROOFLINE=1`` and
        #   never a prerequisite for any other action.
        # * ``select_kernels`` is a prerequisite ONLY for ``run_optimization``
        #   REQUESTs (enforced in ``_sequence_denial_for_request``); it is
        #   NOT a prerequisite for ``params`` / ``backends`` / ``sweep`` /
        #   ``report``.
        if "kernel" in self.role_registry:
            if not self.shared_state.last_profile_trace:
                return (
                    "TODO 2/4: profile is required now. Baseline exists but "
                    "last_profile_trace is empty; propose/delegate only `profile`. "
                    "Do not run backends/params/sweep yet."
                )
            pending_kid = self._kernel_opt_keep_pending()
            if pending_kid:
                return (
                    f"TODO 3/4: integrate is required now. kernel_opt "
                    f"returned KEEP for kernel_id={pending_kid!r} but the "
                    "patch has not been integrated into optimization_stack. "
                    "Emit request{target_agent='kernel', kind='integrate', "
                    f"params={{kernel_id: {pending_kid!r}}} (or "
                    "propose/delegate `integrate` / `recover` / "
                    "`validate_stack` / `report`) before any further "
                    "explore."
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

    def _baseline_self_loop_denial(
        self, proposed_params: dict[str, Any] | None,
    ) -> PolicyDenied | None:
        """Reject a fresh baseline proposal that just replays the last failure.

        The Orchestration prompt's FAILURE RECOVERY section instructs the
        LLM to introduce a new ``benchmark_script`` / ``result_dir`` /
        ``extra_sglang_args`` override after a baseline failure. This
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
            "extra_sglang_args / extra_envs."
        )
        return PolicyDenied(
            "action='baseline' denied: same-fingerprint failure streak",
            rule="baseline_self_loop",
            hint=hint,
        )

    def _sequence_denial_for_action(
        self,
        action_name: str,
        proposed_params: dict[str, Any] | None = None,
    ) -> PolicyDenied | None:
        """Reject orchestration action/delegate attempts that skip required steps.

        Phase 2 addition: once optimization_stack has unvalidated KEEPs,
        the only allowed actions are ``validate_stack`` itself, ``recover``,
        and ``report`` (the last only when stop_reason is set, which we
        already short-circuit above). Everything else is denied with a
        ``validate_stack_required`` rule so Orchestration sees the
        ``policy_denied`` and self-corrects on the next tick.

        ``proposed_params`` is the ``intent.payload["params"]`` dict
        (propose_action / delegate path). Currently only consumed by
        the baseline self-loop guard above, but the kwarg signature is
        kept open so other per-action stop-losses (e.g. params/backends
        loop guards) can plug in without further call-site churn.
        """
        action = str(action_name or "").strip()
        sequence_actions = {
            "target_analysis",
            "baseline", "profile", "pmc_roofline",
            "backends", "params", "sweep", "report",
            "integrate", "validate_stack",
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
        # the role registry — no-kernel mode skips them. Two related gates
        # are intentionally NOT enforced at the action layer:
        # * ``pmc_roofline`` is opt-in advisory enrichment for
        #   ``kernel_opt`` via ``HYPERLOOM_ENABLE_PMC_ROOFLINE=1`` and
        #   never blocks any other action. A platform that cannot run
        #   rocprof must not deadlock the explore / kernel pipeline.
        # * ``select_kernels`` is enforced at the REQUEST layer
        #   (``_sequence_denial_for_request``) for ``run_optimization``
        #   only. ``params`` / ``backends`` / ``sweep`` / ``report`` are
        #   never gated on a fresh ``last_select_kernels`` cache — those
        #   actions don't need kernel candidates to make progress.
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
            # integrate gate: kernel_opt KEEP awaiting integrate. Allow
            # integrate / validate_stack / report through; recover is not
            # in `sequence_actions` and therefore already bypasses this
            # function (no-op early return).
            pending_kid = self._kernel_opt_keep_pending()
            if pending_kid and action not in {
                "integrate", "validate_stack", "report",
            }:
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
        derived = derive_pmc_roofline_params_from_config(
            config_path,
            framework=self.shared_state.framework,
            model_path=self.shared_state.model_path,
            gpu_type=self.shared_state.gpu_type,
            output_dir=str(Path(self.session_dir) / "runs" / "pmc_roofline" / "auto"),
        )
        return derived

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
        elif it == IntentType.SPECIALIST_DONE:
            # v0.8 §3.5 §10 + KB_gaps/Gap-03 — terminal intent of a
            # specialist task. PolicyGate R3 has already validated the
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
        # v0.8 M1 — T2 anchor: mint optimization_node + hypothesize edge.
        # Best-effort + isolated so a KB hiccup never blocks the
        # Critic-review pipeline (Inv-6.2 + KB_design §3.13 M1 §5.2).
        await self._cortex_t2_hook(pending)
        self.state.pending_proposals[msg.msg_id] = pending

    async def _cortex_t2_hook(self, pending: PendingProposal) -> None:
        """Mint optimization_node + hypothesize edge(s) for a propose_action.

        Two paths:

        * **explore + grid** (v0.8 KB_gaps/Gap-07 + KB_design §3.13 M5
          §5 step 6): mints one optimization_node + one hypothesize
          edge **per variant**, populating :attr:`PendingProposal.kb_edge_ids`
          + :attr:`kb_opt_canonicals` keyed by variant name. The
          representative variant (first one with a non-empty edge_id)
          also lands on the legacy :attr:`kb_edge_id` /
          :attr:`kb_opt_canonical` fields so the existing T3 hook
          (per-proposal) keeps working until Gap-08 upgrades it.
        * **non-grid** (kernel_opt / integrate / sweep / profile /
          legacy backends / params): single optimization_node + single
          hypothesize edge — identical to the v0.8 M1 behaviour.

        Best-effort: every Cortex KB failure is downgraded to an NDJSON
        enqueue by the client itself; partial failures within a
        per-variant batch are logged but don't poison the other
        variants in the same proposal.

        Gap-anchor selection (KB_gaps/Gap-07 + Gap-09):
        :meth:`_resolve_issue_canonical` consults
        ``payload.gap_canonical_id`` and ``payload.params.gap_canonical_id``
        before falling back to the M1 ``workload_canonical_id``
        anchor. Once Gap-09 lands the gaps[] ledger, the per-gap
        canonical id will also be looked up here.
        """
        if self.cortex_kb is None or not self.cortex_kb.enabled:
            return
        sid = (self.shared_state.cortex_session_id or "").strip()
        if not sid:
            return

        # Detect the per-variant path (v0.8 explore + grid).
        params = pending.payload.get("params") or {}
        grid = params.get("grid") if isinstance(params, dict) else None
        if (
            pending.action_name == "explore"
            and isinstance(grid, list)
            and grid
        ):
            await self._cortex_t2_hook_per_variant(pending, sid=sid, grid=grid)
        else:
            await self._cortex_t2_hook_single(pending, sid=sid)

    async def _cortex_t2_hook_single(
        self, pending: PendingProposal, *, sid: str,
    ) -> None:
        """Single optimization_node + single hypothesize edge.

        v0.8 M1 path — preserved verbatim for non-grid proposals
        (kernel_opt / integrate / sweep / profile / legacy
        backends / params). Per-variant grid proposals route through
        :meth:`_cortex_t2_hook_per_variant` instead.
        """
        opt_canonical = optimization_canonical_id(sid, pending.proposal_msg_id)
        gap_canonical = self._resolve_issue_canonical(pending)
        try:
            self.cortex_kb.propose_point(
                canonical_id=opt_canonical,
                kind="optimization_node",
                authority="HYPOTHESIZED",
                attrs={
                    "action":              pending.action_name,
                    "from_agent":          pending.from_agent,
                    "predicted_gain_pct":  pending.predicted_gain_pct,
                    "proposal_msg_id":     pending.proposal_msg_id,
                },
                evidence=[f"log:proposal-{pending.proposal_msg_id}"],
            )
        except CortexKBError as exc:  # defensive (client already swallows)
            log.warning("cortex T2 propose_point failed: %s", exc)
        pending.kb_opt_canonical = opt_canonical
        try:
            outcome = self.cortex_kb.hypothesize(
                sid=sid,
                from_canonical=gap_canonical,
                to_canonical=opt_canonical,
                edge_type="hypothetical",
                reason=str(pending.payload.get("reasoning") or "")[:512],
                attrs={
                    "role":   pending.from_agent,
                    "action": pending.action_name,
                    "proposal_msg_id": pending.proposal_msg_id,
                    # v0.8 M2 — phase provenance on every edge so
                    # cross-session reachability queries can filter
                    # by phase (KB_design §3.6.5.1 / §3.2 §7).
                    "phase":  (self.shared_state.phase or "").upper() or "UNKNOWN",
                },
                evidence=[f"log:proposal-{pending.proposal_msg_id}"],
            )
        except CortexKBError as exc:
            log.warning("cortex T2 hypothesize failed: %s", exc)
            outcome = {}
        edge_id = str(outcome.get("tentative_edge_id") or "").strip()
        pending.kb_edge_id = edge_id
        self._append_pending_kb_edge_row({
            "proposal_msg_id": pending.proposal_msg_id,
            "opt_canonical":   opt_canonical,
            "gap_canonical":   gap_canonical,
            "edge_id":         edge_id,
            "action":          pending.action_name,
            "ts":              datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    async def _cortex_t2_hook_per_variant(
        self,
        pending: PendingProposal,
        *,
        sid: str,
        grid: list[Any],
    ) -> None:
        """Per-variant T2: one optimization_node + one hypothesize edge
        per ``grid[i]``.

        KB_design §3.13 M5 §5 step 6 / KB_gaps/Gap-07.
        ``pending.kb_edge_ids`` / ``kb_opt_canonicals`` are populated
        keyed by variant name. The legacy single-id fields
        (``kb_edge_id`` / ``kb_opt_canonical``) carry the
        *representative* variant's ids so the existing per-proposal
        T3 hook continues to work until Gap-08 ships the per-variant
        verify path.

        Partial failures: a per-variant exception is logged and that
        variant is skipped — other variants in the same proposal
        still mint successfully. Variants without a ``name`` are
        skipped (canonical id requires one).
        """
        gap_canonical = self._resolve_issue_canonical(pending)
        phase = (self.shared_state.phase or "").upper() or "UNKNOWN"
        reason = str(pending.payload.get("reasoning") or "")[:512]
        ts_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        variant_edges: dict[str, str] = {}
        variant_canonicals: dict[str, str] = {}
        for variant in grid:
            if not isinstance(variant, dict):
                continue
            variant_name = str(variant.get("name") or "").strip()
            if not variant_name:
                # Skip nameless variants — the executor itself rejects
                # them downstream, so we don't even mint a phantom edge.
                continue
            opt_canonical = (
                f"opt.session-{sid}.proposal-{pending.proposal_msg_id}"
                f".variant-{variant_name}"
            )
            # 1. optimization_node per variant.
            try:
                self.cortex_kb.propose_point(
                    canonical_id=opt_canonical,
                    kind="optimization_node",
                    authority="HYPOTHESIZED",
                    attrs={
                        "action":              pending.action_name,
                        "from_agent":          pending.from_agent,
                        "predicted_gain_pct":  pending.predicted_gain_pct,
                        "proposal_msg_id":     pending.proposal_msg_id,
                        "variant_name":        variant_name,
                        "extra_sglang_args":   str(
                            variant.get("extra_sglang_args")
                            or variant.get("extra_args") or ""
                        ),
                        "extra_envs":          dict(
                            variant.get("extra_envs") or {}
                        ),
                        # ``provenance`` was stamped by explore.py's
                        # parser (``default_grid`` / ``llm_direct`` /
                        # ``specialist:<domain>``); record it so KB
                        # consumers can answer "did this variant come
                        # from a specialist?".
                        "provenance":          str(
                            variant.get("provenance") or "llm_direct"
                        ),
                    },
                    evidence=[
                        f"log:proposal-{pending.proposal_msg_id}",
                        f"variant:{variant_name}",
                    ],
                )
            except CortexKBError as exc:
                log.warning(
                    "cortex T2 propose_point failed for variant=%s: %s",
                    variant_name, exc,
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                log.exception(
                    "cortex T2 propose_point unexpected error for variant=%s: %r",
                    variant_name, exc,
                )
            variant_canonicals[variant_name] = opt_canonical

            # 2. hypothesize edge per variant.
            try:
                outcome = self.cortex_kb.hypothesize(
                    sid=sid,
                    from_canonical=gap_canonical,
                    to_canonical=opt_canonical,
                    edge_type="hypothetical",
                    reason=reason,
                    attrs={
                        "role":             pending.from_agent,
                        "action":           pending.action_name,
                        "proposal_msg_id":  pending.proposal_msg_id,
                        "variant_name":     variant_name,
                        "provenance":       str(
                            variant.get("provenance") or "llm_direct"
                        ),
                        "phase":            phase,
                    },
                    evidence=[
                        f"log:proposal-{pending.proposal_msg_id}",
                        f"variant:{variant_name}",
                    ],
                )
            except CortexKBError as exc:
                log.warning(
                    "cortex T2 hypothesize failed for variant=%s: %s",
                    variant_name, exc,
                )
                outcome = {}
            except Exception as exc:  # noqa: BLE001 — defensive
                log.exception(
                    "cortex T2 hypothesize unexpected error for variant=%s: %r",
                    variant_name, exc,
                )
                outcome = {}
            edge_id = str(outcome.get("tentative_edge_id") or "").strip()
            variant_edges[variant_name] = edge_id

        # Stash on PendingProposal for downstream consumers
        # (``_materialize_approved_proposal`` stamps these into the
        # grid; Gap-08 T3 will iterate the map).
        pending.kb_opt_canonicals = variant_canonicals
        pending.kb_edge_ids = variant_edges
        # Representative legacy fields — first variant with a non-empty
        # edge_id wins. Falls back to first variant when every edge
        # failed (legacy T3 path treats empty edge_id as "late propose-edge
        # fallback" anyway).
        rep_name = next(
            (n for n, eid in variant_edges.items() if eid),
            next(iter(variant_edges), ""),
        )
        if rep_name:
            pending.kb_edge_id = variant_edges.get(rep_name, "")
            pending.kb_opt_canonical = variant_canonicals.get(rep_name, "")

        # Record a single pending_kb_edges row for back-compat with the
        # existing T3 hook. The full per-variant maps live in the
        # ``variant_edges`` / ``variant_canonicals`` extension fields
        # so Gap-08 can iterate without a schema bump.
        self._append_pending_kb_edge_row({
            "proposal_msg_id":    pending.proposal_msg_id,
            "opt_canonical":      pending.kb_opt_canonical,
            "gap_canonical":      gap_canonical,
            "edge_id":            pending.kb_edge_id,
            "action":             pending.action_name,
            "variant_edges":      dict(variant_edges),
            "variant_canonicals": dict(variant_canonicals),
            "ts":                 ts_iso,
        })

    def _append_pending_kb_edge_row(self, row: dict[str, Any]) -> None:
        """Append to ``shared_state.pending_kb_edges`` + persist.

        Centralised so both T2 paths (single + per-variant) share the
        same capping / save-error semantics.
        """
        pending_edges = list(self.shared_state.pending_kb_edges or [])
        pending_edges.append(row)
        # Cap to a reasonable size so resume doesn't pay quadratic costs.
        if len(pending_edges) > 256:
            pending_edges = pending_edges[-256:]
        self.shared_state.pending_kb_edges = pending_edges
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive; T2 must not crash.
            log.exception("cortex T2 SharedState.save failed")

    def _resolve_issue_canonical(self, pending: PendingProposal) -> str:
        """Find the issue_node canonical_id this proposal addresses.

        Priority (KB_gaps/Gap-07 + Gap-09 contract):

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

    def _gap_anchor_canonical_id(self) -> str:
        """M1 placeholder for the gap anchor.

        Per KB_design §3.13 M1: "M1 simplification — use
        ``workload_node.canonical_id`` as the from side of every
        hypothesize edge". M5 specialist framework will introduce real
        ``issue_node`` anchors keyed by gap descriptors. Centralising
        the derivation here keeps the migration to M5 a single-line
        change.
        """
        from ..cortex_kb_client import workload_canonical_id
        workload = self.shared_state.model_name or "unknown_model"
        hw = self.shared_state.gpu_type or "unknown_gpu"
        return workload_canonical_id(workload, hw)

    def _pop_pending_kb_edge(self, proposal_msg_id: str) -> dict[str, Any] | None:
        """Remove + return the pending edge entry for a proposal_msg_id.

        Used by T3 (KEEP/REVERT) to confirm or refute the matching
        hypothetical edge. ``None`` when the entry is missing (resume
        from a stale state.json, or the T2 hook was skipped).
        """
        edges = list(self.shared_state.pending_kb_edges or [])
        found: dict[str, Any] | None = None
        rest: list[dict[str, Any]] = []
        for row in edges:
            if not isinstance(row, dict):
                continue
            if found is None and row.get("proposal_msg_id") == proposal_msg_id:
                found = row
            else:
                rest.append(row)
        if found is None:
            return None
        self.shared_state.pending_kb_edges = rest
        return found

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
        # v0.8 KB_gaps/Gap-07 — stamp per-variant kb_edge_id into the
        # grid so the explore executor (which already reads
        # ``variant.get("kb_edge_id")``) can carry the id through to
        # the result rows the ledger writer + T3 hook will consume.
        # No-op when the proposal isn't ``explore`` or when the T2
        # hook didn't populate the map (e.g. ``--no-cortex`` runs).
        if (
            pending.action_name == "explore"
            and pending.kb_edge_ids
            and isinstance(params.get("grid"), list)
        ):
            stamped_grid: list[dict[str, Any]] = []
            for variant in params["grid"]:
                if not isinstance(variant, dict):
                    stamped_grid.append(variant)
                    continue
                variant_copy = dict(variant)
                vname = str(variant_copy.get("name") or "").strip()
                edge_id = pending.kb_edge_ids.get(vname, "")
                if edge_id and not variant_copy.get("kb_edge_id"):
                    variant_copy["kb_edge_id"] = edge_id
                stamped_grid.append(variant_copy)
            params["grid"] = stamped_grid
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
            # The ``backends_search`` ledger (Phase 2 of the dedup-by-
            # fingerprint plan) covers everything ``tested_variant_names``
            # used to provide and more: it records ALL tested variants
            # (not just winners), and is consulted for both the default
            # grid and LLM-supplied ``params.grid``.
            if pending.action_name == "backends":
                params.setdefault(
                    "synergy_attempted",
                    list(self.shared_state.synergy_attempted),
                )
                params.setdefault("max_candidates_per_round", 5)
                params.setdefault("max_synergy_combos", 4)
                params.setdefault(
                    "backends_search", self.shared_state.backends_search,
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
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=pending.action_name,
            params=params,
            idempotency_key=f"approved-{pending.proposal_msg_id}",
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
        if self.shared_state.is_pruned(action_name):
            await self._record_policy_denied(
                source, intent,
                PolicyDenied(
                    f"delegate{{action_name={action_name!r}}} pruned",
                    rule="family_pruned",
                    hint=(
                        f"{action_name!r} is in pruned_families; pick another "
                        f"action from Action scores"
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
        # v0.8 §3.5 + KB_gaps/Gap-01 — specialist pre-dispatch warmup.
        # When the Orchestration role delegates a specialist task, the
        # Coordinator is the only place with the KnowledgePlane facade
        # in scope. Warm the prompt's external-knowledge sections here
        # so SpecialistRunner's prompt assembly sees them via task
        # params. ``setdefault`` lets the caller (Orchestration) pre-
        # supply values; we only fill the gaps.
        if action_name == "specialist":
            self._warm_specialist_params(params)
        # T2 — same synergy-dedup + per-round cap plumbing as the
        # proposal-review path. Dedup against previously-tested variants
        # rides on the shared ``backends_search`` ledger (Phase 2 of the
        # dedup-by-fingerprint plan).
        if action_name == "backends":
            params.setdefault(
                "synergy_attempted",
                list(self.shared_state.synergy_attempted),
            )
            params.setdefault("max_candidates_per_round", 5)
            params.setdefault("max_synergy_combos", 4)
            params.setdefault(
                "backends_search", self.shared_state.backends_search,
            )
        raw_key = intent.payload.get("idempotency_key")
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
            task, was_existing = await self.tasks.create_or_return_existing(
                kind=action_name,
                params=params,
                idempotency_key=idempotency_key,
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
        await self.bus.append_and_seq(Message.new(
            "coordinator", "*", "event",
            {"kind": "task_queued", "task_id": task.task_id, "source": source, "action": action_name},
        ))

    # ------------------------------------------------------------------
    # v0.8 §3.5 + KB_gaps/Gap-01 — specialist pre-dispatch warmup
    # ------------------------------------------------------------------
    def _warm_specialist_params(self, params: dict[str, Any]) -> None:
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
        * ``warm_start_recipe`` / ``warm_start_pitfalls`` — mirror of
          ``SharedState.warm_start_*`` (already T0'd by cli).
        * ``framework_source_roots`` — picked up from
          :func:`resolve_source_file_allowlist` so the LLM has a stable
          local-source navigation hint without needing
          ``$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`` propagated by
          hand.
        * GPU hardware hints (``gpu_type`` / ``tp``) from SharedState.

        Gap-09 (gaps[] field) will later expose ``gap_symptom`` /
        ``gap_layer`` / ``gap_attempts`` / ``kb_subgraph`` from a real
        gap ledger; until then those stay empty (PromptBuilder
        gracefully degrades).
        """
        state = self.shared_state
        plane = self.knowledge_plane

        domain = str(params.get("domain") or "").strip()

        # PR feed (Gap-02 ↔ Gap-01 contract): if the plane is wired and
        # PR Monitor is enabled, fetch the per-domain warm cache. Any
        # exception falls back to an empty list + non-fatal warning.
        if plane is not None and "pr_feed" not in params:
            try:
                prs, _warnings = plane.pr_feed_warm(domain=domain)
                params["pr_feed"] = [
                    self._pr_summary_to_dict(p) for p in prs
                ]
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "specialist warmup: pr_feed_warm(domain=%r) failed: %r",
                    domain, exc,
                )
                params.setdefault("pr_feed", [])
        else:
            params.setdefault("pr_feed", [])

        if "pr_monitor_available" not in params:
            params["pr_monitor_available"] = bool(
                plane is not None and getattr(plane, "pr_monitor_enabled", True)
            )

        # Warm-start recipe + pitfalls from T0 anchor (KB_design §3.6).
        if state.warm_start_recipe and "warm_start_recipe" not in params:
            params["warm_start_recipe"] = dict(state.warm_start_recipe)
        if state.warm_start_pitfalls and "warm_start_pitfalls" not in params:
            params["warm_start_pitfalls"] = list(state.warm_start_pitfalls)

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

        # Hardware hints (cheap; pulled from SharedState which T0
        # populates from manifest).
        params.setdefault("gpu_type", state.gpu_type or "")
        # tp lives on baseline_config when seeded; otherwise the prompt
        # builder is fine with the dataclass default of 1.

        # v0.8 KB_gaps/Gap-09 — fill gap-specific anchors from the
        # gaps[] ledger. Orchestration carries a ``gap_canonical_id``
        # via ``delegate.params`` (and also as the M5 ``gap`` field);
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
    # v0.8 KB_gaps/Gap-09 — gaps[] ledger refresh (KB_design §3.3 / §3.5)
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
                "domain_hint": "framework_specialist",
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
                "domain_hint": "framework_specialist",
                "source": "attempts",
            })
        return gaps

    @staticmethod
    def _gap_layer_for_action(action: str) -> tuple[str, str]:
        """Map an action name → (layer, domain_hint) for gap rows.

        Centralised so the four call sites (baseline / attempts /
        explore / specialist_done) agree on the routing. Falls back
        to ``("framework", "framework_specialist")`` for unknown
        action names — that's the broadest catch-all in the M5
        specialist catalogue.
        """
        a = str(action or "").strip().lower()
        if a in {"kernel_opt", "integrate", "select_kernels", "run_optimization"}:
            return ("kernel", "kernel_specialist")
        if a in {"profile", "pmc_roofline"}:
            return ("kernel", "kernel_specialist")
        if a in {"sweep"}:
            return ("framework", "framework_specialist")
        if a in {"backends", "params", "validate_stack", "explore"}:
            return ("framework", "framework_specialist")
        if a in {"baseline"}:
            return ("system", "system_specialist")
        return ("framework", "framework_specialist")

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
                "domain_hint": "framework_specialist",
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
    # v0.8 §3.5 §10 + KB_gaps/Gap-03 — specialist_done bookkeeping
    # ------------------------------------------------------------------
    async def _handle_specialist_done(
        self, source: str, intent: Intent,
    ) -> None:
        """Handle a ``specialist_done`` intent (KB_design §3.5 §10).

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

        T2 per-variant hypothesize (KB_design §3.13 M5 §5 step 6) is
        intentionally **not** triggered here — it belongs to
        :ref:`Gap-07` which up-shifts the existing per-proposal
        ``_cortex_t2_hook`` to per-variant. Threading a stub call
        site now would silently no-op (the per-variant edge map
        ``PendingProposal.kb_edge_ids`` doesn't exist yet); the
        Gap-07 PR will add the call.
        """
        domain = str(done_payload.get("domain") or "").strip()
        proposals = done_payload.get("proposal_set") or []
        if not isinstance(proposals, list):
            proposals = []
        is_empty = bool(done_payload.get("empty")) or len(proposals) == 0

        round_entry = self._build_specialist_round_entry(
            task=task, done_payload=done_payload, source=source,
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

        # v0.8 KB_gaps/Gap-09 — refresh the gaps ledger after a
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

    def _build_specialist_round_entry(
        self,
        *,
        task: Task,
        done_payload: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        """Translate a specialist done payload into a row for
        ``SharedState.specialist_rounds[]`` (KB_design §3.12 §4.3).

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
        return {
            "round_id":          round_id,
            "task_id":           task.task_id,
            "source":            source or "coordinator",
            "completed_at":      datetime.now(timezone.utc).isoformat(),
            "domain":            str(done_payload.get("domain") or ""),
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
                        # Ran but did not promote — count toward the
                        # no-promote streak rather than the failure
                        # streak (the optimization itself didn't crash).
                        self._score_action_no_promote("kernel_opt")
                    else:
                        # Unknown decision — treat as a measurement: bump
                        # runs without rewarding or penalising the action.
                        self._score_action_keep("kernel_opt", gain_pct=0.0)
                    self.shared_state.save(self.session_dir)
                if kind == "integrate":
                    if result.get("status") != "skipped":
                        self.shared_state.record_kernel_integrate_result(result)
                    decision = str(result.get("decision", "")).upper()
                    int_status = str(result.get("status") or "")
                    int_failed = int_status in {"failed", "error"} or bool(
                        result.get("error_class")
                    )
                    if decision == "KEEP":
                        self._record_integrate_keep(result)
                        gain = result.get("gain_pct")
                        gain_f = (
                            float(gain) if isinstance(gain, (int, float)) else 0.0
                        )
                        self._score_action_keep("integrate", gain_pct=max(0.0, gain_f))
                    elif decision in {"REVERT", "NEEDS_REVIEW"}:
                        if int_failed:
                            self._score_action_failure("integrate")
                        else:
                            self._score_action_no_promote("integrate")
                    elif int_status != "skipped":
                        if int_failed:
                            self._score_action_failure("integrate")
                        else:
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
        """Process ``escalate_strategy_change`` (KB_design §3.8 §7.3 +
        §3.13 M7 §5.3).

        Two responsibilities:

        1. **Always** broadcast a priority-0 ``strategy_change`` event
           so other reactors / breakdown see the LLM's intent (kept
           verbatim from v0.6 for back-compat).
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
        # Always emit the broadcast first (back-compat with v0.6 tests
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
        # the domain (KB_design §3.5 / §3.9). Unknown domain suffixes
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
        streak = self.shared_state.record_policy_denial(
            action_name=resolved_action,
            rule=str(denied.rule or ""),
            hint=str(denied.hint or ""),
            intent_type=intent.type.value,
            tick=int(self.shared_state.tick or 0),
            intent_payload=intent.payload,
        )
        # v0.8 §3.9 — the streak counter is still a fact (LLM sees it
        # via policy_denial_history), but we no longer mirror it onto
        # a scoreboard ``locked_reason``. The phase machine's
        # ``policy_loop`` stop_reason at streak ≥ 10 (below) remains
        # the only system-side reaction.
        if resolved_action and streak >= 5:
            self.shared_state.prune_family(resolved_action)
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind": "auto_prune",
                    "action": resolved_action,
                    "rule": denied.rule,
                    "streak": streak,
                },
            )
        if streak >= 10:
            self.shared_state.set_stop_reason("policy_loop")

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
            # Per-entry incremental gain (current_best vs. baseline at the
            # moment this integrate KEEP landed). Keeps
            # ``gain_per_stack_entry`` index-aligned with
            # ``optimization_stack`` for session_breakdown's
            # capability_summary attribution.
            gain_pct_entry: float | None = None
            try:
                bt = float(self.shared_state.baseline_tput or 0.0)
                if bt > 0:
                    gain_pct_entry = (float(new_tput) - bt) / bt * 100.0
            except (TypeError, ValueError):
                gain_pct_entry = None
            self.shared_state.gain_per_stack_entry.append(gain_pct_entry)

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
        """Record a failed / unpromotable task result into SharedState.

        Previously this was a baseline-only branch. The audit-trail plan
        broadens it to:

        * Always (every task kind, including kernel-owned) append to
          :attr:`SharedState.last_action_failures` via
          :meth:`SharedState.record_action_failure` — that's the rich
          rolling log Orchestration consults after the inbox rotates.
        * For the 6 audit kinds (baseline/profile/backends/params/sweep/
          validate_stack) also append a ``status="failed"`` entry to
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
            # detect "same params failed twice; refuse a third
            # attempt". Only baseline is fingerprinted today
            # (validate_stack uses the same surface but its own audit
            # branch sits in ``_promote_to_shared_state``); other
            # actions would need their own per-action fingerprint key
            # set before this stamp is meaningful.
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
        if any_changed:
            self.shared_state.save(self.session_dir)
        if baseline_event_payload is not None:
            await self.bus.append_and_seq(Message.new(
                "coordinator", "*", "event", baseline_event_payload,
            ))

    async def _pump_dispatcher_once(self) -> None:
        """Dispatch queued tasks, respecting per-lane capacity.

        v0.8 M6 (KB_design §3.7 §4.3) — concurrent dispatch path:

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
        equivalent to v0.6 serial dispatch (one task per conflict
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
        spawned: list[tuple[Task, asyncio.Task[SubAgentResult]]] = []
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
            spawned.append((
                task,
                asyncio.create_task(
                    self.sub.run_task(task, prebound_lease=lease),
                ),
            ))
        if not spawned:
            return
        # Gather; we want to surface per-task results in the order the
        # tasks finished but keep tick semantics simple by awaiting
        # all of them. Exceptions are folded into SubAgentResult
        # (run_task catches inside its body) but defensively absorb
        # anything that leaks here too.
        results = await asyncio.gather(
            *(t for _, t in spawned), return_exceptions=True,
        )
        for (task, _), maybe_result in zip(spawned, results):
            if isinstance(maybe_result, BaseException):
                log.exception(
                    "dispatcher: spawned task %s raised: %r",
                    task.task_id, maybe_result,
                )
                continue
            result: SubAgentResult = maybe_result
            await self.bus.append_and_seq(Message.new(
                "coordinator", "*", "delegated_result",
                {"task_id": task.task_id, "kind": task.kind,
                 "state": result.state, "result": result.result,
                 "error": result.error},
            ))
            # v0.8 §3.5 §10 + KB_gaps/Gap-03 — specialist bookkeeping.
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
            if kept:
                await self._promote_to_shared_state(
                    task.kind, result.result, task=task,
                )
            else:
                await self._handle_unpromotable_result(task, result.result)
            # v0.8 M1 — T3 anchor (KB_design §3.13 M1 §5.3). Always called
            # so KEEP / REVERT each get a corresponding ingest-attempt +
            # verify pair. Best-effort: failures are absorbed into the
            # NDJSON queue by the client.
            await self._cortex_t3_hook(task=task, result=result, kept=kept)
            # v0.8 KB_gaps/Gap-09 — explore-round gap update.
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
    # v0.8 M1 — Cortex KB T3 hook (KEEP / REVERT mirror)
    # ------------------------------------------------------------------
    def _proposal_msg_id_for_task(self, task: "Task") -> str:
        """Recover the original ``proposal_msg_id`` from a task.

        ``_materialize_approved_proposal`` writes the task with
        ``idempotency_key=f"approved-{proposal_msg_id}"`` (DESIGN §18,
        also referenced by the resume path). Tasks created via direct
        delegate (no review) carry a different idempotency_key shape;
        for those we return ``""`` and the T3 hook falls back to
        propose-edge (``late_verified``).
        """
        key = (task.idempotency_key or "").strip()
        if key.startswith("approved-"):
            return key[len("approved-"):]
        return ""

    async def _cortex_t3_hook(
        self,
        *,
        task: "Task",
        result: Any,
        kept: bool,
    ) -> None:
        """T3 dispatcher — KB_design §3.13 M5 §5 step 7 / KB_gaps/Gap-08.

        When the task is an ``explore`` action that returned a non-empty
        ``per_variant_outcomes`` list, fan out to the per-variant path:
        one ``ingest_attempt`` + ``verify`` per variant, keyed by the
        per-variant ``kb_edge_id`` minted in T2 (Gap-07).

        Everything else (kernel_opt, integrate, baseline, profile,
        backends, params, sweep, ...) keeps the legacy per-task path:
        a single attempt + single verify against the representative
        edge_id (still recorded by T2 for back-compat).
        """
        if self.cortex_kb is None or not self.cortex_kb.enabled:
            return
        sid = (self.shared_state.cortex_session_id or "").strip()
        if not sid:
            return
        result_dict = result.result if hasattr(result, "result") else (result or {})
        if not isinstance(result_dict, dict):
            result_dict = {}
        per_variant = result_dict.get("per_variant_outcomes")
        if (
            task.kind == "explore"
            and isinstance(per_variant, list)
            and per_variant
        ):
            await self._cortex_t3_per_variant(
                task=task, sid=sid, outcomes=per_variant,
            )
        else:
            await self._cortex_t3_per_task(
                task=task, sid=sid, result_dict=result_dict, kept=kept,
            )
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001 — defensive; T3 must not crash.
            log.exception("cortex T3 SharedState.save failed")

    async def _cortex_t3_per_task(
        self,
        *,
        task: "Task",
        sid: str,
        result_dict: dict[str, Any],
        kept: bool,
    ) -> None:
        """Legacy per-task T3 — one attempt + one verify per task.

        Used by all non-explore actions (kernel_opt / integrate /
        baseline / profile / backends / params / sweep / ...).
        """
        # Mint per-task attempt_node for cross-session reachability.
        attempt_canonical = attempt_canonical_id(sid, task.task_id)
        outcome_label = "PASS" if kept else "FAIL"
        if not kept:
            status = str(result_dict.get("status", "")).lower()
            if status in ("partial", "needs_review"):
                outcome_label = "PARTIAL"
        metrics: dict[str, Any] = {}
        for key in (
            "output_throughput", "gain_pct", "validated_gain_pct",
            "accuracy", "ttft_mean_ms", "e2el_mean_ms", "decision",
            "variant_name", "error_class",
        ):
            if key in result_dict and result_dict[key] is not None:
                metrics[key] = result_dict[key]
        metrics.setdefault("task_kind", task.kind)
        metrics.setdefault("task_id",   task.task_id)
        try:
            self.cortex_kb.propose_point(
                canonical_id=attempt_canonical,
                kind="attempt_node",
                authority="EXPERIENTIAL",
                attrs={
                    "task_kind": task.kind,
                    "task_id":   task.task_id,
                    "outcome":   outcome_label,
                },
                evidence=[f"log:task-{task.task_id}"],
            )
        except CortexKBError as exc:
            log.warning("cortex T3 propose_point attempt failed: %s", exc)
        proposal_msg_id = self._proposal_msg_id_for_task(task)
        edge_entry = (
            self._pop_pending_kb_edge(proposal_msg_id) if proposal_msg_id else None
        )
        edge_id = (edge_entry or {}).get("edge_id", "") if edge_entry else ""
        plan_edge = edge_id or ""
        try:
            self.cortex_kb.ingest_attempt(
                sid=sid,
                iter_id=int(self.shared_state.tick or 0),
                outcome=outcome_label,
                metrics=metrics,
                plan_edge=plan_edge,
                evidence=[
                    f"log:task-{task.task_id}",
                    f"point_id:{attempt_canonical}",
                ],
            )
        except CortexKBError as exc:
            log.warning("cortex T3 ingest_attempt failed: %s", exc)
        if edge_entry and edge_id:
            verify_outcome = "confirmed" if kept else "refuted"
            promote_authority = "EXPERIENTIAL" if kept else None
            try:
                self.cortex_kb.verify(
                    sid=sid,
                    edge_id=edge_id,
                    outcome=verify_outcome,
                    evidence=[f"log:task-{task.task_id}"],
                    promote_authority=promote_authority,
                )
            except CortexKBError as exc:
                log.warning("cortex T3 verify failed: %s", exc)
        elif edge_entry and not edge_id:
            # T2 fell through to NDJSON without a sync edge id. Fall back
            # to a late propose-edge: signal the verdict via attempt
            # outcome only; the flusher will eventually replay the
            # hypothesize NDJSON row and Cortex will dedup by canonical_id.
            log.info(
                "cortex T3 late_verified (no edge_id for proposal %s)",
                proposal_msg_id or "(no msg_id)",
            )

    async def _cortex_t3_per_variant(
        self,
        *,
        task: "Task",
        sid: str,
        outcomes: list[dict[str, Any]],
    ) -> None:
        """Per-variant T3 — one attempt + one verify per variant.

        KB_design §3.13 M5 §5 step 7 / KB_gaps/Gap-08. Iterates
        ``per_variant_outcomes`` from the explore executor result and
        binds each entry to the matching edge_id minted by T2 (Gap-07,
        stored under ``pending_kb_edges[].variant_edges``).

        Per-variant outcome → KB encoding:

        * ``KEEP``           → attempt PASS + verify ``confirmed``
                               (promote_authority=EXPERIENTIAL).
        * ``REVERT`` /
          ``FAILED`` /
          ``KEEP_UNSTABLE``  → attempt FAIL + verify ``refuted``
                               (no promotion).
        * ``SKIPPED_DEDUP``  → no KB activity (no edge was minted).

        Partial failures: a single variant's ``verify`` or
        ``ingest_attempt`` exception is logged but does not abort the
        remaining variants. The pending_kb_edges row is popped once
        up front so resume + idempotency stay clean.
        """
        proposal_msg_id = self._proposal_msg_id_for_task(task)
        edge_entry = (
            self._pop_pending_kb_edge(proposal_msg_id) if proposal_msg_id else None
        )
        variant_edges_map: dict[str, str] = {}
        if isinstance(edge_entry, dict):
            raw = edge_entry.get("variant_edges") or {}
            if isinstance(raw, dict):
                variant_edges_map = {
                    str(k): str(v or "") for k, v in raw.items()
                }
        if not edge_entry and not variant_edges_map:
            # T2 hook never ran (e.g. --no-cortex during materialize) or
            # the row was already popped on a resume — fall back to the
            # per-variant ``kb_edge_id`` stamped on the executor's
            # result, which mirrors the same map.
            for vo in outcomes:
                name = str(vo.get("variant_name") or "")
                edge_id = str(vo.get("kb_edge_id") or "")
                if name and edge_id:
                    variant_edges_map[name] = edge_id
        if not variant_edges_map and isinstance(edge_entry, dict):
            # Last-resort: only a single edge_id was recorded (e.g. the
            # proposal pre-dates Gap-07). Map every KEEP/REVERT variant
            # to it so we at least confirm/refute *something* — matches
            # the per-task fallback behaviour.
            single = str(edge_entry.get("edge_id") or "")
            if single:
                for vo in outcomes:
                    name = str(vo.get("variant_name") or "")
                    if name and vo.get("outcome") in (
                        "KEEP", "REVERT", "FAILED", "KEEP_UNSTABLE",
                    ):
                        variant_edges_map[name] = single

        any_terminal = False  # at least one KEEP/REVERT processed
        for vo in outcomes:
            variant_name = str(vo.get("variant_name") or "")
            outcome = str(vo.get("outcome") or "")
            if outcome == "SKIPPED_DEDUP":
                continue
            if not variant_name:
                # Nameless variants slip through if the executor mutates
                # the list — skip rather than mint a phantom attempt.
                continue
            any_terminal = True
            if outcome == "KEEP":
                attempt_outcome = "PASS"
                verify_outcome = "confirmed"
                promote_authority: str | None = "EXPERIENTIAL"
            else:  # REVERT / FAILED / KEEP_UNSTABLE
                attempt_outcome = "FAIL"
                verify_outcome = "refuted"
                promote_authority = None

            # 1. Per-variant attempt_node — canonical id encodes the
            #    variant name so cross-session diffs stay precise.
            variant_attempt_canonical = (
                f"{attempt_canonical_id(sid, task.task_id)}.variant-{variant_name}"
            )
            attempt_attrs = {
                "task_kind":    task.kind,
                "task_id":      task.task_id,
                "variant_name": variant_name,
                "outcome":      attempt_outcome,
            }
            if vo.get("provenance"):
                attempt_attrs["provenance"] = str(vo.get("provenance"))
            try:
                self.cortex_kb.propose_point(
                    canonical_id=variant_attempt_canonical,
                    kind="attempt_node",
                    authority="EXPERIENTIAL",
                    attrs=attempt_attrs,
                    evidence=[
                        f"log:task-{task.task_id}",
                        f"variant:{variant_name}",
                    ],
                )
            except CortexKBError as exc:
                log.warning(
                    "cortex T3 propose_point attempt failed for variant=%s: %s",
                    variant_name, exc,
                )

            # 2. ingest_attempt — carries per-variant metrics + plan_edge.
            edge_id = variant_edges_map.get(variant_name, "")
            metrics: dict[str, Any] = {
                "task_kind":    task.kind,
                "task_id":      task.task_id,
                "variant_name": variant_name,
            }
            raw_metrics = vo.get("metrics") or {}
            if isinstance(raw_metrics, dict):
                for mk, mv in raw_metrics.items():
                    if mv is not None:
                        metrics[mk] = mv
            if vo.get("reason"):
                metrics["reason"] = str(vo.get("reason"))
            try:
                self.cortex_kb.ingest_attempt(
                    sid=sid,
                    iter_id=int(self.shared_state.tick or 0),
                    outcome=attempt_outcome,
                    metrics=metrics,
                    plan_edge=edge_id,
                    evidence=[
                        f"log:task-{task.task_id}",
                        f"variant:{variant_name}",
                        f"point_id:{variant_attempt_canonical}",
                    ],
                    idempotency_key=(
                        f"ingest_attempt:{sid}:{task.task_id}:{variant_name}"
                    ),
                )
            except CortexKBError as exc:
                log.warning(
                    "cortex T3 ingest_attempt failed for variant=%s: %s",
                    variant_name, exc,
                )

            # 3. verify — only when we have a real edge_id; otherwise
            #    fall through (T2 NDJSON replay covers it eventually).
            if edge_id:
                try:
                    self.cortex_kb.verify(
                        sid=sid,
                        edge_id=edge_id,
                        outcome=verify_outcome,
                        evidence=[
                            f"log:task-{task.task_id}",
                            f"variant:{variant_name}",
                        ],
                        promote_authority=promote_authority,
                        idempotency_key=(
                            f"verify:{sid}:{edge_id}:{variant_name}"
                        ),
                    )
                except CortexKBError as exc:
                    log.warning(
                        "cortex T3 verify failed for variant=%s: %s",
                        variant_name, exc,
                    )
            else:
                log.info(
                    "cortex T3 late_verified (no edge_id for variant=%s"
                    " proposal=%s)",
                    variant_name, proposal_msg_id or "(no msg_id)",
                )

        if not any_terminal and edge_entry:
            log.info(
                "cortex T3 per-variant: all variants skipped for proposal=%s"
                " (round was 100%% dedup); pending edge row dropped",
                proposal_msg_id or "(no msg_id)",
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
                # Mirror append into ``gain_per_stack_entry`` so indexes
                # stay aligned across the two parallel lists. See the
                # SharedState docstring for the contract.
                gain_pct_entry: float | None = None
                try:
                    bt = float(self.shared_state.baseline_tput or 0.0)
                    if bt > 0:
                        gain_pct_entry = (float(best_tput) - bt) / bt * 100.0
                except (TypeError, ValueError):
                    gain_pct_entry = None
                self.shared_state.gain_per_stack_entry.append(gain_pct_entry)

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
        # Audit-trail bookkeeping for the 6 non-kernel actions. Each
        # in-scope branch sets ``audit_decision`` (and optionally
        # ``audit_extras``); after all branches we call
        # ``record_action_attempt`` once so adding a new branch is a
        # local change. ``audit_decision`` remaining ``None`` means
        # either the kind is out of scope (kernel-owned, pmc_roofline)
        # or the branch had nothing to record.
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
            self.shared_state.current_best = {
                "action": "baseline",
                "tput": float(tput) if isinstance(tput, (int, float)) else None,
                "ttft_mean_ms": result.get("ttft_mean_ms"),
                "e2el_mean_ms": result.get("e2el_mean_ms"),
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
            # v0.8 KB_gaps/Gap-09 — seed the gaps[] ledger from baseline.
            # Best-effort; failure is logged + absorbed inside the helper.
            await self._refresh_gaps(reason="baseline_done")
        elif task_kind == "profile":
            audit_decision = "promoted"
            audit_extras = {
                "trace_path": None,
                "profile_args": None,
                "pmc_summary_path": result.get("pmc_summary_path"),
                "roofline_path": result.get("roofline_path"),
                "kernel_breakdown_path": result.get("kernel_breakdown_path"),
                "output_throughput": result.get("output_throughput"),
            }
            # Bug C fix: surface the trace path produced by ProfileExecutor
            # to SharedState so Orch can pass a real path to the kernel
            # `select_kernels` REQUEST instead of fabricating one.
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
                # Stale select_kernels cache no longer matches this trace.
                self.shared_state.last_select_kernels = {}
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
                # Validate-stack is a *measurement*, not a decision
                # gate: even when the re-bench lands at or below the
                # baseline we still record the number so the timeline /
                # state.json reflects ground truth. The warning below
                # flags the regression so an operator (or a future
                # rollback policy) can react, but we deliberately leave
                # ``optimization_stack`` / ``current_best`` /
                # ``cumulative_gain`` untouched here.
                VALIDATE_STACK_WARN_THRESHOLD_PCT = 0.0
                if gain <= VALIDATE_STACK_WARN_THRESHOLD_PCT:
                    log.warning(
                        "validate_stack: cumulative_gain_validated=%.2f%% <= %.1f%% "
                        "(tput=%.2f vs baseline=%.2f, stack_len=%d). Recording the "
                        "measurement but NOT rolling back optimization_stack — "
                        "validate_stack remains a measurement, not a decision gate.",
                        gain, VALIDATE_STACK_WARN_THRESHOLD_PCT, float(tput),
                        self.shared_state.baseline_tput,
                        len(self.shared_state.optimization_stack),
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
                audit_decision = "promoted" if gain > 0 else "discarded"
                audit_extras = {
                    "validated_stack_len": (
                        self.shared_state.cumulative_gain_validated_stack_len
                    ),
                    "gain_pct": float(gain),
                    "baseline_tput_ref": float(self.shared_state.baseline_tput),
                    "validated_tput": float(tput),
                }
            else:
                # Record the failed-to-measure case as a discard so the
                # audit trail still captures *why* the validation didn't
                # produce a number (NaN tput, baseline_tput == 0, etc.).
                audit_decision = "discarded"
                audit_extras = {
                    "validated_stack_len": stack_len_at_run,
                    "gain_pct": None,
                    "baseline_tput_ref": float(self.shared_state.baseline_tput),
                    "validated_tput": (
                        float(tput) if isinstance(tput, (int, float)) else None
                    ),
                }
        elif task_kind in ("backends", "params", "sweep"):
            # backends/params content-fingerprint ledgers (Phase 4 of the
            # dedup-by-fingerprint plan). Persist BEFORE the promotion
            # logic below so a winner appended to ``accepted`` further
            # down sees the latest ``tested`` / ``rejected`` already in
            # place. ``apply_*_search_update`` preserves the existing
            # ``accepted`` list (the executor never writes it) so the
            # Coordinator remains the sole writer for promotion history.
            if task_kind == "backends" and isinstance(
                result.get("backends_search_update"), dict
            ):
                self.shared_state.apply_backends_search_update(
                    result["backends_search_update"],
                )
                changed = True
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
                err = disc_update.get("discovery_error")
                if err:
                    self.shared_state.discovered_flags_error = str(err)
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
                # Audit trail (sweep is in _AUDIT_ACTIONS even though it
                # never promotes a current_best — Orchestration still
                # benefits from seeing what the sweep grid produced).
                # Recorded FIRST so the subsequent ``record_sweep`` call
                # gets the final word on ``last_sweep`` (which is a
                # richer grid-summary payload than the uniform audit
                # entry; see ``_format_last_sweep`` and
                # test_p2_5_grid_promotion). ``sweep_attempts`` still
                # captures the audit row.
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
            # current best by a meaningful margin. We use 0.2% as the
            # 1-shot KEEP threshold — relaxed from marathon's original
            # 1.0% per the resume5 9h finding (35/38 winners landed in
            # the 0.3–0.84% band but never promoted because each
            # individual run sat under 1.0%), but kept above the
            # session-to-session noise floor (~0.1%) so a single noisy
            # round doesn't lock in a non-improvement. AND, as a
            # separate path, promote ANY consistent winner that wins
            # ≥ 2 of last 3 rounds with average gain ≥ 0.1% — that's
            # the cross-round signal-vs-noise check; rounds individually
            # under 0.2% still stack up when the same variant keeps
            # winning, so we don't lose real but small gains.
            PROMOTE_THRESHOLD_PCT = 0.2
            CROSS_ROUND_LOOKBACK = 3
            CROSS_ROUND_MIN_APPEARANCES = 2
            # The cross-round bar must stay strictly under
            # PROMOTE_THRESHOLD_PCT, otherwise the path is mathematically
            # unreachable (any 2 sub-threshold rounds whose average
            # crosses the cross-round bar would also have at least one
            # round above the 1-shot bar, triggering single-shot promote
            # first). 0.1% gives us a real cross-round signal between
            # the noise floor and the 1-shot bar.
            CROSS_ROUND_MIN_AVG_GAIN_PCT = 0.1
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
                    extra_sglang_args=str(
                        bv.get("candidate_extra_sglang_args")
                        or bv.get("extra_sglang_args") or ""
                    ),
                    extra_envs=dict(bv.get("extra_envs") or {}),
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
                # Phase 4 of the dedup-by-fingerprint plan: Coordinator
                # is the sole writer to ``backends_search.accepted``. On
                # a backends promote, stamp the winner so the next round
                # filter treats this variant as accepted, not rejected.
                if task_kind == "backends" and isinstance(bv, dict) and bv:
                    promote_entry = dict(bv)
                    promote_entry.setdefault(
                        "gain_pct",
                        float(gain_vs_cb) if isinstance(
                            gain_vs_cb, (int, float)
                        ) else None,
                    )
                    self.shared_state.record_backends_accepted(promote_entry)
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
            # Audit-trail for backends / params. Sweep already recorded
            # above (it short-circuits with ``return``).
            audit_decision = "promoted" if promoted else "discarded"
            audit_extras = {
                "round_id": (
                    self.shared_state.backend_winners_history[-1].get("round_id")
                    if self.shared_state.backend_winners_history
                    and isinstance(
                        self.shared_state.backend_winners_history[-1], dict,
                    )
                    else None
                ),
                "best_variant_name": (
                    bv.get("name") if isinstance(bv, dict) else None
                ),
                "candidate_extra_sglang_args": (
                    bv.get("extra_sglang_args")
                    if isinstance(bv, dict) else None
                ),
                "extra_envs": (
                    dict(bv.get("extra_envs") or {})
                    if isinstance(bv, dict) else {}
                ),
                "best_gain_pct_vs_base": best_gain,
                "gain_vs_cb": (
                    float(gain_vs_cb)
                    if isinstance(gain_vs_cb, (int, float)) else None
                ),
            }
        # Out-of-band score updates for the task_kinds that don't have a
        # promoted-vs-discard notion (profile / pmc_roofline / validate_stack
        # bump runs + cooldown; baseline is treated as a gate and skipped
        # inside the helper). Sweep + backends/params/kernel branches above
        # already called the helper themselves, so we filter to the
        # measurement-style kinds here.
        if task_kind in {"profile", "pmc_roofline", "validate_stack"}:
            self._apply_action_score_update(task_kind, result)
            changed = True
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
