"""Coordinator main loop.

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

Everything else (real backends, accuracy gate, phase machine, checkpoint
cadence) lands in P0-5 and beyond.
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
    experiment_canonical_id,
)
from . import phase_state as _phase_state
from ..paths import db_path_for, make_session_dir
from ..storage.connection import SqliteConnection
from .action_registry import ActionRegistry
from .agent_role import AgentRole, default_role_registry
from .backends.base import Backend, BackendError, BackendTurnResult
from .cursor_store import CursorStore
from .intent_parser import Intent, IntentType, NoIntentEmitted
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
    "baseline", "profile", "sweep", "explore",
    # F1-3 + N10: see shared_state._AUDIT_ACTIONS for the rationale.
    # The composite roofline action runs profile + trace_analyze
    # atomically; each invocation is visible in `roofline_attempts`.
    "roofline",
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

# IR-7 — closed enum of session_steward_specialist recommendations.
# Any value outside this set is coerced to ``stop_session`` in
# :meth:`Coordinator._route_steward_verdict` (defense in depth — the
# LLM can write any string but only the enum drives a phase-routing
# change).
_STEWARD_RECS: frozenset[str] = frozenset({
    "continue_explore", "advance_to_kernel", "stop_session",
})

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


def _resolve_silent_ticks_closing_threshold() -> int:
    """N33: how many consecutive idle ticks must elapse before the
    Coordinator force-enters closing phase.

    "Idle" = ``shared_state.consecutive_silent_ticks`` was bumped
    because the tick had no queued tasks, no running tasks, no pending
    proposals and no ``current_action``. Default 120 ticks; with the
    prod ``tick_interval_sec=5.0`` that is ~10 minutes of total LLM
    silence before we short-circuit. Override via the env knob; ``0``
    disables the early-close (legacy behaviour: idle until the wall-
    clock deadline). Negative / non-numeric values fall back to the
    default.
    """
    raw = os.environ.get(
        "INFERENCE_OPTIMIZER_IDLE_CLOSE_TICKS", "",
    ).strip()
    if not raw:
        return 120
    try:
        v = int(raw)
    except ValueError:
        return 120
    return v if v >= 0 else 120


def _parse_iso_unix(ts: str) -> float:
    """Parse an ISO 8601 UTC timestamp into unix seconds.

    Returns ``0.0`` on any parse failure so callers can treat a missing
    timestamp as "no information"; never raises. Used by the
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


def _summarize_failed_variants(
    all_results: Any, *, max_entries: int = 10,
) -> list[dict[str, Any]]:
    """Project the ``status=='failed'`` rows of a grid_runner result list.

    Returns a compact ``[{name, error_class, error_excerpt,
    extra_sglang_args}, ...]`` so the audit-trail extras can carry
    per-variant failure context without ballooning the prompt context.

    Why this exists: explore / sweep executors run a multi-variant grid
    via ``run_grid`` and reduce it to one ``record_action_attempt`` entry
    (1 task = 1 attempt, by Coordinator design). Before this helper the
    only place a failed variant landed was the ``all_results`` blob
    inside the raw delegated_result; the LLM prompt assembled from
    SharedState therefore could not see prior silent aborts and might
    keep re-proposing the same variant on the next round.

    Truncation: at most ``max_entries`` failed rows so a runaway grid
    can't bloat attempts_history. ``error_excerpt`` is capped at 400
    chars (smaller than the per-entry 2000-char cap used by
    ``_write_variant_abort_marker`` because this lives inside a
    promptable audit trail, not on-disk forensics).
    """
    if not isinstance(all_results, list):
        return []
    failed: list[dict[str, Any]] = []
    for row in all_results:
        if not isinstance(row, dict):
            continue
        if str(row.get("status") or "") != "failed":
            continue
        err = str(row.get("error") or "")
        failed.append({
            "name": str(row.get("name") or ""),
            "error_class": str(row.get("error_class") or "") or None,
            "error_excerpt": err[:400] if err else None,
            "extra_sglang_args": str(row.get("extra_sglang_args") or ""),
        })
        if len(failed) >= max_entries:
            break
    return failed


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


def _dedupe_extra_sglang_args(args_str: str) -> str:
    """Collapse repeated ``--flag value`` pairs into a unique launch string.

    SGLang / vLLM argparse uses ``action="store"`` for almost every
    knob, so if ``--cuda-graph-max-bs 32 --cuda-graph-max-bs 128
    --cuda-graph-max-bs 256`` end up on the same command line, only the
    last value is honored. The original cmdline still works, but
    ``final.extra_sglang_args`` exists for dashboard / replay use and
    looking at it with the same flag repeated 3-15 times is misleading
    (it reads as "this run actually used N values" when really only the
    last won).

    Promote / validate_stack rounds previously fed each ``candidate_args``
    block into ``previous + candidate`` concatenation; when the next
    round's candidate kept the same multi-value combo, the whole block
    was re-appended verbatim. Dedupe is a sane normalization: keep each
    flag once, with the value of its last occurrence — same semantics
    argparse would have applied at launch.

    Bare flags (no value, e.g. ``--enable-prefix-caching``) are also
    deduped (kept once, position preserved by first appearance).

    Args:
        args_str: space-separated ``--flag value`` pairs.

    Returns:
        Deduped equivalent. Empty input → empty output.
    """
    if not args_str:
        return ""
    tokens = args_str.split()
    # Track each flag's last-seen pair, plus its first-seen position so
    # we emit in stable order (last-wins-for-value, first-wins-for-order).
    pair_by_flag: dict[str, list[str]] = {}
    order: list[str] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("--"):
            flag = t
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                pair = [flag, tokens[i + 1]]
                i += 2
            else:
                pair = [flag]
                i += 1
            if flag not in pair_by_flag:
                order.append(flag)
            pair_by_flag[flag] = pair
        else:
            # Stray positional token; preserve as-is, in order. Use a
            # synthetic key so it doesn't collide with anything.
            key = f"__positional_{len(order)}__"
            order.append(key)
            pair_by_flag[key] = [t]
            i += 1
    out: list[str] = []
    for k in order:
        out.extend(pair_by_flag[k])
    return " ".join(out)


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
    # tentative_edge_id returned by Cortex T2
    # ``session hypothesize``.
    # Empty when T2 failed sync and went to NDJSON; T3 then falls back
    # to ``propose-edge + late_verified`` rather than ``verify``.
    #
    # Back-compat surface: when the proposal is a multi-variant
    # ``explore`` grid, this field carries the
    # *representative* edge_id (first variant) so the legacy T3 hook
    # — which still verifies one edge per proposal — keeps working.
    # The full per-variant map lives in :attr:`kb_edge_ids` below;
    # Gap-08 will extend T3 to iterate that map.
    kb_edge_id: str = ""
    # The experiment canonical_id minted at T2
    # (``exp:{sid}:{session_iter_index:04d}``). Stored on the proposal
    # so the T3 verify path can still emit a ``propose-edge`` even when
    # the sync hypothesize failed.
    kb_opt_canonical: str = ""
    # Monotonic experiment iter index assigned at T2 (mirrors
    # ``SharedState.session_iter_index``). Per-variant grids reuse this
    # parent iter and append ``.variant-{name}`` to the canonical_id.
    experiment_iter_index: int = 0
    # per-variant edge_ids minted
    # by ``_cortex_t2_hook`` when the proposal is an ``explore``
    # action with a non-empty ``params.grid``. Keyed by variant
    # name; empty dict for non-grid proposals (kernel_opt / integrate
    # / etc.). The explore executor reads each variant's
    # ``kb_edge_id`` via :meth:`_materialize_approved_proposal`
    # stamping, so cross-session KB queries can locate the exact
    # variant that confirmed / refuted a hypothesis (instead of the
    # M3 per-proposal aggregate).
    kb_edge_ids: dict[str, str] = field(default_factory=dict)
    # per-variant opt_canonical
    # ids parallel to :attr:`kb_edge_ids`. Stored separately so the
    # T3 hook can rebuild ``propose-edge`` fallbacks per variant
    # (Gap-08) even when the synchronous T2 hypothesize failed.
    kb_opt_canonicals: dict[str, str] = field(default_factory=dict)
    # per-variant verdicts
    # surfaced by the Critic agent's batch review (``verdict_map``).
    # Keyed by variant_name, value carries ``{verdict, rationale}``.
    # Empty for single-verdict (kernel_opt / integrate / report /
    # ...) proposals; the Coordinator's ``_handle_verdict_map`` writer
    # filters the grid down to the ``approve`` subset before
    # materializing the explore task and fires KB ``refuted`` for
    # every reject (so the Cortex view captures the
    # critic-rejected edge in addition to the KEEP/REVERT
    # signal the explore executor produces).
    verdict_map: dict[str, dict[str, Any]] = field(default_factory=dict)


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
        # Cortex KB client.  When
        # ``None`` (legacy cli path or ``--degraded-kb``) all T2/T3/T4
        # hooks become no-ops; the rest of the Coordinator behaves
        # identically to v0.6.  The client itself is stateless apart
        # from the per-session NDJSON queue, so it can be shared across
        # threads (the Coordinator is single-event-loop anyway).
        self.cortex_kb: CortexKBClient | None = cortex_kb
        # KnowledgePlane facade.
        # When non-None, ``_handle_delegate`` pre-warms ``pr_feed`` +
        # ``kb_subgraph`` for ``delegate{action='specialist'}`` tasks
        # before enqueue, so the SpecialistRunner prompt assembly sees
        # the latest knowledge.  ``None`` keeps the legacy code path
        # (no warmup; specialist still runs but sees empty knowledge
        # surface).
        self.knowledge_plane: Any = knowledge_plane
        # phase budget percentages (KB_design §3.8 §5.3 +
        # §3.13 M2 §7). ``None`` means library defaults; CLI flags
        # populate this dict from ``--max-minutes-<phase>-pct``. We
        # normalise once at construction so downstream judges can
        # rely on a complete dict.
        self._phase_budget_pct: dict[str, float] = _phase_state.normalize_budget_pct(
            phase_budget_pct
        )
        # specialist stale scan threshold (seconds).
        # M5 wires real specialist sub-agents; M2 only ships the
        # scanner so the Robustness prompt block lights up the moment
        # M5 lands. Env override mirrors the rest of the knobs.
        try:
            self._specialist_stale_sec: float = max(
                0.0,
                float(os.environ.get(
                    "INFERENCE_OPTIMIZER_SPECIALIST_STALE_SEC", "600",
                )),
            )
        except ValueError:
            self._specialist_stale_sec = 600.0
        # opt-out switch for
        # the legacy ``params_no_promote_streak`` plateau proxy.
        # When set, ``compute_next_phase`` skips the m2_proxy branch
        # entirely; legacy resume sessions without signals fall
        # through to the wall-clock budget exhaustion exit.
        self._legacy_plateau_proxy_disabled: bool = (
            os.environ.get(
                "INFERENCE_OPTIMIZER_DISABLE_PLATEAU_PROXY", "",
            ).strip().lower() in ("1", "true", "yes")
        )
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

        self._resumed_from = self._detect_resume_state()
        # initialise phase machine. Fresh session enters
        # PRELUDE; resume from v0.6 (no phase field) infers a phase via
        # :func:`phase_state.infer_phase_from_state`.  Always idempotent:
        # second construction on the same session_dir is a no-op.
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
            # ``_handle_verdict_map`` always
            # mirrors a *summary* ``verdict`` alongside the per-
            # variant ``verdict_map`` so this rebuild stays
            # backward-compatible. A verdict_map-only event with no
            # summary string still marks the proposal as decided
            # (the Critic spoke on every variant, no further round
            # needed) — we synthesise a ``needs_review`` placeholder
            # so the prompt's /status surface still shows something.
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
        # T4 anchor. Drains the
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

        when the CLOSE phase
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
        #    upper bound.
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
        # Fresh OR legacy resume.
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
        * Skip when ``cortex_kb.enabled is False`` (``--degraded-kb``).
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
        if client is None or not getattr(client, "enabled", False):
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
        extra_attrs = {
            "framework":   getattr(state, "framework", "") or "",
            "model_class": getattr(state, "model_class", "") or "",
            "claw_session_id": getattr(state, "claw_session_id", "") or "",
            "boot_origin": "coordinator_fallback",
        }
        try:
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
            disable_legacy_proxy=self._legacy_plateau_proxy_disabled,
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
            max_hours=max_hours_arg,
        )
        if next_phase is None:
            # IR-7 — on plateau but no steward verdict yet,
            # compute_next_phase returns None and we enqueue the
            # steward here. The dispatcher's normal loop picks it up;
            # the next tick will see the verdict and route accordingly.
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
        # phase-entry side effects (KB_gaps/Gap-02 PR 5.4 +
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
        """
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
                    # Stamp a summary row so the give-up decision shows
                    # up in phase_history alongside the per-attempt
                    # ``framework_pr_discover_failed`` rows — without
                    # it the final flip to ``phase_done=True`` is
                    # silent and operators have to infer the reason
                    # from the retry trail. PR-327 P2.b follow-up.
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
        # P1.b: Critic gate before apply. The Critic sees the PR
        # metadata (diff URL + title + gap target) and returns an
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

    def _record_framework_pr_phase_done(
        self, *, reason: str, failure_count: int,
    ) -> None:
        """Append a single ``framework_pr_phase_done`` row to
        ``phase_history`` describing why the pump gave up.

        Per-attempt ``framework_pr_discover_failed`` rows already cover
        each individual error; this is the summary row so the give-up
        decision is not silent (PR-327 P2.b follow-up).
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
        gaps = []
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
        if not gaps:
            gaps = [{"gap_canonical_id": "", "gap_description": ""}]
        framework = ""
        try:
            framework = str(getattr(state, "framework", "") or "").strip().lower()
        except Exception:  # noqa: BLE001
            framework = ""
        timeout_sec = float(
            getattr(self, "framework_pr_discover_timeout_sec", 0.0)
            or _fa_client.DEFAULT_FA_PHASE_TIMEOUT_SEC
        )
        try:
            payload = await _fa_client.phase_discover(
                model=str(getattr(state, "model", "") or ""),
                framework=framework or "sglang",
                gpu_type=str(getattr(state, "gpu_type", "") or ""),
                gaps=gaps,
                session_dir=self.session_dir,
                timeout_sec=timeout_sec,
            )
        except Exception as exc:  # noqa: BLE001 — defensive
            failures = int(getattr(state, "framework_pr_discover_failures", 0) or 0) + 1
            state.framework_pr_discover_failures = failures
            log.warning(
                "fa phase-discover failed (attempt %d/%d): %r",
                failures, _fa_client.DISCOVER_FAILURE_RETRY_LIMIT, exc,
            )
            try:
                history = getattr(state, "phase_history", None)
                if isinstance(history, list):
                    history.append({
                        "event":   "framework_pr_discover_failed",
                        "attempt": failures,
                        "limit":   _fa_client.DISCOVER_FAILURE_RETRY_LIMIT,
                        "error":   repr(exc),
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
        candidates = (payload or {}).get("candidates") or []
        if not isinstance(candidates, list) or not candidates:
            return False
        batch_id = str((payload or {}).get("batch_id") or "")
        # Normalise each candidate so the executor's slug helper has
        # consistent fields and the progress ledger has a stable id.
        norm: list[dict[str, Any]] = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            cand_id = str(
                c.get("pr_url") or c.get("ref")
                or f"{c.get('repo','')}-{c.get('pr_number','')}"
            )
            norm.append({
                **c,
                "candidate_id": cand_id,
                "batch_id": batch_id,
            })
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
        """No-op KERNEL entry hook.

        Roofline is auto-enqueued at PRELUDE (initial) and on every
        10% watermark crossing of ``last_roofline_tput`` — see
        :meth:`_maybe_enqueue_watermark_roofline`. The KERNEL phase
        no longer needs an entry-time profile anchor: the watermark
        refresh keeps ``analysis.md`` aligned with stack progress,
        and ``select_kernels`` / ``kernel_opt`` read
        ``last_profile_trace`` written by the same roofline executor.
        """
        if not self._kernel_enabled():
            # Should not happen — compute_next_phase routes
            # --no-kernel runs straight EXPLORE → SWEEP.
            log.info(
                "KERNEL entry hook fired with kernel_enabled=False "
                "(from=%s)", from_phase or "<unknown>",
            )

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
        so the PRELUDE initial roofline enqueue (driven by the
        baseline-completion hook) is the sole entry point before the
        first roofline lands.

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
        if last_rl <= 0:
            return False
        if (state.auto_roofline_pending_task_id or "").strip():
            return False
        cur = self._current_tput_from_validated_gain()
        if cur <= 0:
            return False
        return cur / last_rl >= self._ROOFLINE_WATERMARK_RATIO

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
            cb_args = str(cb.get("extra_sglang_args") or "")
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
        task, was_existing = await self.tasks.create_or_return_existing(
            kind=kind,
            params=params,
            idempotency_key=f"internal-analysis-{reason}",
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
    # v0.8 §3.2 §5.4 + SWEEP phase auto-dispatch
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
    # v0.8 §3.2 §5.5 + CLOSE phase 5-step sequencer
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
        """CLOSE phase 5-step sequencer.

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
        sid.

        The sequencer runs INLINE inside the hook — it doesn't wait
        for the reactor / dispatcher tick boundary. Steps 1 and 2
        enqueue tasks then poll ``_wait_for_task_terminal`` until the
        dispatcher (which the same Coordinator.run() loop drives)
        picks them up and finishes. Step 3 + 4 call the cortex_kb
        client synchronously. Step 5 is a single SharedState write.
        """
        log.info("CLOSE entered (from=%s); starting 4-step close sequence",
                 from_phase or "<unknown>")
        await self._record_close_step("sequencer_started", status="running")

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
        # ``STOP_REASON_VOCAB``. The ``not stop_reason`` guard
        # preserves Step 4's ``cortex_commit_failed`` setter.
        if not self.shared_state.stop_reason:
            self.shared_state.set_stop_reason("time_exhausted")
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

    async def _enqueue_internal_steward_task(
        self, *, reason: str,
    ) -> "Task | None":
        """IR-7 — enqueue a Coordinator-owned session_steward_specialist task.

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
        # Avoid re-enqueueing if a steward verdict already landed in
        # this round (paranoia — wants_steward_assessment should have
        # returned False, but defense-in-depth).
        last = self.shared_state.last_remaining_gaps_assessment or {}
        if isinstance(last, dict) and last.get(
            "round_at_assessment"
        ) == round_id and last.get("recommendation"):
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
            # Replay rebuilt ``_proposals_awaiting_roofline`` from
            # ``proposal_materialize_blocked`` observations. If the
            # analysis task already completed during shutdown, the
            # normal drain hook in ``_promote_to_shared_state`` will
            # not fire on restart, so kick the drain explicitly. It
            # re-checks the roofline gate per proposal and re-queues
            # any that are still blocked.
            if self._proposals_awaiting_roofline:
                await self._drain_proposals_awaiting_roofline()
        for _ in range(n):
            self.shared_state.increment_tick()
            for name in self._tick_roles:
                await self._reactor_pass(name)
            await self._pump_dispatcher_once()
            # FRAMEWORK_PR phase pump: enqueue next candidate / fetch
            # next batch when no framework_pr task is in flight. Best-
            # effort; failures degrade to phase_done so we never wedge.
            try:
                await self._pump_framework_pr_phase()
            except Exception:  # noqa: BLE001 — defensive
                log.exception("FRAMEWORK_PR pump (tick) failed")
            # phase machine advance at tick boundary.
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
        idle_close_ticks_threshold = _resolve_silent_ticks_closing_threshold()
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
            # Same drain reasoning as ``tick(...)`` — see the comment
            # there. Without this, a session that restarted while
            # analysis was complete but deferred proposals were still
            # queued would never re-dispatch them.
            if self._proposals_awaiting_roofline:
                await self._drain_proposals_awaiting_roofline()

        tick_n = 0
        stop_reason = ""
        closing_deadline: float | None = None
        try:
            while not stop_reason:
                tick_n += 1
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
                    try:
                        await self._pump_framework_pr_phase()
                    except Exception:  # noqa: BLE001 — defensive
                        log.exception("FRAMEWORK_PR pump (run) failed")
                # phase machine advance at tick boundary.
                # Runs even when ``in_closing`` so CLOSE phase still gets
                # recorded into phase_history when the final breakdown
                # writer transitions us in.
                await self._advance_phase_if_needed()

                # N33: bump ``consecutive_silent_ticks`` when the post-
                # tick state shows nothing in flight (no queued / running
                # task, no pending proposal, no ``current_action``). Any
                # non-empty signal means the run is still making forward
                # progress (LLM proposed, executor running, critic
                # reviewing, etc.) so we reset the counter to 0. Skipped
                # while we're already in closing to avoid double-firing
                # the closing-phase trigger below.
                if not in_closing:
                    try:
                        queued_now = len(await self.tasks.queued())
                        running_now = len(await self.tasks.running())
                    except Exception:  # noqa: BLE001
                        queued_now = running_now = 0
                    tick_is_idle = (
                        queued_now == 0
                        and running_now == 0
                        and not self.state.pending_proposals
                        and not (self.shared_state.current_action or "").strip()
                    )
                    if tick_is_idle:
                        self.shared_state.consecutive_silent_ticks += 1
                    else:
                        self.shared_state.consecutive_silent_ticks = 0

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
                # N33: if the run has been silent for
                # ``idle_close_ticks_threshold`` consecutive ticks (LLM
                # has stopped proposing anything actionable, no tasks in
                # flight, no pending proposals), short-circuit to closing
                # phase NOW instead of idling until the wall-clock
                # deadline. This is the common failure mode where the
                # LLM keeps re-proposing rejected ``report`` actions (or
                # no actions at all) and would otherwise burn the
                # remaining budget for nothing. ``threshold <= 0``
                # disables the early-close (legacy behaviour).
                if (
                    idle_close_ticks_threshold > 0
                    and not in_closing
                    and self.shared_state.consecutive_silent_ticks
                        >= idle_close_ticks_threshold
                ):
                    log.warning(
                        "Coordinator: idle for %d consecutive ticks "
                        "(threshold=%d); entering closing phase early "
                        "to flush final report instead of waiting for "
                        "wall-clock deadline (max_minutes=%.0f).",
                        self.shared_state.consecutive_silent_ticks,
                        idle_close_ticks_threshold,
                        max_minutes_value,
                    )
                    if grace_sec <= 0:
                        stop_reason = "idle_timeout"
                        break
                    closing_deadline = await self._enter_closing_phase(
                        grace_sec=grace_sec,
                    )
                    self.shared_state.consecutive_silent_ticks = 0
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
        """Return True when automated explore/kernel levers are exhausted.

        v0.8 plateau judgment lives in :mod:`phase_state`; this helper
        is the safety-net check the closing-phase path consults before
        winding the session down. We require:

        1. baseline + current_best present (otherwise we are still in
           PRELUDE / never finished a round)
        2. no pending proposals or queued / running tasks
        3. ``params_no_promote_streak >= 5`` (proxy still used as
           the cross-phase plateau hint; see KB_gaps/Gap-15)
        4. every reusable kernel_id is rejected.
        """
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
        return self._all_reusable_kernels_rejected()

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
        # ``stack rebench required`` signal (v0.8 M3 / KB_gaps/Gap-10).
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
                        "explore rounds or validate_stack will likely be cut "
                        "by the deadline."
                    )

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
                    "`explore` rounds (which inline the stack rebench) "
                    "will likely be cut by the deadline."
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
            required_step = self._required_next_step()
            if required_step:
                sections.append("=== Execution checklist (Coordinator-enforced) ===")
                sections.append(required_step)

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

    def _required_next_step(self) -> str:
        """Return the coordinator-enforced next step, or empty if flexible.

        The Orchestration prompt says baseline -> profile -> analyze ->
        kernel_opt -> integrate -> validate_stack, but the LLM can still
        skip ahead. This guard makes that sequence deterministic and
        visible in the prompt every tick.

        Pipeline (the ``analyze`` step sits between profile and integrate
        so the TraceLens ``analysis.md`` contract is honored before
        kernel_opt fires. The TODO is purely guidance —
        `_sequence_denial_for_action` still does NOT block explore actions
        (params/backends/sweep) on a stale `last_select_kernels` cache;
        that demoted gate stays demoted. Only
        `_sequence_denial_for_request` enforces it for `run_optimization`
        REQUESTs, as before):

            TODO 0  target_analysis  (only when --compare-against-gpu set)
            TODO 1  baseline
            TODO 2  profile          (kernel mode only)
            TODO 3  integrate        (kernel mode only, after kernel_opt KEEP)
            TODO 4  stack rebench    (when unvalidated KEEPs landed)

        The stack-rebench TODO is critical: once at least one KEEP has
        been added to ``optimization_stack`` since the last rebench,
        skipping it would let the LLM keep stacking per-round gains
        that don't compose linearly and therefore over-report
        ``cumulative_gain`` in the final report. v0.8 M3 + KB_gaps/Gap-10
        inlined the rebench into ``explore`` (the legacy standalone
        ``validate_stack`` action is denied at PolicyGate), so the
        TODO surfaces with a "propose explore" hint.
        """
        if self.shared_state.stop_reason:
            return ""
        if not self._target_analysis_baseline_exists():
            if self._compare_against_gpu:
                return (
                    f"TODO 0/5: target_analysis is required now. "
                    f"--compare-against-gpu="
                    f"{self._compare_against_gpu!r} was set but "
                    "$SESSION_DIR/target_analysis/target_baseline.json is "
                    "missing; propose/delegate only `target_analysis` until "
                    "the external InferenceX reference has been fetched."
                )
            return (
                "TODO 0/5: target_analysis is required now. "
                "$SESSION_DIR/target_analysis/target_baseline.json is "
                "missing; propose/delegate only `target_analysis` so a "
                "reason='no_target_gpu_configured' marker JSON is written "
                "(no --compare-against-gpu was supplied, so this writes a "
                "skipped marker rather than fetching InferenceX data)."
            )
        if self.shared_state.baseline_tput <= 0:
            return (
                "TODO 1/5: baseline is required now. Propose/delegate only "
                "`baseline` until baseline_tput > 0."
            )
        # Profile / analyze / integrate guards only apply when the kernel
        # agent is alive — no-kernel runs have no way to service the
        # request and the mandate would be meaningless.
        # ``select_kernels`` is a prerequisite ONLY for
        # ``run_optimization`` REQUESTs (enforced in
        # ``_sequence_denial_for_request``); the action-layer hard-gate
        # stays demoted (explore actions are not blocked).
        if "kernel" in self.role_registry:
            if not self.shared_state.last_profile_trace:
                return (
                    "TODO 2/5: waiting for the Coordinator-internal analysis "
                    "task to populate ``last_profile_trace``. The Coordinator "
                    "auto-enqueues `roofline` (default) or `profile` (under "
                    "``--no-enable-roofline``) at the end of PRELUDE and on "
                    "every +10% watermark crossing; PolicyGate denies any "
                    "LLM-proposed roofline/profile with "
                    "``rule='analysis_action_not_llm_proposable'``. If "
                    "``auto_roofline_pending_task_id`` is stuck (failed / "
                    "cancelled task that never cleared the field), emit "
                    "`recover` as your escape hatch — the analysis lane is "
                    "Coordinator-owned and not LLM-proposable."
                )
            # analysis.md contract: require `select_kernels` to have
            # run against the current trace so analysis.md exists on disk
            # before any kernel_opt / integrate cycle can fire.  Guidance
            # only -- explore actions (params/backends/sweep/report) are
            # not blocked; only run_optimization is hard-gated on the
            # same cache by `_sequence_denial_for_request`.
            cached = self.shared_state.last_select_kernels or {}
            current_trace = self.shared_state.last_profile_trace
            cache_matches_trace = (
                isinstance(cached, dict)
                and cached.get("trace_input") == current_trace
            )
            if not cache_matches_trace:
                return (
                    "TODO 3/5: analyze is required now. last_profile_trace "
                    f"={current_trace!r} but last_select_kernels is "
                    "empty/stale; TraceLens has not yet produced "
                    "analysis.md for this trace. Emit "
                    "request{target_agent='kernel', kind='select_kernels', "
                    "params={trace_input: <last_profile_trace>, top_k: 10}}. "
                    "Do NOT propose kernel_opt / run_optimization / "
                    "integrate until this cache populates."
                )
            pending_kid = self._kernel_opt_keep_pending()
            if pending_kid:
                return (
                    f"TODO 4/5: integrate is required now. kernel_opt "
                    f"returned KEEP for kernel_id={pending_kid!r} but the "
                    "patch has not been integrated into optimization_stack. "
                    "Emit request{target_agent='kernel', kind='integrate', "
                    f"params={{kernel_id: {pending_kid!r}}} (or "
                    "propose/delegate `integrate` / `recover` / `report`) "
                    "before any further explore."
                )
            # PR-C TODO 4a/5: hot-kernel must-try gate. Surfaces the
            # untried hot reusable kernel queue so Orchestration knows
            # it has to ``run_optimization`` (not ``report``) until the
            # gpu_pct >= 3% set is drained. Same source of truth as the
            # ``_sequence_denial_for_action('report')`` denial.
            #
            # Only fires when N19c (cheap-exhausted) gate has opened --
            # otherwise the LLM would propose ``run_optimization``,
            # bounce off ``execution_order`` repeatedly, and hit the
            # policy_loop auto-stop (Qwen3-30B-A3B-Base 20260523T014653Z
            # died at tick=14 this way). When cheap is still earning
            # marginal gain (last_cheap_delta_gain >= EPSILON), let the
            # LLM keep exploring; PR-C re-activates the instant N19c
            # unlocks.
            if self._kernel_opt_unlocked():
                untried_hot = self.shared_state.untried_hot_reusable_kernels()
                if untried_hot:
                    untried_str = ", ".join(untried_hot)
                    return (
                        f"TODO 4a/5: kernel_opt required on untried hot "
                        f"reusable kernels [{untried_str}]. Each kernel "
                        "with gpu_pct >= 3% (capped at top 5 by gpu_pct) "
                        "must get at least one full backend ladder "
                        "(GEAK -> Claude -> Codex). Emit request"
                        "{target_agent='kernel', kind='run_optimization', "
                        "params={candidates_path="
                        "<last_trace_analyze.candidates_path>}} -- batch "
                        "mode fans out automatically. Failed ladders "
                        "retire the kernel (max_failures=1), so this list "
                        "shrinks monotonically. `report` is denied until "
                        "empty."
                    )
        if self.shared_state.optimization_stack_has_unvalidated_keeps():
            return (
                "TODO 4/4: stack rebench required. New KEEP'd entries have "
                "landed on optimization_stack since the last rebench "
                f"(stack_len={len(self.shared_state.optimization_stack)}, "
                f"validated_at_len="
                f"{self.shared_state.cumulative_gain_validated_stack_len}). "
                "Propose/delegate `explore` — its per-KEEP stack rebench "
                "is inlined and updates cumulative_gain_validated as a "
                "side effect. "
                "Per-round gains do NOT compose linearly — the final "
                "report quotes the validated number, so this is the only "
                "honest gain."
            )
        # TODO 5/5 (current_best path): params/backends can advance
        # ``current_best.tput`` without populating ``optimization_stack``
        # (e.g. when the executor's best_variant lacks an explicit
        # variant_name + candidate_args pair, the lift updates tput but
        # appends nothing to the stack). The final report otherwise
        # reads ``cumulative_gain_validated=0.0%`` with no validation on
        # record. Surface a guidance TODO once the per-round gain crosses
        # a small but meaningful threshold so Orchestration can fire one
        # validate_stack and put an honest number into the report.
        # Guidance only — no PolicyGate denial (the explore loop is not
        # locked) since opt_stack-based unvalidated-KEEP detection still
        # owns that.
        _CB_VALIDATE_THRESHOLD_PCT = 0.5
        cum_gain = float(self.shared_state.cumulative_gain or 0.0)
        already_validated = bool(
            self.shared_state.cumulative_gain_validated_ts
        )
        if (
            cum_gain >= _CB_VALIDATE_THRESHOLD_PCT
            and not self.shared_state.optimization_stack
            and not already_validated
        ):
            cb_action = (self.shared_state.current_best or {}).get(
                "action", "?",
            )
            return (
                f"TODO 5/5: validate_stack recommended. current_best "
                f"advanced to +{cum_gain:.2f}% via "
                f"{cb_action!r} but optimization_stack is empty (no "
                "kernel-opt KEEPs landed). Emit `validate_stack` so "
                "cumulative_gain_validated reflects the current "
                "configuration end-to-end before the final report. "
                "Guidance only — explore actions remain unlocked."
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

        Phase 2 addition: once optimization_stack has unvalidated KEEPs,
        the next ``explore`` round must carry the inlined stack rebench
        (PolicyGate rule ``stack_rebench_required``). The legacy
        ``validate_stack`` standalone action was retired in the legacy release; this
        function now only enforces the cross-action ordering (target_analysis
        before baseline, baseline before everything else, baseline
        self-loop guard, profile/integrate kernel-agent guards).

        ``proposed_params`` is the ``intent.payload["params"]`` dict
        (propose_action / delegate path). Currently only consumed by
        the baseline self-loop guard above, but the kwarg signature is
        kept open so other per-action stop-losses can plug in without
        further call-site churn.
        """
        action = str(action_name or "").strip()
        # the legacy
        # ``backends`` / ``params`` / ``validate_stack`` names are
        # already denied by PolicyGate with ``rule='action_deprecated'``
        # before reaching this function, so they intentionally do not
        # appear in the sequence-gate allow-list.
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
        # ``select_kernels`` is enforced at the REQUEST layer
        # (``_sequence_denial_for_request``) for ``run_optimization``
        # only. ``params`` / ``backends`` / ``sweep`` / ``report`` are
        # never gated on a fresh ``last_select_kernels`` cache — those
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
        # PR-C (0270b67) + c900791 yield-to-N19c: hot-kernel report-gate.
        # Block ``report`` when any reusable hot kernel with gpu_pct >=
        # 3% has not yet been tried (and is not rejected / integrated).
        # Prevents the log1 (164910Z) failure mode where tick=8 ->
        # report_emitted with k001=24% / k002=37% / k004=9.7% untouched.
        #
        # Allowed through the gate:
        #   - kernel_opt request itself (handled at request layer)
        #   - integrate (still needs to drain prior KEEPs; the
        #     integrate-pending gate above already handles ordering)
        #   - recover (not in sequence_actions, bypasses entirely)
        # Blocked:
        #   - report -- the LLM cannot declare the session done
        #     while a meaningful kernel lever exists.
        #
        # The hot_kernel_unfinished rule only fires when
        # ``run_optimization`` is actually dispatchable; otherwise N19c
        # would reject the LLM's resulting request and we'd deadlock
        # the LLM between two opposing gates (death-spiral observed on
        # 20260523T014653Z — see c900791).
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
        # (the wind-down). ``validate_stack`` itself is now denied at
        # the PolicyGate boundary with ``rule='action_deprecated'``,
        # so it can no longer be the recovery action.
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
        # select_kernels / trace_analyze IS the prerequisite request: it
        # produces ``last_select_kernels`` / ``last_trace_analyze`` cache
        # the rest of the chain consults. It is also used directly by
        # tests / tools passing an explicit ``trace_input``, so allow it
        # through; later backends/params/sweep are guarded until the
        # result is cached in SharedState. (Main M4 renamed
        # ``select_kernels`` → ``trace_analyze``; on this branch both
        # request kinds dispatch to the same handler via the
        # back-compat alias in ``kernel_request_handlers.py``, so the
        # allowlist must accept both names to keep this carve-out
        # working under both pre- and post-rename test surfaces.)
        if req_kind in ("select_kernels", "trace_analyze"):
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
                    "select_kernels / run_optimization; analysis is "
                    "auto-enqueued at PRELUDE and on every +10% watermark "
                    "crossing. The analysis lane is Coordinator-owned and not "
                    "LLM-proposable (PolicyGate denies with "
                    "``rule='analysis_action_not_llm_proposable'``). If "
                    "``auto_roofline_pending_task_id`` is stuck, emit "
                    "`recover` as the escape hatch."
                ),
            )
        # Main M4 renamed the cache field from ``last_select_kernels`` to
        # ``last_trace_analyze``; this branch populates BOTH (legacy +
        # canonical) on each handler success, but tests and external
        # callers may seed only the canonical one. Accept either as the
        # prerequisite cache.
        select = (
            self.shared_state.last_trace_analyze
            or self.shared_state.last_select_kernels
            or {}
        )
        needs_select = select.get("trace_input") != self.shared_state.last_profile_trace
        if needs_select and req_kind not in ("select_kernels", "trace_analyze"):
            return PolicyDenied(
                f"request kind={req_kind!r} denied: trace_analyze must run first",
                rule="execution_order",
                hint="emit request kind='trace_analyze' for last_profile_trace",
            )
        # Note (c900791): main also gates ``run_optimization`` here on a
        # gain-driven N19c rule (snapshot_id >= 1 + cheap-attempt
        # recorded + last_cheap_delta_gain < EPSILON). On this branch
        # the equivalent lives in PolicyGate as
        # ``_validate_gain_driven_kernel_opt`` (F3-5 N19c, reads
        # ``gain_per_stack_entry`` instead of v0.6
        # backends_attempts/params_attempts/last_cheap_delta_gain), so
        # the Coordinator-side gate is intentionally omitted here.
        return None

    @staticmethod
    def _allow_early_kernel_opt() -> bool:
        """Escape hatch — ``INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT=1``
        opens the kernel_opt request gate unconditionally (skips both
        the snapshot check and the gain-driven N19c check). Used by
        v0-baseline-comparison flows and unit tests."""
        return os.environ.get(
            "INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT", "",
        ).strip().lower() in {"1", "true", "yes", "on"}

    def _kernel_opt_unlocked(self) -> bool:
        """Return True when ``run_optimization`` is actually dispatchable
        right now — i.e. when the hot-kernel report-gate (PR-C) should
        fire and the LLM should be pushed toward kernel_opt.

        Mirrors PolicyGate's F3-5 N19c ``_validate_gain_driven_kernel_opt``
        check so the two gates never disagree. Without this yield-to-N19c
        on the Coordinator side the LLM oscillates between
        ``hot_kernel_unfinished`` (PR-C, deny report) and
        ``n19c_gain_driven_kernel_opt`` (PolicyGate, deny kernel_opt
        proposal) until ``policy_loop`` auto-stops the session
        (Qwen3-30B-A3B-Base 20260523T014653Z died this way at tick=14).

        The gate is open when ANY of the following hold:
          * escape hatch env set;
          * the gain-driven N19c toggle is off (legacy F3 default);
          * the last ``_N19C_HISTORY_WINDOW`` cheap-round deltas in
            ``gain_per_stack_entry`` average below the F3-5 epsilon
            (i.e. cheap exploration has actually plateaued).
        """
        if self._allow_early_kernel_opt():
            return True
        ss = self.shared_state
        ta = ss.last_trace_analyze or {}
        snapshot_id = ta.get("roofline_snapshot_id", 0)
        if not isinstance(snapshot_id, int) or snapshot_id < 1:
            return False
        if not bool(getattr(ss, "gain_driven_kernel_opt", False)):
            # F3-5 toggle off → mirror PolicyGate's early-return; gate
            # is considered open once a roofline snapshot exists.
            return True
        try:
            from .policy import PolicyGate as _PolicyGate
            window = _PolicyGate._N19C_HISTORY_WINDOW
            epsilon = _PolicyGate._N19C_EPSILON_PCT
        except Exception:  # noqa: BLE001 — defensive, fall back to F3-5 defaults
            window, epsilon = 3, 0.5
        history = list(getattr(ss, "gain_per_stack_entry", []) or [])
        deltas: list[float] = []
        for entry in reversed(history):
            if isinstance(entry, dict):
                d = entry.get("delta_pct")
                if isinstance(d, (int, float)):
                    deltas.append(float(d))
            if len(deltas) >= window:
                break
        if len(deltas) < window:
            return False
        return (sum(deltas) / float(len(deltas))) < float(epsilon)

    # Note: main commit c900791 also ports the N22 keyword-implied
    # advice (``_record_keyword_implied_advice``,
    # ``_registered_variants_for``). On this branch the equivalent
    # functionality lives in
    # ``orchestrator/_analysis_keyword_map.py`` + PolicyGate's
    # ``analysis_keyword_advisory`` rule. The Coordinator-side
    # helpers from main are therefore omitted here (they would also
    # re-import the dropped backends.py / params.py grids).

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
            # v0.8 §3.5 §10 + terminal intent of a
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
        # Gate proposals on a pending auto-roofline / auto-profile task
        # *before* paying for the Critic round-trip. This is the cheap
        # path; ``_materialize_approved_proposal`` carries a symmetric
        # check for the race where the watermark trips while a proposal
        # is already in front of the Critic.
        if action_name in _ROOFLINE_GATED_ACTIONS:
            roofline_denied = await self._auto_roofline_pending_denial(
                action_name=action_name,
            )
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
        # T2 anchor: mint optimization_node + hypothesize edge.
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
          hypothesize edge — identical to the legacy M1 behaviour.

        Best-effort: every Cortex KB failure is downgraded to an NDJSON
        enqueue by the client itself; partial failures within a
        per-variant batch are logged but don't poison the other
        variants in the same proposal.

        Gap-anchor selection:
        :meth:`_resolve_issue_canonical` consults
        ``payload.gap_canonical_id`` and ``payload.params.gap_canonical_id``
        before falling back to the M1 ``recipe_canonical_id``
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
        iter_idx = self.shared_state.increment_session_iter_index()
        pending.experiment_iter_index = iter_idx
        opt_canonical = experiment_canonical_id(sid, iter_idx)
        gap_canonical = self._resolve_issue_canonical(pending)
        try:
            self.cortex_kb.propose_point(
                canonical_id=opt_canonical,
                kind="experiment",
                authority="HYPOTHESIZED",
                attrs={
                    "session_id":          sid,
                    "iter_index":          iter_idx,
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
                    # phase provenance on every edge so
                    # cross-session reachability queries can filter
                    # by phase.
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
        if pending.experiment_iter_index <= 0:
            pending.experiment_iter_index = (
                self.shared_state.increment_session_iter_index()
            )
        parent_exp_canonical = experiment_canonical_id(
            sid, pending.experiment_iter_index,
        )
        # Anchor the registered ``experiment`` parent once (idempotent).
        # Variants below pin to it via ``.variant-{name}`` canonical
        # suffix on the unregistered ``optimization_node`` kind.
        try:
            self.cortex_kb.propose_point(
                canonical_id=parent_exp_canonical,
                kind="experiment",
                authority="HYPOTHESIZED",
                attrs={
                    "session_id":         sid,
                    "iter_index":         pending.experiment_iter_index,
                    "action":             pending.action_name,
                    "from_agent":         pending.from_agent,
                    "predicted_gain_pct": pending.predicted_gain_pct,
                    "proposal_msg_id":    pending.proposal_msg_id,
                    "phase":              phase,
                    "variants":           len(grid),
                },
                evidence=[f"log:proposal-{pending.proposal_msg_id}"],
            )
        except CortexKBError as exc:
            log.warning("propose_point parent experiment failed: %s", exc)
        for variant in grid:
            if not isinstance(variant, dict):
                continue
            variant_name = str(variant.get("name") or "").strip()
            if not variant_name:
                # Skip nameless variants — the executor itself rejects
                # them downstream, so we don't even mint a phantom edge.
                continue
            opt_canonical = f"{parent_exp_canonical}.variant-{variant_name}"
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

    def _gap_anchor_canonical_id(self) -> str:
        """M1 placeholder for the gap anchor.

        Per "M1 simplification — use
        ``workload_node.canonical_id`` as the from side of every
        hypothesize edge". M5 specialist framework will introduce real
        ``issue_node`` anchors keyed by gap descriptors. Centralising
        the derivation here keeps the migration to M5 a single-line
        change.
        """
        from ..cortex_kb_client import recipe_canonical_id
        workload = self.shared_state.model_name or "unknown_model"
        hw = self.shared_state.gpu_type or "unknown_gpu"
        return recipe_canonical_id(workload, hw)

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
        """Route a Critic ``review_verdict`` to the per-variant or
        legacy single-verdict handler.

        v0.8 KB_gaps/Gap-11: the
        intent_parser already validated that exactly one of
        ``verdict`` / ``verdict_map`` is present. We branch on
        ``verdict_map`` first so the batch Explore path takes
        precedence; everything else (kernel_opt / integrate /
        report / specialist dispatch) falls through to the legacy
        single-verdict path.
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
        if isinstance(verdict_map, dict) and verdict_map:
            await self._handle_verdict_map(
                source=source,
                pending=pending,
                verdict_map=verdict_map,
                rationale=str(intent.payload.get("reasoning") or ""),
            )
            return
        await self._handle_single_verdict(
            source=source,
            pending=pending,
            verdict=str(single_verdict or ""),
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

        PR-A7 (Arbor-into-Hyperloom) adds the integrate_patch critic
        gate: when the pending proposal is an ``integrate_patch`` for
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
        # PR-A7: mirror specialist / integrate_patch verdicts onto
        # SharedState so PolicyGate's integrate_patch gate can
        # consult them on the next tick.
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
                    "PR-A7: failed to mirror critic verdict for "
                    "specialist task=%s", sid_candidate,
                )
        if verdict == "approve":
            await self._materialize_approved_proposal(pending)

    async def _handle_verdict_map(
        self,
        *,
        source: str,
        pending: "PendingProposal",
        verdict_map: dict[str, dict[str, Any]],
        rationale: str,
    ) -> None:
        """v0.8 KB_gaps/Gap-11 per-variant Critic verdict handler.

        Splits a batch ``verdict_map`` into approved / rejected
        partitions and:

        1. Pins the full map on the :class:`PendingProposal` so
           subsequent reactor passes (resume rebuild,
           ``decided`` checks) can see the per-variant decisions
           without re-parsing the bus.
        2. Mirrors the map back onto the bus as a single
           ``review_verdict`` event carrying both the per-variant
           detail *and* a summary single ``verdict`` (``approve``
           when any variant survived; ``reject`` when all are
           rejected; ``needs_review`` otherwise) so legacy
           consumers (resume's verdict_by_target rebuild,
           breakdown.kb_writes_summary) keep their meaning.
        3. Fires KB ``refuted`` for every rejected variant via
           :meth:`_cortex_t3_critic_rejected` (Gap-11 §5.4) — the
           critic's reject is itself negative KB evidence; we
           don't need to wait for explore to run.
        4. Materialises an ``explore`` task whose ``grid`` is the
           approved subset (passes the names down via
           :meth:`_materialize_approved_proposal`).

        Non-``explore`` proposals never reach this method
        (``_handle_review_verdict`` routes them through the legacy
        single-verdict path); we still defend against operator
        misuse by short-circuiting with an observation when
        somehow we land here on a non-grid proposal.
        """
        if pending.action_name != "explore":
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind":             "verdict_map_for_non_explore",
                    "target":           pending.proposal_msg_id,
                    "action_name":      pending.action_name,
                    "variants_in_map":  sorted(verdict_map.keys()),
                },
            )
            return

        # Index the proposal's original grid by variant_name so we can
        # (a) drop ``verdict_map`` keys that don't match a real
        # variant, (b) carry the variant's payload (extra_args,
        # extra_envs, ...) through to the materialised task, and
        # (c) report a structured "unknown_variants" observation
        # for the operator audit log.
        original_grid: list[dict[str, Any]] = []
        raw_grid = (pending.payload.get("params") or {}).get("grid")
        if isinstance(raw_grid, list):
            for v in raw_grid:
                if isinstance(v, dict) and v.get("name"):
                    original_grid.append(v)
        original_names = {str(v.get("name")): v for v in original_grid}
        approved_names: list[str] = []
        rejected: list[tuple[str, str]] = []
        unknown: list[str] = []
        for vname, entry in verdict_map.items():
            name = str(vname)
            sub_verdict = str((entry or {}).get("verdict") or "").strip()
            sub_rationale = str((entry or {}).get("rationale") or "")
            if name not in original_names:
                unknown.append(name)
                continue
            if sub_verdict == "approve":
                approved_names.append(name)
            elif sub_verdict == "reject":
                rejected.append((name, sub_rationale))
            # ``redirect`` / ``advise`` / ``needs_review`` neither
            # land in the executor grid nor fire a KB refute — they
            # surface in the audit log via the bus mirror below.

        summary_verdict = (
            "approve" if approved_names
            else "reject" if rejected and not approved_names
            else "needs_review"
        )

        # 1. Pin on the pending proposal.
        pending.decided = True
        pending.verdict = summary_verdict
        pending.verdict_map = {
            str(k): dict(v) if isinstance(v, dict) else {"verdict": str(v)}
            for k, v in verdict_map.items()
        }

        # 2. Bus mirror — single event carrying the full map AND a
        #    summary verdict so legacy consumers keep working.
        await self.bus.append_and_seq(Message.new(
            source, pending.from_agent, "review_verdict",
            {
                "target_proposal_msg_id": pending.proposal_msg_id,
                "verdict":                summary_verdict,
                "verdict_map":            pending.verdict_map,
                "approved_variants":      sorted(approved_names),
                "rejected_variants":      sorted(n for n, _ in rejected),
                "unknown_variants":       sorted(unknown),
                "reasoning":              rationale,
            },
            priority=0 if summary_verdict == "reject" else 1,
            in_reply_to=pending.proposal_msg_id,
        ))

        # 3. KB refuted for critic-rejected variants (Gap-11 §5.4).
        for vname, sub_rationale in rejected:
            try:
                await self._cortex_t3_critic_rejected(
                    pending=pending,
                    variant_name=vname,
                    rationale=sub_rationale,
                )
            except Exception:  # noqa: BLE001 — best-effort KB write
                log.exception(
                    "cortex T3 critic-rejected failed for proposal=%s "
                    "variant=%s", pending.proposal_msg_id, vname,
                )

        # 4. Materialise only the approved subset.
        if approved_names:
            await self._materialize_approved_proposal(
                pending,
                approved_variant_names=set(approved_names),
            )
        else:
            # Whole grid rejected — observation only; downstream
            # plateau judges + breakdown reads the bus mirror to
            # surface "critic rejected K of K".
            await self._record_observation(
                "coordinator", "observation",
                {
                    "kind":              "verdict_map_all_rejected",
                    "target":            pending.proposal_msg_id,
                    "rejected_variants": sorted(n for n, _ in rejected),
                    "unknown_variants":  sorted(unknown),
                },
            )

    async def _cortex_t3_critic_rejected(
        self,
        *,
        pending: "PendingProposal",
        variant_name: str,
        rationale: str,
    ) -> None:
        """KB ``refuted`` mirror for a critic-rejected variant.

        KB_gaps/Gap-11 §5.4: when the Critic rejects a variant at
        the verdict_map stage we record an immediate ``verify``
        with ``outcome='refuted'`` on the matching T2 edge so the
        Cortex view distinguishes "critic refused" from "executor
        ran and failed". Depends on Gap-07's per-variant
        ``kb_edge_ids`` map — a noop when the proposal predates
        the T2 hook (no edge to refute).
        """
        if not variant_name:
            return
        edge_id = (pending.kb_edge_ids or {}).get(variant_name) or ""
        if not edge_id:
            # No per-variant edge — Gap-07 didn't fire (e.g. --no-
            # cortex, or T2 hook was skipped). Skip; the explore
            # executor never runs this variant either, so there's
            # nothing to refute.
            return
        cortex = getattr(self, "cortex_kb", None)
        if cortex is None or not getattr(cortex, "enabled", False):
            return
        sid = (self.shared_state.cortex_session_id or "").strip()
        if not sid:
            return
        try:
            cortex.verify(
                sid=sid,
                edge_id=edge_id,
                outcome="refuted",
                evidence=[
                    f"proposal:{pending.proposal_msg_id}",
                    f"variant:{variant_name}",
                    "stage:critic",
                    (f"rationale:{rationale[:200]}" if rationale else "rationale:none"),
                ],
                promote_authority=None,
                idempotency_key=(
                    f"verify_critic_reject:{sid}:{edge_id}:{variant_name}"
                ),
            )
        except Exception:  # noqa: BLE001 — defensive
            log.exception(
                "cortex T3 critic-rejected verify failed for variant=%s",
                variant_name,
            )

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

        ``approved_variant_names`` (v0.8 KB_gaps/Gap-11): when set,
        filter the ``explore`` grid down to the named subset before
        materialising. Used by the per-variant Critic verdict_map
        path so a 4-variant proposal with 2 approved + 2 rejected
        dispatches only the 2 approved variants to the executor.
        ``None`` (legacy single-verdict path) keeps the full grid.
        """
        # Safety net for the race where the watermark fires while a
        # proposal is already in front of the Critic.
        # ``_handle_propose_action`` carries the same check (the cheaper
        # path); this one catches proposals the Critic approved between
        # the watermark crossing and the dispatch tick. Defer rather
        # than drop — the Critic already approved, so re-running the
        # round-trip would be wasted budget.
        if pending.action_name in _ROOFLINE_GATED_ACTIONS:
            roofline_denied = await self._auto_roofline_pending_denial(
                action_name=pending.action_name,
            )
            if roofline_denied is not None:
                self._proposals_awaiting_roofline.append(
                    (pending, approved_variant_names),
                )
                # Resume contract: this observation carries everything
                # ``replay_for_resume`` needs to rebuild the deferred
                # queue after a restart (proposal_msg_id keys the
                # PendingProposal in the bus's proposal-topic events;
                # approved_variant_names + kb_edge_ids preserve the
                # per-variant filter the Critic produced and the
                # Cortex T2 edge stamping). A subsequent
                # ``approved_proposal`` decision carrying the same
                # proposal_msg_id signals that the drain has dispatched
                # the proposal so resume should skip it.
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
                return
        params = dict(pending.payload.get("params") or {})
        # stamp per-variant kb_edge_id into the
        # grid so the explore executor (which already reads
        # ``variant.get("kb_edge_id")``) can carry the id through to
        # the result rows the ledger writer + T3 hook will consume.
        # No-op when the proposal isn't ``explore`` or when the T2
        # hook didn't populate the map (e.g. ``--degraded-kb`` runs).
        if (
            pending.action_name == "explore"
            and isinstance(params.get("grid"), list)
        ):
            stamped_grid: list[dict[str, Any]] = []
            for variant in params["grid"]:
                if not isinstance(variant, dict):
                    # v0.8 KB_gaps/Gap-11: non-dict slots can't carry
                    # a name, so they are *always* dropped when a
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
                variant_copy = dict(variant)
                if pending.kb_edge_ids:
                    edge_id = pending.kb_edge_ids.get(vname, "")
                    if edge_id and not variant_copy.get("kb_edge_id"):
                        variant_copy["kb_edge_id"] = edge_id
                stamped_grid.append(variant_copy)
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
            str(cb.get("extra_sglang_args") or "")
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
            # Fix E (per-variant overtime kill — Q1 baseline-anchored,
            # Q5 default-on-with-flag). These two fields are NOT LLM
            # strategy: they're operational knobs the Coordinator
            # owns. We always inject them so the ExploreExecutor can
            # derive ``soft_deadline_sec = baseline_runtime_sec *
            # explore_overtime_kill_ratio`` for the single-variant
            # phase (stack rebench intentionally bypasses this gate
            # — Q4). ``setdefault`` leaves an LLM-supplied override
            # in place for one-off rebench / debug variants.
            br = float(getattr(self.shared_state, "baseline_runtime_sec", 0.0) or 0.0)
            if br > 0:
                params.setdefault("baseline_runtime_sec", br)
            kill_ratio = float(getattr(
                self.shared_state, "explore_overtime_kill_ratio", 0.0,
            ) or 0.0)
            if kill_ratio > 0:
                params.setdefault("explore_overtime_kill_ratio", kill_ratio)
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
        # Critic gate for explore grids: every ``delegate{action_name='explore',
        # params={grid: [...]}}`` is reviewed per-variant by the Critic before
        # any benchmark runs. We re-route through ``_handle_propose_action``
        # so the proposal lands in ``pending_proposals``, the Critic emits a
        # ``verdict_map``, and ``_handle_verdict_map`` materialises only the
        # approved subset (variants the Critic rejects fire a KB ``refuted``
        # edge via ``_cortex_t3_critic_rejected`` and never reach the executor).
        # The proposal path re-runs is_pruned + _sequence_denial_for_action,
        # and ``_materialize_approved_proposal`` writes the task via
        # ``tasks.create_or_return_existing`` directly, so this re-route
        # cannot recurse back into ``_handle_delegate``. Empty/missing grid
        # falls through to the legacy delegate path (nothing to filter).
        params_preview = intent.payload.get("params") or {}
        grid_preview = (
            params_preview.get("grid") if isinstance(params_preview, dict) else None
        )
        if (
            action_name == "explore"
            and isinstance(grid_preview, list)
            and grid_preview
        ):
            await self._handle_propose_action(source, intent)
            return
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
        # Fix E (parity with _materialize_approved_proposal): inject
        # overtime-kill operational knobs for direct explore delegates
        # too. The common path is the Critic-gated reroute above, but
        # this branch covers anything that bypasses the verdict map
        # (legacy resume, tests that delegate explore directly).
        if action_name == "explore":
            br = float(getattr(self.shared_state, "baseline_runtime_sec", 0.0) or 0.0)
            if br > 0:
                params.setdefault("baseline_runtime_sec", br)
            kill_ratio = float(getattr(
                self.shared_state, "explore_overtime_kill_ratio", 0.0,
            ) or 0.0)
            if kill_ratio > 0:
                params.setdefault("explore_overtime_kill_ratio", kill_ratio)
        # IR-7 — ``assess_remaining_gaps`` is a thin wrapper: rewrite
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
        if action_name in _ROOFLINE_GATED_ACTIONS:
            denied = await self._auto_roofline_pending_denial(
                action_name=action_name,
            )
            if denied is not None:
                await self._record_policy_denied(
                    source, intent, denied, action_name=action_name,
                )
                return

        # v0.8 §3.5 + specialist pre-dispatch warmup.
        # When the Orchestration role delegates a specialist task, the
        # Coordinator is the only place with the KnowledgePlane facade
        # in scope. Warm the prompt's external-knowledge sections here
        # so SpecialistRunner's prompt assembly sees them via task
        # params. ``setdefault`` lets the caller (Orchestration) pre-
        # supply values; we only fill the gaps.
        if action_name == "specialist":
            await self._warm_specialist_params(params)
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
    # v0.8 §3.5 + specialist pre-dispatch warmup
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

        # PR-A5 (Arbor-into-Hyperloom): KB sub-graph traverse warmup.
        # The plain ``pr_feed`` covers PR Monitor; this fills the
        # specialist prompt's ``## 4. KB SUB-GRAPH`` section so the LLM
        # starts with the Cortex KB anchor expanded rather than having
        # to call ``mcp__cortex_kb__traverse`` itself.
        if (
            plane is not None
            and getattr(plane, "cortex_enabled", False)
            and "kb_subgraph" not in params
        ):
            try:
                # PR-A10: pass the (lowercased) hw_slug so the per-domain
                # fallback in select_kb_for_domain can filter recipe
                # candidates to the same GPU.
                from ..cortex_kb_client import _slug as _kb_slug
                hw_slug = _kb_slug(state.gpu_type or "", "")
                subgraph = plane.select_kb_for_domain(
                    domain, hw_slug=hw_slug or None,
                )
                params["kb_subgraph"] = subgraph or {}
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "specialist warmup: select_kb_for_domain(domain=%r) "
                    "failed: %r",
                    domain, exc,
                )
                params.setdefault("kb_subgraph", {})
        else:
            params.setdefault("kb_subgraph", {})

        # Warm-start recipe + pitfalls from T0 anchor.
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

        # fill gap-specific anchors from the
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
        if a in {"kernel_opt", "integrate", "select_kernels", "run_optimization"}:
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
    # v0.8 §3.5 §10 + specialist_done bookkeeping
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

        T2 per-variant hypothesize is
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

        # IR-7 — route session_steward_specialist verdicts. Done payload
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

    async def _route_steward_verdict(
        self, *, task: "Task", done_payload: dict[str, Any],
    ) -> None:
        """IR-7 — process a session_steward_specialist verdict.

        Side effects depending on ``recommendation``:

        * ``stop_session``    → ``state.set_stop_reason('no_more_leverage')``.
        * ``advance_to_kernel`` → ``state.set_pending_escalate_hint('skip_to_kernel')``.
        * ``continue_explore`` → append ``next_gap_canonical_id`` to
          ``state.gaps[]``, reset ``params_no_promote_streak`` and
          per-domain empty streak counters, set
          ``steward_continuation_used=True``.

        Antiloop: only one continuation per session. A second
        ``continue_explore`` is coerced to ``advance_to_kernel`` so
        the EXPLORE→KERNEL transition becomes mandatory.

        Out-of-vocab ``recommendation`` values are coerced to
        ``stop_session`` (defense in depth — the LLM can write any
        string, but we only honour the closed enum).
        """
        raw_rec = str(done_payload.get("recommendation") or "").strip().lower()
        if raw_rec not in _STEWARD_RECS:
            log.warning(
                "steward: out-of-vocab recommendation=%r for task=%s; "
                "coercing to 'stop_session'",
                raw_rec, task.task_id,
            )
            raw_rec = "stop_session"
        # Antiloop: only one continuation per session.
        if (
            raw_rec == "continue_explore"
            and bool(getattr(
                self.shared_state, "steward_continuation_used", False,
            ))
        ):
            log.info(
                "steward: continue_explore requested but continuation "
                "already used this session; coercing to "
                "'advance_to_kernel' (task=%s)",
                task.task_id,
            )
            raw_rec = "advance_to_kernel"
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
            self.shared_state.set_stop_reason("no_more_leverage")
            log.info(
                "steward: recommendation='stop_session' for task=%s "
                "-> stop_reason='no_more_leverage'",
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
            self.shared_state.params_no_promote_streak = 0
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
        entry: dict[str, Any] = {
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
                # PR-X (1cd9f7d): force batch dispatch for run_optimization.
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
                            "extra_sglang_args": rejected.get("extra_sglang_args", ""),
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
                        # still wins. See PR-B follow-up.
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
                # Cache select_kernels / trace_analyze output so subsequent
                # identical requests are short-circuited next tick. Only
                # cache real successful runs, not failures, to avoid
                # sticky errors. Main M4 renamed ``select_kernels`` →
                # ``trace_analyze`` so accept both request kinds at the
                # cache write site too — the back-compat alias on
                # ``KERNEL_REQUEST_HANDLERS`` lets either name reach
                # here.
                if (
                    kind in ("select_kernels", "trace_analyze")
                    and cache_hit_source is None
                    and result.get("status") in ("ok", "succeeded")
                ):
                    self.shared_state.record_select_kernels(merged_payload, result)
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
                    # Note: main commit ce36e70 also routes the decision
                    # into a per-action scoreboard (KEEP / no-promote
                    # accounting on the action-priority table). The
                    # scoreboard was retired by KB_design §3.9 on this
                    # branch (                     # §4), so the post-record bookkeeping is omitted.
                    self.shared_state.save(self.session_dir)
                if kind == "integrate":
                    if result.get("status") != "skipped":
                        self.shared_state.record_kernel_integrate_result(result)
                    decision = str(result.get("decision", "")).upper()
                    if decision == "KEEP":
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

        Main M4 renamed ``select_kernels`` → ``trace_analyze``; both
        request kinds dispatch to the same handler via the back-compat
        alias and the cache lives under either of ``last_trace_analyze``
        (canonical post-M4) or ``last_select_kernels`` (legacy resume
        parity). Prefer the canonical field; fall back to the legacy
        one so cached requests work regardless of which field the
        handler write path populated last.
        """
        if kind not in ("select_kernels", "trace_analyze"):
            return None
        cached = (
            self.shared_state.last_trace_analyze
            or self.shared_state.last_select_kernels
            or {}
        )
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
        streak = self.shared_state.record_policy_denial(
            action_name=resolved_action,
            rule=str(denied.rule or ""),
            hint=str(denied.hint or ""),
            intent_type=intent.type.value,
            tick=int(self.shared_state.tick or 0),
            intent_payload=intent.payload,
        )
        # the streak counter is still a fact (LLM sees it
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
                extra_sglang_args=extra_args,
                ts=entry["ts"],
            )

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
        return result.get("status") != "failed"

    def _record_intervention_for_task(
        self, task: "Task", result: Any,
    ) -> None:
        """PR-A8 (Arbor-into-Hyperloom): log the change_type of a
        completed task into ``SharedState.intervention_mix``.

        Mapping:
        * ``task.kind == "explore"``       → ``change_type = "config"``
          (every explore KEEP is a config tweak per v0.8 M3).
        * ``task.kind == "integrate_patch"`` AND the executor reported
          ``status == "kept"`` → ``change_type = "code_patch"``.
        * ``integrate_patch`` with any other status (reverted /
          apply_failed / rejected_by_critic / applied_no_bench) →
          NOT recorded; only successful KEEPs roll the counter.

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
            if str(result.get("status") or "") != "kept":
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
            # v0.8 §3.5 §10 + specialist bookkeeping.
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
                # PR-A8 (Arbor-into-Hyperloom): bump the per-EXPLORE
                # specialist dispatch counter. Robustness reads this
                # in its prompt context to detect storms (many
                # specialists dispatched with no winning proposal).
                try:
                    self.shared_state.bump_specialist_dispatched()
                except Exception:  # noqa: BLE001
                    log.exception("PR-A8: bump_specialist_dispatched failed")
            # PR-A8 — intervention-mix ledger: when an explore or
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
            # N34 Bug #4 (May 2026): a successful ``report`` task is the
            # canonical terminal signal — ``final.md`` / ``final.json``
            # are already on disk, so any further LLM-driven exploration
            # is waste. Set ``stop_reason='report_emitted'`` so the next
            # run-loop iteration breaks out. Skip when ``stop_reason``
            # is already set (signal / other terminal already won) so
            # we don't paper over an earlier failure.
            # N34 Bug #4 (May 2026) only fires for LLM-driven report
            # tasks (kind="report" outside the closing phase). When the
            # closing-phase report task is already in flight,
            # ``in_closing`` owns the stop-reason transition — let the
            # run loop's existing ``time_exhausted`` / ``no_more_leverage``
            # path resolve it. Otherwise an LLM-proposed mid-run report
            # would burn the rest of the budget without this guard.
            if (
                task.kind == "report"
                and result.state == "succeeded"
                and not (self.shared_state.stop_reason or "").strip()
                and not self.shared_state.closing_report_task_id
            ):
                log.info(
                    "Coordinator: report task %s succeeded; setting "
                    "stop_reason='report_emitted' to terminate the run "
                    "loop (N34 Bug #4 fix).",
                    task.task_id,
                )
                self.shared_state.stop_reason = "report_emitted"
                self.shared_state.save(self.session_dir)
            # T3 anchor. Always called
            # so KEEP / REVERT each get a corresponding ingest-attempt +
            # verify pair. Best-effort: failures are absorbed into the
            # NDJSON queue by the client.
            await self._cortex_t3_hook(task=task, result=result, kept=kept)
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
    # Cortex KB T3 hook (KEEP / REVERT mirror)
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
            # T2 hook never ran (e.g. --degraded-kb during materialize) or
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
        # Dedupe repeated ``--flag value`` pairs so the cumulative
        # extra_sglang_args reflects what argparse will actually honor
        # (last value wins). Without this, every promote that re-applies
        # a multi-value combo candidate (e.g.
        # ``--cuda-graph-max-bs 32 --cuda-graph-max-bs 128 --cuda-graph-max-bs 256``)
        # leaves multiple copies of the same flag in the final args
        # string, which breaks reproduce-launch dashboards and the
        # final.extra_sglang_args field surfaced by session_breakdown.
        full_args = _dedupe_extra_sglang_args(full_args)

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
                # SharedState docstring for the V1 StackGainEntry contract.
                self.shared_state.append_stack_gain_entry(
                    action=task_kind,
                    variant_name=variant_name,
                    new_tput=best_tput,
                    extra_sglang_args=full_args,
                )

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
            # Fix E: promote the baseline Magpie wall-clock so the
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
            # PRELUDE bootstrap: enqueue the first analysis action
            # (kind picked by ``shared_state.enable_roofline``:
            # ``roofline`` by default, ``profile`` when
            # ``--no-enable-roofline``). Idempotency-keyed via the
            # fixed ``prelude_initial`` reason so a resume after the
            # baseline completion edge does not double-enqueue.
            #
            # Skip conditions:
            # * baseline tput missing or invalid;
            # * an analysis task is already in-flight (gate field set).
            if (
                isinstance(tput, (int, float)) and tput > 0
                and not (self.shared_state.auto_roofline_pending_task_id or "").strip()
            ):
                try:
                    rl_task = await self._enqueue_internal_analysis_task(
                        reason="prelude_initial",
                    )
                    self.shared_state.auto_roofline_pending_task_id = rl_task.task_id
                    log.info(
                        "PRELUDE: baseline landed (tput=%.2f); auto-enqueued "
                        "initial %s task=%s",
                        float(tput), rl_task.kind, rl_task.task_id,
                    )
                except Exception as exc:  # noqa: BLE001 — defensive
                    log.exception(
                        "PRELUDE: failed to enqueue initial analysis task "
                        "after baseline: %r", exc,
                    )
        elif task_kind == "profile":
            # IR-8 fallback: profile_executor used to short-circuit on
            # FRAMEWORK=atom with status="skipped" + error_class=
            # "atom_no_profiler". That short-circuit was removed once
            # Magpie's atom_mi*x.sh learned to bridge PROFILE=1 to
            # atom's --torch-profiler-dir (atom natively supports torch
            # profiler via /start_profile + /stop_profile). This
            # ``skipped`` arm is now a *defensive* path: it still runs
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
                # Stale select_kernels / trace_analyze cache no longer matches
                # this trace. Clear both (M4 main merge: ``last_trace_analyze``
                # is the canonical post-rename field; ``last_select_kernels``
                # is kept on this branch for legacy resume parity).
                self.shared_state.last_select_kernels = {}
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
                # IR-8 fallback: the roofline composite used to short-
                # circuit on FRAMEWORK=atom with status="skipped"
                # because profile was a hard dependency and atom had
                # no profiler wiring. That short-circuit was removed
                # once Magpie's atom_mi*x.sh learned to bridge PROFILE=1
                # to atom's --torch-profiler-dir. This arm is now the
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
                # N10: prefer the executor's already-published
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
                # N27 — reset the outer roofline failure streak on a
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
                # N27 — bump the outer failure streak. The action_failure
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
            # v0.8 M3 + KB_gaps/Dead-A.5 (prerequisite to Gap-10) —
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
                    self._lift_to_current_best(
                        "explore", float(best_tput), best_winner,
                    )
                    promoted = True
                    changed = True
            if promoted:
                # Reset the plateau proxy on a successful KEEP so
                # the M2 transitional fallback in phase_state stays
                # aligned with the unified ledger. The proxy is dual-
                # tracked for resume parity.
                self.shared_state.params_no_promote_streak = 0
                # v0.8 M3 §4.4 + explore inlines the
                # per-KEEP stack rebench, so the post-rebench
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
                # No KEEP cleared the rebench. Bump the proxy so the
                # plateau judges see the no-progress run.
                self.shared_state.params_no_promote_streak += 1
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
                    "candidate_extra_sglang_args": "",
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
