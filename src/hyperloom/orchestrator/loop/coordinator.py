# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from hyperloom.orchestrator.knowledge.recipe_kb import RecipeKB

# Recipe snapshot severity tags (schema has no fixed enum).
_SEVERITY_CRASH: str = "crash"
_SEVERITY_REGRESS: str = "regress"

# Bounded transient-failure auto-retry for specialist dispatches: a subprocess
# timeout / crash / stale-heartbeat re-dispatches up to this many times before
# the failure is recorded normally. Infra-only (see classify_specialist_failure);
# semantic empties are left for the orchestrator. Env override / disable via
# INFERENCE_OPTIMIZER_SPECIALIST_AUTO_RETRY (set "0" to disable).
SPECIALIST_AUTO_RETRY_MAX: int = 2

# Periodic in-process maintenance/reaper cadence (lease reaping + DB retention). Runs
# every N coordinator ticks: actively reaps expired serving + GPU leases and
# prunes the events/tasks DB so a multi-day single-session run never leaks
# capacity or grows the DB unbounded (no process restart clears them). Env
# override via INFERENCE_OPTIMIZER_MAINTENANCE_EVERY_TICKS ("0" disables).
MAINTENANCE_EVERY_TICKS: int = 50

# Default per-macro-cycle wall-clock window (hours) in cyclic mode. Each
# phase's budget fraction (DEFAULT_PHASE_BUDGET_PCT) applies to this window
# rather than the whole run. Env override: INFERENCE_OPTIMIZER_CYCLE_HOURS.
DEFAULT_CYCLE_HOURS: float = 24.0
# Trailing window for the crash-rate emergency stop: the threshold counts only
# crashes within this many seconds so old crashes age out on long runs/resume.
_CRASH_EMERGENCY_WINDOW_SEC: float = 24.0 * 3600.0
# Combined baseline-failure backstop: fast-fail after this many TOTAL baseline
# failures (any error_class), so mixed classes can't dodge the per-class streaks.
_BASELINE_MAX_TOTAL_FAILURES: int = 3
# Enablement stall cap: consecutive enablement rounds that neither made the
# combo runnable nor advanced to a NEW failure signature. Reaching it stops the
# loop with stop_reason ``enablement_stalled`` instead of re-deriving the same
# fix until the wall-clock deadline. A *progressing* round resets the streak, so
# an N-gap serial enablement is bounded by N + this cap, not capped at N.
_ENABLEMENT_MAX_STALL: int = int(os.environ.get("INFERENCE_OPTIMIZER_ENABLEMENT_MAX_STALL", "3") or "3")
# Floor on the per-repo framework-PR discover timeout so a slow repo still gets a
# usable budget even when the phase timeout is spread thin across many repos.
_FRAMEWORK_MIN_PER_REPO_TIMEOUT_SEC: float = 30.0
# Default min TRANSFER confidence a warm-replay champion must clear to be enqueued.
_DEFAULT_WARM_REPLAY_MIN_CONFIDENCE: float = 0.7
# Default resume-drift floor (%): a re-measured current_best below this fraction
# of its recorded tput is flagged as drift. Overridable via env.
_DEFAULT_RESUME_DRIFT_FLOOR_PCT: float = 95.0
from ..phases import machine_state as _phase_state
from ..state.optimization_journal import Journal
from hyperloom.inference_optimizer.session.paths import db_path_for
from ..actions.registry import ActionRegistry
from ..roles.agent_role import AgentRole, default_role_registry
from ..roles.base import Backend, BackendError, BackendTurnResult
from ..bus.cursor_store import CursorStore
from ..bus.storage.connection import SqliteConnection
from hyperloom.inference_optimizer.protocol.intent import NoIntentEmitted
from ..bus.message_bus import Message, MessageBus
from ..state.objective import Objective, TimeOnlyObjective
from ..policy.gate import (
    PolicyGate,
    SPECIALIST_FROM_AGENT_PREFIX,  # noqa: F401 - re-exported for callers/tests
)
from ..bus.gpu_pool import (
    SpecialistGpuPool,
    resolve_gpu_specialist_devices,
    resolve_whole_machine_devices,
)
from ..bus.resource_lock import (
    ResourceLockManager,
    SqliteLeaseBackend,
)
from ..state.shared_state import SharedState
from .intent_router import IntentRouter
from .result_recorder import ResultRecorder
from .sub_agent_runner import SubAgentRunner
from ..state.task_registry import TaskRegistry
from ..trace.llm_trace import LLMCallRecord, append_llm_call
from .coordinator_helpers import (  # noqa: F401 - re-exported for callers/tests
    _BASELINE_FINGERPRINT_KEYS,
    _baseline_params_fingerprint,
    _dedupe_extra_server_args,
    _infer_model_class_from_config,
    _merge_cumulative_extra_sglang_args,
    _parse_baseline_workload_extra,
    _parse_iso_unix,
    _resolve_roofline_watermark_ratio,
    effective_closing_grace_sec,
    format_exc_brief,
    serialize_verdict_advisory,
)


log = logging.getLogger(__name__)


# Audit-trail kinds (must match shared_state._AUDIT_ACTIONS); kernel_agent-owned actions excluded.
_AUDIT_ACTIONS: frozenset[str] = frozenset(
    {
        "baseline",
        "profile",
        "sweep",
        "explore",
        # Composite roofline runs profile + trace_analyze atomically.
        "roofline",
    }
)

# Default per-repo candidate cap for ``fa phase-discover`` (FRAMEWORK).
DEFAULT_FRAMEWORK_MAX_CANDIDATES: int = 8


def _extract_enablement_launch_log(result_payload: dict[str, Any] | None) -> str:
    """Extract launch/traceback text from a failed baseline result payload.

    Feeds ``framework_agent.enablement.classify_failure``. Concatenates the
    most likely error-bearing fields (``error`` / ``stderr`` / ``log_tail`` /
    ``traceback`` / ``reason``) so a "can't even boot" baseline failure becomes
    classifiable text. Returns ``""`` when nothing usable is present.

    Args:
        result_payload: The failed task's result dict (``None`` treated empty).

    Returns:
        str: Concatenated, trimmed launch-log text (may be ``""``).
    """
    if not isinstance(result_payload, dict):
        return ""
    parts: list[str] = []
    for key in ("error", "stderr", "log_tail", "log_excerpt", "traceback", "reason"):
        val = result_payload.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
        elif isinstance(val, (list, tuple)):
            joined = "\n".join(str(x) for x in val if str(x).strip())
            if joined.strip():
                parts.append(joined.strip())
    return "\n".join(parts).strip()


def _resolvable_artifacts_from_done(
    done_payload: dict[str, Any] | None,
    resolve_bases: list[Path],
) -> list[dict[str, Any]]:
    """Return ``artifacts_written`` entries whose ``source`` file exists on disk.

    A FRAMEWORK/EXPLORE specialist may deliver a non-diff tuned artifact (e.g.
    an autotuned aiter ``tuned_fmoe`` CSV) via ``artifacts_written`` instead of
    a source patch. Such a deliverable is a first-class result: it flows through
    the ``integrate_patch`` artifact-install channel (backup + install + bench +
    accuracy gate + REVERT). This is the SHARED routable-signal used by BOTH the
    autosubmit bridge (``_maybe_autosubmit_specialist_patches``) and the
    empty-outcome bridge (``_record_framework_agent_authoring_empty_outcome``)
    so they share one rule: an artifact is routable only when its ``source``
    resolves to a real file inside the specialist worktree/workspace. (Each
    bridge passes its own already-unwrapped ``specialist_done`` view — outer for
    autosubmit, ``payload``-unwrapped for empty-outcome — so agreement holds as
    long as ``specialist_done`` is not double-wrapped, the same assumption the
    ``patches_written`` / config-lever routing already relies on.) Keeping the
    bridges on one signal prevents the FRAMEWORK livelock (skip-stamp without a
    following integrate_patch).

    Source containment: every ``source`` — relative (resolved under
    ``resolve_bases``) or absolute — is accepted ONLY when it resolves to a real
    file inside one of ``resolve_bases``. This matches integrate_patch's sandbox
    so the signal never routes a deliverable integrate_patch would reject as
    ``source_outside_workspace`` (incl. a relative ``..`` that escapes the
    sandbox yet still resolves to an existing file).

    Target scope (intentional trade-off): this pure signal validates the
    ``source`` only; it does NOT re-resolve ``target`` against the framework
    allowlist (that would couple a testable pure function to on-disk framework
    roots). A malformed / out-of-allowlist ``target`` is therefore still routed
    and rejected downstream by integrate_patch as a terminal ``no_patches`` row
    (no livelock — the FRAMEWORK pump still advances), rather than skip-stamped
    as ``authored_empty`` here.

    Args:
        done_payload: The specialist ``specialist_done`` payload (unwrapped).
        resolve_bases: Dirs to resolve a relative ``source`` against (typically
            ``[<spec>/worktree, <spec>]``).

    Returns:
        list[dict]: ``artifacts_written`` entries with a valid ``source``/
            ``target`` and an existing source file (possibly empty).
    """
    if not isinstance(done_payload, dict):
        return []
    arts = done_payload.get("artifacts_written")
    if not isinstance(arts, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in arts:
        if not isinstance(entry, dict):
            continue
        src = str(entry.get("source") or "").strip()
        tgt = str(entry.get("target") or "").strip()
        if not src or not tgt:
            continue
        raw = Path(src)
        # Candidate paths: an absolute ``source`` is checked as-is; a relative
        # ``source`` is resolved under each base. In BOTH cases the resolved
        # path must be a real file that stays inside a base — the SAME
        # containment integrate_patch's ``_resolve_artifact_specs`` enforces,
        # rejecting ``source_outside_workspace`` AND ``..`` escapes (a relative
        # ``../../x`` that ``is_file()`` alone would otherwise mis-route).
        cands = [raw] if raw.is_absolute() else [base / raw for base in resolve_bases]
        bases_resolved = [base.resolve() for base in resolve_bases]
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


def _framework_config_levers_from_done(
    done_payload: dict[str, Any] | None,
) -> dict[str, str]:
    """Extract a config-lever set from a FRAMEWORK specialist deliverable.

    A specialist may translate an upstream PR into a CONFIG win (serving flags /
    env vars already reachable on this build) instead of a source patch. Such a
    deliverable is a first-class FRAMEWORK result: it flows through the
    existing ``integrate_patch`` ``config_changes`` channel (apply + bench +
    accuracy gate), NOT an authored_empty skip.

    The levers are read from the FIRST ``proposal_set`` entry that carries
    ``extra_args`` and/or ``extra_envs`` (the standard explore-variant schema).
    ``extra_args`` (a server-arg string or list) and ``extra_envs`` (a mapping)
    are flattened into a single ``{KEY: value}`` config-changes dict that
    ``integrate_patch`` layers onto the launch env. Returns ``{}`` when no
    config lever is present (the caller then treats the deliverable as a patch
    or an empty outcome as before).

    Args:
        done_payload: The specialist ``specialist_done`` payload (already
            unwrapped of any envelope).

    Returns:
        dict[str, str]: The flattened config-change mapping, or ``{}``.
    """
    if not isinstance(done_payload, dict):
        return {}
    # A patch deliverable takes precedence — it is not a config-only outcome.
    patches = done_payload.get("patches_written") or []
    if isinstance(patches, list) and patches:
        return {}
    proposals = done_payload.get("proposal_set") or []
    if not isinstance(proposals, list):
        return {}
    for entry in proposals:
        if not isinstance(entry, dict):
            continue
        levers: dict[str, str] = {}
        envs = entry.get("extra_envs")
        if isinstance(envs, dict):
            for k, v in envs.items():
                key = str(k).strip()
                if key:
                    levers[key] = str(v)
        args = entry.get("extra_args")
        arg_tokens: list[str] = []
        if isinstance(args, str) and args.strip():
            arg_tokens = args.split()
        elif isinstance(args, (list, tuple)):
            arg_tokens = [str(a) for a in args if str(a).strip()]
        # Fold ``--flag value`` / ``--flag=value`` / bare ``--flag`` pairs into
        # the config dict so integrate_patch can re-emit them as server args.
        i = 0
        while i < len(arg_tokens):
            tok = arg_tokens[i].strip()
            if not tok:
                i += 1
                continue
            if "=" in tok and tok.startswith("-"):
                k, _, v = tok.partition("=")
                levers[k.strip()] = v.strip()
                i += 1
                continue
            if tok.startswith("-") and i + 1 < len(arg_tokens) and not arg_tokens[i + 1].startswith("-"):
                levers[tok] = str(arg_tokens[i + 1]).strip()
                i += 2
                continue
            levers[tok] = ""
            i += 1
        if levers:
            return levers
    return {}

# Hard-trigger thresholds: EXPLORE rounds a domain may go without a
# specialist dispatch / a KEEP before the Coordinator force-dispatches one (a
# real scheduling event, not an advisory nudge). Overridable via SharedState.
FORCE_STALLED_SPECIALIST_ROUNDS: int = 8
FORCE_STALLED_KEEP_ROUNDS: int = 12


# Result keys surfaced in delegated_result inbox line; first match wins per group.
_OUTCOME_GAIN_KEYS: tuple[str, ...] = (
    "validated_gain_pct",
    "gain_pct",
    "predicted_gain_pct",
    "delta_pct",
)
_OUTCOME_TPUT_KEYS: tuple[str, ...] = (
    "tokens_per_s",
    "tput",
    "throughput",
    "tput_tok_s",
)
_OUTCOME_STATUS_KEYS: tuple[str, ...] = ("status", "verdict", "outcome")

def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    """Return ``d[k]`` for the first ``k`` in ``keys`` present + non-None.

    Args:
        d: Mapping to look up; a non-dict argument yields ``None``.
        keys: Candidate keys checked in order; the first whose value is
            non-None wins.

    Returns:
        The first present, non-None value, or ``None`` if none match.
    """
    if not isinstance(d, dict):
        return None
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return None


def _format_inbox_event(m: "Message") -> str:
    """Render one inbox ``Message`` as a compact, high-signal line.

    Args:
        m: The inbox message to render; its topic selects a per-topic
            formatting branch (delegated_result, policy_denial, review_verdict,
            observation) with a generic fallback.

    Returns:
        A single-line string summarising the message header and payload.
    """
    topic = (m.topic or "").strip()
    payload = m.payload if isinstance(m.payload, dict) else {}
    # Canonical inbox header ordering that downstream parsers anchor on.
    if getattr(m, "msg_id", None):
        head = f"seq={m.seq} msg_id={m.msg_id} from={m.from_agent} topic={topic}"
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

    if topic in ("policy_denial", "denial") or (topic == "observation" and payload.get("kind") == "policy_denial"):
        return (
            f"{head} action={payload.get('action_name')!r} "
            f"rule={payload.get('rule')!r} "
            f"hint={str(payload.get('hint') or '')[:140]!r}"
        )

    if topic == "review_verdict":
        parts = [
            f"{head} target={payload.get('target_proposal_msg_id')!r} "
            f"verdict={payload.get('verdict')!r} "
            f"reasoning={str(payload.get('reasoning') or '')[:140]!r}"
        ]
        advisory = serialize_verdict_advisory(payload)
        required_evidence = advisory.get("required_evidence")
        if required_evidence:
            shown = "; ".join(str(item) for item in required_evidence[:3])
            parts.append(f"required_evidence[{len(required_evidence)}]={shown[:140]!r}")
        risks = advisory.get("risks")
        if risks:
            parts.append(f"risks={len(risks)}")
        advice_text = advisory.get("advice_text")
        if advice_text:
            parts.append(f"advice={advice_text[:140]!r}")
        return " ".join(parts)

    if topic == "observation":
        kind = payload.get("kind")
        if kind is not None:
            return f"{head} kind={kind!r} payload={payload}"

    return f"{head} payload={payload}"


@dataclass
class PendingProposal:
    """A propose_action intent waiting for Critic Review."""

    proposal_msg_id: str
    from_agent: str
    action_name: str
    predicted_gain_pct: float
    payload: dict[str, Any]
    decided: bool = False
    verdict: str | None = None  # approve / reject / redirect / advise / needs_review


# Path-like keys surfaced from a kernel handler
# payload (inputs) or result (outputs) so operators can see where a step's
# artifacts went without enumerating every per-handler return shape.
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
    "tracelens_agent_transcript",
    "tracelens_agent_report",
    # TraceLens analysis outputs surfaced by trace_analyze_handler — the
    # analysis.md report, its alias, the per-run audit summary, the roofline
    # sidecar and the CLI log — so operators can reach them from lifecycle END.
    "trace_report_path",
    "analysis_report_path",
    "tracelens_summary_path",
    "kernel_roofline_path",
    "cli_log_path",
)


def _lifecycle_paths(payload: Any) -> dict[str, str]:
    """Extract present, non-empty path-like fields from a kernel handler
    payload or result dict. Best-effort: a non-dict argument yields
    an empty mapping so callers never have to guard the type.

    Args:
        payload: A kernel handler payload or result; non-dict inputs are
            tolerated and produce an empty mapping.

    Returns:
        Mapping of recognised path-like key to its non-empty string value.
    """
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


class _CoordinatorMeta(type):
    """Class-level delegation for extracted collaborator methods. Instance access is handled by ``Coordinator.__getattr__``; class
    access (``Coordinator._extracted_method`` — used by tests that copy methods
    onto stub classes / call them unbound) resolves here to the owning
    collaborator *class*'s function. Collaborator modules are imported lazily to
    avoid an import cycle (they import from ``coordinator`` at module top)."""

    def __getattr__(cls, name):  # noqa: N805 - metaclass first arg is the class
        prop = cls._DELEGATED.get(name)
        if prop is not None:
            import importlib

            mod, clsname = cls._COLLAB_MODULES[prop]
            module = importlib.import_module(f"hyperloom.orchestrator.{mod}")
            return getattr(getattr(module, clsname), name)
        raise AttributeError(f"type object {cls.__name__!r} has no attribute {name!r}")


class Coordinator(metaclass=_CoordinatorMeta):
    """The single Coordinator instance per session."""

    # property name -> (module, collaborator class) for class-level delegation
    # (instance-level uses the lazy properties directly). Kept in sync with the
    # lazy collaborator properties + ``_DELEGATED``.
    _COLLAB_MODULES = {
        # Phase handlers, registered in call-chain order: machine -> prelude ->
        # sweep -> close -> internal ->
        # kernel_stack -> kernel -> explore -> framework. ``framework`` is
        # placed last because it owns the most delegated methods (48, the
        # largest of the 9 phase clusters). Module paths are relative to
        # ``hyperloom.orchestrator``.
        "phase_machine": ("phases.machine", "MachinePhase"),
        "phase_prelude": ("phases.prelude", "PreludePhase"),
        "phase_sweep": ("phases.sweep", "SweepPhase"),
        "phase_close": ("phases.close", "ClosePhase"),
        "phase_internal": ("phases.internal", "InternalTasksPhase"),
        "phase_kernel_stack": ("phases.kernel_stack", "KernelStackPhase"),
        "phase_kernel": ("phases.kernel", "KernelPhase"),
        "phase_explore": ("phases.explore", "ExplorePhase"),
        "phase_framework": ("phases.framework", "FrameworkPhase"),
        "router": ("loop.intent_router", "IntentRouter"),
        "recorder": ("loop.result_recorder", "ResultRecorder"),
        "maintenance": ("loop.maintenance", "MaintenanceCollaborator"),
        "resume_helper": ("loop.resume", "ResumeCollaborator"),
        "writeback": ("loop.writeback", "WritebackCollaborator"),
        "gating": ("loop.gating", "GatingCollaborator"),
        "dispatcher": ("loop.dispatcher", "DispatcherCollaborator"),
        "proposals": ("loop.proposals", "ProposalsCollaborator"),
        "advisory": ("loop.advisory", "AdvisoryCollaborator"),
        "inline_actions": ("loop.inline_actions", "InlineActionsCollaborator"),
        "conversation": ("loop.conversation", "ConversationCollaborator"),
    }

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
        self._phase_budget_pct: dict[str, float] = _phase_state.normalize_budget_pct(phase_budget_pct)
        # Specialist stale scan threshold (seconds).
        try:
            self._specialist_stale_sec: float = max(
                0.0,
                float(
                    os.environ.get(
                        "INFERENCE_OPTIMIZER_SPECIALIST_STALE_SEC",
                        "600",
                    )
                ),
            )
        except ValueError:
            self._specialist_stale_sec = 600.0
        # External launcher config; decides whether target_analysis fetches real rows.
        self._compare_against_gpu: str = (compare_against_gpu or "").strip()
        self._model_class_override: str = (model_class or "").strip()

        # Validate every reactor has a backend wired.
        for name in self.role_registry:
            if name not in backends:
                raise ValueError(f"missing backend for role {name!r} (provide via Coordinator(backends={{...}}))")
        self.backends = dict(backends)

        # Persistence layer
        db_path = db_path_for(self.session_dir)
        self.db = SqliteConnection(db_path)

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
        # Serving-disjoint physics invariant: the
        # live serving process holds the first ``serving_tp`` cards, so they are
        # carved off the specialist pool to avoid shared-card measurement
        # corruption. ``shared_state.tp`` is restored on resume; the ``TP`` env
        # (exported by the CLI before construction) is the fresh-start fallback.
        self.gpu_specialist_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_gpu_specialist_devices(
                int(getattr(self.shared_state, "gpu_specialist_capacity", 0) or 0),
                serving_tp=self._resolve_serving_tp(),
            ),
        )
        # Framework-authoring pool over the whole node (serving cards not carved
        # off, no gpu_specialist_capacity gate). Shares the ``gpu_leases`` table
        # with ``gpu_specialist_pool``; the cap-1 ``gpu_research_lane`` mutex
        # serializes the two so they never hold cards at the same time.
        self.framework_gpu_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_whole_machine_devices(),
        )
        # Dispatcher re-scan poll: while awaiting in-flight tasks, re-scan the
        # queue at this cadence so a queued GPU task starts the moment its lane
        # frees (instead of waiting out a long specialist / integrate_patch that
        # was already being awaited). FIRST_COMPLETED handles lane frees that
        # coincide with a task completion; the poll covers TTL/external releases.
        try:
            self._dispatcher_poll_sec = max(
                0.05,
                float(os.environ.get("INFERENCE_OPTIMIZER_DISPATCHER_POLL_SECONDS", "10") or 10.0),
            )
        except (TypeError, ValueError):
            self._dispatcher_poll_sec = 10.0
        # Sync research_lane capacity into lane_capacity so acquire_many honours the cap.
        try:
            from ..bus.storage.schema import set_lane_capacity as _set_lane_capacity

            cap = int(self.shared_state.research_lane_capacity or 0)
            if cap >= 0:
                _set_lane_capacity(self.db.raw, "research_lane", cap)
        except Exception:  # noqa: BLE001 — non-fatal; default seed wins
            log.exception("failed to sync research_lane_capacity to leases DB")
        # WS2: gpu_research_lane stays capacity-1 (strictly serial GPU
        # specialists). The co-acquisition lock model can't express a
        # multi-holder lane that is also mutually exclusive with the cap-1
        # serving lanes, so a single GPU specialist holds the whole machine at a
        # time (gpu_count up to the whole machine); the GPU pool partitions the
        # physical cards within that one lease. The seed default (1) is correct;
        # no runtime sync needed.
        # `strict_paths` defers to the env flag (on in production, off in tests).
        self.policy = PolicyGate(
            role_registry=self.role_registry,
            session_dir=self.session_dir,
            shared_state=self.shared_state,
        )
        # Attach read-only context-pull MCP tools to Orchestration backend.
        self._attach_orchestration_context_tools()
        # Resume detection must run before any boot-time state.json write.
        self._resumed_from = self._detect_resume_state()
        # Derive model_class once at boot if not supplied; never overwrite a resume.
        if not (self.shared_state.model_class or "").strip():
            self.shared_state.model_class = self._model_class_override or _infer_model_class_from_config(
                self.shared_state.model_path or os.environ.get("MODEL_PATH", "")
            )
        self.state = CoordinatorState()
        self._stop = asyncio.Event()
        self._tasks_running: list[asyncio.Task] = []
        # Orchestration prompt mode: first turn full SEED, later turns DELTA.
        self._orchestration_seeded: bool = False
        # Orchestration working-memory checkpoint policy + tracker.
        from ..state import orchestration_memory as _orch_mem

        # Context-token guardrail (#3): derive soft/hard budgets from the
        # orchestration model's window × fraction (env-overridable). Falls back to
        # a conservative 200k window for unknown models; 0 budgets => token
        # triggers disabled (char/tick/time cadence still applies).
        def _ckpt_fraction(env_key: str, default: float) -> float:
            try:
                v = float(os.environ.get(env_key, "").strip() or default)
            except (TypeError, ValueError):
                v = default
            return v if 0.0 < v <= 1.0 else default

        _orch_model = str(getattr(self.backends.get("orchestration"), "model", "") or "")
        _ctx_window = _orch_mem.context_window_for_model(_orch_model)
        _soft_frac = _ckpt_fraction(
            "INFERENCE_OPTIMIZER_CTX_SOFT_FRACTION",
            _orch_mem.DEFAULT_CONTEXT_TOKEN_SOFT_FRACTION,
        )
        _hard_frac = _ckpt_fraction(
            "INFERENCE_OPTIMIZER_CTX_HARD_FRACTION",
            _orch_mem.DEFAULT_CONTEXT_TOKEN_HARD_FRACTION,
        )
        self._checkpoint_policy = _orch_mem.CheckpointPolicy(
            context_token_soft=int(_ctx_window * _soft_frac),
            context_token_hard=int(_ctx_window * _hard_frac),
        )
        self._checkpoint_tracker = _orch_mem.CheckpointTracker(
            last_phase=str(getattr(self.shared_state, "phase", "") or ""),
        )
        # Minimum ticks between orchestration-memory compactions. A compaction
        # resets the persistent conversation and forces a full SEED re-push next
        # tick, so allowing two compactions back-to-back can wedge the run in a
        # checkpoint-every-tick loop (the SEED's own tokens re-trip the budget).
        # A true near-window emergency bypasses this floor (see _maybe_checkpoint).
        try:
            self._checkpoint_min_tick_gap: int = max(
                1, int(os.environ.get("INFERENCE_OPTIMIZER_CHECKPOINT_MIN_TICK_GAP", "").strip() or 3)
            )
        except (TypeError, ValueError):
            self._checkpoint_min_tick_gap = 3
        # Consecutive degenerate checkpoint replies (#1); resets on a good one.
        self._consec_degenerate_ckpt: int = 0
        # Disable checkpointing entirely via env.
        self._checkpoint_enabled: bool = os.environ.get(
            "INFERENCE_OPTIMIZER_DISABLE_ORCH_CHECKPOINT",
            "",
        ).strip().lower() not in {"1", "true", "yes", "on"}
        # Seed memory rendered into the next full SEED push (resume recovery source).
        # Operator rollback (#1): INFERENCE_OPTIMIZER_ORCH_MEMORY_ROLLBACK=<n>
        # re-seeds from the n-th-from-newest history snapshot instead of the live
        # memory, to recover when the most recent compaction(s) lost a key thread.
        _seed_memory = dict(getattr(self.shared_state, "orchestration_memory", {}) or {})
        _rollback_raw = os.environ.get("INFERENCE_OPTIMIZER_ORCH_MEMORY_ROLLBACK", "").strip()
        if _rollback_raw:
            try:
                _n = int(_rollback_raw)
                _hist = list(getattr(self.shared_state, "orchestration_memory_history", []) or [])
                if _n >= 1 and len(_hist) >= _n:
                    _seed_memory = dict(_hist[-_n])
                    self.shared_state.orchestration_memory = _seed_memory
                    log.warning(
                        "Coordinator: orchestration memory rolled back to history[-%d] "
                        "(of %d snapshots)",
                        _n,
                        len(_hist),
                    )
                else:
                    log.warning(
                        "Coordinator: ORCH_MEMORY_ROLLBACK=%s out of range (history has %d); "
                        "using live memory",
                        _rollback_raw,
                        len(_hist),
                    )
            except (TypeError, ValueError):
                log.warning(
                    "Coordinator: invalid ORCH_MEMORY_ROLLBACK=%r; using live memory",
                    _rollback_raw,
                )
        self._orchestration_seed_memory: str = _orch_mem.render_memory_for_seed(_seed_memory)
        # No-progress circuit-breaker telemetry; threshold = high-severity cutoff.
        self._progress_marker: dict[str, Any] = {}
        try:
            self._no_progress_threshold: int = max(
                1,
                int(
                    os.environ.get(
                        "INFERENCE_OPTIMIZER_NO_PROGRESS_TICKS",
                        "15",
                    )
                ),
            )
        except ValueError:
            self._no_progress_threshold = 15

        # Periodic maintenance/reaper cadence (lease reaping + DB retention). 0 disables.
        try:
            self._maintenance_every_ticks: int = max(
                0,
                int(
                    os.environ.get(
                        "INFERENCE_OPTIMIZER_MAINTENANCE_EVERY_TICKS",
                        str(MAINTENANCE_EVERY_TICKS),
                    )
                ),
            )
        except ValueError:
            self._maintenance_every_ticks = MAINTENANCE_EVERY_TICKS

        # R2: when cyclic mode is on, pin a per-macro-cycle budget window so the
        # per-phase budget fractions apply per cycle, not per whole run. The
        # window only *takes effect* for long/unbounded runs — ``_budget_minutes``
        # (and the reloop gate) ignore it for short bounded runs (--max-hours ≤
        # 24), whose real budget is not even known here at __init__ (it is set
        # later in ``run()``). Keeping the assignment unconditional is therefore
        # harmless: short runs stay anchored on the whole session.
        if _phase_state.is_cyclic_phases_enabled() and float(getattr(self.shared_state, "cycle_minutes", 0) or 0) <= 0:
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

        # R6: medium-intensity soft restart at each macro-cycle boundary
        # (reap/prune/clear-caches + compacted-memory conversation reset). On by
        # default in cyclic mode; opt out via the env flag.
        self._cycle_soft_restart: bool = os.environ.get(
            "INFERENCE_OPTIMIZER_DISABLE_CYCLE_SOFT_RESTART",
            "",
        ).strip().lower() not in {"1", "true", "yes", "on"}
        # The soft restart's inference-server deep-clean kills lingering server
        # processes (vLLM/SGLang/atom workers). It is safe at a cycle boundary
        # (no benchmark in flight) but is the highest-blast-radius step, so it is
        # separately gated and defaults ON within the soft restart; opt out via
        # the env flag (tests set it to avoid touching real /proc).
        self._cycle_restart_servers: bool = os.environ.get(
            "INFERENCE_OPTIMIZER_DISABLE_CYCLE_SERVER_RESTART",
            "",
        ).strip().lower() not in {"1", "true", "yes", "on"}

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

        # Stable tick order from the live role_registry (NOT the cached
        # module-level roles_for_run, which keeps "kernel_agent" under --no-kernel).
        _CANONICAL_ORDER = ("orchestration", "kernel_agent", "critic", "robustness")
        self._tick_roles: tuple[str, ...] = tuple(r for r in _CANONICAL_ORDER if r in self.role_registry)

        # Action registry — yaml catalogue mapping action_name → metadata;
        # load failure falls back to ``None`` (handled gracefully).
        try:
            self.action_registry: ActionRegistry | None = ActionRegistry().load()
        except Exception:  # noqa: BLE001 — defensive; missing yaml shouldn't kill the run.
            log.exception("Coordinator: failed to load ActionRegistry.")
            self.action_registry = None
        # Inline fast-action execution: run cheap lane-light action in-turn. Default ON.
        _inline_raw = (
            os.environ.get(
                "INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS",
                "",
            )
            .strip()
            .lower()
        )
        self._inline_fast_actions_enabled: bool = _inline_raw not in {
            "0",
            "false",
            "no",
            "off",
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

    @property
    def router(self) -> IntentRouter:
        """Intent routing collaborator (extracted from this class).

        The ``_handle_*`` intent handlers were moved verbatim into
        :class:`IntentRouter`; the ``_handle_*`` methods remaining on this class
        are thin forwarding shims that delegate here. Built lazily and cached so
        that test doubles constructed via ``Coordinator.__new__`` (bypassing
        ``__init__``) still resolve a router on first access.
        """
        r = self.__dict__.get("_router")
        if r is None:
            r = IntentRouter(self)
            self.__dict__["_router"] = r
        return r

    @property
    def recorder(self) -> ResultRecorder:
        """Result-recording / fact-synthesis collaborator (extracted from this class).

        The ``_record_*`` / ``_build_*`` / ``cortex_finalize_recipe_and_journal``
        / research-evidence methods were moved verbatim into
        :class:`ResultRecorder`; the methods remaining here are thin forwarding
        shims. Built lazily and cached, same as :attr:`router`, so test doubles
        constructed via ``Coordinator.__new__`` resolve a recorder on first use.
        """
        r = self.__dict__.get("_recorder")
        if r is None:
            r = ResultRecorder(self)
            self.__dict__["_recorder"] = r
        return r

    # Methods extracted into collaborator objects are delegated back by name here
    # (symmetric to each collaborator's
    # ``__getattr__`` back to this coordinator). ``coord.foo`` / ``self.foo`` /
    # ``self._coord.foo`` all keep resolving, and instance-attr monkeypatches
    # still shadow them (normal lookup wins over __getattr__). Each value is the
    # name of the property returning the owning collaborator.
    _DELEGATED = {
        # router
        "_handle_intent": "router", "_handle_propose_action": "router",
        "_handle_review_verdict": "router", "_handle_single_verdict": "router",
        "_handle_delegate": "router", "_handle_specialist_done": "router",
        "_handle_request": "router", "_handle_response": "router",
        "_handle_kill_task": "router", "_handle_prune_branch": "router",
        "_handle_escalate_strategy_change": "router", "_handle_send_message": "router",
        "_handle_alert": "router", "_handle_update_state": "router",
        # recorder
        "_aggregate_research_evidence": "recorder", "_harvest_research_scout": "recorder",
        "_record_specialist_result": "recorder",
        # Phase handlers, grouped in the same call-chain order as
        # _COLLAB_MODULES/the @property block above:
        # machine -> prelude -> sweep -> close -> internal -> kernel_stack ->
        # kernel -> explore -> framework (framework last: largest cluster).
        "_ensure_phase_initialised": "phase_machine",
        "_ensure_cortex_t0_anchored": "phase_machine",
        "_kernel_enabled": "phase_machine",
        "_explore_enabled": "phase_machine",
        "_advance_phase_if_needed": "phase_machine",
        "_on_phase_entered": "phase_machine",
        "_record_phase_entry_evidence": "phase_machine",
        "_internal_analysis_kind": "phase_prelude",
        "_warm_recipe_proven_items": "phase_prelude",
        "_inject_warm_recipe_history_into_ledger": "phase_prelude",
        "_filter_warm_patches_with_kg": "phase_prelude",
        "_maybe_enqueue_warm_replay": "phase_prelude",
        "_promote_warm_replay": "phase_prelude",
        "_maybe_enqueue_prelude_initial_analysis_after_baseline": "phase_prelude",
        "_enqueue_internal_analysis_task": "phase_prelude",
        "_on_enter_sweep": "phase_sweep",
        "_enqueue_internal_conc_sweep_task": "phase_sweep",
        "_enqueue_internal_sweep_task": "phase_sweep",
        "_build_sweep_params_from_recipe": "phase_sweep",
        "_derive_close_stop_reason": "phase_close",
        "_session_integrated_kernel_patch": "phase_close",
        "_maybe_run_close_post_opt_roofline": "phase_close",
        "_on_enter_close": "phase_close",
        "_enqueue_internal_report_task": "phase_close",
        "_enqueue_internal_session_breakdown_task": "phase_close",
        "_record_close_step": "phase_close",
        "_enter_closing_phase": "phase_close",
        "_closing_report_terminal": "phase_close",
        "_enqueue_internal_research_scout_task": "phase_internal",
        "_maybe_enqueue_prelude_research_scout": "phase_internal",
        "_maybe_enqueue_explore_research_scout": "phase_internal",
        "_enqueue_internal_static_recon_task": "phase_internal",
        "_maybe_enqueue_prelude_static_recon": "phase_internal",
        "_maybe_enqueue_trajectory_reviewer": "phase_internal",
        "_consume_static_recon": "phase_internal",
        "_drain_pending_keep_integrates": "phase_kernel_stack",
        "_positive_needs_review_integrates": "phase_kernel_stack",
        "_stack_resolved_kernel_ids": "phase_kernel_stack",
        "_mark_stack_validation_entries_resolved": "phase_kernel_stack",
        "_stack_component_identities": "phase_kernel_stack",
        "_mark_stack_validation_in_progress": "phase_kernel_stack",
        "_clear_stack_validation_in_progress": "phase_kernel_stack",
        "_clear_pending_stack_validation_checkpoints": "phase_kernel_stack",
        "_recover_interrupted_stack_validation": "phase_kernel_stack",
        "_stack_entries_for_validation": "phase_kernel_stack",
        "_finalize_stack_validation_outcome": "phase_kernel_stack",
        "_maybe_validate_positive_needs_review_stack": "phase_kernel_stack",
        "_run_kernel_stack_validation_e2e": "phase_kernel_stack",
        "_auto_enqueue_pending_integrations": "phase_kernel_stack",
        "_maybe_reprofile_for_kernel": "phase_kernel",
        "_geak_enabled": "phase_kernel",
        "_on_enter_kernel": "phase_kernel",
        "_run_bf16_dense_gemm_fallback": "phase_kernel",
        "_should_run_bf16_dense_gemm_fallback": "phase_kernel",
        "_bf16_dense_gemm_fallback_pending": "phase_kernel",
        "_bf16_dense_gemm_fallback_attempted": "phase_kernel",
        "_is_bf16_dense_gemm_fallback_attempt": "phase_kernel",
        "_resolve_bench_protocol": "phase_kernel",
        "_geak_timeouts": "phase_kernel",
        "_run_geak_kernel_phase": "phase_kernel",
        "_geak_win_already_recorded": "phase_kernel",
        "_geak_legacy_promote": "phase_kernel",
        "_parse_geak_accepted_config": "phase_kernel",
        "_record_geak_candidate": "phase_kernel",
        "_promote_geak_from_candidate": "phase_kernel",
        "_promote_geak_result": "phase_kernel",
        "_record_geak_kernel_journey": "phase_kernel",
        "_ck_blockscale_switch_eligible": "phase_kernel",
        "_ck_switch_precision_is_fp8": "phase_kernel",
        "_handle_gemm_tuning_result": "phase_kernel",
        "_journal_gemm_tuning_keep": "phase_kernel",
        "_promote_gemm_tuning_keep": "phase_kernel",
        "_replace_latest_gemm_tuning_attempt": "phase_kernel",
        "_validate_forge_gemm_tuning_e2e": "phase_kernel",
        "_should_continue_kernel_after_gemm": "phase_kernel",
        "_run_kernel_opt_after_gemm": "phase_kernel",
        "_current_tput_from_validated_gain": "phase_kernel",
        "_last_measured_roofline_tput": "phase_kernel",
        "_needs_roofline_for_watermark": "phase_kernel",
        "_maybe_enqueue_watermark_roofline": "phase_kernel",
        "_cached_kernel_request": "phase_kernel",
        "_negative_ledger_domain_counts": "phase_explore",
        "_plan_cycle_focus": "phase_explore",
        "_record_cycle_strategy_for_current_cycle": "phase_explore",
        "_cycle_strategy_seed_block": "phase_explore",
        "_apply_macro_cycle_reloop": "phase_explore",
        "_run_cycle_soft_restart": "phase_explore",
        "_restart_inference_servers": "phase_explore",
        "_on_enter_explore": "phase_explore",
        "_maybe_force_stalled_domain_specialist": "phase_explore",
        "_seed_gaps_from_research_hints": "phase_explore",
        "_scan_stale_specialists": "phase_explore",
        "_fan_out_specialist_wave": "phase_explore",
        "_maybe_auto_retry_specialist": "phase_explore",
        "_warm_specialist_params": "phase_explore",
        "_refresh_gaps": "phase_explore",
        "_extract_gaps_from_baseline": "phase_explore",
        "_extract_gaps_from_attempts": "phase_explore",
        "_gap_layer_for_action": "phase_explore",
        "_record_explore_round_gaps": "phase_explore",
        "_task_id_from_specialist_source": "phase_explore",
        "_maybe_materialize_mn_explore": "phase_explore",
        "_maybe_autosubmit_specialist_patches": "phase_explore",
        "_maybe_autosubmit_framework_config": "phase_explore",
        "_build_specialist_round_entry": "phase_explore",
        "_on_enter_framework": "phase_framework",
        "_pump_framework_agent_phase": "phase_framework",
        "_framework_agent_authoring_inflight": "phase_framework",
        "_framework_agent_audit_skip_confident": "phase_framework",
        "_framework_agent_roots_have_git": "phase_framework",
        "_audit_framework_agent_candidate": "phase_framework",
        "_framework_agent_audit_seed_lines": "phase_framework",
        "_record_framework_agent_audit_skip": "phase_framework",
        "_enqueue_framework_agent_authoring_specialist": "phase_framework",
        "_coerce_needs_gpu": "phase_framework",
        "_framework_gpu_params": "phase_framework",
        "_framework_authoring_lanes_ttl": "phase_framework",
        "_build_enablement_specialist_params": "phase_framework",
        "_read_enablement_source_context": "phase_framework",
        "_derive_checkpoint_weight_facts": "phase_framework",
        "_discover_enablement_candidate_refs": "phase_framework",
        "_maybe_enqueue_enablement_specialist": "phase_framework",
        "_maybe_record_enablement_human_review": "phase_framework",
        "_maybe_rearm_enablement": "phase_framework",
        "_framework_candidate_key": "phase_framework",
        "_framework_processed_candidate_keys": "phase_framework",
        "_unprocessed_framework_agent_candidates": "phase_framework",
        "_select_next_framework_agent_candidate": "phase_framework",
        "_select_best_framework_agent_candidate": "phase_framework",
        "_rank_framework_agent_candidates_llm": "phase_framework",
        "_match_framework_agent_candidate": "phase_framework",
        "_framework_agent_ranker_model": "phase_framework",
        "_framework_agent_ranker_client": "phase_framework",
        "_framework_known_candidate_ids": "phase_framework",
        "_framework_tried_refs": "phase_framework",
        "_build_framework_working_memory": "phase_framework",
        "_render_framework_memory_for_prompt": "phase_framework",
        "_framework_agent_discover_repo_urls": "phase_framework",
        "_write_prs_tested_from_framework_agent": "phase_framework",
        "_emit_kg_decision": "phase_framework",
        "_record_framework_agent_phase_done": "phase_framework",
        "_discover_next_framework_batch": "phase_framework",
        "_enqueue_framework_agent_task": "phase_framework",
        "_collect_framework_agent_candidate_priors": "phase_framework",
        "_submit_framework_agent_candidate_for_review": "phase_framework",
        "_materialize_framework_agent_candidate": "phase_framework",
        "_stamp_framework_progress": "phase_framework",
        "_record_framework_agent_critic_denied": "phase_framework",
        "_maybe_reauthor_from_critic_feedback": "phase_framework",
        "_pump_framework_agent_phase_safely": "phase_framework",
        "_pump_enablement_safely": "phase_framework",
        "_record_framework_agent_authored_outcome": "phase_framework",
        "_record_framework_agent_authoring_empty_outcome": "phase_framework",
        "_framework_agent_repo_url_origin_framework": "phase_framework",
        "_build_framework_config_grid": "phase_framework",
        "_framework_config_explore_params": "phase_framework",
        "_run_framework_config_exploration": "phase_framework",
        "_framework_config_exploration_inflight": "phase_framework",
        "_framework_config_max_rounds": "phase_framework",
        "_framework_config_new_variants": "phase_framework",
        "_finish_framework_config_lane": "phase_framework",
        "_framework_config_generation_context_lines": "phase_framework",
        "_dispatch_framework_config_generation_specialist": "phase_framework",
        "_framework_config_generation_inflight": "phase_framework",
        "_framework_config_grid_from_proposals": "phase_framework",
        "_ingest_framework_config_generation": "phase_framework",
        "_start_framework_config_generation": "phase_framework",
        "_framework_config_lane_should_engage": "phase_framework",
        "_maybe_hold_for_framework_config_lane": "phase_framework",
        "_record_framework_config_exploration_result": "phase_framework",
        "_orchestration_conversational": "conversation",
        "_reset_orchestration_conversation": "conversation",
        "_conversation_progress_signal": "conversation",
        "_attach_orchestration_context_tools": "conversation",
        "_context_inbox_reader": "conversation",
        "_context_recent_outcomes_reader": "conversation",
        "_context_analysis_reader": "conversation",
        "_record_reactor_conversation": "conversation",
        "_compose_prompt": "conversation",
        "_load_system_prompt": "conversation",
        "_inline_action_whitelist": "inline_actions",
        "_run_action_now_sync": "inline_actions",
        "_run_action_now": "inline_actions",
        "_plateau_advisory_block": "advisory",
        "_dominant_roofline_direction": "advisory",
        "_bottleneck_redirect_advisory_block": "advisory",
        "_acceptance_threshold_advisory_block": "advisory",
        "_target_gap_advisory_block": "advisory",
        "_current_primary_gap": "advisory",
        "_recent_proposed_variants": "advisory",
        "_priors_match_advisory_block": "advisory",
        "_resolve_issue_canonical": "proposals",
        "_workload_canonical_id": "proposals",
        "_gap_anchor_canonical_id": "proposals",
        "_read_local_recipe_row": "proposals",
        "_extract_kept_best_config": "proposals",
        "_kb_best_config_overrides_for_keep": "proposals",
        "_kb_amend_recipe": "proposals",
        "_inject_explore_runtime_params": "proposals",
        "_decaying_keep_threshold_pct": "proposals",
        "_materialize_approved_proposal": "proposals",
        "_record_proposal_task_map": "proposals",
        "_registry_lanes_ttl": "dispatcher",
        "_cycle_idem_suffix": "dispatcher",
        "_wait_for_task_terminal": "dispatcher",
        "_cursor_advance_to_latest": "dispatcher",
        "_dispatch_paused_for_phase_budget": "dispatcher",
        "_pump_dispatcher_once": "dispatcher",
        "_spawn_fitting_queued": "dispatcher",
        "_run_dispatched_with_gpu_release": "dispatcher",
        "_specialist_wall_budget_sec": "dispatcher",
        "_resolve_serving_tp": "dispatcher",
        "_gpu_lease_ttl_sec": "dispatcher",
        "_reap_dispatched_task": "dispatcher",
        "_lanes_fit": "dispatcher",
        "_target_analysis_baseline_exists": "gating",
        "_kernel_opt_keep_pending": "gating",
        "_sequence_denial_for_action": "gating",
        "_sequence_denial_for_request": "gating",
        "_skip_gemm_tuning": "gating",
        "_gemm_tuning_required_before_kernel_opt": "gating",
        "_emit_lifecycle": "writeback",
        "_record_policy_denied": "writeback",
        "_record_observation": "writeback",
        "_record_kernel_opt_partial": "writeback",
        "_record_integrate_keep": "writeback",
        "_is_promotable_result": "writeback",
        "_record_intervention_for_task": "writeback",
        "_handle_unpromotable_result": "writeback",
        "_source_session_id": "writeback",
        "_fact_write_hook": "writeback",
        "_ensure_journal": "writeback",
        "_pitfall_severity_for": "writeback",
        "_journal_entry_phase": "writeback",
        "_record_fact_per_task": "writeback",
        "_build_statement": "writeback",
        "_build_measured_impact": "writeback",
        "_predicted_gain": "writeback",
        "_record_fact_per_variant": "writeback",
        "_collect_workload_tags": "writeback",
        "_build_kernel_optimizations_from_state": "writeback",
        "_collect_attempt_provenance": "writeback",
        "_build_recipe_attrs_from_state": "writeback",
        "cortex_finalize_recipe_and_journal": "writeback",
        "_lift_to_current_best": "writeback",
        "_promote_to_shared_state": "writeback",
        "_detect_resume_state": "resume_helper",
        "replay_for_resume": "resume_helper",
        "_materialize_stack_config_for_resume": "resume_helper",
        "build_env_spec": "resume_helper",
        "_resume_consistency_pass": "resume_helper",
        "_resume_reenter_kernel_if_needed": "resume_helper",
        "_replay_keep_from_result": "resume_helper",
        "_resume_rollback_pending_integrate": "resume_helper",
        "_resume_recover_pending_integrate": "resume_helper",
        "_resume_recover_orphaned_keeps": "resume_helper",
        "_enqueue_internal_stack_rebench": "resume_helper",
        "_validate_geak_via_geak_harness": "resume_helper",
        "resumed_from": "resume_helper",
        "_replay_resume_if_needed": "resume_helper",
        "_maybe_run_maintenance_tick": "maintenance",
        "_maybe_prune_runs_for_disk": "maintenance",
        "_maybe_checkpoint_orchestration": "maintenance",
    }

    def __getattr__(self, name: str):
        # Only fires for genuinely-missing attributes (not shadowed instance
        # attrs / real methods). Delegate extracted collaborator methods to their
        # owner; everything else is a real AttributeError.
        owner = Coordinator._DELEGATED.get(name)
        if owner is not None:
            return getattr(getattr(self, owner), name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _collaborator(self, attr: str, factory):
        """Lazily build + cache a collaborator object (like ``router``/``recorder``);
        works for ``Coordinator.__new__`` test doubles too (uses ``__dict__``)."""
        obj = self.__dict__.get(attr)
        if obj is None:
            obj = factory(self)
            self.__dict__[attr] = obj
        return obj

    # Phase handlers, in call-chain order (machine -> prelude -> sweep ->
    # close -> internal -> kernel_stack -> kernel -> explore -> framework);
    # ``framework`` is last as it owns the most delegated methods (48).
    @property
    def phase_machine(self):
        from ..phases.machine import MachinePhase
        return self._collaborator("_phase_machine", MachinePhase)

    @property
    def phase_prelude(self):
        from ..phases.prelude import PreludePhase
        return self._collaborator("_phase_prelude", PreludePhase)

    @property
    def phase_sweep(self):
        from ..phases.sweep import SweepPhase
        return self._collaborator("_phase_sweep", SweepPhase)

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
    def phase_explore(self):
        from ..phases.explore import ExplorePhase
        return self._collaborator("_phase_explore", ExplorePhase)

    @property
    def phase_framework(self):
        from ..phases.framework import FrameworkPhase
        return self._collaborator("_phase_framework", FrameworkPhase)

    @property
    def conversation(self):
        from .conversation import ConversationCollaborator
        return self._collaborator("_conversation", ConversationCollaborator)

    @property
    def inline_actions(self):
        from .inline_actions import InlineActionsCollaborator
        return self._collaborator("_inline_actions", InlineActionsCollaborator)

    @property
    def advisory(self):
        from .advisory import AdvisoryCollaborator
        return self._collaborator("_advisory", AdvisoryCollaborator)

    @property
    def proposals(self):
        from .proposals import ProposalsCollaborator
        return self._collaborator("_proposals", ProposalsCollaborator)

    @property
    def dispatcher(self):
        from .dispatcher import DispatcherCollaborator
        return self._collaborator("_dispatcher", DispatcherCollaborator)

    @property
    def gating(self):
        from .gating import GatingCollaborator
        return self._collaborator("_gating", GatingCollaborator)

    @property
    def writeback(self):
        from .writeback import WritebackCollaborator
        return self._collaborator("_writeback", WritebackCollaborator)

    @property
    def resume_helper(self):
        from .resume import ResumeCollaborator
        return self._collaborator("_resume_helper", ResumeCollaborator)

    @property
    def maintenance(self):
        from .maintenance import MaintenanceCollaborator
        return self._collaborator("_maintenance", MaintenanceCollaborator)





    # Advisory disk guard: when the session partition runs low, LRU-trim the
    # bulkiest churn (per-task runs/ workspaces) while never touching durable
    # optimization state. Optimization wins live in state.json / journal /
    # reports, none of which this method removes.
    _DISK_FREE_MIN_GB: float = 20.0
    _DISK_USED_MAX_FRAC: float = 0.85
    _DISK_RUNS_KEEP_PER_ACTION: int = 50
    _STATE_JSON_WARN_BYTES: int = 50 * 1024 * 1024






    # Inline fast-action execution; deny report/session_breakdown (CLOSE artifacts).
    _INLINE_ACTION_DENY: frozenset[str] = frozenset(
        {
            "report",
            "session_breakdown",
        }
    )















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
                # Expected: we just cancelled these tasks; swallow the
                # cancellation so shutdown drains every task cleanly.
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













































    # Statuses that mean the candidate was ADOPTED (positive). Everything else
    # in framework_agent_phase_progress is a negative signal for the ranker.
    _FRAMEWORK_KEEP_STATUSES: frozenset[str] = frozenset({"kept"})

    # Max tried-candidate rows fed into the ranker/discovery working memory
    # (most-recent-first), to keep the prompt bounded.
    _FRAMEWORK_TRIED_MEMORY_CAP: int = 12









    _CRITIC_PRIORS_DECISION_TAIL: int = 5
    _CRITIC_PRIORS_OUTCOME_TAIL: int = 5






























    # Auto-roofline — PRELUDE bootstrap + 10% watermark refresh anchored on last_roofline_tput.
    _ROOFLINE_WATERMARK_RATIO: float = 1.10  # 10% step over last roofline
    # Relative-change floor for the pre-GEAK reprofile: any |cur-last|/last above
    # this re-runs profile+TraceLens. Tiny value (validated gain is rounded to 3
    # decimals) so it is effectively "any change", just absorbing float noise.
    _REPROFILE_CHANGE_TOL: float = 1e-5

    # Max re-author rounds per candidate on a needs_review verdict.
    _MAX_REAUTHOR_ATTEMPTS: int = 1

    # Backstop: max Critic-review submissions for a single candidate before
    # the pump force-stamps ``repeated_review_abort`` and stops re-selecting it.
    _MAX_REPEATED_REVIEW_SUBMISSIONS: int = 3

































    # CLOSE phase sequencer; class-level wait-for-task timeouts (overridable per-instance in tests).
    CLOSE_REPORT_TIMEOUT_SEC: float = 600.0
    CLOSE_SESSION_BREAKDOWN_TIMEOUT_SEC: float = 300.0
    CLOSE_NDJSON_DRAIN_TIMEOUT_SEC: float = 60.0
    # CLOSE step 0 post-opt roofline hard cap; on timeout the optimized snapshot
    # is skipped so report/breakdown always run.
    CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC: float = 600.0


    # optimization_stack actions that change kernel-level performance and thus
    # warrant a post-opt roofline: source-patch integrate plus GEMM tuning and
    # geak. Pure param-search (explore/sweep) is excluded.
    _POST_OPT_ROOFLINE_ACTIONS = frozenset(
        {"integrate", "integrate_patch", "gemm_tuning", "geak_e2e"}
    )






















    async def tick(self, n: int = 1) -> None:
        """Run exactly ``n`` reactor passes for every agent; dispatcher pumps at pass end, lazy resume replay on tick 1.

        Args:
            n: Number of full reactor+dispatcher passes to run (default 1).
        """
        await self._replay_resume_if_needed()
        for _ in range(n):
            self.shared_state.increment_tick()
            for name in self._tick_roles:
                await self._reactor_pass(name)
            await self._pump_dispatcher_once()
            # FRAMEWORK_AGENT phase pump: enqueue next candidate / fetch next batch. Best-effort.
            await self._pump_framework_agent_phase_safely(caller="tick")
            # Phase-independent enablement pump: repair a non-runnable combo
            # before it wedges the run in PRELUDE (see method docstring).
            await self._pump_enablement_safely(caller="tick")
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
        """Record a Coordinator-side exception without killing the session.

        Args:
            stage: The pipeline stage where the exception occurred.
            exc: The caught exception, recorded with type/message/traceback.
            tick: Optional tick number; defaults to the current SharedState
                tick when ``None``.
            agent: Optional agent role associated with the failure.
        """
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
        except Exception:  # noqa: BLE001
            log.exception("failed to persist Coordinator exception metadata")

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
        """Run reactor + dispatcher until a stop condition fires (priority order): signal, target_reached, time_exhausted (via closing phase), emergency, custom, max_ticks. Sets + saves + returns shared_state.stop_reason.

        Args:
            objective: Stop objective; ``None`` uses a :class:`TimeOnlyObjective`.
            max_minutes: Wall-clock budget in minutes; ``None``/falsy runs
                unbounded (capped at the container lifetime).
            tick_interval_sec: Sleep between ticks; ``0.0`` keeps tests fast.
            max_ticks: Optional hard cap on the number of ticks.
            stop_when: Optional custom predicate (sync or async) evaluated each
                tick; a truthy result stops the run.
            install_signal_handlers: Whether to install SIGINT/SIGTERM handlers
                that set the stop event.
            crash_emergency_threshold: Recent-crash count within the emergency
                window that triggers an emergency stop.
            closing_grace_sec: Grace window for the closing report phase;
                ``None`` derives a default from ``max_minutes``.

        Returns:
            The persisted ``shared_state.stop_reason`` describing why the run
            stopped.
        """
        objective = objective or TimeOnlyObjective()
        # Stash so _compose_prompt can update target_gap_pct.
        self._current_objective = objective
        grace_sec = effective_closing_grace_sec(max_minutes, closing_grace_sec)
        # Unbounded runs (max_minutes falsy) are capped at the container lifetime
        # so a long run always has a final wall-clock safety net; bounded runs
        # keep their explicit deadline unchanged.
        effective_minutes = max_minutes if max_minutes else _phase_state.DEFAULT_LONGRUN_MAX_MINUTES
        deadline = time.monotonic() + effective_minutes * 60.0
        self._run_started_monotonic = time.monotonic()
        self._run_deadline = deadline
        # Capture the live loop so the inline fast-action context tool can marshal coroutines back here.
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
                        # Orchestration checkpoint/compaction; cadence-based, no-op off conversational.
                        if not self._stop.is_set():
                            try:
                                await self._maybe_checkpoint_orchestration(
                                    tick=tick_n,
                                )
                            except Exception:  # noqa: BLE001
                                log.exception("Coordinator.run: orchestration checkpoint raised")
                    if not self._stop.is_set():
                        await self._pump_dispatcher_once()
                    # FRAMEWORK_AGENT phase pump: see ``tick()`` for rationale.
                    if not in_closing:
                        await self._pump_framework_agent_phase_safely(caller="run")
                        # Phase-independent enablement pump (see method docstring):
                        # repair a non-runnable combo stuck in PRELUDE.
                        await self._pump_enablement_safely(caller="run")
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
                    # Periodic reaper + DB retention (R5 + R4); cadence-gated, best-effort.
                    try:
                        await self._maybe_run_maintenance_tick(tick=tick_n)
                    except Exception:  # noqa: BLE001
                        log.exception("maintenance tick raised")
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
                if deadline is not None and time.monotonic() >= deadline and not in_closing:
                    if grace_sec <= 0:
                        stop_reason = "time_exhausted"
                        break
                    closing_deadline = await self._enter_closing_phase(
                        grace_sec=grace_sec,
                    )
                    continue
                if in_closing:
                    report_terminal = await self._closing_report_terminal()
                    grace_blown = closing_deadline is not None and time.monotonic() >= closing_deadline
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
                        stop_reason = "signal"
                        break
                    except asyncio.TimeoutError:
                        # Normal path: no stop signal within the tick interval —
                        # fall through and run the next tick.
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
                "Coordinator.run: stopped tick=%d reason=%s baseline_tput=%.1f cumulative_gain=%.2f%% max_minutes=%.0f",
                tick_n,
                stop_reason or "unknown",
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
                    # Signal handlers are unsupported off the main thread / on
                    # some platforms; teardown is best-effort, so ignore.
                    pass
        return self.shared_state.stop_reason



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
        # Conversation-growth accounting happens AFTER the turn returns, from the
        # backend's reported token usage (see below) — a delta-prompt char count
        # before the call badly undercounts the cached history in a persistent
        # conversation.
        sys_prompt = await self._load_system_prompt(agent_name)
        tools = self.policy.allowed_tools_for_agent(agent_name)
        # Stamp the timeline keys onto backends that self-write their trace row
        # (critic writes its own llm_calls row from inside run()). No-op for
        # backends without the hook. Best-effort: never block the turn.
        _set_trace_ctx = getattr(backend, "set_trace_context", None)
        if callable(_set_trace_ctx):
            try:
                _set_trace_ctx(
                    tick=int(self.shared_state.tick or 0),
                    phase=(self.shared_state.phase or "") or None,
                )
            except Exception:  # noqa: BLE001
                pass
        # max_turns=0 → backend default; ClaudeBackend needs ≥2 for tool_use→tool_result→final-text.
        _t0 = time.perf_counter()
        try:
            result: BackendTurnResult = await backend.run(
                prompt=prompt,
                system_prompt=sys_prompt,
                tools=tools,
                max_turns=0,
            )
        except BackendError as exc:
            await self._record_observation(
                "coordinator",
                "observation",
                {"kind": "backend_error", "agent": agent_name, "error": repr(exc)},
            )
            await self._track_backend_error_streak(agent_name, exc)
            return
        except NoIntentEmitted as exc:
            # No parseable intents; surface as observation so the next tick self-corrects instead of killing the run.
            await self._record_observation(
                "coordinator",
                "observation",
                {"kind": "no_intent_emitted", "agent": agent_name, "error": str(exc)[:500]},
            )
            return
        except Exception as exc:  # noqa: BLE001
            # Catch-all so one agent's bad turn never stops the loop (repeated crashes → emergency stop).
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
        # Reset the streak — a successful turn proves the backend is alive again.
        if self._backend_error_streak.get(agent_name):
            self._backend_error_streak[agent_name] = 0
            self._backend_error_alarm_armed[agent_name] = True
        # Record this reactor turn's token spend on the
        # unified ledger. One call site covers every in-process reactor
        # role (orchestration / kernel) whose backend reports usage on
        # metadata (ClaudeBackend + CodexBackend). Best-effort: a trace
        # failure must never affect intent routing.
        latency_ms = int((time.perf_counter() - _t0) * 1000)
        self._trace_reactor_llm_call(agent_name, result, latency_ms=latency_ms)
        # Full-trace (conversations): persist the full, redacted
        # prompt+response for this reactor turn. Separate file from the
        # token ledger; same best-effort posture.
        self._record_reactor_conversation(agent_name, result)
        # Context-token water level: the persistent conversation's true size
        # is the backend's reported input usage (input + cache_read +
        # cache_creation). Fall back to a full-turn char accumulation only when
        # the backend reports no usage (e.g. codex).
        if agent_name == "orchestration" and self._orchestration_conversational():
            try:
                md = getattr(result, "metadata", None) or {}
                it = md.get("input_tokens")
                cr = md.get("cache_read_input_tokens")
                cc = md.get("cache_creation_input_tokens")
                if it is not None or cr is not None or cc is not None:
                    self._checkpoint_tracker.set_context_tokens(
                        int(it or 0) + int(cr or 0) + int(cc or 0)
                    )
                else:
                    self._checkpoint_tracker.chars_add(
                        len(prompt) + len(getattr(result, "raw_text", "") or "")
                    )
            except Exception:  # noqa: BLE001 — accounting must never break routing
                pass
        # Completed orchestration turn means SEED delivered; flip flag so later turns send DELTA.
        if agent_name == "orchestration":
            self._orchestration_seeded = True
        for intent in result.intents:
            await self._handle_intent(agent_name, intent)

    def _trace_reactor_llm_call(
        self,
        agent_name: str,
        result: BackendTurnResult,
        *,
        latency_ms: int | None = None,
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

        Args:
            agent_name: The reactor role; doubles as trace component and role.
            result: The backend turn result whose metadata carries token
                counters.
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
                latency_ms=latency_ms,
            )
            append_llm_call(session_dir=self.session_dir, record=record)
        except Exception:  # noqa: BLE001 — trace must never break the loop
            log.debug(
                "full-trace: reactor llm_call append failed for %s",
                agent_name,
                exc_info=True,
            )


    async def _track_backend_error_streak(
        self,
        agent_name: str,
        exc: BackendError,
    ) -> None:
        """Increment the per-agent ``BackendError`` streak; emit one backend_unhealthy event on crossing the threshold (re-arms only after a successful turn).

        Args:
            agent_name: The agent role whose error streak is incremented.
            exc: The backend error; its repr is included in the emitted event.
        """
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
                        "backend (e.g. --robustness-mock / --critic-mock) "
                        "while the underlying transport is repaired"
                    ),
                },
            )







































    # Multi-node only: cap on how many specialist proposal_set entries
    # are auto-materialised into a single explore grid per specialist
    # round. Keeps the deterministic bridge from flooding the action
    # queue when an LLM specialist returns a large proposal_set.
    _MN_AUTO_EXPLORE_GRID_CAP = 6






























    # Phases whose long, serially-drained GPU grids must not starve the
    # per-phase cyclic budget exit. PRELUDE (baseline/roofline bootstrap),
    # SWEEP, CLOSE, RECOVER own mandatory work and keep draining normally.
    _BUDGET_GATED_DISPATCH_PHASES: frozenset[str] = frozenset({"EXPLORE", "KERNEL_AGENT", "FRAMEWORK_AGENT"})












    # Fact-write surface — journal + direct KB lesson/pitfall/recipe writes (the fact side of KB integration).
    PITFALL_REGRESS_THRESHOLD_PCT: float = -5.0  # gain_pct ≤ this → pitfall

















__all__ = [
    "Coordinator",
    "CoordinatorState",
    "PendingProposal",
    "SharedState",
    # Re-exported from coordinator_helpers for callers/tests that reference
    # them via ``coordinator.<name>``. Declared so the re-export is
    # intentional rather than a flagged unused import.
    "_BASELINE_FINGERPRINT_KEYS",
    "_baseline_params_fingerprint",
    "_dedupe_extra_server_args",
    "_infer_model_class_from_config",
    "_merge_cumulative_extra_sglang_args",
    "_parse_baseline_workload_extra",
    "_parse_iso_unix",
    "_resolve_roofline_watermark_ratio",
    "effective_closing_grace_sec",
    # Re-exported from policy.gate; referenced via ``coordinator.<name>`` in
    # tests. Declared so the re-export is intentional, not a flagged import.
    "SPECIALIST_FROM_AGENT_PREFIX",
]
