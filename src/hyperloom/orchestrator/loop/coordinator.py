# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator main loop and runtime protocol manager."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from hyperloom.orchestrator.actions.executors._grid_server_args import (
    tokenize_server_args_preserving_json,
)
from hyperloom.orchestrator.knowledge.config import KnowledgeConfig, KnowledgeStoreMode
from hyperloom.orchestrator.knowledge.recipe_kb import RecipeKB

# Recipe snapshot severity tags (schema has no fixed enum).
_SEVERITY_CRASH: str = "crash"
_SEVERITY_REGRESS: str = "regress"

# Bounded transient-failure auto-retry for specialist dispatches (infra-only).
SPECIALIST_AUTO_RETRY_MAX: int = 2

# Periodic in-process maintenance/reaper cadence (lease reaping + DB retention),
# in coordinator ticks.
MAINTENANCE_EVERY_TICKS: int = 50

# Default per-macro-cycle wall-clock window (hours) in cyclic mode.
DEFAULT_CYCLE_HOURS: float = 24.0
# Trailing window for the crash-rate emergency stop, in seconds.
_CRASH_EMERGENCY_WINDOW_SEC: float = 24.0 * 3600.0
# Combined baseline-failure backstop: fast-fail after this many TOTAL baseline failures.
_BASELINE_MAX_TOTAL_FAILURES: int = 3
# Enablement stall cap: consecutive enablement rounds that neither made the combo
# runnable nor advanced to a NEW failure signature; reaching it stops the loop
# with ``enablement_stalled``. A progressing round resets the streak.
_ENABLEMENT_MAX_STALL: int = 5
# Unified authored-lane max attempts (apply-failure retries + Critic reauthor).
_AUTHORED_LANE_MAX_ATTEMPTS: int = 3
# Floor on the per-repo framework-PR discover timeout.
_FRAMEWORK_MIN_PER_REPO_TIMEOUT_SEC: float = 30.0
# Default min TRANSFER confidence a warm-replay champion must clear to be enqueued.
_DEFAULT_WARM_REPLAY_MIN_CONFIDENCE: float = 0.7
# Default resume-drift floor (%): a re-measured current_best below this fraction
# of its recorded tput is flagged as drift.
_DEFAULT_RESUME_DRIFT_FLOOR_PCT: float = 95.0
from ..phases import machine_state as _phase_state
from ..state.failure_evidence import UNMEASURED_OUTCOMES, render_failure_line
from ..state.optimization_journal import Journal
from hyperloom.inference_optimizer.session.paths import db_path_for
from hyperloom.inference_optimizer.protocol.action_surfaces import ACTION_CATALOGUE, ActionMetadata
from ..roles.agent_role import AgentRole, default_role_registry
from ..roles.base import Backend, BackendError, BackendTurnResult, LLMCallFailed
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
from ..state.shared_state import SharedState, effective_closing_grace_sec, timed_teardown_step
from .intent_router import IntentRouter
from .sub_agent_runner import SubAgentRunner
from ..state.task_registry import TaskRegistry
from ..trace.llm_trace import LLMCallRecord, append_llm_call
from hyperloom.common.prompt_safety import defang_prompt_structure as _defang_prompt_structure
from hyperloom.common.prompt_safety import flatten_for_prompt as _flatten_for_inbox
from ..trace.orchestration_trace import (
    write_mcp_setup_once,
)
from .coordinator_helpers import (
    _infer_model_class_from_config,
    format_exc_brief,
    serialize_verdict_advisory,
)


log = logging.getLogger(__name__)


# Audit-trail kinds (must match shared_state._AUDIT_ACTIONS).
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

    A FRAMEWORK_AGENT specialist may deliver a non-diff tuned artifact via
    ``artifacts_written`` instead of a source patch; it flows through the
    ``integrate_patch`` artifact-install channel. This is the shared routable
    signal used by both the autosubmit and empty-outcome bridges: an artifact is
    routable only when its ``source`` resolves to a real file inside one of
    ``resolve_bases`` (matching integrate_patch's sandbox, rejecting ``..``
    escapes). Only the ``source`` is validated; a bad ``target`` is rejected
    downstream by integrate_patch rather than skip-stamped here.

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
        # An absolute ``source`` is checked as-is; a relative one is resolved
        # under each base. In both cases the resolved path must be a real file
        # inside a base (rejecting ``..`` escapes). Scan all bases and take the
        # first that yields a contained real file.
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


def _framework_config_levers_from_done(
    done_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    """Extract a config-lever set from a FRAMEWORK specialist deliverable.

    A specialist may translate an upstream PR into a config win (serving flags /
    env vars) instead of a source patch. Levers are read from the first
    ``proposal_set`` entry carrying ``extra_args`` and/or ``extra_envs`` while
    preserving the separate server-argument and environment channels.

    Args:
        done_payload: The specialist ``specialist_done`` payload (already
            unwrapped of any envelope).

    Returns:
        dict[str, Any]: The server arguments and environment overrides, or
            ``{}``.
    """
    if not isinstance(done_payload, dict):
        return {}
    # A patch deliverable takes precedence.
    patches = done_payload.get("patches_written") or []
    if isinstance(patches, list) and patches:
        return {}
    proposals = done_payload.get("proposal_set") or []
    if not isinstance(proposals, list):
        return {}
    for entry in proposals:
        if not isinstance(entry, dict):
            continue
        extra_envs: dict[str, str] = {}
        envs = entry.get("extra_envs")
        if isinstance(envs, dict):
            for k, v in envs.items():
                key = str(k).strip()
                if key:
                    extra_envs[key] = str(v)
        args = entry.get("extra_args")
        extra_server_args = ""
        if isinstance(args, str) and args.strip():
            parsed_args = tokenize_server_args_preserving_json(args)
            if parsed_args is None:
                log.warning(
                    "FRAMEWORK config lever %r has server args unsupported by "
                    "Magpie's unquoted argv transport; dropping the args%s",
                    entry.get("name"),
                    " while preserving its environment overrides" if extra_envs else "",
                )
                if not extra_envs:
                    continue
            else:
                extra_server_args = parsed_args[0]
        elif isinstance(args, (list, tuple)):
            arg_tokens = [str(a) for a in args if str(a).strip()]
            if any(any(ch.isspace() for ch in token) for token in arg_tokens):
                log.warning(
                    "FRAMEWORK config lever %r has a whitespace-bearing argv token; dropping the args%s",
                    entry.get("name"),
                    " while preserving its environment overrides" if extra_envs else "",
                )
                if not extra_envs:
                    continue
            else:
                parsed_args = tokenize_server_args_preserving_json(" ".join(arg_tokens))
                if parsed_args is None:
                    log.warning(
                        "FRAMEWORK config lever %r has unparseable server args; dropping the args%s",
                        entry.get("name"),
                        " while preserving its environment overrides" if extra_envs else "",
                    )
                    if not extra_envs:
                        continue
                else:
                    extra_server_args = parsed_args[0]
        if extra_server_args or extra_envs:
            return {
                "extra_server_args": extra_server_args,
                "extra_envs": extra_envs,
            }
    return {}


# Hard-trigger thresholds: optimisation rounds a domain may go without a specialist
# dispatch / a KEEP before the Coordinator force-dispatches one.
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
_OUTCOME_STATUS_KEYS: tuple[str, ...] = ("status", "verdict", "outcome", "runner_status")
# Notes rendered per inbox line.
_OUTCOME_NOTES_MAX: int = 3


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


def _defang_alert_payload(value: Any) -> Any:
    """Recursively defang string leaves of an alert payload (keys untouched).

    Only string *values* are neutralised so dict structure / keys that
    downstream rendering relies on are preserved; numbers/bools pass through.
    """
    if isinstance(value, str):
        return _defang_prompt_structure(value)
    if isinstance(value, dict):
        return {k: _defang_alert_payload(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_defang_alert_payload(v) for v in value]
    return value


def _format_inbox_event(m: "Message", *, max_variant_rows: int = 3) -> str:
    """Render one inbox ``Message`` as a compact, high-signal line.

    Args:
        m: The inbox message to render; its topic selects a per-topic
            formatting branch (delegated_result, policy_denial, review_verdict,
            observation) with a generic fallback.
        max_variant_rows: Cap on the indented per-variant failure lines
            appended to a ``delegated_result``; 0 suppresses them.

    Returns:
        The header line, plus one indented continuation line per rendered
        variant failure.
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
        notes: list[Any] = []
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
            # Executors that never raise report the failure inside the result
            # envelope, leaving the top-level error None.
            if not error:
                error = result.get("error")
            raw_notes = result.get("notes")
            if isinstance(raw_notes, list):
                notes = [n for n in raw_notes if n][:_OUTCOME_NOTES_MAX]
        if error:
            parts.append(f"error={str(error)[:200]!r}")
        if notes:
            shown = "; ".join(str(n) for n in notes)
            parts.append(f"notes={shown[:300]!r}")
        header_line = " ".join(parts)
        if max_variant_rows <= 0 or not isinstance(result, dict):
            return header_line
        pvos = result.get("per_variant_outcomes")
        if not isinstance(pvos, list):
            return header_line
        failures = [
            v for v in pvos if isinstance(v, dict) and str(v.get("outcome") or "").upper() in UNMEASURED_OUTCOMES
        ]
        if not failures:
            return header_line
        lines = [header_line]
        for vo in failures[:max_variant_rows]:
            row = dict(vo)
            row["error_excerpt"] = _flatten_for_inbox(vo.get("error_excerpt") or vo.get("reason") or "")
            lines.append("  failure: " + render_failure_line(row, excerpt_chars=120))
        elided = len(failures) - max_variant_rows
        if elided > 0:
            lines.append(f"  (+{elided} more failures; pull get_variant_failures)")
        return "\n".join(lines)

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

    if topic == "alert":
        # Alert payloads can embed attacker-influenceable server.log excerpts;
        # defang string leaves so a log line can't inject prompt structure.
        # Only the alert topic is treated this way — proposal/other topics keep
        # their exact payload so the Critic inbox parser still literal_evals it.
        return f"{head} payload={_defang_alert_payload(payload)}"

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


# Path-like keys surfaced from a kernel handler payload/result so operators can
# see where a step's artifacts went.
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
    """Extract present, non-empty path-like fields from a kernel handler
    payload or result dict. A non-dict argument yields an empty mapping.

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
    """Class-level delegation for extracted collaborator methods.

    Instance access is handled by ``Coordinator.__getattr__``; class access
    resolves here to the owning collaborator class's function. Collaborator
    modules are imported lazily to avoid an import cycle.
    """

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

    # property name -> (module, collaborator class) for class-level delegation.
    # Kept in sync with the lazy collaborator properties + ``_DELEGATED``.
    _COLLAB_MODULES = {
        # Phase handlers in call-chain order. Module paths are relative to
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
        "gpu_lanes": ("gpu_lanes", "GpuLanes"),
        "enablement_params": ("enablement.params", "EnablementParams"),
        "enablement_lane": ("enablement.lane", "EnablementLane"),
        "enablement_build": ("enablement.build", "EnablementBuild"),
        "enablement_revalidation": ("enablement.revalidation", "EnablementRevalidation"),
        "router": ("loop.intent_router", "IntentRouter"),
        "maintenance": ("loop.maintenance", "MaintenanceCollaborator"),
        "build_lifecycle": ("loop.build_lifecycle", "BuildLifecycleCollaborator"),
        "writeback": ("loop.writeback", "WritebackCollaborator"),
        "dispatcher": ("loop.dispatcher", "DispatcherCollaborator"),
        "proposals": ("loop.proposals", "ProposalsCollaborator"),
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
        self.role_registry = role_registry or default_role_registry()
        # KnowledgePlane owns RecipeKB. Keep the explicit parameter as a
        # compatibility injection path for library callers during Phase 1.
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
        # Lifecycle save debounce: terminal events flush immediately; bursty
        # non-terminal markers coalesce within a short window.
        # ``_lifecycle_last_save`` is a monotonic timestamp.
        self._lifecycle_last_save: float = 0.0
        self._lifecycle_save_min_interval_s: float = 2.0
        # Thread live SharedState into the runner so executors get it via
        # ctx.extra.
        self.sub.shared_state = self.shared_state
        # Serving-disjoint invariant: the live serving process holds the first
        # ``serving_tp`` cards, carved off the specialist pool. ``shared_state.tp``
        # is restored on resume; the ``TP`` env is the fresh-start fallback.
        self.gpu_specialist_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_gpu_specialist_devices(
                int(getattr(self.shared_state, "gpu_specialist_capacity", 0) or 0),
                serving_tp=self._resolve_serving_tp(),
            ),
        )
        # Framework-authoring pool over the whole node. Serialized against
        # ``gpu_specialist_pool`` via the cap-1 ``gpu_research_lane`` mutex.
        self.framework_gpu_pool = SpecialistGpuPool(
            self.db,
            gpu_ids=resolve_whole_machine_devices(),
        )
        # Dispatcher re-scan poll cadence: re-scan the queue while awaiting
        # in-flight tasks so a queued GPU task starts the moment its lane frees.
        self._dispatcher_poll_sec = 10.0
        # Sync research_lane capacity into lane_capacity so acquire_many honours the cap.
        try:
            from ..bus.storage.schema import set_lane_capacity as _set_lane_capacity

            cap = int(self.shared_state.research_lane_capacity or 0)
            if cap >= 0:
                _set_lane_capacity(self.db.raw, "research_lane", cap)
        except Exception:  # noqa: BLE001 — non-fatal; default seed wins
            log.exception("failed to sync research_lane_capacity to leases DB")
        # gpu_research_lane stays capacity-1 (strictly serial GPU specialists);
        # the GPU pool partitions physical cards within that one lease.
        # `strict_paths` defers to the env flag.
        self.policy = PolicyGate(
            role_registry=self.role_registry,
            session_dir=self.session_dir,
            shared_state=self.shared_state,
        )
        self.sub.policy = self.policy
        # Attach read-only context-pull MCP tools to Orchestration backend.
        self._attach_orchestration_context_tools()
        # Resume detection must run before any boot-time state.json write.
        self._resumed_from = self._detect_resume_state()
        # Reap serving processes orphaned by a prior monitor-process crash
        # (e.g. a raylet death that took the optimizer down mid-benchmark),
        # scoped strictly to this session's own pidfiles. No-op on a fresh
        # session; on resume it clears a leftover SGLang/vLLM server so it
        # cannot pollute the shared benchmark port. Best-effort; never fatal.
        self._reap_orphaned_servers_best_effort()
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

        # Context-token guardrail: derive soft/hard budgets from the
        # orchestration model's window × fraction (env-overridable). 0 budgets
        # disable token triggers (char/tick/time cadence still applies).
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
        self._checkpoint_policy = _orch_mem.CheckpointPolicy(
            context_token_soft=int(_ctx_window * _soft_frac),
        )
        # Kept so a provider that reports its own window per turn can replace the
        # table's guess without re-deriving the operator's fraction.
        self._checkpoint_soft_fraction = _soft_frac
        self._checkpoint_tracker = _orch_mem.CheckpointTracker(
            last_phase=str(getattr(self.shared_state, "phase", "") or ""),
        )
        # Consecutive degenerate checkpoint replies; resets on a good one.
        self._consec_degenerate_ckpt: int = 0
        # Disable checkpointing entirely via env.
        self._checkpoint_enabled: bool = os.environ.get(
            "INFERENCE_OPTIMIZER_DISABLE_ORCH_CHECKPOINT",
            "",
        ).strip().lower() not in {"1", "true", "yes", "on"}
        # Seed memory rendered into the next full SEED push (resume recovery).
        # INFERENCE_OPTIMIZER_ORCH_MEMORY_ROLLBACK=<n> re-seeds from the
        # n-th-from-newest history snapshot instead of the live memory.
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
                        "Coordinator: orchestration memory rolled back to history[-%d] (of %d snapshots)",
                        _n,
                        len(_hist),
                    )
                else:
                    log.warning(
                        "Coordinator: ORCH_MEMORY_ROLLBACK=%s out of range (history has %d); using live memory",
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

        # Pin a per-macro-cycle budget window so per-phase budget fractions
        # apply per cycle. Only takes effect for long/unbounded runs; short
        # bounded runs stay anchored on the whole session.
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

        # Medium-intensity soft restart at each macro-cycle boundary. On by
        # default in cyclic mode; opt out via the env flag.
        self._cycle_soft_restart: bool = os.environ.get(
            "INFERENCE_OPTIMIZER_DISABLE_CYCLE_SOFT_RESTART",
            "",
        ).strip().lower() not in {"1", "true", "yes", "on"}
        # The soft restart's inference-server deep-clean kills lingering server
        # processes; separately gated, defaults ON within the soft restart.
        self._cycle_restart_servers: bool = os.environ.get(
            "INFERENCE_OPTIMIZER_DISABLE_CYCLE_SERVER_RESTART",
            "",
        ).strip().lower() not in {"1", "true", "yes", "on"}

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
        _CANONICAL_ORDER = ("orchestration", "critic", "robustness")
        self._tick_roles: tuple[str, ...] = tuple(r for r in _CANONICAL_ORDER if r in self.role_registry)

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
        # Closing-grace bound; used only while ``closing_phase`` is set so CLOSE
        # work is not skipped just because the session deadline has passed.
        self._closing_deadline: float | None = None
        # Set while a success terminal (a met objective) is being routed into
        # CLOSE. Distinct from ``closing_phase``, which means the wall clock ran
        # out and the sequencer should shed expensive work; a met target has to
        # produce the full set of artifacts. ``True`` lifts the session bound for
        # that routing without claiming a rescue is under way.
        self._terminal_closing: bool = False
        # Latest objective wired by run(); refreshes target_gap_pct each tick. None outside a run.
        self._current_objective: Objective | None = None

        # Initialise phase machine (fresh session enters PRELUDE). Idempotent.
        self._ensure_phase_initialised()
        # Recipe KB T0 defensive fallback for direct SDK/test callers; best-effort.
        self._ensure_recipe_kb_t0_anchored()

    @property
    def router(self) -> IntentRouter:
        """Intent routing collaborator (extracted from this class).

        The ``_handle_*`` intent handlers live on :class:`IntentRouter`; the
        methods remaining here are thin forwarding shims. Built lazily and
        cached.
        """
        r = self.__dict__.get("_router")
        if r is None:
            r = IntentRouter(self)
            self.__dict__["_router"] = r
        return r

    # Methods extracted into collaborator objects are delegated back by name here
    # (symmetric to each collaborator's
    # ``__getattr__`` back to this coordinator). ``coord.foo`` / ``self.foo`` /
    # ``self._coord.foo`` all keep resolving, and instance-attr monkeypatches
    # still shadow them (normal lookup wins over __getattr__). Each value is the
    # name of the property returning the owning collaborator.
    _DELEGATED = {
        # router
        "_handle_intent": "router",
        "_handle_propose_action": "router",
        "_handle_review_verdict": "router",
        "_handle_single_verdict": "router",
        "_handle_delegate": "router",
        "_handle_request": "router",
        "_handle_response": "router",
        "_handle_extend_lease": "router",
        "_deliver_specialist_inbox": "router",
        "_handle_prune_branch": "router",
        "_handle_escalate_strategy_change": "router",
        "_handle_send_message": "router",
        "_handle_alert": "router",
        "_handle_update_state": "router",
        # recorder (folded into writeback)
        "_aggregate_research_evidence": "writeback",
        "_harvest_research_scout": "writeback",
        "_record_specialist_result": "writeback",
        "_drain_queued_baselines": "writeback",
        # Phase handlers, grouped in the same call-chain order as
        # _COLLAB_MODULES/the @property block above:
        # machine -> prelude -> sweep -> close -> internal -> kernel_stack ->
        # kernel -> explore -> framework (framework last: largest cluster).
        "_ensure_phase_initialised": "phase_machine",
        "_ensure_recipe_kb_t0_anchored": "phase_machine",
        "_kernel_enabled": "phase_machine",
        "_optimize_enabled": "phase_machine",
        "_advance_phase_if_needed": "phase_machine",
        "_on_phase_entered": "phase_machine",
        "_reseed_orch_prompt_for_phase": "phase_machine",
        "_record_phase_entry_evidence": "phase_machine",
        "_internal_analysis_kind": "phase_prelude",
        "_measured_analysis_cost_sec": "phase_prelude",
        "_record_prelude_arm_dropped": "phase_prelude",
        "_warm_recipe_proven_items": "phase_prelude",
        "_inject_warm_recipe_history_into_ledger": "phase_prelude",
        "_maybe_enqueue_warm_replay": "phase_prelude",
        "_promote_warm_replay": "phase_prelude",
        "_maybe_enqueue_prelude_initial_analysis_after_baseline": "phase_prelude",
        "_enqueue_internal_analysis_task": "phase_prelude",
        "_on_enter_sweep": "phase_sweep",
        "_enqueue_internal_conc_sweep_task": "phase_sweep",
        "_enqueue_internal_sweep_task": "phase_sweep",
        "_build_sweep_params_from_recipe": "phase_sweep",
        "_record_session_budget_conc_sweep_skip": "phase_sweep",
        "_record_terminal_conc_sweep_skip": "phase_sweep",
        "_derive_close_stop_reason": "phase_close",
        "_session_integrated_kernel_patch": "phase_close",
        "_maybe_run_close_post_opt_roofline": "phase_close",
        "_drain_geak_rebench_for_close": "phase_close",
        "_on_enter_close": "phase_close",
        "_enqueue_runnable_internal_task": "phase_close",
        "_enqueue_internal_report_task": "phase_close",
        "_enqueue_internal_session_breakdown_task": "phase_close",
        "_run_close_task": "phase_close",
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
        "_collective_required_before_kernel_opt": "phase_kernel",
        "_on_enter_kernel": "phase_kernel",
        "_resolve_bench_protocol": "phase_kernel",
        "_geak_timeouts": "phase_kernel",
        "_run_geak_kernel_phase": "phase_kernel",
        "_geak_win_already_recorded": "phase_kernel",
        "_parse_geak_accepted_config": "phase_kernel",
        "_record_geak_candidate": "phase_kernel",
        "_promote_geak_from_candidate": "phase_kernel",
        "_record_geak_kernel_journey": "phase_kernel",
        "_ck_blockscale_switch_eligible": "phase_kernel",
        "_ck_switch_precision_is_fp8": "phase_kernel",
        "_handle_gemm_tuning_result": "phase_kernel",
        "_sync_profile_state_after_gemm_roofline": "phase_kernel",
        "_journal_gemm_tuning_keep": "phase_kernel",
        "_replace_latest_gemm_tuning_attempt": "phase_kernel",
        "_gemm_e2e_candidates": "phase_kernel",
        "_validate_gemm_tuning_e2e": "phase_kernel",
        "_kernel_opt_work_remains": "phase_kernel",
        "_run_kernel_opt_nomination": "phase_kernel",
        "_current_tput_from_validated_gain": "phase_kernel",
        "_last_measured_roofline_tput": "phase_kernel",
        "_needs_roofline_for_watermark": "phase_kernel",
        "_maybe_enqueue_watermark_roofline": "phase_kernel",
        "_cached_kernel_request": "phase_kernel",
        "_negative_ledger_domain_counts": "phase_explore",
        "_plan_cycle_focus": "phase_explore",
        "_record_cycle_strategy_for_current_cycle": "phase_explore",
        "_cycle_strategy_seed_block": "phase_explore",
        "_cycle_directive_fallback": "phase_explore",
        "_reseed_orch_prompt_for_cycle": "phase_explore",
        "_apply_macro_cycle_reloop": "phase_explore",
        "_run_cycle_soft_restart": "phase_explore",
        "_restart_inference_servers": "phase_explore",
        "_on_cycle_start_reprofile": "phase_explore",
        "_maybe_force_stalled_domain_specialist": "phase_explore",
        "_seed_gaps_from_research_hints": "phase_explore",
        "_fan_out_specialist_wave": "phase_explore",
        "_maybe_auto_retry_specialist": "phase_explore",
        "_record_specialist_retry_exhausted": "phase_explore",
        "_warm_specialist_params": "phase_explore",
        "_refresh_gaps": "phase_explore",
        "_extract_gaps_from_baseline": "phase_explore",
        "_extract_gaps_from_attempts": "phase_explore",
        "_gap_layer_for_action": "phase_explore",
        "_record_explore_round_gaps": "phase_explore",
        "_record_explore_variant_failures": "phase_explore",
        "_task_id_from_specialist_source": "phase_explore",
        "_maybe_materialize_mn_explore": "phase_explore",
        "_maybe_autosubmit_specialist_patches": "phase_explore",
        "_maybe_autosubmit_framework_config": "phase_explore",
        "_build_specialist_round_entry": "phase_explore",
        "_on_enter_framework": "phase_framework",
        "_pump_framework_agent_phase": "phase_framework",
        "_framework_agent_authoring_inflight": "phase_framework",
        "_enqueue_framework_agent_authoring_specialist": "phase_framework",
        "_coerce_needs_gpu": "gpu_lanes",
        "_framework_gpu_params": "gpu_lanes",
        "_framework_authoring_lanes_ttl": "gpu_lanes",
        "_build_enablement_specialist_params": "enablement_params",
        "_read_enablement_source_context": "enablement_params",
        "_derive_checkpoint_weight_facts": "enablement_params",
        "_discover_enablement_candidate_refs": "enablement_params",
        "_maybe_enqueue_enablement_specialist": "enablement_lane",
        "_maybe_record_enablement_human_review": "enablement_lane",
        "_enablement_in_flight": "enablement_lane",
        "_maybe_rearm_enablement": "enablement_lane",
        "_maybe_escalate_to_targeted_build": "enablement_build",
        "_maybe_enqueue_specialist_requested_build": "enablement_build",
        "_maybe_route_build_outcomes": "enablement_build",
        "_route_succeeded_build": "enablement_build",
        "_route_failed_build": "enablement_build",
        "_build_routing_record": "enablement_build",
        "_note_build_routed": "enablement_build",
        "_build_probe_was_cancelled": "enablement_build",
        "_enqueue_build_launch_probe": "enablement_build",
        "_maybe_rearm_authored_lane": "phase_framework",
        "_enqueue_author_specialist": "phase_framework",
        "_drain_apply_fail_retry_pending": "phase_framework",
        "_framework_candidate_key": "phase_framework",
        "_framework_processed_candidate_keys": "phase_framework",
        "_unprocessed_framework_agent_candidates": "phase_framework",
        "_select_next_framework_agent_candidate": "phase_framework",
        "_framework_known_candidate_ids": "phase_framework",
        "_framework_tried_refs": "phase_framework",
        "_build_framework_working_memory": "phase_framework",
        "_framework_agent_discover_repo_urls": "phase_framework",
        "_record_framework_agent_phase_done": "phase_framework",
        "_enqueue_framework_agent_task": "phase_framework",
        "_collect_framework_agent_candidate_priors": "phase_framework",
        "_submit_framework_agent_candidate_for_review": "phase_framework",
        "_materialize_framework_agent_candidate": "phase_framework",
        "_stamp_framework_progress": "phase_framework",
        "_record_framework_agent_critic_denied": "phase_framework",
        "_maybe_reauthor_from_critic_feedback": "phase_framework",
        "_pump_framework_agent_phase_safely": "phase_framework",
        "_pump_enablement_safely": "enablement_lane",
        "_maybe_enqueue_enablement_baseline_revalidation": "enablement_revalidation",
        "_open_revalidation_row": "enablement_revalidation",
        "_open_row_past_spent_generations": "enablement_revalidation",
        "_record_framework_agent_authored_outcome": "phase_framework",
        "_recover_framework_agent_authoring_outcome": "phase_framework",
        "_record_framework_agent_authoring_empty_outcome": "phase_framework",
        "_record_framework_agent_dispatch_failure": "phase_framework",
        "_maybe_enqueue_candidate_discovery": "phase_framework",
        "_candidate_discovery_inflight": "phase_framework",
        "_ingest_candidate_discovery": "phase_framework",
        "_candidates_from_discovery_proposals": "phase_framework",
        "_orchestration_conversational": "conversation",
        "_orchestration_context_tools_mounted": "conversation",
        "_orchestration_needs_seed": "conversation",
        "_reset_orchestration_conversation": "conversation",
        "_conversation_progress_signal": "conversation",
        "_attach_orchestration_context_tools": "conversation",
        "_context_inbox_reader": "conversation",
        "_context_recent_outcomes_reader": "conversation",
        "_context_running_tasks_reader": "conversation",
        "_task_heartbeat_age_sec": "conversation",
        "_context_analysis_reader": "conversation",
        "_record_reactor_conversation": "conversation",
        "_compose_prompt": "conversation",
        "_load_system_prompt": "conversation",
        "_inline_action_whitelist": "dispatcher",
        "_run_action_now_sync": "dispatcher",
        "_run_action_now": "dispatcher",
        "_plateau_advisory_block": "conversation",
        "_dominant_roofline_direction": "conversation",
        "_bottleneck_redirect_advisory_block": "conversation",
        "_acceptance_threshold_advisory_block": "conversation",
        "_target_gap_advisory_block": "conversation",
        "_current_primary_gap": "conversation",
        "_recent_proposed_variants": "conversation",
        "_priors_match_advisory_block": "conversation",
        "_workload_canonical_id": "proposals",
        "_read_local_recipe_row": "proposals",
        "_extract_kept_best_config": "proposals",
        "_kb_best_config_overrides_for_keep": "proposals",
        "_kb_amend_recipe": "proposals",
        "_inject_explore_runtime_params": "proposals",
        "_materialize_approved_proposal": "proposals",
        "_record_proposal_task_map": "proposals",
        "_registry_lanes_ttl": "dispatcher",
        "_cycle_idem_suffix": "dispatcher",
        "_advance_rendered_cursor": "conversation",
        "_dispatch_paused_for_phase_budget": "dispatcher",
        "_pump_dispatcher_once": "dispatcher",
        "_spawn_fitting_queued": "dispatcher",
        "run_task_registered": "dispatcher",
        "_specialist_wall_budget_sec": "dispatcher",
        "_specialist_progress_publisher": "dispatcher",
        "_resolve_serving_tp": "dispatcher",
        "_gpu_lease_ttl_sec": "dispatcher",
        "_reap_dispatched_task": "dispatcher",
        "_account_dead_holder_failures": "dispatcher",
        "_lanes_fit": "dispatcher",
        "_sequence_denial_for_action": "dispatcher",
        "_time_budget_denial_for_action": "dispatcher",
        "_admission_denial_for_action": "dispatcher",
        "_sequence_denial_for_request": "dispatcher",
        "_skip_gemm_tuning": "dispatcher",
        "_gemm_tuning_required_before_kernel_opt": "dispatcher",
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
        "_record_fact_per_variant": "writeback",
        "_collect_workload_tags": "writeback",
        "_build_kernel_optimizations_from_state": "writeback",
        "_collect_attempt_provenance": "writeback",
        "_build_recipe_attrs_from_state": "writeback",
        "ensure_recipe_finalized": "writeback",
        "finalize_recipe_and_journal": "writeback",
        "_lift_to_current_best": "writeback",
        "_update_cumulative_gain_validated": "writeback",
        "_promote_to_shared_state": "writeback",
        "_should_run_prelude_bootstrap": "writeback",
        "_detect_resume_state": "writeback",
        "replay_for_resume": "writeback",
        "_current_best_launch_config": "writeback",
        "build_env_spec": "writeback",
        "_resume_consistency_pass": "writeback",
        "_resume_reenter_kernel_if_needed": "writeback",
        "_replay_keep_from_result": "writeback",
        "_resume_rollback_pending_integrate": "writeback",
        "_resume_recover_pending_integrate": "writeback",
        "_resume_recover_orphaned_keeps": "writeback",
        "_enqueue_internal_stack_rebench": "writeback",
        "_validate_geak_via_geak_harness": "writeback",
        "resumed_from": "writeback",
        "_replay_resume_if_needed": "writeback",
        "_maybe_run_maintenance_tick": "maintenance",
        "_maybe_prune_runs_for_disk": "maintenance",
        "_maybe_checkpoint_orchestration": "maintenance",
        "enqueue_targeted_build": "build_lifecycle",
    }

    def __getattr__(self, name: str):
        # Only fires for genuinely-missing attributes (not shadowed instance
        # attrs / real methods). Delegate extracted collaborator methods to their
        # owner; everything else is a real AttributeError.
        owner = Coordinator._DELEGATED.get(name)
        if owner is not None:
            target = getattr(self, owner)
            try:
                # Do not invoke the collaborator's fallback ``__getattr__`` here:
                # a stale _DELEGATED entry would otherwise bounce back to this
                # Coordinator and recurse until RecursionError.
                return object.__getattribute__(target, name)
            except AttributeError as exc:
                raise AttributeError(
                    f"{type(self).__name__!r} delegates {name!r} to {owner!r}, but that collaborator does not define it"
                ) from exc
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def _inline_action_whitelist(self) -> frozenset[str]:
        return self.dispatcher._inline_action_whitelist()

    def _run_action_now_sync(self, action_name: str, params: dict[str, Any] | None = None) -> str:
        return self.dispatcher._run_action_now_sync(action_name, params)

    async def _run_action_now(self, action_name: str, params: dict[str, Any] | None = None) -> str:
        return await self.dispatcher._run_action_now(action_name, params)

    def _collaborator(self, attr: str, factory):
        """Lazily build + cache a collaborator object (like ``router``/``writeback``);
        works for ``Coordinator.__new__`` test doubles too (uses ``__dict__``)."""
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

        return self._collaborator("_dispatcher", DispatcherCollaborator)

    @property
    def writeback(self):
        from .writeback import WritebackCollaborator

        return self._collaborator("_writeback", WritebackCollaborator)

    @property
    def maintenance(self):
        from .maintenance import MaintenanceCollaborator

        return self._collaborator("_maintenance", MaintenanceCollaborator)

    @property
    def build_lifecycle(self):
        from .build_lifecycle import BuildLifecycleCollaborator

        return self._collaborator("_build_lifecycle", BuildLifecycleCollaborator)

    def _kb_hardware_slug(self) -> str:
        """Topology-aware hardware dimension for the recipe ``canonical_id``.

        Single-node: bare ``gpu_type`` (existing keys/data unchanged).
        Multi-node: ``gpu_type`` + ``_ws{world_size}`` so multi-node runs never
        share a recipe key with — and overwrite the ``best_config`` of — the
        single-node recipe. MUST match ``recipe_kb_t0.run_t0_anchor``'s derivation
        so warm-start reads and KEEP/REVERT/CLOSE writes target the same row.

        Returns:
            The topology-aware hardware slug for the current run.
        """
        from hyperloom.orchestrator.actions.executors._multi_node_env import resolve_kb_topology
        from hyperloom.inference_optimizer.recipe_snapshot_constants import kb_hardware_slug

        ss = self.shared_state
        return kb_hardware_slug(ss.gpu_type or "unknown_gpu", **resolve_kb_topology())

    def _reap_orphaned_servers_best_effort(self) -> None:
        """Reap leftover single-node serving processes from a prior crash.

        Scoped to this session's own pidfiles and gated on a cmdline match, so a
        co-located session's server and a recycled pid are never touched. A
        no-op on multi-node (servers live on pods) and on a fresh session.
        Best-effort: any failure is logged and swallowed so boot never fails.
        """
        try:
            from ..actions.executors._multi_node_env import is_multi_node

            if is_multi_node():
                return
            from ..actions.executors._server_lifecycle import reap_orphaned_servers

            reaped = reap_orphaned_servers(self.session_dir)
            if reaped:
                log.warning(
                    "coordinator: reaped %d orphaned serving process(es) at boot: %s",
                    len(reaped),
                    reaped,
                )
        except Exception:  # noqa: BLE001 - boot-time cleanup must never be fatal
            log.exception("coordinator: orphan server reaper failed (ignored)")

    # Advisory disk guard: when the session partition runs low, LRU-trim the
    # bulkiest churn (per-task runs/ workspaces); durable state is never touched.
    _DISK_FREE_MIN_GB: float = 20.0
    _DISK_USED_MAX_FRAC: float = 0.85
    _DISK_RUNS_KEEP_PER_ACTION: int = 50
    _STATE_JSON_WARN_BYTES: int = 50 * 1024 * 1024

    # Action catalogue mapping action_name -> metadata. Class-level so a
    # partially-built Coordinator still resolves it.
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
        """Signal shutdown, cancel in-flight work, finalize, and close the DB.

        Sets the stop event, cancels and awaits the dispatched actions still
        running plus every running reactor task, runs the Recipe KB T4
        safety-net finalize hook when CLOSE never reached a terminal
        publication status or its earlier attempt failed, then closes the
        SQLite connection. Exceptions raised by reactor tasks during teardown
        are logged, not propagated.

        Dispatched actions are cancelled first and awaited: the stop event alone
        only asks the loop to stop between ticks, so a teardown that skipped
        them would close the database out from under work still using it.
        """
        self._stop.set()
        try:
            await self.dispatcher.cancel_inflight_actions(reason="coordinator_stop")
        except Exception:  # noqa: BLE001 — teardown proceeds even if cancellation misbehaves
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
            except Exception:  # noqa: BLE001
                log.exception("reactor task raised on shutdown")
        # Safety net: recipe/journal finalize when CLOSE sequencer didn't run.
        await self._recipe_kb_t4_hook()
        self.db.close()

    def _bind_session_deadline(
        self,
        *,
        max_minutes: float | None,
        closing_grace_sec: float | None,
    ) -> tuple[float, float, float]:
        """Stamp the persisted deadline once and size this process's loop clock.

        Bounded sessions persist ``deadline_unix`` from ``start_ts + budget`` on
        the first ``run()`` and keep it on resume, so this process cannot
        reissue a full ``max_minutes``. Unbounded sessions keep the container
        cap as a local monotonic deadline and do not persist one.

        Args:
            max_minutes: Operator wall-clock budget, or ``None``/0 for unbounded.
            closing_grace_sec: Operator CLOSE window; ``None`` derives a default.

        Returns:
            ``(grace_sec, monotonic_deadline, max_minutes_value)``.
        """
        grace_sec = effective_closing_grace_sec(max_minutes, closing_grace_sec)
        self.shared_state.closing_grace_sec = closing_grace_sec
        max_minutes_value = max_minutes if max_minutes is not None else 0
        if max_minutes:
            # Stamp from the float budget before persisting ``int(max_minutes)``.
            # ``int(0.0001)`` is 0, and stamping after that truncation used to
            # leave ``deadline_unix`` unset so remaining-time checks read unbounded.
            self.shared_state.stamp_deadline_unix(budget_minutes=float(max_minutes))
            self.shared_state.max_minutes = int(max_minutes)
            self.shared_state.save(self.session_dir)
            deadline = self.shared_state.monotonic_session_deadline_sec()
            if deadline is None:
                deadline = time.monotonic()
        else:
            self.shared_state.deadline_unix = 0.0
            if max_minutes is not None:
                self.shared_state.max_minutes = int(max_minutes)
                self.shared_state.save(self.session_dir)
            deadline = time.monotonic() + _phase_state.DEFAULT_LONGRUN_MAX_MINUTES * 60.0
        self._run_started_monotonic = time.monotonic()
        self._run_deadline = deadline
        return grace_sec, deadline, float(max_minutes_value)

    async def _recipe_kb_t4_hook(self) -> None:
        """Finalize or retry on graceful teardown/Ctrl-C.

        Terminal publication statuses are idempotent no-ops. An unfinished
        CLOSE sequence, a missing status, or a prior retryable failure receives
        one more in-process attempt before the database closes.

        This in-process hook cannot run after SIGKILL, container force-delete,
        host loss, or interpreter failure.
        """
        if bool(getattr(getattr(self, "knowledge_plane", None), "kb_disabled", False)):
            return
        finalize_status = str(getattr(self.shared_state, "recipe_finalize_status", "") or "")
        if getattr(self.shared_state, "close_sequence_done", False) and finalize_status in {
            "written",
            "skipped",
            "disabled",
        }:
            return
        try:
            config = getattr(getattr(self, "knowledge_plane", None), "config", None) or KnowledgeConfig.from_env()
            if config.mode is KnowledgeStoreMode.LOCAL:
                if self.recipe_kb is None:
                    return
                sid = (self.shared_state.recipe_kb_session_id or "").strip()
                if not sid:
                    return
            self.ensure_recipe_finalized(source="t4_fallback")
        except Exception:  # noqa: BLE001 — defensive
            log.exception("recipe KB T4 fact_finalize fallback failed")
        try:
            self.shared_state.save(self.session_dir)
        except Exception:  # noqa: BLE001
            log.exception("recipe KB T4 SharedState.save failed")

    # Statuses that mean the candidate was ADOPTED; everything else is a negative
    # signal for the ranker.
    _FRAMEWORK_KEEP_STATUSES: frozenset[str] = frozenset({"kept"})

    # Max tried-candidate rows fed into the ranker/discovery working memory.
    _FRAMEWORK_TRIED_MEMORY_CAP: int = 12

    _CRITIC_PRIORS_OUTCOME_TAIL: int = 5

    # Auto-roofline — PRELUDE bootstrap + 10% watermark refresh.
    _ROOFLINE_WATERMARK_RATIO: float = 1.10  # 10% step over last roofline
    # Relative-change floor for the pre-GEAK reprofile: any change above this
    # re-runs profile+TraceLens (effectively "any change", absorbing float noise).
    _REPROFILE_CHANGE_TOL: float = 1e-5

    # Max re-author rounds per candidate on a needs_review verdict.
    _MAX_REAUTHOR_ATTEMPTS: int = _AUTHORED_LANE_MAX_ATTEMPTS

    # Backstop: max Critic-review submissions for a single candidate before
    # the pump force-stamps ``repeated_review_abort`` and stops re-selecting it.
    _MAX_REPEATED_REVIEW_SUBMISSIONS: int = 3

    # CLOSE step 0 post-opt roofline hard cap; on timeout the optimized snapshot
    # is skipped so report/breakdown always run.
    CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC: float = 600.0

    # optimization_stack actions warranting a post-opt roofline; pure
    # param-search (explore/sweep) is excluded.
    _POST_OPT_ROOFLINE_ACTIONS = frozenset({"collective", "integrate", "integrate_patch", "gemm_tuning", "geak_e2e"})

    async def tick(self, n: int = 1) -> None:
        """Run exactly ``n`` reactor passes for every agent; dispatcher pumps at pass end, lazy resume replay on tick 1.

        Args:
            n: Number of full reactor+dispatcher passes to run (default 1).
        """
        await self._replay_resume_if_needed()
        for _ in range(n):
            self.shared_state.increment_tick()
            # A phase-entry hook may have finished by setting a pending phase
            # hint (for example current GEAK returning no_gain -> skip_to_sweep).
            # Consume that before prompting agents again so stale phase prompts
            # cannot enqueue legacy work.
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

    def _seconds_until_session_bound(self) -> float | None:
        """Seconds left on the active run or closing bound; ``None`` if unbounded.

        During CLOSE the session deadline has already passed, so the bound
        switches to ``_closing_deadline`` and CLOSE work is not skipped.

        A success terminal is unbounded here: the sequencer's own per-step
        timeouts (``CLOSE_POST_OPT_ROOFLINE_TIMEOUT_SEC`` is 600s on its own)
        are the budget. An outer bound short enough to matter would cancel the
        step mid-flight and drop the run onto the safety net -- the very outcome
        routing into CLOSE exists to avoid.

        Returns:
            Remaining seconds, or ``None`` when no bound is armed.
        """
        if self._terminal_closing:
            return None
        if bool(getattr(self.shared_state, "closing_phase", False)):
            bound = self._closing_deadline
        else:
            bound = self._run_deadline
        if bound is None:
            return None
        return float(bound) - time.monotonic()

    async def _await_within_session_bound(
        self,
        factory: Callable[[], Awaitable[Any]],
        *,
        stage: str,
    ) -> None:
        """Run one tick step, cancelling it when the session/closing bound elapses.

        The wall-clock stop lives at the end of each tick. A conversational
        reactor turn or a long phase-enter await that never returns would skip
        that stop. Cancelling the step lets the tick finish and enter CLOSE.

        Args:
            factory: Builds the awaitable so a skipped step is never started.
            stage: Label for the warning log.
        """
        remaining = self._seconds_until_session_bound()
        if remaining is not None and remaining <= 0.0:
            log.warning("Coordinator: skipping %s; session bound already elapsed", stage)
            return
        try:
            # ``timeout=None`` waits until the step finishes (unbounded run).
            await asyncio.wait_for(factory(), timeout=remaining)
        except asyncio.TimeoutError:
            log.warning(
                "Coordinator: %s hit the session bound after %.1fs; cancelled so the tick can close",
                stage,
                remaining,
            )

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
        """Run reactor + dispatcher until a stop condition fires (priority order): signal, target_reached (via the CLOSE phase sequencer), time_exhausted (via closing phase), emergency, custom, max_ticks. Sets + saves + returns shared_state.stop_reason.

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
        # Capture the live loop for the inline fast-action context tool.
        try:
            self._coordinator_loop = asyncio.get_running_loop()
        except RuntimeError:
            self._coordinator_loop = None

        previous_handlers: dict[int, Any] = {}
        if install_signal_handlers:
            try:
                loop = asyncio.get_running_loop()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, self._stop.set)
                    previous_handlers[sig] = True
                log.info("Coordinator.run: SIGINT/SIGTERM handlers installed")
            except (NotImplementedError, RuntimeError) as exc:  # noqa: BLE001
                # add_signal_handler unavailable off the main thread / on Windows.
                log.info("Coordinator.run: signal handlers not installed (%s)", exc)
                previous_handlers = {}

        await self._replay_resume_if_needed()
        grace_sec, deadline, max_minutes_value = self._bind_session_deadline(
            max_minutes=max_minutes,
            closing_grace_sec=closing_grace_sec,
        )

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
                    except Exception as exc:  # noqa: BLE001
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
                            if self._stop.is_set():
                                break
                            await self._await_within_session_bound(
                                lambda n=name: self._reactor_pass(n),
                                stage=f"reactor:{name}",
                            )
                        # Orchestration checkpoint/compaction; cadence-based.
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
                        # Phase-independent enablement pump.
                        await self._pump_enablement_safely(caller="run")
                    # phase machine advance; runs even in_closing so CLOSE is recorded.
                    try:
                        await self._await_within_session_bound(
                            self._advance_phase_if_needed,
                            stage="advance_phase",
                        )
                    except Exception as exc:  # noqa: BLE001
                        log.exception("phase advance (run) failed")
                        self._record_coordinator_exception(
                            stage="advance_phase",
                            exc=exc,
                            tick=tick_n,
                        )
                    # Periodic reaper + DB retention; cadence-gated.
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
                    # Route the terminal through CLOSE. ``machine_state``
                    # registers ``target_reached`` as an "any phase -> CLOSE"
                    # transition reason, but nothing ever produced it, so a met
                    # target skipped the 7-step close sequencer and left only
                    # the cli safety-net report.
                    #
                    # The transition runs in THIS tick rather than the next one.
                    # Every ``_advance_phase_if_needed`` in the tick body sits
                    # behind ``_await_within_session_bound``, which skips the
                    # step once the session bound has elapsed -- so a target met
                    # at or after the deadline would never get another advance,
                    # and deferring the close would silently fall back to the
                    # safety net.
                    if not self.shared_state.stop_reason:
                        self.shared_state.set_stop_reason("target_reached")
                        # Best-effort: the terminal is already set in memory and
                        # CLOSE persists state itself, so a failed write must not
                        # cost the run its close sequence.
                        try:
                            self.shared_state.save(self.session_dir)
                        except Exception:  # noqa: BLE001
                            log.exception(
                                "Coordinator: persisting target_reached failed; closing anyway",
                            )
                        # Lift the session bound for this one advance so the
                        # elapsed run deadline cannot skip it. ``closing_phase``
                        # is deliberately NOT set: that flag means the wall clock
                        # ran out, and CLOSE reads it to shed expensive work --
                        # ``_maybe_run_close_post_opt_roofline`` returns early on
                        # it, which would drop the very artifact this routing
                        # exists to produce. The sequencer's own per-step
                        # timeouts bound the work.
                        self._terminal_closing = True
                        try:
                            await self._await_within_session_bound(
                                self._advance_phase_if_needed,
                                stage="advance_phase_target_reached",
                            )
                        except Exception:  # noqa: BLE001
                            log.exception(
                                "Coordinator: close transition on target_reached failed",
                            )
                        finally:
                            self._terminal_closing = False
                    stop_reason = self.shared_state.stop_reason or "target_reached"
                    break
                if deadline is not None and time.monotonic() >= deadline and not in_closing:
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
                        # Normal path: no stop signal within the tick interval.
                        pass
        finally:
            if self.shared_state.closing_phase:
                self.shared_state.closing_phase = False
            # Resuming a terminal session can break out before stop_reason is set.
            self.shared_state.set_stop_reason(
                stop_reason
                or self.shared_state.stop_reason
                or ("coordinator_exception" if last_tick_exc is not None else "unknown")
            )
            self.shared_state.save(self.session_dir)
            # Every graceful terminal path gets one idempotent Recipe finalize
            # attempt, including stop-check exits that never enter PHASE_CLOSE.
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
            # Best-effort cleanup of installed signal handlers.
            if previous_handlers:
                try:
                    loop = asyncio.get_running_loop()
                    for sig in previous_handlers:
                        loop.remove_signal_handler(sig)
                except (NotImplementedError, RuntimeError):
                    # Teardown is best-effort; signal handlers may be unsupported.
                    pass
            with timed_teardown_step(self.shared_state, "close_backends"):
                await self._close_backends()
            self.shared_state.save(self.session_dir)
        return self.shared_state.stop_reason

    async def _close_backends(self) -> None:
        """Release every backend holding a live agent session.

        A session-scoped backend (Codex) keeps a child process and a private
        state directory open for the whole run, and the loop is the only place
        that knows the run is over. Failing to close one must not change the
        stop reason, so each failure is logged and the rest still close.
        """
        for name, backend in list(self.backends.items()):
            closer = getattr(backend, "aclose", None)
            if not callable(closer):
                continue
            try:
                await closer()
            except Exception:  # noqa: BLE001 — teardown must not mask the stop reason
                log.exception("Coordinator: closing the %s backend failed", name)

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
        # The system prompt is loaded first because the SEED/DELTA gate inside
        # _compose_prompt has to know whether THIS prompt replaces the backend's
        # conversation. A re-scoped prompt opens a new thread inside the turn, so
        # a gate that only sees the thread as it stands would compose a delta for
        # a conversation that is about to be emptied.
        sys_prompt = await self._load_system_prompt(agent_name)
        prompt = await self._compose_prompt(agent_name, system_prompt=sys_prompt)
        tools = self.policy.allowed_tools_for_agent(agent_name)
        # Stamp timeline keys onto backends that self-write their trace row.
        # No-op for backends without the hook. Presence of the hook is also what
        # tells the failure path below to stay out of the way: such a backend
        # records its own row, with the review model and its own latency, and a
        # second row from here would count one provider failure twice.
        _set_trace_ctx = getattr(backend, "set_trace_context", None)
        backend_self_traces = callable(_set_trace_ctx)
        if backend_self_traces:
            try:
                _set_trace_ctx(
                    tick=int(self.shared_state.tick or 0),
                    phase=(self.shared_state.phase or "") or None,
                    macro_cycle=int(self.shared_state.macro_cycle or 0),
                )
            except Exception:  # noqa: BLE001
                pass
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
        except Exception as exc:  # noqa: BLE001
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
        # Context-token water level. Only a per-request figure is comparable to
        # the window; a backend reporting none leaves the level at 0 and the
        # char ledger carries the growth signal alone.
        if agent_name == "orchestration" and self._orchestration_conversational():
            try:
                md = getattr(result, "metadata", None) or {}
                self._checkpoint_tracker.set_context_tokens(int(md.get("context_tokens_peak") or 0))
                self._checkpoint_tracker.chars_add(len(prompt) + len(getattr(result, "raw_text", "") or ""))
            except Exception:  # noqa: BLE001 — accounting must never break routing
                pass
        # Kept out of the ledger's try above: that one exists so a missing token
        # figure still leaves the char ledger updating, and a throw from here
        # would stop it too.
        if agent_name == "orchestration" and self._orchestration_conversational():
            try:
                md = getattr(result, "metadata", None) or {}
                self._checkpoint_policy.adopt_context_window(
                    int(md.get("model_context_window") or 0),
                    self._checkpoint_soft_fraction,
                )
            except Exception:  # noqa: BLE001 — accounting must never break routing
                pass
        # Completed orchestration turn means SEED delivered; later turns send DELTA.
        if agent_name == "orchestration":
            self._orchestration_seeded = True
        for intent in result.intents:
            await self._handle_intent(agent_name, intent)
        await self._advance_rendered_cursor(agent_name)

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
        except Exception:  # noqa: BLE001
            log.debug("orchestration mcp setup trace failed", exc_info=True)

    def _trace_reactor_llm_call(
        self,
        agent_name: str,
        result: BackendTurnResult,
        *,
        latency_ms: int | None = None,
    ) -> None:
        """Append one ``llm_calls.jsonl`` row for a reactor turn.

        The reactor role name doubles as both the trace ``component`` and
        ``role``. Only rows carrying real token counters are written. Any error
        in trace assembly degrades to a logged warning rather than breaking the
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

    def _trace_reactor_llm_failure(
        self,
        agent_name: str,
        error: LLMCallFailed,
        *,
        latency_ms: int | None = None,
    ) -> None:
        """Append one ``status="error"`` ``llm_calls.jsonl`` row for a failed turn.

        The success path (:meth:`_trace_reactor_llm_call`) can only run once a
        ``BackendTurnResult`` exists, so without this the ledger — and therefore
        Langfuse — has no trace of a turn whose model call never returned. The
        row carries no token counters; it exists to make the failure countable.

        Only :class:`LLMCallFailed` reaches here, and only from a backend that
        does not trace itself. A plain ``BackendError`` can come from a
        deterministic local fault (unreadable ``emit.json``, missing
        ``--review`` path, absent SDK) that never touched the provider, and
        recording those would make the Langfuse error rate meaningless; a
        self-tracing backend (the critic) has already written a richer row of
        its own, so writing here too would double-count one failure.

        Args:
            agent_name: The reactor role; doubles as trace component and role.
            error: The model-call failure that ended the turn.
            latency_ms: Time spent before failing, when measured.
        """
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
        except Exception:  # noqa: BLE001 — trace must never break the loop
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

    # Multi-node only: cap on specialist proposal_set entries auto-materialised
    # into a single explore grid per round.
    _MN_AUTO_EXPLORE_GRID_CAP = 6

    # Phases whose long, serially-drained GPU grids must not starve the
    # per-phase cyclic budget exit. PRELUDE/SWEEP/CLOSE/RECOVER drain normally.
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
