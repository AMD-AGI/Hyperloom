# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PolicyGate — single chokepoint: every parsed Intent passes through ``validate_intent`` before side-effects."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from ..framework.paths import (
    is_rocm_hip_writable_path,
    resolve_session_framework_root,
    resolve_source_file_allowlist,
    resolved_within,
    source_file_candidates,
)
from ..bus.gpu_pool import (
    resolve_gpu_specialist_devices,
    resolve_whole_machine_devices,
)
from hyperloom.common.visible_devices import COUNTING_VISIBLE_DEVICE_VARS
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.protocol.action_surfaces import (
    COORDINATOR_INTERNAL_ACTIONS,
    COORDINATOR_OWNED_KERNEL_REQUEST_KINDS,
    KERNEL_AGENT_OWNED_ACTIONS,
    ROBUSTNESS_DELEGATE_ONLY_ACTIONS,
)
from ..specialists.domains import (
    KNOWLEDGE_DOMAIN_TAG_SET,
    SPECIALIST_MAX_TURNS_HARD_CAP,
    domain_for_tag,
    get_domain,
    normalize_dispatch_tags,
)
from ..specialists.profile import (
    SCOPE_DOMAIN as SPECIALIST_SCOPE_DOMAIN,
    SCOPE_DOMAINS as SPECIALIST_SCOPE_DOMAINS,
    SCOPE_FREEFORM as SPECIALIST_SCOPE_FREEFORM,
    SCOPE_VALUES as SPECIALIST_SCOPE_VALUES,
)
from ..specialists.patch_safety import parse_patch_targets

if TYPE_CHECKING:  # pragma: no cover — type-only
    from ..roles.agent_role import AgentRole


log = logging.getLogger(__name__)


def _value_is_present(value: Any) -> bool:
    """Present iff a non-empty string OR non-empty container; ``None`` / whitespace count as absent."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0
    return True


def _delegate_field_present(payload: dict[str, Any], field_name: str) -> bool:
    """True iff ``field_name`` is present at the top of ``payload`` OR nested under ``payload[\"params\"]`` (robustness uses params)."""
    if _value_is_present(payload.get(field_name)):
        return True
    nested = payload.get("params")
    if isinstance(nested, dict) and _value_is_present(nested.get(field_name)):
        return True
    return False


class PolicyDenied(RuntimeError):
    """Intent rejected by PolicyGate."""

    def __init__(self, reason: str, *, rule: str | None = None, hint: str | None = None):
        """Initialise the denial with a human-readable reason and metadata."""
        super().__init__(reason)
        self.rule = rule
        self.hint = hint


ROBUSTNESS_ONLY_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"robustness"})

# Per-action delegate source allowlist, derived so it cannot drift from ROBUSTNESS_DELEGATE_ONLY_ACTIONS; unlisted
# actions carry no source restriction.
DELEGATE_ACTION_SOURCE_ALLOWLIST: dict[str, frozenset[str]] = {
    action: ROBUSTNESS_ONLY_SOURCE_ALLOWLIST for action in ROBUSTNESS_DELEGATE_ONLY_ACTIONS
}


# Per-action delegate required payload fields (minimum evidence for the audit trail; missing/empty raise PolicyDenied).
DELEGATE_ACTION_REQUIRED_PAYLOAD: dict[str, tuple[str, ...]] = {
    "recover": ("reason", "evidence"),
}


# Specialist dispatch action name.
SPECIALIST_ACTION_NAME: str = "specialist"

# Orchestrator-side patch integration step (gated by a Critic verdict).
INTEGRATE_PATCH_ACTION_NAME: str = "integrate_patch"

# GEMM tuning action; the hook that guards it is called for every action, so it needs its own name to answer only for
# itself.
GEMM_TUNING_ACTION_NAME: str = "gemm_tuning"


# Specialist / Explore parallelism caps — single source of truth across layers.
RESEARCH_LANE_CEILING_FALLBACK: int = 2


def detect_gpu_count() -> int:
    """Count visible GPUs from env masks, then ``rocm-smi``; return 0 on failure."""
    for env_name in COUNTING_VISIBLE_DEVICE_VARS:
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        raw = raw.strip()
        if raw == "":
            return 0
        ids = [tok for tok in raw.split(",") if tok.strip() != ""]
        if ids:
            return len(ids)
    import subprocess

    try:
        proc = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, ValueError, subprocess.TimeoutExpired):
        return 0
    if proc.returncode != 0:
        return 0
    indices: set[str] = set()
    for line in (proc.stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("GPU["):
            idx, _, _ = stripped[4:].partition("]")
            if idx:
                indices.add(idx)
    return len(indices)


def research_lane_ceiling() -> int:
    """Dynamic ceiling on concurrent research-lane specialists (``2 × GPU``; falls back to :data:`RESEARCH_LANE_CEILING_FALLBACK`)."""
    gpus = detect_gpu_count()
    if gpus > 0:
        return 2 * gpus
    return RESEARCH_LANE_CEILING_FALLBACK


def gpu_specialist_ceiling(shared_state: Any | None = None) -> int:
    """Configured GPU specialist capacity (separate from serving lanes; 0 disables ``needs_gpu=true`` dispatch)."""
    if shared_state is not None:
        try:
            return max(0, int(getattr(shared_state, "gpu_specialist_capacity", 0) or 0))
        except (TypeError, ValueError):
            return 0
    try:
        return max(0, int(os.environ.get("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "0") or "0"))
    except ValueError:
        return 0


def _serving_tp_for_policy(shared_state: Any | None = None) -> int:
    """Resolve serving TP for policy-time specialist GPU validation."""
    if shared_state is not None:
        try:
            tp = int(getattr(shared_state, "tp", 0) or 0)
        except (TypeError, ValueError):
            tp = 0
        if tp > 0:
            return tp
    try:
        return max(0, int(os.environ.get("TP", "0") or 0))
    except ValueError:
        return 0


def _effective_gpu_specialist_pool_size(shared_state: Any | None = None) -> int:
    """Actual policy-time GPU specialist pool size after serving carve."""
    ceiling = gpu_specialist_ceiling(shared_state)
    if ceiling <= 0:
        return 0
    return len(
        resolve_gpu_specialist_devices(
            ceiling,
            serving_tp=_serving_tp_for_policy(shared_state),
        ),
    )


def _whole_machine_pool_size() -> int:
    """Policy-time size of the whole-machine (framework/bench) GPU pool."""
    return len(resolve_whole_machine_devices())


# Verdicts that allow ``integrate_patch`` without an operator override (``advise`` = soft approval, ``approve`` = green light).
INTEGRATE_PATCH_PERMISSIVE_VERDICTS: frozenset[str] = frozenset(
    {
        "approve",
        "advise",
    }
)


def patch_verdict_subject(params: Mapping[str, Any]) -> str:
    """Return the id an ``integrate_patch``'s Critic verdict is filed under."""
    sid = str(params.get("specialist_task_id") or "").strip()
    return sid or str(params.get("framework_agent_candidate_id") or "").strip()


# Source roles allowed to dispatch a specialist via ``delegate{action='specialist'}``.
SPECIALIST_DISPATCH_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"orchestration"})

# Free-form (``scope='freeform'``) sanity-gate limits; the real ceiling is the research_lane capacity and the GPU
# specialist pool.
SPECIALIST_FREEFORM_WAVE_MAX: int = 16
SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS: int = 8000

# Specialist task identity prefix: the dispatcher stamps ``specialist:<task_id>`` as the source of a specialist result
# (explore parses the task_id back out), and a send_message addressed to ``specialist:<task_id>`` is delivered to that
# specialist's inbox.
SPECIALIST_FROM_AGENT_PREFIX: str = "specialist:"


# R5 — external tool whitelist registry (single source of truth for PolicyGate + SpecialistRunner).

#: PR Monitor *readonly* surfaces. R5 same role gating.
PR_MONITOR_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "mcp__pr_monitor__pr_repos_list",
        "mcp__pr_monitor__pr_repo_stats",
        "mcp__pr_monitor__pr_list",
        "mcp__pr_monitor__pr_get",
        "mcp__pr_monitor__pr_files",
        "mcp__pr_monitor__pr_file_patch",
        "mcp__pr_monitor__pr_patches",
        "mcp__pr_monitor__pr_blob",
        "mcp__pr_monitor__pr_commit_files",
        "mcp__pr_monitor__pr_commit_file",
        "mcp__pr_monitor__pr_pr_file_baseline",
        "mcp__pr_monitor__pr_search",
    }
)

#: Web tools. R5 — specialist-only in this map (other roles get ``tool_whitelist_role``);
#: usable in any phase. Note :meth:`PolicyGate.allowed_tools_for_agent` separately grants
#: ``WebSearch`` / ``WebFetch`` to the orchestration agent.
WEB_TOOL_NAMES: frozenset[str] = frozenset({"WebSearch", "WebFetch"})

#: Role→allowed-toolset map (R5). Only the specialist sub-agent may invoke PR Monitor / web
#: tools as an action name.
TOOL_WHITELIST_BY_ROLE: dict[str, frozenset[str]] = {
    "specialist": (WEB_TOOL_NAMES | PR_MONITOR_TOOL_NAMES),
    # Empty sets listed explicitly so a role-name typo is a key error, not a silent allow.
    "orchestration": frozenset(),
    "critic": frozenset(),
    "robustness": frozenset(),
}

#: Convenience superset of every known external tool name (R5 collision check).
ALL_KNOWN_EXTERNAL_TOOL_NAMES: frozenset[str] = PR_MONITOR_TOOL_NAMES | WEB_TOOL_NAMES


# REQUEST/RESPONSE routing matrix: source role → allowed target_agents (only orchestration→kernel).
REQUEST_ROUTING: dict[str, frozenset[str]] = {
    "orchestration": frozenset({"kernel_agent"}),
}


# Critic-only: REVIEW_VERDICT
REVIEW_VERDICT_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"critic"})

# Verdict vocabulary for review_verdict
REVIEW_VERDICTS: frozenset[str] = frozenset(
    {
        "approve",
        "reject",
        "redirect",
        "advise",
        "needs_review",
    }
)


# prune_branch scopes.
PRUNE_BRANCH_SCOPE_FAMILY: str = "family"
PRUNE_BRANCH_SCOPE_QUEUED: str = "queued"
PRUNE_BRANCH_ALLOWED_SCOPES: frozenset[str] = frozenset(
    {
        PRUNE_BRANCH_SCOPE_FAMILY,
        PRUNE_BRANCH_SCOPE_QUEUED,
    }
)

# Ceiling on a single extend_lease step; repeated extensions are allowed.
EXTEND_LEASE_MAX_SEC: int = 3600

ROBUSTNESS_ONLY_INTENTS: frozenset[IntentType] = frozenset(
    {
        IntentType.PRUNE_BRANCH,
        IntentType.ESCALATE_STRATEGY_CHANGE,
    }
)

# Per-intent source override: PRUNE_BRANCH + ESCALATE_STRATEGY_CHANGE widen to orchestration.
_ROBUSTNESS_ONLY_INTENT_SOURCES: dict[IntentType, frozenset[str]] = {
    IntentType.PRUNE_BRANCH: frozenset({"robustness", "orchestration"}),
    IntentType.ESCALATE_STRATEGY_CHANGE: frozenset(
        {
            "robustness",
            "orchestration",
        }
    ),
}


# SESSION_DIR path containment: PATH_LIKE_FIELDS must point inside session_dir or a framework source allowlist (checked recursively).
PATH_LIKE_FIELDS: frozenset[str] = frozenset(
    {
        "trace_input",
        "candidates_path",
        "patch_path",
        "target_file",
        "resolved_patch_targets",
        "config_path",
        "output_dir",
        "workspace",
        "workspace_path",
        "trace_dir",
        "main_trace_path",
        "report_path",
        "json_path",
        "md_path",
        "session_dir",
        "backup_root",
        "manifest_path",
    }
)

# `source_file` and `framework_source_root` may point at trusted installed source scopes outside the session
# directory.
SOURCE_LIKE_FIELDS: frozenset[str] = frozenset({"source_file", "framework_source_root"})

# Payload fields that name files modified by patch/install actions.
ROCM_WRITE_PATH_FIELDS: frozenset[str] = frozenset({"patch_path", "target_file", "resolved_patch_targets"})

# Coordinator-owned warm replay may deploy a KB patch into the active framework checkout.
_WARM_REPLAY_ACTION = "replay_warm_recipe"
_REMOTE_RECIPE_FILES_PARTS = ("runtime", "remote_recipe", "files")
_MAX_POLICY_PATCH_BYTES = 4 * 1024 * 1024

# Placeholder/not-found sentinels that upstream lookups (or an LLM restating a miss as prose) can leave in a
# SOURCE_LIKE_FIELDS value instead of leaving the field empty.
_SOURCE_FILE_ABSENT_SENTINELS: frozenset[str] = frozenset(
    {
        "not found",
        "none",
        "n/a",
        "null",
        "unknown",
        "unresolved",
        "missing",
        "tbd",
        "<unresolved>",
        "aiter (vendor)",
        "triton (vendor)",
    }
)


# Multi-node profile trace dirs live outside session_dir but must be referenceable by trace_dir / main_trace_path / trace_input (runtime-resolved).
def _trace_path_allowlist() -> tuple[str, ...]:
    """Multi-node profile trace path allowlist (runtime-resolved)."""
    from hyperloom.inference_optimizer.session.paths import mn_profile_trace_root

    root = str(mn_profile_trace_root()).rstrip("/") + "/"
    return (root,)


# Subset of PATH_LIKE_FIELDS that also accept :func:`_trace_path_allowlist` (others stay strictly session-rooted).
TRACE_PATH_LIKE_FIELDS: frozenset[str] = frozenset(
    {
        "trace_dir",
        "main_trace_path",
        "trace_input",
    }
)


# Core SharedState fields that only the Coordinator may mutate.
CORE_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "current_best",
        "stop_reason",
        # Paired with stop_reason and written by the same setter: locking one without the other lets an update_state
        # move the session's end time away from the reason it was stamped for.
        "stop_ts",
        "last_tick_exception",
        "cumulative_gain_validated",
        "cumulative_gain_validated_ts",
        "cumulative_gain_validated_stack_len",
        "pending_integrate",
        "resume_pending_revalidation",
        "baseline_tput",
        "baseline_accuracy",
        "session_id",
        "model_path",
        "model_name",
        "model_class",
        # The topology every number in the session was measured on, established once at launch from a read of the
        # card.
        "compute_partition",
        "start_ts",
        # Where the current run leg begins; a forged value hands a previous leg's CLOSE transition back the right to
        # speak for this one.
        "resumed_ts",
        "max_minutes",
        # Absolute session deadline.
        "deadline_unix",
        # Sizes the closing reserve, so it decides how much of ``max_minutes`` is still usable: locking the budget
        # without locking this one leaves the same forgery one field over -- a large value spends the session
        # outright, a zero one erases the window the CLOSE report needs.
        "closing_grace_sec",
        # fact-layer KEEP ledger; Coordinator is the sole writer.
        "optimization_stack",
        "gain_per_stack_entry",
        "schema_version",
        # Recipe KB integration fields (Coordinator-only writes).
        "recipe_kb_session_id",
        "warm_start_recipe",
        "warm_start_pitfalls",
        "warm_start_lessons",
        "warm_start_ts",
        "warm_start_context",
        "kb_stage_outbox",
        "kb_stage_dead_letter",
        "recipe_finalize_status",
        "recipe_finalize_attempts",
        "recipe_finalize_outcome",
        # KB tag completeness (Coordinator-populated; LLM reads via prompt).
        "stack_fingerprint_meta",
        "baseline_workload_extra",
        "last_profile_workload",
        "last_profile_workload_action",
        # warm-recipe replay one-shot guard + outcome; LLM cannot edit.
        "warm_replay_attempted",
        "warm_replay_outcome",
        "warm_history_injected",
        # phase state machine fields (managed by ``Coordinator._advance_phase_if_needed``).
        "phase",
        "phase_started_ts",
        "phase_started_unix",
        "phase_history",
        "phase_budget_pct",
        "explore_elapsed_accum_s",
        "phase_elapsed_totals",
        # KERNEL idle-streak bookkeeping.
        "kernel_idle_ticks",
        "kernel_progress_fingerprint",
        "kernel_idle_since_unix",
        # Cyclic phase-machine state; Coordinator-only writers.
        "macro_cycle",
        "cycle_minutes",
        "gain_at_cycle_start",
        "no_gain_cycle_streak",
        "pending_bottleneck_switch",
        "last_cycle_bottleneck",
        "saturated_directions",
        "bottleneck_shift",
        "cycle_strategy_log",
        # operator-facing lifecycle event log; Coordinator-only writer so the LLM cannot forge lifecycle events.
        "lifecycle",
        # specialist sub-agent ledger; Coordinator-only writer.
        "specialist_rounds",
        # per-kb_anchor coverage counters; Coordinator-only writers.
        "rounds_since_last_specialist",
        "rounds_since_last_keep",
        "last_specialist",
        # research_lane / GPU capacity set once at CLI/manifest time; locked.
        "research_lane_capacity",
        "gpu_specialist_capacity",
        # phase-machine escalation plumbing; LLM blocked (defense in depth).
        "pending_escalate_hint",
        "last_consumed_escalate_hint",
        "last_consumed_escalate_hint_ts",
        "last_discarded_escalate_hint",
        "last_discarded_escalate_hint_ts",
        "plateau_overrides",
        # CLOSE-phase sequencer flag; LLM must not toggle it.
        "close_sequence_done",
        # Objective-met marker; the Coordinator is its only writer.
        "target_reached_at",
        # explore search ledger; Coordinator-only writers (LLM rewrite would bypass dedup-by-fingerprint).
        "explore_search",
        # structured gaps ledger; Coordinator-only writers (``_refresh_gaps``, ``_seed_gaps_from_research_hints``,
        # ``_record_explore_round_gaps``, ``_consume_static_recon``), all via ``SharedState.upsert_gap``.
        "gaps",
        # Orchestration working-memory checkpoint; Coordinator-authored.
        "orchestration_memory",
        # Bounded rollback ring of prior good orchestration_memory records; Coordinator-only writer, locked in
        # lock-step with its parent.
        "orchestration_memory_history",
        # Advisory model-architecture profile from the SKILL launcher; locked.
        "model_arch",
        # Architecture-identity tags from config.json; locked against pollution.
        "model_architectures",
        "model_type",
        # Multimodal text-fallback degraded-run markers (cli._preflight); Coordinator/preflight are the sole writers.
        "degraded_mode",
        "model_warnings",
        # Kernel-opt ledgers + Critic patch-verdict store; Coordinator/kernel-agent are the sole writers.
        "specialist_patch_verdicts",
        "last_trace_analyze",
        "last_kernel_opt",
        "kernel_opt_task_attempts",
        "pending_kernel_integrations",
        "last_collective",
        "collective_attempts",
        "collective_only_mode",
        # closing_phase and baseline_config_path are Coordinator-only fact fields, locked here so non-coordinator
        # roles cannot mutate them via UPDATE_STATE.
        "closing_phase",
        "baseline_config_path",
        # Structured failure evidence; Coordinator-only writer.
        "failures",
    }
)


@dataclass
class PolicyGate:
    """Validate every intent emitted by an agent reactor."""

    role_registry: dict[str, "AgentRole"]
    session_dir: Path | None = None
    strict_paths: bool = False
    shared_state: Any | None = None

    def __post_init__(self) -> None:  # noqa: D401 — dataclass hook
        """Apply the ``INFERENCE_OPTIMIZER_STRICT_PATHS`` override."""
        import os as _os

        if not self.strict_paths and _os.environ.get("INFERENCE_OPTIMIZER_STRICT_PATHS", "").strip() in (
            "1",
            "true",
            "yes",
        ):
            self.strict_paths = True

    # Public API
    def validate_intent(self, from_agent: str, intent: Intent) -> None:
        """Raise :class:`PolicyDenied` if the intent is not allowed (cheapest checks first: role → allowed_intents → structural → cross-source)."""
        role = self.role_registry.get(from_agent)
        if role is None:
            raise PolicyDenied(f"unknown agent {from_agent!r}", rule="role")

        if intent.type not in role.allowed_intents:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit intent_type={intent.type.value!r}",
                rule="role",
            )

        payload = intent.payload or {}

        # Per-intent structural validators
        if intent.type == IntentType.DELEGATE:
            self._validate_delegate(role, payload)
        elif intent.type == IntentType.PROPOSE_ACTION:
            self._validate_propose_action(role, payload)
        elif intent.type == IntentType.UPDATE_STATE:
            self._validate_state_transition(role, payload)
        elif intent.type == IntentType.SEND_MESSAGE:
            self._validate_send_message_topic(payload)
        elif intent.type == IntentType.REQUEST:
            self._validate_request(role, payload)
        elif intent.type == IntentType.RESPONSE:
            self._validate_response(payload)
        elif intent.type == IntentType.REVIEW_VERDICT:
            self._validate_review_verdict(role, payload)
        elif intent.type == IntentType.EXTEND_LEASE:
            self._validate_extend_lease(payload)
        elif intent.type in ROBUSTNESS_ONLY_INTENTS:
            self._validate_robustness_only(role, intent.type, payload)
        # ALERT carries no extra checks beyond the role gate.

        # Path-containment guard for PATH_LIKE_FIELDS in the payload.
        self._validate_payload_paths(role, intent.type, payload)

    def validate_dispatched_task(
        self,
        action_name: str,
        params: dict[str, Any] | None,
    ) -> None:
        """Re-validate a persisted queued task before executor dispatch."""
        kind = str(action_name or "").strip()
        if not kind:
            raise PolicyDenied("dispatched task missing kind", rule="payload")
        params_dict = dict(params or {}) if isinstance(params, dict) else {}
        payload = {"action_name": kind, "params": params_dict}
        role = self.role_registry.get("orchestration")
        if role is None:
            raise PolicyDenied("unknown agent 'orchestration'", rule="role")
        trusted_framework_targets: frozenset[str] = frozenset()
        if kind == _WARM_REPLAY_ACTION:
            trusted_framework_targets = self._validate_warm_replay_targets(params_dict)
        self._validate_payload_paths(
            role,
            IntentType.DELEGATE,
            payload,
            trusted_framework_targets=trusted_framework_targets,
        )
        # Coordinator-dispatched internal actions get path checks only.
        if kind in COORDINATOR_INTERNAL_ACTIONS:
            return
        self._validate_delegate_body(
            role,
            payload,
            check_source=False,
        )

    def allowed_tools_for_agent(self, agent_name: str) -> list[str]:
        """Return the Claude tool list a reactor may use (Codex → []; Claude → emit_intent; orchestration also gets context-pull tools + sandboxed Read + web search)."""
        role = self.role_registry.get(agent_name)
        if role is None:
            return []
        if role.no_tools:
            return []
        tools = ["emit_intent"]
        if agent_name == "orchestration":
            from ..roles.mcp_context_tools import CONTEXT_TOOL_NAMES

            tools.extend(CONTEXT_TOOL_NAMES)
            tools.append("Read")
            tools.extend(["WebSearch", "WebFetch"])
        return tools

    # Per-intent validators
    def _validate_delegate(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate a ``DELEGATE`` intent against the full delegate rule set."""
        self._validate_delegate_body(role, payload)

    def _validate_delegate_body(
        self,
        role: "AgentRole",
        payload: dict[str, Any],
        *,
        check_source: bool = True,
    ) -> None:
        """Shared delegate validation for intents and dispatched task rows."""
        if not role.can_delegate_side_effects:
            raise PolicyDenied(
                f"role={role.name!r} cannot delegate side-effecting actions",
                rule="role",
            )
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied("delegate intent missing action_name", rule="payload")
        # ``specialist`` bypasses the catalogue; ``_validate_specialist_dispatch`` owns its contract.
        if action_name == SPECIALIST_ACTION_NAME:
            self._validate_specialist_dispatch(role, payload)
            return
        # ``integrate_patch`` requires a non-reject Critic verdict.
        if action_name == INTEGRATE_PATCH_ACTION_NAME:
            self._validate_integrate_patch_critic_gate(payload)
        self._validate_gemm_tuning_action(action_name, intent_kind="delegate")
        if check_source:
            self._validate_coordinator_managed_action(action_name, intent_kind="delegate")
            self._validate_baseline_not_mid_authoring(action_name)
            # Robustness delegates nothing beyond its own declared action set.
            if role.name in ROBUSTNESS_ONLY_SOURCE_ALLOWLIST and action_name not in ROBUSTNESS_DELEGATE_ONLY_ACTIONS:
                raise PolicyDenied(
                    f"role={role.name!r} cannot delegate action={action_name!r}; "
                    f"allowed: {sorted(ROBUSTNESS_DELEGATE_ONLY_ACTIONS)!r}",
                    rule="role",
                )
            allowed_sources = DELEGATE_ACTION_SOURCE_ALLOWLIST.get(action_name)
            if allowed_sources is not None and role.name not in allowed_sources:
                raise PolicyDenied(
                    f"role={role.name!r} cannot delegate action={action_name!r} (allowed: {sorted(allowed_sources)!r})",
                    rule="delegate_action_source",
                    hint=(
                        "side-effecting actions like `recover` are reserved for "
                        "the robustness agent; emit an ALERT and let robustness "
                        "escalate via its action-ladder instead"
                    ),
                )
        # Per-action required-payload guard (e.g. ``recover`` must carry ``reason`` + ``evidence``); top-level or under ``params``.
        required = DELEGATE_ACTION_REQUIRED_PAYLOAD.get(action_name)
        if required:
            missing = [field_name for field_name in required if not _delegate_field_present(payload, field_name)]
            if missing:
                raise PolicyDenied(
                    f"delegate(action_name={action_name!r}) missing required payload field(s): {missing!r}",
                    rule="delegate_action_evidence",
                    hint=(
                        "side-effecting delegates must carry the symptom "
                        "evidence that justified them (e.g. "
                        "{'reason': 'gpu_memory_leaked', "
                        "'evidence': {...}})"
                    ),
                )
        # R5 — block a delegate whose action_name invokes an external tool.
        self._validate_tool_whitelist_collision(
            role.name,
            action_name,
            intent_kind="delegate",
        )

    def _validate_propose_action(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate a ``PROPOSE_ACTION`` intent (the advisory channel)."""
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied("propose_action missing action_name", rule="payload")
        self._validate_coordinator_managed_action(action_name, intent_kind="propose_action")
        self._validate_baseline_not_mid_authoring(action_name)
        # Per-action source allowlist (e.g. ``recover`` is robustness-only); mirrors the delegate-path guard.
        allowed_sources = DELEGATE_ACTION_SOURCE_ALLOWLIST.get(action_name)
        if allowed_sources is not None and role.name not in allowed_sources:
            raise PolicyDenied(
                f"role={role.name!r} cannot propose action={action_name!r} (allowed: {sorted(allowed_sources)!r})",
                rule="propose_action_source",
                hint=(
                    "side-effecting actions like `recover` are reserved for "
                    "the robustness agent; emit an ALERT and let robustness "
                    "escalate via its action-ladder instead"
                ),
            )
        self._validate_gemm_tuning_action(action_name, intent_kind="propose_action")
        # R5 — defense in depth on propose_action.
        self._validate_tool_whitelist_collision(
            role.name,
            action_name,
            intent_kind="propose_action",
        )

    def _validate_coordinator_managed_action(self, action_name: str, *, intent_kind: str) -> None:
        """Deny an agent-initiated copy of an action the Coordinator dispatches itself."""
        if action_name not in COORDINATOR_INTERNAL_ACTIONS:
            return
        raise PolicyDenied(
            f"action {action_name!r} is Coordinator-managed and not LLM-proposable ({intent_kind})",
            rule="coordinator_managed_action",
            hint="read the outcome in SharedState rather than ordering a second run.",
        )

    def _validate_baseline_not_mid_authoring(self, action_name: str) -> None:
        """Deny an LLM ``baseline`` while an enablement authoring round is in flight."""
        if action_name != "baseline" or self.shared_state is None:
            return
        inflight = self.shared_state.enablement.inflight_task_id
        if not inflight:
            return
        raise PolicyDenied(
            f"baseline: an enablement authoring round is in flight (task={inflight})",
            rule="enablement_round_in_flight",
            hint="wait for the enablement specialist to finish and rearm before re-running baseline.",
        )

    def _validate_state_transition(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate an ``UPDATE_STATE`` intent's ``changes`` against core fields."""
        changes = payload.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise PolicyDenied(
                "update_state.payload.changes must be a non-empty dict",
                rule="payload",
                hint=("include at least one allowed field, e.g. {'changes': {'current_action': '<action_name>'}}"),
            )
        violating = sorted(set(changes.keys()) & CORE_STATE_FIELDS)
        if violating:
            raise PolicyDenied(
                f"role={role.name!r} cannot mutate core state fields: {violating!r}",
                rule="state_field",
            )

    def _validate_send_message_topic(self, payload: dict[str, Any]) -> None:
        """Require a non-empty ``topic`` on a ``SEND_MESSAGE`` intent."""
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            raise PolicyDenied("send_message missing topic", rule="payload")

    def _validate_request(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate a ``REQUEST`` intent against the routing matrix."""
        targets = REQUEST_ROUTING.get(role.name)
        if not targets:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit REQUEST",
                rule="request_role",
            )
        target = str(payload.get("target_agent", "")).strip()
        if not target:
            raise PolicyDenied("request missing target_agent", rule="payload")
        if target not in targets:
            raise PolicyDenied(
                f"role={role.name!r} cannot request target_agent={target!r} (allowed: {sorted(targets)!r})",
                rule="request_target",
            )
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            raise PolicyDenied("request missing kind", rule="payload")
        if kind in COORDINATOR_OWNED_KERNEL_REQUEST_KINDS:
            raise PolicyDenied(
                f"request kind {kind!r} is a Coordinator-owned kernel lane and not LLM-requestable",
                rule="request_kind",
                hint=(
                    "the lane runs at KERNEL entry once its own gate passes, "
                    "targeted from the nomination and bounded by the lane budget; "
                    "its outcome arrives as <kind>_done. Wait for that event and "
                    "`integrate` the KEEPs it queues."
                ),
            )
        self._validate_gemm_tuning_action(kind, intent_kind="request")
        # R5 — a REQUEST.kind cannot smuggle an external tool either.
        self._validate_tool_whitelist_collision(
            role.name,
            kind,
            intent_kind="request",
        )

    def _validate_response(self, payload: dict[str, Any]) -> None:
        """Require ``in_reply_to`` and ``kind`` on a ``RESPONSE`` intent."""
        in_reply_to = str(payload.get("in_reply_to", "")).strip()
        if not in_reply_to:
            raise PolicyDenied("response missing in_reply_to", rule="payload")
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            raise PolicyDenied("response missing kind", rule="payload")

    def _validate_review_verdict(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate a ``REVIEW_VERDICT`` intent (Critic-only)."""
        if role.name not in REVIEW_VERDICT_SOURCE_ALLOWLIST:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit review_verdict (allowed: {sorted(REVIEW_VERDICT_SOURCE_ALLOWLIST)!r})",
                rule="review_verdict_source",
            )
        target = str(payload.get("target_proposal_msg_id", "")).strip()
        if not target:
            raise PolicyDenied(
                "review_verdict missing target_proposal_msg_id",
                rule="payload",
            )
        # Accept the single ``verdict`` or the per-variant ``verdict_map``.
        has_single = "verdict" in payload
        verdict_map = payload.get("verdict_map")
        has_map = isinstance(verdict_map, dict) and bool(verdict_map)
        if has_single == has_map:
            raise PolicyDenied(
                "review_verdict: exactly one of 'verdict' or 'verdict_map' must be present",
                rule="payload",
                hint=(
                    "single-proposal review: emit {target_proposal_msg_id, "
                    "verdict, reasoning, failure_reason_code?}. Explore batch "
                    "review: emit {target_proposal_msg_id, verdict_map: "
                    "{variant_name: {verdict, rationale?, "
                    "failure_reason_code?}}}"
                ),
            )
        if has_single:
            verdict = str(payload.get("verdict", "")).strip()
            if verdict not in REVIEW_VERDICTS:
                raise PolicyDenied(
                    f"review_verdict.verdict={verdict!r} not in allowed set {sorted(REVIEW_VERDICTS)!r}",
                    rule="payload",
                    hint="use one of approve/reject/redirect/advise/needs_review",
                )
            return
        # verdict_map path — every entry's verdict must be in the closed vocab.
        for vname, entry in verdict_map.items():
            v = str((entry or {}).get("verdict") or "").strip()
            if v not in REVIEW_VERDICTS:
                raise PolicyDenied(
                    f"review_verdict.verdict_map[{vname!r}].verdict="
                    f"{v!r} not in allowed set "
                    f"{sorted(REVIEW_VERDICTS)!r}",
                    rule="payload",
                    hint=("every per-variant verdict must be one of approve/reject/redirect/advise/needs_review"),
                )

    # GEMM tuning ownership
    def _validate_gemm_tuning_action(
        self,
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Refuse a model-proposed GEMM tuning run; the Coordinator owns the lane."""
        # Called unconditionally for every action, so it must answer only for its own; it used to never raise, which
        # hid that.
        if action_name != GEMM_TUNING_ACTION_NAME:
            return
        raise PolicyDenied(
            f"{action_name!r} is a Coordinator-owned kernel lane and not model-requestable ({intent_kind})",
            rule="phase_incompatible",
            hint=(
                "GEMM tuning is dispatched by the Coordinator at KERNEL entry once its "
                "deterministic gate passes; it draws on a lane budget rather than a "
                "per-request one, so it cannot be re-issued per tick."
            ),
        )

    # R5 — tool_whitelist_role
    def _validate_tool_whitelist_collision(
        self,
        role_name: str,
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject an external tool name not on the caller's role whitelist (:data:`TOOL_WHITELIST_BY_ROLE` grants PR Monitor + web tools to ``specialist`` only)."""
        if not action_name:
            return
        if action_name not in ALL_KNOWN_EXTERNAL_TOOL_NAMES:
            return
        allowed_for_role = TOOL_WHITELIST_BY_ROLE.get(role_name, frozenset())
        if action_name in allowed_for_role:
            return
        raise PolicyDenied(
            f"role={role_name!r} cannot invoke tool {action_name!r}",
            rule="tool_whitelist_role",
            hint=(
                f"Tool {action_name!r} is restricted to "
                f"specialist sub-agents as an action name. The "
                f"primary agents (orchestration / kernel / critic / "
                f"robustness) reach KB / PR Monitor through the "
                f"Coordinator-mediated KnowledgePlane facade instead; "
                f"orchestration additionally holds WebSearch / WebFetch "
                f"directly via allowed_tools_for_agent."
            ),
        )

    def _validate_integrate_patch_critic_gate(
        self,
        payload: dict[str, Any],
    ) -> None:
        """Enforce a permissive Critic verdict on the patch's review subject."""
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise PolicyDenied(
                "integrate_patch: params must be a dict",
                rule="integrate_patch_requires_critic_verdict",
                hint=("pass params={specialist_task_id: <id>, ...}; see actions/integrate_patch.md"),
            )
        # Enablement build launch probe: an ``enablement_launch_only`` integrate runs the (already artifact-verified)
        # built runtime through the runnable gate WITHOUT applying any patch.
        if params.get("enablement_launch_only"):
            return
        sid = patch_verdict_subject(params)
        if not sid:
            raise PolicyDenied(
                "integrate_patch.params names no Critic review subject",
                rule="integrate_patch_requires_critic_verdict",
                hint=(
                    "set params.specialist_task_id to the task_id of "
                    "the completed specialist whose worktree carries "
                    "the patches you want to apply, or "
                    "params.framework_agent_candidate_id to the "
                    "pre-screened upstream-PR candidate."
                ),
            )
        ss = getattr(self, "shared_state", None)
        verdict = ""
        if ss is not None:
            try:
                verdict = ss.get_specialist_patch_verdict(sid)
            except AttributeError:
                # Guards a null specialist_patch_verdicts deserialized from state.json.
                verdict = ""
        if not verdict:
            raise PolicyDenied(
                f"integrate_patch: no Critic verdict on record for subject={sid!r}",
                rule="integrate_patch_requires_critic_verdict",
                hint=(
                    "Wait for the Critic to emit a "
                    "review_verdict{target_proposal_msg_id=<patch "
                    "proposal>, verdict=approve|reject|...} for this "
                    "specialist. The Critic verdict "
                    "is recorded on SharedState.specialist_patch_verdicts."
                ),
            )
        if verdict.lower() not in INTEGRATE_PATCH_PERMISSIVE_VERDICTS:
            raise PolicyDenied(
                f"integrate_patch: Critic verdict for subject "
                f"{sid!r} is {verdict!r}; integrate_patch only "
                f"runs on "
                f"{sorted(INTEGRATE_PATCH_PERMISSIVE_VERDICTS)!r}",
                rule="integrate_patch_requires_critic_verdict",
                hint=(
                    "Either ask the Critic to re-review (next "
                    "review_verdict overwrites this one), or drop the "
                    "patch (specialist_done.patches_written=[])."
                ),
            )

    def _validate_specialist_dispatch(
        self,
        role: "AgentRole",
        payload: dict[str, Any],
    ) -> None:
        """Enforce the specialist-delegate contract (Inv-11.2): orchestration-only, gap_canonical_id required, max_turns ≤ cap."""
        if role.name not in SPECIALIST_DISPATCH_SOURCE_ALLOWLIST:
            raise PolicyDenied(
                f"role={role.name!r} cannot dispatch specialists "
                f"(allowed: {sorted(SPECIALIST_DISPATCH_SOURCE_ALLOWLIST)!r})",
                rule="specialist_dispatch_source",
                hint=(
                    "Only the Orchestration role may dispatch specialists. "
                    "Robustness should escalate via "
                    "escalate_strategy_change with "
                    "hint='need_specialist:<domain>'; the orchestration "
                    "tick will pick it up."
                ),
            )
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise PolicyDenied(
                "delegate{action='specialist'}: params must be a dict",
                rule="specialist_dispatch_source",
                hint="pass params={tags, gap_canonical_id, ...} per §3.5 §6",
            )

        # scope='freeform' has no domain anchor: it skips the tag / gap vocabulary checks and runs a lightweight
        # mechanical sanity gate instead.
        scope_raw = str(params.get("scope") or "").strip().lower()
        if scope_raw == SPECIALIST_SCOPE_FREEFORM:
            self._validate_freeform_specialist_dispatch(params)
            return

        # ``params.tags`` is canonical; ``params.domain`` is accepted as a single-tag alias.
        tags = normalize_dispatch_tags(params)
        # A bare dispatch (no scope, no anchor) defaults to the cheap freeform lane; its gate still requires a
        # non-empty task_description.
        if not scope_raw and not tags:
            self._validate_freeform_specialist_dispatch(params)
            return

        # Observed, not enforced: resolve_specialist_profile re-infers the scope and the runner synthesizes an empty
        # result for an unresolvable anchor.
        if not tags:
            log.info("specialist dispatch declares a scope but no tags; profile will re-infer")
        unknown_tags = [t for t in tags if t not in KNOWLEDGE_DOMAIN_TAG_SET]
        if unknown_tags:
            log.info(
                "specialist dispatch carries out-of-vocabulary tag(s)=%r (known: %r)",
                unknown_tags,
                sorted(KNOWLEDGE_DOMAIN_TAG_SET),
            )

        if scope_raw and scope_raw not in SPECIALIST_SCOPE_VALUES:
            log.info(
                "specialist dispatch scope=%r not in %r; re-inferred from tags",
                scope_raw,
                sorted(SPECIALIST_SCOPE_VALUES),
            )
        elif scope_raw == SPECIALIST_SCOPE_DOMAINS and len(tags) < 2:
            log.info("specialist dispatch scope='domains' with %d tag(s)=%r", len(tags), tags)
        elif scope_raw == SPECIALIST_SCOPE_DOMAIN and len(tags) > 1:
            log.info("specialist dispatch scope='domain' with %d tags=%r", len(tags), tags)

        gap = str(params.get("gap_canonical_id") or params.get("gap") or "").strip()
        if not gap:
            # Backfill the gap id from the gaps[] ledger by matching the dispatch anchor against each gap's
            # ``domain_hint``; only mutates on a match.
            gap = self._autofill_gap_from_ledger(params, tags)
        if not gap:
            raise PolicyDenied(
                "delegate{action='specialist'}: params.gap_canonical_id required",
                rule="specialist_dispatch_source",
                hint=(
                    "Provide a canonical gap id (e.g. "
                    "'gap.attention.fp8_kv_cache.session-<sid>') so the "
                    "specialist can anchor its KB traversal."
                ),
            )
        max_turns_raw = params.get("max_turns")
        validate_specialist_max_turns_raw(max_turns_raw, where="params.max_turns")

        self._validate_specialist_gpu_request(params)

    def _validate_specialist_gpu_request(self, params: dict[str, Any]) -> None:
        """Validate a specialist's optional GPU request against the GPU specialist-pool ceiling."""
        needs_gpu_raw = params.get("needs_gpu", False)
        if isinstance(needs_gpu_raw, str):
            needs_gpu = needs_gpu_raw.strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
        else:
            needs_gpu = bool(needs_gpu_raw)
        if not needs_gpu:
            from ..specialists.profile import resolve_specialist_profile

            if resolve_specialist_profile(params).reserves_benchmark_lane:
                needs_gpu = True
        if not needs_gpu:
            return
        serving_tp = _serving_tp_for_policy(self.shared_state)
        from ..specialists.profile import (
            resolve_specialist_profile,
            uses_whole_machine_gpu_lane,
        )

        # Whole-machine bench specialists lease from ``framework_gpu_pool``, so their default gpu_count matches the
        # dispatcher: serving_tp when known, else the whole-machine pool capacity (the serving_tp == 0 case).
        if uses_whole_machine_gpu_lane(params) and serving_tp == 0:
            default_gpu_count = _whole_machine_pool_size() or 1
        else:
            default_gpu_count = serving_tp or 1
        gpu_count_raw = params.get("gpu_count", default_gpu_count)
        if gpu_count_raw is None or (isinstance(gpu_count_raw, str) and not gpu_count_raw.strip()):
            gpu_count_raw = default_gpu_count
        try:
            gpu_count = int(gpu_count_raw)
        except (TypeError, ValueError):
            # The dispatcher re-parses with the same default.
            log.info("specialist dispatch gpu_count=%r not an integer; using %d", gpu_count_raw, default_gpu_count)
            gpu_count = int(default_gpu_count)
        if gpu_count <= 0:
            raise PolicyDenied(
                "delegate{action='specialist'}: gpu_count must be > 0 when needs_gpu=true",
                rule="specialist_gpu_request_invalid",
            )
        ceiling = gpu_specialist_ceiling(self.shared_state)
        if ceiling <= 0 and not (uses_whole_machine_gpu_lane(params) and _whole_machine_pool_size() > 0):
            raise PolicyDenied(
                "delegate{action='specialist'}: needs_gpu=true but the GPU specialist pool is disabled",
                rule="specialist_gpu_pool_disabled",
                hint=(
                    "Start the session with --gpu-specialist-capacity > 0 "
                    "or set INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY "
                    "before dispatching GPU specialists."
                ),
            )

        if resolve_specialist_profile(params).reserves_benchmark_lane and serving_tp > 0 and gpu_count < serving_tp:
            gpu_count = serving_tp
        # Whole-machine, time-shared specialists validate against the whole-machine pool, not the serving-disjoint
        # pool: they route to ``framework_gpu_pool`` and serialize with serving, so the carve does not apply (else
        # they'd be denied when serving owns the node).
        if uses_whole_machine_gpu_lane(params):
            effective_pool_size = _whole_machine_pool_size()
            pool_desc = "whole-machine GPU pool"
        else:
            effective_pool_size = _effective_gpu_specialist_pool_size(self.shared_state)
            pool_desc = "serving-disjoint GPU specialist pool"
        if gpu_count > effective_pool_size:
            raise PolicyDenied(
                "delegate{action='specialist'}: "
                f"effective gpu_count={gpu_count} exceeds {pool_desc} "
                f"size={effective_pool_size} "
                f"(configured capacity={ceiling}, serving_tp={serving_tp})",
                rule="specialist_gpu_request_exceeds_capacity",
                hint=(
                    "Lower params.gpu_count for non-bench probes, omit it for "
                    "bench specialists only when the pool has at least serving "
                    "TP free cards, or start a session with a larger GPU pool."
                ),
            )

    def _autofill_gap_from_ledger(
        self,
        params: dict[str, Any],
        tags: list[str],
    ) -> str:
        """Backfill ``params.gap_canonical_id`` from the gaps[] ledger."""
        state = getattr(self, "shared_state", None)
        gaps = list(getattr(state, "gaps", None) or []) if state is not None else []
        if not gaps:
            return ""

        # Build the anchor candidate set the gap's domain_hint may name.
        candidates: set[str] = set()
        domain_key = str(params.get("domain") or "").strip()
        if domain_key:
            candidates.add(domain_key.lower())
            d = get_domain(domain_key)
            if d and d.kb_anchor:
                candidates.add(d.kb_anchor.lower())
        for t in tags:
            t_l = str(t).strip().lower()
            if t_l:
                candidates.add(t_l)
            dt = domain_for_tag(t)
            if dt:
                candidates.add(dt.key.lower())
                if dt.kb_anchor:
                    candidates.add(dt.kb_anchor.lower())
        if not candidates:
            return ""

        severity_rank = {"high": 3, "medium": 2, "low": 1}

        def _selection_key(g: dict[str, Any]) -> tuple[int, int, str]:
            """Sort key ranking gaps by actionability for autofill."""
            sev = severity_rank.get(str(g.get("severity") or "").lower(), 0)
            attempts = len(g.get("attempts") or [])
            first_seen = str(g.get("first_seen_ts") or "")
            return (-sev, attempts, first_seen)

        matches = [
            g
            for g in gaps
            if isinstance(g, dict)
            and str(g.get("canonical_id") or "").strip()
            and str(g.get("domain_hint") or "").strip().lower() in candidates
        ]
        if not matches:
            return ""
        matches.sort(key=_selection_key)
        chosen = str(matches[0].get("canonical_id") or "").strip()
        if chosen:
            params["gap_canonical_id"] = chosen
        return chosen

    def _validate_freeform_specialist_dispatch(
        self,
        params: dict[str, Any],
    ) -> None:
        """Lightweight mechanical sanity gate for ``scope='freeform'`` specialists."""
        # Freeform applies the same max_turns contract as domain dispatches.
        validate_specialist_max_turns_raw(params.get("max_turns"), where="params.max_turns")
        self._validate_specialist_gpu_request(params)
        wave = params.get("tasks")
        # A malformed or empty wave falls through to the single-task path in the fan-out, which re-checks shape per
        # entry.
        if isinstance(wave, list) and wave:
            if len(wave) > SPECIALIST_FREEFORM_WAVE_MAX:
                raise PolicyDenied(
                    f"delegate{{action='specialist',scope='freeform'}}: wave "
                    f"size={len(wave)} exceeds cap "
                    f"{SPECIALIST_FREEFORM_WAVE_MAX}",
                    rule="specialist_freeform_wave_too_large",
                    hint=(f"Split the wave into batches of at most {SPECIALIST_FREEFORM_WAVE_MAX} tasks."),
                )
            for i, task in enumerate(wave):
                validate_freeform_wave_task(task, index=i)
                if isinstance(task, dict):
                    validate_specialist_max_turns_raw(
                        task.get("max_turns"),
                        where=f"tasks[{i}].max_turns",
                    )
            return
        desc = str(params.get("task_description") or "").strip()
        self._check_freeform_task_description(desc, where="params")

    @staticmethod
    def _check_freeform_task_description(desc: str, *, where: str) -> None:
        """Per-task structural checks for a free-form ``task_description``: non-empty and length-bounded."""
        if not desc:
            raise PolicyDenied(
                f"delegate{{action='specialist',scope='freeform'}}: {where} task_description must be non-empty",
                rule="specialist_freeform_empty_description",
                hint=("Each freeform task needs a natural-language task_description (the whole mandate)."),
            )
        if len(desc) > SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS:
            raise PolicyDenied(
                f"delegate{{action='specialist',scope='freeform'}}: "
                f"{where} task_description is {len(desc)} chars > cap "
                f"{SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS}",
                rule="specialist_freeform_description_too_long",
            )

    def _validate_extend_lease(self, payload: dict[str, Any]) -> None:
        """Validate an ``EXTEND_LEASE`` intent."""
        task_id = str(payload.get("task_id", "")).strip()
        if not task_id:
            raise PolicyDenied("extend_lease missing task_id", rule="payload")
        try:
            extra_sec = int(payload.get("extra_sec") or 0)
        except (TypeError, ValueError) as exc:
            raise PolicyDenied(
                f"extend_lease extra_sec must be an integer, got {payload.get('extra_sec')!r}",
                rule="payload",
            ) from exc
        if extra_sec <= 0 or extra_sec > EXTEND_LEASE_MAX_SEC:
            raise PolicyDenied(
                f"extend_lease extra_sec={extra_sec} outside (0, {EXTEND_LEASE_MAX_SEC}]",
                rule="extend_lease_bounds",
                hint=(
                    "Extend in bounded steps and re-check get_running_tasks; "
                    "a lease must not outlive the session budget."
                ),
            )

    def _path_under_session(self, value: str) -> bool:
        """Return whether a path resolves inside the active session_dir."""
        if self.session_dir is None:
            return True
        try:
            sd = self.session_dir.resolve()
            v = Path(str(value)).resolve()
        except (OSError, RuntimeError):
            return False
        return v == sd or v.is_relative_to(sd)

    def _path_in_source_allowlist(self, value: str) -> bool:
        """Return whether a path falls under a trusted installed source scope."""
        return any(resolved_within(value, p) for p in resolve_source_file_allowlist())

    def _path_in_trace_allowlist(self, value: str) -> bool:
        """Match a value against runtime-resolved trace path prefixes (multi-node shared profile dir outside session_dir)."""
        return any(resolved_within(value, p) for p in _trace_path_allowlist())

    def _remote_recipe_files_root(self) -> Path | None:
        """Return the session-owned root containing downloaded KB artifacts."""
        if self.session_dir is None:
            return None
        try:
            return self.session_dir.resolve().joinpath(*_REMOTE_RECIPE_FILES_PARTS)
        except (OSError, RuntimeError):
            return None

    @staticmethod
    def _patch_declared_targets(patch_path: Path) -> frozenset[str]:
        """Read safe relative targets from unified-diff headers."""
        try:
            if not patch_path.is_file() or patch_path.stat().st_size > _MAX_POLICY_PATCH_BYTES:
                return frozenset()
            text = patch_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return frozenset()

        try:
            return frozenset(parse_patch_targets(text).all)
        except ValueError:
            return frozenset()

    def _framework_relative_candidates(self, target_file: str) -> frozenset[str]:
        """Return the target path relative to this Session's active root."""
        raw_root = resolve_session_framework_root()
        if not raw_root:
            return frozenset()
        try:
            target = Path(target_file).resolve()
            root = Path(raw_root).resolve()
            relative = target.relative_to(root)
        except (OSError, RuntimeError):
            return frozenset()
        except ValueError:
            return frozenset()
        relative_posix = relative.as_posix()
        return frozenset({relative_posix}) if relative_posix and relative_posix != "." else frozenset()

    def _validate_warm_replay_targets(
        self,
        params: dict[str, Any],
    ) -> frozenset[str]:
        """Validate the sole framework-target exception for warm replay."""
        if self.session_dir is None or not self.strict_paths:
            return frozenset()
        plan = params.get("warm_kernel_plan") or []
        if not isinstance(plan, list):
            raise PolicyDenied(
                "replay_warm_recipe warm_kernel_plan must be a list",
                rule="warm_replay_plan_invalid",
            )

        kb_root = self._remote_recipe_files_root()
        admitted: set[str] = set()
        patch_coverage: dict[str, tuple[frozenset[str], set[str]]] = {}
        for index, entry in enumerate(plan):
            if not isinstance(entry, dict):
                raise PolicyDenied(
                    f"replay_warm_recipe warm_kernel_plan[{index}] must be an object",
                    rule="warm_replay_plan_invalid",
                )
            raw_targets = entry.get("resolved_patch_targets") or []
            if not raw_targets:
                continue
            if not isinstance(raw_targets, list) or not all(
                isinstance(target, str) and target.strip() for target in raw_targets
            ):
                raise PolicyDenied(
                    f"replay_warm_recipe warm_kernel_plan[{index}].resolved_patch_targets "
                    "must be a flat non-empty string list",
                    rule="warm_replay_plan_invalid",
                )

            raw_patch = entry.get("patch_path")
            if not isinstance(raw_patch, str) or not raw_patch.strip():
                raise PolicyDenied(
                    f"replay_warm_recipe resolved_patch_targets={raw_targets!r} has no patch_path",
                    rule="warm_replay_patch_missing",
                )
            if kb_root is None or not resolved_within(raw_patch, str(kb_root)):
                raise PolicyDenied(
                    f"replay_warm_recipe patch_path={raw_patch!r} is outside the session KB download root={kb_root!s}",
                    rule="warm_replay_patch_outside_kb_download",
                )

            declared_targets = self._patch_declared_targets(Path(raw_patch))
            try:
                patch_key = str(Path(raw_patch).resolve())
            except (OSError, RuntimeError) as exc:
                raise PolicyDenied(
                    f"replay_warm_recipe patch_path={raw_patch!r} cannot be resolved",
                    rule="warm_replay_patch_outside_kb_download",
                ) from exc
            known_targets, covered_targets = patch_coverage.setdefault(
                patch_key,
                (declared_targets, set()),
            )
            if known_targets != declared_targets:
                raise PolicyDenied(
                    f"replay_warm_recipe patch_path={raw_patch!r} changed during validation",
                    rule="warm_replay_patch_target_mismatch",
                )
            for raw_target in raw_targets:
                active_root = resolve_session_framework_root()
                if not active_root or not resolved_within(raw_target, active_root):
                    raise PolicyDenied(
                        f"replay_warm_recipe target_file={raw_target!r} is outside the Session active framework root",
                        rule="warm_replay_target_outside_framework_roots",
                    )
                target_candidates = self._framework_relative_candidates(raw_target)
                if not declared_targets or declared_targets.isdisjoint(target_candidates):
                    raise PolicyDenied(
                        f"replay_warm_recipe target_file={raw_target!r} does not "
                        f"match patch targets={sorted(declared_targets)!r}",
                        rule="warm_replay_patch_target_mismatch",
                    )
                covered_targets.update(target_candidates)
                try:
                    admitted.add(str(Path(raw_target).resolve()))
                except (OSError, RuntimeError) as exc:
                    raise PolicyDenied(
                        f"replay_warm_recipe target_file={raw_target!r} cannot be resolved",
                        rule="warm_replay_target_outside_framework_roots",
                    ) from exc
        for patch_key, (declared_targets, covered_targets) in patch_coverage.items():
            uncovered = declared_targets - covered_targets
            if uncovered:
                raise PolicyDenied(
                    f"replay_warm_recipe patch_path={patch_key!r} declares "
                    f"targets with no matching target_file={sorted(uncovered)!r}",
                    rule="warm_replay_patch_target_mismatch",
                )
        return frozenset(admitted)

    def _validate_payload_paths(
        self,
        role: "AgentRole",
        intent_type: IntentType,
        payload: dict[str, Any],
        *,
        trusted_framework_targets: frozenset[str] = frozenset(),
    ) -> None:
        """Walk payload (recursively); reject path-like values escaping session_dir. No-op when session_dir is None or strict_paths is False."""
        if self.session_dir is None or not self.strict_paths:
            return

        def visit(node: Any, path_keys: tuple[str, ...]) -> None:
            """Recursively scan a payload node for escaping path values."""
            if isinstance(node, dict):
                for k, v in node.items():
                    visit(v, path_keys + (str(k),))
                return
            if isinstance(node, (list, tuple)):
                for item in node:
                    visit(item, path_keys)
                return
            if not isinstance(node, str) or not node.strip():
                return
            key = path_keys[-1] if path_keys else ""
            if key in SOURCE_LIKE_FIELDS:
                if node.strip().lower() in _SOURCE_FILE_ABSENT_SENTINELS:
                    log.info(
                        "role=%r %s payload field %r=%r is an absent-value "
                        "sentinel; treating as omitted and admitting the delegate",
                        role.name,
                        intent_type.value,
                        key,
                        node,
                    )
                    return
                if any(
                    self._path_in_source_allowlist(c) or self._path_under_session(c)
                    for c in source_file_candidates(node)
                ):
                    return
                raise PolicyDenied(
                    f"role={role.name!r} {intent_type.value} payload field "
                    f"{key!r}={node!r} is not under session_dir or a trusted "
                    f"installed source scope from "
                    f"{list(resolve_source_file_allowlist())!r}",
                    rule="source_file_outside_trusted_scope",
                    hint=(
                        "source_file and framework_source_root must resolve under "
                        "an active site/dist-packages, configured framework root, "
                        "or session directory"
                    ),
                )
            if key not in PATH_LIKE_FIELDS:
                return
            if key in ROCM_WRITE_PATH_FIELDS and not is_rocm_hip_writable_path(node):
                raise PolicyDenied(
                    f"role={role.name!r} {intent_type.value} payload field "
                    f"{key!r}={node!r} is a ROCm runtime path, not HIP source",
                    rule="rocm_runtime_write_denied",
                    hint="ROCm writes are limited to source, header, and CMake files",
                )
            if not self._path_under_session(node):
                if key in {"target_file", "resolved_patch_targets"} and trusted_framework_targets:
                    try:
                        resolved = str(Path(node).resolve())
                    except (OSError, RuntimeError):
                        resolved = ""
                    if resolved in trusted_framework_targets:
                        return
                # Multi-node profile traces live outside session_dir; allow trace-input fields against the trace allowlist.
                if key in TRACE_PATH_LIKE_FIELDS and self._path_in_trace_allowlist(node):
                    return
                raise PolicyDenied(
                    f"role={role.name!r} {intent_type.value} payload field "
                    f"{key!r}={node!r} escapes session_dir={self.session_dir!s}",
                    rule="path_outside_session_dir",
                    hint=(
                        "emit paths verbatim from SharedState (e.g. "
                        "last_profile_trace) or under SESSION_DIR; "
                        "multi-node trace fields may also resolve under "
                        f"{list(_trace_path_allowlist())!r}"
                    ),
                )

        visit(payload, ())

    def _validate_robustness_only(self, role: "AgentRole", intent_type: IntentType, payload: dict[str, Any]) -> None:
        """Enforce that only allowed roles emit robustness-only intents."""
        # Per-intent source override takes precedence; default is robustness-only.
        allowed_sources = _ROBUSTNESS_ONLY_INTENT_SOURCES.get(
            intent_type,
            ROBUSTNESS_ONLY_SOURCE_ALLOWLIST,
        )
        if role.name not in allowed_sources:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit {intent_type.value} (allowed: {sorted(allowed_sources)!r})",
                rule="robustness_only_source",
            )
        if intent_type == IntentType.PRUNE_BRANCH:
            family = str(payload.get("family", "")).strip()
            if not family:
                raise PolicyDenied("prune_branch missing family", rule="payload")
            scope = str(payload.get("scope") or PRUNE_BRANCH_SCOPE_FAMILY).strip()
            if scope not in PRUNE_BRANCH_ALLOWED_SCOPES:
                raise PolicyDenied(
                    f"prune_branch scope={scope!r} not allowed (allowed: {sorted(PRUNE_BRANCH_ALLOWED_SCOPES)!r})",
                    rule="prune_scope",
                    hint=(
                        f"{PRUNE_BRANCH_SCOPE_FAMILY!r} retires the action for "
                        f"the rest of the run; {PRUNE_BRANCH_SCOPE_QUEUED!r} "
                        f"only cancels the queued backlog."
                    ),
                )


# Policy-denial write-owner functions: they take ``state`` first and own the denial-streak bookkeeping + its prompt
# summary.
def record_policy_denial(
    state,
    *,
    action_name: str,
    rule: str,
    hint: str,
    intent_type: str,
    tick: int,
    intent_payload: dict[str, Any] | None = None,
) -> int:
    """Append a PolicyGate denial row and bump the per-(action, rule) streak."""
    from ..state.shared_state import _now_iso

    key = f"{action_name or '*'}:{rule}"
    streak = int(state.policy_denial_streak.get(key, 0)) + 1
    state.policy_denial_streak[key] = streak
    entry = {
        "tick": int(tick),
        "action_name": action_name or "",
        "rule": rule,
        "hint": hint or "",
        "intent_type": intent_type,
        "streak": streak,
        "ts": _now_iso(),
    }
    if intent_payload:
        entry["intent_payload_keys"] = sorted(intent_payload.keys())
    history = list(state.policy_denial_history or [])
    history.append(entry)
    if len(history) > state._POLICY_DENIAL_HISTORY_CAP:
        history = history[-state._POLICY_DENIAL_HISTORY_CAP :]
    state.policy_denial_history = history
    return streak


def reset_policy_denial_streak(state, action_name: str) -> None:
    """Clear all consecutive-denial streaks for a given action."""
    if not action_name:
        return
    prefix = f"{action_name}:"
    state.policy_denial_streak = {
        k: v for k, v in (state.policy_denial_streak or {}).items() if not k.startswith(prefix)
    }


def to_policy_denial_summary(state, *, top_k: int = 6) -> str:
    """Render the most recent PolicyGate denials for prompt injection."""
    if not state.policy_denial_history:
        return ""
    rows = list(state.policy_denial_history)[-top_k:]
    lines = [f"=== Recent policy denials (newest last, total={len(state.policy_denial_history)}) ==="]
    for r in rows:
        lines.append(
            f"  tick={r.get('tick')} action={r.get('action_name')!r} "
            f"rule={r.get('rule')!r} streak={r.get('streak')} "
            f"hint={str(r.get('hint') or '')[:140]!r}"
        )
    return "\n".join(lines)


def validate_specialist_max_turns_raw(
    max_turns_raw: Any,
    *,
    where: str,
) -> None:
    """Validate an optional specialist ``max_turns`` dial."""
    if max_turns_raw is None:
        return
    try:
        max_turns = int(max_turns_raw)
    except (TypeError, ValueError) as exc:
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: {where} max_turns must be int, got {max_turns_raw!r}",
            rule="specialist_dispatch_source",
        ) from exc
    if max_turns < 0:
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: {where} max_turns={max_turns} must be >= 0",
            rule="specialist_dispatch_source",
            hint=(
                "Use a non-negative integer. "
                "0 = unbounded (bounded by the wall-clock budget); "
                "omit max_turns to use the default turn cap."
            ),
        )
    if max_turns > SPECIALIST_MAX_TURNS_HARD_CAP:
        raise PolicyDenied(
            f"delegate{{action='specialist'}}: {where} max_turns={max_turns} "
            f"exceeds the hard cap {SPECIALIST_MAX_TURNS_HARD_CAP}",
            rule="specialist_dispatch_source",
            hint=(
                f"max_turns must be <= {SPECIALIST_MAX_TURNS_HARD_CAP} "
                "(0 = unbounded; depth is bounded by the wall-clock "
                "budget, so omit max_turns unless capping a probe early)."
            ),
        )


def validate_freeform_wave_task(task: Any, *, index: int) -> str:
    """Validate one entry in a freeform specialist ``tasks`` wave."""
    if not isinstance(task, dict):
        raise PolicyDenied(
            f"delegate{{action='specialist',scope='freeform'}}: tasks[{index}] must be a dict",
            rule="specialist_freeform_wave_invalid_task",
            hint="Each wave entry must be an object with task_description.",
        )
    desc = str(task.get("task_description") or task.get("task_summary") or "").strip()
    PolicyGate._check_freeform_task_description(desc, where=f"tasks[{index}]")
    return desc


__all__ = [
    "CORE_STATE_FIELDS",
    "DELEGATE_ACTION_REQUIRED_PAYLOAD",
    "DELEGATE_ACTION_SOURCE_ALLOWLIST",
    "EXTEND_LEASE_MAX_SEC",
    "INTEGRATE_PATCH_PERMISSIVE_VERDICTS",
    "KERNEL_AGENT_OWNED_ACTIONS",
    "PATH_LIKE_FIELDS",
    "PRUNE_BRANCH_ALLOWED_SCOPES",
    "PRUNE_BRANCH_SCOPE_FAMILY",
    "PRUNE_BRANCH_SCOPE_QUEUED",
    "PolicyDenied",
    "PolicyGate",
    "patch_verdict_subject",
    "validate_freeform_wave_task",
    "validate_specialist_max_turns_raw",
    "REQUEST_ROUTING",
    "REVIEW_VERDICTS",
    "REVIEW_VERDICT_SOURCE_ALLOWLIST",
    "ROCM_WRITE_PATH_FIELDS",
    "ROBUSTNESS_ONLY_INTENTS",
    "ROBUSTNESS_ONLY_SOURCE_ALLOWLIST",
    "TRACE_PATH_LIKE_FIELDS",
    "SOURCE_LIKE_FIELDS",
]
