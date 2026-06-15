# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""PolicyGate — single chokepoint: every parsed Intent passes through ``validate_intent`` before side-effects."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .framework_paths import resolve_source_file_allowlist
from ..protocol.intent import Intent, IntentType
from ..protocol.action_surfaces import (
    COORDINATOR_INTERNAL_ACTIONS,
    INTERNAL_ONLY_ACTION_NAMES,
    KERNEL_OWNED_ACTIONS,
    ROBUSTNESS_DELEGATE_ONLY_ACTIONS,
)
from .phase_state import (
    PHASE_KERNEL,
    PHASE_NAMES,
    PHASE_SWEEP,
    is_action_allowed_in_phase,
    is_action_llm_proposable_in_phase_with_interleave,
    llm_proposable_actions_for_with_interleave,
)
from .specialist_domains import (
    KNOWLEDGE_DOMAIN_TAG_SET,
    SPECIALIST_DOMAIN_KEYS,
    SPECIALIST_MAX_TURNS_HARD_CAP,
    domain_for_tag,
    get_domain,
    normalize_dispatch_tags,
)
from .specialist_profile import (
    SCOPE_DOMAIN as SPECIALIST_SCOPE_DOMAIN,
    SCOPE_DOMAINS as SPECIALIST_SCOPE_DOMAINS,
    SCOPE_FREEFORM as SPECIALIST_SCOPE_FREEFORM,
    SCOPE_VALUES as SPECIALIST_SCOPE_VALUES,
)

if TYPE_CHECKING:  # pragma: no cover — type-only
    from .agent_role import AgentRole


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
    """True iff ``field_name`` is present at the top of ``payload`` OR nested under ``payload["params"]`` (robustness uses params)."""
    if _value_is_present(payload.get(field_name)):
        return True
    nested = payload.get("params")
    if isinstance(nested, dict) and _value_is_present(nested.get(field_name)):
        return True
    return False


class PolicyDenied(RuntimeError):
    """Intent rejected by PolicyGate.

    Attributes:
        rule: short identifier of the rule that fired.
        hint: optional one-line agent-actionable suggestion.
    """

    def __init__(self, reason: str, *, rule: str | None = None,
                 hint: str | None = None):
        """Initialise the denial with a human-readable reason and metadata.

        Args:
            reason (str): human-readable explanation passed to the base
                ``RuntimeError``; surfaced in logs and the policy_denied
                observation event.
            rule (str | None): short identifier of the rule that fired,
                used by the Coordinator to classify the denial. Defaults
                to ``None``.
            hint (str | None): optional one-line, agent-actionable
                suggestion describing the canonical fix. Defaults to
                ``None``.
        """
        super().__init__(reason)
        self.rule = rule
        self.hint = hint


FP8_ONLY_ACTIONS: frozenset[str] = frozenset({
    "gemm_tuning",
    "run_gemm_tuning",
})


# Per-action delegate source allowlist (action_name → source roles); unlisted actions fall through to the general delegate rules.
DELEGATE_ACTION_SOURCE_ALLOWLIST: dict[str, frozenset[str]] = {
    "recover": frozenset({"robustness"}),
}


# Per-action delegate required payload fields (minimum evidence for the audit trail; missing/empty raise PolicyDenied).
DELEGATE_ACTION_REQUIRED_PAYLOAD: dict[str, tuple[str, ...]] = {
    "recover": ("reason", "evidence"),
}


# Specialist dispatch action name (central so R2 sub-rules enforce the contract uniformly).
SPECIALIST_ACTION_NAME: str = "specialist"

# Orchestrator-side patch integration step (EXPLORE phase, gated by a Critic verdict).
INTEGRATE_PATCH_ACTION_NAME: str = "integrate_patch"

# Merged explore action.
EXPLORE_ACTION_NAME: str = "explore"

# Sweep actions; named constants so the ``*_phase_singleton`` rules have a single source of truth.
SWEEP_ACTION_NAME: str = "sweep"
CONC_SWEEP_ACTION_NAME: str = "conc_sweep"

# Specialist / Explore parallelism caps — single source of truth across layers.
# Research-lane ceiling fallback used when the GPU count cannot be probed.
RESEARCH_LANE_CEILING_FALLBACK: int = 2


def detect_gpu_count() -> int:
    """Best-effort visible-GPU count: env masks first, then ``rocm-smi``; 0 when nothing can be probed."""
    for env_name in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        raw = raw.strip()
        if raw == "":
            return 0
        ids = [tok for tok in raw.split(",") if tok.strip() != ""]
        if ids:
            return len(ids)
    try:
        import subprocess

        proc = subprocess.run(
            ["rocm-smi", "--showid"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, OSError, ValueError):
        return 0
    except Exception:  # noqa: BLE001
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
            return max(0, int(
                getattr(shared_state, "gpu_specialist_capacity", 0) or 0
            ))
        except (TypeError, ValueError):
            return 0
    try:
        return max(0, int(
            os.environ.get("INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY", "0")
            or "0"
        ))
    except ValueError:
        return 0


# Ceiling snapshot at import for callers needing a plain int; recomputed lazily by :func:`research_lane_ceiling`.
MAX_RESEARCH_LANE_CAPACITY: int = research_lane_ceiling()

# Canonical name of the LLM-sub-agent resource lane shared by specialists.
RESEARCH_LANE_NAME: str = "research_lane"
DEFAULT_SPECIALIST_MAX_PROPOSALS: int = 12

# Verdicts that allow ``integrate_patch`` without an operator override (``advise`` = soft approval, ``approve`` = green light).
INTEGRATE_PATCH_PERMISSIVE_VERDICTS: frozenset[str] = frozenset({
    "approve", "advise",
})

# Source roles allowed to dispatch a specialist via ``delegate{action='specialist'}``.
SPECIALIST_DISPATCH_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"orchestration"})

# Free-form (``scope='freeform'``) sanity-gate limits (absorbed from the
# retired dynamic_specialist wave channel).
SPECIALIST_FREEFORM_WAVE_MAX: int = 8
SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS: int = 8000
# Lightweight mechanical red-line tripwire over free-form task descriptions:
# obviously-destructive host commands a dispatch must never embed. This is a
# fail-fast sanity check, NOT a security boundary — the isolated worktree, the
# Critic review, and the integrate_patch gate remain the real boundaries.
_FREEFORM_REDLINE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\brm\s+-rf?\s+(?:/|~|\$HOME|\*)", re.IGNORECASE),
    re.compile(r"\bmkfs\.", re.IGNORECASE),
    re.compile(r"\bdd\s+if=.*\bof=/dev/", re.IGNORECASE),
    re.compile(r">\s*/dev/sd[a-z]"),
    re.compile(r":\(\)\s*\{.*\};\s*:"),  # fork bomb
    re.compile(r"\bshutdown\b|\breboot\b", re.IGNORECASE),
)

# Prefix the SubAgentRunner stamps on specialist emit-intents (``from_agent='specialist:<task_id>'``).
SPECIALIST_FROM_AGENT_PREFIX: str = "specialist:"


# R4 / R5 — external tool whitelist registry (single source of truth for PolicyGate + SpecialistRunner).

#: KB *write* surfaces. R4 ``kb_write_unauthorized`` denies any intent invoking one.
KB_WRITE_TOOL_NAMES: frozenset[str] = frozenset({
    "mcp__cortex_kb__propose_point",
})

#: KB *readonly* surfaces. R5 ``tool_whitelist_role`` requires a specialist sub-agent caller.
CORTEX_KB_READ_TOOL_NAMES: frozenset[str] = frozenset({
    "mcp__cortex_kb__traverse",
    "mcp__cortex_kb__find_recipe",
    "mcp__cortex_kb__query",
})

#: PR Monitor *readonly* surfaces. R5 same role gating.
PR_MONITOR_TOOL_NAMES: frozenset[str] = frozenset({
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
})

#: Web tools. R5 — specialist-only (other roles get ``tool_whitelist_role``); usable in any phase.
WEB_TOOL_NAMES: frozenset[str] = frozenset({"WebSearch", "WebFetch"})

#: Role→allowed-toolset map (R5). Only the specialist sub-agent touches external knowledge tools.
TOOL_WHITELIST_BY_ROLE: dict[str, frozenset[str]] = {
    "specialist": (
        WEB_TOOL_NAMES
        | PR_MONITOR_TOOL_NAMES
        | CORTEX_KB_READ_TOOL_NAMES
    ),
    # Empty sets listed explicitly so a role-name typo is a key error, not a silent allow.
    "orchestration": frozenset(),
    "kernel": frozenset(),
    "critic": frozenset(),
    "robustness": frozenset(),
}

#: Convenience superset of every known external tool name (R4 collision check).
ALL_KNOWN_EXTERNAL_TOOL_NAMES: frozenset[str] = (
    KB_WRITE_TOOL_NAMES
    | CORTEX_KB_READ_TOOL_NAMES
    | PR_MONITOR_TOOL_NAMES
    | WEB_TOOL_NAMES
)


# Synthetic stub used as ``role`` for specialist path-containment checks (only ``name`` is needed).
class _SpecialistPseudoRole:
    """Minimal stand-in role used when path-validating specialist intents.

    Specialist intents are routed through ``_validate_specialist_*`` rather
    than the conventional ``role.allowed_intents`` matrix, so the path
    containment check only needs a ``name`` attribute for its error
    messages. This stub supplies that single field.

    Attributes:
        name (str): the synthetic role name, always ``"specialist"``.
    """

    name = "specialist"


_SPECIALIST_PSEUDO_ROLE = _SpecialistPseudoRole()


# REQUEST/RESPONSE routing matrix (DESIGN §7.6 / §13.4): source role → allowed target_agents (only orchestration→kernel).
REQUEST_ROUTING: dict[str, frozenset[str]] = {
    "orchestration": frozenset({"kernel"}),
}


# Critic-only: REVIEW_VERDICT (DESIGN §18.2)
REVIEW_VERDICT_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"critic"})

# Verdict vocabulary for review_verdict (DESIGN §18.2)
REVIEW_VERDICTS: frozenset[str] = frozenset({
    "approve", "reject", "redirect", "advise", "needs_review",
})


# Robustness-only: kill_task + scheduling-police intents (DESIGN §7.4 / §19.3)
KILL_TASK_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"robustness"})
KILL_TASK_ALLOWED_SCOPES: frozenset[str] = frozenset({"task"})

ROBUSTNESS_ONLY_INTENTS: frozenset[IntentType] = frozenset({
    IntentType.FORCE_DISPATCH,
    IntentType.PRUNE_BRANCH,
    IntentType.ESCALATE_STRATEGY_CHANGE,
})
ROBUSTNESS_ONLY_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"robustness"})

# Per-intent source override: PRUNE_BRANCH + ESCALATE_STRATEGY_CHANGE widen to orchestration; FORCE_DISPATCH stays robustness-only.
_ROBUSTNESS_ONLY_INTENT_SOURCES: dict[IntentType, frozenset[str]] = {
    IntentType.PRUNE_BRANCH: frozenset({"robustness", "orchestration"}),
    IntentType.ESCALATE_STRATEGY_CHANGE: frozenset({
        "robustness", "orchestration",
    }),
}


# SESSION_DIR path containment: PATH_LIKE_FIELDS must point inside session_dir or a framework source allowlist (checked recursively).
PATH_LIKE_FIELDS: frozenset[str] = frozenset({
    "trace_input",
    "candidates_path",
    "patch_path",
    "target_file",
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
})

# `source_file` is special: may match :func:`resolve_source_file_allowlist` (framework trees outside session_dir, resolved at check time).
SOURCE_LIKE_FIELDS: frozenset[str] = frozenset({"source_file"})


# Multi-node profile trace dirs live outside session_dir but must be referenceable by trace_dir / main_trace_path / trace_input (runtime-resolved).
def _trace_path_allowlist() -> tuple[str, ...]:
    """Multi-node profile trace path-prefix allowlist (runtime-resolved; trailing ``/`` is load-bearing for the startswith check)."""
    from ..paths import mn_profile_trace_root
    root = str(mn_profile_trace_root()).rstrip("/") + "/"
    return (root,)

# Subset of PATH_LIKE_FIELDS that also accept :func:`_trace_path_allowlist` (others stay strictly session-rooted).
TRACE_PATH_LIKE_FIELDS: frozenset[str] = frozenset({
    "trace_dir",
    "main_trace_path",
    "trace_input",
})


# Core SharedState fields that only the Coordinator may mutate.
CORE_STATE_FIELDS: frozenset[str] = frozenset({
    "current_best",
    "stop_reason",
    "last_tick_exception",
    "cumulative_gain",
    "cumulative_gain_validated",
    "cumulative_gain_validated_ts",
    "cumulative_gain_validated_stack_len",
    "baseline_tput",
    "baseline_accuracy",
    "session_id",
    "model_path",
    "model_name",
    "model_class",
    "start_ts",
    "max_minutes",
    # fact-layer KEEP ledger; Coordinator is the sole writer.
    "optimization_stack",
    "gain_per_stack_entry",
    # schema_version migration breadcrumb; LLM must not roll state.json back.
    "schema_version",
    # Cortex KB integration fields (Coordinator-only writes; LLM read is fine).
    "cortex_session_id",
    "cortex_session_summary",
    "warm_start_recipe",
    "warm_start_pitfalls",
    "warm_start_lessons",
    "warm_start_ts",
    "warm_start_context",
    # KB tag completeness (Coordinator-populated from manifest + baseline config; LLM reads via prompt, Coordinator writes).
    "stack_fingerprint_meta",
    "baseline_workload_extra",
    # warm-recipe replay one-shot guard + outcome; LLM cannot edit (bypasses replay budget).
    "warm_replay_attempted",
    "warm_replay_outcome",
    "warm_history_injected",
    # phase state machine fields (managed by ``Coordinator._advance_phase_if_needed``).
    "phase",
    "phase_started_ts",
    "phase_started_unix",
    "phase_history",
    "phase_budget_pct",
    # R1/R2/R7 cyclic phase-machine state (Coordinator-only writers:
    # ``_apply_macro_cycle_reloop`` + ``should_reloop_to_explore`` accounting).
    # Locked so an LLM ``update_state`` cannot forge the macro-cycle counter,
    # the per-cycle budget window, the per-cycle gain anchor / no-gain streak
    # (which drive global-convergence + the decaying acceptance curve), or the
    # cross-cycle bottleneck-switch handoff.
    "macro_cycle",
    "cycle_minutes",
    "gain_at_cycle_start",
    "no_gain_cycle_streak",
    "pending_bottleneck_switch",
    "last_cycle_bottleneck",
    # operator-facing lifecycle event log (#266). Coordinator-only writer
    # (SharedState.record_lifecycle_event); LLM update_state must not be
    # able to forge "phase X finished, outputs at <path>" events.
    "lifecycle",
    # specialist sub-agent ledger; LLM cannot inject entries (proposals go via the R3 path).
    "specialist_rounds",
    "specialist_domain_empty_streak",
    # per-kb_anchor coverage counters (point 1); Coordinator-only writers.
    "rounds_since_last_specialist",
    "rounds_since_last_keep",
    "last_specialist",
    # research_lane / GPU capacity set once at CLI/manifest time; locked so the LLM can't raise it mid-flight.
    "research_lane_capacity",
    "gpu_specialist_capacity",
    # phase-machine escalation plumbing; LLM blocked (defense in depth) so it can't force ``skip_to_close``.
    "pending_escalate_hint",
    "last_consumed_escalate_hint",
    "last_consumed_escalate_hint_ts",
    "plateau_overrides",
    # CLOSE-phase sequencer flag; LLM must not toggle it (would skip cli.finally's safety net).
    "close_sequence_done",
    # explore search ledger; Coordinator-only writers (LLM rewrite would bypass dedup-by-fingerprint).
    "explore_search",
    # structured gaps ledger; single writer (``_refresh_gaps``) so the LLM can't inject fake gaps.
    "gaps",
    # Orchestration working-memory checkpoint; Coordinator-authored (LLM must not self-author durable memory).
    "orchestration_memory",
    # FRAMEWORK_PR per-repo discovery budget; set once, locked against LLM inflation.
    "framework_pr_max_candidates",
    # Advisory model-architecture profile from the SKILL launcher; locked as the sole source of truth.
    "model_arch",
    # Architecture-identity tags from config.json fanned into recipe-snapshot extras; locked against pollution.
    "model_architectures",
    "model_type",
    # Multimodal text-fallback degraded-run markers (cli._preflight). Coordinator/
    # preflight are the sole writers; locked so an LLM update_state cannot forge
    # or clear "degraded run" — it drives the final report's degraded warning
    # (report.py) and must reflect the real preflight verdict, not LLM intent.
    "degraded_mode",
    "model_warnings",
})


@dataclass
class PolicyGate:
    """Validate every intent emitted by an agent reactor.

    ``strict_paths`` (or ``$INFERENCE_OPTIMIZER_STRICT_PATHS=1``) requires
    PATH_LIKE_FIELDS to resolve under session_dir / the source-file allowlist.
    """

    role_registry: dict[str, "AgentRole"]
    action_registry: Any | None = None
    session_dir: Path | None = None
    strict_paths: bool = False
    shared_state: Any | None = None
    # R1 phase enforcement: False (default) warns only; ``INFERENCE_OPTIMIZER_STRICT_PHASE=1`` fails closed.
    strict_phase: bool = False

    def __post_init__(self) -> None:  # noqa: D401 — dataclass hook
        """Apply environment overrides for strict-mode flags.

        Lets ``INFERENCE_OPTIMIZER_STRICT_PATHS`` and
        ``INFERENCE_OPTIMIZER_STRICT_PHASE`` enable strict behavior
        without threading a constructor argument through every caller.
        """
        # Allow env to enable strict mode without threading a constructor arg through every caller.
        import os as _os
        if not self.strict_paths and _os.environ.get(
            "INFERENCE_OPTIMIZER_STRICT_PATHS", ""
        ).strip() in ("1", "true", "yes"):
            self.strict_paths = True
        if not self.strict_phase and _os.environ.get(
            "INFERENCE_OPTIMIZER_STRICT_PHASE", ""
        ).strip() in ("1", "true", "yes"):
            self.strict_phase = True

    # Public API
    def validate_intent(self, from_agent: str, intent: Intent) -> None:
        """Raise :class:`PolicyDenied` if the intent is not allowed (cheapest checks first: role → allowed_intents → structural → cross-source)."""
        # specialist sub-agents emit under an ephemeral ``specialist:<task_id>`` identity routed to a synthetic role.
        if from_agent.startswith(SPECIALIST_FROM_AGENT_PREFIX):
            self._validate_specialist_intent(from_agent, intent)
            self._validate_payload_paths(
                _SPECIALIST_PSEUDO_ROLE, intent.type, intent.payload or {},
            )
            return

        role = self.role_registry.get(from_agent)
        if role is None:
            raise PolicyDenied(f"unknown agent {from_agent!r}", rule="role")

        if intent.type not in role.allowed_intents:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit intent_type={intent.type.value!r}",
                rule="role",
            )

        closing_denied = self._closing_phase_denial(from_agent, intent)
        if closing_denied is not None:
            raise closing_denied

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
        elif intent.type == IntentType.KILL_TASK:
            self._validate_kill_task(role, payload)
        elif intent.type in ROBUSTNESS_ONLY_INTENTS:
            self._validate_robustness_only(role, intent.type, payload)
        # ANSWER / ASK_QUESTION / UPDATE_PERSONA / ALERT carry no extra checks beyond the role gate.

        # Path-containment guard for PATH_LIKE_FIELDS in the payload.
        self._validate_payload_paths(role, intent.type, payload)

    def _closing_phase_denial(
        self, source: str, intent: Intent,
    ) -> PolicyDenied | None:
        """During closing phase, allow only harmless intents and ``report`` proposals."""
        state = self.shared_state
        if state is None or not getattr(state, "closing_phase", False):
            return None
        if intent.type in (
            IntentType.SEND_MESSAGE,
            IntentType.UPDATE_PERSONA,
            IntentType.ALERT,
            IntentType.ASK_QUESTION,
            IntentType.ANSWER,
        ):
            return None
        if (
            intent.type == IntentType.PROPOSE_ACTION
            and (intent.payload or {}).get("action_name") == "report"
        ):
            return None
        return PolicyDenied(
            f"closing_phase: {intent.type.value} denied "
            f"(only `report` proposals allowed during wind-down)",
            rule="closing_phase_only_report",
            hint="run is winding down; new tasks are dropped",
        )

    def allowed_tools_for_agent(self, agent_name: str) -> list[str]:
        """Return the Claude tool list a reactor may use (Codex → []; Claude → emit_intent; orchestration also gets context-pull tools + sandboxed Read)."""
        role = self.role_registry.get(agent_name)
        if role is None:
            return []
        if role.no_tools:
            return []
        tools = ["emit_intent"]
        if agent_name == "orchestration":
            from .backends.mcp_context_tools import CONTEXT_TOOL_NAMES
            tools.extend(CONTEXT_TOOL_NAMES)
            tools.append("Read")
        return tools

    def allowed_tools_for_action(self, action_name: str) -> list[str]:
        """Per-action tool intersection; action's declared ``allowed_tools`` or the default ``["emit_intent"]``."""
        if self.action_registry is None:
            return ["emit_intent"]
        meta = self.action_registry.get(action_name)
        if meta is None:
            return ["emit_intent"]
        return list(meta.allowed_tools)

    # Per-intent validators
    def _validate_delegate(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate a ``DELEGATE`` intent against the full delegate rule set.

        Enforces, in order: the role's ``can_delegate_side_effects``
        capability, presence of ``action_name``, the
        analysis/internal-only gate, the kernel-owned-action guard, and
        the per-action specialised paths (``specialist`` / ``dynamic_action``
        / ``integrate_patch`` / ``explore`` / ``sweep``). It then applies
        the FP8-only, gain-driven and explore-minimum kernel_opt gates, the
        ActionRegistry unknown-action lookup, per-action source and
        required-payload guards, the phase-compatibility check, and the
        external-tool collision guards (R4 / R5).

        Args:
            role (AgentRole): the resolved role of the emitting agent.
            payload (dict[str, Any]): the delegate intent payload, expected
                to carry ``action_name`` and optional ``params``.

        Returns:
            None: returns silently when the delegate is permitted.

        Raises:
            PolicyDenied: if any delegate rule fails; the ``rule``
                attribute identifies which guard fired.
        """
        if not role.can_delegate_side_effects:
            raise PolicyDenied(
                f"role={role.name!r} cannot delegate side-effecting actions",
                rule="role",
            )
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied("delegate intent missing action_name", rule="payload")
        # Plan A — kernel-owned actions are not directly delegatable.
        if action_name in KERNEL_OWNED_ACTIONS:
            raise PolicyDenied(
                f"action={action_name!r} is owned by the kernel agent; "
                f"emit REQUEST(target_agent='kernel', kind='...') instead "
                f"of delegate(action_name={action_name!r})",
                rule="kernel_owned_by_kernel_agent",
            )
        # R2 ``specialist`` bypasses ActionRegistry; its contract is enforced by ``_validate_specialist_dispatch``.
        if action_name == SPECIALIST_ACTION_NAME:
            self._validate_specialist_dispatch(role, payload)
            self._validate_phase_action(role, action_name, intent_kind="delegate")
            return
        # ``integrate_patch`` requires a non-reject Critic verdict (``bypass_critic=True`` overrides, audit-visible).
        if action_name == INTEGRATE_PATCH_ACTION_NAME:
            self._validate_integrate_patch_critic_gate(payload)
        # sweep_phase_singleton: deny LLM sweep once the auto-enqueue landed (concurrent sweeps crash both vllm engines).
        if action_name == SWEEP_ACTION_NAME:
            self._validate_sweep_singleton(payload, intent_kind="delegate")
        # conc_sweep_phase_singleton (Bug #11): block duplicate conc_sweep proposals.
        if action_name == CONC_SWEEP_ACTION_NAME:
            self._validate_conc_sweep_singleton(payload, intent_kind="delegate")
        self._validate_fp8_only_action(action_name, intent_kind="delegate")
        # Refuse delegate for unknown action names when an ActionRegistry is wired (no registry → fall through).
        if self.action_registry is not None and self.action_registry.get(action_name) is None:
            raise PolicyDenied(
                f"unknown action_name={action_name!r} (not in ActionRegistry)",
                rule="unknown_action",
                hint="register a yaml under inference_optimizer/actions/_meta/<name>.yaml",
            )
        # Per-action source allowlist (e.g. ``recover`` is robustness-only).
        allowed_sources = DELEGATE_ACTION_SOURCE_ALLOWLIST.get(action_name)
        if allowed_sources is not None and role.name not in allowed_sources:
            raise PolicyDenied(
                f"role={role.name!r} cannot delegate action={action_name!r} "
                f"(allowed: {sorted(allowed_sources)!r})",
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
            missing = [
                field_name
                for field_name in required
                if not _delegate_field_present(payload, field_name)
            ]
            if missing:
                raise PolicyDenied(
                    f"delegate(action_name={action_name!r}) missing required "
                    f"payload field(s): {missing!r}",
                    rule="delegate_action_evidence",
                    hint=(
                        "side-effecting delegates must carry the symptom "
                        "evidence that justified them (e.g. "
                        "{'reason': 'gpu_memory_leaked', "
                        "'evidence': {...}})"
                    ),
                )
        # R1 phase_incompatible. Runs after the structural checks so cheaper denials win.
        self._validate_phase_action(role, action_name, intent_kind="delegate")
        # R4 / R5 — block a delegate whose action_name invokes an external tool.
        self._validate_no_kb_write_collision(
            action_name, intent_kind="delegate",
        )
        self._validate_tool_whitelist_collision(
            role.name, action_name, intent_kind="delegate",
        )

    def _validate_propose_action(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate a ``PROPOSE_ACTION`` intent (the advisory channel).

        Requires ``action_name`` and applies the internal-only gate. The
        ActionRegistry lookup here is soft — unknown names are only
        rejected when a registry is wired and the action is neither
        registered nor kernel-owned. Mirrors the delegate channel's
        explore-grid, sweep-singleton, FP8-only, gain-driven /
        explore-minimum kernel_opt, phase, and external-tool collision
        gates so an LLM cannot sidestep them by proposing instead of
        delegating.

        Args:
            role (AgentRole): the resolved role of the emitting agent.
            payload (dict[str, Any]): the propose_action payload, expected
                to carry ``action_name`` and optional ``params``.

        Returns:
            None: returns silently when the proposal is permitted.

        Raises:
            PolicyDenied: if ``action_name`` is missing, unknown (with a
                registry wired), internal-only, or fails one of the
                mirrored action gates.
        """
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied("propose_action missing action_name", rule="payload")
        # Kernel-owned actions are REQUEST-only — mirror the delegate guard so the
        # ownership contract is enforced on BOTH the propose and delegate channels
        # (action_surfaces.KERNEL_OWNED_ACTIONS). Without this a propose_action
        # would pass the gate and materialize as a ``kind=<kernel action>`` task
        # that bypasses the kernel REQUEST handler.
        if action_name in KERNEL_OWNED_ACTIONS:
            raise PolicyDenied(
                f"action={action_name!r} is owned by the kernel agent; "
                f"emit REQUEST(target_agent='kernel', kind='...') instead "
                f"of propose_action(action_name={action_name!r})",
                rule="kernel_owned_by_kernel_agent",
            )
        # Soft check — reject only if registry is wired AND name is unknown.
        if (
            self.action_registry is not None
            and self.action_registry.get(action_name) is None
        ):
            raise PolicyDenied(
                f"propose_action: unknown action_name={action_name!r} "
                f"(not in ActionRegistry)",
                rule="unknown_action",
            )
        # sweep_phase_singleton (defense in depth on the propose_action channel).
        if action_name == SWEEP_ACTION_NAME:
            self._validate_sweep_singleton(
                payload, intent_kind="propose_action",
            )
        # conc_sweep_phase_singleton (Bug #11) on propose_action.
        if action_name == CONC_SWEEP_ACTION_NAME:
            self._validate_conc_sweep_singleton(
                payload, intent_kind="propose_action",
            )
        # Per-action source allowlist (e.g. ``recover`` is robustness-only); mirrors the delegate-path guard.
        allowed_sources = DELEGATE_ACTION_SOURCE_ALLOWLIST.get(action_name)
        if allowed_sources is not None and role.name not in allowed_sources:
            raise PolicyDenied(
                f"role={role.name!r} cannot propose action={action_name!r} "
                f"(allowed: {sorted(allowed_sources)!r})",
                rule="propose_action_source",
                hint=(
                    "side-effecting actions like `recover` are reserved for "
                    "the robustness agent; emit an ALERT and let robustness "
                    "escalate via its action-ladder instead"
                ),
            )
        self._validate_fp8_only_action(action_name, intent_kind="propose_action")
        # R1 phase_incompatible.
        self._validate_phase_action(role, action_name, intent_kind="propose_action")
        # R4 / R5 — defense in depth on propose_action.
        self._validate_no_kb_write_collision(
            action_name, intent_kind="propose_action",
        )
        self._validate_tool_whitelist_collision(
            role.name, action_name, intent_kind="propose_action",
        )

    def _validate_state_transition(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate an ``UPDATE_STATE`` intent's ``changes`` against core fields.

        Requires a non-empty ``changes`` dict. Roles with
        ``can_mutate_core_state`` (the Coordinator) may write anything;
        every other role is blocked from mutating any field in
        :data:`CORE_STATE_FIELDS`.

        Args:
            role (AgentRole): the resolved role of the emitting agent.
            payload (dict[str, Any]): the update_state payload, expected to
                contain a ``changes`` mapping of field → new value.

        Returns:
            None: returns silently when the state transition is permitted.

        Raises:
            PolicyDenied: if ``changes`` is missing/empty, or a
                non-privileged role attempts to mutate core state fields.
        """
        changes = payload.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise PolicyDenied(
                "update_state.payload.changes must be a non-empty dict",
                rule="payload",
                hint=("include at least one allowed field, e.g. "
                      "{'changes': {'current_action': '<action_name>'}}"),
            )
        violating = sorted(set(changes.keys()) & CORE_STATE_FIELDS)
        if violating:
            raise PolicyDenied(
                f"role={role.name!r} cannot mutate core state fields: {violating!r}",
                rule="state_field",
            )

    def _validate_send_message_topic(self, payload: dict[str, Any]) -> None:
        """Require a non-empty ``topic`` on a ``SEND_MESSAGE`` intent.

        Unknown topics are intentionally not rejected here — the
        Coordinator soft-degrades them to ``"observation"`` (DESIGN
        §13.2) — so agents can still surface unstructured observations.

        Args:
            payload (dict[str, Any]): the send_message payload, expected to
                carry a ``topic`` string.

        Returns:
            None: returns silently when a topic is present.

        Raises:
            PolicyDenied: with ``rule='payload'`` when ``topic`` is missing
                or blank.
        """
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            raise PolicyDenied("send_message missing topic", rule="payload")
        # Unknown topics are soft-degraded by the Coordinator to "observation" (DESIGN §13.2); not rejected here.

    def _validate_request(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate a ``REQUEST`` intent against the routing matrix.

        Checks that the role may emit a REQUEST at all (per
        :data:`REQUEST_ROUTING`), that ``target_agent`` is in the role's
        allowed-target set, and that ``kind`` is present. For
        orchestration→kernel requests the ``kind`` is treated as the action
        name, so the internal-only, phase, FP8-only and external-tool
        collision guards are applied to it as defense in depth.

        Args:
            role (AgentRole): the resolved role of the emitting agent.
            payload (dict[str, Any]): the request payload, expected to
                carry ``target_agent`` and ``kind``.

        Returns:
            None: returns silently when the request is permitted.

        Raises:
            PolicyDenied: if the role cannot emit REQUEST, the target is
                missing/disallowed, ``kind`` is missing, or one of the
                applied action guards fires.
        """
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
                f"role={role.name!r} cannot request target_agent={target!r} "
                f"(allowed: {sorted(targets)!r})",
                rule="request_target",
            )
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            raise PolicyDenied("request missing kind", rule="payload")
        # R1 phase_incompatible: treat REQUEST kind as the action name for kernel-owned + coordinator-internal kinds.
        if (
            target == "kernel" and kind in KERNEL_OWNED_ACTIONS
        ) or kind in COORDINATOR_INTERNAL_ACTIONS:
            self._validate_phase_action(role, kind, intent_kind="request")
        self._validate_fp8_only_action(kind, intent_kind="request")
        # R4 / R5 — a REQUEST.kind cannot smuggle a KB write / external tool either.
        self._validate_no_kb_write_collision(kind, intent_kind="request")
        self._validate_tool_whitelist_collision(
            role.name, kind, intent_kind="request",
        )

    def _validate_response(self, payload: dict[str, Any]) -> None:
        """Require ``in_reply_to`` and ``kind`` on a ``RESPONSE`` intent.

        Args:
            payload (dict[str, Any]): the response payload, expected to
                carry ``in_reply_to`` (the message id being answered) and
                ``kind``.

        Returns:
            None: returns silently when both fields are present.

        Raises:
            PolicyDenied: with ``rule='payload'`` when ``in_reply_to`` or
                ``kind`` is missing or blank.
        """
        in_reply_to = str(payload.get("in_reply_to", "")).strip()
        if not in_reply_to:
            raise PolicyDenied("response missing in_reply_to", rule="payload")
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            raise PolicyDenied("response missing kind", rule="payload")

    def _validate_review_verdict(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate a ``REVIEW_VERDICT`` intent (Critic-only).

        Enforces that the source role is on
        :data:`REVIEW_VERDICT_SOURCE_ALLOWLIST`, that
        ``target_proposal_msg_id`` is present, and that exactly one of the
        single ``verdict`` field or the per-variant ``verdict_map`` is
        supplied. Every verdict string (single or per-variant) must belong
        to the closed :data:`REVIEW_VERDICTS` vocabulary.

        Args:
            role (AgentRole): the resolved role of the emitting agent.
            payload (dict[str, Any]): the review_verdict payload, carrying
                ``target_proposal_msg_id`` and either ``verdict`` or
                ``verdict_map``.

        Returns:
            None: returns silently when the verdict is well-formed.

        Raises:
            PolicyDenied: if the role is not a Critic, the target id is
                missing, neither/both verdict forms are present, or a
                verdict string is outside ``REVIEW_VERDICTS``.
        """
        if role.name not in REVIEW_VERDICT_SOURCE_ALLOWLIST:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit review_verdict "
                f"(allowed: {sorted(REVIEW_VERDICT_SOURCE_ALLOWLIST)!r})",
                rule="review_verdict_source",
            )
        target = str(payload.get("target_proposal_msg_id", "")).strip()
        if not target:
            raise PolicyDenied(
                "review_verdict missing target_proposal_msg_id", rule="payload",
            )
        # Accept the legacy single ``verdict`` or the per-variant ``verdict_map``; here we validate verdict strings against REVIEW_VERDICTS.
        has_single = "verdict" in payload
        verdict_map = payload.get("verdict_map")
        has_map = isinstance(verdict_map, dict) and bool(verdict_map)
        if has_single == has_map:
            # Both or neither — defense in depth.
            raise PolicyDenied(
                "review_verdict: exactly one of 'verdict' or "
                "'verdict_map' must be present",
                rule="payload",
                hint=(
                    "single-proposal review: emit {target_proposal_msg_id, "
                    "verdict, reasoning}. Explore batch review: emit "
                    "{target_proposal_msg_id, verdict_map: {variant_name: "
                    "{verdict, rationale?}}}"
                ),
            )
        if has_single:
            verdict = str(payload.get("verdict", "")).strip()
            if verdict not in REVIEW_VERDICTS:
                raise PolicyDenied(
                    f"review_verdict.verdict={verdict!r} not in allowed set "
                    f"{sorted(REVIEW_VERDICTS)!r}",
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
                    hint=(
                        "every per-variant verdict must be one of "
                        "approve/reject/redirect/advise/needs_review"
                    ),
                )

    # NOTE: no ``framework_atom_action_unsupported`` rule exists; the guards
    # that enforce this live in ``tests/test_policy_atom_invariants.py``.

    # R1 phase_incompatible
    def _validate_phase_action(
        self,
        role: "AgentRole",
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject an action the LLM cannot propose in the current phase (``strict_phase`` True raises, False warns; no-op when phase missing)."""
        if action_name in COORDINATOR_INTERNAL_ACTIONS:
            raise PolicyDenied(
                f"action {action_name!r} is Coordinator-managed and not "
                f"LLM-proposable ({intent_kind})",
                rule="phase_incompatible",
                hint=(
                    "roofline / profile / replay_warm_recipe / framework_pr "
                    "are driven by the Coordinator (PRELUDE bootstrap, +10% "
                    "watermark refresh, warm-recipe replay, FRAMEWORK_PR "
                    "pump) and never appear in any phase's LLM-proposable "
                    "set. Propose ``specialist`` or ``explore`` instead."
                ),
            )
        state = self.shared_state
        if state is None:
            return
        phase = (getattr(state, "phase", "") or "").strip().upper()
        if not phase or phase not in PHASE_NAMES:
            return
        explore_enabled = bool(getattr(state, "explore_enabled", True))
        # --no-explore is a hard intent: EXPLORE work is disabled for the whole
        # run, so the interleave grey channel must not let KERNEL re-introduce
        # an ``explore`` grid. This denial is ALWAYS fail-closed (independent of
        # ``strict_phase``) because it reflects an explicit operator decision,
        # not the softer per-phase action contract.
        if (
            not explore_enabled
            and phase == PHASE_KERNEL
            and action_name == EXPLORE_ACTION_NAME
        ):
            raise PolicyDenied(
                f"action {EXPLORE_ACTION_NAME!r} is disabled for this run "
                f"(--no-explore); KERNEL may not borrow the interleave "
                f"channel to run an explore grid",
                rule="explore_disabled",
                hint=(
                    "--no-explore skips the EXPLORE phase entirely. The "
                    "phase-interleave grey channel cannot reintroduce "
                    "`explore` into KERNEL. Use kernel-owned actions "
                    "(kernel_opt / integrate / ...), or `specialist` / "
                    "`integrate_patch` if you need patch research/integration."
                ),
            )
        # Robustness-delegate-only actions (e.g. ``recover``) are absent from the LLM-proposable set but still delegatable by robustness; accept if phase-allowed.
        if (
            intent_kind == "delegate"
            and action_name in ROBUSTNESS_DELEGATE_ONLY_ACTIONS
            and is_action_allowed_in_phase(action_name, phase)
        ):
            return
        if is_action_llm_proposable_in_phase_with_interleave(
            action_name, phase, explore_enabled=explore_enabled,
        ):
            return
        allowed = tuple(sorted(
            llm_proposable_actions_for_with_interleave(
                phase, explore_enabled=explore_enabled,
            )
        ))
        hint = (
            f"you are in phase={phase}; action {action_name!r} is not in "
            f"the LLM-proposable set {list(allowed)!r}. Either propose an "
            f"action from that list, or wait for the Coordinator to "
            f"advance the phase. See KB_design §3.2 for the per-phase "
            f"action contract."
        )
        if not self.strict_phase:
            # Warn-only: keep the run flowing but record the denial in the audit trail.
            try:
                state.record_policy_denial(
                    action_name=action_name,
                    rule="phase_incompatible",
                    hint=hint,
                    intent_type=intent_kind,
                    tick=int(getattr(state, "tick", 0) or 0),
                    intent_payload={"phase": phase},
                )
            except Exception:  # noqa: BLE001 — best-effort audit
                pass
            return
        raise PolicyDenied(
            f"action {action_name!r} not allowed in phase={phase}",
            rule="phase_incompatible",
            hint=hint,
        )

    # FP8-only actions
    def _validate_fp8_only_action(
        self,
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject GEMM tuning for non-FP8 sessions (it drives FP8 block-scale GEMM dispatch; the handler repeats the check)."""
        if not action_name or action_name not in FP8_ONLY_ACTIONS:
            return
        state = self.shared_state
        if state is None:
            return
        precision = str(getattr(state, "precision", "") or "").strip().lower()
        if precision == "fp8":
            return
        raise PolicyDenied(
            f"action {action_name!r} is FP8-only but session precision={precision or '(unset)'!r}",
            rule="fp8_only_action",
            hint=(
                f"intent_kind={intent_kind!r}: GEAK GEMM tuning only applies "
                "to FP8 block-scale workloads. Set PRECISION=fp8 / "
                "--precision fp8, or skip gemm_tuning and continue with "
                "non-FP8 actions."
            ),
        )

    # R4 — kb_write_unauthorized
    def _validate_no_kb_write_collision(
        self,
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject any intent whose ``action_name`` / ``request.kind`` equals a Cortex KB write tool name (defense in depth; KB_design §3.11 §4.4 / Inv-11.3)."""
        if not action_name:
            return
        if action_name not in KB_WRITE_TOOL_NAMES:
            return
        raise PolicyDenied(
            f"intent={intent_kind!r} cannot invoke KB write surface "
            f"{action_name!r}",
            rule="kb_write_unauthorized",
            hint=(
                "Direct KB writes are not allowed. "
                "The Coordinator owns all KB writes. Express your "
                "intent via propose_action / delegate / "
                "specialist_done.proposal_set / review_verdict / "
                "kb_writes (critic-agent commit-review) instead."
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
        """Reject an external tool name not on the caller's role whitelist (KB/PR/Web tools are specialist-only; KB_design §3.11 §4.5)."""
        if not action_name:
            return
        # Skip KB write names (R4 owns them) so a write attempt yields ``kb_write_unauthorized``, not the less-specific R5 code.
        if action_name in KB_WRITE_TOOL_NAMES:
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
                f"specialist sub-agents. The "
                f"primary agents (orchestration / kernel / critic / "
                f"robustness) consult KB / PR Monitor via the "
                f"Coordinator-mediated KnowledgePlane facade instead."
            ),
        )

    # R4 / R5 public helper — pure validator for the SpecialistRunner tool-list builder.
    def validate_tool_invocation(
        self,
        tool_name: str,
        *,
        source_role: str,
        phase: str | None = None,
    ) -> None:
        """Raise :class:`PolicyDenied` if ``tool_name`` is not allowed for ``source_role`` (pure, Inv-11.1; ``phase`` no longer gates)."""
        tool_name = (tool_name or "").strip()
        if not tool_name:
            raise PolicyDenied(
                "validate_tool_invocation: tool_name is empty",
                rule="payload",
                hint="caller must pass the canonical tool name",
            )
        # R4 — KB writes are categorically off-limits.
        if tool_name in KB_WRITE_TOOL_NAMES:
            raise PolicyDenied(
                f"KB write tool {tool_name!r} cannot be invoked by "
                f"role={source_role!r}",
                rule="kb_write_unauthorized",
                hint=(
                    "Direct KB writes are not allowed (KB_design §3.11 "
                    "R4). The Coordinator owns all KB writes."
                ),
            )
        # R5 — role whitelist for the known external tools.
        if tool_name in ALL_KNOWN_EXTERNAL_TOOL_NAMES:
            allowed_for_role = TOOL_WHITELIST_BY_ROLE.get(
                source_role, frozenset(),
            )
            if tool_name not in allowed_for_role:
                raise PolicyDenied(
                    f"role={source_role!r} cannot invoke tool "
                    f"{tool_name!r}",
                    rule="tool_whitelist_role",
                    hint=(
                        f"{tool_name} is restricted to specialist "
                        f"sub-agents. KB_design §3.11 §4.5."
                    ),
                )
        # Anything else is implicitly allowed; internal tools are filtered by the SpecialistRunner's own denylist.

    # ``sweep_phase_singleton``
    def _validate_sweep_singleton(
        self, payload: dict[str, Any], *, intent_kind: str,
    ) -> None:
        """Enforce one sweep per SWEEP phase (Inv-9.4): deny agent sweeps once the auto-enqueued sweep landed (concurrent sweeps crash both vllm engines). Escape: ``params.bypass_sweep_singleton=True``."""
        params = payload.get("params") or {}
        if isinstance(params, dict) and params.get("bypass_sweep_singleton"):
            return
        ss = getattr(self, "shared_state", None)
        if ss is None:
            return
        history = getattr(ss, "phase_history", None) or []
        if not history:
            return
        latest = history[-1]
        if not isinstance(latest, dict):
            return
        if str(latest.get("to_phase") or "").strip() != PHASE_SWEEP:
            return
        evidence = latest.get("evidence")
        if not isinstance(evidence, dict):
            return
        auto_id = str(evidence.get("auto_sweep_task_id") or "").strip()
        if not auto_id:
            return
        raise PolicyDenied(
            f"sweep: SWEEP phase already has an auto-enqueued sweep "
            f"task (auto_sweep_task_id={auto_id!r}); concurrent "
            f"sweep proposals would race for the same GPUs and "
            f"port and crash both vllm engines on init.",
            rule="sweep_phase_singleton",
            hint=(
                "The Coordinator's SWEEP-entry hook already covers "
                "the SKILL.md default grid plus the Cortex "
                "recipe.sweep_grid field — no further sweep "
                "proposal is needed. Wait for the auto-sweep to "
                "finish (SWEEP→CLOSE transitions automatically). "
                "If you genuinely need a second grid for debug, "
                f"set params.bypass_sweep_singleton=True on the "
                f"{intent_kind} payload (the override is recorded "
                f"on the audit trail)."
            ),
        )

    # ``conc_sweep_phase_singleton`` (Bug #11)
    def _validate_conc_sweep_singleton(
        self, payload: dict[str, Any], *, intent_kind: str,
    ) -> None:
        """Enforce one conc_sweep per SWEEP phase (Bug #11); re-proposals burn GPU for no new data. Escape: ``params.bypass_conc_sweep_singleton=True``."""
        params = payload.get("params") or {}
        if isinstance(params, dict) and params.get("bypass_conc_sweep_singleton"):
            return
        ss = getattr(self, "shared_state", None)
        if ss is None:
            return
        history = getattr(ss, "phase_history", None) or []
        if not history:
            return
        latest = history[-1]
        if not isinstance(latest, dict):
            return
        if str(latest.get("to_phase") or "").strip() != PHASE_SWEEP:
            return
        evidence = latest.get("evidence")
        if not isinstance(evidence, dict):
            return
        auto_id = str(evidence.get("auto_conc_sweep_task_id") or "").strip()
        if not auto_id:
            return
        raise PolicyDenied(
            f"conc_sweep: SWEEP phase already has an auto-enqueued "
            f"conc_sweep task (auto_conc_sweep_task_id={auto_id!r}); "
            f"duplicate runs reproduce the same baseline + current_best "
            f"comparison and add no new data while burning 30-150 min "
            f"of GPU time.",
            rule="conc_sweep_phase_singleton",
            hint=(
                "Coordinator's post-sweep hook already dispatched "
                "conc_sweep — wait for SWEEP→CLOSE. If you need a "
                "second run for debug, set "
                f"params.bypass_conc_sweep_singleton=True on the "
                f"{intent_kind} payload (recorded on the audit trail)."
            ),
        )

    def _validate_integrate_patch_critic_gate(
        self, payload: dict[str, Any],
    ) -> None:
        """PR-A7: enforce ``integrate_patch_requires_critic_verdict`` (needs specialist_task_id + permissive verdict, unless ``params.bypass_critic=True``)."""
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise PolicyDenied(
                "integrate_patch: params must be a dict",
                rule="integrate_patch_requires_critic_verdict",
                hint=(
                    "pass params={specialist_task_id: <id>, ...}; "
                    "see actions/integrate_patch.md"
                ),
            )
        sid = str(params.get("specialist_task_id") or "").strip()
        if not sid:
            raise PolicyDenied(
                "integrate_patch.params.specialist_task_id is required",
                rule="integrate_patch_requires_critic_verdict",
                hint=(
                    "set params.specialist_task_id to the task_id of "
                    "the completed specialist whose worktree carries "
                    "the patches you want to apply."
                ),
            )
        bypass = bool(params.get("bypass_critic"))
        if bypass:
            return
        ss = getattr(self, "shared_state", None)
        verdict = ""
        if ss is not None:
            try:
                verdict = ss.get_specialist_patch_verdict(sid)
            except AttributeError:
                # Older SharedState without the field → no verdict on record.
                verdict = ""
        if not verdict:
            raise PolicyDenied(
                f"integrate_patch: no Critic verdict on record for "
                f"specialist_task_id={sid!r}",
                rule="integrate_patch_requires_critic_verdict",
                hint=(
                    "Wait for the Critic to emit a "
                    "review_verdict{target_proposal_msg_id=<patch "
                    "proposal>, verdict=approve|reject|...} for this "
                    "specialist, or override with "
                    "params.bypass_critic=True. The Critic verdict "
                    "is recorded on SharedState.specialist_patch_verdicts."
                ),
            )
        if verdict.lower() not in INTEGRATE_PATCH_PERMISSIVE_VERDICTS:
            raise PolicyDenied(
                f"integrate_patch: Critic verdict for specialist "
                f"task {sid!r} is {verdict!r}; integrate_patch only "
                f"runs on "
                f"{sorted(INTEGRATE_PATCH_PERMISSIVE_VERDICTS)!r}",
                rule="integrate_patch_requires_critic_verdict",
                hint=(
                    "Either ask the Critic to re-review (next "
                    "review_verdict overwrites this one), drop the "
                    "patch (specialist_done.patches_written=[]), or "
                    "set params.bypass_critic=True to force "
                    "integration with an explicit operator audit "
                    "trail."
                ),
            )

    def _validate_specialist_dispatch(
        self, role: "AgentRole", payload: dict[str, Any],
    ) -> None:
        """Enforce the specialist-delegate contract (Inv-11.2): orchestration-only, tags ∈ vocab, gap_canonical_id required, max_turns ≤ cap."""
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

        # scope='freeform' (absorbed dynamic_specialist) has no domain anchor:
        # it skips the tag / gap vocabulary checks and runs a lightweight
        # mechanical sanity gate instead.
        scope_raw = str(params.get("scope") or "").strip().lower()
        if scope_raw == SPECIALIST_SCOPE_FREEFORM:
            self._validate_freeform_specialist_dispatch(params)
            return

        # ``params.tags`` is canonical; a single ``params.domain`` is a backward-compatible alias.
        tags = normalize_dispatch_tags(params)
        # A *bare* dispatch — no explicit scope and no domain/tag anchor —
        # defaults to the cheap, read-only freeform lane (point 3: safe & cheap
        # first) instead of being rejected. The freeform gate below still
        # requires a non-empty task_description, so a fully empty dispatch is
        # rejected there rather than running an anchorless patch specialist.
        if not scope_raw and not tags:
            self._validate_freeform_specialist_dispatch(params)
            return

        if not tags:
            raise PolicyDenied(
                "delegate{action='specialist'}: at least one tag is "
                "required (params.tags or the legacy params.domain alias)",
                rule="specialist_unknown_domain",
                hint=(
                    f"set params.tags to a non-empty subset of "
                    f"{sorted(KNOWLEDGE_DOMAIN_TAG_SET)!r}"
                ),
            )
        # Each tag must belong to the controlled knowledge-domain vocabulary.
        unknown_tags = [t for t in tags if t not in KNOWLEDGE_DOMAIN_TAG_SET]
        if unknown_tags:
            raise PolicyDenied(
                f"delegate{{action='specialist'}}: unknown knowledge-domain "
                f"tag(s)={unknown_tags!r}",
                rule="specialist_unknown_domain",
                hint=(
                    f"every tag must be one of "
                    f"{sorted(KNOWLEDGE_DOMAIN_TAG_SET)!r}"
                ),
            )

        # ``scope`` dial (domain | domains | freeform). Absent => legacy
        # single-domain default. ``domains`` is the cross-domain channel that
        # absorbed the retired dynamic_action worker and requires >1 distinct
        # tag; ``domain`` is single-tag. (``freeform`` has its own gate.)
        scope = str(params.get("scope") or "").strip().lower()
        if scope and scope not in SPECIALIST_SCOPE_VALUES:
            raise PolicyDenied(
                f"delegate{{action='specialist'}}: unknown scope={scope!r}",
                rule="specialist_scope_invalid",
                hint=(
                    f"scope must be one of {sorted(SPECIALIST_SCOPE_VALUES)!r}"
                ),
            )
        if scope == SPECIALIST_SCOPE_DOMAINS and len(tags) < 2:
            raise PolicyDenied(
                "delegate{action='specialist'}: scope='domains' is the "
                "cross-domain channel and requires at least 2 distinct "
                f"tags; got {tags!r}",
                rule="specialist_scope_too_narrow",
                hint=(
                    "Declare every domain the patch must touch together in "
                    "params.tags, or use scope='domain' for a single-domain "
                    "specialist."
                ),
            )
        if scope == SPECIALIST_SCOPE_DOMAIN and len(tags) > 1:
            raise PolicyDenied(
                "delegate{action='specialist'}: scope='domain' is "
                f"single-domain but got {len(tags)} tags {tags!r}",
                rule="specialist_scope_mismatch",
                hint=(
                    "Use scope='domains' for a cross-domain specialist, or "
                    "pass a single tag."
                ),
            )

        # ``sub_kind`` is a free-form prompt selector (not constrained to a catalogue).
        gap = str(params.get("gap_canonical_id") or params.get("gap") or "").strip()
        if not gap:
            # Friction symmetry (point 4): a domain/tag-anchored dispatch that
            # omits the gap id is backfilled from the gaps[] ledger by matching
            # the dispatch anchor against each gap's ``domain_hint`` — so the
            # LLM doesn't have to hand-copy a canonical id (and isn't pushed
            # toward freeform out of friction). Only mutates when a match is
            # found; otherwise the rejection below still fires.
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
        if max_turns_raw is not None:
            try:
                max_turns = int(max_turns_raw)
            except (TypeError, ValueError) as exc:
                raise PolicyDenied(
                    f"delegate{{action='specialist'}}: max_turns must be "
                    f"int, got {max_turns_raw!r}",
                    rule="specialist_dispatch_source",
                ) from exc
            if max_turns <= 0 or max_turns > SPECIALIST_MAX_TURNS_HARD_CAP:
                raise PolicyDenied(
                    f"delegate{{action='specialist'}}: max_turns={max_turns} "
                    f"outside (0, {SPECIALIST_MAX_TURNS_HARD_CAP}]",
                    rule="specialist_dispatch_source",
                    hint=(
                        f"max_turns must be in (0, {SPECIALIST_MAX_TURNS_HARD_CAP}]; "
                        f"the prompt default is 8."
                    ),
                )

        self._validate_specialist_gpu_request(params)

    def _validate_specialist_gpu_request(self, params: dict[str, Any]) -> None:
        """Validate a specialist's optional GPU request against the GPU
        specialist-pool ceiling.

        Shared by the domain-anchored gate (``_validate_specialist_dispatch``)
        and the freeform gate (``_validate_freeform_specialist_dispatch``) so a
        ``scope='freeform'`` dispatch that sets ``needs_gpu`` is governed by the
        same ceiling instead of slipping past it via the freeform early-return.
        No-op when the dispatch needs no GPU.

        A bench-enabled specialist (``mode=patch`` & ``bench=true``) does not
        have to set ``needs_gpu`` explicitly: the Coordinator's
        ``_warm_specialist_params`` auto-defaults it to True at dispatch so the
        worktree micro-benchmark holds a GPU lease. We mirror that auto-default
        here, otherwise such a dispatch slips past this gate (no explicit
        ``needs_gpu``) and later becomes a GPU-needing queued task that can stall
        forever when the pool is disabled (``--gpu-specialist-capacity 0``)
        rather than being rejected at the policy layer.
        """
        needs_gpu_raw = params.get("needs_gpu", False)
        if isinstance(needs_gpu_raw, str):
            needs_gpu = needs_gpu_raw.strip().lower() in (
                "1", "true", "yes", "y", "on",
            )
        else:
            needs_gpu = bool(needs_gpu_raw)
        if not needs_gpu:
            from .specialist_profile import resolve_specialist_profile
            if resolve_specialist_profile(params).grants_bench_tool:
                needs_gpu = True
        if not needs_gpu:
            return
        gpu_count_raw = params.get("gpu_count", 1)
        try:
            gpu_count = int(gpu_count_raw)
        except (TypeError, ValueError) as exc:
            raise PolicyDenied(
                "delegate{action='specialist'}: gpu_count must be "
                f"an integer, got {gpu_count_raw!r}",
                rule="specialist_gpu_request_invalid",
            ) from exc
        if gpu_count <= 0:
            raise PolicyDenied(
                "delegate{action='specialist'}: gpu_count must be > 0 "
                "when needs_gpu=true",
                rule="specialist_gpu_request_invalid",
            )
        ceiling = gpu_specialist_ceiling(self.shared_state)
        if ceiling <= 0:
            raise PolicyDenied(
                "delegate{action='specialist'}: needs_gpu=true but the "
                "GPU specialist pool is disabled",
                rule="specialist_gpu_pool_disabled",
                hint=(
                    "Start the session with --gpu-specialist-capacity > 0 "
                    "or set INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY "
                    "before dispatching GPU specialists."
                ),
            )
        if gpu_count > ceiling:
            raise PolicyDenied(
                f"delegate{{action='specialist'}}: gpu_count={gpu_count} "
                f"exceeds GPU specialist capacity={ceiling}",
                rule="specialist_gpu_request_exceeds_capacity",
                hint=(
                    "Lower params.gpu_count or start a session with a "
                    "larger GPU specialist pool."
                ),
            )

    def _autofill_gap_from_ledger(
        self, params: dict[str, Any], tags: list[str],
    ) -> str:
        """Backfill ``params.gap_canonical_id`` from the gaps[] ledger.

        Matches the dispatch anchor (domain key, its kb_anchor, and the
        knowledge-domain ``tags``) against each gap's ``domain_hint``. Among the
        matches, prefers the most actionable: highest severity, then the
        least-attempted, then the oldest (most-stalled) gap. Mutates ``params``
        in place and returns the chosen canonical id (``""`` when nothing
        matches, leaving the caller's required-gap rejection intact).
        """
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
            sev = severity_rank.get(str(g.get("severity") or "").lower(), 0)
            attempts = len(g.get("attempts") or [])
            first_seen = str(g.get("first_seen_ts") or "")
            # Highest severity first, then fewest attempts, then oldest.
            return (-sev, attempts, first_seen)

        matches = [
            g for g in gaps
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
        self, params: dict[str, Any],
    ) -> None:
        """Lightweight mechanical sanity gate for ``scope='freeform'``
        specialists (absorbed from the retired dynamic_specialist wave
        channel). Free-form dispatches carry no domain/tag/gap anchor, so this
        validates only structural shape: a single ``task_description`` or a
        ``tasks=[...]`` wave (bounded by SPECIALIST_FREEFORM_WAVE_MAX), each
        with a non-empty, length-bounded description that survives the
        red-line tripwire."""
        # Freeform deliberately skips the domain-anchored max_turns gate: a
        # free-form investigation has no domain/gap to bound its depth, so it is
        # constrained by the task TIMEOUT (lease TTL / wall-clock) rather than a
        # turn cap. This is by design, NOT an oversight — do not re-add a
        # max_turns bound here (see Issue 5b review). A GPU request must still
        # clear the same pool ceiling as a domain specialist, otherwise
        # scope='freeform' would be a hole around the GPU accounting.
        self._validate_specialist_gpu_request(params)
        wave = params.get("tasks")
        if wave is not None:
            if not isinstance(wave, list) or not wave:
                raise PolicyDenied(
                    "delegate{action='specialist',scope='freeform'}: "
                    "params.tasks must be a non-empty list",
                    rule="specialist_freeform_wave_invalid",
                    hint=(
                        "Pass tasks=[{task_description: ...}, ...] or a single "
                        "params.task_description for a one-off freeform "
                        "specialist."
                    ),
                )
            if len(wave) > SPECIALIST_FREEFORM_WAVE_MAX:
                raise PolicyDenied(
                    f"delegate{{action='specialist',scope='freeform'}}: wave "
                    f"size={len(wave)} exceeds cap "
                    f"{SPECIALIST_FREEFORM_WAVE_MAX}",
                    rule="specialist_freeform_wave_too_large",
                    hint=(
                        f"Split the wave into batches of at most "
                        f"{SPECIALIST_FREEFORM_WAVE_MAX} tasks."
                    ),
                )
            for i, task in enumerate(wave):
                if not isinstance(task, dict):
                    raise PolicyDenied(
                        f"delegate{{action='specialist',scope='freeform'}}: "
                        f"tasks[{i}] must be an object",
                        rule="specialist_freeform_task_invalid",
                    )
                desc = str(
                    task.get("task_description")
                    or task.get("task_summary")
                    or ""
                ).strip()
                self._check_freeform_task_description(desc, where=f"tasks[{i}]")
            return
        desc = str(params.get("task_description") or "").strip()
        self._check_freeform_task_description(desc, where="params")

    @staticmethod
    def _check_freeform_task_description(desc: str, *, where: str) -> None:
        """Per-task structural checks for a free-form ``task_description``:
        non-empty, length-bounded, and clear of the red-line tripwire."""
        if not desc:
            raise PolicyDenied(
                f"delegate{{action='specialist',scope='freeform'}}: "
                f"{where} task_description must be non-empty",
                rule="specialist_freeform_empty_description",
                hint=(
                    "Each freeform task needs a natural-language "
                    "task_description (the whole mandate)."
                ),
            )
        if len(desc) > SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS:
            raise PolicyDenied(
                f"delegate{{action='specialist',scope='freeform'}}: "
                f"{where} task_description is {len(desc)} chars > cap "
                f"{SPECIALIST_FREEFORM_TASK_DESC_MAX_CHARS}",
                rule="specialist_freeform_description_too_long",
            )
        for pat in _FREEFORM_REDLINE_PATTERNS:
            if pat.search(desc):
                raise PolicyDenied(
                    f"delegate{{action='specialist',scope='freeform'}}: "
                    f"{where} task_description tripped the red-line scan "
                    f"(pattern={pat.pattern!r})",
                    rule="specialist_freeform_redline",
                    hint=(
                        "Free-form mandates must not embed destructive host "
                        "commands. Describe the investigation, not raw "
                        "destructive shell."
                    ),
                )

    # R3 ``specialist_done_source``
    def _validate_specialist_intent(
        self, from_agent: str, intent: Intent,
    ) -> None:
        """Validate any intent emitted under a ``specialist:<task_id>`` identity (Inv-5.2: only SEND_MESSAGE, ALERT, and one SPECIALIST_DONE)."""
        task_id = from_agent.removeprefix(SPECIALIST_FROM_AGENT_PREFIX).strip()
        if not task_id:
            raise PolicyDenied(
                "specialist from_agent missing task_id suffix "
                f"(got {from_agent!r})",
                rule="specialist_done_source",
                hint=(
                    "Specialist sub-agents must stamp "
                    "from_agent='specialist:<task_id>' where <task_id> "
                    "matches the dispatched task."
                ),
            )
        if intent.type == IntentType.SPECIALIST_DONE:
            self._validate_specialist_done_payload(task_id, intent.payload or {})
            return
        # Allowed ancillary intents (heartbeat / advice / alert).
        if intent.type in (
            IntentType.SEND_MESSAGE,
            IntentType.ALERT,
        ):
            return
        raise PolicyDenied(
            f"specialist={from_agent!r} cannot emit "
            f"intent_type={intent.type.value!r}",
            rule="specialist_done_source",
            hint=(
                "Specialists may only emit specialist_done (exit), "
                "send_message (heartbeat/advice), or alert. Use "
                "specialist_done with proposal_set + summary instead."
            ),
        )

    def _validate_specialist_done_payload(
        self, task_id: str, payload: dict[str, Any],
    ) -> None:
        """Per-field R3 structural checks for the ``specialist_done`` payload (gap_canonical_id, domain ∈ keys, proposal_set, empty+reason, summary ≤4096, confidence ∈ [0,1])."""
        gap = str(payload.get("gap_canonical_id") or "").strip()
        if not gap:
            raise PolicyDenied(
                "specialist_done missing gap_canonical_id",
                rule="specialist_done_source",
                hint=(
                    "Payload must echo the gap_canonical_id that was "
                    "passed to delegate{action='specialist'} so "
                    "Coordinator can cross-check the dispatch."
                ),
            )
        domain = str(payload.get("domain") or "").strip()
        if not domain:
            raise PolicyDenied(
                "specialist_done missing domain",
                rule="specialist_done_source",
            )
        if domain not in SPECIALIST_DOMAIN_KEYS:
            raise PolicyDenied(
                f"specialist_done: unknown domain={domain!r}",
                rule="specialist_done_source",
                hint=(
                    f"domain must be one of {sorted(SPECIALIST_DOMAIN_KEYS)!r}"
                ),
            )
        proposal_set = payload.get("proposal_set")
        if not isinstance(proposal_set, list):
            raise PolicyDenied(
                "specialist_done.proposal_set must be a list",
                rule="specialist_done_source",
                hint="set proposal_set=[] when empty=true",
            )
        empty_flag = bool(payload.get("empty"))
        if empty_flag:
            if proposal_set:
                raise PolicyDenied(
                    "specialist_done: empty=true implies proposal_set=[]",
                    rule="specialist_done_source",
                )
            reason_field = str(
                payload.get("reason") or payload.get("summary") or ""
            ).strip()
            if not reason_field:
                raise PolicyDenied(
                    "specialist_done: empty=true requires a reason / summary "
                    "describing why no proposals were emitted",
                    rule="specialist_done_source",
                )
        else:
            for i, variant in enumerate(proposal_set):
                if not isinstance(variant, dict):
                    raise PolicyDenied(
                        f"specialist_done.proposal_set[{i}] must be a dict",
                        rule="specialist_done_source",
                    )
                if not str(variant.get("name") or "").strip():
                    raise PolicyDenied(
                        f"specialist_done.proposal_set[{i}].name required",
                        rule="specialist_done_source",
                        hint=(
                            "Every variant needs a unique name "
                            "(round-scoped). See §3.4 §5.1 for the full "
                            "variant schema."
                        ),
                    )
        summary = str(payload.get("summary") or "")
        if len(summary) > 4096:
            raise PolicyDenied(
                "specialist_done.summary too long "
                f"({len(summary)} > 4096 chars)",
                rule="specialist_done_source",
                hint="KB_design §3.5 §7 caps summary at ~500 chars; "
                     "4096 is the defensive hard limit.",
            )
        confidence_raw = payload.get("confidence")
        if confidence_raw is not None:
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError) as exc:
                raise PolicyDenied(
                    f"specialist_done.confidence must be float, "
                    f"got {confidence_raw!r}",
                    rule="specialist_done_source",
                ) from exc
            if not 0.0 <= confidence <= 1.0:
                raise PolicyDenied(
                    f"specialist_done.confidence={confidence} not in [0, 1]",
                    rule="specialist_done_source",
                )

    def _validate_kill_task(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        """Validate a ``KILL_TASK`` intent (robustness-only).

        Requires the source role to be on
        :data:`KILL_TASK_SOURCE_ALLOWLIST`, a non-empty ``task_id`` and
        ``reason``, and a ``scope`` within :data:`KILL_TASK_ALLOWED_SCOPES`
        (``task`` only — server/process kills stay out per IR-5).

        Args:
            role (AgentRole): the resolved role of the emitting agent.
            payload (dict[str, Any]): the kill_task payload carrying
                ``task_id``, ``reason`` and optional ``scope``.

        Returns:
            None: returns silently when the kill request is permitted.

        Raises:
            PolicyDenied: when the role is not allowed
                (``kill_task_source``), ``task_id`` / ``reason`` is missing
                (``payload``), or the scope is disallowed (``kill_scope``).
        """
        if role.name not in KILL_TASK_SOURCE_ALLOWLIST:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit kill_task "
                f"(allowed: {sorted(KILL_TASK_SOURCE_ALLOWLIST)!r})",
                rule="kill_task_source",
            )
        task_id = str(payload.get("task_id", "")).strip()
        if not task_id:
            raise PolicyDenied("kill_task missing task_id", rule="payload")
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise PolicyDenied("kill_task missing reason", rule="payload")
        scope = str(payload.get("scope") or "task").strip()
        if scope not in KILL_TASK_ALLOWED_SCOPES:
            raise PolicyDenied(
                f"kill_task scope={scope!r} not allowed "
                f"(allowed: {sorted(KILL_TASK_ALLOWED_SCOPES)!r}; "
                f"v0.6 keeps server/process kills out per IR-5)",
                rule="kill_scope",
            )

    def _path_under_session(self, value: str) -> bool:
        """Return whether a path resolves inside the active session_dir.

        Args:
            value (str): the path string to test.

        Returns:
            bool: True when :attr:`session_dir` is unset (check disabled),
                or when ``value`` resolves to or under the session
                directory; False if it escapes or cannot be resolved.
        """
        if self.session_dir is None:
            return True
        try:
            sd = self.session_dir.resolve()
            v = Path(str(value)).resolve()
        except (OSError, RuntimeError):
            return False
        try:
            return v == sd or v.is_relative_to(sd)
        except AttributeError:  # pragma: no cover — Python <3.9
            try:
                v.relative_to(sd)
                return True
            except ValueError:
                return False

    def _path_in_source_allowlist(self, value: str) -> bool:
        """Return whether a path falls under a framework source allowlist.

        Args:
            value (str): the path string to test.

        Returns:
            bool: True when ``value`` starts with any prefix returned by
                :func:`resolve_source_file_allowlist` (the aiter / sglang /
                vllm source trees); False otherwise.
        """
        s = str(value)
        return any(s.startswith(p) for p in resolve_source_file_allowlist())

    def _path_in_trace_allowlist(self, value: str) -> bool:
        """Match a value against runtime-resolved trace path prefixes (multi-node shared profile dir outside session_dir)."""
        s = str(value)
        return any(s.startswith(p) for p in _trace_path_allowlist())

    def _validate_payload_paths(
        self, role: "AgentRole", intent_type: IntentType, payload: dict[str, Any],
    ) -> None:
        """Walk payload (recursively); reject path-like values escaping session_dir. No-op when session_dir is None or strict_paths is False."""
        if self.session_dir is None or not self.strict_paths:
            return

        def visit(node: Any, path_keys: tuple[str, ...]) -> None:
            """Recursively scan a payload node for escaping path values.

            Args:
                node (Any): the current payload node (dict, list/tuple,
                    string, or scalar) being walked.
                path_keys (tuple[str, ...]): the chain of dict keys leading
                    to ``node``; its last element is the field name used to
                    decide which allowlist applies.

            Returns:
                None.

            Raises:
                PolicyDenied: when a path-like string escapes the session
                    directory and its applicable allowlists.
            """
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
                if self._path_in_source_allowlist(node) or self._path_under_session(node):
                    return
                raise PolicyDenied(
                    f"role={role.name!r} {intent_type.value} payload field "
                    f"{key!r}={node!r} is not under session_dir or any of "
                    f"{list(resolve_source_file_allowlist())!r}",
                    rule="source_file_not_allowlisted",
                    hint=("kernel-opt may only target framework source trees "
                          "under aiter/sglang/vllm; reject the request"),
                )
            if key not in PATH_LIKE_FIELDS:
                return
            if not self._path_under_session(node):
                # Multi-node profile traces live outside session_dir; allow only the trace-input fields against the trace allowlist.
                if (
                    key in TRACE_PATH_LIKE_FIELDS
                    and self._path_in_trace_allowlist(node)
                ):
                    return
                raise PolicyDenied(
                    f"role={role.name!r} {intent_type.value} payload field "
                    f"{key!r}={node!r} escapes session_dir={self.session_dir!s}",
                    rule="path_outside_session_dir",
                    hint=("emit paths verbatim from SharedState (e.g. "
                          "last_profile_trace) or under SESSION_DIR; "
                          "multi-node trace fields may also resolve under "
                          f"{list(_trace_path_allowlist())!r}"),
                )

        visit(payload, ())

    def _validate_robustness_only(
        self, role: "AgentRole", intent_type: IntentType, payload: dict[str, Any]
    ) -> None:
        """Enforce that only allowed roles emit robustness-only intents.

        Args:
            role: The agent role attempting to emit the intent.
            intent_type: The intent being validated.
            payload: The intent payload (checked for required fields).

        Raises:
            PolicyDenied: If the role is not permitted to emit the intent,
                or a required payload field (e.g. ``family`` for
                ``PRUNE_BRANCH``) is missing.
        """
        # Per-intent source override takes precedence; default is robustness-only.
        allowed_sources = _ROBUSTNESS_ONLY_INTENT_SOURCES.get(
            intent_type, ROBUSTNESS_ONLY_SOURCE_ALLOWLIST,
        )
        if role.name not in allowed_sources:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit {intent_type.value} "
                f"(allowed: {sorted(allowed_sources)!r})",
                rule="robustness_only_source",
            )
        if intent_type == IntentType.PRUNE_BRANCH:
            family = str(payload.get("family", "")).strip()
            if not family:
                raise PolicyDenied("prune_branch missing family", rule="payload")


__all__ = [
    "CORE_STATE_FIELDS",
    "DELEGATE_ACTION_REQUIRED_PAYLOAD",
    "DELEGATE_ACTION_SOURCE_ALLOWLIST",
    "INTERNAL_ONLY_ACTION_NAMES",
    "KERNEL_OWNED_ACTIONS",
    "KILL_TASK_ALLOWED_SCOPES",
    "KILL_TASK_SOURCE_ALLOWLIST",
    "PATH_LIKE_FIELDS",
    "PolicyDenied",
    "PolicyGate",
    "REQUEST_ROUTING",
    "REVIEW_VERDICTS",
    "REVIEW_VERDICT_SOURCE_ALLOWLIST",
    "ROBUSTNESS_ONLY_INTENTS",
    "ROBUSTNESS_ONLY_SOURCE_ALLOWLIST",
    "TRACE_PATH_LIKE_FIELDS",
    "SOURCE_LIKE_FIELDS",
]
