"""PolicyGate

Single chokepoint: every parsed Intent passes through ``validate_intent``
before the Coordinator commits side-effects. PolicyGate converges:

    * Role permission   — does this agent's role allow this intent type?
    * Source allowlist  — REVIEW_VERDICT is critic-only;
                          KILL_TASK / FORCE_DISPATCH / PRUNE_BRANCH /
                          ESCALATE_STRATEGY_CHANGE are robustness-only
    * REQUEST routing   — only orchestration→kernel is allowed in the legacy release
    * Kernel ownership  — 5 kernel-owned actions can NOT be `delegate`d;
                          orchestration must REQUEST(target_agent="kernel")
    * Core state guard  — only the Coordinator can mutate
                          CORE_STATE_FIELDS (current_best, etc.)

PolicyGate stays *pure* — it does not touch the bus or the DB. The
Coordinator catches :class:`PolicyDenied` and emits a ``policy_denied``
observation event so the LLM can self-correct on its next replay turn.

v0.6 changes vs v0.5:

* Removed mode / FeatureFlags coupling — single full mode (ADR-34)
* Removed quick-mode bash allow/deny lists
* Renamed TRIAGE_ONLY → ROBUSTNESS_ONLY (matches new agent name)
* Added REVIEW_VERDICT validation (Critic-only, §18.2)
* KERNEL_OWNED_ACTIONS expanded to all 5 (DESIGN §7.2 / §16.1)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .framework_paths import resolve_source_file_allowlist
from .intent_parser import Intent, IntentType
from .phase_state import (
    PHASE_EXPLORE,
    PHASE_NAMES,
    PHASE_SWEEP,
    allowed_actions_for,
    is_action_allowed_in_phase,
)
from .specialist_domains import (
    SPECIALIST_DOMAIN_KEYS,
    SPECIALIST_MAX_TURNS_HARD_CAP,
)

if TYPE_CHECKING:  # pragma: no cover — type-only
    from .agent_role import AgentRole


# ---------------------------------------------------------------------------
def _value_is_present(value: Any) -> bool:
    """Treat a value as present iff it is a non-empty string OR a
    non-empty container (dict/list/tuple/set). ``None`` and whitespace-
    only strings count as absent. Used by the delegate required-payload
    check, where ``reason`` is a short string and ``evidence`` is a dict
    (per-GPU snapshot, consecutive_hits, ...)."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list, tuple, set)):
        return len(value) > 0
    return True


def _delegate_field_present(payload: dict[str, Any], field_name: str) -> bool:
    """Return True iff ``field_name`` is present (per :func:`_value_is_present`)
    at the top of ``payload`` OR nested under ``payload["params"]``.

    Robustness builds delegate envelopes with the action knobs (``reason``,
    ``evidence``, ``force_gpu_cleanup``) under ``payload["params"]`` so the
    downstream executor reads them via ``ctx.task.params``. We accept either
    location so PolicyGate is the chokepoint regardless of the producer's
    payload-shape choice — see ``robustness_agent/role/envelope.py``
    ``build_delegate`` and ``recover_executor.__call__``.
    """
    if _value_is_present(payload.get(field_name)):
        return True
    nested = payload.get("params")
    if isinstance(nested, dict) and _value_is_present(nested.get(field_name)):
        return True
    return False


class PolicyDenied(RuntimeError):
    """Intent rejected by PolicyGate.

    Attributes:
        rule: short identifier of the rule that fired (``role``, ``payload``,
              ``kernel_owned_by_kernel_agent``, ``request_target``,
              ``kill_task_source``, ``review_verdict_source``,
              ``robustness_only_source``, ``state_field``, ...)
        hint: optional one-line agent-actionable suggestion describing the
              canonical fix.
    """

    def __init__(self, reason: str, *, rule: str | None = None,
                 hint: str | None = None):
        super().__init__(reason)
        self.rule = rule
        self.hint = hint


# ---------------------------------------------------------------------------
# Plan A — kernel-owned actions (DESIGN §7.2 / §16.1)
#
# These 5 actions are owned end-to-end by the Kernel agent. Orchestration
# (and any other role) MUST NOT delegate them directly; the only valid path
# is REQUEST(target_agent="kernel", kind="...") + RESPONSE.
# ---------------------------------------------------------------------------
KERNEL_OWNED_ACTIONS: frozenset[str] = frozenset({
    "kernel_opt",
    "integrate",
    "deep_kernel_analysis",
    "operator_tuning",
    "vendor_kernel_config",
    "gemm_tuning",
})

FP8_ONLY_ACTIONS: frozenset[str] = frozenset({
    "gemm_tuning",
    "run_gemm_tuning",
})


# ---------------------------------------------------------------------------
# Per-action delegate source allowlist (action_name → set of source roles).
#
# This is the action-name analogue of ROBUSTNESS_ONLY_SOURCE_ALLOWLIST
# (which gates IntentType, not action_name). Some actions have side
# effects narrow enough that even roles with ``can_delegate_side_effects``
# must NOT initiate them — e.g. ``recover`` walks SIGTERM/SIGKILL against
# matching processes and is env-gated to optionally invoke
# ``rocm-smi --gpureset``. Letting Orchestration drive it bypasses the
# robustness escalation path (symptom → ActionLadder → delegate), so we
# limit the source to the robustness agent only.
#
# Actions not listed here fall through to the general delegate rules
# (kernel-owned guard + ActionRegistry lookup).
# ---------------------------------------------------------------------------
DELEGATE_ACTION_SOURCE_ALLOWLIST: dict[str, frozenset[str]] = {
    "recover": frozenset({"robustness"}),
}


# ---------------------------------------------------------------------------
# Per-action delegate required payload fields. The values are stringified
# and stripped; empty / missing fields raise PolicyDenied. This is the
# minimum evidence we require alongside a side-effecting delegate so the
# downstream executor + result.json audit have something to anchor on.
# ---------------------------------------------------------------------------
DELEGATE_ACTION_REQUIRED_PAYLOAD: dict[str, tuple[str, ...]] = {
    "recover": ("reason", "evidence"),
}


# ---------------------------------------------------------------------------
# specialist sub-agent action.
#
# ``specialist`` is a synthetic action_name that the Orchestration role
# uses to delegate work to an LLM specialist (research_lane, capacity-1
# in M5). It does NOT have a yaml meta under ``actions/_meta/`` —
# domain-specific behaviour is parameterised via ``params.domain``
# instead (see ``specialist_domains.SPECIALIST_DOMAINS``).
#
# PolicyGate accepts ``delegate{action_name='specialist'}`` from
# Orchestration only (R2 ``specialist_dispatch_source``) regardless of
# whether the action is in ActionRegistry; this constant tells the
# generic ``_validate_delegate`` unknown_action gate to skip the
# registry lookup. R2's sub-rules then enforce the param contract.
# ---------------------------------------------------------------------------
SPECIALIST_ACTION_NAME: str = "specialist"

# PR-A7 (Arbor-into-Hyperloom): the orchestrator-side patch
# integration step. Lives in the EXPLORE phase, gated by a Critic
# verdict before bench (Inv-5.1 / Inv-3 single-tenant GPU preserved).
INTEGRATE_PATCH_ACTION_NAME: str = "integrate_patch"

# PR-A9 (Arbor-into-Hyperloom): the merged explore action — kept
# as a named constant alongside SPECIALIST_ACTION_NAME so the
# explore-provenance gate has a single source of truth.
EXPLORE_ACTION_NAME: str = "explore"

# the sweep action; named constant so the
# ``sweep_phase_singleton`` rule (deny LLM-emitted sweep when
# Coordinator's auto-enqueue already landed one in SWEEP phase) has
# a single source of truth.  See _validate_sweep_singleton.
SWEEP_ACTION_NAME: str = "sweep"

# Provenance values that pass the explore-provenance gate. Anything
# else (``llm_direct``, missing, unknown) is denied. ``dynamic`` is a
# strict single-literal stamp; composite forms are rejected.
EXPLORE_PERMISSIVE_PROVENANCE_PREFIXES: tuple[str, ...] = (
    "specialist:",
)
EXPLORE_PERMISSIVE_PROVENANCE_LITERALS: frozenset[str] = frozenset({
    "default_grid",
    "dynamic",
})

# Specialist / Explore parallelism caps — single source of truth
# imported by cli.py, specialist_runner.py, specialist_prompt_builder.py,
# and prompt_builder.py so the limits never drift between layers.
#   * ``MAX_RESEARCH_LANE_CAPACITY`` — hard cap on concurrent specialist
#     sub-agents (the M6 ceiling; CLI clamps higher operator values down
#     to this without warning).
#   * ``DEFAULT_SPECIALIST_MAX_PROPOSALS`` — per-specialist proposal_set
#     cap; enforced both in the specialist prompt (self-curation) and on
#     the SpecialistRunner write path (hard truncate before persist).
#   * ``MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS`` — number of
#     ``provenance='specialist:*'`` variants Orchestration may stack in
#     one ``explore`` grid. ``default_grid`` variants are unaffected
#     (cold-start path).
MAX_RESEARCH_LANE_CAPACITY: int = 6

# Canonical name of the LLM-sub-agent resource lane shared by
# specialists + dynamic_action; kept in lockstep with
# :data:`resource_lock.LANE_PRIORITY`.
RESEARCH_LANE_NAME: str = "research_lane"
DEFAULT_SPECIALIST_MAX_PROPOSALS: int = 3
MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS: int = 1

# Verdicts that allow ``integrate_patch`` to proceed without an
# explicit operator override. ``advise`` is treated as a soft
# approval (Critic provided guidance but didn't block); ``approve``
# is the canonical green light.
INTEGRATE_PATCH_PERMISSIVE_VERDICTS: frozenset[str] = frozenset({
    "approve", "advise",
})

# Source roles allowed to dispatch a specialist via
# ``delegate{action='specialist'}``. KB_design §3.5 §11 / §3.11 §4.2.
SPECIALIST_DISPATCH_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"orchestration"})

# Prefix the SubAgentRunner stamps on every emit-intent originating
# from a specialist task. ``from_agent='specialist:<task_id>'``.
SPECIALIST_FROM_AGENT_PREFIX: str = "specialist:"


# ---------------------------------------------------------------------------
# dynamic_action — supplementary cross-domain ReAct sub-agent channel.
# Shares the research_lane with specialists; independent round caps so
# the two pools never starve each other. Red-line checks live in
# :meth:`PolicyGate._validate_dynamic_action_dispatch`.
# ---------------------------------------------------------------------------
DYNAMIC_ACTION_NAME: str = "dynamic_action"

# Roles allowed to dispatch a ``dynamic_action`` (sub-agents may not
# recursively spawn one).
DYNAMIC_ACTION_DISPATCH_SOURCE_ALLOWLIST: frozenset[str] = frozenset({
    "orchestration",
})

# At most one dispatch and one sourced variant per EXPLORE round.
MAX_DYNAMIC_PER_ROUND: int = 1
MAX_DYNAMIC_SOURCED_VARIANTS: int = 1

# Floor on the ``scope_domains`` list length.
DYNAMIC_ACTION_MIN_SCOPE_DOMAINS: int = 2

# Allowed values of the optional ``budget_hint`` field.
DYNAMIC_ACTION_BUDGET_HINTS: frozenset[str] = frozenset({
    "low", "medium", "high",
})

# Side-effect categories ``dynamic_action`` may never declare; mirrors
# the dispatch red lines (no own metric / accuracy gate / server /
# Magpie process).
DYNAMIC_ACTION_SIDE_EFFECT_RED_LINES: frozenset[str] = frozenset({
    "metric",
    "accuracy_gate",
    "server",
    "magpie",
})

# A ``scope_domains`` list consisting only of this literal collapses
# the dispatch to a kernel-only patch; rejected at dispatch.
DYNAMIC_ACTION_KERNEL_DOMAIN_LITERAL: str = "kernel"


# ---------------------------------------------------------------------------
# Analysis actions that are Coordinator-internal only
#
# ``roofline`` (composite: profile + trace_analyze + analysis.md
# snapshot) and ``profile`` (lightweight trace capture) are auto-managed
# by the Coordinator: enqueued at PRELUDE after baseline lands, and
# again on every +10% watermark crossing of ``last_roofline_tput``.
# Which kind runs is selected by ``shared_state.enable_roofline``
# (CLI flag ``--enable-roofline`` / ``--no-enable-roofline``, default
# on). The LLM does not propose either name; PolicyGate denies them at
# the intent boundary so the policy_denial event in the prompt nudges
# the LLM toward the proposable surface (``specialist`` / ``explore``
# / ``integrate_patch``).
#
# The denial is symmetric across delegate / propose_action / request:
# the rule fires *before* the kernel-owned + phase + unknown checks, so
# the canonical hint always wins.
# ---------------------------------------------------------------------------
INTERNAL_ONLY_ACTION_NAMES: frozenset[str] = frozenset({
    "roofline",
    "profile",
    # GAP 1 — ``replay_warm_recipe`` is enqueued exclusively by the
    # Coordinator at PRELUDE (after baseline lands) when the T0
    # warm-start ladder returned a high-confidence prior. Letting the
    # LLM propose it would (a) race the one-shot guard, (b) let a
    # specialist hand-craft a "warm replay" with adversarial args
    # that ought to go through ``explore`` instead. Same gate as
    # roofline / profile.
    "replay_warm_recipe",
})

# FRAMEWORK_PR phase: ``framework_pr`` is Coordinator-internal too
# (the new phase pumps candidates serially; the LLM never proposes the
# action). Kept as a separate set so the denial rule fires a distinct
# ``framework_pr_action_not_llm_proposable`` name with a hint pointing
# at ``--no-framework`` rather than the roofline/profile hint.
FRAMEWORK_PR_INTERNAL_ACTION_NAMES: frozenset[str] = frozenset({
    "framework_pr",
})


# ---------------------------------------------------------------------------
# v0.8 §3.11 R4 / R5 — external tool whitelist registry
#
# Tool names live here (the *policy* layer) so PolicyGate AND the
# SpecialistRunner share a single source of truth. The runner builds
# its per-task tool list from the role-whitelist table below; PolicyGate
# uses these constants for the intent-level R4 + R5 second pass
#.
#
# Naming convention follows the Claude / Cursor tool surface.
# ---------------------------------------------------------------------------

#: KB *write* surfaces. R4 ``kb_write_unauthorized`` denies any
#: intent that tries to invoke one — directly or via an action_name /
#: request.kind collision.
KB_WRITE_TOOL_NAMES: frozenset[str] = frozenset({
    "mcp__cortex_kb__propose_point",
    # The methods these tool names map to (propose_edge / hypothesize /
    # ingest_attempt / verify / commit) have been retired from the
    # client, but the tool names stay on the denylist because the
    # safety contract — KB writes are Coordinator-owned, not
    # specialist-callable — is independent of which methods currently
    # exist. Specialists that attempt to invoke any of these get an
    # immediate ``kb_write_unauthorized`` denial rather than a confusing
    # "tool not found" downstream.
    "mcp__cortex_kb__propose_edge",
    "mcp__cortex_kb__hypothesize",
    "mcp__cortex_kb__ingest_attempt",
    "mcp__cortex_kb__verify",
    "mcp__cortex_kb__commit",
})

#: KB *readonly* surfaces. R5 ``tool_whitelist_role`` requires the
#: caller to be a specialist sub-agent.
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

#: Web tools. R5 — specialist-only AND EXPLORE-phase-only (KB_design
#: §3.11 §4.5 table). Other roles / phases get
#: ``tool_whitelist_role`` / ``tool_whitelist_phase``.
WEB_TOOL_NAMES: frozenset[str] = frozenset({"WebSearch", "WebFetch"})

#: Role-→-allowed-toolset map (R5). The default agents (orchestration /
#: kernel / critic / robustness) never touch external knowledge tools;
#: only the specialist sub-agent does. Keep this map flat and explicit
#: so PolicyGate can do an O(1) membership check.
TOOL_WHITELIST_BY_ROLE: dict[str, frozenset[str]] = {
    "specialist": (
        WEB_TOOL_NAMES
        | PR_MONITOR_TOOL_NAMES
        | CORTEX_KB_READ_TOOL_NAMES
    ),
    # Empty sets — listing the roles explicitly so a typo
    # (``"orchestrator"`` vs ``"orchestration"``) becomes a key
    # error instead of a silent allow.
    "orchestration": frozenset(),
    "kernel": frozenset(),
    "critic": frozenset(),
    "robustness": frozenset(),
}

#: Phase whitelist for tools that are time-sensitive. Currently only
#: the ``WebSearch`` / ``WebFetch`` block carries a phase restriction;
#: KB readonly + PR Monitor are allowed in any phase (KB_design §3.11
#: §4.5 table).
PHASE_RESTRICTED_TOOLS: dict[str, frozenset[str]] = {
    "WebSearch": frozenset({"EXPLORE"}),
    "WebFetch": frozenset({"EXPLORE"}),
}

#: Convenience superset — every tool name PolicyGate knows about
#: (read or write). Useful for the R4 collision check on
#: ``propose_action.action_name`` so an LLM can't smuggle a tool
#: invocation through the action registry.
ALL_KNOWN_EXTERNAL_TOOL_NAMES: frozenset[str] = (
    KB_WRITE_TOOL_NAMES
    | CORTEX_KB_READ_TOOL_NAMES
    | PR_MONITOR_TOOL_NAMES
    | WEB_TOOL_NAMES
)


# Synthetic dataclass-ish stub used as ``role`` argument when validating
# path containment for specialist intents. We only need ``name`` for the
# error messages; specialist intents go through ``_validate_specialist_*``
# directly so the conventional role.allowed_intents matrix doesn't fire.
class _SpecialistPseudoRole:
    name = "specialist"


_SPECIALIST_PSEUDO_ROLE = _SpecialistPseudoRole()


# ---------------------------------------------------------------------------
# REQUEST/RESPONSE routing matrix (DESIGN §7.6 / §13.4)
#
# Maps source role → set of allowed target_agent names. v0.6: only
# orchestration→kernel is allowed.
# ---------------------------------------------------------------------------
REQUEST_ROUTING: dict[str, frozenset[str]] = {
    "orchestration": frozenset({"kernel"}),
}


# ---------------------------------------------------------------------------
# Critic-only: REVIEW_VERDICT (DESIGN §18.2)
# ---------------------------------------------------------------------------
REVIEW_VERDICT_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"critic"})

# Verdict vocabulary for review_verdict (DESIGN §18.2)
REVIEW_VERDICTS: frozenset[str] = frozenset({
    "approve", "reject", "redirect", "advise", "needs_review",
})


# ---------------------------------------------------------------------------
# Robustness-only: kill_task + scheduling-police intents (DESIGN §7.4 / §19.3)
# ---------------------------------------------------------------------------
KILL_TASK_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"robustness"})
KILL_TASK_ALLOWED_SCOPES: frozenset[str] = frozenset({"task"})

ROBUSTNESS_ONLY_INTENTS: frozenset[IntentType] = frozenset({
    IntentType.FORCE_DISPATCH,
    IntentType.PRUNE_BRANCH,
    IntentType.ESCALATE_STRATEGY_CHANGE,
})
ROBUSTNESS_ONLY_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"robustness"})

# Roofline-v2 C3: per-intent source allowlist override. PRUNE_BRANCH widens
# to ``orchestration`` as well, because the ``roofline`` action (C4) produces
# structured prune suggestions that the main Orchestration LLM consumes via
# the rendered prompt (C5) and then forwards to the Coordinator. The other
# two scheduling-police intents (FORCE_DISPATCH, ESCALATE_STRATEGY_CHANGE)
# stay robustness-only — they are recovery-shaped intents that bypass normal
# task accounting and shouldn't be reachable from optimisation-flow LLMs.
#
# Lookups fall through to ROBUSTNESS_ONLY_SOURCE_ALLOWLIST when an intent is
# not listed here, so adding a new ROBUSTNESS_ONLY_INTENTS entry remains
# robustness-only by default.
_ROBUSTNESS_ONLY_INTENT_SOURCES: dict[IntentType, frozenset[str]] = {
    IntentType.PRUNE_BRANCH: frozenset({"robustness", "orchestration"}),
}


# ---------------------------------------------------------------------------
# SESSION_DIR path containment ().
#
# Any payload field listed in _PATH_LIKE_FIELDS must point either
# (a) inside the active session_dir, OR
# (b) under one of the framework source allowlists below (so kernel-agent
#     can reference aiter/sglang/vllm source trees without violation).
#
# The check is applied recursively to dict values; nested dicts (e.g.
# request.params.trace_input) are walked.
# ---------------------------------------------------------------------------
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

# `source_file` is a special case — kernel agents reference framework
# source trees that legitimately live outside session_dir. We allowlist
# the well-known parents here; anything else falls through to the
# session_dir containment check.
# Field name-only allowlist: when the payload key is `source_file`, the
# value may match :func:`resolve_source_file_allowlist` instead of being
# session-rooted. Resolved at check time so importlib/glob discovery and
# env overrides apply without a process restart.
SOURCE_LIKE_FIELDS: frozenset[str] = frozenset({"source_file"})


# Multi-node profile trace shared dirs. In multi-node runs, server pods
# write torch traces to a shared-FS path that the sandbox also mounts;
# that path lives outside session_dir but must be referenceable by
# trace_dir / main_trace_path / trace_input so kernel-agent input flows
# work. The allowlist intentionally only covers prefixes mkdir'd by the
# sandbox CLI under our namespace; arbitrary writes remain blocked.
#
# Runtime-resolved (via :func:`_trace_path_allowlist`) so the allowlist
# follows ``$USER_DATA_PATH`` instead of hard-coding a cluster mount
# point. See :func:`inference_optimizer.paths.mn_profile_trace_root`.
def _trace_path_allowlist() -> tuple[str, ...]:
    """Multi-node profile trace path-prefix allowlist (runtime-resolved).

    Returns the set of path prefixes (each terminated by ``/``) that
    PolicyGate accepts for ``TRACE_PATH_LIKE_FIELDS`` values escaping
    ``session_dir``. The trailing ``/`` is load-bearing — without it a
    ``str.startswith`` check would match a sibling dir whose name shares
    the prefix as a substring.
    """
    from ..paths import mn_profile_trace_root
    root = str(mn_profile_trace_root()).rstrip("/") + "/"
    return (root,)

# Subset of PATH_LIKE_FIELDS for which :func:`_trace_path_allowlist`
# is also accepted (in addition to session_dir containment). Other path
# fields such as workspace, output_dir, report_path remain strictly
# session-rooted to preserve sandbox-isolation guarantees.
TRACE_PATH_LIKE_FIELDS: frozenset[str] = frozenset({
    "trace_dir",
    "main_trace_path",
    "trace_input",
})


# ---------------------------------------------------------------------------
# Core SharedState fields that only the Coordinator may mutate.
# ---------------------------------------------------------------------------
CORE_STATE_FIELDS: frozenset[str] = frozenset({
    "current_best",
    "stop_reason",
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
    # the fact-layer KEEP ledger. Coordinator is the
    # sole writer (Inv-1 / Inv-10.2); LLM update_state can never
    # rewrite the stack, even though the LLM proposes the entries that
    # land in it via emit_intent → execute → promote flows.
    "optimization_stack",
    "gain_per_stack_entry",
    # schema_version is a migration breadcrumb; an
    # LLM update_state must not be able to roll the state.json back to
    # a reader by setting ``schema_version=1``.
    "schema_version",
    # Cortex KB integration fields (KB_design §3.6, §3.10,
    # §3.13 M1). Coordinator-only writes; LLM agents reading is fine.
    "cortex_session_id",
    "cortex_session_summary",
    "warm_start_recipe",
    "warm_start_pitfalls",
    "warm_start_lessons",
    "warm_start_ts",
    # GAP 5 KB tag completeness — populated by Coordinator from
    # manifest + baseline materialized config. LLM agents can read
    # them via prompt sections, but only Coordinator writes.
    "stack_fingerprint_meta",
    "baseline_workload_extra",
    # GAP 1 warm-recipe replay — one-shot guard + outcome record.
    # Coordinator-only writes; LLM cannot edit them via update_state
    # (would let a misbehaving LLM bypass the replay budget).
    "warm_replay_attempted",
    "warm_replay_outcome",
    "warm_history_injected",
    # phase state machine fields (KB_design §3.2, §3.10,
    # §3.13 M2). All managed by ``Coordinator._advance_phase_if_needed``;
    # LLM update_state never reaches these.
    "phase",
    "phase_started_ts",
    "phase_started_unix",
    "phase_history",
    "phase_budget_pct",
    # specialist sub-agent ledger (KB_design §3.5 §10 / §3.10
    # §4.1 / §3.11). Coordinator-only writes; LLM cannot inject
    # arbitrary entries via update_state (specialist_done carries
    # proposals through the dedicated R3 path instead).
    "specialist_rounds",
    "specialist_domain_empty_streak",
    "last_specialist",
    # research_lane capacity is set once at CLI/manifest time
    # and mirrored into SharedState. Locking it as CORE prevents an LLM
    # from raising capacity mid-flight.
    "research_lane_capacity",
    # phase-machine escalation plumbing (KB_design §3.8 §7.3 /
    # §3.13 M7). Coordinator's ``_handle_escalate_strategy_change``
    # writes ``pending_escalate_hint`` via the validated
    # ``SharedState.set_pending_escalate_hint`` helper; LLM
    # ``update_state`` is blocked here as a defense-in-depth measure
    # so an arbitrary intent can't drop the phase machine into
    # ``skip_to_close``.
    "pending_escalate_hint",
    "last_consumed_escalate_hint",
    "last_consumed_escalate_hint_ts",
    "plateau_overrides",
    # CLOSE phase sequencer flag.
    # Set by Coordinator at the end of the 5-step sequencer so
    # ``cli.finally`` can short-circuit its emergency breakdown
    # write. LLM update_state must not be able to toggle this and
    # trick the cli into skipping its safety net.
    "close_sequence_done",
    # explore search ledger.
    # Coordinator's ``apply_explore_search_update`` / ``record_explore_accepted``
    # are the sole writers; LLM ``update_state`` must not rewrite the ledger
    # directly (would bypass dedup-by-fingerprint + Inv-1).
    "explore_search",
    # structured gaps ledger (KB_design §3.3 /
    # §3.5 / §3.9 §6). Coordinator's ``_refresh_gaps`` is the sole
    # writer; LLM agents read via prompt injection. Locking the field
    # closes the proxy gap (last_action_failures + winners_history)
    # against an arbitrary update_state that would inject fake gaps
    # to bias specialist domain selection (Inv-1 / Inv-10.2).
    "gaps",
    # Coordinator-only writes on the dynamic_action aggregate view +
    # round counter so the LLM cannot self-narrate its dispatch
    # outcomes via UPDATE_STATE.
    "dynamic_actions",
    "dynamic_action_round_count",
})


# ---------------------------------------------------------------------------
@dataclass
class PolicyGate:
    """Validate every intent emitted by an agent reactor.

    Attributes:
        role_registry:   name → AgentRole lookup
        action_registry: optional name → ActionMetadata lookup; v0.6 fallback
                         accepts any action name not on KERNEL_OWNED_ACTIONS
                         (no mode gating; single full mode per ADR-34).
        session_dir:     active session root for path-containment checks.
                         When None, the path check is skipped.
        strict_paths:    when True (production), payload field values that
                         match :data:`PATH_LIKE_FIELDS` MUST resolve under
                         ``session_dir`` (or the source-file allowlist for
                         ``source_file``). Production CLI flips this on;
                         legacy tests with ``/tmp/<fixture>.json`` fixtures
                         keep it False. ``$INFERENCE_OPTIMIZER_STRICT_PATHS=1``
                         in env also enables it for in-process callers.
    """

    role_registry: dict[str, "AgentRole"]
    action_registry: Any | None = None
    session_dir: Path | None = None
    strict_paths: bool = False
    shared_state: Any | None = None
    # phase-incompatible R1 enforcement mode.  When False
    # (default) the rule only emits a warning entry into the audit log;
    # production CLI flips this on via ``INFERENCE_OPTIMIZER_STRICT_PHASE=1``
    # to fail-closed.  Two-stage rollout matches the legacy →
    # v0.8 transition strategy in KB_design §3.13 M2.
    strict_phase: bool = False

    def __post_init__(self) -> None:  # noqa: D401 — dataclass hook
        # Allow env to enable strict mode without threading a constructor
        # arg through every Coordinator caller (tests use a monkeypatched
        # env to opt in).
        import os as _os
        if not self.strict_paths and _os.environ.get(
            "INFERENCE_OPTIMIZER_STRICT_PATHS", ""
        ).strip() in ("1", "true", "yes"):
            self.strict_paths = True
        if not self.strict_phase and _os.environ.get(
            "INFERENCE_OPTIMIZER_STRICT_PHASE", ""
        ).strip() in ("1", "true", "yes"):
            self.strict_phase = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate_intent(self, from_agent: str, intent: Intent) -> None:
        """Raise :class:`PolicyDenied` if the intent is not allowed.

        Order of checks (cheapest first):

            1. Agent must be a known role (or a ``specialist:<task_id>``
               ephemeral identity)
            2. ``intent.type`` must be in ``role.allowed_intents``
            3. Per-intent type structural rules
            4. Cross-source allowlists (review_verdict / kill_task /
               robustness-only)
        """
        # specialist sub-agents emit intents under an ephemeral
        # ``specialist:<task_id>`` identity. They get
        # routed to a synthetic role with a tightly-scoped intent set
        # (specialist_done + base inbox intents) and the R3 from-agent
        # contract is enforced against the task_id suffix.
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
        # ANSWER / ASK_QUESTION / UPDATE_PERSONA / ALERT carry no
        # extra side-effect checks beyond the role gate.

        # Path-containment guard — every payload that travels through
        # the bus is scanned for `_PATH_LIKE_FIELDS`; offending paths
        # raise PolicyDenied(rule="path_outside_session_dir").
        self._validate_payload_paths(role, intent.type, payload)

    def _closing_phase_denial(
        self, source: str, intent: Intent,
    ) -> PolicyDenied | None:
        """During closing phase, only harmless intents and ``report`` proposals."""
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
        """Return the Claude tool list a reactor may use.

        Codex roles → ``[]`` (no-tools). Claude roles → ``["emit_intent"]``
        in the legacy release; per-action Read/Bash/Edit injection happens in
        SubAgentRunner (P0-3) and via :meth:`allowed_tools_for_action`.
        """
        role = self.role_registry.get(agent_name)
        if role is None:
            return []
        if role.no_tools:
            return []
        return ["emit_intent"]

    def allowed_tools_for_action(self, action_name: str) -> list[str]:
        """Per-action tool intersection used by SubAgentRunner.

        Returns the action's declared ``allowed_tools`` from metadata, or
        the conservative default ``["emit_intent"]`` when no
        ActionRegistry is wired or the action is unknown.
        """
        if self.action_registry is None:
            return ["emit_intent"]
        meta = self.action_registry.get(action_name)
        if meta is None:
            return ["emit_intent"]
        return list(meta.allowed_tools)

    # ------------------------------------------------------------------
    # Per-intent validators
    # ------------------------------------------------------------------
    def _validate_delegate(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        if not role.can_delegate_side_effects:
            raise PolicyDenied(
                f"role={role.name!r} cannot delegate side-effecting actions",
                rule="role",
            )
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied("delegate intent missing action_name", rule="payload")
        # analysis_action_not_llm_proposable —
        # roofline / profile are Coordinator-internal (PRELUDE bootstrap
        # + watermark-triggered). Block the LLM from racing the auto
        # path regardless of which channel (delegate / propose / request)
        # it tries to smuggle the name through.
        self._validate_action_not_llm_proposable(
            action_name, intent_kind="delegate",
        )
        # Plan A — kernel-owned actions are not directly delegatable.
        if action_name in KERNEL_OWNED_ACTIONS:
            raise PolicyDenied(
                f"action={action_name!r} is owned by the kernel agent; "
                f"emit REQUEST(target_agent='kernel', kind='...') instead "
                f"of delegate(action_name={action_name!r})",
                rule="kernel_owned_by_kernel_agent",
            )
        # R2 ``specialist`` is a synthetic action that bypasses
        # ActionRegistry. The
        # per-payload contract (domain / gap / max_turns) is enforced by
        # ``_validate_specialist_dispatch`` instead of the generic
        # registry path.
        if action_name == SPECIALIST_ACTION_NAME:
            self._validate_specialist_dispatch(role, payload)
            self._validate_phase_action(role, action_name, intent_kind="delegate")
            return
        # dynamic_action — phase, source, payload, and red-line checks
        # live in ``_validate_dynamic_action_dispatch``. The dedicated
        # validator runs before the generic phase check so a wrong-phase
        # emit surfaces ``dynamic_phase_violation`` (not the generic
        # ``phase_incompatible``).
        if action_name == DYNAMIC_ACTION_NAME:
            self._validate_dynamic_action_dispatch(role, payload)
            return
        # PR-A7 (Arbor-into-Hyperloom) — ``integrate_patch`` requires a
        # non-reject Critic verdict on the specialist's patches before
        # the orchestrator can apply them to framework_source_roots.
        # See SharedState.specialist_patch_verdicts /
        # record_specialist_patch_verdict; ``bypass_critic=True`` lets
        # an operator override the gate (rare, audit-trail visible
        # via the policy_denied → bypass override pattern).
        if action_name == INTEGRATE_PATCH_ACTION_NAME:
            self._validate_integrate_patch_critic_gate(payload)
            # Continue into the standard registry + phase checks below.
        # PR-A9 (Arbor-into-Hyperloom) — single-agent explore is retired.
        # Every explore grid variant must trace to either a specialist
        # (provenance='specialist:<domain>') or the cold-start default
        # grid (provenance='default_grid'). The legacy 'llm_direct'
        # provenance — where the orchestration LLM authored the grid
        # from a single prompt window without any specialist research —
        # is denied here so the EXPLORE phase becomes specialist-first
        # for every round, matching Arbor's optimization loop.
        if action_name == EXPLORE_ACTION_NAME:
            self._validate_explore_provenance(payload)
            self._validate_explore_grid_size(payload)
        # sweep_phase_singleton: deny LLM-emitted
        # sweep when the Coordinator's SWEEP-entry hook already
        # auto-enqueued one. Two concurrent sweep tasks crash both
        # vllm engines on init; see _validate_sweep_singleton.
        if action_name == SWEEP_ACTION_NAME:
            self._validate_sweep_singleton(payload, intent_kind="delegate")
        # Same gain-driven / explore-minimum gates apply at the
        # delegate channel so kernel_opt cannot bypass them by skipping
        # propose_action.
        self._validate_fp8_only_action(action_name, intent_kind="delegate")
        self._validate_gain_driven_kernel_opt(action_name)
        self._validate_explore_minimum_before_kernel_opt(action_name)
        # If an ActionRegistry is wired, refuse delegate for unknown action names.
        # No registry → fall through (P0 / dev-mode where registry isn't loaded).
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
        # Per-action required-payload guard (e.g. ``recover`` must carry
        # ``reason`` + ``evidence`` so the audit trail captures the symptom).
        # Fields are accepted at the top of the payload OR nested under
        # ``payload["params"]`` (the structure robustness emits).
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
        # R1 phase_incompatible. Runs **after** the role +
        # kernel-ownership + unknown_action checks so the cheaper /
        # structural denials win when both apply (Inv-11.3 orthogonality).
        self._validate_phase_action(role, action_name, intent_kind="delegate")
        # v0.8 §3.11 R4 / R5 — block any ``delegate`` whose action_name
        # tries to invoke an external tool via the intent channel.
        self._validate_no_kb_write_collision(
            action_name, intent_kind="delegate",
        )
        self._validate_tool_whitelist_collision(
            role.name, action_name, intent_kind="delegate",
        )

    def _validate_propose_action(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied("propose_action missing action_name", rule="payload")
        # analysis_action_not_llm_proposable
        # (propose channel) — same rule, same hint.
        self._validate_action_not_llm_proposable(
            action_name, intent_kind="propose_action",
        )
        # Soft check — propose is advisory; only reject if registry is wired
        # AND the name is unknown AND it's not a kernel-owned action (which
        # are listed in metadata under their canonical names).
        if (
            self.action_registry is not None
            and action_name not in KERNEL_OWNED_ACTIONS
            and self.action_registry.get(action_name) is None
        ):
            raise PolicyDenied(
                f"propose_action: unknown action_name={action_name!r} "
                f"(not in ActionRegistry)",
                rule="unknown_action",
            )
        # sweep_phase_singleton (defense in depth on
        # the propose_action channel; same shape as the delegate
        # validator). See _validate_sweep_singleton.
        if action_name == SWEEP_ACTION_NAME:
            self._validate_sweep_singleton(
                payload, intent_kind="propose_action",
            )
        # PR-A9 + explore_specialist_grid_max_one — same explore-grid
        # gates as the delegate channel. Without these, an LLM that
        # cannot delegate{action_name='explore', ...} (e.g. wrong role
        # or phase guard fires first) can still propose_action an
        # explore grid full of llm_direct or many specialist:* variants
        # and have the Coordinator materialise it. Mirror both rules
        # here so the propose advisory path can never sidestep PR-A9
        # or the new per-round specialist cap.
        if action_name == EXPLORE_ACTION_NAME:
            self._validate_explore_provenance(payload)
            self._validate_explore_grid_size(payload)
        # F3-2 (Roofline-v2 N19c): gain-driven kernel_opt lock — until
        # cheap rounds have plateaued, kernel_opt is denied so we don't
        # spend a 30 min GPU lane on a cheap-still-earning frontier.
        self._validate_fp8_only_action(action_name, intent_kind="propose_action")
        self._validate_gain_driven_kernel_opt(action_name)
        # F3-5: explore-attempt minimum guard — kernel_opt requires at
        # least one successful explore round on record.
        self._validate_explore_minimum_before_kernel_opt(action_name)
        # R1 phase_incompatible.
        self._validate_phase_action(role, action_name, intent_kind="propose_action")
        # v0.8 §3.11 R4 / R5 — defense in depth on propose_action.
        self._validate_no_kb_write_collision(
            action_name, intent_kind="propose_action",
        )
        self._validate_tool_whitelist_collision(
            role.name, action_name, intent_kind="propose_action",
        )

    # ------------------------------------------------------------------
    # Propose_action sub-gates (F3 series)
    #
    # Reading order in the dispatch path:
    #   1. _validate_gain_driven_kernel_opt                 (N19c)
    #   2. _validate_explore_minimum_before_kernel_opt      (F3-5)
    # ------------------------------------------------------------------

    _N19C_HISTORY_WINDOW: int = 3
    _N19C_EPSILON_PCT: float = 0.5

    def _validate_gain_driven_kernel_opt(self, action_name: str) -> None:
        """F3-2 (N19c): lock ``kernel_opt`` until cheap exploration
        plateaus.

        Reads the last ``_N19C_HISTORY_WINDOW`` entries of
        :attr:`SharedState.gain_per_stack_entry` (the canonical
        per-KEEP delta_pct ledger this branch already maintains; v0.8
        replacement for v0.6's ``last_cheap_explore_gain_pct``).
        Denies if the moving-average ``delta_pct`` is still
        ``>= _N19C_EPSILON_PCT``: cheap rounds are still earning,
        burning a kernel-opt lane is premature.

        Toggle: :attr:`SharedState.gain_driven_kernel_opt` (F0-10,
        default off).
        """
        if action_name != "kernel_opt":
            return
        ss = getattr(self, "shared_state", None)
        if ss is None:
            return
        if not bool(getattr(ss, "gain_driven_kernel_opt", False)):
            return
        # Defer to the phase allowlist when kernel_opt is proposed
        # outside the KERNEL phase — the phase_incompatible rule's
        # message is more actionable there. N19c is a within-KERNEL
        # gate, not a phase gate.
        if str(getattr(ss, "phase", "") or "") != "KERNEL":
            return
        history = list(getattr(ss, "gain_per_stack_entry", []) or [])
        # Keep only entries with a real per-round delta on record
        # (resumed / seeded entries use ``delta_pct=None``).
        deltas: list[float] = []
        for entry in reversed(history):
            if not isinstance(entry, dict):
                continue
            d = entry.get("delta_pct")
            if isinstance(d, (int, float)):
                deltas.append(float(d))
            if len(deltas) >= self._N19C_HISTORY_WINDOW:
                break
        if len(deltas) < self._N19C_HISTORY_WINDOW:
            raise PolicyDenied(
                f"propose_action: kernel_opt is locked: only "
                f"{len(deltas)} cheap-round delta(s) on record "
                f"(need {self._N19C_HISTORY_WINDOW} for the gain-trend "
                f"window). Run more explore / specialist rounds first.",
                rule="n19c_gain_driven_kernel_opt",
                hint=(
                    "propose_action{action_name='explore'} or "
                    "delegate{action_name='specialist', ...} until the "
                    "ledger has at least "
                    f"{self._N19C_HISTORY_WINDOW} integrated rounds."
                ),
            )
        avg = sum(deltas) / float(len(deltas))
        if avg >= self._N19C_EPSILON_PCT:
            raise PolicyDenied(
                f"propose_action: kernel_opt is locked: cheap rounds "
                f"still earning (last {len(deltas)} avg "
                f"{avg:+.2f}% >= {self._N19C_EPSILON_PCT:.2f}%). "
                f"Continue cheap rounds until the average drops below "
                f"the threshold.",
                rule="n19c_gain_driven_kernel_opt",
                hint=(
                    "propose_action{action_name='explore'} or "
                    "delegate{action_name='specialist', ...}"
                ),
            )

    def _validate_explore_minimum_before_kernel_opt(
        self, action_name: str,
    ) -> None:
        """F3-5: ``kernel_opt`` requires at least one successful explore
        round on record.

        Successful = at least one entry in ``gain_per_stack_entry``,
        which is appended only when ``optimization_stack`` accepts a
        variant. This is also the same signal Coordinator uses to
        compute ``cumulative_gain_validated``.

        Always-on (no toggle) — this is a baseline correctness rule;
        without it the LLM could enter KERNEL phase and burn a
        kernel-opt lane before it knows whether cheap rounds would
        have closed the gap.
        """
        if action_name != "kernel_opt":
            return
        ss = getattr(self, "shared_state", None)
        if ss is None:
            return
        # Phase allowlist owns the "kernel_opt outside KERNEL" denial;
        # F3-5 is a within-KERNEL correctness rule. Skipping here keeps
        # the legacy ``phase_incompatible`` message fired by
        # _validate_phase_action when an LLM proposes kernel_opt in
        # PRELUDE / FRAMEWORK_PR / EXPLORE.
        if str(getattr(ss, "phase", "") or "") != "KERNEL":
            return
        history = list(getattr(ss, "gain_per_stack_entry", []) or [])
        accepted = sum(1 for e in history if isinstance(e, dict))
        if accepted >= 1:
            return
        raise PolicyDenied(
            f"propose_action: kernel_opt requires at least one "
            f"successful explore round first (gain_per_stack_entry "
            f"length = {accepted}). Run explore + integrate_patch "
            f"before invoking kernel_opt.",
            rule="explore_attempts_minimum_before_kernel_opt",
            hint="propose_action{action_name='explore'} first.",
        )

    def _validate_state_transition(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        changes = payload.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise PolicyDenied(
                "update_state.payload.changes must be a non-empty dict",
                rule="payload",
                hint=("include at least one allowed field, e.g. "
                      "{'changes': {'current_action': '<action_name>'}}"),
            )
        if role.can_mutate_core_state:
            return
        violating = sorted(set(changes.keys()) & CORE_STATE_FIELDS)
        if violating:
            raise PolicyDenied(
                f"role={role.name!r} cannot mutate core state fields: {violating!r}",
                rule="state_field",
            )

    def _validate_send_message_topic(self, payload: dict[str, Any]) -> None:
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            raise PolicyDenied("send_message missing topic", rule="payload")
        # Unknown topics are soft-degraded by the Coordinator to "observation"
        # (DESIGN §13.2). PolicyGate doesn't reject them outright so agents
        # can still surface unstructured observations.

    def _validate_request(self, role: "AgentRole", payload: dict[str, Any]) -> None:
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
        # analysis_action_not_llm_proposable —
        # defense in depth: nobody REQUESTs roofline/profile today, but
        # an extension that does would race the auto-managed gate.
        self._validate_action_not_llm_proposable(kind, intent_kind="request")
        # R1 phase_incompatible. For
        # orchestration → kernel REQUEST we treat the request *kind* as
        # the action name (kernel-owned actions named identically to
        # their REQUEST kind: kernel_opt / integrate / etc.).
        if target == "kernel" and kind in KERNEL_OWNED_ACTIONS:
            self._validate_phase_action(role, kind, intent_kind="request")
        self._validate_fp8_only_action(kind, intent_kind="request")
        # v0.8 §3.11 R4 / R5 — defense in depth: a REQUEST.kind cannot
        # smuggle a KB write / external tool invocation either.
        self._validate_no_kb_write_collision(kind, intent_kind="request")
        self._validate_tool_whitelist_collision(
            role.name, kind, intent_kind="request",
        )

    def _validate_response(self, payload: dict[str, Any]) -> None:
        in_reply_to = str(payload.get("in_reply_to", "")).strip()
        if not in_reply_to:
            raise PolicyDenied("response missing in_reply_to", rule="payload")
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            raise PolicyDenied("response missing kind", rule="payload")

    def _validate_review_verdict(self, role: "AgentRole", payload: dict[str, Any]) -> None:
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
        #         # accept either the legacy single ``verdict`` field or the
        # per-variant ``verdict_map``. The intent_parser already
        # enforced mutual exclusion + structural shape; here we
        # validate the *content* (verdict strings must be in the
        # closed REVIEW_VERDICTS vocab).
        has_single = "verdict" in payload
        verdict_map = payload.get("verdict_map")
        has_map = isinstance(verdict_map, dict) and bool(verdict_map)
        if has_single == has_map:
            # Both or neither — defense in depth (intent_parser
            # should have caught this already).
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
        # verdict_map path — every entry's verdict string must be in
        # the same closed vocab. variant_name vs original-grid
        # membership is checked by Coordinator's
        # ``_handle_verdict_map`` once the grid is in scope.
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

    # ------------------------------------------------------------------
    # analysis_action_not_llm_proposable —
    # deny LLM proposals of ``roofline`` / ``profile``
    # ------------------------------------------------------------------
    def _validate_action_not_llm_proposable(
        self,
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject LLM-proposed analysis actions.

        ``roofline`` and ``profile`` are Coordinator-auto-managed at
        PRELUDE bootstrap and on every +10% watermark crossing; the
        kind selected is controlled by
        :attr:`SharedState.enable_roofline` (``--enable-roofline`` /
        ``--no-enable-roofline``). The LLM has no business proposing
        either name — doing so would race the auto-managed pending-task
        gate and could double-enqueue.

        No-op when ``action_name`` isn't in
        :data:`INTERNAL_ONLY_ACTION_NAMES`. Fires *before* the
        kernel-owned / phase / unknown gates, so the canonical hint
        always wins.
        """
        if not action_name:
            return
        if action_name in FRAMEWORK_PR_INTERNAL_ACTION_NAMES:
            raise PolicyDenied(
                f"action {action_name!r} is Coordinator-internal; the LLM "
                f"must not propose it ({intent_kind})",
                rule="framework_pr_action_not_llm_proposable",
                hint=(
                    "``framework_pr`` is driven by the FRAMEWORK_PR phase "
                    "pump (one candidate per tick, plateau-based exit). "
                    "Propose ``specialist`` or ``explore`` instead, or "
                    "pass ``--no-framework`` to skip the phase entirely."
                ),
            )
        if action_name not in INTERNAL_ONLY_ACTION_NAMES:
            return
        raise PolicyDenied(
            f"action {action_name!r} is Coordinator-internal; the LLM "
            f"must not propose it ({intent_kind})",
            rule="analysis_action_not_llm_proposable",
            hint=(
                "roofline / profile / replay_warm_recipe are auto-enqueued "
                "by the Coordinator (PRELUDE bootstrap + +10% watermark "
                "crossings + warm-recipe replay). Selection is controlled "
                "by ``--enable-roofline`` / ``--no-enable-roofline`` / "
                "``--no-warm-replay``. Propose ``specialist`` or "
                "``explore`` instead — these analysis snapshots will "
                "refresh automatically when their gate fires."
            ),
        )

    # ------------------------------------------------------------------
    # NOTE: no ``framework_atom_action_unsupported`` rule exists. atom
    # has no action that needs framework-specific denial at the
    # PolicyGate layer — multi-node is guarded at the CLI level, and
    # ``framework_pr`` is still caught for all frameworks by the
    # earlier ``framework_pr_action_not_llm_proposable`` rule (LLMs
    # cannot propose ``framework_pr`` regardless of framework; the
    # Coordinator drives it directly).
    #
    # Anti-regression guards live in
    # ``inference_optimizer/tests/test_policy_atom_invariants.py``
    # (asserts the constant + helper symbols stay absent) so a future
    # reintroduction has to be intentional.

    # ------------------------------------------------------------------
    # R1 phase_incompatible
    # ------------------------------------------------------------------
    def _validate_phase_action(
        self,
        role: "AgentRole",
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject an action that isn't legal in the current phase.

        Behaviour matrix (mode flipping via ``strict_phase``):

        * ``strict_phase=True`` (production) → raise PolicyDenied with
          ``rule='phase_incompatible'`` so the LLM self-corrects via
          the inbox ``policy_denied`` event.
        * ``strict_phase=False`` (legacy / tests) → swallow the denial
          but bump :attr:`policy_denial_streak` so the audit trail
          surfaces the would-be denial. Returns silently.

        Cheap path: when SharedState is missing / phase isn't
        initialised, the rule is a no-op (Inv-2.1 doesn't apply to a
        run that hasn't entered the machine yet — Coordinator sets
        phase before the first reactor tick anyway).
        """
        state = self.shared_state
        if state is None:
            return
        phase = (getattr(state, "phase", "") or "").strip().upper()
        if not phase or phase not in PHASE_NAMES:
            return
        if is_action_allowed_in_phase(action_name, phase):
            return
        allowed = allowed_actions_for(phase)
        hint = (
            f"you are in phase={phase}; action {action_name!r} is not in "
            f"the allowed set {list(allowed)!r}. Either propose an "
            f"action from that list, or wait for the Coordinator to "
            f"advance the phase. See KB_design §3.2 for the per-phase "
            f"action contract."
        )
        if not self.strict_phase:
            # Warn-only: keep the run flowing for legacy tests but make
            # the audit trail visible via policy_denial_streak.
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

    # ------------------------------------------------------------------
    # FP8-only actions
    # ------------------------------------------------------------------
    def _validate_fp8_only_action(
        self,
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject GEMM tuning for non-FP8 sessions.

        ``gemm_tuning`` drives aiter A8W8 block-scale FP8 GEMM CSV
        dispatch. Running it on BF16 / non-quantized workloads wastes the
        serving lane and may patch the wrong framework path, so enforce the
        precision contract at the intent boundary. The handler repeats the
        same check as defense in depth for programmatic callers.
        """
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

    # ------------------------------------------------------------------
    # v0.8 §3.11 R4 — kb_write_unauthorized
    # ------------------------------------------------------------------
    def _validate_no_kb_write_collision(
        self,
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject any intent whose ``action_name`` / ``request.kind``
        equals a Cortex KB *write* tool name.

        Defense in depth — none of the canonical actions in
        ``ActionRegistry`` ever collide with these names, so a real
        v0.8 run will never reach this branch via a valid registry
        entry. The rule fires when an LLM tries to smuggle a KB write
        via the propose / delegate / request channels (or an
        operator extension accidentally registers an action with a
        cortex_kb name). KB_design §3.11 §4.4 / Inv-11.3.
        """
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

    # ------------------------------------------------------------------
    # v0.8 §3.11 R5 — tool_whitelist_role / tool_whitelist_phase
    #
    # ------------------------------------------------------------------
    def _validate_tool_whitelist_collision(
        self,
        role_name: str,
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject any intent whose ``action_name`` / ``request.kind``
        equals an external tool name not on the caller's role
        whitelist.

        Read-only Cortex KB / PR Monitor / Web tools are specialist-
        only; the four primary agents (orchestration / kernel /
        critic / robustness) never reach for them through an intent.
        KB_design §3.11 §4.5.
        """
        if not action_name:
            return
        # Only externally-known tool names trigger R5. The R4 check
        # already covers KB write names — keep them out of this
        # branch so a write attempt produces ``kb_write_unauthorized``
        # (R4) rather than a less-specific ``tool_whitelist_role`` (R5).
        if action_name in KB_WRITE_TOOL_NAMES:
            return
        if action_name not in ALL_KNOWN_EXTERNAL_TOOL_NAMES:
            return
        allowed_for_role = TOOL_WHITELIST_BY_ROLE.get(role_name, frozenset())
        if action_name in allowed_for_role:
            # Role allows it — check the phase restriction if any.
            phase = ""
            if self.shared_state is not None:
                phase = (
                    getattr(self.shared_state, "phase", "") or ""
                ).strip().upper()
            phase_allowed = PHASE_RESTRICTED_TOOLS.get(action_name)
            if phase and phase_allowed and phase not in phase_allowed:
                raise PolicyDenied(
                    f"tool {action_name!r} is restricted to "
                    f"phase={sorted(phase_allowed)!r}; current "
                    f"phase={phase!r}",
                    rule="tool_whitelist_phase",
                    hint=(
                        f"{action_name} is restricted to "
                        f"{sorted(phase_allowed)} phase(s). Wait for "
                        f"the phase transition or use a phase-allowed "
                        f"alternative (e.g. KB readonly / PR Monitor "
                        f"are any-phase). KB_design §3.11 §4.5."
                    ),
                )
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

    # ------------------------------------------------------------------
    # v0.8 §3.11 R4 / R5 public helper — pure validator usable by the
    # SpecialistRunner per-task tool-list builder.
    # ------------------------------------------------------------------
    def validate_tool_invocation(
        self,
        tool_name: str,
        *,
        source_role: str,
        phase: str | None = None,
    ) -> None:
        """Raise :class:`PolicyDenied` if ``tool_name`` is not
        allowed for ``source_role`` (and optional ``phase``).

        Pure function — Inv-11.1. Returns ``None`` when the tool is
        allowed. Intended for tool-list builders that need a single
        source of truth: the SpecialistRunner can call this on each
        candidate tool before passing the list to the LLM backend.

        ``phase`` overrides the live ``shared_state.phase`` — handy in
        unit tests that don't wire a SharedState.
        """
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
        # R5 — role + phase whitelist for the known external tools.
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
            phase_value = (
                (phase or "").strip().upper()
                if phase is not None
                else (
                    (getattr(self.shared_state, "phase", "") or "")
                    .strip().upper()
                    if self.shared_state is not None else ""
                )
            )
            phase_allowed = PHASE_RESTRICTED_TOOLS.get(tool_name)
            if phase_value and phase_allowed and phase_value not in phase_allowed:
                raise PolicyDenied(
                    f"tool {tool_name!r} is restricted to "
                    f"phase={sorted(phase_allowed)!r}; current "
                    f"phase={phase_value!r}",
                    rule="tool_whitelist_phase",
                    hint=(
                        f"{tool_name} is restricted to "
                        f"{sorted(phase_allowed)} phase(s). KB_design "
                        f"§3.11 §4.5."
                    ),
                )
        # Anything else is implicitly allowed — PolicyGate doesn't
        # try to enumerate every internal tool (Read / Grep / Glob /
        # emit_intent / Bash). The SpecialistRunner still filters
        # against its own ``SPECIALIST_TOOL_DENYLIST`` for the local
        # tools that don't pass through PolicyGate.

    # ------------------------------------------------------------------
    # R2 ``specialist_dispatch_source``
    # ------------------------------------------------------------------
    def _validate_explore_provenance(
        self, payload: dict[str, Any],
    ) -> None:
        """PR-A9 (Arbor-into-Hyperloom): retire single-agent explore.

        Every explore-variant must trace to either:

        * a specialist's ``proposal_set`` entry
          (``provenance='specialist:<domain>'``); OR
        * the cold-start default grid
          (``provenance='default_grid'``).

        The legacy ``provenance='llm_direct'`` — where the Orchestration
        LLM authored the variant from a single prompt window without
        any specialist research — is denied. The denial hint instructs
        the LLM to dispatch a specialist first OR stamp the variant as
        ``default_grid`` for the cold-start path. This keeps every
        EXPLORE round specialist-first while still letting cold-start
        rounds proceed when no specialist proposal_set exists yet.

        Empty grid (or grid omitted) is NOT denied here — the executor
        surfaces a structured ``empty_grid`` failure instead.
        """
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            return
        grid = params.get("grid")
        if not isinstance(grid, list) or not grid:
            return
        permitted_count = 0
        denied_examples: list[str] = []
        for v in grid:
            if not isinstance(v, dict):
                continue
            prov = str(v.get("provenance") or "").strip()
            if (
                prov in EXPLORE_PERMISSIVE_PROVENANCE_LITERALS
                or any(
                    prov.startswith(p)
                    for p in EXPLORE_PERMISSIVE_PROVENANCE_PREFIXES
                )
            ):
                permitted_count += 1
            else:
                denied_examples.append(
                    f"{v.get('name', '?')}={prov or '<missing>'}"
                )
        if permitted_count == 0:
            raise PolicyDenied(
                "explore: every grid variant carries the legacy "
                "provenance='llm_direct' (or no provenance at all, "
                "which defaults to 'llm_direct'). PR-A9 retired the "
                "single-agent explore path; EXPLORE rounds must "
                "trace each variant to a specialist proposal_set "
                "or the cold-start default_grid.",
                rule="explore_requires_specialist_provenance",
                hint=(
                    "Either:\n"
                    "  1. delegate{action_name='specialist', "
                    "params={domain, gap_canonical_id, ...}} first, "
                    "wait for the specialist_done, then re-emit "
                    "explore with grid variants stamped "
                    "provenance='specialist:<domain>'; OR\n"
                    "  2. stamp every cold-start variant with "
                    "provenance='default_grid' (signals 'no specialist "
                    "yet — use the executor's built-in grid')."
                ),
            )

    # ------------------------------------------------------------------
    # ``explore_specialist_grid_max_one`` — cap on how many
    # ``provenance='specialist:*'`` variants Orchestration may stack
    # into one explore round. ``default_grid`` variants are unaffected
    # (cold-start path). Backstops the prompt instruction by hard-denying
    # at PolicyGate so an over-eager LLM cannot ship a 5-way grid even
    # if the orchestration prompt is later softened.
    # ------------------------------------------------------------------
    def _validate_explore_grid_size(
        self, payload: dict[str, Any],
    ) -> None:
        params = payload.get("params") or {}
        if not isinstance(params, dict):
            return
        grid = params.get("grid")
        if not isinstance(grid, list) or not grid:
            return
        specialist_sourced = sum(
            1 for v in grid
            if isinstance(v, dict)
            and str(v.get("provenance") or "").startswith("specialist:")
        )
        if specialist_sourced > MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS:
            raise PolicyDenied(
                f"explore: grid contains {specialist_sourced} "
                f"specialist-sourced variants; max "
                f"{MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS} per round.",
                rule="explore_specialist_grid_max_one",
                hint=(
                    "Across all specialist_done.proposal_set entries in "
                    "the inbox, select AT MOST one variant per explore "
                    "round to stamp provenance='specialist:<domain>'. "
                    "If multiple specialist proposals look attractive, "
                    "defer the runners-up to a subsequent explore round. "
                    "``default_grid`` variants are unaffected (cold-start "
                    "path)."
                ),
            )
        # Per-round cap on variants stamped with ``provenance='dynamic'``;
        # locked at the grid surface so loosening upstream caps cannot
        # silently break the IR-4 invariant.
        dynamic_sourced = sum(
            1 for v in grid
            if isinstance(v, dict)
            and str(v.get("provenance") or "").strip() == "dynamic"
        )
        if dynamic_sourced > MAX_DYNAMIC_SOURCED_VARIANTS:
            raise PolicyDenied(
                f"explore: grid contains {dynamic_sourced} "
                f"dynamic-sourced variants; max "
                f"{MAX_DYNAMIC_SOURCED_VARIANTS} per round.",
                rule="dynamic_sourced_variant_cap_exceeded",
                hint=(
                    f"At most {MAX_DYNAMIC_SOURCED_VARIANTS} variant "
                    f"with provenance='dynamic' per explore round; "
                    f"defer runners-up to a subsequent round."
                ),
            )

    # ------------------------------------------------------------------
    # ``sweep_phase_singleton``
    # ------------------------------------------------------------------
    def _validate_sweep_singleton(
        self, payload: dict[str, Any], *, intent_kind: str,
    ) -> None:
        """Enforce one sweep per SWEEP phase.

        ``Coordinator._on_enter_sweep`` (KB_design §3.2 §5.4 +
        KB_gaps/Gap-05) auto-enqueues a single internal sweep task on
        SWEEP entry, stamping
        ``state.phase_history[-1].evidence.auto_sweep_task_id`` with
        the resulting task id. The Coordinator's own enqueue bypasses
        PolicyGate (it calls TaskRegistry.create_or_return_existing
        directly), so this rule is dormant for the auto-path.

        For agent-emitted intents (``delegate{action_name='sweep'}``
        and ``propose_action{action_name='sweep'}``), this rule
        denies any sweep proposal once the auto-enqueue has committed
        within the active SWEEP phase. Concrete signal: latest
        phase_history row has ``to_phase='SWEEP'`` (cheaper to read
        than ``state.phase`` and immune to stale-phase reads after a
        crash) AND ``evidence.auto_sweep_task_id`` is non-empty.

        Why this is the right shape:

        * Two concurrent sweep tasks make every variant fail engine
          init: both ``vllm serve`` instances race for the same 8
          GPUs and the same TCP port. ``HSA_STATUS_ERROR_OUT_OF_
          RESOURCES`` for both, all sweep variants written as
          ``success=false``, the report's workload-curve section is
          empty.
        * The auto-enqueue already covers the SKILL.md default grid
          + the Cortex ``recipe.sweep_grid`` field, which together
          are the entirety of the documented sweep contract — there
          is no remaining workload the LLM could legitimately add.
        * The rule self-clears at SWEEP→CLOSE: phase_history[-1]
          becomes the new CLOSE row, ``evidence.auto_sweep_task_id``
          is no longer present, the gate goes back to inert.

        Operator escape hatch: ``params.bypass_sweep_singleton=True``
        is honoured so a debug session can intentionally run a
        second sweep with a custom grid (e.g. ``CONC=128``). The
        denial-then-bypass pattern keeps the override on the audit
        trail (Inv-9.4).
        """
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

    def _validate_integrate_patch_critic_gate(
        self, payload: dict[str, Any],
    ) -> None:
        """PR-A7: enforce ``integrate_patch_requires_critic_verdict``.

        Reject when:

        * ``params.specialist_task_id`` is missing (cannot identify the
          patch source).
        * SharedState has no recorded critic verdict for the task and
          ``params.bypass_critic`` is not truthy.
        * The recorded verdict is ``reject`` (Critic asked to abort).
        * The recorded verdict is ``needs_review`` / ``redirect`` and
          ``params.bypass_critic`` is not set (the operator must
          consciously override these terminal states).

        ``params.bypass_critic=True`` always wins so the operator can
        force-integrate (the policy_denial event will still surface
        the override on the next tick for audit).
        """
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
        # SharedState lookup — every PolicyGate instance carries a
        # reference; defensive None check covers tests that build
        # PolicyGate without a SharedState.
        ss = getattr(self, "shared_state", None)
        verdict = ""
        if ss is not None:
            try:
                verdict = ss.get_specialist_patch_verdict(sid)
            except AttributeError:
                # Older SharedState (no PR-A7 field). Treat as no
                # verdict on record.
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
        """Enforce the specialist-delegate contract.

        Single rule ``specialist_dispatch_source`` with several sub-rules
        surfaced in the ``hint`` so the LLM gets actionable feedback
        (Inv-11.2). Order matches §3.5 §11 / §3.13 M5 §4.

        - source role must be Orchestration (R2 main).
        - ``params.domain`` ∈ SPECIALIST_DOMAIN_KEYS.
        - ``params.gap_canonical_id`` non-empty.
        - ``params.max_turns`` (if set) ≤ SPECIALIST_MAX_TURNS_HARD_CAP.
        """
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
                hint="pass params={domain, gap_canonical_id, ...} per §3.5 §6",
            )
        domain = str(params.get("domain") or "").strip()
        if not domain:
            raise PolicyDenied(
                "delegate{action='specialist'}: params.domain is required",
                rule="specialist_dispatch_source",
                hint=(
                    f"set params.domain to one of "
                    f"{sorted(SPECIALIST_DOMAIN_KEYS)!r}"
                ),
            )
        if domain not in SPECIALIST_DOMAIN_KEYS:
            raise PolicyDenied(
                f"delegate{{action='specialist'}}: unknown domain={domain!r}",
                rule="specialist_dispatch_source",
                hint=(
                    f"params.domain must be one of "
                    f"{sorted(SPECIALIST_DOMAIN_KEYS)!r} "
                    f""
                ),
            )

        # Per-domain sub_kind validation. Default sub_kind (None / "")
        # is always allowed — the specialist runs the canonical per-
        # domain prompt. Non-empty sub_kind must appear in the
        # domain's ``sub_kinds`` tuple. (``FRAMEWORK_AGENT_GATED_SUB_KINDS``
        # / ``framework_pr_scout`` were removed when framework-agent
        # was promoted to the FRAMEWORK_PR phase.)
        sub_kind = str(params.get("sub_kind") or "").strip()
        if sub_kind:
            from .specialist_domains import get_domain
            domain_obj = get_domain(domain)
            allowed = tuple(domain_obj.sub_kinds) if domain_obj else ()
            if sub_kind not in allowed:
                raise PolicyDenied(
                    f"delegate{{action='specialist'}}: domain={domain!r} "
                    f"does not support sub_kind={sub_kind!r}",
                    rule="specialist_dispatch_source",
                    hint=(
                        f"params.sub_kind must be empty (= default "
                        f"per-domain prompt) or one of "
                        f"{sorted(allowed)!r} for domain={domain!r}."
                    ),
                )

        gap = str(params.get("gap_canonical_id") or params.get("gap") or "").strip()
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

    # ------------------------------------------------------------------
    # dynamic_action dispatch validation
    # ------------------------------------------------------------------
    def _validate_dynamic_action_dispatch(
        self, role: "AgentRole", payload: dict[str, Any],
    ) -> None:
        """Reject every dispatch that would cross a ``dynamic_action``
        red line.

        Four check groups:

        - **A** phase (EXPLORE only) + source role (orchestration only)
        - **B** payload schema completeness
        - **C** ``side_effects_declared`` red-line boundary
        - **D** round-cap accounting

        Each failure raises :class:`PolicyDenied` with a distinct
        ``rule=dynamic_*`` code. Group D + the IR-4 sourced cap depend
        on SharedState; the method falls open when ``shared_state`` is
        absent, keeping legacy unit-test paths stable.
        """
        state = self.shared_state
        phase = ""
        if state is not None:
            phase = str(getattr(state, "phase", "") or "").strip().upper()
        if phase and phase != PHASE_EXPLORE:
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}} only valid in "
                f"phase=EXPLORE; current phase={phase!r}",
                rule="dynamic_phase_violation",
                hint=(
                    "dynamic_action is an EXPLORE-only channel; wait "
                    "for the EXPLORE phase before dispatching."
                ),
            )
        if role.name not in DYNAMIC_ACTION_DISPATCH_SOURCE_ALLOWLIST:
            raise PolicyDenied(
                f"role={role.name!r} cannot dispatch dynamic_action "
                f"(allowed: "
                f"{sorted(DYNAMIC_ACTION_DISPATCH_SOURCE_ALLOWLIST)!r})",
                rule="dynamic_source_violation",
                hint=(
                    "Only the orchestration role may dispatch "
                    "dynamic_action; sub-agents must not recursively "
                    "spawn one."
                ),
            )

        params = payload.get("params") or {}
        if not isinstance(params, dict):
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: params must "
                f"be a dict",
                rule="dynamic_payload_schema",
                hint=(
                    "params must carry motivation_gap_text, "
                    "scope_domains, side_effects_declared (and optional "
                    "budget_hint)."
                ),
            )
        motivation = str(params.get("motivation_gap_text") or "").strip()
        if not motivation:
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: "
                f"params.motivation_gap_text is required and non-empty",
                rule="dynamic_payload_schema",
                hint=(
                    "Provide a free-form motivation_gap_text explaining "
                    "why a single specialist cannot cover this patch "
                    "combination (audit only; PolicyGate does not parse "
                    "the semantics)."
                ),
            )
        scope_domains_raw = params.get("scope_domains")
        if not isinstance(scope_domains_raw, (list, tuple)):
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: "
                f"params.scope_domains must be a list of specialist "
                f"domain keys",
                rule="dynamic_payload_schema",
            )
        # Dedup (order-preserving) so a repeated entry cannot inflate a
        # single-domain dispatch into a fake cross-domain one.
        scope_domains = list(dict.fromkeys(
            d for d in (str(d or "").strip() for d in scope_domains_raw) if d
        ))
        # All-kernel scope is a kernel-only patch in disguise — checked
        # before the min-length rule so it keeps its dedicated reason
        # code even after dedup collapses a repeated kernel literal.
        if scope_domains and all(
            d.lower() == DYNAMIC_ACTION_KERNEL_DOMAIN_LITERAL
            for d in scope_domains
        ):
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: every "
                f"scope_domains entry is "
                f"{DYNAMIC_ACTION_KERNEL_DOMAIN_LITERAL!r}; that is a "
                f"kernel-only patch in disguise",
                rule="dynamic_kernel_only_disallowed",
                hint=(
                    "Kernel-only patches must go through the kernel "
                    "agent (REQUEST{target_agent='kernel', ...}); "
                    "dynamic_action is for genuine cross-domain "
                    "synthesis."
                ),
            )
        if len(scope_domains) < DYNAMIC_ACTION_MIN_SCOPE_DOMAINS:
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: "
                f"scope_domains has {len(scope_domains)} distinct "
                f"entries; minimum is {DYNAMIC_ACTION_MIN_SCOPE_DOMAINS}",
                rule="dynamic_scope_too_narrow",
                hint=(
                    "dynamic_action is for cross-domain patches; "
                    "declare at least 2 distinct specialist domains. "
                    "For single-domain patches, dispatch a specialist."
                ),
            )
        unknown_domains = [
            d for d in scope_domains
            if d not in SPECIALIST_DOMAIN_KEYS
            and d != DYNAMIC_ACTION_KERNEL_DOMAIN_LITERAL
        ]
        if unknown_domains:
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: "
                f"scope_domains contains unregistered keys: "
                f"{unknown_domains!r}",
                rule="dynamic_scope_unknown_domain",
                hint=(
                    f"Every scope_domains entry must be one of "
                    f"{sorted(SPECIALIST_DOMAIN_KEYS)!r} (or the "
                    f"reserved literal "
                    f"{DYNAMIC_ACTION_KERNEL_DOMAIN_LITERAL!r})."
                ),
            )
        side_effects_raw = params.get("side_effects_declared")
        if not isinstance(side_effects_raw, (list, tuple)):
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: "
                f"params.side_effects_declared must be a list",
                rule="dynamic_payload_schema",
                hint=(
                    "Declare every action category the sub-agent "
                    "expects to touch (e.g. ['framework_source']); "
                    "verified against the red-line set below."
                ),
            )
        side_effects = [str(s or "").strip() for s in side_effects_raw]
        side_effects = [s for s in side_effects if s]
        if not side_effects:
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: "
                f"params.side_effects_declared cannot be empty",
                rule="dynamic_payload_schema",
                hint=(
                    "Even a noop-shaped dynamic_action must declare "
                    "its target side-effect category (e.g. "
                    "['framework_source'])."
                ),
            )
        budget_hint_raw = params.get("budget_hint")
        if budget_hint_raw is not None:
            budget_hint = str(budget_hint_raw or "").strip().lower()
            if budget_hint and budget_hint not in DYNAMIC_ACTION_BUDGET_HINTS:
                raise PolicyDenied(
                    f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: "
                    f"budget_hint={budget_hint!r} not in "
                    f"{sorted(DYNAMIC_ACTION_BUDGET_HINTS)!r}",
                    rule="dynamic_payload_schema",
                )

        offending_side_effects: list[str] = []
        for se in side_effects:
            se_norm = se.lower()
            if (
                se_norm in KERNEL_OWNED_ACTIONS
                or se_norm in DYNAMIC_ACTION_SIDE_EFFECT_RED_LINES
            ):
                offending_side_effects.append(se)
        if offending_side_effects:
            raise PolicyDenied(
                f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: "
                f"side_effects_declared crosses a red line: "
                f"{offending_side_effects!r}",
                rule="dynamic_side_effects_red_line",
                hint=(
                    "dynamic_action cannot declare kernel-owned "
                    "actions, metric / accuracy_gate ownership, or "
                    "independent server lifecycle."
                ),
            )
        # Round-cap accounting; only enforced when shared_state is
        # wired (legacy unit tests pass without it).
        if state is not None:
            cur = int(
                getattr(state, "dynamic_action_round_count", 0) or 0
            )
            if cur >= MAX_DYNAMIC_PER_ROUND:
                raise PolicyDenied(
                    f"delegate{{action='{DYNAMIC_ACTION_NAME}'}}: "
                    f"round cap exhausted "
                    f"({cur}/{MAX_DYNAMIC_PER_ROUND} dispatched in the "
                    f"current EXPLORE round)",
                    rule="dynamic_round_cap_exhausted",
                    hint=(
                        f"At most {MAX_DYNAMIC_PER_ROUND} dynamic_action "
                        f"per EXPLORE round; wait for the next round."
                    ),
                )
        # Registry-backed phase allowlist as a final defense.
        self._validate_phase_action(
            role, DYNAMIC_ACTION_NAME, intent_kind="delegate",
        )

    # ------------------------------------------------------------------
    # R3 ``specialist_done_source``
    # ------------------------------------------------------------------
    def _validate_specialist_intent(
        self, from_agent: str, intent: Intent,
    ) -> None:
        """Validate any intent emitted under a ``specialist:<task_id>`` identity.

        Specialists are tightly scoped (Inv-5.2): they may emit
        SEND_MESSAGE (heartbeat / advice), ALERT, and exactly one
        SPECIALIST_DONE. Anything else fires R3
        ``specialist_done_source`` (the rule label covers the whole
        specialist intent surface — sub-rule reported in hint).
        """
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
        """Per-field R3 checks for the ``specialist_done`` payload.

        Schema:

        * gap_canonical_id: str (matches dispatch task_id's gap)
        * domain: str ∈ SPECIALIST_DOMAIN_KEYS
        * proposal_set: list of variant dicts (may be empty when empty=true)
        * empty: bool (true → proposal_set must be []; reason required)
        * summary: str (≤ 500 chars per design; we cap at 4096 defensively)
        * confidence?: float ∈ [0, 1]
        * new_findings?: list
        * residual_questions?: list

        The dispatch-side gap/domain match (R3
        ``specialist_done_gap_mismatch`` / ``_domain_mismatch``) is
        delegated to the SharedState-aware caller path; PolicyGate
        here checks the structural shape so a malformed envelope
        never reaches the Coordinator dispatcher.
        """
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
        s = str(value)
        return any(s.startswith(p) for p in resolve_source_file_allowlist())

    def _path_in_trace_allowlist(self, value: str) -> bool:
        """Match a value against runtime-resolved trace path prefixes.

        Used only for trace-input-style fields in multi-node mode where
        the shared profile dir lives outside session_dir (on a cluster-
        shared filesystem anchored on ``$USER_DATA_PATH``; see
        :func:`_trace_path_allowlist`).
        """
        s = str(value)
        return any(s.startswith(p) for p in _trace_path_allowlist())

    def _validate_payload_paths(
        self, role: "AgentRole", intent_type: IntentType, payload: dict[str, Any],
    ) -> None:
        """Walk payload dict; reject path-like values escaping session_dir.

        Recursive: nested dicts (request.params, response.result, ...)
        are scanned. Lists of strings are also scanned. The check is
        a no-op when either ``self.session_dir`` is None OR
        ``self.strict_paths`` is False (P0 / legacy paths).
        """
        if self.session_dir is None or not self.strict_paths:
            return

        def visit(node: Any, path_keys: tuple[str, ...]) -> None:
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
                # Multi-node profile traces live on a shared-FS path
                # outside session_dir by design; allow only the specific
                # trace-input fields, only against the runtime-resolved
                # trace path allowlist (anchored on $USER_DATA_PATH).
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
        # Roofline-v2 C3: per-intent source allowlist takes precedence; the
        # generic ROBUSTNESS_ONLY_SOURCE_ALLOWLIST remains the default so
        # FORCE_DISPATCH / ESCALATE_STRATEGY_CHANGE stay robustness-only.
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
