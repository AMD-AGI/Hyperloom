"""PolicyGate — DESIGN v0.6 §14.5.

Single chokepoint: every parsed Intent passes through ``validate_intent``
before the Coordinator commits side-effects. PolicyGate converges:

    * Role permission   — does this agent's role allow this intent type?
    * Source allowlist  — REVIEW_VERDICT is critic-only;
                          KILL_TASK / FORCE_DISPATCH / PRUNE_BRANCH /
                          ESCALATE_STRATEGY_CHANGE are robustness-only
    * REQUEST routing   — only orchestration→kernel is allowed in v0.6
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

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .framework_paths import resolve_source_file_allowlist
from .intent_parser import Intent, IntentType
from .message_bus import TOPIC_ALLOWLIST
from .phase_state import (
    PHASE_ALLOWED_ACTIONS,
    PHASE_EXPLORE,
    PHASE_NAMES,
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
})


# ---------------------------------------------------------------------------
# v0.8 M5 — specialist sub-agent action (KB_design §3.5 + §3.13 M5).
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

# Source roles allowed to dispatch a specialist via
# ``delegate{action='specialist'}``. KB_design §3.5 §11 / §3.11 §4.2.
SPECIALIST_DISPATCH_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"orchestration"})

# Prefix the SubAgentRunner stamps on every emit-intent originating
# from a specialist task. ``from_agent='specialist:<task_id>'``.
SPECIALIST_FROM_AGENT_PREFIX: str = "specialist:"


# ---------------------------------------------------------------------------
# v0.8 M3 / KB_gaps/Gap-10 — ``action_deprecated`` rule (KB_design §3.13 M3 §PR7)
#
# KB_design §3.4 / §3.15 §2.3: v0.8 merged ``backends`` / ``params`` /
# ``validate_stack`` into a single ``explore`` action (with the per-KEEP
# stack-rebench inlined). The legacy executors stay in the tree so v0.6
# resumes that still carry the corresponding ``*_attempts`` ledger
# fields don't crash on load (Inv-10.1 fact-layer survival), but new
# Orchestration intents that name one of the deprecated actions must
# be rejected at the PolicyGate boundary so the LLM gets a structured
# ``policy_denied{rule='action_deprecated'}`` event and a clear
# replacement hint.
#
# Why this lives in PolicyGate and not just in
# ``phase_state.PHASE_ALLOWED_ACTIONS``:
#
# * ``phase_incompatible`` (R1) denies an action for the wrong reason —
#   "not in phase=EXPLORE allowed set" hides the *real* cause (the
#   action is gone, not the phase). The LLM gets a misleading hint to
#   "wait for the phase to advance".
# * R1 fires *only when ``strict_phase=True``* (production). Without
#   ``action_deprecated`` an early-dev / test run with
#   ``strict_phase=False`` happily accepts ``propose_action='backends'``
#   and dispatches it; the resulting ``no_executor`` failure surfaces
#   far later in the dispatcher.
# * Inv-11.3 orthogonality wants exactly one rule per intent;
#   ``action_deprecated`` fires *before* R1 so deprecation always wins
#   over phase-allowed checks.
# ---------------------------------------------------------------------------
DEPRECATED_ACTION_NAMES: frozenset[str] = frozenset({
    "backends",
    "params",
    "validate_stack",
})

DEPRECATED_ACTION_REPLACEMENTS: dict[str, str] = {
    "backends": "explore",
    "params":   "explore",
    "validate_stack": "explore (per-KEEP stack rebench is inlined)",
}


# ---------------------------------------------------------------------------
# v0.8 §3.11 R4 / R5 — external tool whitelist registry
#
# Tool names live here (the *policy* layer) so PolicyGate AND the
# SpecialistRunner share a single source of truth. The runner builds
# its per-task tool list from the role-whitelist table below; PolicyGate
# uses these constants for the intent-level R4 + R5 second pass
# (KB_design §3.11 §5 "PolicyGate 仅作 *intent 层面* 的二次校验").
#
# Naming convention follows the Claude / Cursor tool surface.
# ---------------------------------------------------------------------------

#: KB *write* surfaces. R4 ``kb_write_unauthorized`` denies any
#: intent that tries to invoke one — directly or via an action_name /
#: request.kind collision.
KB_WRITE_TOOL_NAMES: frozenset[str] = frozenset({
    "mcp__cortex_kb__propose_point",
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


# ---------------------------------------------------------------------------
# SESSION_DIR path containment (DESIGN v0.6.1 §23 / §14.5).
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
SOURCE_FILE_ALLOWLIST: tuple[str, ...] = resolve_source_file_allowlist()

# Field name-only allowlist: when the payload key is `source_file`, the
# value may match SOURCE_FILE_ALLOWLIST instead of being session-rooted.
SOURCE_LIKE_FIELDS: frozenset[str] = frozenset({"source_file"})


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
    # v0.8 §3.10 §6.2 — the fact-layer KEEP ledger. Coordinator is the
    # sole writer (Inv-1 / Inv-10.2); LLM update_state can never
    # rewrite the stack, even though the LLM proposes the entries that
    # land in it via emit_intent → execute → promote flows.
    "optimization_stack",
    "gain_per_stack_entry",
    # v0.8 §3.10 §5.1 — schema_version is a migration breadcrumb; an
    # LLM update_state must not be able to roll the state.json back to
    # a v0.6 reader by setting ``schema_version=1``.
    "schema_version",
    # v0.8 M1 — Cortex KB integration fields (KB_design §3.6, §3.10,
    # §3.13 M1). Coordinator-only writes; LLM agents reading is fine.
    "cortex_session_id",
    "cortex_session_summary",
    "pending_kb_edges",
    "warm_start_recipe",
    "warm_start_pitfalls",
    "warm_start_ts",
    # v0.8 M2 — phase state machine fields (KB_design §3.2, §3.10,
    # §3.13 M2). All managed by ``Coordinator._advance_phase_if_needed``;
    # LLM update_state never reaches these.
    "phase",
    "phase_started_ts",
    "phase_started_unix",
    "phase_history",
    "phase_budget_pct",
    # v0.8 M5 — specialist sub-agent ledger (KB_design §3.5 §10 / §3.10
    # §4.1 / §3.11). Coordinator-only writes; LLM cannot inject
    # arbitrary entries via update_state (specialist_done carries
    # proposals through the dedicated R3 path instead).
    "specialist_rounds",
    "specialist_domain_empty_streak",
    "last_specialist",
    # v0.8 M5 — research_lane capacity is set once at CLI/manifest time
    # and mirrored into SharedState. Locking it as CORE prevents an LLM
    # from raising capacity mid-flight (KB_design §3.7 §4.4).
    "research_lane_capacity",
    # v0.8 M7 — phase-machine escalation plumbing (KB_design §3.8 §7.3 /
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
    # v0.8 §3.2 §5.5 / KB_gaps/Gap-06 — CLOSE phase sequencer flag.
    # Set by Coordinator at the end of the 5-step sequencer so
    # ``cli.finally`` can short-circuit its emergency breakdown
    # write. LLM update_state must not be able to toggle this and
    # trick the cli into skipping its safety net.
    "close_sequence_done",
    # v0.8 KB_gaps/Gap-14 — explore / backends / params search ledgers
    # (KB_design §3.10 §6.2). Coordinator's
    # ``apply_{explore,backends,params}_search_update`` /
    # ``record_explore_accepted`` / ``record_backends_accepted`` are the
    # sole writers; LLM ``update_state`` must not rewrite the ledger
    # directly (would bypass dedup-by-fingerprint + Inv-1).
    "explore_search",
    "backends_search",
    "params_search",
    # v0.8 KB_gaps/Gap-09 — structured gaps ledger (KB_design §3.3 /
    # §3.5 / §3.9 §6). Coordinator's ``_refresh_gaps`` is the sole
    # writer; LLM agents read via prompt injection. Locking the field
    # closes the proxy gap (last_action_failures + winners_history)
    # against an arbitrary update_state that would inject fake gaps
    # to bias specialist domain selection (Inv-1 / Inv-10.2).
    "gaps",
    # KB_design_continue §3.5 — monotonic experiment counter feeding
    # ``experiment_canonical_id(sid, iter)``. Coordinator's T2 hook is
    # the sole writer (via ``SharedState.increment_session_iter_index``);
    # LLM update_state must not rewrite the index or duplicate KB
    # ``exp:{sid}:{iter:04d}`` anchors would collide.
    "session_iter_index",
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
    # v0.8 M2 — phase-incompatible R1 enforcement mode.  When False
    # (default) the rule only emits a warning entry into the audit log;
    # production CLI flips this on via ``INFERENCE_OPTIMIZER_STRICT_PHASE=1``
    # to fail-closed.  Two-stage rollout matches the v0.6 →
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
        # v0.8 M5 — specialist sub-agents emit intents under an ephemeral
        # ``specialist:<task_id>`` identity (KB_design §3.5 §7). They get
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
        if intent.type == IntentType.PROPOSE_ACTION:
            if (intent.payload or {}).get("action_name") == "report":
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
        in v0.6; per-action Read/Bash/Edit injection happens in
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
        # v0.8 M3 / KB_gaps/Gap-10 — ``action_deprecated`` (KB_design
        # §3.13 M3 §PR7). Fires *before* the kernel-owned + phase
        # checks so the LLM gets the canonical "use ``explore``
        # instead" hint regardless of which other rule would have
        # fired. Inv-11.3: one deprecated action triggers exactly one
        # rule.
        self._validate_action_not_deprecated(action_name, intent_kind="delegate")
        # Plan A — kernel-owned actions are not directly delegatable.
        if action_name in KERNEL_OWNED_ACTIONS:
            raise PolicyDenied(
                f"action={action_name!r} is owned by the kernel agent; "
                f"emit REQUEST(target_agent='kernel', kind='...') instead "
                f"of delegate(action_name={action_name!r})",
                rule="kernel_owned_by_kernel_agent",
            )
        # v0.8 M5 — R2 ``specialist`` is a synthetic action that bypasses
        # ActionRegistry (KB_design §3.5 §10 "specialist 没有 yaml"). The
        # per-payload contract (domain / gap / max_turns) is enforced by
        # ``_validate_specialist_dispatch`` instead of the generic
        # registry path.
        if action_name == SPECIALIST_ACTION_NAME:
            self._validate_specialist_dispatch(role, payload)
            self._validate_phase_action(role, action_name, intent_kind="delegate")
            return
        # If an ActionRegistry is wired, refuse delegate for unknown action names.
        # No registry → fall through (P0 / dev-mode where registry isn't loaded).
        if self.action_registry is not None and self.action_registry.get(action_name) is None:
            raise PolicyDenied(
                f"unknown action_name={action_name!r} (not in ActionRegistry)",
                rule="unknown_action",
                hint="register a yaml under inference_optimizer/actions/_meta/<name>.yaml",
            )
        # v0.8 M2 — R1 phase_incompatible. Runs **after** the role +
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
        # v0.8 M3 / KB_gaps/Gap-10 — same ``action_deprecated`` gate
        # as ``_validate_delegate``. Catches advisory LLM proposals so
        # the policy_denial event surfaces in the prompt before the
        # delegate is even attempted.
        self._validate_action_not_deprecated(action_name, intent_kind="propose_action")
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
        # v0.8 M2 — R1 phase_incompatible (KB_design §3.11 §4.1).
        self._validate_phase_action(role, action_name, intent_kind="propose_action")
        # v0.8 §3.11 R4 / R5 — defense in depth on propose_action.
        self._validate_no_kb_write_collision(
            action_name, intent_kind="propose_action",
        )
        self._validate_tool_whitelist_collision(
            role.name, action_name, intent_kind="propose_action",
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
        # v0.8 M3 / KB_gaps/Gap-10 — ``action_deprecated`` covers the
        # REQUEST channel too (defense in depth). None of the v0.8
        # request kinds (select_kernels / kernel_opt / integrate /
        # ...) collide with the deprecated set today, but an operator
        # extension that re-uses one of the legacy names via
        # ``request.kind`` would still be caught here.
        self._validate_action_not_deprecated(kind, intent_kind="request")
        # v0.8 M2 — R1 phase_incompatible (KB_design §3.11 §4.1). For
        # orchestration → kernel REQUEST we treat the request *kind* as
        # the action name (kernel-owned actions named identically to
        # their REQUEST kind: kernel_opt / integrate / etc.).
        if target == "kernel" and kind in KERNEL_OWNED_ACTIONS:
            self._validate_phase_action(role, kind, intent_kind="request")
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
        # v0.8 KB_gaps/Gap-11 (KB_design §3.5 §5 / M5 §5 step 5) —
        # accept either the legacy single ``verdict`` field or the
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
    # v0.8 M3 / KB_gaps/Gap-10 — action_deprecated (KB_design §3.13 M3 §PR7)
    # ------------------------------------------------------------------
    def _validate_action_not_deprecated(
        self,
        action_name: str,
        *,
        intent_kind: str,
    ) -> None:
        """Reject a v0.6 action name that v0.8 has replaced.

        The deprecation is *hard* in v0.8 (KB_design §3.13 M3 §PR7):
        new sessions cannot use ``backends`` / ``params`` /
        ``validate_stack`` directly; all three flows merge into
        ``explore``. The denial carries a structured replacement hint
        so the LLM can self-correct in one tick.

        ``intent_kind`` is folded into the denial message so the
        policy_denial event in the prompt names the actual smuggling
        channel (delegate vs propose_action vs request).

        No-op when ``action_name`` isn't in
        :data:`DEPRECATED_ACTION_NAMES` — keeps the helper cheap on
        the hot path. Fires *before* phase / role gates so the
        deprecation hint always wins over phase_incompatible /
        unknown_action (Inv-11.3 orthogonality).
        """
        if not action_name:
            return
        if action_name not in DEPRECATED_ACTION_NAMES:
            return
        replacement = DEPRECATED_ACTION_REPLACEMENTS.get(
            action_name, "explore",
        )
        raise PolicyDenied(
            f"action {action_name!r} is deprecated since v0.8 (M3); "
            f"use {replacement!r} instead",
            rule="action_deprecated",
            hint=(
                f"intent_kind={intent_kind!r}: emit "
                f"delegate{{action_name='explore', params={{grid: [...]}}}} "
                f"or propose_action{{action_name='explore', ...}}. "
                f"KB_design §3.4 / §3.15 §2.3 — v0.8 merged "
                f"backends/params/validate_stack into the single "
                f"explore action with per-KEEP stack rebench inlined."
            ),
        )

    # ------------------------------------------------------------------
    # v0.8 M2 — R1 phase_incompatible (KB_design §3.11 §4.1)
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
    # v0.8 §3.11 R4 — kb_write_unauthorized (KB_design §3.11 §4.4)
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
                "Direct KB writes are not allowed (KB_design §3.11 R4). "
                "The Coordinator owns all KB writes. Express your "
                "intent via propose_action / delegate / "
                "specialist_done.proposal_set / review_verdict / "
                "kb_writes (critic-agent commit-review) instead."
            ),
        )

    # ------------------------------------------------------------------
    # v0.8 §3.11 R5 — tool_whitelist_role / tool_whitelist_phase
    # (KB_design §3.11 §4.5)
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
                f"specialist sub-agents (KB_design §3.11 §4.5). The "
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
    # v0.8 M5 — R2 ``specialist_dispatch_source`` (KB_design §3.11 §4.2)
    # ------------------------------------------------------------------
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
                    f"(KB_design §3.5 §5)"
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
    # v0.8 M5 — R3 ``specialist_done_source`` (KB_design §3.11 §4.3)
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

        Schema (KB_design §3.5 §7):

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
        return any(s.startswith(p) for p in SOURCE_FILE_ALLOWLIST)

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
                    f"{list(SOURCE_FILE_ALLOWLIST)!r}",
                    rule="source_file_not_allowlisted",
                    hint=("kernel-opt may only target framework source trees "
                          "under aiter/sglang/vllm; reject the request"),
                )
            if key not in PATH_LIKE_FIELDS:
                return
            if not self._path_under_session(node):
                raise PolicyDenied(
                    f"role={role.name!r} {intent_type.value} payload field "
                    f"{key!r}={node!r} escapes session_dir={self.session_dir!s}",
                    rule="path_outside_session_dir",
                    hint=("emit paths verbatim from SharedState (e.g. "
                          "last_profile_trace) or under SESSION_DIR"),
                )

        visit(payload, ())

    def _validate_robustness_only(
        self, role: "AgentRole", intent_type: IntentType, payload: dict[str, Any]
    ) -> None:
        if role.name not in ROBUSTNESS_ONLY_SOURCE_ALLOWLIST:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit {intent_type.value} "
                f"(allowed: {sorted(ROBUSTNESS_ONLY_SOURCE_ALLOWLIST)!r})",
                rule="robustness_only_source",
            )
        if intent_type == IntentType.PRUNE_BRANCH:
            family = str(payload.get("family", "")).strip()
            if not family:
                raise PolicyDenied("prune_branch missing family", rule="payload")


__all__ = [
    "CORE_STATE_FIELDS",
    "DEPRECATED_ACTION_NAMES",
    "DEPRECATED_ACTION_REPLACEMENTS",
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
    "SOURCE_FILE_ALLOWLIST",
    "SOURCE_LIKE_FIELDS",
]
