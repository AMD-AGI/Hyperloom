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

from .intent_parser import Intent, IntentType
from .message_bus import TOPIC_ALLOWLIST

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
SOURCE_FILE_ALLOWLIST: tuple[str, ...] = (
    "/sgl-workspace/aiter/",
    "/sgl-workspace/sglang/",
    "/sgl-workspace/vllm/",
)

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

    def __post_init__(self) -> None:  # noqa: D401 — dataclass hook
        # Allow env to enable strict mode without threading a constructor
        # arg through every Coordinator caller (tests use a monkeypatched
        # env to opt in).
        import os as _os
        if not self.strict_paths and _os.environ.get(
            "INFERENCE_OPTIMIZER_STRICT_PATHS", ""
        ).strip() in ("1", "true", "yes"):
            self.strict_paths = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate_intent(self, from_agent: str, intent: Intent) -> None:
        """Raise :class:`PolicyDenied` if the intent is not allowed.

        Order of checks (cheapest first):

            1. Agent must be a known role
            2. ``intent.type`` must be in ``role.allowed_intents``
            3. Per-intent type structural rules
            4. Cross-source allowlists (review_verdict / kill_task /
               robustness-only)
        """
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
        # Plan A — kernel-owned actions are not directly delegatable.
        if action_name in KERNEL_OWNED_ACTIONS:
            raise PolicyDenied(
                f"action={action_name!r} is owned by the kernel agent; "
                f"emit REQUEST(target_agent='kernel', kind='...') instead "
                f"of delegate(action_name={action_name!r})",
                rule="kernel_owned_by_kernel_agent",
            )
        # If an ActionRegistry is wired, refuse delegate for unknown action names.
        # No registry → fall through (P0 / dev-mode where registry isn't loaded).
        if self.action_registry is not None and self.action_registry.get(action_name) is None:
            raise PolicyDenied(
                f"unknown action_name={action_name!r} (not in ActionRegistry)",
                rule="unknown_action",
                hint="register a yaml under inference_optimizer/actions/_meta/<name>.yaml",
            )

    def _validate_propose_action(self, role: "AgentRole", payload: dict[str, Any]) -> None:
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied("propose_action missing action_name", rule="payload")
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
        verdict = str(payload.get("verdict", "")).strip()
        if verdict not in REVIEW_VERDICTS:
            raise PolicyDenied(
                f"review_verdict.verdict={verdict!r} not in allowed set "
                f"{sorted(REVIEW_VERDICTS)!r}",
                rule="payload",
                hint="use one of approve/reject/redirect/advise/needs_review",
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
