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
                    f"{list(SOURCE_FILE_ALLOWLIST)!r}",
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
    "TRACE_PATH_LIKE_FIELDS",
    "SOURCE_LIKE_FIELDS",
]
