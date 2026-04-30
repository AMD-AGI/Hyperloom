"""PolicyGate — DESIGN §10.5.7 / §10.5.8.

Single chokepoint: every parsed Intent passes through ``validate_intent``
before the Conductor commits side-effects. PolicyGate is the *only* place
where the four constraints below converge:

    * Role permission   — does this agent's role allow this intent type?
    * Mode allow-list   — does the action declare ``mode`` as supported?
    * Side-effect spec  — does the action declare proper lanes / lease_ttl?
    * Quick-mode bash   — when an action is delegated in quick mode the
                          Bash command set must be on the allowlist and not
                          on the denylist (DESIGN §10.5.8).

PolicyGate stays *pure* — it does not touch the bus or the DB. The Conductor
catches :class:`PolicyDenied` and emits a ``policy_denied`` observation
event so the LLM can see why its intent was rejected on the next replay
turn.

References:
    - DESIGN §5.1.1   Tool Access principle
    - DESIGN §10.5.7  Intent gating pipeline
    - DESIGN §10.5.8  Quick-mode bash allow/deny lists
    - IMPLEMENTATION-CHECKLIST.md Phase 2 §2.15 — §2.30
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

from .execution_mode import ExecutionMode
from .intent_parser import Intent, IntentType
from .message_bus import TOPIC_ALLOWLIST

if TYPE_CHECKING:  # pragma: no cover - type-only
    from .agent_role import AgentRole
    from .feature_flags import FeatureFlags


# ---------------------------------------------------------------------------
class PolicyDenied(RuntimeError):
    """Intent rejected by PolicyGate.

    Attributes:
        rule:    short identifier of the rule that fired
                 (``role``, ``mode``, ``state_field``, ``bash``, ...) so
                 reactors can log structured reasons.
        hint:    optional one-line, agent-actionable suggestion describing
                 the canonical fix (e.g. "add a 'topic' key to the
                 send_message payload"). Surfaced in the
                 ``policy_denied`` observation so the next reactor turn
                 can self-correct without re-reading the failure
                 codebook.
    """

    def __init__(self, reason: str, *, rule: str | None = None,
                 hint: str | None = None):
        super().__init__(reason)
        self.rule = rule
        self.hint = hint


# ---------------------------------------------------------------------------
# §10.5.8 quick-mode Bash allow / deny lists
# ---------------------------------------------------------------------------
QUICK_BASH_ALLOWLIST: tuple[str, ...] = (
    # Server lifecycle (delegated via process_management actions only).
    "pgrep -f sglang.launch_server",
    "pgrep -f vllm.entrypoints.openai.api_server",
    "kill ",                  # only with pid arg
    "scripts/run_baseline.sh",
    # Read-only inspection
    "rocm-smi",
    "nvidia-smi",
    "ls ",
    "cat ",
    "head ",
    "tail ",
    # Sweep helpers
    "scripts/eval_accuracy.sh",
    "scripts/run_sweep.sh",
    "python -m sglang.bench_serving",
    "python -m vllm.entrypoints.benchmark",
)

QUICK_BASH_DENYLIST: tuple[str, ...] = (
    "pkill -f sglang",         # IR-5
    "pkill -f vllm",
    "git commit",
    "git push",
    "patch ",                  # IR-6: only via integrate action
    "patch_inductor.py",
    "geak",                    # marathon only
    "make ",                   # build systems out of executor scope
    "cmake ",
    "ninja",
    "rm -rf",                  # safety belt
    "sudo ",
)


# ---------------------------------------------------------------------------
# Quick-mode action allow-list (used when no ActionRegistry is available)
# ---------------------------------------------------------------------------
# In quick mode the Conductor refuses to delegate any action *not* on this
# list. When an ActionRegistry is wired (Phase 4) the registry's
# ``allowed_modes`` field takes precedence.
DEFAULT_QUICK_ACTION_ALLOWLIST: frozenset[str] = frozenset(
    {
        "baseline",                # MUST be first in every mode (executor.md)
        "server_lifecycle_restart",
        "param_sweep_run",
        "bench_runner",
        "profile",                 # cheap, soft-skip when no trace
        "diagnostic_probe",
        "kb_query",
        "report",                  # quick mode terminates with report
    }
)


# ---------------------------------------------------------------------------
# Plan A — kernel-owned actions
#
# These actions are owned end-to-end by the kernel agent. The executor
# (and any other role) MUST NOT delegate them directly; the only valid
# path is REQUEST(target_agent="kernel", kind="...") + RESPONSE.
# PolicyGate hard-rejects executor.delegate(kernel_opt|integrate) so the
# capability boundary is unambiguous.
# ---------------------------------------------------------------------------
KERNEL_OWNED_ACTIONS: frozenset[str] = frozenset({"kernel_opt", "integrate"})


# ---------------------------------------------------------------------------
# Request / Response routing matrix (DESIGN §5 / standalone_agent_design §3.2)
#
# Maps source role -> set of allowed target_agent names. A REQUEST whose
# (from_agent, target_agent) pair is not on this map is rejected. Today
# only executor->kernel is allowed; expanding here is how new agent-to-
# agent RPC paths are sanctioned.
# ---------------------------------------------------------------------------
REQUEST_ROUTING: dict[str, frozenset[str]] = {
    "executor": frozenset({"kernel"}),
}


# ---------------------------------------------------------------------------
# kill_task source allowlist — v0.4 MVP (standalone_agent_design §13.3)
#
# Only the triage role may emit kill_task intents. Any other role attempting
# kill_task is hard-rejected at PolicyGate. payload.scope is also restricted
# to "task" (process / server kills are NOT in MVP — IR-5 still owns server
# lifecycle; see KILL_TASK_ALLOWED_SCOPES).
# ---------------------------------------------------------------------------
KILL_TASK_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"triage"})
KILL_TASK_ALLOWED_SCOPES: frozenset[str] = frozenset({"task"})


# ---------------------------------------------------------------------------
# Phase G — triage-only "scheduling police" intents.
#
# These are emitted by triage to override executor stalls / dead-end loops:
#
#  * force_dispatch       — bump a queued task to the head of the queue
#  * prune_branch         — cancel queued tasks of a family + add it to
#                            ``state.pruned_families`` so the scheduler
#                            stops scoring it
#  * escalate_strategy_change — broadcast a high-priority strategy hint
#                                that executor's next tick must read
#
# PolicyGate enforces the source allowlist (only triage may emit any of
# these) on top of the per-role ``allowed_intents`` gate. Both layers
# agree; the dual check is intentional belt-and-braces because the
# scheduling-police intents bypass executor's normal authority.
# ---------------------------------------------------------------------------
TRIAGE_ONLY_INTENTS: frozenset[IntentType] = frozenset(
    {
        IntentType.FORCE_DISPATCH,
        IntentType.PRUNE_BRANCH,
        IntentType.ESCALATE_STRATEGY_CHANGE,
    }
)
TRIAGE_ONLY_SOURCE_ALLOWLIST: frozenset[str] = frozenset({"triage"})


# ---------------------------------------------------------------------------
# Core SharedState fields that only the Conductor may mutate
# ---------------------------------------------------------------------------
CORE_STATE_FIELDS: frozenset[str] = frozenset(
    {
        "current_best",          # KEEP/REVERT decision target
        "stop_reason",           # graceful stop owner
        "cumulative_gain",
        "baseline_tput",
        "baseline_accuracy",
        "session_id",
        "model_path",
        "model_name",
        "model_class",
        "start_ts",
        "max_minutes",
        "execution_mode",
    }
)


# ---------------------------------------------------------------------------
@dataclass
class PolicyGate:
    """Validate every intent emitted by an agent reactor.

    Attributes:
        flags:           current FeatureFlags
        mode:            current ExecutionMode
        role_registry:   name -> AgentRole lookup
        action_registry: name -> ActionMetadata lookup; may be ``None``
                         in v0.6 — falls back to ``DEFAULT_QUICK_ACTION_ALLOWLIST``
                         in quick mode and accepts everything in guided/marathon.
        quick_action_allowlist:
                         override for ``DEFAULT_QUICK_ACTION_ALLOWLIST`` (tests).
    """

    flags: "FeatureFlags"
    mode: ExecutionMode
    role_registry: dict[str, "AgentRole"]
    action_registry: Any | None = None
    quick_action_allowlist: frozenset[str] = field(
        default_factory=lambda: DEFAULT_QUICK_ACTION_ALLOWLIST
    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def validate_intent(
        self, from_agent: str, intent: Intent, state: Any | None = None
    ) -> None:
        """Raise :class:`PolicyDenied` if the intent is not allowed.

        Order of checks (cheapest first):

            1. Agent must be a known role
            2. ``intent.type`` must be in ``role.allowed_intents``
            3. Per-intent type structural rules (mode/state/payload)

        Side-effect free.
        """
        role = self.role_registry.get(from_agent)
        if role is None:
            raise PolicyDenied(
                f"unknown agent {from_agent!r}", rule="role",
            )

        if intent.type not in role.allowed_intents:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit intent_type={intent.type.value!r}",
                rule="role",
            )

        payload = intent.payload or {}

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
            self._validate_response(role, payload)
        elif intent.type == IntentType.KILL_TASK:
            self._validate_kill_task(role, payload)
        elif intent.type in TRIAGE_ONLY_INTENTS:
            self._validate_triage_only(role, intent.type, payload)
        # ANSWER / ASK_QUESTION / UPDATE_PERSONA / ALERT carry no
        # policy-relevant side-effects beyond the role gate.

    # ------------------------------------------------------------------
    def allowed_tools_for_agent(
        self, agent_name: str, mode: ExecutionMode | None = None
    ) -> list[str]:
        """Return the Claude tool list a reactor may use.

        For Codex roles the result is always ``[]`` (no-tools by spec). For
        Claude roles we return the canonical ``["emit_intent"]`` set in v0.6;
        Phase 4 adds Read/Bash/Edit gated by per-action policy.
        """
        role = self.role_registry.get(agent_name)
        if role is None:
            return []
        if role.no_tools:
            return []
        return ["emit_intent"]

    def allowed_tools_for_action(
        self, mode: ExecutionMode, action_name: str
    ) -> list[str]:
        """Per-action tool intersection used by SubAgentRunner (Phase 4)."""
        if self.action_registry is None:
            # Without an ActionRegistry we can't compute per-action tools.
            # Conservative default — only emit_intent.
            return ["emit_intent"]
        action = self.action_registry.get(action_name)
        if action is None:
            return ["emit_intent"]
        # Each ActionMetadata declares its own ``allowed_tools`` list.
        return list(getattr(action, "allowed_tools", ["emit_intent"]))

    # ------------------------------------------------------------------
    # Per-intent validators
    # ------------------------------------------------------------------
    def _validate_delegate(
        self, role: "AgentRole", payload: dict[str, Any]
    ) -> None:
        """``delegate`` requires:

            * role.can_delegate_side_effects = True
            * mode allows sub-agent delegation (FeatureFlags)
            * action_name in mode allowlist (or ActionRegistry approval)
            * action_name NOT in :data:`KERNEL_OWNED_ACTIONS` (Plan A —
              kernel agent owns kernel_opt + integrate end-to-end; the
              executor must use REQUEST(target=kernel) instead)
        """
        if not role.can_delegate_side_effects:
            raise PolicyDenied(
                f"role={role.name!r} cannot delegate side-effecting actions",
                rule="role",
            )
        if not self.flags.enable_subagent_delegate:
            raise PolicyDenied(
                f"sub-agent delegation disabled in mode={self.mode.value!r}",
                rule="mode",
            )
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied(
                "delegate intent missing action_name", rule="payload",
            )
        # Plan A — kernel-owned actions are not directly delegatable by any
        # role. Executor must REQUEST the kernel agent instead.
        if action_name in KERNEL_OWNED_ACTIONS:
            raise PolicyDenied(
                f"action={action_name!r} is owned by the kernel agent; "
                f"emit REQUEST(target_agent='kernel', kind='...') instead "
                f"of delegate(action_name={action_name!r})",
                rule="kernel_owned_by_kernel_agent",
            )
        self._validate_mode_allowed(action_name)

    # ------------------------------------------------------------------
    def _validate_request(
        self, role: "AgentRole", payload: dict[str, Any]
    ) -> None:
        """``request`` (agent-to-agent RPC) requires:

            * source role's name is on :data:`REQUEST_ROUTING` (i.e. has at
              least one valid target it can request).
            * ``payload['target_agent']`` is in the source role's allowed
              targets set.
            * ``payload['kind']`` is a non-empty string.

        Kind-level validation (e.g. "executor can only request
        select_kernels / run_optimization / apply_patch from kernel") is
        intentionally left to the kernel agent's own LLM-side discipline
        for v1 simplicity — wire it here later when the kind catalogue
        stabilises.
        """
        targets = REQUEST_ROUTING.get(role.name)
        if not targets:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit REQUEST",
                rule="role",
            )
        target = str(payload.get("target_agent", "")).strip()
        if not target:
            raise PolicyDenied(
                "request missing target_agent", rule="payload",
            )
        if target not in targets:
            raise PolicyDenied(
                f"role={role.name!r} cannot request target_agent={target!r} "
                f"(allowed: {sorted(targets)!r})",
                rule="request_target",
            )
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            raise PolicyDenied(
                "request missing kind", rule="payload",
            )

    def _validate_response(
        self, role: "AgentRole", payload: dict[str, Any]
    ) -> None:
        """``response`` validation.

        The source role must have RESPONSE on its allowed_intents (already
        gated by the role check above). We additionally require:

            * ``payload['in_reply_to']`` is a non-empty string (so the
              Conductor can reverse-route to the original requester).
            * ``payload['kind']`` is a non-empty string.
        """
        in_reply_to = str(payload.get("in_reply_to", "")).strip()
        if not in_reply_to:
            raise PolicyDenied(
                "response missing in_reply_to", rule="payload",
            )
        kind = str(payload.get("kind", "")).strip()
        if not kind:
            raise PolicyDenied(
                "response missing kind", rule="payload",
            )

    def _validate_kill_task(
        self, role: "AgentRole", payload: dict[str, Any]
    ) -> None:
        """``kill_task`` validation — v0.4 MVP (standalone_agent_design §13.3).

        Constraints:
            * source role must be in :data:`KILL_TASK_SOURCE_ALLOWLIST`
              (only ``triage`` today)
            * ``payload['task_id']`` is a non-empty string
            * ``payload['reason']`` is a non-empty string
            * ``payload['scope']`` defaults to ``"task"`` and MUST be in
              :data:`KILL_TASK_ALLOWED_SCOPES`. Explicit ``"process"`` /
              ``"server"`` are rejected — IR-5 still owns server lifecycle
              and OS-level pid kills are not in MVP scope.

        ``force`` (optional bool) is metadata only in MVP — it does not
        change behaviour, only gets recorded into ``findings/kills.jsonl``.
        """
        if role.name not in KILL_TASK_SOURCE_ALLOWLIST:
            raise PolicyDenied(
                f"role={role.name!r} cannot emit kill_task "
                f"(allowed: {sorted(KILL_TASK_SOURCE_ALLOWLIST)!r})",
                rule="kill_task_source",
            )
        task_id = str(payload.get("task_id", "")).strip()
        if not task_id:
            raise PolicyDenied(
                "kill_task missing task_id", rule="payload",
            )
        reason = str(payload.get("reason", "")).strip()
        if not reason:
            raise PolicyDenied(
                "kill_task missing reason", rule="payload",
            )
        scope = str(payload.get("scope") or "task").strip()
        if scope not in KILL_TASK_ALLOWED_SCOPES:
            raise PolicyDenied(
                f"kill_task scope={scope!r} not allowed "
                f"(allowed: {sorted(KILL_TASK_ALLOWED_SCOPES)!r}; "
                f"v0.4 MVP keeps server/process kills out per IR-5)",
                rule="kill_scope",
            )

    def _validate_propose_action(
        self, role: "AgentRole", payload: dict[str, Any]
    ) -> None:
        """``propose_action`` is advisory; we only require a real action_name
        and that the action (when known) lists the current mode."""
        action_name = str(payload.get("action_name", "")).strip()
        if not action_name:
            raise PolicyDenied(
                "propose_action missing action_name", rule="payload",
            )
        # Soft-gate: a proposal for an action that explicitly disallows the
        # current mode is rejected before it even gets stored as a task.
        if self.action_registry is not None:
            action = self.action_registry.get(action_name)
            if action is not None:
                modes = getattr(action, "allowed_modes", None)
                if modes is not None and self.mode not in modes:
                    raise PolicyDenied(
                        f"action={action_name!r} not allowed in mode={self.mode.value!r}",
                        rule="mode",
                    )

    def _validate_state_transition(
        self, role: "AgentRole", payload: dict[str, Any]
    ) -> None:
        """No non-Conductor agent may set fields in :data:`CORE_STATE_FIELDS`
        unless ``role.can_mutate_core_state`` is True."""
        changes = payload.get("changes")
        if not isinstance(changes, dict) or not changes:
            raise PolicyDenied(
                "update_state.payload.changes must be a non-empty dict",
                rule="payload",
                hint=(
                    "include at least one allowed field, e.g. "
                    "{'changes': {'current_action': '<action_name>'}}"
                ),
            )
        if role.can_mutate_core_state:
            return
        violating = sorted(set(changes.keys()) & CORE_STATE_FIELDS)
        if violating:
            raise PolicyDenied(
                f"role={role.name!r} cannot mutate core state fields: {violating!r}",
                rule="state_field",
                hint=(
                    "core fields are conductor-derived; remove "
                    f"{violating!r} from changes and only set role-allowed "
                    "fields like current_action / current_tput / crash_count"
                ),
            )

    def _validate_send_message_topic(self, payload: dict[str, Any]) -> None:
        """``send_message.topic`` validation.

        DESIGN refinement (post-v0.8): Critic / Sage / Watchdog naturally
        invent topic names like ``rca_finding`` / ``kb_status`` /
        ``executor_status`` / ``critic_review_post_baseline`` that aren't
        in :data:`TOPIC_ALLOWLIST`. Hard-denying them flooded the run with
        ``policy_denied`` observations and cost ~80% of LLM tokens for no
        signal.

        Policy now:

            * empty topic           -> deny (still a real bug)
            * topic in allowlist    -> pass
            * topic not in allowlist
                -> mutate ``payload['topic']='observation'`` and stash the
                   original under ``payload['original_topic']`` so the bus
                   receives a valid record AND we don't lose the LLM's
                   intent. The conductor's ``_handle_send_message`` already
                   does this downgrade for non-agent send paths; we extend
                   it to the agent path here.

        This keeps the bus schema strict (only allow-listed topics live in
        the events table) while letting agents talk freely. Unknown topics
        become ``observation`` with the original name preserved for
        debuggability.
        """
        topic = str(payload.get("topic", "")).strip()
        if not topic:
            raise PolicyDenied(
                "send_message missing topic", rule="payload",
                hint=(
                    "add a 'topic' key, e.g. 'heartbeat' / 'observation' / "
                    "'alert' / 'request' / 'response' / 'decision' / 'event'"
                ),
            )
        if topic in TOPIC_ALLOWLIST:
            return
        # Soft downgrade: stash original, normalize to ``observation``.
        payload["original_topic"] = topic
        payload["topic"] = "observation"

    # ------------------------------------------------------------------
    # Per-aspect helpers (re-used by SubAgentRunner)
    # ------------------------------------------------------------------
    def _validate_mode_allowed(self, action_name: str) -> None:
        """Reject when the action does not declare current mode as supported.

        Two paths:
            * ActionRegistry present  — check ``action.allowed_modes``.
            * Quick mode w/o registry — fall back to ``quick_action_allowlist``.
        """
        if self.action_registry is not None:
            action = self.action_registry.get(action_name)
            if action is None:
                # Surface up to 8 candidate names to help the agent recover
                # from typos / hallucinated action ids.
                try:
                    candidates = sorted(self.action_registry.names())[:8]
                except Exception:  # noqa: BLE001
                    candidates = []
                hint = (
                    f"action {action_name!r} not in catalogue; "
                    f"valid examples: {candidates!r}"
                ) if candidates else None
                raise PolicyDenied(
                    f"unknown action {action_name!r}", rule="mode",
                    hint=hint,
                )
            modes = getattr(action, "allowed_modes", None)
            if modes is not None and self.mode not in modes:
                try:
                    allowed_here = sorted(
                        a.name for a in self.action_registry.allowed_for_mode(self.mode)
                    )
                except Exception:  # noqa: BLE001
                    allowed_here = []
                raise PolicyDenied(
                    f"action={action_name!r} not allowed in mode={self.mode.value!r}",
                    rule="mode",
                    hint=(
                        f"actions allowed in {self.mode.value}: {allowed_here!r}"
                        if allowed_here else None
                    ),
                )
            return
        # No registry — fall back to mode-specific allow-list.
        if self.mode is ExecutionMode.QUICK_PARAM_SWEEP:
            if action_name not in self.quick_action_allowlist:
                raise PolicyDenied(
                    f"action={action_name!r} not in quick allowlist",
                    rule="mode",
                    hint=(
                        "quick mode action allowlist: "
                        f"{sorted(self.quick_action_allowlist)!r}"
                    ),
                )
        # In guided/marathon, w/o a registry, accept (Phase 4 tightens this).

    def _validate_triage_only(
        self, role: "AgentRole", intent_type: IntentType,
        payload: dict[str, Any],
    ) -> None:
        """Source-allowlist + payload-shape gate for the Phase G
        scheduling-police intents (FORCE_DISPATCH / PRUNE_BRANCH /
        ESCALATE_STRATEGY_CHANGE). The role gate already filters by
        ``allowed_intents`` (only triage carries them), but we belt-
        and-brace the source check so a future role mis-assignment
        cannot quietly grant scheduler-override rights."""
        if role.name not in TRIAGE_ONLY_SOURCE_ALLOWLIST:
            raise PolicyDenied(
                f"intent_type={intent_type.value!r} restricted to "
                f"{sorted(TRIAGE_ONLY_SOURCE_ALLOWLIST)!r}; "
                f"role={role.name!r} cannot emit it",
                rule="triage_only",
                hint=(
                    "scheduling-police intents (force_dispatch, "
                    "prune_branch, escalate_strategy_change) are reserved "
                    "for the triage agent. Use alert / send_message instead."
                ),
            )
        if intent_type is IntentType.FORCE_DISPATCH:
            if not str(payload.get("task_id", "")).strip():
                raise PolicyDenied(
                    "force_dispatch.payload.task_id must be a non-empty string",
                    rule="payload",
                    hint=(
                        "include the queued task's task_id; check "
                        "$SESSION_DIR/storage/conductor.db tasks table"
                    ),
                )
        elif intent_type is IntentType.PRUNE_BRANCH:
            if not str(payload.get("family", "")).strip():
                raise PolicyDenied(
                    "prune_branch.payload.family must be a non-empty string",
                    rule="payload",
                    hint=(
                        "family is the action.family value, e.g. 'long' / "
                        "'deep_kernel' / 'shallow' / 'creative'"
                    ),
                )
        elif intent_type is IntentType.ESCALATE_STRATEGY_CHANGE:
            if not str(payload.get("next_action_hint", "")).strip():
                raise PolicyDenied(
                    "escalate_strategy_change.payload.next_action_hint "
                    "must be a non-empty string",
                    rule="payload",
                    hint=(
                        "describe the suggested next action / family in "
                        "1–2 sentences so executor can act on it next tick"
                    ),
                )

    def validate_quick_bash(self, command: str) -> None:
        """Public helper for SubAgentRunner — validate a single Bash command
        string against the quick-mode allow / deny lists.

        Out-of-quick-mode this is a no-op; callers that want to enforce
        denylist always should call ``check_bash_denylist`` directly.
        """
        if self.mode is not ExecutionMode.QUICK_PARAM_SWEEP:
            return
        self.check_bash_denylist(command)
        if not _matches_any(command, QUICK_BASH_ALLOWLIST):
            raise PolicyDenied(
                f"bash command not in quick allowlist: {command!r}",
                rule="bash",
            )

    def check_bash_denylist(self, command: str) -> None:
        """Raise if a command matches a denylist entry (any mode)."""
        if _matches_any(command, QUICK_BASH_DENYLIST):
            raise PolicyDenied(
                f"bash command matches denylist: {command!r}",
                rule="bash",
            )


# ---------------------------------------------------------------------------
def _matches_any(command: str, patterns: Iterable[str]) -> bool:
    """Tiny matcher: the command starts with or contains the pattern.

    The patterns are deliberately conservative; we use ``in`` for substrings
    that contain a trailing space (e.g. ``"kill "``) and prefix matching for
    full-program names. This is not a shell parser — anything sneaky should
    already have been blocked at the action allow-list stage.
    """
    for pat in patterns:
        if pat.endswith(" "):
            if pat in command:
                return True
        elif command.startswith(pat) or f" {pat}" in command:
            return True
    return False


__all__ = [
    "CORE_STATE_FIELDS",
    "DEFAULT_QUICK_ACTION_ALLOWLIST",
    "KERNEL_OWNED_ACTIONS",
    "KILL_TASK_ALLOWED_SCOPES",
    "KILL_TASK_SOURCE_ALLOWLIST",
    "PolicyDenied",
    "PolicyGate",
    "QUICK_BASH_ALLOWLIST",
    "REQUEST_ROUTING",
    "QUICK_BASH_DENYLIST",
]
