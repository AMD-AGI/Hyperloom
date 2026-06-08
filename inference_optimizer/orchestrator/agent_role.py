# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Agent role definitions

Each :class:`AgentRole` binds:

    * a backend type (Claude tool-using vs. Codex no-tools)
    * a model slug + the env var holding the API key
    * the system prompt loaded from ``system_prompts/<name>.md``
    * which intent types the role is allowed to emit
    * permission flags consumed by :class:`PolicyGate`

v0.6 roster — 4 persistent reactors, no mode gating::

    ┌──────────────┬──────────┬─────────────────────────────────────────┐
    │ name         │ backend  │ allowed intents (high level)            │
    ├──────────────┼──────────┼─────────────────────────────────────────┤
    │ orchestration│ Claude   │ propose_action / delegate / request /   │
    │              │          │ update_state / send_message / ...       │
    │ kernel       │ Claude   │ response (only) / send_message / alert  │
    │ critic       │ Codex    │ review_verdict (only) / send_message /  │
    │              │ no-tools │ ask_question / answer / update_persona  │
    │              │ + KB Bash│ KB read/write Bash allowlist (§7.3)     │
    │ robustness   │ Claude   │ alert / kill_task / force_dispatch /    │
    │              │          │ prune_branch / escalate_strategy_change │
    │              │          │ + always-on tick                        │
    └──────────────┴──────────┴─────────────────────────────────────────┘

The roster is exactly these four roles. Earlier designs are gone:
the ``sage`` role merged into Critic, the ``triage`` role was renamed
to ``robustness`` (the active 4th role above), and parliament-era
``OBJECTION`` / ``VOTE`` intents were removed. There is no separate
framework role: framework PR work runs as the Coordinator-owned
FRAMEWORK_PR phase, not an agent role.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path

from ..paths import asset_system_prompts_dir
from ..protocol.intent import IntentType


class BackendType(str, Enum):
    """How a role talks to the LLM."""

    CLAUDE = "claude"   # tool-using (emit_intent + Read/Bash/Edit gated by Policy)
    CODEX = "codex"     # no-tools, validated_json_output only (KB Bash exception)


# --------------------------------------------------------------------------
# Built-in default model + API key env table
# --------------------------------------------------------------------------
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7"
DEFAULT_CODEX_MODEL = "gpt-5.4"  # litellm support pending for 5.5

DEFAULT_CLAUDE_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_CODEX_API_KEY_ENV = "OPENAI_API_KEY"


# --------------------------------------------------------------------------
# Role permission catalogue (DESIGN §7.6)
# --------------------------------------------------------------------------
_BASE_INTENTS: frozenset[IntentType] = frozenset({
    IntentType.SEND_MESSAGE,
    IntentType.ASK_QUESTION,
    IntentType.ANSWER,
    IntentType.ALERT,
    IntentType.UPDATE_PERSONA,
})


# Orchestration — proposes / delegates / requests Kernel; the only role with
# REQUEST authority (target_agent="kernel" enforced by PolicyGate).
#
# Roofline-v2 C3: Orchestration is also granted PRUNE_BRANCH so it can act on
# the structured ``suggested_prunes`` advice produced by the ``roofline``
# action (C4 / C5). The intent's per-source allowlist is widened in
# :mod:`.policy` (``_ROBUSTNESS_ONLY_INTENT_SOURCES``); the other two
# scheduling-police intents (FORCE_DISPATCH, ESCALATE_STRATEGY_CHANGE) stay
# robustness-only.
_ORCHESTRATION_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset({
    IntentType.PROPOSE_ACTION,
    IntentType.DELEGATE,
    IntentType.UPDATE_STATE,
    IntentType.REQUEST,
    IntentType.PRUNE_BRANCH,
})


# Kernel — responder-only; never initiates RPC, never proposes / delegates.
_KERNEL_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset({
    IntentType.RESPONSE,
    IntentType.UPDATE_STATE,  # only its own action's metric fields (§7.6 ※5)
})


# Critic — review verdicts only (no propose_action / delegate / request).
# Devil's advocate: send_message(topic="advice"), no parliament intents.
_CRITIC_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset({
    IntentType.REVIEW_VERDICT,
})


# Robustness — always-on health monitoring + RCA + recovery. Holds the entire
# scheduling-police intent set + KILL_TASK exclusively.
_ROBUSTNESS_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset({
    IntentType.UPDATE_STATE,  # crash_count / current_action only
    IntentType.DELEGATE,      # only handle actions: accuracy_gate / recover / server_lifecycle
    IntentType.KILL_TASK,
    IntentType.FORCE_DISPATCH,
    IntentType.PRUNE_BRANCH,
    IntentType.ESCALATE_STRATEGY_CHANGE,
})


# --------------------------------------------------------------------------
# AgentRole dataclass
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentRole:
    """Static role record. Backend instances are created elsewhere.

    Multiple roles can share the same backend instance in dry-runs.
    PolicyGate consumes the permission flags below.
    """

    name: str
    backend_type: BackendType
    model: str
    api_key_env: str
    allowed_intents: frozenset[IntentType]
    can_delegate_side_effects: bool = False
    can_mutate_core_state: bool = False
    no_tools: bool = False  # Codex roles
    system_prompt_filename: str = ""

    @property
    def system_prompt_path(self) -> Path:
        """Path to this role's system prompt markdown file.

        Uses ``system_prompt_filename`` when set, else ``<name>.md`` under
        the shared system-prompts asset directory.

        Returns:
            Path: Absolute path to the role's system prompt file.
        """
        return asset_system_prompts_dir() / (self.system_prompt_filename or f"{self.name}.md")

    def load_system_prompt(self) -> str:
        """Read and return this role's system prompt text.

        Returns:
            str: The UTF-8 decoded contents of :attr:`system_prompt_path`.
        """
        return self.system_prompt_path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Default registry
# --------------------------------------------------------------------------
def default_role_registry() -> dict[str, AgentRole]:
    """Return the canonical v0.6 4-agent registry (PascalCase capable).

    Builds fresh :class:`AgentRole` records for orchestration, kernel,
    critic, and robustness with their default backends, models, and
    permission flags.

    Returns:
        dict[str, AgentRole]: Mapping of role name to its static record.
    """
    return {
        "orchestration": AgentRole(
            name="orchestration",
            backend_type=BackendType.CLAUDE,
            model=DEFAULT_CLAUDE_MODEL,
            api_key_env=DEFAULT_CLAUDE_API_KEY_ENV,
            allowed_intents=_ORCHESTRATION_INTENTS,
            can_delegate_side_effects=True,
            can_mutate_core_state=False,
            no_tools=False,
        ),
        "kernel": AgentRole(
            name="kernel",
            backend_type=BackendType.CLAUDE,
            model=DEFAULT_CLAUDE_MODEL,
            api_key_env=DEFAULT_CLAUDE_API_KEY_ENV,
            allowed_intents=_KERNEL_INTENTS,
            can_delegate_side_effects=False,  # responder-only
            can_mutate_core_state=False,
            no_tools=False,
        ),
        "critic": AgentRole(
            name="critic",
            backend_type=BackendType.CODEX,
            model=DEFAULT_CODEX_MODEL,
            api_key_env=DEFAULT_CODEX_API_KEY_ENV,
            allowed_intents=_CRITIC_INTENTS,
            can_delegate_side_effects=False,
            can_mutate_core_state=False,
            no_tools=True,  # Codex no-tools + KB Bash exception (§7.3)
        ),
        "robustness": AgentRole(
            name="robustness",
            backend_type=BackendType.CLAUDE,
            model=DEFAULT_CLAUDE_MODEL,
            api_key_env=DEFAULT_CLAUDE_API_KEY_ENV,
            allowed_intents=_ROBUSTNESS_INTENTS,
            can_delegate_side_effects=True,  # only handle actions per Policy
            can_mutate_core_state=False,
            no_tools=False,
        ),
    }


@lru_cache(maxsize=1)
def roles_for_run() -> tuple[str, ...]:
    """Stable, deterministic ordering for reactor loop iteration.

    Cached so every caller observes the same tuple instance.

    Returns:
        tuple[str, ...]: Role names in fixed reactor-iteration order.
    """
    return ("orchestration", "kernel", "critic", "robustness")


__all__ = [
    "AgentRole",
    "BackendType",
    "DEFAULT_CLAUDE_API_KEY_ENV",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_CODEX_API_KEY_ENV",
    "DEFAULT_CODEX_MODEL",
    "default_role_registry",
    "roles_for_run",
]
