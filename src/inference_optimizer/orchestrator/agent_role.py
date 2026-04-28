"""Agent role definitions — DESIGN §5.1.

Each ``AgentRole`` binds:

    * a backend type (Claude tool-using vs. Codex no-tools)
    * a model slug + the env var holding the API key
    * the system prompt loaded from ``system_prompts/<name>.md``
    * which intent types the role is allowed to emit
    * whether the role may *delegate* side-effecting actions or *mutate* core
      shared-state fields (consumed by :class:`PolicyGate`)

The design (§5.1) specifies exactly four reactor-capable personas:

    ┌──────────┬──────────┬──────────────┬──────────┐
    │ name     │ backend  │ enabled in   │ tools?   │
    ├──────────┼──────────┼──────────────┼──────────┤
    │ executor │ Claude   │ all modes    │ yes      │
    │ critic   │ Codex    │ guided +     │ no-tools │
    │          │          │ marathon     │          │
    │ watchdog │ Claude   │ marathon     │ yes      │
    │ sage     │ Codex    │ marathon     │ no-tools │
    │          │          │ (resident)   │          │
    │ sage     │ Codex    │ quick+guided │ no-tools │
    │          │          │ (KB query)   │          │
    └──────────┴──────────┴──────────────┴──────────┘

The Conductor itself is *not* a role here — it is the host process that owns
PolicyGate and rejects/accepts intents on behalf of the policy.

References:

    - DESIGN §5.1 Agent roster & responsibilities
    - DESIGN §10.5.4 emit_intent tool schema (Claude side)
    - DESIGN §10.5.5 validated_json_output (Codex side)
    - IMPLEMENTATION-CHECKLIST.md Phase 2 §2.31 — §2.35
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from .execution_mode import ExecutionMode
from .intent_parser import IntentType


class BackendType(str, Enum):
    """How a role talks to the LLM."""

    CLAUDE = "claude"  # tool-using (emit_intent + Read/Bash/Edit gated by Policy)
    CODEX = "codex"    # no-tools, validated_json_output only


# --------------------------------------------------------------------------
# Built-in default model + API key env table
# --------------------------------------------------------------------------
DEFAULT_CLAUDE_MODEL = "claude-opus-4-7"
DEFAULT_CODEX_MODEL = "gpt-5.4"

DEFAULT_CLAUDE_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_CODEX_API_KEY_ENV = "OPENAI_API_KEY"


# --------------------------------------------------------------------------
# Role permission catalogue (DESIGN §5.1.1 + §10.5.7)
# --------------------------------------------------------------------------
# Codex roles never use tools and are not allowed to *trigger* workspace
# side-effects (DESIGN §5.1.1 — "需要工具或 workspace 副作用的动作统一走
# Claude-based Executor / Watchdog / ephemeral sub-agent"). Claude roles may
# delegate but Conductor still has the final say via PolicyGate.

_BASE_INTENTS: frozenset[IntentType] = frozenset(
    {
        IntentType.SEND_MESSAGE,
        IntentType.ASK_QUESTION,
        IntentType.ANSWER,
        IntentType.ALERT,
        IntentType.UPDATE_PERSONA,
    }
)

_EXECUTOR_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.PROPOSE_ACTION,
        IntentType.DELEGATE,
        IntentType.UPDATE_STATE,
    }
)

_WATCHDOG_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.PROPOSE_ACTION,
        # Watchdog can suggest a postmortem/strategic review via DELEGATE in
        # marathon mode (DESIGN §5.1 row "Watchdog"). It still passes through
        # PolicyGate which only allows action_names tagged for watchdogs.
        IntentType.DELEGATE,
        IntentType.UPDATE_STATE,
    }
)

_CRITIC_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.OBJECTION,
        IntentType.VOTE,
    }
)

_SAGE_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.OBJECTION,
        IntentType.VOTE,
        # DESIGN §5.1.2 marathon Sage: "可发 objection 触发议会" plus may
        # propose strategic review actions.
        IntentType.PROPOSE_ACTION,
    }
)


# --------------------------------------------------------------------------
# AgentRole dataclass
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentRole:
    """Static role record. Backend instances are created elsewhere.

    A role is *not* a backend — multiple roles can share the same backend
    instance in a dry-run. PolicyGate consumes the permission flags below.
    """

    name: str
    backend: BackendType
    model: str
    api_key_env: str
    description: str = ""
    no_tools: bool = False
    max_turns: int = 30

    # Permission triplet consumed by PolicyGate.
    allowed_intents: frozenset[IntentType] = field(default_factory=frozenset)
    can_delegate_side_effects: bool = False
    can_mutate_core_state: bool = False

    def system_prompt(self) -> str:
        """Return the system prompt text for this role.

        Looks up ``orchestrator/system_prompts/<name>.md`` first; falls back
        to a minimal generic prompt if the file is missing (so roles can be
        constructed in tests without a real prompt on disk).
        """
        return _read_role_prompt(self.name)


# --------------------------------------------------------------------------
# Factories (DESIGN §5.1; IMPL-CHECKLIST §2.33 / §2.34)
# --------------------------------------------------------------------------
def claude_role(
    name: str,
    *,
    model: str = DEFAULT_CLAUDE_MODEL,
    api_key_env: str = DEFAULT_CLAUDE_API_KEY_ENV,
    max_turns: int = 30,
    allowed_intents: Iterable[IntentType] | None = None,
    can_delegate_side_effects: bool = True,
    can_mutate_core_state: bool = False,
    description: str = "",
) -> AgentRole:
    """Factory for Claude-backed (tool-using) roles."""
    intents = (
        frozenset(allowed_intents)
        if allowed_intents is not None
        else _EXECUTOR_INTENTS
    )
    return AgentRole(
        name=name,
        backend=BackendType.CLAUDE,
        model=model,
        api_key_env=api_key_env,
        description=description,
        no_tools=False,
        max_turns=max_turns,
        allowed_intents=intents,
        can_delegate_side_effects=can_delegate_side_effects,
        can_mutate_core_state=can_mutate_core_state,
    )


def codex_role(
    name: str,
    *,
    model: str = DEFAULT_CODEX_MODEL,
    api_key_env: str = DEFAULT_CODEX_API_KEY_ENV,
    max_turns: int = 20,
    allowed_intents: Iterable[IntentType] | None = None,
    description: str = "",
) -> AgentRole:
    """Factory for Codex-backed (no-tools, validated_json_output) roles.

    Codex roles never delegate workspace side-effects and never mutate
    "core" shared-state fields (DESIGN §5.1.1 / §10.5.5).
    """
    intents = (
        frozenset(allowed_intents)
        if allowed_intents is not None
        else _CRITIC_INTENTS
    )
    return AgentRole(
        name=name,
        backend=BackendType.CODEX,
        model=model,
        api_key_env=api_key_env,
        description=description,
        no_tools=True,
        max_turns=max_turns,
        allowed_intents=intents,
        can_delegate_side_effects=False,
        can_mutate_core_state=False,
    )


# --------------------------------------------------------------------------
# Default registry (DESIGN §5.1)
# --------------------------------------------------------------------------
ROLE_EXECUTOR: AgentRole = claude_role(
    "executor",
    description="Proposes & delegates actions, interprets results, writes predictions.",
    allowed_intents=_EXECUTOR_INTENTS,
    can_delegate_side_effects=True,
    can_mutate_core_state=False,
)

ROLE_WATCHDOG: AgentRole = claude_role(
    "watchdog",
    description="event_log RCA, crash analysis, health monitoring (marathon-only).",
    allowed_intents=_WATCHDOG_INTENTS,
    # Watchdog may set crash_count / current_action via update_state but never
    # current_best / stop_reason — that remains Conductor-owned.
    can_delegate_side_effects=True,
    can_mutate_core_state=False,
)

ROLE_CRITIC: AgentRole = codex_role(
    "critic",
    description="Reviews proposals, independent predictions, post-mortem; ephemeral RCA in guided emergency.",
    allowed_intents=_CRITIC_INTENTS,
)

ROLE_SAGE: AgentRole = codex_role(
    "sage",
    description="KB curation, cross-run synthesis, devil's advocate (marathon resident).",
    allowed_intents=_SAGE_INTENTS,
)


def default_role_registry() -> dict[str, AgentRole]:
    """Return a fresh dict of the four built-in roles keyed by name."""
    return {
        ROLE_EXECUTOR.name: ROLE_EXECUTOR,
        ROLE_WATCHDOG.name: ROLE_WATCHDOG,
        ROLE_CRITIC.name: ROLE_CRITIC,
        ROLE_SAGE.name: ROLE_SAGE,
    }


# --------------------------------------------------------------------------
# Mode → roster (DESIGN §5.1 — "按 mode 启用")
# --------------------------------------------------------------------------
def roles_for_mode(
    mode: ExecutionMode,
    registry: dict[str, AgentRole] | None = None,
) -> list[AgentRole]:
    """Return the *resident reactor* roles for a given execution mode.

    The mapping follows DESIGN §5.1:

        quick      -> [executor]
        guided     -> [executor, critic]
        marathon   -> [executor, critic, watchdog, sage]

    Note: Sage in quick/guided is the "KB query service" form (DESIGN §5.1.2),
    not a reactor — it is *not* returned here.
    """
    reg = registry if registry is not None else default_role_registry()
    if mode == ExecutionMode.QUICK_PARAM_SWEEP:
        names = ("executor",)
    elif mode == ExecutionMode.GUIDED_KERNEL_OPT:
        names = ("executor", "critic")
    elif mode == ExecutionMode.MARATHON_MULTI_AGENT:
        names = ("executor", "critic", "watchdog", "sage")
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown execution mode: {mode!r}")
    return [reg[n] for n in names if n in reg]


# --------------------------------------------------------------------------
# System prompt loader
# --------------------------------------------------------------------------
def system_prompts_dir() -> Path:
    """Return absolute path to ``orchestrator/system_prompts/``."""
    return Path(__file__).resolve().parent / "system_prompts"


_DEFAULT_PROMPT_TEMPLATE = (
    "You are the {name} agent in the Hyperloom inference optimizer. "
    "Emit only structured intents according to the configured transport."
)


def load_system_prompt(name: str) -> str:
    """Read ``system_prompts/<name>.md`` and return its text.

    Falls back to a minimal generic prompt when the file is missing — useful
    for tests with custom roles that don't have an authored prompt yet.
    """
    return _read_role_prompt(name)


@lru_cache(maxsize=64)
def _read_role_prompt(name: str) -> str:
    path = system_prompts_dir() / f"{name}.md"
    if path.exists():
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            pass
    return _DEFAULT_PROMPT_TEMPLATE.format(name=name)


def _clear_prompt_cache() -> None:
    """Test helper — drop cached prompts so a new tmp dir takes effect."""
    _read_role_prompt.cache_clear()


__all__ = [
    "AgentRole",
    "BackendType",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_CLAUDE_API_KEY_ENV",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_API_KEY_ENV",
    "ROLE_CRITIC",
    "ROLE_EXECUTOR",
    "ROLE_SAGE",
    "ROLE_WATCHDOG",
    "claude_role",
    "codex_role",
    "default_role_registry",
    "load_system_prompt",
    "roles_for_mode",
    "system_prompts_dir",
]
