"""Agent role definitions — DESIGN §5.1 / standalone_agent_design §13 (v0.4 MVP).

Each ``AgentRole`` binds:

    * a backend type (Claude tool-using vs. Codex no-tools)
    * a model slug + the env var holding the API key
    * the system prompt loaded from ``system_prompts/<name>.md``
    * which intent types the role is allowed to emit
    * whether the role may *delegate* side-effecting actions or *mutate* core
      shared-state fields (consumed by :class:`PolicyGate`)

v0.4 MVP roster (4 Claude-backed reactors — see standalone_agent_design §13.1):

    ┌──────────┬──────────┬──────────────────────────────┬──────────────────┐
    │ name     │ backend  │ enabled in                   │ allowed intents  │
    ├──────────┼──────────┼──────────────────────────────┼──────────────────┤
    │ executor │ Claude   │ quick + guided + marathon    │ propose/delegate │
    │          │          │                              │ /update_state/   │
    │          │          │                              │ /request         │
    │ critic   │ Claude   │ guided + marathon            │ KEEP/REVERT obs  │
    │          │          │ (was Codex pre-v0.4)         │ via send_message │
    │ triage   │ Claude   │ all modes (always-on,        │ alert/update_st/ │
    │          │          │ tick=60s)                    │ /kill_task (only)│
    │ kernel   │ Claude   │ guided + marathon            │ response only    │
    │          │          │ (Plan A — kernel-opt)        │ (executor RPC)   │
    └──────────┴──────────┴──────────────────────────────┴──────────────────┘

Removed in v0.4 (vs v0.3 Plan A):
    * sage role (KB merged later; no KB in MVP)
    * watchdog role (renamed to triage with broader powers)
    * OBJECTION / VOTE intents (parliament removed entirely)

The Conductor itself is *not* a role here — it is the host process that owns
PolicyGate and rejects/accepts intents on behalf of the policy.

References:

    - DESIGN §5.1 Agent roster & responsibilities
    - standalone_agent_design §13 (v0.4 MVP)
    - DESIGN §10.5.4 emit_intent tool schema (Claude side)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from ..paths import asset_system_prompts_dir
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
# Role permission catalogue (DESIGN §5.1.1 + §10.5.7 + standalone §13.4)
# --------------------------------------------------------------------------
# v0.4 MVP — all 4 reactor roles use Claude. Codex factory is retained for
# v0.5+ but no role is bound to it today.

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
        # Executor is the only role that may emit REQUEST today (kernel agent
        # is the sole valid target). PolicyGate enforces both the source
        # role and the target_agent payload value (DESIGN §5 / standalone
        # _agent_design §3).
        IntentType.REQUEST,
    }
)

# Critic in v0.4 MVP has no OBJECTION/VOTE — parliament is removed entirely.
# Its only output is plain `send_message` carrying KEEP/REVERT verdicts after
# decision events arrive in its inbox (the standard `to_agent="*"` mirror
# already covers this — see standalone §13.9.4).
_CRITIC_INTENTS: frozenset[IntentType] = _BASE_INTENTS

# Triage in v0.4 MVP — always-on cross-layer health watcher. The only
# privileged intent it carries is KILL_TASK (PolicyGate enforces source
# allowlist). It never delegates / proposes / requests; its job is
# observation + alert + kill.
_TRIAGE_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.UPDATE_STATE,
        IntentType.KILL_TASK,
    }
)

# Kernel agent is a "responder" persona — it never initiates RPCs (no
# REQUEST), never delegates side-effecting actions, and never proposes the
# next inference-optimization action. Its sole new privilege is RESPONSE,
# which it emits in reply to executor REQUESTs. send_message + alert +
# update_persona + ask_question + answer are inherited from _BASE_INTENTS so
# the kernel agent can post observations / heartbeats during long GEAK runs.
_KERNEL_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.RESPONSE,
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

# Critic in v0.4 — Claude-backed (was Codex pre-v0.4). KEEP/REVERT review +
# Brier prediction via plain `send_message`. No OBJECTION / VOTE because
# parliament is removed in MVP.
ROLE_CRITIC: AgentRole = claude_role(
    "critic",
    description="Reviews decisions; emits KEEP/REVERT verdicts via send_message; never delegates.",
    allowed_intents=_CRITIC_INTENTS,
    can_delegate_side_effects=False,
    can_mutate_core_state=False,
)

# Triage in v0.4 — always-on Claude reactor (replaces watchdog + sage as
# cross-layer health watcher). Reads sibling agent outbox/inbox via the
# launcher's expanded --add-dir; emits kill_task to cancel stuck tasks.
ROLE_TRIAGE: AgentRole = claude_role(
    "triage",
    description=(
        "Always-on cross-layer health watcher. Scans event_log + sibling"
        " agent jsonl. The ONLY role allowed to emit kill_task."
    ),
    allowed_intents=_TRIAGE_INTENTS,
    # Triage never delegates side-effecting actions — its workspace footprint
    # is read-only Bash (tail/head/cat/ls/pgrep). kill_task is a separate
    # intent and is gated by KILL_TASK_SOURCE_ALLOWLIST in policy.py.
    can_delegate_side_effects=False,
    can_mutate_core_state=False,
)

# Kernel agent: persistent reactor that owns kernel-opt / integrate work.
# Backed by Claude (needs Bash/Read/Edit to drive geak_ray_submit.py /
# oob_ray_submit.py / patch_inductor.py). Intent set is intentionally narrow
# — it only RESPONSE-s to executor REQUESTs, never initiates work itself.
ROLE_KERNEL: AgentRole = claude_role(
    "kernel",
    description=(
        "Kernel-opt + integrate specialist. Responds to executor REQUEST"
        " RPCs (select_kernels / run_optimization / apply_patch); never"
        " delegates or proposes."
    ),
    allowed_intents=_KERNEL_INTENTS,
    can_delegate_side_effects=False,
    can_mutate_core_state=False,
)


def default_role_registry() -> dict[str, AgentRole]:
    """Return a fresh dict of the four built-in v0.4 roles keyed by name."""
    return {
        ROLE_EXECUTOR.name: ROLE_EXECUTOR,
        ROLE_CRITIC.name: ROLE_CRITIC,
        ROLE_TRIAGE.name: ROLE_TRIAGE,
        ROLE_KERNEL.name: ROLE_KERNEL,
    }


# --------------------------------------------------------------------------
# Mode → roster (DESIGN §5.1 — "按 mode 启用")
# --------------------------------------------------------------------------
def roles_for_mode(
    mode: ExecutionMode,
    registry: dict[str, AgentRole] | None = None,
) -> list[AgentRole]:
    """Return the *resident reactor* roles for a given execution mode.

    v0.4 MVP mapping (standalone_agent_design §13.2):

        quick      -> [executor, triage]
        guided     -> [executor, critic, kernel, triage]
        marathon   -> [executor, critic, kernel, triage]

    triage is always-on (active in every mode) — it is the only cross-mode
    reactor. critic + kernel activate from guided onwards. quick forbids
    kernel-opt actions so kernel agent stays absent there.

    guided / marathon roster is intentionally identical — the difference
    collapses to prompt length + checkpoint cadence, simplifying tests.
    """
    reg = registry if registry is not None else default_role_registry()
    if mode == ExecutionMode.QUICK_PARAM_SWEEP:
        names = ("executor", "triage")
    elif mode == ExecutionMode.GUIDED_KERNEL_OPT:
        names = ("executor", "critic", "kernel", "triage")
    elif mode == ExecutionMode.MARATHON_MULTI_AGENT:
        names = ("executor", "critic", "kernel", "triage")
    else:  # pragma: no cover - defensive
        raise ValueError(f"unknown execution mode: {mode!r}")
    return [reg[n] for n in names if n in reg]


# --------------------------------------------------------------------------
# System prompt loader
# --------------------------------------------------------------------------
def system_prompts_dir() -> Path:
    """Return absolute path to ``orchestrator/system_prompts/``."""
    return asset_system_prompts_dir()


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
    "ROLE_KERNEL",
    "ROLE_TRIAGE",
    "claude_role",
    "codex_role",
    "default_role_registry",
    "load_system_prompt",
    "roles_for_mode",
    "system_prompts_dir",
]
