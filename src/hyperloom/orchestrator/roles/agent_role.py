# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent role definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir
from hyperloom.inference_optimizer.protocol.intent import IntentType


class BackendType(str, Enum):
    """How a role talks to the LLM."""

    CLAUDE = "claude"  # tool-using (emit_intent + Read/Bash/Edit gated by Policy)
    CODEX = "codex"  # no-tools, validated_json_output only


DEFAULT_CLAUDE_MODEL = "claude-opus-5"
DEFAULT_CODEX_MODEL = "gpt-5.6-sol"

DEFAULT_CLAUDE_API_KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_CODEX_API_KEY_ENV = "OPENAI_API_KEY"


_BASE_INTENTS: frozenset[IntentType] = frozenset(
    {
        IntentType.SEND_MESSAGE,
        IntentType.ALERT,
    }
)


# Orchestration — only role with REQUEST authority.
_ORCHESTRATION_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.PROPOSE_ACTION,
        IntentType.DELEGATE,
        IntentType.UPDATE_STATE,
        IntentType.REQUEST,
        IntentType.EXTEND_LEASE,
        IntentType.PRUNE_BRANCH,
        IntentType.ESCALATE_STRATEGY_CHANGE,
    }
)


# Critic — review verdicts only.
_CRITIC_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.REVIEW_VERDICT,
    }
)


# Robustness — health monitoring + RCA + recovery.
_ROBUSTNESS_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.UPDATE_STATE,  # crash_count / current_action only
        IntentType.DELEGATE,  # only recover; enforced by PolicyGate ROBUSTNESS_DELEGATE_ONLY_ACTIONS
        IntentType.PRUNE_BRANCH,
        IntentType.ESCALATE_STRATEGY_CHANGE,
    }
)


# Specialist — single exit signal, optional heartbeats and alerts only.
SPECIALIST_INTENTS: frozenset[IntentType] = _BASE_INTENTS | frozenset(
    {
        IntentType.SPECIALIST_DONE,
    }
)


@dataclass(frozen=True)
class AgentRole:
    """Static role record. Backend instances are created elsewhere."""

    name: str
    backend_type: BackendType
    model: str
    api_key_env: str
    allowed_intents: frozenset[IntentType]
    can_delegate_side_effects: bool = False
    can_mutate_core_state: bool = False
    no_tools: bool = False  # Codex roles
    system_prompt_filename: str = ""
    prompt_driven: bool = True  # False = deterministic role; no system prompt is loaded

    @property
    def system_prompt_path(self) -> Path:
        """Path to this role's system prompt markdown file."""
        return asset_system_prompts_dir() / (self.system_prompt_filename or f"{self.name}.md")

    def load_system_prompt(self) -> str:
        """Read and return this role's system prompt text."""
        return self.system_prompt_path.read_text(encoding="utf-8")


def default_role_registry() -> dict[str, AgentRole]:
    """Return the canonical 3-agent role registry."""
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
        "critic": AgentRole(
            name="critic",
            backend_type=BackendType.CODEX,
            model=DEFAULT_CODEX_MODEL,
            api_key_env=DEFAULT_CODEX_API_KEY_ENV,
            allowed_intents=_CRITIC_INTENTS,
            can_delegate_side_effects=False,
            can_mutate_core_state=False,
            no_tools=True,  # Codex no-tools
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
            prompt_driven=False,
        ),
    }


__all__ = [
    "AgentRole",
    "BackendType",
    "DEFAULT_CLAUDE_API_KEY_ENV",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_CODEX_API_KEY_ENV",
    "DEFAULT_CODEX_MODEL",
    "SPECIALIST_INTENTS",
    "default_role_registry",
]
