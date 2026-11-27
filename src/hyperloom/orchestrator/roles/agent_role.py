# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Agent role definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from hyperloom.inference_optimizer.session.paths import asset_system_prompts_dir
from hyperloom.inference_optimizer.protocol.intent import IntentType


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
        IntentType.REQUEST,
        IntentType.EXTEND_LEASE,
        IntentType.PRUNE_BRANCH,
        IntentType.ESCALATE_STRATEGY_CHANGE,
    }
)


@dataclass(frozen=True)
class AgentRole:
    """Static role record. Backend instances are created elsewhere."""

    name: str
    allowed_intents: frozenset[IntentType]

    @property
    def system_prompt_path(self) -> Path:
        """Path to this role's system prompt markdown file."""
        return asset_system_prompts_dir() / f"{self.name}.md"

    def load_system_prompt(self) -> str:
        """Read and return this role's system prompt text."""
        return self.system_prompt_path.read_text(encoding="utf-8")


def default_role_registry() -> dict[str, AgentRole]:
    """Return the canonical role registry: the orchestration role."""
    return {
        "orchestration": AgentRole(
            name="orchestration",
            allowed_intents=_ORCHESTRATION_INTENTS,
        ),
    }


__all__ = [
    "AgentRole",
    "default_role_registry",
]
