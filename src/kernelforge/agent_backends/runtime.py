# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resolve a run's configuration into the Agent provider runtime it asks for."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from kernelforge.agent_backends.base import AgentRuntimeConfig, AgentRuntimeSettings
from kernelforge.agent_backends.registry import resolve_agent_runtime, select_default_agent_provider
from kernelforge.config import resolve_agent_model

if TYPE_CHECKING:
    from kernelforge.config import Config


def agent_runtime_settings(config: Config) -> AgentRuntimeSettings:
    """The configured runtime settings, independent of which provider runs them."""
    return AgentRuntimeSettings(
        executable=config.agent_cli,
        timeout_sec=config.agent_timeout_sec,
        reasoning_effort=config.agent_reasoning_effort,
        sandbox_mode=config.agent_sandbox_mode,
        precheck=config.agent_precheck,
        fallback_provider=config.agent_fallback_provider,
        options=config.agent_options,
    )


def agent_runtime(config: Config) -> AgentRuntimeConfig:
    """Resolve the provider ``config`` selects into one complete runtime config.

    Raises ``ValueError`` for a provider or fallback provider no registered provider answers to.
    """
    provider = config.agent_backend
    if provider == "auto":
        provider = select_default_agent_provider(config.agent_model).name
    # The model variable is per-provider, so it can only be read once the provider is settled -- reading it before
    # ``auto`` resolves is how a Claude model id reaches Codex.
    model = config.agent_model or resolve_agent_model(provider)
    return resolve_agent_runtime(provider, model=model, settings=agent_runtime_settings(config))


def agent_runtime_for(config: Config, provider: str, model: str = "", *, executable: str = "") -> AgentRuntimeConfig:
    """Resolve one explicit provider/model with ``config``'s settings and that provider's own executable."""
    return resolve_agent_runtime(
        provider,
        model=model,
        settings=replace(agent_runtime_settings(config), executable=executable),
    )


__all__ = ["agent_runtime", "agent_runtime_for", "agent_runtime_settings"]
