# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Backend selection for Forge implementer sessions."""

from __future__ import annotations

from forge_llm.agent_backends.base import (
    AgentBackend,
    AgentCapabilities,
    AgentHook,
    AgentHooks,
    AgentProviderError,
    AgentProviderUnavailableError,
    AgentRole,
    AgentRunResult,
    AgentRunSpec,
    AgentRuntimeConfig,
    AgentToolPolicy,
    ResumableAgentBackend,
    StdioMcpServer,
)
from forge_llm.agent_backends.registry import (
    AgentProvider,
    PROVIDER_ENTRY_POINT_GROUP,
    create_registered_backend,
    discover_agent_providers,
    get_agent_provider,
    list_agent_providers,
    register_agent_provider,
    resolve_agent_runtime,
)

__all__ = [
    "AgentBackend",
    "AgentCapabilities",
    "AgentHook",
    "AgentHooks",
    "AgentProvider",
    "AgentProviderError",
    "AgentProviderUnavailableError",
    "PROVIDER_ENTRY_POINT_GROUP",
    "AgentRole",
    "AgentRunResult",
    "AgentRunSpec",
    "AgentRuntimeConfig",
    "AgentToolPolicy",
    "ResumableAgentBackend",
    "StdioMcpServer",
    "create_registered_backend",
    "discover_agent_providers",
    "get_agent_provider",
    "list_agent_providers",
    "register_agent_provider",
    "resolve_agent_runtime",
]
