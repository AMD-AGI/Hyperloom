# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Backend selection for Forge implementer sessions."""

from __future__ import annotations

from kernelforge.agent_backends.base import (
    AGENT_WATCHDOG_GRACE_SEC,
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
    watchdog_timeout_sec,
)
from kernelforge.agent_backends.registry import (
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
    "AGENT_WATCHDOG_GRACE_SEC",
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
    "watchdog_timeout_sec",
]
