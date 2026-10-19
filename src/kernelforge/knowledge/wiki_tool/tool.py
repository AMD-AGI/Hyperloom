# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Attach the read-only knowledge base MCP server to forge agent sessions.

Off unless ``Config.knowledge_wiki_root`` names a checkout. When it does, every
forge role that has tools gets the same server, so the planning roles, which
have no shell, can consult the knowledge base like the implementer can.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

from kernelforge.agent_backends.base import PYTHON_LAUNCH_ENV_VARS, StdioMcpServer
from kernelforge.knowledge.wiki_tool.contract import (
    CALL_TIMEOUT_SEC,
    LOG_ENV,
    ROLE_ENV,
    ROOT_ENV,
    TOOL_NAMES,
)

if TYPE_CHECKING:
    from kernelforge.agent_backends.base import AgentBackend, AgentRuntimeConfig
    from kernelforge.agent_backends.registry import AgentProvider
    from kernelforge.config import Config

SERVER_KEY = "knowledge_wiki"
LOG_NAME = "knowledge_wiki_calls.jsonl"

# The server's launch, and what the checkout's scripts it runs need to find their programs and libraries.
_CHILD_ENV_VARS = (*PYTHON_LAUNCH_ENV_VARS, "PATH", "LD_LIBRARY_PATH")


@dataclass(frozen=True)
class KnowledgeWikiTool:
    """One run's knowledge base checkout and the log its calls are appended to."""

    root: Path
    log_path: Path

    @classmethod
    def from_config(cls, config: Config) -> KnowledgeWikiTool | None:
        if not config.knowledge_wiki_root:
            return None
        log = config.knowledge_wiki_log
        return cls(
            root=Path(config.knowledge_wiki_root),
            log_path=Path(log) if log else Path(config.experiments_dir) / LOG_NAME,
        )

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(f"mcp__{SERVER_KEY}__{name}" for name in TOOL_NAMES)

    def mcp_servers(self, role: str) -> dict[str, StdioMcpServer]:
        env = {name: os.environ[name] for name in _CHILD_ENV_VARS if os.environ.get(name, "").strip()}
        env.update({ROOT_ENV: str(self.root), LOG_ENV: str(self.log_path), ROLE_ENV: role})
        return {
            SERVER_KEY: StdioMcpServer(
                command=sys.executable,
                args=("-m", "kernelforge.knowledge.wiki_tool.mcp_server"),
                env=env,
                startup_timeout_sec=15,
                tool_timeout_sec=CALL_TIMEOUT_SEC + 10,
                tools=self.tool_names,
            )
        }


# Checkouts a --wiki-guided campaign stopped offering once its knowledge base round ended. Module state because every
# role's session is built by a factory the loop never sees; a forge-loop process runs one campaign.
_WITHDRAWN: set[Path] = set()


def withdraw_knowledge_wiki(tool: KnowledgeWikiTool) -> None:
    """Offer ``tool`` to none of the sessions this process starts from now on."""
    _WITHDRAWN.add(tool.root)


def _offered(tool: KnowledgeWikiTool | None) -> bool:
    return tool is not None and tool.root not in _WITHDRAWN


def wiki_servers(tool: KnowledgeWikiTool | None, role: str) -> dict[str, StdioMcpServer]:
    """The knowledge base server for one session, or nothing when none is configured or it was withdrawn."""
    return tool.mcp_servers(role) if _offered(tool) else {}


def wiki_tool_names(tool: KnowledgeWikiTool | None) -> tuple[str, ...]:
    return tool.tool_names if _offered(tool) else ()


def require_mcp_backend(tool: KnowledgeWikiTool | None, backend: AgentBackend | AgentProvider, role: str) -> None:
    """Refuse a configured knowledge base on a backend, or a provider, whose sessions would silently drop it."""
    if tool is not None and not backend.capabilities.mcp:
        raise ValueError(
            f"the knowledge base needs an MCP-capable backend, but the {role} backend {backend.name} is not"
        )


def require_mcp_runtime(tool: KnowledgeWikiTool | None, runtime: AgentRuntimeConfig, role: str) -> None:
    """Refuse a configured knowledge base on a runtime whose backend is built later, by what its providers declare.

    That backend is the runtime's provider or, when that one is unavailable, its fallback, so both must keep the tools.
    """
    if tool is None:
        return
    from kernelforge.agent_backends.registry import get_agent_provider

    for provider in (runtime.provider, runtime.fallback_provider):
        if provider:
            require_mcp_backend(tool, get_agent_provider(provider), role)
