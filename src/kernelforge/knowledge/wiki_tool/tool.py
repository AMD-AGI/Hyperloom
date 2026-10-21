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

from kernelforge.agent_backends.base import StdioMcpServer
from kernelforge.knowledge.wiki_tool.contract import (
    CALL_TIMEOUT_SEC,
    LOG_ENV,
    ROLE_ENV,
    ROOT_ENV,
    TOOL_NAMES,
)

if TYPE_CHECKING:
    from kernelforge.config import Config

SERVER_KEY = "knowledge_wiki"
LOG_NAME = "knowledge_wiki_calls.jsonl"

# What the server's own launch needs; the MCP client does not forward the parent environment.
_CHILD_ENV_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX", "PATH", "LD_LIBRARY_PATH")


@dataclass(frozen=True)
class KnowledgeWikiTool:
    """One run's knowledge base checkout and the log its calls are appended to."""

    root: Path
    log_path: Path

    @classmethod
    def from_config(cls, config: Config) -> KnowledgeWikiTool | None:
        root = str(getattr(config, "knowledge_wiki_root", "") or "").strip()
        if not root:
            return None
        log = str(getattr(config, "knowledge_wiki_log", "") or "").strip()
        log_path = Path(log) if log else Path(config.experiments_dir) / LOG_NAME
        return cls(root=Path(root), log_path=log_path)

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


def wiki_servers(tool: KnowledgeWikiTool | None, role: str) -> dict[str, StdioMcpServer]:
    """The knowledge base server for one session, or nothing when no knowledge base is configured."""
    return {} if tool is None else tool.mcp_servers(role)


def wiki_tool_names(tool: KnowledgeWikiTool | None) -> tuple[str, ...]:
    return () if tool is None else tool.tool_names


def require_mcp_backend(tool: KnowledgeWikiTool | None, backend, role: str) -> None:
    """Refuse a configured knowledge base on a backend that would silently drop it."""
    capabilities = getattr(backend, "capabilities", None)
    if tool is not None and not getattr(capabilities, "mcp", False):
        name = getattr(backend, "name", type(backend).__name__)
        raise ValueError(f"the knowledge base needs an MCP-capable backend, but the {role} backend {name} is not")
