# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A configured kernel knowledge base, served read-only to every forge role with tools."""

from kernelforge.knowledge.wiki_tool.tool import (
    LOG_NAME,
    SERVER_KEY,
    KnowledgeWikiTool,
    require_mcp_backend,
    wiki_servers,
    wiki_tool_names,
)

__all__ = [
    "LOG_NAME",
    "SERVER_KEY",
    "KnowledgeWikiTool",
    "require_mcp_backend",
    "wiki_servers",
    "wiki_tool_names",
]
