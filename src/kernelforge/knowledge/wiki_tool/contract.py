# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a forge session and the knowledge base server agree on: tool names, scripts, launch environment, timeout.

Kept apart from the server so that importing the session side never imports the module ``python -m`` runs.
"""

from __future__ import annotations

SEARCH, PAGE, GREP, GUIDE = "search", "page", "grep", "guide"
# Agents see these as mcp__knowledge_wiki__<name>.
TOOL_NAMES = (SEARCH, PAGE, GREP, GUIDE)

ROOT_ENV = "FORGE_KNOWLEDGE_WIKI_ROOT"
LOG_ENV = "FORGE_KNOWLEDGE_WIKI_LOG"
ROLE_ENV = "FORGE_KNOWLEDGE_WIKI_ROLE"

# The checkout's own scripts each tool runs; a checkout missing one is not a knowledge base.
SCRIPTS = {SEARCH: "query.py", PAGE: "get_page.py", GREP: "grep_wiki.py"}

CALL_TIMEOUT_SEC = 60
