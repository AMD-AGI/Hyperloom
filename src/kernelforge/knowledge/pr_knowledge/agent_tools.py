# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Names of the upstream PR tools an implementer agent may call."""

from __future__ import annotations

# Prefixed so they cannot collide with the upstream PR Monitor's own MCP tools; agents see them as
# mcp__pr_monitor__<name>.
TOOL_NAMES = ("pr_find_references", "pr_get_reference", "pr_get_file_patch", "pr_check_inclusion")
