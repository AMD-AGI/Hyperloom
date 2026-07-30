# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

from hyperloom.orchestrator.specialists.mcp_config import write_specialist_mcp_config


def test_specialist_mcp_config_writes_pr_monitor(tmp_path):
    cfg = write_specialist_mcp_config(
        session_dir=tmp_path,
        pr_monitor_mcp_url="https://pr.example/mcp",
    )
    assert cfg is not None
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    assert payload["mcpServers"]["pr_monitor"]["url"] == "https://pr.example/mcp"
    assert "cortex_kb" not in payload["mcpServers"]


def test_specialist_mcp_config_returns_none_when_no_servers(tmp_path):
    cfg = write_specialist_mcp_config(
        session_dir=tmp_path,
        pr_monitor_mcp_url="",
    )
    assert cfg is None
