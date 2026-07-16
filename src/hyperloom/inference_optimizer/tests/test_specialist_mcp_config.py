# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import json

from hyperloom.orchestrator.specialists.mcp_config import write_specialist_mcp_config


def test_specialist_mcp_config_skips_authorized_cortex(tmp_path):
    cfg = write_specialist_mcp_config(
        session_dir=tmp_path,
        pr_monitor_mcp_url="https://pr.example/mcp",
        cortex_kb_mcp_url="https://kb.example/mcp",
        cortex_kb_mcp_headers={"Authorization": "Bearer secret-token"},
    )
    assert cfg is not None
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    assert "pr_monitor" in payload["mcpServers"]
    assert "cortex_kb" not in payload["mcpServers"]
    assert "secret-token" not in cfg.read_text(encoding="utf-8")


def test_specialist_mcp_config_keeps_headerless_cortex(tmp_path):
    cfg = write_specialist_mcp_config(
        session_dir=tmp_path,
        pr_monitor_mcp_url="",
        cortex_kb_mcp_url="https://kb.example/mcp",
    )
    assert cfg is not None
    payload = json.loads(cfg.read_text(encoding="utf-8"))
    assert payload["mcpServers"]["cortex_kb"]["url"] == "https://kb.example/mcp"
