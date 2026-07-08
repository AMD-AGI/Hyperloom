# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Specialist subprocess MCP config generator.

Generates the ``--mcp-config`` JSON the ``claude`` subprocess reads,
registering up to two streamable-HTTP MCP servers:

- ``pr_monitor`` (at :meth:`KnowledgePlane.specialist_mcp_url`) — the name MUST
  be ``pr_monitor`` so the ``mcp__pr_monitor__*`` whitelist names resolve.
- ``cortex_kb`` (at :meth:`KnowledgePlane.cortex_specialist_mcp_url`) — the
  read-only KB-graph (gbrain) server; the name MUST be ``cortex_kb`` so the
  ``mcp__cortex_kb__*`` whitelist names resolve. Auth headers (bearer token)
  are passed through when supplied.

Schema follows :data:`claude_agent_sdk.types.McpHttpServerConfig`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)


SPECIALIST_MCP_CONFIG_FILENAME = "specialist_mcp.json"


def write_specialist_mcp_config(
    *,
    session_dir: Path | str,
    pr_monitor_mcp_url: str,
    cortex_kb_mcp_url: str = "",
    cortex_kb_mcp_headers: dict[str, str] | None = None,
) -> Path | None:
    """Write the specialist subprocess MCP config and return its path.

    Returns ``None`` when no MCP server is wireable (caller leaves
    ``--mcp-config`` off). Idempotent.

    Args:
        session_dir: Session root; the file lands at
            ``<session_dir>/runtime/<SPECIALIST_MCP_CONFIG_FILENAME>``.
        pr_monitor_mcp_url: ``KnowledgePlane.specialist_mcp_url()``; empty
            means disabled.
        cortex_kb_mcp_url: ``KnowledgePlane.cortex_specialist_mcp_url()``;
            empty means the read-only KB-graph server is disabled.
        cortex_kb_mcp_headers: Optional auth headers for the ``cortex_kb``
            server (e.g. ``{"Authorization": "Bearer …"}``).

    Returns:
        The written config file path, or ``None`` when no MCP server is
        wireable.
    """
    servers: dict[str, dict[str, Any]] = {}
    pr_url = (pr_monitor_mcp_url or "").strip()
    if pr_url:
        servers["pr_monitor"] = {
            "type": "http",
            "url": pr_url,
        }

    kb_url = (cortex_kb_mcp_url or "").strip()
    if kb_url:
        kb_server: dict[str, Any] = {
            "type": "http",
            "url": kb_url,
        }
        kb_headers = {
            str(k): str(v)
            for k, v in (cortex_kb_mcp_headers or {}).items()
            if str(k).strip() and str(v).strip()
        }
        if kb_headers:
            kb_server["headers"] = kb_headers
        servers["cortex_kb"] = kb_server

    if not servers:
        log.info(
            "specialist_mcp_config: no MCP servers to wire (pr_monitor disabled?); skipping config file generation"
        )
        return None

    runtime_dir = Path(session_dir) / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = runtime_dir / SPECIALIST_MCP_CONFIG_FILENAME
    payload = {"mcpServers": servers}
    cfg_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    log.info(
        "specialist_mcp_config: wrote %s (servers=%s)",
        cfg_path,
        sorted(servers.keys()),
    )
    return cfg_path


__all__ = [
    "SPECIALIST_MCP_CONFIG_FILENAME",
    "write_specialist_mcp_config",
]
