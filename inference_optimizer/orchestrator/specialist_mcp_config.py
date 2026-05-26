"""Specialist subprocess MCP config generator.

Generates the ``--mcp-config`` JSON file that the ``claude`` subprocess
spawned by :class:`SpecialistSubprocessDispatcher` reads. Without this
file the subprocess starts with ``mcp_servers=[]`` and any
``mcp__<server>__*`` tool name listed in ``--allowedTools`` resolves to
nothing — the specialist silently loses access to PR Monitor.

The generated config currently registers one server:

* ``pr_monitor`` — streamable-HTTP MCP at the URL advertised by
  :meth:`KnowledgePlane.specialist_mcp_url`. The server name MUST be
  ``pr_monitor`` so the tool names already in the specialist whitelist
  (``mcp__pr_monitor__pr_search`` / ``pr_get`` / …) resolve.

Cortex KB does not expose an MCP surface (REST only); its read
context is pre-warmed into Section 4 of the specialist prompt by
``Coordinator._warm_specialist_params``. Listing dead
``mcp__cortex_kb__*`` names in the whitelist is harmful (LLM tries to
call them and gets tool-not-found errors), so the runner-level
whitelist no longer includes them either.

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
) -> Path | None:
    """Write the specialist subprocess MCP config and return its path.

    Returns ``None`` when no MCP server is wireable (caller should then
    leave ``--mcp-config`` off the claude command line). Mutating the
    on-disk file is idempotent: repeated calls with the same inputs
    overwrite the file with byte-identical content.

    Parameters
    ----------
    session_dir
        Session root. The file lands at
        ``<session_dir>/runtime/<SPECIALIST_MCP_CONFIG_FILENAME>``.
    pr_monitor_mcp_url
        ``KnowledgePlane.specialist_mcp_url()`` — empty string means
        PR Monitor is disabled / degraded.
    """
    servers: dict[str, dict[str, Any]] = {}
    pr_url = (pr_monitor_mcp_url or "").strip()
    if pr_url:
        servers["pr_monitor"] = {
            "type": "http",
            "url": pr_url,
        }

    if not servers:
        log.info(
            "specialist_mcp_config: no MCP servers to wire "
            "(pr_monitor disabled?); skipping config file generation"
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
        cfg_path, sorted(servers.keys()),
    )
    return cfg_path


__all__ = [
    "SPECIALIST_MCP_CONFIG_FILENAME",
    "write_specialist_mcp_config",
]
