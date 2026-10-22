# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A static, self-refreshing HTML dashboard of one forge-loop campaign, written next to its other artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path

from kernelforge.loop.dashboard.render import render_dashboard
from kernelforge.loop.dashboard.snapshot import DASHBOARD_HTML, DASHBOARD_JSON, LiveStatus, collect_snapshot
from kernelforge.tools.durable_io import atomic_write_text

__all__ = ["DASHBOARD_HTML", "LiveStatus", "publish_dashboard"]


def publish_dashboard(campaign_root: str | Path, live: LiveStatus | None = None) -> Path:
    """Write ``dashboard.json`` and ``dashboard.html`` under ``campaign_root``; return the HTML path."""
    root = Path(campaign_root)
    snapshot = collect_snapshot(root, live)
    json_path, html_path = root / DASHBOARD_JSON, root / DASHBOARD_HTML
    atomic_write_text(json_path, json.dumps(snapshot, indent=2, default=str))
    atomic_write_text(html_path, render_dashboard(snapshot))
    # Readable by whoever syncs the page off the node; the atomic write leaves its temporary file's 0600.
    for path in (json_path, html_path):
        os.chmod(path, 0o644)
    return html_path
