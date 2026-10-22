# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A static, self-refreshing HTML dashboard of one forge-loop campaign, written next to its other artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from kernelforge.loop.dashboard.render import render_dashboard
from kernelforge.loop.dashboard.snapshot import DASHBOARD_HTML, DASHBOARD_JSON, LiveStatus, collect_snapshot
from kernelforge.tools.durable_io import atomic_write_text

__all__ = ["DASHBOARD_HTML", "LiveStatus", "publish_dashboard"]


def publish_dashboard(campaign_root: str | Path, live: LiveStatus | None = None) -> Path:
    """Write ``dashboard.json`` and ``dashboard.html`` under ``campaign_root``; return the HTML path."""
    root = Path(campaign_root)
    snapshot = collect_snapshot(root, live)
    atomic_write_text(root / DASHBOARD_JSON, json.dumps(snapshot, indent=2, default=str))
    html_path = root / DASHBOARD_HTML
    atomic_write_text(html_path, render_dashboard(snapshot))
    return html_path
