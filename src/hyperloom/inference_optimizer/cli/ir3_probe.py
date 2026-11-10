# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""IR-3 reachability probe: run the KB / PR Monitor probe script and read the marker it writes."""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger("hyperloom.inference_optimizer.cli")


def run_ir3_probe_script(script: Path, env: dict[str, str], *, marker_path: Path, explicit_pr: bool) -> dict[str, Any]:
    """Run the IR-3 probe script and read its reachability marker, unreachable on any failure."""
    try:
        subprocess.run(
            ["bash", str(script)],
            env=env,
            check=False,
            timeout=60,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        log.warning("IR-3 preflight script error: %s", exc)
        marker: dict[str, Any] = {
            "kb_reachable": False,
            "pr_reachable": False,
            "kb_skipped": True,
            "pr_skipped": explicit_pr,
        }
    else:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("IR-3 marker unreadable: %s", exc)
            marker = {
                "kb_reachable": False,
                "pr_reachable": False,
                "kb_skipped": True,
                "pr_skipped": explicit_pr,
            }
    return marker
