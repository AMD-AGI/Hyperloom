# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ActionRunner for the ``session_breakdown`` action."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class SessionBreakdownExecutor:
    """Materialize ``session_breakdown.json`` for the current session."""

    async def __call__(self, ctx) -> dict[str, Any]:
        """Write ``session_breakdown.json`` and return its path + metadata."""
        session_dir = Path(ctx.extra["session_dir"])
        params = ctx.task.params or {}
        output_path = params.get("output_path")

        from hyperloom.inference_optimizer.breakdown import write_breakdown_json

        # The export also writes the session's Experiences to the local service; that I/O stays off the loop.
        target = await asyncio.to_thread(write_breakdown_json, session_dir, output_path=output_path)

        # Surface warnings + size to the bus event.
        doc = json.loads(target.read_text(encoding="utf-8"))
        warnings = (doc.get("metadata") or {}).get("warnings") or []

        log.info(
            "session_breakdown_executor: wrote %s (%d warnings)",
            target,
            len(warnings),
        )
        return {
            "status": "succeeded",
            "breakdown_path": str(target),
            "warnings": warnings,
            "size_bytes": int(target.stat().st_size) if target.exists() else 0,
        }


session_breakdown_executor = SessionBreakdownExecutor()


__all__ = ["SessionBreakdownExecutor", "session_breakdown_executor"]
