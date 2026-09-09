# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""ActionRunner for the ``session_breakdown`` action."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ...loop.coordinator_helpers import format_exc_brief

log = logging.getLogger(__name__)


class SessionBreakdownExecutor:
    """Materialize ``session_breakdown.json`` for the current session."""

    async def __call__(self, ctx) -> dict[str, Any]:
        """Write ``session_breakdown.json`` and return its path + metadata."""
        session_dir = self._resolve_session_dir(ctx)
        if session_dir is None:
            return {
                "status": "failed",
                "error": "session_breakdown_executor: could not resolve session_dir",
            }

        params = ctx.task.params or {}
        output_path = params.get("output_path")

        from hyperloom.inference_optimizer.breakdown import build, write_breakdown_json

        try:
            target = write_breakdown_json(session_dir, output_path=output_path)
        except Exception as exc:  # noqa: BLE001
            log.exception("session_breakdown export failed")
            return {
                "status": "failed",
                "error": format_exc_brief(exc),
            }

        # Surface warnings + size to the bus event.
        try:
            warnings = build(session_dir).get("warnings") or []
        except Exception:  # noqa: BLE001
            warnings = []

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

    @staticmethod
    def _resolve_session_dir(ctx) -> Path | None:
        """Same resolution order as :class:`ReportExecutor`."""
        extra = getattr(ctx, "extra", None) or {}
        if extra.get("session_dir"):
            return Path(extra["session_dir"])
        params = ctx.task.params or {}
        if params.get("session_dir"):
            return Path(params["session_dir"])
        from hyperloom.inference_optimizer.session.paths import session_dir as _sd

        candidate = _sd()
        # manifest.json (not state.json) so a fresh session yields a partial breakdown.
        if candidate.exists() and (candidate / "manifest.json").exists():
            return candidate
        return None


session_breakdown_executor = SessionBreakdownExecutor()


__all__ = ["SessionBreakdownExecutor", "session_breakdown_executor"]
