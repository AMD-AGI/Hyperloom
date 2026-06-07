# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""ActionRunner for the ``session_breakdown`` action.

Thin wrapper around :func:`inference_optimizer.breakdown.write_breakdown_json`
so the Coordinator / orchestration agent can refresh
``$SESSION_DIR/session_breakdown.json`` on demand (after an ``explore``
KEEP, before a planned ``report``, when a live dashboard is observing
this session, etc.).

The end-of-session safety net lives in ``cli.py``'s finally block — this
action is the agent-driven path used **during** a session for live
refreshes.

Returned shape::

    {
      "status":         "succeeded" | "failed",
      "breakdown_path": str,           # absolute path to the JSON file
      "warnings":       list[str],     # from inside the build
      "size_bytes":     int,
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class SessionBreakdownExecutor:
    """Materialize ``session_breakdown.json`` for the current session.

    Honors ``ctx.task.params`` keys:

    * ``session_dir`` (optional override; same precedence as ReportExecutor)
    * ``output_path`` (optional override; defaults to
      ``<session_dir>/session_breakdown.json``)
    """

    async def __call__(self, ctx) -> dict[str, Any]:
        session_dir = self._resolve_session_dir(ctx)
        if session_dir is None:
            return {
                "status": "failed",
                "error":  "session_breakdown_executor: could not resolve session_dir",
            }

        params = ctx.task.params or {}
        output_path = params.get("output_path")

        from ...breakdown import build, write_breakdown_json
        try:
            target = write_breakdown_json(session_dir, output_path=output_path)
        except Exception as exc:  # noqa: BLE001
            log.exception("session_breakdown export failed")
            return {
                "status": "failed",
                "error":  f"{type(exc).__name__}: {exc}",
            }

        # Re-read the rendered JSON to surface warnings + size to the bus
        # event without parsing it back.
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
            "status":         "succeeded",
            "breakdown_path": str(target),
            "warnings":       warnings,
            "size_bytes":     int(target.stat().st_size) if target.exists() else 0,
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
        from ...paths import session_dir as _sd
        candidate = _sd()
        # Don't require state.json — a fresh session with only manifest can
        # still produce a partial breakdown (with a warning).
        if candidate.exists() and (candidate / "manifest.json").exists():
            return candidate
        return None


session_breakdown_executor = SessionBreakdownExecutor()


__all__ = ["SessionBreakdownExecutor", "session_breakdown_executor"]
