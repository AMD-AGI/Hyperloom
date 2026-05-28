"""Stub ``dynamic_action`` executor — action_dynamic_plan/P1 §7 + P2 §6.

The P1 dispatch skeleton + P2 artefact bundle are the productised
parts; this executor stays a stub until P3 lands the multi-turn
ReAct runner.

Behaviour (P2):

* Locate the artefact dir via ``ctx.task.params['artifact_path']``
  (Coordinator-injected; falls back to ``ctx.extra['workspace']`` for
  back-compat) and append one ``dispatch_history.jsonl`` row.
* Write an empty ``proposal_set.json`` into the same dir so downstream
  consumers see the canonical empty signal.
* Return an empty ``proposal_set`` so the existing empty path
  (specialist-equivalent) takes over without critic / grid runner.

P3 swaps the executor body without touching upstream dispatch wiring
or the artefact layout.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..sub_agent_runner import RunnerContext

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _resolve_artifact_dir(ctx: RunnerContext) -> Path | None:
    """Prefer the Coordinator-injected ``params['artifact_path']``
    (P2 §6); fall back to the per-task workspace for legacy callers
    that bypass the prepare hook (tests). Returns None when neither
    source resolves."""
    explicit = ctx.task.params.get("artifact_path")
    if explicit:
        return Path(explicit)
    workspace_str = ctx.extra.get("workspace")
    if workspace_str:
        return Path(workspace_str)
    return None


async def dynamic_action_executor(ctx: RunnerContext) -> dict[str, Any]:
    """Append one dispatch_history.jsonl row + write empty proposal_set.

    The history row uses the unified ``SUB_AGENT_DONE`` closed schema
    (G2) so consumers of dispatch_history can treat the stub uniformly
    with the real runner.
    """
    dyn_id = str(ctx.task.params.get("dyn_id") or ctx.task.task_id)
    artifact_dir = _resolve_artifact_dir(ctx)
    if artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        try:
            (artifact_dir / "proposal_set.json").write_text(
                json.dumps(
                    {
                        "dyn_id": dyn_id,
                        "proposal_set": [],
                        "empty": True,
                    },
                    sort_keys=True, indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning(
                "dynamic_action stub: proposal_set write failed for "
                "dyn_id=%s: %r",
                dyn_id, exc,
            )
        # Closed-schema history append; only when we can derive the
        # session_dir from the artefact dir layout
        # (``<session_dir>/agents/orchestration/dynamic_actions/<dyn_id>``).
        session_dir = _session_dir_from_artifact(artifact_dir, dyn_id)
        if session_dir is not None:
            from ..dynamic_action_history import (
                DispatchHistoryEvent,
                append_dispatch_history_row,
            )
            try:
                append_dispatch_history_row(
                    session_dir=session_dir,
                    dyn_id=dyn_id,
                    event=DispatchHistoryEvent.SUB_AGENT_DONE,
                    payload={
                        "terminal_state": "COMPLETED_EMPTY",
                        "reason": "stub_empty",
                        "turns_used": 0,
                        "journal_path": "",
                        "proposal_count": 0,
                    },
                )
            except Exception:  # noqa: BLE001 — defensive
                log.exception(
                    "dynamic_action stub: history append failed for "
                    "dyn_id=%s", dyn_id,
                )
    return {
        "dyn_id": dyn_id,
        "proposal_set": [],
        "empty": True,
        "outcome": "stub_empty",
        "summary": (
            "dynamic_action P1+P2 stub executor — no exploration "
            "performed (real ReAct runner lands at P3)."
        ),
    }


def _session_dir_from_artifact(
    artifact_dir: Path, dyn_id: str,
) -> Path | None:
    """Walk the canonical layout back to the session root so the stub
    can call the unified history writer without an extra plumb-through.

    ``artifact_dir`` is
    ``<session_dir>/agents/orchestration/dynamic_actions/<dyn_id>``;
    we strip the four trailing path segments to recover the session
    root. Returns ``None`` if the layout does not match.
    """
    if artifact_dir.name != dyn_id:
        return None
    candidate = artifact_dir.parent.parent.parent.parent
    if (candidate / "agents").is_dir() or candidate.is_dir():
        return candidate
    return None


__all__ = ["dynamic_action_executor"]
