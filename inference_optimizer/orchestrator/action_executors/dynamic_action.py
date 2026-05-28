"""Stub ``dynamic_action`` executor — action_dynamic_plan/P1 §7.

P1 ships only the dispatch skeleton. The real multi-turn ReAct sub-agent
runner replaces this stub at P3 (action_dynamic_plan/P3_subagent_runner.md).

Behaviour:

* Echo the dispatch into ``dispatch_history.jsonl`` under the per-dyn_id
  artefact dir (``runs/dynamic_action/<dyn_id>/``).
* Return an empty ``proposal_set`` so the downstream specialist-equivalent
  empty path takes over (no critic, no grid runner).

The empty ``proposal_set`` is the canonical "stub" signal the P1
acceptance criteria expect; P3 swaps the runner without touching the
upstream dispatch chain.
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


def _dispatch_history_path(workspace: Path) -> Path:
    return workspace / "dispatch_history.jsonl"


async def dynamic_action_executor(ctx: RunnerContext) -> dict[str, Any]:
    """P1 stub: write an empty proposal_set + one dispatch_history row.

    The workspace is pre-created by SubAgentRunner._pre_mkdir_workspace
    (``runs/dynamic_action/<dyn_id>/``). The Coordinator's dispatch
    branch is the one that generates the ``dyn_id`` and seeds
    ``spec.json`` at dispatch time (P1 §6); this executor only logs
    that the stub ran and the proposal_set is empty.
    """
    workspace_str = ctx.extra.get("workspace")
    dyn_id = ctx.task.params.get("dyn_id") or ctx.task.task_id
    ts = _now_iso()
    record = {
        "ts": ts,
        "dyn_id": str(dyn_id),
        "outcome": "stub_empty",
        "task_id": ctx.task.task_id,
    }
    if workspace_str:
        workspace = Path(workspace_str)
        workspace.mkdir(parents=True, exist_ok=True)
        try:
            with _dispatch_history_path(workspace).open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError as exc:
            log.warning(
                "dynamic_action stub: failed to append dispatch_history "
                "for dyn_id=%s: %r",
                dyn_id, exc,
            )
        proposal_path = workspace / "proposal_set.json"
        try:
            proposal_path.write_text(
                json.dumps(
                    {"dyn_id": str(dyn_id), "proposal_set": [], "empty": True},
                    sort_keys=True, indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            log.warning(
                "dynamic_action stub: failed to write proposal_set.json "
                "for dyn_id=%s: %r",
                dyn_id, exc,
            )
    return {
        "dyn_id": str(dyn_id),
        "proposal_set": [],
        "empty": True,
        "outcome": "stub_empty",
        "summary": (
            "dynamic_action P1 stub executor — no exploration performed "
            "(real ReAct runner lands at P3)."
        ),
    }


__all__ = ["dynamic_action_executor"]
