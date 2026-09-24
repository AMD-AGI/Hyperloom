# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MLPerf AgentX KEEP gate: smoke for search, full 613 fail-closed before KEEP."""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from hyperloom.orchestrator.actions.executors._accuracy_gate import AGENTX_ERROR_RATE_THRESHOLD_PCT
from hyperloom.orchestrator.actions.executors._grid_base import GridVariant
from hyperloom.common.coerce import to_str_list

log = logging.getLogger(__name__)

MLPERF_FULL_FLOW = "full"
MLPERF_SMOKE_FLOW = "smoke_test"
MLPERF_FULL_TRAJECTORIES = "613"


def mlperf_full_env_overrides(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """Env the Magpie client needs for online.yaml (613 trajectories)."""
    envs = dict(base or {})
    envs["MLPERF_AGENTIC_FLOW"] = MLPERF_FULL_FLOW
    envs["AGENTIC_NUM_TRAJECTORIES"] = MLPERF_FULL_TRAJECTORIES
    return envs


def clone_variant_for_mlperf_full(gv: Any) -> GridVariant:
    """Copy a grid variant and pin it to the full MLPerf flow."""
    envs = mlperf_full_env_overrides(dict(getattr(gv, "extra_envs", {}) or {}))
    cloned = GridVariant(
        name=f"{gv.name}-mlperf-full",
        extra_server_args=getattr(gv, "extra_server_args", "") or "",
        extra_envs=envs,
        note=f"{getattr(gv, 'note', '')} mlperf_full_keep".strip(),
        remove_args=to_str_list(getattr(gv, "remove_args", [])),
        unset_envs=to_str_list(getattr(gv, "unset_envs", [])),
        args_mode=str(getattr(gv, "args_mode", "append") or "append"),
    )
    runtime = getattr(gv, "runtime_override", None)
    if isinstance(runtime, dict) and runtime:
        cloned.runtime_override = dict(runtime)
    return cloned


def load_mapped_inferencex_result(workspace: str | Path | None) -> dict[str, Any] | None:
    """Newest ``inferencex_result.json`` under a Magpie workspace."""
    if not workspace:
        return None
    root = Path(workspace)
    matches = [Path(p) for p in glob.glob(str(root / "**" / "inferencex_result.json"), recursive=True)]
    if not matches and (root / "inferencex_result.json").is_file():
        matches = [root / "inferencex_result.json"]
    if not matches:
        return None
    latest = max(matches, key=lambda path: path.stat().st_mtime)
    try:
        data = json.loads(latest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def mlperf_full_keep_block(
    workspace: str | Path | None,
    *,
    status: str | None = None,
) -> str:
    """Return a fail-closed reason, or ``\"\"`` when the full run may KEEP."""
    if status and status != "succeeded":
        return status or "mlperf_full_run_failed"
    mapped = load_mapped_inferencex_result(workspace)
    if not mapped:
        return "mlperf_full_result_missing"
    if mapped.get("mlperf_complete") is False or mapped.get("submission_valid") is False:
        reasons = mapped.get("submission_invalid_reasons") or []
        return "mlperf_full_incomplete:" + ",".join(str(r) for r in reasons) if reasons else "mlperf_full_incomplete"
    rate = mapped.get("request_error_rate")
    if not isinstance(rate, (int, float)) or rate > AGENTX_ERROR_RATE_THRESHOLD_PCT:
        return "mlperf_full_error_rate"
    score = mapped.get("accuracy_score")
    if not isinstance(score, (int, float)):
        return "mlperf_full_accuracy_unavailable"
    return ""


def should_validate_mlperf_full(env: Mapping[str, str] | None = None) -> bool:
    from hyperloom.inference_optimizer.agentx.deploy import is_mlperf_backend

    return is_mlperf_backend(env)
