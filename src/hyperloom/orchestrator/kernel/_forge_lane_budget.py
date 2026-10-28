# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One forge lane's share of the phase's remaining time, and the timeouts derived from it."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from hyperloom.common.coerce import optional_positive_int

from ..actions.executors._kernel_agent_tool import HandlerResult
from .lane_budget import INSUFFICIENT_BUDGET, LANE_GEMM, LaneAllocation
from .lane_budget import allocate as _allocate_lane_budgets

if TYPE_CHECKING:
    from kernelforge.gemm_tune import RoutingEvidence, RoutingTarget

log = logging.getLogger(__name__)


# Outer subprocess cap for the whole GEMM-tuning run (all shapes/tuners) when there is
# no lane share to take, as in an unbounded session; sized for large models with many
# GEMM shapes. Override via HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC (or payload timeout_sec).
_DEFAULT_GEMM_TUNING_TIMEOUT_SEC = 5 * 60 * 60


def _lane_budget(state: Any, lane: str, *, gemm_target_costs_sec: tuple[int, ...] = ()) -> LaneAllocation | None:
    """Derive one lane's share of the phase's remaining time.

    Wraps :func:`lane_budget.allocate`, which divides the remaining time between
    the lanes and returns, per lane, both a second budget and how many targets that
    budget can fund. Every lane draws its share from this single probe of session
    state, so the shares stay parts of one whole.
    """
    remaining = _lane_remaining_minutes(state)
    allocation = _allocate_lane_budgets(remaining, gemm_target_costs_sec=gemm_target_costs_sec)[lane]
    if remaining is not None and not allocation.is_fundable:
        return None
    return allocation


def _unfunded_lane_skip(lane: str) -> HandlerResult:
    """What a forge lane returns instead of starting when the phase cannot fund it: a skip, not a verdict."""
    reason = f"what is left of the KERNEL phase time budget cannot fund one {lane} target"
    log.info("forge %s lane skipped: %s", lane, reason)
    return {"status": "skipped", "error_class": INSUFFICIENT_BUDGET, "skip_reason": reason, "backend": "forge"}


def _lane_remaining_minutes(state: Any) -> float | None:
    """Minutes a lane may plan against: the tighter of session and phase.

    The session clock alone overfunds a phase that has already spent most of its
    own slice, and work planned past the phase exit is cut off partway through.
    """
    from ..phases.machine_state import phase_budget_remaining_seconds

    remaining_fn = getattr(state, "remaining_minutes", None)
    session = remaining_fn() if callable(remaining_fn) else None
    if session is None:
        return None
    phase_sec = phase_budget_remaining_seconds(state)
    if phase_sec is None:
        return float(session)
    return min(float(session), float(phase_sec) / 60.0)


def _gemm_tuning_timeout_sec(payload: dict, *, lane_budget_sec: int = 0) -> int:
    """Resolve the GEMM-tuning subprocess timeout in seconds.

    Reads ``payload['timeout_sec']`` then the
    ``HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC`` env var -- both operator inputs, so both
    outrank a derived budget -- then the lane's share of the phase, then the module
    default; the result is floored at 60 seconds.
    """
    raw = payload.get("timeout_sec") or os.environ.get(
        "HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC",
        "",
    )
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = lane_budget_sec if lane_budget_sec > 0 else _DEFAULT_GEMM_TUNING_TIMEOUT_SEC
    return max(60, value)


def _gemm_router_targets(
    *,
    model_path: str,
    target: RoutingTarget,
    evidence: RoutingEvidence,
) -> tuple[tuple[str, int], ...]:
    """Ask the forge router which tuners it would run and what each costs.

    The router is the only place a per-tuner runtime estimate exists, and it
    returns them already in the execution order the gemm lane ceiling assumes.
    """
    try:
        from kernelforge.gemm_tune import analyze_model, select_tuners

        specs = select_tuners(analyze_model(model_path), target, evidence)
    except (OSError, ValueError, TypeError, AttributeError):
        log.debug("GEMM: could not consult the tuner router for lane cost estimates", exc_info=True)
        return ()
    return tuple((str(spec.name), max(0, int(spec.estimated_minutes * 60))) for spec in specs if spec.should_run)


async def _priced_gemm_lane(
    state: Any,
    *,
    model_path: str,
    target: RoutingTarget,
    evidence: RoutingEvidence,
) -> LaneAllocation | None:
    """The gemm lane's share, priced on the router's own per-tuner estimates for ``_gemm_router_targets`` inputs."""
    targets = await asyncio.to_thread(_gemm_router_targets, model_path=model_path, target=target, evidence=evidence)
    return _lane_budget(state, LANE_GEMM, gemm_target_costs_sec=tuple(cost for _, cost in targets))


def _forge_fusion_timeout_sec(payload: dict, *, lane_budget_sec: int = 0) -> int:
    """Resolve the forge-fusion subprocess timeout in seconds.

    Args:
        payload (dict): Request payload that may carry ``timeout``/``timeout_sec``.
        lane_budget_sec (int): The fusion lane's share of the phase. ``0`` means there
            is no share to take, as in an unbounded session, and keeps the module
            default rather than collapsing an unattended lane to a one-second timeout.

    Returns:
        int: The resolved timeout in seconds (>= 1).
    """
    raw = (
        payload.get("timeout")
        or payload.get("timeout_sec")
        or os.environ.get(
            "FORGE_FUSION_TIMEOUT",
            "",
        )
    )
    try:
        value = int(float(raw))
    except (OverflowError, TypeError, ValueError):
        value = lane_budget_sec if lane_budget_sec > 0 else 7200
    return max(1, value)


def _fusion_attn_tp_size(state: object, payload: dict, *, tp: int) -> int:
    """Attention TP shard behind the session's serving ``tp``.

    DP-attention splits the ``tp`` ranks into ``dp`` attention groups, so
    forge-fuse cannot infer the attention shard from ``--tp`` alone -- and it
    stamps the harness group/head dims from exactly this number.
    """
    explicit = optional_positive_int(payload.get("attn_tp"), 0)
    if explicit:
        return explicit
    from hyperloom.inference_optimizer.grid_server_args import server_arg_value
    from hyperloom.inference_optimizer.roofline_ceiling import resolve_runtime_workload

    server_args = str(payload.get("extra_server_args") or "").strip()
    if not server_args:
        server_args = resolve_runtime_workload(state, arm="current_best").server_args

    if "--enable-dp-attention" not in str(server_args).replace("=", " ").split():
        return tp
    # launch_infera_node injects --dp-size = tp when a dp-attention flag arrives
    # without one, and these args are read before that injection.
    dp = (
        optional_positive_int(server_arg_value(server_args, "--dp-size"), 0)
        or optional_positive_int(server_arg_value(server_args, "--data-parallel-size"), 0)
        or tp
    )
    return max(1, tp // dp)


def _fusion_session_serve_args(
    state: object,
    payload: dict,
    *,
    framework: str,
    model_path: str,
) -> dict[str, int]:
    """TP / attention TP / KV block size / max-model-len the smoke must match."""
    tp = optional_positive_int(payload.get("tp") or getattr(state, "tp", 0), 0)
    max_model_len = optional_positive_int(payload.get("max_model_len") or getattr(state, "max_model_len", 0), 0)
    block_size = optional_positive_int(payload.get("block_size"), 0)
    if block_size <= 0 and "vllm" in (framework or "").strip().lower():
        from hyperloom.inference_optimizer.model_config_utils import (
            _sparse_kv_block_size,
        )

        block_size = optional_positive_int(_sparse_kv_block_size(model_path), 0)
    args: dict[str, int] = {}
    if tp:
        args["tp"] = tp
        args["attn_tp"] = _fusion_attn_tp_size(state, payload, tp=tp)
    if block_size:
        args["block_size"] = block_size
    if max_model_len:
        args["max_model_len"] = max_model_len
    return args


def _gemm_tuning_workspace(payload: dict, *, session_dir: Path) -> Path:
    """Resolve the workspace directory for a GEMM-tuning run.

    Honors an explicit ``payload['workspace_path']``; otherwise builds a path
    under ``<session_dir>/runs/gemm_tuning/`` keyed by ``task_id`` /
    ``request_id`` (or a timestamped fallback).
    """
    raw = payload.get("workspace_path")
    if raw:
        return Path(raw)
    suffix = str(payload.get("task_id") or payload.get("request_id") or "").strip()
    if not suffix:
        suffix = f"request_{int(time.time())}"
    return Path(session_dir) / "runs" / "gemm_tuning" / suffix
