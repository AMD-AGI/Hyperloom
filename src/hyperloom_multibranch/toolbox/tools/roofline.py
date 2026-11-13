# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``roofline`` tool (the kernel roofline of the trace a profile test left) and the ``tracelens`` tool (analysis of a
given trace).

Both read a trace that already exists: ``roofline`` runs the orchestrator's ``non-llm`` analysis, which reads the trace in
memory and writes ``kernel_roofline.json`` (every hot kernel with its category, share of the GPU time and bound), with no
model session; ``tracelens`` runs TraceLens' report. The roofline of the orchestrator's own flow profiles first, on a card,
in the process that runs it; a toolbox does not offer that.

The analysis itself needs no card. Its roofline needs the board's achievable peaks, though, and when TraceLens has no arch
spec for the board it measures them with a microbenchmark on a GPU. That card is leased from the pool the measurement tests
run on (:class:`_ArchCard`), so no test is placed on it while the microbenchmark runs.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.framework_registry import is_scriptable
from hyperloom.orchestrator.actions.executors.trace_analyze import ARCH_GPUS_PARAM, trace_analyze_handler
from hyperloom.orchestrator.trace_analysis.analysis_route import AnalysisRoute
from hyperloom.orchestrator.trace_analysis.tracelens_arch_benchmark import (
    build_hyperloom_arch_spec,
    resolve_arch_json_path,
)
from hyperloom.sandbox.socket import SandboxUnavailable
from hyperloom_multibranch.contracts.blackboard.vocab import LeaseKind
from hyperloom_multibranch.contracts.toolbox import BranchLease, LeaseCharge, RequestRejected
from hyperloom_multibranch.toolbox.branch_leases import BranchLeases
from hyperloom_multibranch.toolbox.tools.base import ToolOutput, ToolRequest
from hyperloom_multibranch.toolbox.tools.executor_tool import ScratchHandlerTool
from hyperloom_multibranch.toolbox.tools.op_shares import with_op_shares
from hyperloom_multibranch.toolbox.tools.tracelens_report import (
    DEFAULT_PLATFORM,
    REPORT_ROUTE,
    report_handler,
    unusable_trace,
    with_window_shares,
)

ROOFLINE_TOOL = "roofline"
TRACELENS_TOOL = "tracelens"
#: The only route of the roofline: the orchestrator's analysis with no model session (``agent`` starts one on the host).
ROOFLINE_ROUTE = AnalysisRoute.NON_LLM.value
#: The error class of a roofline that answered ``ok`` with no hot kernel.
ROOFLINE_WITHOUT_KERNELS = "roofline_without_kernels"
HOT_KERNELS_KEY = "hot_kernels"
FAILED = "failed"
#: The error class of a roofline asked for a GPU neither TraceLens nor Hyperloom has peaks for.
UNKNOWN_GPU_ARCH = "unknown_gpu_arch"
#: The holder prefix of a tool job's lease, as ``kernel_tune``'s.
_HOLDER_PREFIX = "tool:"
_ARCH_CARDS = 1


class _SharesOps:
    """Mixin: the tool's pointer carries the analysed profile's per-op shares, when the request names its commit."""

    def output(self, result: Mapping[str, Any], request: ToolRequest) -> ToolOutput:
        return with_op_shares(super().output(result, request), request)  # type: ignore[misc]


class _ArchCard:
    """Mixin: an analysis that would measure the board's arch spec does so on one card leased for it, and on no other.

    The card is leased from ``leases``, the pool the measurement tests run on, held by ``tool:<job id>`` and charged to the
    requesting branch, and released once the analysis returns. The lease is pausing (``lease_pausing``): it waits for a
    card however long the running tests take, and when the tests need every card no new test starts until it is
    released. The analysis is
    told the card in ``arch_gpus``, which the caller cannot set. With no pool (a toolbox whose cards it does not own), or
    a lease refused as the toolbox closes, it is given none, and its roofline reads vendor peaks. A board with an arch
    spec, or a route that never measures, takes no card.
    """

    name: str
    _leases: BranchLeases | None

    def _measures(self, params: Mapping[str, Any]) -> bool:
        return resolve_arch_json_path(_platform(params)) is None

    async def run(self, request: ToolRequest, workdir: Path) -> ToolOutput:
        params = {key: value for key, value in request.params.items() if key != ARCH_GPUS_PARAM}
        if not self._measures(params):
            return await super().run(replace(request, params=params), workdir)  # type: ignore[misc]
        lease = await self._arch_lease(request.branch, workdir)
        try:
            params[ARCH_GPUS_PARAM] = list(lease.gpu_ids) if lease is not None else []
            return await super().run(replace(request, params=params), workdir)  # type: ignore[misc]
        finally:
            if lease is not None and self._leases is not None:
                self._leases.release(lease.lease_id)

    async def _arch_lease(self, branch: str, workdir: Path) -> BranchLease | None:
        if self._leases is None:
            return None
        charge = LeaseCharge(LeaseKind.TOOL_JOB, branch, self.name)
        try:
            return await self._leases.lease_pausing(
                f"{_HOLDER_PREFIX}{workdir.name}", _ARCH_CARDS, workdir, charge=charge
            )
        except (RequestRejected, SandboxUnavailable):
            return None


class _RooflineTool(_ArchCard, ScratchHandlerTool):
    def __init__(self, leases: BranchLeases | None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._leases = leases


class _TracelensTool(_ArchCard, _SharesOps, ScratchHandlerTool):
    def __init__(self, leases: BranchLeases | None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._leases = leases

    def _measures(self, params: Mapping[str, Any]) -> bool:
        return _tracelens_route(params).lower() != REPORT_ROUTE and super()._measures(params)


def _platform(params: Mapping[str, Any]) -> str:
    return str(params.get("target_platform") or "").strip() or DEFAULT_PLATFORM


async def roofline_handler(payload: dict, *, session_dir: Path) -> dict:
    """The kernel roofline of ``payload['trace_input']``: ``non-llm`` analysis for the platform the request names (else
    :data:`~.tracelens_report.DEFAULT_PLATFORM`). Any other ``analysis_route`` fails the job by name instead of being
    ignored, and the route the host's environment names is not read: it could start a model session. A trace that is missing or
    holds no GPU kernel fails by name: the analysis itself answers ``ok`` with no kernel for both. So does an analysis that
    answers ``ok`` with no hot kernel for a trace that has some (it scanned none of its events)."""
    route = str(payload.get("analysis_route") or "").strip().lower()
    if route and route != ROOFLINE_ROUTE:
        return {
            "status": FAILED,
            "error_class": "invalid_analysis_route",
            "error": f"the roofline runs only the {ROOFLINE_ROUTE!r} route, not {route!r}",
        }
    unusable = await asyncio.to_thread(unusable_trace, str(payload["trace_input"]))
    if unusable is not None:
        return unusable
    platform = _platform(payload)
    unknown = _unknown_arch(platform)
    if unknown is not None:
        return unknown
    result = await trace_analyze_handler(
        {**payload, "analysis_route": ROOFLINE_ROUTE, "target_platform": platform}, session_dir=session_dir
    )
    return _without_kernels(result) or result


def _unknown_arch(platform: str) -> dict[str, str] | None:
    """The failed result of a GPU TraceLens' perf report has no peaks for; ``None`` when it has some. The analysis gives
    TraceLens the arch TraceLens bundles, else the one its microbenchmark measures on the leased card, else Hyperloom's
    vendor peaks (built from ``hyperloom_common.gpu_identity.AMD_GPU_BOARDS``): with no bundled spec and no vendor peaks,
    a failed measurement leaves the report nothing to read."""
    if resolve_arch_json_path(platform) is not None or build_hyperloom_arch_spec(platform) is not None:
        return None
    return {
        "status": FAILED,
        "error_class": UNKNOWN_GPU_ARCH,
        "error": f"no peak FLOPS or memory bandwidth is known for the GPU {platform!r}: name a GPU TraceLens or Hyperloom "
        "has peaks for in target_platform",
    }


def _without_kernels(result: Mapping[str, Any]) -> dict[str, Any] | None:
    """The failed result of an ``ok`` analysis that found no hot kernel in a trace that has GPU kernels; ``None`` otherwise."""
    if result.get("status") != "ok" or result.get(HOT_KERNELS_KEY):
        return None
    warnings = [str(w.get("code") or "") for w in result.get("trace_health_warnings") or () if isinstance(w, Mapping)]
    return {
        **result,
        "status": FAILED,
        "error_class": ROOFLINE_WITHOUT_KERNELS,
        "error": (
            "the roofline analysis answered ok with no hot kernel, though the trace has GPU kernels: it scanned none of "
            f"their events (trace health warnings: {', '.join(filter(None, warnings)) or 'none'})"
        ),
    }


def roofline_tool(leases: BranchLeases | None = None) -> ScratchHandlerTool:
    """Analyse the roofline of a torch trace that already exists. Params: ``trace_input`` (a trace directory or file),
    ``framework``, ``model_name`` and ``target_platform`` (a GPU name such as MI355X, the default). Its pointer carries no op
    shares (they come from the TraceLens report of ``tracelens``); the kernels are in the result. ``leases`` is the pool a
    card to measure the board's arch spec on is leased from (:class:`_ArchCard`)."""
    return _RooflineTool(
        leases,
        ROOFLINE_TOOL,
        roofline_handler,
        ok_text="roofline analysed",
        required=("trace_input",),
    )


async def tracelens_handler(payload: dict, *, session_dir: Path) -> dict:
    """``analysis_route`` (else ``HYPERLOOM_TRACE_ANALYSIS_ROUTE``) picks the analysis: ``report`` is TraceLens' deterministic
    report with no agent, ``agent`` and ``non-llm`` are the orchestrator's own routes. Unnamed, a serving framework gets
    ``report`` and a scriptable one (xDiT, custom: no steady-state window to split) the orchestrator's default."""
    if _tracelens_route(payload).lower() == REPORT_ROUTE:
        return await report_handler(payload, session_dir=session_dir)
    result = await trace_analyze_handler(payload, session_dir=session_dir)
    return await asyncio.to_thread(with_window_shares, result, str(payload.get("trace_input") or ""), session_dir)


def _tracelens_route(payload: Mapping[str, Any]) -> str:
    """The route :func:`tracelens_handler` takes for ``payload``; empty is the orchestrator's default."""
    raw = payload.get("analysis_route")
    route = ("" if raw is None else str(raw).strip()) or os.environ.get("HYPERLOOM_TRACE_ANALYSIS_ROUTE", "").strip()
    if not route and not is_scriptable(payload.get("framework")):
        return REPORT_ROUTE
    return route


def tracelens_tool(leases: BranchLeases | None = None) -> ScratchHandlerTool:
    """Analyse a torch trace that already exists. Params: ``trace_input`` (a trace directory or file), optional
    ``analysis_route`` (``report`` by default, ``agent``, ``non-llm``), ``framework`` and ``model_name`` (the orchestrator's
    routes); ``commit_id`` (the commit the trace was profiled on) and, when the trace came from a PROFILE test,
    ``profile_request_id``, which the pointer carries with the op shares. ``leases`` is the pool a card to measure the
    board's arch spec on is leased from, for the orchestrator's routes (:class:`_ArchCard`)."""
    return _TracelensTool(
        leases,
        TRACELENS_TOOL,
        tracelens_handler,
        ok_text="trace analysed",
        required=("trace_input",),
    )
