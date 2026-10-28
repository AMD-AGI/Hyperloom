# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""One GEMM-tuning run: the input it is given, the CLI call, and the CSVs it leaves behind."""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, NamedTuple

from hyperloom.common.coerce import optional_positive_int
from kernelforge.gemm_tune import RoutingEvidence, RoutingTarget

from . import forge_gemm_tuning
from ..actions.executors._kernel_agent_tool import HandlerResult, _run_subprocess, _shape_tool_result
from ._aiter_tuner_support import _align_forge_shapes_for_aiter, _write_fmoe_untuned_csv_from_log
from ._forge_gemm_probe import (
    _derive_gemm_skip_reason,
    _forge_gemm_tune_available,
    _parse_forge_gemm_sentinel,
    _read_forge_result_json,
    _resolve_aiter_root_for_forge,
    _resolve_forge_precision_and_quant,
)
from ._forge_lane_budget import (
    _gemm_tuning_timeout_sec,
    _gemm_tuning_workspace,
    _priced_gemm_lane,
    _unfunded_lane_skip,
)
from ._forge_shapes import (
    _normalize_forge_shapes_json,
    _normalize_tokens,
    _path_is_existing_file,
    _resolve_forge_shapes,
    _resolve_trace_shape_manifest,
)
from ._integrate_payload import _requested_framework, _requested_setting
from .serving_log_evidence import resolve_forge_server_log, _tokens_from_serving_log
from ._shape_capture import (
    SuppliedShapes,
    _capture_vllm_tunableop_shapes,
    _reuse_vllm_block_fp8_roofline_shapes,
    _vllm_block_fp8_profile_capture_required,
    _vllm_dense_shape_capture_required,
)
from ._untuned_gemm_csv import _resolve_forge_untuned_csv
from ._vllm_aiter_routing import _forge_framework_for_vllm, _resolve_vllm_aiter_routing
from .lane_budget import LANE_GEMM, gemm_per_tuner_timeout_sec

log = logging.getLogger(__name__)


async def _invoke_forge_gemm_tune(input_payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    """Run ``kernelforge gemm-tune`` for ``input_payload`` and return its result, or a failed result."""
    try:
        cmd = forge_gemm_tuning.build_command(input_payload)
    except ValueError as exc:
        return {
            "status": "failed",
            "micro_decision": "failed",
            "error_class": exc.__class__.__name__,
            "error": repr(exc),
        }
    aiter_root = _resolve_aiter_root_for_forge()
    if aiter_root:
        cmd = ["env", f"AITER_ROOT_DIR={aiter_root}", *cmd]
    try:
        rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout, label="forge_gemm_tuning")
    except subprocess.TimeoutExpired as exc:
        # Reaped by the process-group kill in _run_subprocess; shape a failed result.
        cmd_repr = " ".join(str(c) for c in (getattr(exc, "cmd", None) or cmd))
        return {
            "status": "failed",
            "error_class": "subprocess_timeout",
            "error": f"TimeoutExpired after {timeout}s: {cmd_repr[:1500]}",
        }
    result = _parse_forge_gemm_sentinel(stdout)
    return result if result is not None else _shape_tool_result(rc, stdout, stderr)


def _bridge_forge_decision(
    result: dict[str, Any],
    *,
    raw_model_path: str,
    session_dir: Path,
) -> None:
    micro = str(result.get("micro_decision") or "").strip().lower()
    if micro == "candidate" and result.get("recommended_env"):
        result.setdefault("decision", "KEEP")
        # Make the tuned CSV durable + recipe-portable (mirrors integrate_patch's
        # source-layer snapshot): copy it into the serving aiter config dir,
        # repoint the env there, and snapshot it so the KEEP survives with the
        # recipe instead of referencing the ephemeral tuner-workspace path.
        # Keep the logical ID here: a resolved HF snapshot basename is a commit
        # hash, which would make durable artifact names unstable across revisions.
        _durable_envs, _snap_dir = _persist_forge_gemm_csv_durably(
            dict(result["recommended_env"]),
            model_path=raw_model_path,
            session_dir=session_dir,
        )
        result.setdefault("extra_envs", _durable_envs)
        if _snap_dir:
            result.setdefault("source_snapshot", _snap_dir)
        # Derive best_speedup from tuners_run when absent.
        if "best_speedup" not in result:
            best = 1.0
            for t in result.get("tuners_run") or []:
                if isinstance(t, dict):
                    sp = float(t.get("best_micro_speedup") or 1.0)
                    if sp > best:
                        best = sp
            if best > 1.0:
                result["best_speedup"] = best
        # Micro-only result: E2E validation still needed.
        result.setdefault("requires_e2e_validation", True)
    elif micro in ("no_improvement", "skipped"):
        # Left unadorned on purpose: the wordings below are only legible against it.
        result.setdefault("decision", "REVERT")
    elif micro in _FORGE_BARREN_MICRO_DECISIONS:
        result.setdefault("decision", "REVERT")
        if micro == "failed":
            result.setdefault("status", "failed")
        result.setdefault("error_class", f"forge_{micro}")


#: forge wordings that carry nothing deployable, each distinct from an honest
#: ``no_improvement``. ``build_report`` checks ``has_candidate`` first, so a run
#: holding a usable env reports ``candidate`` even when a sibling crashed --
#: these arrive only with nothing to deploy.
_FORGE_BARREN_MICRO_DECISIONS = (
    "failed",
    "empty_output",
    "partial_failure",
    "partial_output",
)


class _ForgeGemmRun(NamedTuple):
    session_dir: Path
    workspace: Path
    framework: str
    tuning_framework: str
    precision: str
    quant_type: str
    gpu_type: str
    tp: int
    conc: int
    mp: int
    model_dir: str
    model_id: str
    server_log: str
    thorough: bool


class _ForgeShapeSources(NamedTuple):
    tokens: str
    shapes_json: str
    untuned_csv: str
    moe_untuned_csv: str
    moe_key_report: dict[str, Any]
    tunableop_input: str
    shapes_manifest: str
    demand_json: str
    capture: HandlerResult | None


def _requested_existing_file(payload: dict, key: str) -> str:
    path = str(payload.get(key) or "").strip()
    return path if _path_is_existing_file(path) else ""


def _forge_gemm_run(state: Any, payload: dict, session_dir: Path) -> _ForgeGemmRun:
    from hyperloom.common.model_paths import resolve_local_model_dir, resolve_serving_model_path
    from hyperloom.common.visible_devices import detect_gpu_count

    workspace = _gemm_tuning_workspace(payload, session_dir=session_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    # Resolve precision from actual runtime, not just session-level state.
    precision, quant_type = _resolve_forge_precision_and_quant(state, payload)

    # Bootstrap already walked HL_MODEL_BASE and the hub cache to decide what to
    # serve; probing only the hub cache here would reject a repo id that the
    # running server resolved fine.
    model_id = _requested_setting(payload, state, "model_path", os.environ.get("MODEL_PATH") or "")
    model_dir = resolve_local_model_dir(resolve_serving_model_path(model_id) or model_id) if model_id else None

    tp = int(payload.get("tp") or state.tp or os.environ.get("TP") or 1)
    # Default mp = all visible GPUs.
    detected_gpus = detect_gpu_count() or tp
    mp = int(payload.get("mp") or os.environ.get("FORGE_GEMM_TUNE_MP") or detected_gpus)
    return _ForgeGemmRun(
        session_dir=session_dir,
        workspace=workspace,
        framework=_requested_framework(payload, state, "sglang"),
        tuning_framework="",
        precision=precision,
        quant_type=quant_type,
        gpu_type=_requested_setting(payload, state, "gpu_type", os.environ.get("GPU_TYPE") or "mi300x").lower(),
        tp=tp,
        conc=int(payload.get("conc") or state.conc or os.environ.get("CONC") or 64),
        mp=mp,
        model_dir=str(model_dir) if model_dir is not None else "",
        model_id=model_id,
        server_log="",
        # Exhaustive search when budget allows (>= 24h) and mp >= 4.
        thorough=bool(float(getattr(state, "max_minutes", 0) or 0) >= 1440 and mp >= 4),
    )


async def _forge_shape_sources(state: Any, payload: dict, run: _ForgeGemmRun) -> _ForgeShapeSources:
    tokens = _normalize_tokens(payload.get("tokens"))
    if not tokens and run.server_log:
        tokens = _normalize_tokens(await asyncio.to_thread(_tokens_from_serving_log, run.server_log))
        if tokens:
            log.info("GEMM: derived --tokens=%s from observed M in %s", tokens, run.server_log)

    moe_untuned_csv = _requested_existing_file(payload, "moe_untuned_csv")
    moe_key_report: dict[str, Any] = {}
    if not moe_untuned_csv:
        moe_untuned_csv, moe_key_report = _write_fmoe_untuned_csv_from_log(run.server_log, tokens, run.workspace)

    shapes_json = _normalize_forge_shapes_json(payload.get("shapes_json"), run.workspace)
    untuned_csv = _requested_existing_file(payload, "untuned_csv")
    if not shapes_json and not untuned_csv and run.framework != "vllm":
        shapes_json = _resolve_forge_shapes(state, run.session_dir, require_fresh_profile=True, precision=run.precision)
        if not shapes_json:
            untuned_csv = _resolve_forge_untuned_csv(run.session_dir, run.precision, run.quant_type, run.model_dir)

    return _ForgeShapeSources(
        tokens=tokens,
        shapes_json=shapes_json,
        untuned_csv=untuned_csv,
        moe_untuned_csv=moe_untuned_csv,
        moe_key_report=moe_key_report,
        tunableop_input=str(payload.get("tunableop_input") or "").strip(),
        shapes_manifest="",
        demand_json=_requested_existing_file(payload, "demand_json"),
        capture=None,
    )


def _profiled_block_fp8_shapes(
    state: Any,
    payload: dict,
    run: _ForgeGemmRun,
    sources: _ForgeShapeSources,
) -> _ForgeShapeSources:
    if not _vllm_block_fp8_profile_capture_required(
        framework=run.framework,
        precision=run.precision,
        quant_type=run.quant_type,
        supplied=SuppliedShapes(shapes_json=sources.shapes_json, tunableop_input=sources.tunableop_input),
        dry_run=bool(payload.get("dry_run")),
    ):
        return sources
    traced_shapes = _resolve_forge_shapes(state, run.session_dir, require_fresh_profile=True, precision=run.precision)
    if traced_shapes:
        return sources._replace(shapes_json=traced_shapes, untuned_csv="")
    reused = _reuse_vllm_block_fp8_roofline_shapes(
        state,
        workspace=run.workspace,
        current_workload=state.current_profile_workload_context(payload),
    )
    if reused is None:
        return sources
    return sources._replace(shapes_json=str(reused["shapes_json"]), untuned_csv="", capture=reused)


async def _captured_forge_shapes(
    state: Any,
    payload: dict,
    run: _ForgeGemmRun,
    sources: _ForgeShapeSources,
    priced_lane: Any,
) -> tuple[_ForgeShapeSources, HandlerResult | None]:
    sources = _profiled_block_fp8_shapes(state, payload, run, sources)
    supplied = SuppliedShapes(shapes_json=sources.shapes_json, tunableop_input=sources.tunableop_input)
    block_fp8_capture = _vllm_block_fp8_profile_capture_required(
        framework=run.framework,
        precision=run.precision,
        quant_type=run.quant_type,
        supplied=supplied,
        dry_run=bool(payload.get("dry_run")),
    )
    # Keyed on the routed framework: a run handed to the AITER tuner family has
    # no use for a TunableOp recording pass, and paying for one costs a full
    # extra server boot.
    tunableop_capture = (
        _vllm_dense_shape_capture_required(
            framework=run.tuning_framework,
            model_path=run.model_dir,
            supplied=supplied,
            dry_run=bool(payload.get("dry_run")),
        )
        and not block_fp8_capture
    )
    if not block_fp8_capture and not tunableop_capture:
        return sources, None
    # A capture costs a full extra server boot, wasted on a lane the phase cannot fund once it has the shapes.
    lane_after_capture = await priced_lane(
        evidence=RoutingEvidence(
            kernel_signature_log=run.server_log or None,
            has_untuned_csv=tunableop_capture and bool(sources.untuned_csv),
            has_shapes_json=block_fp8_capture,
            has_tunableop_input=tunableop_capture,
        )
    )
    if lane_after_capture is None:
        return sources, _unfunded_lane_skip(LANE_GEMM)

    capture_payload = dict(payload)
    if block_fp8_capture:
        capture_payload["_shape_capture_mode"] = "block_fp8_profile"
    capture = await _capture_vllm_tunableop_shapes(
        state=state,
        session_dir=run.session_dir,
        payload=capture_payload,
        workspace=run.workspace,
    )
    if capture.get("status") != "ok":
        return sources, _stamp_forge_gemm_provenance(capture, run)
    captured = sources._replace(
        tunableop_input=str(capture.get("tunableop_input") or "").strip(),
        capture=capture,
    )
    captured_shapes = str(capture.get("shapes_json") or "").strip()
    if not captured_shapes:
        return captured, None
    # Forge dense tuners prefer untuned_csv over shapes_json. A fresh profile
    # capture is workload-matched and must supersede any stale specialist CSV
    # resolved before the capture pass.
    return captured._replace(shapes_json=captured_shapes, untuned_csv=""), None


async def _forge_dense_shape_manifest(
    state: Any,
    payload: dict,
    run: _ForgeGemmRun,
    sources: _ForgeShapeSources,
) -> str:
    requested = str(payload.get("shapes_manifest") or "").strip()
    if requested:
        return requested if _path_is_existing_file(requested) else ""
    if sources.untuned_csv or sources.shapes_json:
        log.debug(
            "GEMM: not scavenging a trace shape manifest; this run already has "
            "a workload-matched dense-shape source (%s)",
            "untuned_csv" if sources.untuned_csv else "shapes_json",
        )
        return ""
    # Off the event loop for the same reason: a ``**/`` walk of a session tree
    # that holds thousands of run artifacts.
    scavenged = await asyncio.to_thread(_resolve_trace_shape_manifest, state, run.session_dir)
    return scavenged if _path_is_existing_file(scavenged) else ""


def _forge_gemm_tune_input(
    payload: dict,
    run: _ForgeGemmRun,
    sources: _ForgeShapeSources,
    timeout: int,
    tuner_ceiling: int,
) -> dict[str, Any]:
    return {
        "model_path": run.model_dir,
        "framework": run.tuning_framework,
        "precision": run.precision,
        "quant_type": run.quant_type,
        "gpu_type": run.gpu_type,
        "tp": run.tp,
        "conc": run.conc,
        "mp": run.mp,
        "output_dir": str(run.workspace),
        # Passing the same value to both made the producer's own
        # min(per_tuner, remaining) an identity, so the first tuner could
        # consume the entire session and every later one was skipped for lack of
        # time. The per-target cap must stay strictly below the global one.
        "timeout": gemm_per_tuner_timeout_sec(timeout),
        # Bounds the whole session across all tuners.
        "global_timeout": timeout,
        "skip_gpu_check": True,
        "tokens": sources.tokens,
        "untuned_csv": sources.untuned_csv,
        "moe_untuned_csv": sources.moe_untuned_csv,
        "shapes_json": sources.shapes_json,
        "shapes_manifest": sources.shapes_manifest,
        "demand_json": sources.demand_json,
        "tunableop_input": sources.tunableop_input,
        "kernel_signature_log": run.server_log,
        "tuner": str(payload.get("tuner") or ""),
        # How many routed tuners the lane's share pays for. Omitted when none
        # could be derived, which leaves the producer's own routing intact.
        **({"max_tuners": tuner_ceiling} if tuner_ceiling > 0 else {}),
        "thorough": run.thorough,
    }


def _stamp_forge_gemm_provenance(
    result: HandlerResult,
    run: _ForgeGemmRun,
    tuning_framework: str = "",
) -> HandlerResult:
    result.setdefault("backend", "forge")
    result.setdefault("engine", "forge")
    result.setdefault("workspace", str(run.workspace))
    result.setdefault("precision", run.precision)
    result.setdefault("framework", run.framework)
    if tuning_framework:
        result.setdefault("tuning_framework", tuning_framework)
    result.setdefault("model_path", run.model_id)
    return result


def _report_forge_shape_provenance(
    result: HandlerResult,
    sources: _ForgeShapeSources,
    shape_alignment: dict[str, Any] | None,
) -> None:
    if sources.moe_key_report:
        result.setdefault("moe_key_source", sources.moe_key_report)
    if shape_alignment is not None:
        result.setdefault("shape_alignment", shape_alignment)
    if sources.capture is None:
        return
    result.setdefault(
        "shape_capture",
        {
            "status": "ok",
            "workspace": sources.capture.get("shape_capture_workspace"),
            "tunableop_input": sources.tunableop_input,
            "shapes_json": sources.shapes_json,
            "shape_count": sources.capture.get("shape_count"),
            "capture_mode": sources.capture.get("capture_mode", "tunableop"),
            "source_profile_trace": sources.capture.get("source_profile_trace"),
        },
    )


def _fill_forge_tuner_outcomes(result: HandlerResult, workspace: Path) -> None:
    on_disk = _read_forge_result_json(workspace)
    for key in ("tuners_skipped", "failed_tuners"):
        if not result.get(key) and on_disk.get(key):
            result[key] = on_disk[key]
    if not result.get("skip_reason"):
        reason = _derive_gemm_skip_reason(result.get("tuners_skipped"))
        if reason:
            result["skip_reason"] = reason
    failed = result.get("failed_tuners")
    for row in failed if isinstance(failed, list) else []:
        if isinstance(row, dict):
            log.warning(
                "forge gemm tuner %s failed (%s): %s",
                row.get("tuner") or "?",
                row.get("error_class") or "?",
                row.get("error") or "",
            )
    rows = result.get("tuners_run")
    rows = rows if isinstance(rows, list) else []
    for key in ("error_class", "error"):
        if result.get(key):
            continue
        reported = next((str(row[key]) for row in rows if isinstance(row, dict) and row.get(key)), "")
        if reported:
            result[key] = reported


async def _run_forge_gemm_tuning(
    payload: dict,
    *,
    session_dir: Path,
) -> HandlerResult:
    """Deterministic GEMM tuning via the ``kernelforge gemm-tune`` CLI.

    Supports bf16/fp8/fp4 + sglang/vllm. Only micro-benchmarks;
    returns recommended_env for Hyperloom E2E validation.

    ``model_path`` accepts either a local directory or a Hugging Face repo ID.
    Forge receives a validated local directory, while result provenance and
    durable artifact names retain the original logical model identifier.
    Missing inputs return ``model_path_missing``; inputs that cannot resolve to
    a local directory return ``model_path_unavailable`` as a ``skipped`` result,
    because forge never ran and so has no verdict to report.
    """
    from ..state.shared_state import SharedState

    state = SharedState.load_or_init(session_dir)

    # Importing kernelforge.cli is deliberately isolated in a subprocess, but
    # that subprocess may still take until the bounded timeout to fail. Keep the
    # synchronous probe off the orchestrator reactor.
    if not await asyncio.to_thread(_forge_gemm_tune_available):
        return {
            "status": "failed",
            "error_class": "forge_gemm_tune_not_found",
            "error": (
                "forge-gemm-tune is not runnable in this interpreter: "
                f"'{sys.executable} -m kernelforge.cli gemm-tune --help' failed. "
                "kernelforge ships with this distribution, so this means a "
                "partial install: reinstall with 'pip install -e .[forge]'."
                f" (interpreter: {sys.executable!r})"
            ),
            "backend": "forge",
        }

    run = _forge_gemm_run(state, payload, session_dir)
    if not run.model_id:
        return {"status": "failed", "error_class": "model_path_missing", "error": "model_path is required"}
    if not run.model_dir:
        # Forge needs the config on disk to derive shapes, so it cannot run --
        # but not running one tuning backend is a skip, not a session failure.
        # Reporting it as failed would spend a REVERT verdict on an experiment
        # that never started.
        return {
            "status": "skipped",
            "error_class": "model_path_unavailable",
            "skip_reason": (
                f"Model path {run.model_id!r} is neither an existing local "
                "directory nor an available Hugging Face cache snapshot"
            ),
            "backend": "forge",
        }
    # Off the event loop: this walks runs/ and byte-scans server logs that measure
    # ~17MB apiece on the fleet. Inline, it stalled every other coroutine on this
    # orchestrator -- heartbeats included -- for the duration.
    server_log = str(payload.get("kernel_signature_log") or "").strip() or await asyncio.to_thread(
        resolve_forge_server_log, state, session_dir
    )
    run = run._replace(
        server_log=server_log,
        tuning_framework=_forge_framework_for_vllm(
            framework=run.framework,
            precision=run.precision,
            quant_type=run.quant_type,
            tunableop_input=str(payload.get("tunableop_input") or "").strip(),
            aiter_routing=_resolve_vllm_aiter_routing(model_path=run.model_dir, server_log=server_log, tp=run.tp),
        ),
    )
    priced_gemm_lane = functools.partial(
        _priced_gemm_lane,
        state,
        model_path=run.model_dir,
        target=RoutingTarget(
            framework=run.tuning_framework,
            precision=run.precision,
            quant_type=run.quant_type,
            gpu_type=run.gpu_type,
        ),
    )

    sources = await _forge_shape_sources(state, payload, run)
    sources, capture_failure = await _captured_forge_shapes(state, payload, run, sources, priced_gemm_lane)
    if capture_failure is not None:
        return capture_failure
    sources = sources._replace(shapes_manifest=await _forge_dense_shape_manifest(state, payload, run, sources))

    gemm_lane = await priced_gemm_lane(
        evidence=RoutingEvidence(
            kernel_signature_log=run.server_log or None,
            has_untuned_csv=bool(sources.untuned_csv),
            has_shapes_json=bool(sources.shapes_json or sources.shapes_manifest or sources.demand_json),
            has_tunableop_input=bool(sources.tunableop_input),
        )
    )
    if gemm_lane is None:
        return _unfunded_lane_skip(LANE_GEMM)
    timeout = _gemm_tuning_timeout_sec(payload, lane_budget_sec=gemm_lane.budget_sec)
    shape_alignment: dict[str, Any] | None = None
    if sources.shapes_json:
        aligned_shapes, shape_alignment = _align_forge_shapes_for_aiter(
            sources.shapes_json,
            forge_framework=run.tuning_framework,
            workspace=run.workspace,
            budget_sec=timeout,
            mp=run.mp,
        )
        sources = sources._replace(shapes_json=aligned_shapes)

    requested_tuners = optional_positive_int(payload.get("max_tuners"), 0)
    tuner_ceiling = 0 if str(payload.get("tuner") or "").strip() else (requested_tuners or gemm_lane.max_targets)
    result = await _invoke_forge_gemm_tune(
        _forge_gemm_tune_input(payload, run, sources, timeout, tuner_ceiling), timeout
    )
    _stamp_forge_gemm_provenance(result, run, tuning_framework=run.tuning_framework)
    _report_forge_shape_provenance(result, sources, shape_alignment)
    _fill_forge_tuner_outcomes(result, run.workspace)
    _bridge_forge_decision(result, raw_model_path=run.model_id, session_dir=run.session_dir)
    return result


def _persist_forge_gemm_csv_durably(extra_envs: dict, *, model_path: str, session_dir: Path) -> tuple[dict, str]:
    """Make forge GEMM tuned CSVs durable + recipe-portable.

    The forge KEEP references tuned CSVs by their ephemeral tuner-workspace paths,
    so a recipe replayed after the workspace is gone (or on another box) loses the
    tuning and aiter falls back to its default config. Mirror integrate_patch's
    durability: copy each CSV into the serving aiter config tree, repoint the env
    there, and snapshot the realized files via :func:`snapshot_source_layer` so
    they travel with the recipe.

    The copy lands one level below ``configs/model_configs/`` on purpose. aiter
    merges every ``model_configs/*{table}*.csv`` it can glob whenever the env var
    is unset, and that glob is not recursive. Writing directly into that
    directory would hand the table to every later server start -- including after
    E2E rejected the candidate, and including servers for other models, since the
    scan does not discriminate by model. Replay does not need the scan: it
    restores the env var explicitly (see ``prelude._warm_kernel_extra_envs``) and
    defers a GEMM column that has no env at all.

    The snapshot lands under ``<session_dir>/optimization_stack/src/`` (the same
    durable, run-cleanup-surviving location integrate_patch uses) -- NOT under the
    ephemeral ``runs/gemm_tuning`` workspace, which would be cleaned away and
    defeat the cross-environment recipe-portability this exists for.

    A failed durable copy raises; a failed snapshot keeps the copy and returns no
    snapshot dir. Returns ``(extra_envs, source_snapshot_dir)``.
    """
    # Below model_configs/, out of reach of aiter's non-recursive auto-merge glob.
    _FORGE_DURABLE_SUBDIR = "hyperloom"
    _forge_durable_env_stems = {
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE": "a8w8_blockscale_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "a8w8_blockscale_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE": "a8w8_bpreshuffle_tuned_gemm",
        "AITER_CONFIG_GEMM_A8W8": "a8w8_tuned_gemm",
        "AITER_CONFIG_GEMM_A4W4": "a4w4_blockscale_tuned_gemm",
        "AITER_CONFIG_GEMM_BF16": "bf16_tuned_gemm",
        "AITER_CONFIG_FMOE": "tuned_fmoe",
    }
    slug = (
        "".join(c if (c.isalnum() or c in "._-") else "_" for c in Path(model_path).name).strip("_").lower() or "model"
    )

    pending: list[tuple[str, str, Path]] = []
    for env_key, stem in _forge_durable_env_stems.items():
        src_csv = str(extra_envs.get(env_key) or "").strip()
        if not src_csv or not Path(src_csv).is_file():
            continue
        rel = f"configs/model_configs/{_FORGE_DURABLE_SUBDIR}/{stem}_{slug}.csv"
        pending.append((env_key, rel, Path(src_csv)))
    if not pending:
        return extra_envs, ""

    # Step 1 -- commit durable copies + env repoints. This is what makes the
    # KEEP survive: each CSV lands in aiter's config dir and the env points
    # there instead of the ephemeral tuner workspace.
    import importlib.util

    spec = importlib.util.find_spec("aiter")
    if spec is None or not spec.origin:
        return extra_envs, ""
    aiter_pkg = Path(spec.origin).resolve().parent
    updated = dict(extra_envs)
    rel_paths: list[str] = []
    for env_key, rel, src_path in pending:
        dst = aiter_pkg / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst)
        updated[env_key] = str(dst)
        rel_paths.append(rel)

    # Step 2 -- recipe-portability snapshot. Separate best-effort concern: a
    # snapshot failure must NOT discard the copy + repoint committed above.
    snap_dir = ""
    try:
        from ..source_snapshot import SourceLayer, snapshot_source_layer

        snap = snapshot_source_layer(
            SourceLayer(
                framework_root=aiter_pkg,
                base_sha=None,
                rel_paths=rel_paths,
            ),
            dest_dir=Path(session_dir) / "optimization_stack" / "src" / f"forge_gemm_{slug}",
            provenance="kernelforge.gemm_tune",
            extra={
                "env_keys": [env_key for env_key, _, _ in pending],
                "model": slug,
            },
        )
        snap_dir = str((snap or {}).get("snapshot_dir") or "")
    except OSError:
        log.exception("forge gemm CSV snapshot failed; durable copy + repoint kept")
    return updated, snap_dir
