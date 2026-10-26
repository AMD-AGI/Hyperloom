# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Serving the workload once for the sole purpose of recording the GEMM shapes it runs."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

from hyperloom.common.env import env_bool
from hyperloom.common.ports import pick_server_port
from hyperloom.orchestrator.trace_analysis._capture_shapes import is_capture_fragment as _shared_is_capture_fragment

from ..actions.executors._kernel_agent_tool import HandlerResult

log = logging.getLogger(__name__)


async def _served_capture_workload(
    executor: Any,
    run: _ShapeCaptureRun,
    capture_state: Any,
    task_params: dict[str, Any],
) -> tuple[dict[str, Any], HandlerResult | None]:
    from ..loop.sub_agent_runner import RunnerContext
    from ..state.task_registry import Task

    ctx = RunnerContext(
        task=Task(
            task_id=run.task_id,
            kind="gemm_shape_capture",
            state="running",
            params=task_params,
            idempotency_key=f"{run.task_id}-run",
        ),
        lease=None,
    )
    ctx.extra = {
        "shared_state": capture_state,
        "session_dir": run.session_dir,
        "workspace": run.capture_dir,
    }
    try:
        benchmark_result = await executor(ctx)
    except Exception as exc:  # per-run boundary: the fault becomes the run's recorded capture verdict
        log.warning("vLLM TunableOp shape capture raised", exc_info=True)
        return {}, _shape_capture_failure(f"vLLM TunableOp shape capture raised {exc!r}", run.capture_dir)
    return (benchmark_result if isinstance(benchmark_result, dict) else {}), None


async def _capture_block_fp8_profile_shapes(run: _ShapeCaptureRun, state: Any, payload: dict) -> HandlerResult:
    from ..actions.executors.roofline import RooflineExecutor

    task_params = _capture_task_params(run, state, payload)
    task_params["workspace_path"] = str(run.capture_dir / "tracelens")
    benchmark_script = state.accepted_baseline_script()
    if benchmark_script:
        task_params["benchmark_script"] = benchmark_script
    benchmark_result, fault = await _served_capture_workload(
        RooflineExecutor(shared_state=state), run, state, task_params
    )
    if fault is not None:
        return fault

    steady_state_trace = str(benchmark_result.get("steady_state_trace") or "").strip()
    shapes_json, shape_count = "", 0
    if steady_state_trace:
        shapes_json, shape_count = _extract_vllm_block_fp8_profile_shapes(
            Path(steady_state_trace),
            output_dir=run.capture_dir,
        )
    if benchmark_result.get("status") in {"succeeded", "recovered"} and shape_count > 0:
        return {
            "status": "ok",
            "shapes_json": shapes_json,
            "shape_capture_workspace": str(run.capture_dir),
            "shape_count": shape_count,
            "capture_mode": "block_fp8_profile",
            "source_profile_trace": steady_state_trace,
        }
    return _shape_capture_failure(
        f"vLLM block-FP8 profile capture produced no structured GEMM shapes{_benchmark_failure_detail(benchmark_result)}",
        run.capture_dir,
        shape_count=shape_count,
        capture_mode="block_fp8_profile",
    )


async def _capture_tunableop_recording(run: _ShapeCaptureRun, state: Any, payload: dict) -> HandlerResult:
    import copy

    from ..actions.executors.baseline import BaselineExecutor

    config_path = str(payload.get("config_path") or getattr(state, "baseline_config_path", "") or "").strip()
    if not config_path or not Path(config_path).is_file():
        return _shape_capture_failure(
            "vLLM TunableOp shape capture requires an existing baseline_config_path",
            run.capture_dir,
        )
    try:
        capture_port = _resolve_shape_capture_port(payload.get("shape_capture_port"))
    except ValueError as exc:
        return _shape_capture_failure(str(exc), run.capture_dir)

    try:
        timeout_sec = int(
            payload.get("shape_capture_timeout_sec")
            or os.environ.get("HYPERLOOM_GEMM_SHAPE_CAPTURE_TIMEOUT_SEC")
            or _DEFAULT_CAPTURE_TIMEOUT_SEC
        )
    except (TypeError, ValueError):
        timeout_sec = _DEFAULT_CAPTURE_TIMEOUT_SEC

    # unset_envs is applied after extra_envs, so any PYTORCH_TUNABLEOP_* name left here strips the recording.
    kept_unset = [name for name in run.unset_envs if not name.startswith("PYTORCH_TUNABLEOP_")]
    task_params = _capture_task_params(run, state, payload)
    task_params.update(
        {
            "extra_envs": _tunableop_recording_envs(run.envs, capture_port, run.capture_dir),
            "unset_envs": [*kept_unset, "HL_TUNABLEOP_MODE", "HL_TUNABLEOP_FILE", "HL_TUNABLEOP_VERBOSE"],
            "remove_args": [*run.remove_args, "--port"],
            "config_path": config_path,
            "timeout_sec": max(60, timeout_sec),
            "baseline_double_run": False,
        }
    )
    capture_state = copy.deepcopy(state)
    capture_state.baseline_eager_fallback = False
    benchmark_result, fault = await _served_capture_workload(
        BaselineExecutor(session_dir=run.session_dir, shared_state=capture_state),
        run,
        capture_state,
        task_params,
    )
    if fault is not None:
        return fault

    untuned_base = run.capture_dir / _TUNABLEOP_UNTUNED_NAME
    row_count = _merge_tunableop_untuned_files(untuned_base)
    if benchmark_result.get("status") == "succeeded" and row_count > 0:
        return {
            "status": "ok",
            "tunableop_input": str(untuned_base),
            "shape_capture_workspace": str(run.capture_dir),
            "shape_count": row_count,
        }
    try:
        untuned_base.unlink(missing_ok=True)
    except OSError:
        log.debug("shape capture: failed to remove incomplete TunableOp recording %s", untuned_base, exc_info=True)
    return _shape_capture_failure(
        f"vLLM TunableOp shape capture produced no complete workload recording"
        f"{_benchmark_failure_detail(benchmark_result)}",
        run.capture_dir,
        shape_count=row_count,
    )


#: Filenames PyTorch TunableOp reads and writes inside one capture directory.
_TUNABLEOP_UNTUNED_NAME = "tunableop_untuned.csv"

_TUNABLEOP_RESULTS_NAME = "tunableop_results.csv"

#: How long a recording pass may serve the workload when nothing else says.
_DEFAULT_CAPTURE_TIMEOUT_SEC = 1800

#: The vLLM operators whose profiled arguments carry a block-FP8 GEMM's shape.
_VLLM_BLOCK_FP8_TRACE_OPS = (
    "w8a8_triton_block_scaled_mm",
    "rocm_aiter_gemm_a8w8_blockscale",
    "rocm_aiter_triton_gemm_a8w8_blockscale",
)


def _is_vllm_block_fp8(precision: str, quant_type: str) -> bool:
    return precision == "fp8" and quant_type.strip().lower() in {
        "blockscale",
        "block_scale",
        "a8w8_blockscale",
        "fp8_blockscale",
    }


@dataclass(frozen=True)
class SuppliedShapes:
    """The GEMM shape inputs a forge run already holds, either of which makes a capture pass unnecessary."""

    shapes_json: str
    tunableop_input: str


def _vllm_block_fp8_profile_capture_required(
    *,
    framework: str,
    precision: str,
    quant_type: str,
    supplied: SuppliedShapes,
    dry_run: bool,
) -> bool:
    """Return whether block-FP8 needs a profiled runtime-shape capture pass."""
    if (
        framework != "vllm"
        or not _is_vllm_block_fp8(precision, quant_type)
        or dry_run
        or supplied.shapes_json
        or supplied.tunableop_input
        or not env_bool("HYPERLOOM_GEMM_SHAPE_CAPTURE", True)
    ):
        return False
    from ..actions.executors._multi_node_env import is_multi_node

    return not is_multi_node()


def _trace_event_block_fp8_shape(event: Any) -> tuple[int, int, int] | None:
    """Extract one (M, N, K) tuple from a profiled block-FP8 linear event."""
    if not isinstance(event, dict):
        return None
    name = str(event.get("name") or "").lower()
    if not any(marker in name for marker in _VLLM_BLOCK_FP8_TRACE_OPS):
        return None
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    dims = args.get("Input Dims") or args.get("Input dims") or args.get("input_shapes")
    if not isinstance(dims, list) or len(dims) < 2:
        return None
    a_dims, b_dims = dims[0], dims[1]
    if not isinstance(a_dims, list) or not isinstance(b_dims, list) or len(a_dims) < 2 or len(b_dims) < 2:
        return None
    try:
        m = int(a_dims[-2])
        k = int(a_dims[-1])
        if int(b_dims[-1]) == k:
            n = int(b_dims[-2])
        elif int(b_dims[-2]) == k:
            n = int(b_dims[-1])
        else:
            return None
    except (TypeError, ValueError):
        return None
    return (m, n, k) if min(m, n, k) > 0 else None


def _extract_vllm_block_fp8_profile_shapes(
    trace_input: Path,
    *,
    output_dir: Path | None = None,
) -> tuple[str, int]:
    """Convert Kineto block-FP8 events into Forge's structured shapes JSON."""
    import gzip

    def _is_capture_sidecar(path: Path) -> bool:
        # Shared classifier, so a layout the kernel-agent routes demote is also
        # kept out of the shape harvest here; the exact-``capture_traces`` test
        # this replaced missed ``graph_capture_profile/``.
        return _shared_is_capture_fragment(path, trace_input if trace_input.is_dir() else trace_input.parent)

    shapes: set[tuple[int, int, int]] = set()
    # ``Path("")`` normalizes to ``Path(".")``, which would otherwise walk the
    # whole process CWD and harvest shapes from unrelated traces.
    if str(trace_input) in ("", "."):
        return "", 0
    if trace_input.is_file():
        trace_paths = [] if _is_capture_sidecar(trace_input) else [trace_input]
    elif trace_input.is_dir():
        trace_paths = [
            path
            for path in sorted(trace_input.rglob("*.json")) + sorted(trace_input.rglob("*.json.gz"))
            if not _is_capture_sidecar(path)
        ]
    else:
        return "", 0
    for path in trace_paths:
        try:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
                    data = json.load(stream)
            else:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            continue
        events = data.get("traceEvents") if isinstance(data, dict) else None
        if not isinstance(events, list):
            continue
        for event in events:
            shape = _trace_event_block_fp8_shape(event)
            if shape is not None:
                shapes.add(shape)
    if not shapes:
        return "", 0
    destination = output_dir or (trace_input if trace_input.is_dir() else trace_input.parent)
    out = destination / "forge_shapes.json"
    payload = [{"M": m, "N": n, "K": k} for m, n, k in sorted(shapes)]
    try:
        destination.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        return "", 0
    return str(out), len(payload)


def _reuse_vllm_block_fp8_roofline_shapes(
    state: Any,
    *,
    workspace: Path,
    current_workload: dict[str, Any] | None = None,
) -> HandlerResult | None:
    """Reuse block-FP8 runtime shapes from the latest Roofline profile trace."""
    last_trace_analyze = getattr(state, "last_trace_analyze", None)
    if not isinstance(last_trace_analyze, dict):
        return None
    source_trace = str(last_trace_analyze.get("steady_state_trace") or "").strip()
    if not source_trace:
        log.info(
            "vLLM block-FP8 shape capture: latest Roofline has no selected "
            "steady-state trace; running a standard Roofline fallback"
        )
        return None
    profile_trace = str(getattr(state, "last_profile_trace", "") or "").strip()
    analyzed_trace = str(last_trace_analyze.get("trace_input") or "").strip()
    try:
        profile_trace_id = str(Path(profile_trace).expanduser().resolve(strict=False))
        analyzed_trace_id = str(Path(analyzed_trace).expanduser().resolve(strict=False))
    except OSError:
        profile_trace_id = profile_trace
        analyzed_trace_id = analyzed_trace
    if not profile_trace or not analyzed_trace or profile_trace_id != analyzed_trace_id:
        log.info(
            "vLLM block-FP8 shape capture: steady-state trace provenance does "
            "not match the latest profile; running a standard Roofline fallback"
        )
        return None
    if not state.profile_trace_usable():
        log.info(
            "vLLM block-FP8 shape capture: latest Roofline profile left no usable trace; "
            "running a standard Roofline fallback"
        )
        return None
    profile_workload = getattr(state, "last_profile_workload", None)
    expected_workload = current_workload or state.current_profile_workload_context()
    recorded_workload = profile_workload if isinstance(profile_workload, dict) else {}
    if recorded_workload != expected_workload:
        mismatches = sorted(
            key
            for key in set(recorded_workload) | set(expected_workload)
            if recorded_workload.get(key) != expected_workload.get(key)
        )
        log.info(
            "vLLM block-FP8 shape capture: Roofline workload mismatch (%s); running a standard Roofline fallback",
            ", ".join(mismatches) or "missing profile workload metadata",
        )
        return None
    shapes_json, shape_count = _extract_vllm_block_fp8_profile_shapes(
        Path(source_trace),
        output_dir=workspace,
    )
    if shape_count == 0:
        log.info(
            "vLLM block-FP8 shape capture: Roofline trace %s contains no reusable "
            "block-FP8 shapes; running a standard Roofline fallback",
            source_trace,
        )
        return None
    log.info(
        "vLLM block-FP8 shape capture: reusing %d shape(s) from Roofline trace %s",
        shape_count,
        source_trace,
    )
    return {
        "status": "ok",
        "shapes_json": shapes_json,
        "shape_capture_workspace": str(workspace),
        "shape_count": shape_count,
        "capture_mode": "roofline_profile_reuse",
        "source_profile_trace": source_trace,
    }


def _vllm_dense_shape_capture_required(
    *,
    framework: str,
    model_path: str,
    supplied: SuppliedShapes,
    dry_run: bool,
) -> bool:
    """Return whether Forge needs an automatic TunableOp recording pass."""
    if (
        framework != "vllm"
        or dry_run
        or supplied.shapes_json
        or supplied.tunableop_input
        or not env_bool("HYPERLOOM_GEMM_SHAPE_CAPTURE", True)
    ):
        return False
    from ..actions.executors._multi_node_env import is_multi_node

    if is_multi_node():
        return False

    from hyperloom.inference_optimizer.model_config_utils import summarize_model_config

    summary = summarize_model_config(model_path)
    if not summary or bool(summary.get("is_moe")):
        return False
    try:
        hidden_size = int(summary.get("hidden_size") or 0)
        intermediate_size = int(summary.get("intermediate_size") or 0)
    except (TypeError, ValueError):
        return False
    return hidden_size > 0 and intermediate_size > 0


def _resolve_shape_capture_port(value: Any) -> int:
    """Resolve an isolated capture port and reject the production port."""
    if value in (None, ""):
        return pick_server_port()
    try:
        port = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid shape_capture_port: {value!r}") from exc
    if port <= 0 or port > 65535 or port == 8888:
        raise ValueError(f"shape_capture_port must be 1..65535 and not 8888: {port}")
    return port


def _is_tunableop_untuned_row(line: str) -> bool:
    """Recognize a native PyTorch TunableOp offline-input row."""
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "Validator")):
        return False
    fields = [field.strip() for field in stripped.split(",")]
    if len(fields) < 2 or "TunableOp" not in fields[0] or not fields[1]:
        return False
    dimensions = [int(value) for value in re.findall(r"\d+", fields[1])]
    return sum(value > 0 for value in dimensions) >= 3


def _merge_tunableop_untuned_files(base_path: Path) -> int:
    """Merge per-device TunableOp recordings into one deterministic input."""
    rows: list[str] = []
    seen: set[str] = set()
    pattern = f"{base_path.stem}*{base_path.suffix}"
    for path in sorted(base_path.parent.glob(pattern)):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            row = line.strip()
            if _is_tunableop_untuned_row(row) and row not in seen:
                seen.add(row)
                rows.append(row)
    if not rows:
        return 0

    tmp_path = base_path.with_suffix(f"{base_path.suffix}.tmp")
    try:
        tmp_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        os.replace(tmp_path, base_path)
    except OSError:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            log.debug(
                "shape capture: failed to remove temporary TunableOp file %s",
                tmp_path,
                exc_info=True,
            )
        return 0
    return len(rows)


class _ShapeCaptureRun(NamedTuple):
    capture_dir: Path
    session_dir: Path
    task_id: str
    envs: dict[str, str]
    server_args: str
    remove_args: list[str]
    unset_envs: list[str]
    args_mode: str


def _shape_capture_failure(
    error: str,
    capture_dir: Path,
    *,
    shape_count: int | None = None,
    capture_mode: str = "",
) -> HandlerResult:
    failure: HandlerResult = {
        "status": "failed",
        "decision": "REVERT",
        "requires_e2e_validation": False,
        "error_class": "shape_capture_failed",
        "error": error,
        "shape_capture_workspace": str(capture_dir),
    }
    if shape_count is not None:
        failure["shape_count"] = shape_count
    if capture_mode:
        failure["capture_mode"] = capture_mode
    return failure


def _benchmark_failure_detail(benchmark_result: Mapping[str, Any]) -> str:
    error = str(benchmark_result.get("error") or benchmark_result.get("error_class") or "").strip()
    return f": {error}" if error else ""


def _fill_capture_workload_envs(envs: dict[str, str], state: Any, payload: dict) -> None:
    for env_name, state_name in (
        ("TP", "tp"),
        ("CONC", "conc"),
        ("ISL", "isl"),
        ("OSL", "osl"),
        ("MAX_MODEL_LEN", "max_model_len"),
    ):
        value = payload.get(state_name)
        if value in (None, ""):
            value = envs.get(env_name)
        if value in (None, ""):
            value = getattr(state, state_name, 0)
        try:
            resolved = int(value or 0)
        except (TypeError, ValueError):
            resolved = 0
        if resolved > 0:
            envs[env_name] = str(resolved)


def _shape_capture_run(state: Any, payload: dict, workspace: Path, session_dir: Path) -> _ShapeCaptureRun:
    capture_dir = workspace / "shape_capture" / f"attempt-{time.time_ns()}"
    capture_dir.mkdir(parents=True, exist_ok=True)
    current_best = getattr(state, "current_best", None) or {}

    inherited_envs = dict(current_best.get("extra_envs") or {})
    inherited_envs.update(dict(payload.get("extra_envs") or {}))
    envs = {str(key): str(value) for key, value in inherited_envs.items()}
    _fill_capture_workload_envs(envs, state, payload)

    inherited_unset = payload.get("unset_envs", current_best.get("unset_envs")) or []
    inherited_remove = payload.get("remove_args", current_best.get("remove_args")) or []
    return _ShapeCaptureRun(
        capture_dir=capture_dir,
        session_dir=session_dir,
        task_id=f"{str(payload.get('task_id') or workspace.name)}-shape-capture",
        envs=envs,
        server_args=(
            str(payload.get("extra_server_args") or "")
            if "extra_server_args" in payload
            else str(current_best.get("extra_server_args") or "")
        ),
        remove_args=[inherited_remove] if isinstance(inherited_remove, str) else [str(arg) for arg in inherited_remove],
        unset_envs=[inherited_unset] if isinstance(inherited_unset, str) else [str(key) for key in inherited_unset],
        args_mode=str(payload.get("args_mode") or current_best.get("args_mode") or "append"),
    )


def _capture_task_params(run: _ShapeCaptureRun, state: Any, payload: dict) -> dict[str, Any]:
    from hyperloom.inference_optimizer.breakdown.recorder.event_ids import INLINE_EVENT_PARAM
    from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import kernel_event_id

    return {
        "output_dir": str(run.capture_dir),
        "framework": "vllm",
        "model_path": str(payload.get("model_path") or getattr(state, "model_path", "") or ""),
        "gpu_type": str(payload.get("gpu_type") or getattr(state, "gpu_type", "") or ""),
        "extra_server_args": run.server_args,
        "extra_envs": run.envs,
        "remove_args": run.remove_args,
        "unset_envs": run.unset_envs,
        "args_mode": run.args_mode,
        # Either arm of the capture is a sub-step of the KERNEL phase's own
        # event, not a dispatched measurement, so it records into that event
        # rather than leaving one of its own.
        INLINE_EVENT_PARAM: kernel_event_id(int(getattr(state, "macro_cycle", 0) or 0)),
    }


def _tunableop_recording_envs(inherited: Mapping[str, str], port: int, capture_dir: Path) -> dict[str, str]:
    envs = {
        key: value for key, value in inherited.items() if not key.startswith(("PYTORCH_TUNABLEOP_", "HL_TUNABLEOP_"))
    }
    envs.update(
        {
            "PORT": str(port),
            "RUN_EVAL": "false",
            "HL_TUNABLEOP_MODE": "",
            "HL_TUNABLEOP_FILE": "",
            "HL_TUNABLEOP_VERBOSE": "",
            "PYTORCH_TUNABLEOP_ENABLED": "1",
            "PYTORCH_TUNABLEOP_TUNING": "0",
            "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "1",
            "PYTORCH_TUNABLEOP_UNTUNED_FILENAME": str(capture_dir / _TUNABLEOP_UNTUNED_NAME),
            "PYTORCH_TUNABLEOP_FILENAME": str(capture_dir / _TUNABLEOP_RESULTS_NAME),
        }
    )
    return envs


async def _capture_vllm_tunableop_shapes(
    *,
    state: Any,
    session_dir: Path,
    payload: dict,
    workspace: Path,
) -> HandlerResult:
    run = _shape_capture_run(state, payload, workspace, session_dir)
    if payload.get("_shape_capture_mode") == "block_fp8_profile":
        return await _capture_block_fp8_profile_shapes(run, state, payload)
    return await _capture_tunableop_recording(run, state, payload)
